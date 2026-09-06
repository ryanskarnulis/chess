"""Deterministic game core: python-chess owns board truth.

The agent layer never touches `chess.Board` directly — it goes through
`GameSession`, which accepts or rejects moves and reports state.
"""

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import chess
import chess.pgn

_COLOR_NAMES = {chess.WHITE: "white", chess.BLACK: "black"}

# How many `alternatives` a rejected move comes back with. A cap, because the
# fallback branch is "everything legal here" and a midgame position has forty of
# them — this is a suggestion list, not the complete set, and it is named
# `alternatives` rather than `legal_moves` so nothing reads it as one.
ALTERNATIVES_MAX = 8

_SQUARE = re.compile(r"[a-h][1-8]")
_SAN_PIECE = re.compile(r"^[KQRBN]")

# Pawn values, the conventional ones, for counting who is ahead. The king is
# never off the board, so it has no value here.
_PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
}

# The order `piece_placement` names piece types in — the order a player reads a
# position out loud, king first. Spelled out rather than taken from
# python-chess's own enum, which runs pawn-first and would open every spoken
# description with the pawn structure.
_PLACEMENT_ORDER = (
    chess.KING,
    chess.QUEEN,
    chess.ROOK,
    chess.BISHOP,
    chess.KNIGHT,
    chess.PAWN,
)

_TERMINATION_NAMES = {
    chess.Termination.CHECKMATE: "checkmate",
    chess.Termination.STALEMATE: "stalemate",
    chess.Termination.INSUFFICIENT_MATERIAL: "insufficient_material",
    chess.Termination.SEVENTYFIVE_MOVES: "seventyfive_moves",
    chess.Termination.FIVEFOLD_REPETITION: "fivefold_repetition",
    chess.Termination.THREEFOLD_REPETITION: "threefold_repetition",
    chess.Termination.FIFTY_MOVES: "fifty_moves",
}

# The draws a player *claims* rather than getting automatically, in the order
# python-chess's own `outcome(claim_draw=True)` resolves them — so the first name
# `claimable_draws()` reports is the termination a claim would actually produce.
_DRAW_CLAIM_RULES = (
    ("fifty_moves", chess.Board.can_claim_fifty_moves),
    ("threefold_repetition", chess.Board.can_claim_threefold_repetition),
)


@dataclass(frozen=True)
class MoveResult:
    """Outcome of a move submission. `legal` is the engine's final word.

    `capture` (the symbol of the piece the move took, or None) and `check`
    (whether it left the opponent in check) are here because the observe beat
    between the player's move and the engine's reply narrates a verified move,
    and those are the two facts a reaction is made of. They are board truth, so
    the session derives them at move time — the caller never re-reads the board
    to work out what just happened, and the model is never asked.

    `alternatives` is the same idea applied to a *rejection* (audit item 14):
    legal moves that plausibly answer what was asked for, so the agent can
    correct without a second round trip spent asking what is legal. Empty on a
    move that landed, and on one refused because the game is already over —
    there is no alternative to a finished game.
    """

    legal: bool
    san: str | None = None
    uci: str | None = None
    game_over: bool = False
    reason: str | None = None
    capture: str | None = None
    check: bool = False
    alternatives: tuple[str, ...] = ()


@dataclass(frozen=True)
class UndoResult:
    """Outcome of a takeback request. `undone` is SAN, in pop order."""

    ok: bool
    undone: tuple[str, ...] = ()
    reason: str | None = None


@dataclass(frozen=True)
class Outcome:
    termination: str
    winner: str | None
    result: str


def _validate_player_color(color: str) -> str:
    if color not in _COLOR_NAMES.values():
        raise ValueError(f"invalid player color: {color!r}")
    return color


def _validate_started(value: Any) -> str | None:
    """The start date off a save file, or None for one written before games
    recorded it. Checked like every other loaded field: a `Date` header is
    written straight into the PGN a player hands to a viewer, so a string that
    is not a date has no business reaching one."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"invalid started date: {value!r}")
    try:
        date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"invalid started date: {value!r}") from exc
    return value


class GameSession:
    """One chess game. Moves come in as SAN or UCI strings; the board decides."""

    def __init__(self, fen: str | None = None, player_color: str = "white"):
        self._board = chess.Board(fen) if fen is not None else chess.Board()
        self._resigned: chess.Color | None = None
        # Whether a claimable draw was claimed. Session-level like `_resigned`:
        # a board knows a claim is *available*, never that anybody made it.
        self._draw_claimed = False
        # Whether the two sides agreed a draw (`agree_draw`). Session-level for
        # the same reason: agreement is not a property of any position.
        self._draw_agreed = False
        self._player_color = _validate_player_color(player_color)
        self._started: str | None = date.today().isoformat()
        self._revision = 0

    @property
    def started(self) -> str | None:
        """The local date this game began, ISO (`"2026-09-04"`), or None for a
        save written before games recorded one.

        Session state like `_resigned`: a board holds a position, never the day
        somebody sat down at it, and PGN's `Date` tag wants exactly that day.
        Set at construction — a session rooted on a mid-game FEN is still a game
        starting now — and again by `new_game`, because that is a different
        game and dating it by the last one would be wrong on any board left up
        overnight.
        """
        return self._started

    @property
    def revision(self) -> int:
        """How many times this session's board truth has changed, from 0.

        The chokepoint the shared context's board version is built on: every
        mutating method here bumps it, and only after the mutation actually
        happened — an illegal move, a takeback of more plies than were played,
        a resignation of a finished game all leave it where it was, because
        nothing changed. Reads never touch it.

        It counts *mutations*, not plies: a full exchange is two, an undo of two
        plies is one. Nobody outside compares it to a move count — what a client
        needs is only whether the number it holds is still the current one.
        """
        return self._revision

    @property
    def turn(self) -> str:
        return _COLOR_NAMES[self._board.turn]

    @property
    def player_color(self) -> str:
        """Which side the human plays; the engine owns the other. Session
        state (not board truth): it decides orientation and who the opening
        move belongs to, never legality."""
        return self._player_color

    @property
    def fullmove_number(self) -> int:
        """The move number a score sheet would write. The board's own count, so
        a session rooted on a mid-game FEN reports the number that FEN carried
        instead of restarting at 1."""
        return self._board.fullmove_number

    @property
    def plies(self) -> int:
        """Half-moves played *in this session*, from 0.

        Not the move number doubled: the two answer different questions on a
        board rebuilt from a FEN, where the number came in with the position
        and nothing has been played on it since.
        """
        return len(self._board.move_stack)

    def fen(self) -> str:
        return self._board.fen()

    def new_game(self, player_color: str | None = None) -> None:
        """Reset the board. `player_color` reassigns the human's side;
        None keeps the current assignment."""
        if player_color is not None:
            player_color = _validate_player_color(player_color)
        self._board.reset()
        self._resigned = None
        self._draw_claimed = False
        self._draw_agreed = False
        self._started = date.today().isoformat()
        if player_color is not None:
            self._player_color = player_color
        self._revision += 1

    def resign(self, color: str | None = None) -> Outcome:
        """Record a resignation. Defaults to the side to move.

        Boards don't model resignation, so it's session-level state folded
        into `outcome()` / `is_game_over()`.
        """
        if self.is_game_over():
            raise ValueError("cannot resign: game is already over")
        if color is None:
            resigner = self._board.turn
        else:
            by_name = {name: c for c, name in _COLOR_NAMES.items()}
            if color not in by_name:
                raise ValueError(f"invalid color: {color!r}")
            resigner = by_name[color]
        self._resigned = resigner
        self._revision += 1
        return self.outcome()

    def claimable_draws(self) -> tuple[str, ...]:
        """The draw rules the side to move may claim right now.

        python-chess's answer, whole: `can_claim_fifty_moves` and
        `can_claim_threefold_repetition` are the rules, and their semantics
        include the part a hand-rolled count would miss — a claim is available as
        soon as a legal move *would* complete the repetition or the count, not
        only once it has. Nothing here re-derives a chess rule.

        Empty on a finished game, like `legal_moves()`: there is nothing left to
        claim, including once the draw itself has been claimed.
        """
        if self.is_game_over():
            return ()
        return tuple(
            name for name, claimable in _DRAW_CLAIM_RULES if claimable(self._board)
        )

    def claim_draw(self) -> Outcome:
        """Claim an available draw. Refuses when there is nothing to claim.

        Which rule the claim lands under is not the caller's choice and not a
        second implementation of the rules: `outcome()` asks the board with
        `claim_draw=True` and reports whatever python-chess resolves it to, in
        the same precedence order `claimable_draws()` lists.

        Boards don't model *claiming*, only claimability, so — exactly like
        resignation — the fact lives here and folds into `outcome()` /
        `is_game_over()`. A refusal leaves the session untouched, revision
        included.
        """
        if self.is_game_over():
            raise ValueError("cannot claim a draw: game is already over")
        if not self.claimable_draws():
            raise ValueError("cannot claim a draw: no draw is available to claim")
        self._draw_claimed = True
        self._revision += 1
        return self.outcome()

    def agree_draw(self) -> Outcome:
        """Record a draw by agreement. Refuses on a finished game.

        The third session-level ending beside `resign` and `claim_draw`, and
        the one with no board predicate at all: a claim is *available* on a
        position, a resignation is *by* a side, but agreement is a fact about
        the two players and nothing on the board can hold it. Whether the
        engine's side agrees is not decided here — `chessapp.draw_offer` owns
        that rule; this only records that it did. A refusal leaves the session
        untouched, revision included.
        """
        if self.is_game_over():
            raise ValueError("cannot agree a draw: game is already over")
        self._draw_agreed = True
        self._revision += 1
        return self.outcome()

    def submit_move(self, move_str: str) -> MoveResult:
        if self.is_game_over():
            return MoveResult(legal=False, game_over=True, reason="game is over")

        move = self._parse(move_str)
        if move is None or move not in self._board.legal_moves:
            return self._reject(move_str)

        san = self._board.san(move)
        capture = self._captured_symbol(move)
        self._board.push(move)
        self._revision += 1
        return MoveResult(
            legal=True,
            san=san,
            uci=move.uci(),
            game_over=self._board.is_game_over(),
            capture=capture,
            check=self._board.is_check(),
        )

    def _captured_symbol(self, move: chess.Move) -> str | None:
        """The symbol of the piece `move` takes, or None for a quiet move.
        Read before the push, and en-passant-aware: there the captured pawn is
        not on the destination square (the same rule `captured_pieces` replays).
        """
        if not self._board.is_capture(move):
            return None
        if self._board.is_en_passant(move):
            return chess.piece_symbol(chess.PAWN)
        piece = self._board.piece_at(move.to_square)
        return chess.piece_symbol(piece.piece_type) if piece is not None else None

    def export_pgn(self, headers: Mapping[str, str] | None = None) -> str:
        """The game so far as PGN. Result reflects the session-level endings
        (resignation, a claimed draw) too, since it comes off `outcome()`.

        `headers` are who played, where and when — facts the core cannot know,
        so the caller composes them (`tools.pgn_headers`) and this only writes
        them down. Applied before `Result` so nothing a caller passes can
        overwrite the one header that is board truth. Omitted, the export is
        exactly what it always was: python-chess's `?` placeholders, which is
        what an offline export and the older tests get.
        """
        game = chess.pgn.Game.from_board(self._board)
        for tag, value in (headers or {}).items():
            game.headers[tag] = value
        outcome = self.outcome()
        game.headers["Result"] = outcome.result if outcome is not None else "*"
        return str(game)

    def to_dict(self) -> dict[str, Any]:
        """Serialized form: root FEN + UCI moves + the two session-level endings
        (resignation, a claimed draw), plus the player's color and the date the
        game began. `player_color`, `draw_claimed` and `started` are additive —
        older readers ignore them, and older saves that lack them load at the
        defaults those games were played under."""
        resigned = self._resigned
        return {
            "version": 1,
            "root_fen": self._board.root().fen(),
            "moves": [move.uci() for move in self._board.move_stack],
            "resigned": _COLOR_NAMES[resigned] if resigned is not None else None,
            "player_color": self._player_color,
            "draw_claimed": self._draw_claimed,
            "draw_agreed": self._draw_agreed,
            "started": self._started,
        }

    @classmethod
    def from_dict(cls, data: Any) -> "GameSession":
        """Rebuild a session by replaying moves through the legality gate,
        so a corrupted file can never produce an inconsistent board."""
        if not isinstance(data, dict):
            raise ValueError("save data must be an object")
        version = data.get("version")
        # bool is an int subclass, so equality alone would accept `true` as
        # version 1 even though it is the wrong serialized shape.
        if type(version) is not int or version != 1:
            raise ValueError(f"unsupported save version: {version!r}")
        missing = {"root_fen", "moves", "resigned"} - data.keys()
        if missing:
            raise ValueError(f"save data missing keys: {sorted(missing)}")

        root_fen = data["root_fen"]
        moves = data["moves"]
        resigned = data["resigned"]
        player_color = data.get("player_color", "white")
        draw_claimed = data.get("draw_claimed", False)
        if not isinstance(draw_claimed, bool):
            raise ValueError(f"invalid draw_claimed flag: {draw_claimed!r}")
        draw_agreed = data.get("draw_agreed", False)
        if not isinstance(draw_agreed, bool):
            raise ValueError(f"invalid draw_agreed flag: {draw_agreed!r}")
        if not isinstance(root_fen, str):
            raise ValueError("save data root_fen must be a string")
        if not isinstance(moves, list):
            raise ValueError("save data moves must be a list")
        if not all(isinstance(move, str) for move in moves):
            raise ValueError("save data moves must contain only strings")
        if resigned is not None and resigned not in _COLOR_NAMES.values():
            raise ValueError(f"invalid resigned color: {resigned!r}")
        _validate_player_color(player_color)
        started = _validate_started(data.get("started"))

        # Saves that predate the player-color field default to white — the
        # implicit assignment those games were played under.
        session = cls(fen=root_fen, player_color=player_color)
        # A resumed game keeps the day it was played, not the day it was
        # reopened; one saved before games recorded a date has none, and the
        # PGN says so rather than claiming today.
        session._started = started
        for uci in moves:
            result = session.submit_move(uci)
            if not result.legal:
                raise ValueError(f"save data contains illegal move: {uci!r}")
        if resigned is not None:
            session.resign(resigned)
        if draw_claimed:
            # Through the same validation a live claim goes through, for the same
            # reason the moves are replayed rather than trusted: a file claiming a
            # draw the position does not support must not produce a drawn board.
            session.claim_draw()
        if draw_agreed:
            # The same refusal a live agreement gets on a finished game: a file
            # carrying both a checkmate and an agreement is not a game that was
            # played.
            session.agree_draw()
        return session

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "GameSession":
        return cls.from_dict(json.loads(Path(path).read_text()))

    def undo(self, plies: int = 1) -> UndoResult:
        """Take back the last `plies` half-moves.

        The core pops exactly what it's asked to; pairing policy for
        vs-engine takebacks (plies=2) belongs to the caller.
        """
        if self._resigned is not None:
            return UndoResult(ok=False, reason="cannot undo: game ended by resignation")
        if self._draw_claimed:
            # Same rule, same reason: the claim was about *this* position, and
            # popping plies out from under it would leave a game over by a claim
            # the board no longer supports.
            return UndoResult(
                ok=False, reason="cannot undo: game ended by a claimed draw"
            )
        if self._draw_agreed:
            return UndoResult(ok=False, reason="cannot undo: game ended by agreement")
        available = len(self._board.move_stack)
        if plies < 1:
            return UndoResult(ok=False, reason="plies must be at least 1")
        if plies > available:
            return UndoResult(
                ok=False, reason=f"cannot undo {plies} plies: only {available} played"
            )
        undone = tuple(reversed(self.move_history()[-plies:]))
        for _ in range(plies):
            self._board.pop()
        self._revision += 1
        return UndoResult(ok=True, undone=undone)

    def legal_moves(self) -> list[str]:
        """Legal moves in the current position, in SAN. Empty once the game
        is over — including session-level terminations like resignation."""
        if self.is_game_over():
            return []
        return [self._board.san(move) for move in self._board.legal_moves]

    def legal_captures(self) -> dict[str, str]:
        """What each legal capturing move takes, by SAN: `{"exd5": "pawn"}`.

        SAN says *that* a move captures and never what — `exd5` takes whatever
        stands on d5 — and the board is the only thing that knows. The agent is
        handed this beside `legal_moves` because a capture asked for by its
        victim ("take the pawn") is otherwise unanswerable from the list alone:
        on move 1 it came back "which pawn?" for a board with nothing to take
        (2026-09-04 walkthrough). En passant names the pawn it takes, like
        `captured_pieces`. Empty once the game is over, like `legal_moves`.
        """
        if self.is_game_over():
            return {}
        captures: dict[str, str] = {}
        for move in self._board.legal_moves:
            symbol = self._captured_symbol(move)
            if symbol is not None:
                captures[self._board.san(move)] = chess.piece_name(
                    chess.PIECE_SYMBOLS.index(symbol)
                )
        return captures

    def legal_destinations(self) -> dict[str, list[str]]:
        """Legal moves grouped by origin square, in coordinate form
        (`"e2": ["e3", "e4"]`), for a board UI's move hints. Empty once the
        game is over. python-chess stays the sole source of move truth — the
        frontend renders these, it never generates them.
        """
        if self.is_game_over():
            return {}
        dests: dict[str, list[str]] = {}
        for move in self._board.legal_moves:
            origin = chess.square_name(move.from_square)
            target = chess.square_name(move.to_square)
            # A promotion yields four moves to the same square; list it once.
            targets = dests.setdefault(origin, [])
            if target not in targets:
                targets.append(target)
        return dests

    def move_history(self) -> list[str]:
        """Moves played so far, in SAN, derived from the board's move stack."""
        board = self._board.root()
        sans = []
        for move in self._board.move_stack:
            sans.append(board.san(move))
            board.push(move)
        return sans

    def move_history_by_color(self) -> dict[str, list[str]]:
        """The moves each color played, in order — `move_history()` split by
        whose turn it was.

        `captured_pieces()`'s sibling, and derived the same way: by replaying
        the stack, so the side is the board's answer and not an index's parity.
        A session rebuilt from a FEN can start with Black to move, and assuming
        White moved first would credit every move to the wrong player.

        Who played a move is a fact the honesty guard checks — commentary that
        credits a move to a side ("I played Nf3") has to name the side that
        really played it.
        """
        by_color: dict[str, list[str]] = {"white": [], "black": []}
        board = self._board.root()
        for move in self._board.move_stack:
            by_color[_COLOR_NAMES[board.turn]].append(board.san(move))
            board.push(move)
        return by_color

    def position_fens(self) -> list[str]:
        """FEN of every position reached, root first, current last — one per
        ply plus the root. Derived by replaying the move stack, like
        `move_history()`, so the board stays the sole source of truth."""
        board = self._board.root()
        fens = [board.fen()]
        for move in self._board.move_stack:
            board.push(move)
            fens.append(board.fen())
        return fens

    def captured_pieces(self) -> dict[str, list[str]]:
        """Piece symbols each color has captured, in capture order.

        Derived by replaying the move stack — never tracked separately.
        """
        board = self._board.root()
        captured: dict[str, list[str]] = {"white": [], "black": []}
        for move in self._board.move_stack:
            if board.is_capture(move):
                if board.is_en_passant(move):
                    symbol = chess.PAWN
                else:
                    symbol = board.piece_at(move.to_square).piece_type
                captured[_COLOR_NAMES[board.turn]].append(chess.piece_symbol(symbol))
            board.push(move)
        return captured

    def material_balance(self) -> int:
        """The player's material advantage in pawns — positive when ahead.

        Counted off the board rather than from `captured_pieces()`, because a
        promotion adds material nobody captured: the count has to be what is
        standing there, not the history of what left.
        """
        totals = {chess.WHITE: 0, chess.BLACK: 0}
        for piece_type, value in _PIECE_VALUES.items():
            for color in (chess.WHITE, chess.BLACK):
                totals[color] += value * len(self._board.pieces(piece_type, color))
        player = self.player_color == "white"
        return totals[player] - totals[not player]

    def material_profile(self) -> dict[str, Any]:
        """The material facts a draw offer is judged on, off the board:
        whether any queen stands on it, each side's non-pawn material in pawns
        (`_PIECE_VALUES`, so a promoted queen counts as the queen it is), and
        `material_balance()` for the player. Reported whole so the caller's
        rule (`chessapp.draw_offer`) reads facts and holds no chess of its own.
        """
        non_pawn = {}
        for color, name in _COLOR_NAMES.items():
            non_pawn[name] = sum(
                value * len(self._board.pieces(piece_type, color))
                for piece_type, value in _PIECE_VALUES.items()
                if piece_type != chess.PAWN
            )
        return {
            "queens": bool(self._board.pieces(chess.QUEEN, chess.WHITE))
            or bool(self._board.pieces(chess.QUEEN, chess.BLACK)),
            "non_pawn": non_pawn,
            "balance": self.material_balance(),
        }

    def piece_placement(self) -> dict[str, dict[str, list[str]]]:
        """Where each side's pieces stand: `{"white": {"king": ["e1"], ...}}`.

        Read off the board like `material_balance`, and for the same reason: a
        promoted pawn is a queen standing on the board, whatever the move list
        calls it.

        Ordering is part of the answer, because the consumer is prose
        (`tools.describe_position` composes a sentence out of this): types come
        in reading order (`_PLACEMENT_ORDER`) and squares in name order, so the
        same position always reads the same way. A type a side has none of is
        absent rather than an empty list — there is nothing to say about it.
        """
        placement: dict[str, dict[str, list[str]]] = {}
        for color, color_name in _COLOR_NAMES.items():
            by_type: dict[str, list[str]] = {}
            for piece_type in _PLACEMENT_ORDER:
                squares = sorted(
                    chess.square_name(square)
                    for square in self._board.pieces(piece_type, color)
                )
                if squares:
                    by_type[chess.piece_name(piece_type)] = squares
            placement[color_name] = by_type
        return placement

    def castling_status(self) -> dict[str, dict[str, Any]]:
        """Whether each side has castled, and what it may still do:
        `{"white": {"castled": "kingside"|"queenside"|None, "rights": [...]}}`.

        Two facts, and a board holds only the second. That a king already went
        is history, so it is replayed off the move stack the way
        `captured_pieces` is — and asked of the position *before* each push,
        which is the only board that can still recognize the move as a
        castling. The rights are the current board's own answer, so nothing
        here re-derives the rule about a rook that has moved.
        """
        castled: dict[str, str | None] = {name: None for name in _COLOR_NAMES.values()}
        board = self._board.root()
        for move in self._board.move_stack:
            mover = _COLOR_NAMES[board.turn]
            if board.is_kingside_castling(move):
                castled[mover] = "kingside"
            elif board.is_queenside_castling(move):
                castled[mover] = "queenside"
            board.push(move)
        status: dict[str, dict[str, Any]] = {}
        for color, color_name in _COLOR_NAMES.items():
            rights = []
            if self._board.has_kingside_castling_rights(color):
                rights.append("kingside")
            if self._board.has_queenside_castling_rights(color):
                rights.append("queenside")
            status[color_name] = {"castled": castled[color_name], "rights": rights}
        return status

    def _parse(self, move_str: str) -> chess.Move | None:
        try:
            return self._board.parse_san(move_str)
        except ValueError:
            pass
        try:
            return chess.Move.from_uci(move_str)
        except ValueError:
            return None

    def _reject(self, move_str: str) -> MoveResult:
        """A refused move, with the legal moves that answer what was asked.

        Two rejections, told apart because the fix differs: an *ambiguous* move
        names something real that more than one piece can do ("Nd2" with both
        knights in range), and the alternatives are the disambiguated forms —
        the answer is one of them. An *illegal* one names something no piece can
        do, and the alternatives are the nearest thing the position offers.

        Both come from `move_alternatives`, which needs no ambiguity check of
        its own: the moves matching a square and a piece type *are* the
        candidates a SAN ambiguity is between.
        """
        ambiguous = False
        try:
            self._board.parse_san(move_str)
        except chess.AmbiguousMoveError:
            ambiguous = True
        except ValueError:
            pass
        reason = (
            f"ambiguous move: {move_str} — more than one piece can play it"
            if ambiguous
            else f"illegal move: {move_str}"
        )
        return MoveResult(
            legal=False,
            reason=reason,
            alternatives=tuple(self.move_alternatives(move_str)),
        )

    def move_alternatives(self, move_str: str) -> list[str]:
        """Legal moves (SAN) that plausibly answer `move_str`.

        Board truth, so it lives here rather than in the tool that reports it.
        The request is read for the two things a chess move string always
        carries — a destination square and, when it says so, a piece type — and
        the position is filtered by whichever of them it gave:

        - moves to the named square, narrowed to the named piece if both are
          known (this is the branch that turns "Nd2" into `Nbd2`/`Nfd2`);
        - failing that, every move by the named piece ("Qh4" with the queen
          walled in offers the queen nothing, so the whole request was wrong);
        - failing that, whatever the position allows, capped.

        Capped throughout (`ALTERNATIVES_MAX`) — see the constant. Empty once
        the game is over, because `legal_moves` is.
        """
        legal = self.legal_moves()
        if not legal:
            return []
        squares = _SQUARE.findall(move_str)
        destination = squares[-1] if squares else None
        piece = self._requested_piece(move_str, squares)

        def matching(*, by_square: bool, by_piece: bool) -> list[str]:
            picked = []
            for move in self._board.legal_moves:
                if by_square and chess.square_name(move.to_square) != destination:
                    continue
                if by_piece and self._board.piece_type_at(move.from_square) != piece:
                    continue
                picked.append(self._board.san(move))
            return picked

        for by_square, by_piece in (
            (destination is not None, piece is not None),
            (destination is not None, False),
            (False, piece is not None),
        ):
            if not (by_square or by_piece):
                continue
            if found := matching(by_square=by_square, by_piece=by_piece):
                return found[:ALTERNATIVES_MAX]
        return legal[:ALTERNATIVES_MAX]

    def _requested_piece(
        self, move_str: str, squares: list[str]
    ) -> chess.PieceType | None:
        """Which piece the request named, from SAN's leading letter or, for a
        UCI-shaped string, from whatever actually stands on its origin square.
        None when the string says nothing about a piece (a pawn SAN like 'e5'
        deliberately included — a bare square is a square, not a claim)."""
        if letter := _SAN_PIECE.match(move_str):
            return chess.PIECE_SYMBOLS.index(letter.group().lower())
        if len(squares) >= 2 and move_str.startswith(squares[0]):
            origin = chess.parse_square(squares[0])
            return self._board.piece_type_at(origin)
        return None

    def is_check(self) -> bool:
        """Whether the side to move is in check."""
        return self._board.is_check()

    def is_game_over(self) -> bool:
        # `claim_draw` is the session's own flag, passed to the same board call
        # `outcome()` makes: one derivation, so the two can never disagree about
        # whether a claimed game is finished. It is only ever True on a position
        # whose claim was available when it was made, and every mutation is
        # refused from then on, so the board's answer cannot go stale.
        return (
            self._resigned is not None
            or self._draw_agreed
            or self._board.is_game_over(claim_draw=self._draw_claimed)
        )

    def outcome(self) -> Outcome | None:
        if self._resigned is not None:
            winner = not self._resigned
            return Outcome(
                termination="resignation",
                winner=_COLOR_NAMES[winner],
                result="1-0" if winner == chess.WHITE else "0-1",
            )
        if self._draw_agreed:
            return Outcome(termination="agreement", winner=None, result="1/2-1/2")
        raw = self._board.outcome(claim_draw=self._draw_claimed)
        if raw is None:
            return None
        fallback = raw.termination.name.lower()
        return Outcome(
            termination=_TERMINATION_NAMES.get(raw.termination, fallback),
            winner=_COLOR_NAMES[raw.winner] if raw.winner is not None else None,
            result=raw.result(),
        )
