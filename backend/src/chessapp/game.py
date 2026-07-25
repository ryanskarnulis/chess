"""Deterministic game core: python-chess owns board truth.

The agent layer never touches `chess.Board` directly — it goes through
`GameSession`, which accepts or rejects moves and reports state.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chess
import chess.pgn

_COLOR_NAMES = {chess.WHITE: "white", chess.BLACK: "black"}

_TERMINATION_NAMES = {
    chess.Termination.CHECKMATE: "checkmate",
    chess.Termination.STALEMATE: "stalemate",
    chess.Termination.INSUFFICIENT_MATERIAL: "insufficient_material",
    chess.Termination.SEVENTYFIVE_MOVES: "seventyfive_moves",
    chess.Termination.FIVEFOLD_REPETITION: "fivefold_repetition",
}


@dataclass(frozen=True)
class MoveResult:
    """Outcome of a move submission. `legal` is the engine's final word.

    `capture` (the symbol of the piece the move took, or None) and `check`
    (whether it left the opponent in check) are here because the observe beat
    between the player's move and the engine's reply narrates a verified move,
    and those are the two facts a reaction is made of. They are board truth, so
    the session derives them at move time — the caller never re-reads the board
    to work out what just happened, and the model is never asked.
    """

    legal: bool
    san: str | None = None
    uci: str | None = None
    game_over: bool = False
    reason: str | None = None
    capture: str | None = None
    check: bool = False


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


class GameSession:
    """One chess game. Moves come in as SAN or UCI strings; the board decides."""

    def __init__(self, fen: str | None = None, player_color: str = "white"):
        self._board = chess.Board(fen) if fen is not None else chess.Board()
        self._resigned: chess.Color | None = None
        self._player_color = _validate_player_color(player_color)

    @property
    def turn(self) -> str:
        return _COLOR_NAMES[self._board.turn]

    @property
    def player_color(self) -> str:
        """Which side the human plays; the engine owns the other. Session
        state (not board truth): it decides orientation and who the opening
        move belongs to, never legality."""
        return self._player_color

    def fen(self) -> str:
        return self._board.fen()

    def new_game(self, player_color: str | None = None) -> None:
        """Reset the board. `player_color` reassigns the human's side;
        None keeps the current assignment."""
        if player_color is not None:
            player_color = _validate_player_color(player_color)
        self._board.reset()
        self._resigned = None
        if player_color is not None:
            self._player_color = player_color

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
        return self.outcome()

    def submit_move(self, move_str: str) -> MoveResult:
        if self.is_game_over():
            return MoveResult(legal=False, game_over=True, reason="game is over")

        move = self._parse(move_str)
        if move is None or move not in self._board.legal_moves:
            return MoveResult(legal=False, reason=f"illegal move: {move_str}")

        san = self._board.san(move)
        capture = self._captured_symbol(move)
        self._board.push(move)
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

    def export_pgn(self) -> str:
        """The game so far as PGN. Result reflects resignation too."""
        game = chess.pgn.Game.from_board(self._board)
        outcome = self.outcome()
        game.headers["Result"] = outcome.result if outcome is not None else "*"
        return str(game)

    def to_dict(self) -> dict[str, Any]:
        """Serialized form: root FEN + UCI moves + resignation flag, plus the
        player's color (additive — older readers ignore it)."""
        resigned = self._resigned
        return {
            "version": 1,
            "root_fen": self._board.root().fen(),
            "moves": [move.uci() for move in self._board.move_stack],
            "resigned": _COLOR_NAMES[resigned] if resigned is not None else None,
            "player_color": self._player_color,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GameSession":
        """Rebuild a session by replaying moves through the legality gate,
        so a corrupted file can never produce an inconsistent board."""
        if data.get("version") != 1:
            raise ValueError(f"unsupported save version: {data.get('version')!r}")
        missing = {"root_fen", "moves", "resigned"} - data.keys()
        if missing:
            raise ValueError(f"save data missing keys: {sorted(missing)}")

        # Saves that predate the player-color field default to white — the
        # implicit assignment those games were played under.
        session = cls(
            fen=data["root_fen"],
            player_color=data.get("player_color", "white"),
        )
        for uci in data["moves"]:
            result = session.submit_move(uci)
            if not result.legal:
                raise ValueError(f"save data contains illegal move: {uci!r}")
        if data["resigned"] is not None:
            session.resign(data["resigned"])
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
        return UndoResult(ok=True, undone=undone)

    def legal_moves(self) -> list[str]:
        """Legal moves in the current position, in SAN. Empty once the game
        is over — including session-level terminations like resignation."""
        if self.is_game_over():
            return []
        return [self._board.san(move) for move in self._board.legal_moves]

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

    def _parse(self, move_str: str) -> chess.Move | None:
        try:
            return self._board.parse_san(move_str)
        except ValueError:
            pass
        try:
            return chess.Move.from_uci(move_str)
        except ValueError:
            return None

    def is_check(self) -> bool:
        """Whether the side to move is in check."""
        return self._board.is_check()

    def is_game_over(self) -> bool:
        return self._resigned is not None or self._board.is_game_over()

    def outcome(self) -> Outcome | None:
        if self._resigned is not None:
            winner = not self._resigned
            return Outcome(
                termination="resignation",
                winner=_COLOR_NAMES[winner],
                result="1-0" if winner == chess.WHITE else "0-1",
            )
        raw = self._board.outcome()
        if raw is None:
            return None
        fallback = raw.termination.name.lower()
        return Outcome(
            termination=_TERMINATION_NAMES.get(raw.termination, fallback),
            winner=_COLOR_NAMES[raw.winner] if raw.winner is not None else None,
            result=raw.result(),
        )
