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
    """Outcome of a move submission. `legal` is the engine's final word."""

    legal: bool
    san: str | None = None
    uci: str | None = None
    game_over: bool = False
    reason: str | None = None


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


class GameSession:
    """One chess game. Moves come in as SAN or UCI strings; the board decides."""

    def __init__(self, fen: str | None = None):
        self._board = chess.Board(fen) if fen is not None else chess.Board()
        self._resigned: chess.Color | None = None

    @property
    def turn(self) -> str:
        return _COLOR_NAMES[self._board.turn]

    def fen(self) -> str:
        return self._board.fen()

    def new_game(self) -> None:
        self._board.reset()
        self._resigned = None

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
        self._board.push(move)
        return MoveResult(
            legal=True,
            san=san,
            uci=move.uci(),
            game_over=self._board.is_game_over(),
        )

    def export_pgn(self) -> str:
        """The game so far as PGN. Result reflects resignation too."""
        game = chess.pgn.Game.from_board(self._board)
        outcome = self.outcome()
        game.headers["Result"] = outcome.result if outcome is not None else "*"
        return str(game)

    def to_dict(self) -> dict[str, Any]:
        """Serialized form: root FEN + UCI moves + resignation flag."""
        resigned = self._resigned
        return {
            "version": 1,
            "root_fen": self._board.root().fen(),
            "moves": [move.uci() for move in self._board.move_stack],
            "resigned": _COLOR_NAMES[resigned] if resigned is not None else None,
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

        session = cls(fen=data["root_fen"])
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

    def move_history(self) -> list[str]:
        """Moves played so far, in SAN, derived from the board's move stack."""
        board = self._board.root()
        sans = []
        for move in self._board.move_stack:
            sans.append(board.san(move))
            board.push(move)
        return sans

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
