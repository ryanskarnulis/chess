"""Deterministic game core: python-chess owns board truth.

The agent layer never touches `chess.Board` directly — it goes through
`GameSession`, which accepts or rejects moves and reports state.
"""

from dataclasses import dataclass

import chess

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
class Outcome:
    termination: str
    winner: str | None
    result: str


class GameSession:
    """One chess game. Moves come in as SAN or UCI strings; the board decides."""

    def __init__(self, fen: str | None = None):
        self._board = chess.Board(fen) if fen is not None else chess.Board()

    @property
    def turn(self) -> str:
        return _COLOR_NAMES[self._board.turn]

    def fen(self) -> str:
        return self._board.fen()

    def new_game(self) -> None:
        self._board.reset()

    def submit_move(self, move_str: str) -> MoveResult:
        if self._board.is_game_over():
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
        return self._board.is_game_over()

    def outcome(self) -> Outcome | None:
        raw = self._board.outcome()
        if raw is None:
            return None
        fallback = raw.termination.name.lower()
        return Outcome(
            termination=_TERMINATION_NAMES.get(raw.termination, fallback),
            winner=_COLOR_NAMES[raw.winner] if raw.winner is not None else None,
            result=raw.result(),
        )
