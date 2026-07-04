"""Stockfish bridge via python-chess UCI.

Stockfish is a calculation tool, not the referee: it proposes moves, but
every move still enters the game through `GameSession.submit_move`, and
`choose_move` works on a copy of the position so the engine can never
touch session truth.
"""

import chess
import chess.engine

from chessapp.game import GameSession, MoveResult

SKILL_MIN, SKILL_MAX = 0, 20
# Stockfish's own UCI_Elo bounds.
ELO_MIN, ELO_MAX = 1320, 3190

DEFAULT_MOVE_TIME = 0.1


def validate_skill_level(level: object) -> int:
    if not isinstance(level, int) or isinstance(level, bool):
        raise ValueError(f"skill level must be an int, got {level!r}")
    if not SKILL_MIN <= level <= SKILL_MAX:
        raise ValueError(f"skill level must be {SKILL_MIN}-{SKILL_MAX}, got {level}")
    return level


def validate_elo(elo: object) -> int:
    if not isinstance(elo, int) or isinstance(elo, bool):
        raise ValueError(f"elo must be an int, got {elo!r}")
    if not ELO_MIN <= elo <= ELO_MAX:
        raise ValueError(f"elo must be {ELO_MIN}-{ELO_MAX}, got {elo}")
    return elo


class EnginePlayer:
    """One Stockfish process playing moves at a configurable strength."""

    def __init__(self, path: str = "stockfish", move_time: float = DEFAULT_MOVE_TIME):
        self._engine = chess.engine.SimpleEngine.popen_uci(path)
        self._limit = chess.engine.Limit(time=move_time)

    def __enter__(self) -> "EnginePlayer":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        self._engine.quit()

    def set_skill_level(self, level: int) -> None:
        validate_skill_level(level)
        self._engine.configure({"UCI_LimitStrength": False, "Skill Level": level})

    def set_elo(self, elo: int) -> None:
        validate_elo(elo)
        self._engine.configure({"UCI_LimitStrength": True, "UCI_Elo": elo})

    def choose_move(self, session: GameSession) -> str:
        """The engine's move for the current position, as UCI."""
        if session.is_game_over():
            raise ValueError("cannot choose a move: game is over")
        board = chess.Board(session.fen())
        result = self._engine.play(board, self._limit)
        if result.move is None:
            raise ValueError("engine returned no move")
        return result.move.uci()

    def play_move(self, session: GameSession) -> MoveResult:
        """Choose a move and submit it through the session's legality gate."""
        return session.submit_move(self.choose_move(session))
