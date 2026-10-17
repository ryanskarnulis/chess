"""Maia-3: the human-like move behind the tiers below `maximum` (#510).

Maia-3 (CSSLab, U of Toronto) is a transformer trained on Lichess blitz games
to play the move a human of a given rating would. It runs as its own UCI
process (`maia3-uci`, the 79M model on CPU), like Stockfish, and like
Stockfish it only proposes: its move still enters the game through
`GameSession.submit_move`. It does no search, so it never analyses; reviews,
hints and every evaluation stay Stockfish's.

Each tier's Maia settings live in `engine.DIFFICULTY_TIERS`; the numbers were
measured against the old Stockfish tiers in docs/maia-tiers.md (#511). Every
call passes all three options, so the process holds no state a relaunch would
have to restore.

Maia is handed the game's moves, not a bare FEN: with `--use-uci-history` it
reads the last eight positions from the `position ... moves` command, and a
lone FEN would show it one position eight times.

It is optional. `CHESSAPP_MAIA` names the `maia3-uci` executable, and
`CHESSAPP_MAIA_CHECKPOINT` the weights file when they are local (the image
bakes them in; unset, Maia fetches them from Hugging Face once and caches
them). An unset `CHESSAPP_MAIA`, or a process that will not start, means no
Maia, and the tiers play their Stockfish settings instead (`EnginePlayer`),
logged, never a crash. A process that dies mid-game is started again once, and
if that fails too the tier plays Stockfish's move for that turn.
"""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from typing import Any

import chess
import chess.engine

from chessapp.game import GameSession

logger = logging.getLogger(__name__)

# The model and how it runs, owned here rather than by the environment: the
# tier map was measured with exactly these (docs/maia-tiers.md).
MAIA_ARGS = (
    "--model",
    "maia3-79m",
    "--device",
    "cpu",
    "--no-use-amp",
    "--use-uci-history",
)
# Torch's CPU threads for one forward pass: ~150 ms a move at 4, ~430 at 1.
MAIA_THREADS = 4
# Loading the checkpoint takes ~1.2 s; the first start may be slower.
STARTUP_TIMEOUT = 60.0


@dataclass(frozen=True)
class MaiaSettings:
    """One tier's Maia: the rating it plays as, and how it samples. `top_p`
    cuts the long tail a low-rated policy has, which otherwise shuffles pieces
    back and forth (docs/maia-tiers.md)."""

    elo: int
    temperature: float = 1.0
    top_p: float = 0.9

    def options(self) -> dict[str, Any]:
        # Maia declares Temperature and TopP as string options.
        return {
            "Elo": self.elo,
            "Temperature": str(self.temperature),
            "TopP": str(self.top_p),
        }


def game_board(session: GameSession) -> chess.Board:
    """The session's position with its moves, from the position it began
    at, so the UCI command carries the history Maia reads."""
    board = chess.Board(session.position_fens()[0])
    for san in session.move_history():
        board.push_san(san)
    return board


def _open_uci(command: list[str]) -> Any:
    env = {**os.environ, "OMP_NUM_THREADS": str(MAIA_THREADS)}
    engine = chess.engine.SimpleEngine.popen_uci(
        command, timeout=STARTUP_TIMEOUT, env=env
    )
    # Maia answers `uci` before it loads the checkpoint and loads it on
    # `isready`: ping now so a broken install fails here, not on a move.
    engine.ping()
    return engine


class MaiaEngine:
    """One Maia process. Thread-safe like `EnginePlayer`: python-chess
    serialises calls, and a relaunch happens once under a lock."""

    def __init__(
        self,
        path: str,
        checkpoint: str | None = None,
        open_engine: Callable[[list[str]], Any] = _open_uci,
    ) -> None:
        self._command = [path, *MAIA_ARGS]
        if checkpoint:
            self._command += ["--checkpoint-path", checkpoint]
        self._open_engine = open_engine
        self._engine = open_engine(self._command)
        self._relaunch_lock = threading.Lock()
        self._closed = False

    def choose_move(self, session: GameSession, settings: MaiaSettings) -> str:
        """Maia's move for the side to move, as UCI."""
        board = game_board(session)
        options = settings.options()
        result = self._run(
            lambda engine: engine.play(
                board, chess.engine.Limit(nodes=1), options=options
            )
        )
        if result.move is None:
            raise ValueError("maia returned no move")
        return result.move.uci()

    def _run(self, work: Callable[[Any], Any]) -> Any:
        """One call, starting the process again once if it has died, as
        `EnginePlayer._run` does for Stockfish (#329)."""
        engine = self._engine
        try:
            return work(engine)
        except chess.engine.EngineTerminatedError:
            if self._closed:
                raise
            logger.warning("maia_died_relaunching", exc_info=True)
            with self._relaunch_lock:
                if self._engine is engine:
                    self._relaunch()
            return work(self._engine)

    def _relaunch(self) -> None:
        with suppress(Exception):
            self._engine.close()
        try:
            self._engine = self._open_engine(self._command)
        except Exception as exc:
            raise chess.engine.EngineTerminatedError(
                f"maia could not be relaunched: {exc}"
            ) from exc

    def close(self) -> None:
        self._closed = True
        with suppress(Exception):
            self._engine.quit()


def maia_from_env() -> MaiaEngine | None:
    """Maia from `CHESSAPP_MAIA`, or None: unset, or a process that will not
    start, means the tiers play their Stockfish settings."""
    path = os.environ.get("CHESSAPP_MAIA")
    if not path:
        return None
    try:
        return MaiaEngine(path, os.environ.get("CHESSAPP_MAIA_CHECKPOINT"))
    except Exception:
        logger.warning("maia_unavailable path=%s", path, exc_info=True)
        return None
