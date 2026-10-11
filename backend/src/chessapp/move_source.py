"""What chooses Glitch's reply: Stockfish, or Glitch himself (#471, #473).

The coordinator asks one `MoveSource` for every reply, on every route
(`TurnCoordinator._move_source`). A source only *chooses*: it returns a
`SourcedMove` and never touches the session, so the move still enters the game
through `session.submit_move`, in the coordinator, and a source can be wrong
without the game being wrong.

Two sources:

- `EngineSource`: Stockfish at the set difficulty, in every tier but `glitch`.
- `GlitchSource`: the `glitch` tier (docs/glitch-difficulty.md). The mover
  (`mover.py`) chooses; when it fails — retries spent, deadline passed, a
  truncated reply, the provider down — Stockfish plays at `beginner`
  (decision 4). The engine is already configured as `beginner` in this tier
  (`engine.DIFFICULTY_TIERS["glitch"]`), so the fallback is simply the
  engine's move, and a failure never makes Glitch stronger than he plays on
  his own. The fallback is marked, never silent: `chooser` says so, and the
  coordinator writes it to the PGN, the trace and the reply's result.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Protocol

from chessapp.game import GameSession

if TYPE_CHECKING:
    from chessapp.brain import ModelCall
    from chessapp.mover import Mover

# Who chose a reply. `fallback` is Stockfish standing in for a mover that
# returned no move, in the `glitch` tier.
CHOOSER_STOCKFISH = "stockfish"
CHOOSER_GLITCH = "glitch"
CHOOSER_FALLBACK = "fallback"
Chooser = Literal["stockfish", "glitch", "fallback"]

# Why a fallback happened when there was no mover to ask at all: the tier is
# set but the brain is gone. Assembly and `set_difficulty` keep this from
# happening; if it does, the fallback is still marked.
NO_MOVER = "no_mover"


@dataclass(frozen=True)
class SourcedMove:
    """One chosen reply: the move, who chose it, and for the `glitch` tier the
    mover's own record (`MoverChoice.as_trace` or `MoverFailed.as_trace`) and
    its model calls. `fallback_reason` is the mover's failure kind on a
    fallback, and empty otherwise."""

    uci: str
    chooser: Chooser = CHOOSER_STOCKFISH
    mover: dict[str, Any] | None = None
    calls: tuple[ModelCall, ...] = field(default=(), repr=False)
    fallback_reason: str = ""

    @property
    def fallback(self) -> bool:
        return self.chooser == CHOOSER_FALLBACK


class MoveSource(Protocol):
    """What chooses the reply's move.

    `background` says whether the coordinator may compute it off the main
    thread while the turn goes on (Stockfish, ~0.1 s, on its own process). The
    mover may not: it shares the one llama-server with the planner and the
    narrator, and a background call would compete with them for it.
    """

    background: bool

    def choose(self, session: GameSession) -> SourcedMove: ...


class _Engine(Protocol):
    def choose_move(self, session: GameSession) -> str: ...


class EngineSource:
    """Stockfish's move at the set difficulty."""

    background = True

    def __init__(self, engine: _Engine) -> None:
        self.engine = engine

    def choose(self, session: GameSession) -> SourcedMove:
        return SourcedMove(self.engine.choose_move(session))


class GlitchSource:
    """Glitch's own move, from the mover; Stockfish at `beginner` when the mover
    fails. A dead Stockfish under a failed mover raises, and the coordinator
    leaves the reply owed exactly as it does for any engine failure (#284)."""

    background = False

    def __init__(self, mover: Mover | None, engine: _Engine) -> None:
        self.mover = mover
        self.engine = engine

    def choose(self, session: GameSession) -> SourcedMove:
        from chessapp.mover import MoverFailed

        if self.mover is None:
            return self._fallback(session, NO_MOVER, None, ())
        try:
            choice = self.mover.choose(session)
        except MoverFailed as failed:
            return self._fallback(
                session, str(failed.kind), failed.as_trace(), failed.calls
            )
        return SourcedMove(
            choice.uci,
            CHOOSER_GLITCH,
            mover=choice.as_trace(),
            calls=choice.calls,
        )

    def _fallback(
        self,
        session: GameSession,
        reason: str,
        record: dict[str, Any] | None,
        calls: tuple[ModelCall, ...],
    ) -> SourcedMove:
        uci = self.engine.choose_move(session)
        mover = {"outcome": reason, **(record or {}), "fallback": True}
        return SourcedMove(
            uci,
            CHOOSER_FALLBACK,
            mover=mover,
            calls=calls,
            fallback_reason=reason,
        )


def fallback_comment(reason: str) -> str:
    """The PGN comment on a fallback move: a reader of the game sees which of
    Glitch's moves were not his own."""
    return f"Glitch's mover failed ({reason}); Stockfish played this move at beginner"
