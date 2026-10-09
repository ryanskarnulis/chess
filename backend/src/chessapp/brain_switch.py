"""The brain setting (#434): which model Glitch thinks with, switched at the
next turn boundary.

The player picks one of `profiles.BRAIN_CHOICES` (`Settings.brain`); this is
the `Brain` the app holds instead of a `LlamaBrain`, and it hands every call to
the brain it built for the current choice. The choice only takes effect at
`at_boundary`, which the app calls as an interaction starts, while it holds
`ctx.mutation_lock` and before any brain call: a turn already running holds
that lock, so a switch asked for mid-turn waits for it to end, and one turn
never plans on one model and narrates on another.

Game truth is untouched by a switch. The board, the pending question and the
record live in the `ToolContext`, so the new brain picks up where the old one
left off; only the crutches move with it, because they follow the planner's
profile (`ctx.crutches`). Stockfish still plays the moves.

The first turn on a new brain is slow — a model load, then the whole history
read with no cache — so the switch says it is `cold` until that turn is done,
and the API passes that on for the UI to say so.
"""

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from chessapp.brain import AgentResponse, Answer, Brain, Narration
from chessapp.conversation import Recall
from chessapp.tools import ToolContext

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BuiltBrain:
    """One brain as assembly built it for a model: the brain, the crutches its
    planner's profile asks for, and what the trace names it by."""

    model: str
    brain: Brain
    crutches: frozenset[str]
    serving_identity: Callable[[], dict[str, str]] | None = None
    # Told once the brain is the one serving (the manifest's announcement).
    on_installed: Callable[[], None] | None = None


class BrainSwitch:
    """A `Brain` that serves the player's chosen model, swapping only at a
    turn boundary. `build` makes a brain for a model id; `default` is the
    deployment's own model, what `Settings.brain = None` means."""

    def __init__(
        self,
        ctx: ToolContext,
        current: BuiltBrain,
        *,
        build: Callable[[str], BuiltBrain],
        default: str,
    ) -> None:
        self._ctx = ctx
        self._build = build
        self._default = default
        self._lock = threading.Lock()
        self._current = current
        self._cold = False

    @property
    def chosen(self) -> str:
        return self._ctx.settings.brain or self._default

    @property
    def current(self) -> BuiltBrain:
        with self._lock:
            return self._current

    def at_boundary(self) -> bool:
        """Install the chosen brain if it is not the one serving; True if it
        swapped. Called only while `ctx.mutation_lock` is held, before the
        interaction's first brain call. A brain that cannot be built is logged
        and the old one keeps serving: a setting never costs a turn."""
        chosen = self.chosen
        if chosen == self.current.model:
            return False
        try:
            built = self._build(chosen)
        except Exception:
            logger.warning("brain_switch_failed model=%s", chosen, exc_info=True)
            return False
        self._ctx.crutches = built.crutches
        with self._lock:
            previous = self._current.model
            self._current = built
            self._cold = True
        logger.info("brain_switched from=%s to=%s", previous, built.model)
        if built.on_installed is not None:
            try:
                built.on_installed()
            except Exception:
                logger.warning("brain_switch_listener_failed", exc_info=True)
        return True

    def status(self) -> dict[str, Any]:
        """What the settings report: the choice, the brain serving now (they
        differ until the next turn boundary), and whether that brain has yet
        to finish a turn — the first one pays a model load and a cold cache."""
        with self._lock:
            serving, cold = self._current.model, self._cold
        chosen = self.chosen
        return {
            "brain": chosen,
            "brain_serving": serving,
            "brain_cold": cold or chosen != serving,
        }

    def serving_identity(self) -> dict[str, str]:
        identity = self.current.serving_identity
        return identity() if identity is not None else {}

    def _warmed(self, built: BuiltBrain) -> None:
        with self._lock:
            if self._current is built:
                self._cold = False

    # --- Brain -------------------------------------------------------------

    def get_agent_response(
        self,
        board_state: dict[str, Any],
        command: str,
        *,
        earlier: Recall | None = None,
    ) -> AgentResponse:
        built = self.current
        response = built.brain.get_agent_response(board_state, command, earlier=earlier)
        self._warmed(built)
        return response

    def narrate(
        self,
        board_state: dict[str, Any],
        changes: list[dict[str, Any]],
        *,
        command: str = "",
        earlier: Recall | None = None,
    ) -> Narration:
        built = self.current
        narration = built.brain.narrate(
            board_state, changes, command=command, earlier=earlier
        )
        self._warmed(built)
        return narration

    def read_answer(self, question: str, text: str) -> Answer:
        return self.current.brain.read_answer(question, text)
