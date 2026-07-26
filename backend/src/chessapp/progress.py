"""Live turn progress: what the app is doing, while it is still doing it.

A turn used to be a black box with a spinner over it. It now has intentional
phases — the player's move is validated, Glitch reacts to it, Stockfish answers,
Glitch closes — and a spinner is the wrong shape for that: it says "waiting"
when what the player wants to know is *what for*. This module is the seam those
phases report through (the audit's item 19).

**Nothing here narrates.** Every event is emitted by the chokepoint that already
owns the thing it describes — the coordinator's phase setter, the registry's
dispatch, the brain's own two phases — for the same reason `mutations` is a
`board_version` delta rather than a tally: a hand-placed "now we are calculating"
line is a second copy of the truth, and second copies drift. The reporter's job
is only to say *which interaction* an event belongs to and hand it to a sink.

That "which interaction" is a `ContextVar`, not an attribute, because the
pipeline's blocking steps run **off the event loop** — they have to, or a
blocked loop delivers nothing until the turn it is describing is already over —
and the interaction has to travel into the worker thread with the copied
context. Outside an interaction there is nothing to label, so an event is
dropped rather than invented: a tool dispatched over MCP and a phase moved by a
board button belong to no turn the UI is watching.

`correlation_id` is deliberately the trace record's id (`trace.new_correlation_id`),
so a progress line the player saw and the record of the turn that produced it
are one search apart — that is the whole reason the trace slice minted the id
before anything on the wire carried it.

Two rules make this safe to sprinkle through the hot path:

- **Reporting never costs a turn.** A sink that raises (a socket that went away
  mid-send) is logged and dropped, exactly as a failing tracer is. The player's
  move has already been played; a lost label is not their problem.
- **An interaction is always closed.** `end` fires from a `finally`, so a turn
  that raised does not leave a progress line spinning on screen forever.

The one thing here that is *not* decoration is `on_narrating`: the narrator turn
**is** the coordinator's observation beat, and a brain's report that it started
narrating is the only signal the app gets that the beat opened (the brain holds
no coordinator, by design — `docs/turn-coordinator.md` predicted exactly this
seam). So the hook fires whether or not a sink is bound, and its failure is
swallowed for the same reason a sink's is.
"""

import contextvars
import logging
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# The five kinds of thing a turn can say about itself. `begin`/`end` bracket the
# interaction and carry no name; the other three name whatever moved.
KIND_BEGIN = "begin"
KIND_TOOL = "tool"
KIND_PHASE = "phase"
KIND_BRAIN = "brain"
KIND_END = "end"

# The brain's own two phases, in the brain's vocabulary rather than the
# coordinator's: the model is either deciding what to do or writing what to say.
# A `Brain` that reports these needs to know nothing about turns.
BRAIN_PLANNING = "planning"
BRAIN_NARRATING = "narrating"


@dataclass(frozen=True)
class ProgressEvent:
    """One thing that happened inside one interaction.

    Flat and tiny on purpose: this goes out many times per turn over the same
    socket the authoritative state document uses, and it is worth nothing if it
    arrives late. `name` is read according to `kind` — a tool name, a
    `TurnPhase` value, or a brain phase — which keeps one shape on the wire
    instead of four.
    """

    correlation_id: str
    turn_id: int
    kind: str
    name: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "correlation_id": self.correlation_id,
            "turn_id": self.turn_id,
            "kind": self.kind,
            "name": self.name,
        }


@dataclass(frozen=True)
class _Interaction:
    correlation_id: str
    turn_id: int


_CURRENT: contextvars.ContextVar[_Interaction | None] = contextvars.ContextVar(
    "chessapp_progress_interaction", default=None
)


class ProgressReporter:
    """Where the turn's chokepoints report, and where the sink is bound.

    Built before the things that report through it (the coordinator, the
    registry, the brain) and bound to a sink afterwards by whoever owns the
    transport — which is what keeps the wiring acyclic: nothing here knows what
    a websocket is, and nothing that reports knows there is one.
    """

    def __init__(self, on_narrating: Callable[[], None] | None = None) -> None:
        self._sink: Callable[[ProgressEvent], None] | None = None
        self._on_narrating = on_narrating

    def bind(self, sink: Callable[[ProgressEvent], None]) -> None:
        """Point the reporter at a transport. Unbound is a normal state — a
        reporter built for a process with no UI attached still marks the
        observation beat, it just says nothing about it."""
        self._sink = sink

    @contextmanager
    def interaction(self, correlation_id: str, turn_id: int) -> Iterator[None]:
        """Bracket one user interaction: everything reported inside belongs to
        it, and the `end` is guaranteed even if the turn raises."""
        token = _CURRENT.set(_Interaction(correlation_id, turn_id))
        try:
            self._emit(KIND_BEGIN, "")
            yield
        finally:
            self._emit(KIND_END, "")
            _CURRENT.reset(token)

    def phase(self, phase: Any) -> None:
        """The coordinator moved. Called by the phase setter itself, so the
        report cannot disagree with the machine."""
        self._emit(KIND_PHASE, str(phase))

    def tool(self, name: str) -> None:
        """A tool is about to run. Called by `ToolRegistry.dispatch`, the one
        road every tool call takes."""
        self._emit(KIND_TOOL, name)

    def brain(self, name: str) -> None:
        """The brain changed phase (`planning` / `narrating`).

        Narrating is not only a label: it is the observation beat opening, and
        this report is the only place the app can see it happen. The mark comes
        first and runs unconditionally — it is game-machine business, while the
        event is decoration.
        """
        if name == BRAIN_NARRATING and self._on_narrating is not None:
            try:
                self._on_narrating()
            except Exception:
                logger.warning("progress_observation_failed", exc_info=True)
        self._emit(KIND_BRAIN, name)

    def _emit(self, kind: str, name: str) -> None:
        current = _CURRENT.get()
        sink = self._sink
        if current is None or sink is None:
            return
        try:
            sink(
                ProgressEvent(
                    correlation_id=current.correlation_id,
                    turn_id=current.turn_id,
                    kind=kind,
                    name=name,
                )
            )
        except Exception:
            logger.warning("progress_emit_failed", exc_info=True)
