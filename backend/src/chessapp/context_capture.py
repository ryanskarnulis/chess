"""Context capture: the exact bytes of every model call, for a human to read (#359).

The turn trace (`trace.py`) records what the agent *decided* — tools, results,
guard, FENs — and what each call *cost*. It deliberately does not record what
the model was shown, and that is the question a misread turn leaves open: was
the decision wrong because the context was missing something, or because it
held too much? This answers it by keeping, per model call, the request body
byte-for-byte as it went to llama-server, the response body byte-for-byte as it
came back (thought blocks and raw tool-call text included), and the templated
prompt string the server rendered those messages into — which is the text the
model actually tokenized, not our JSON.

Capture happens at the one seam every call already goes through
(`LlamaCppProvider._post`), so no phase can be missed and none can be
described second-hand. The provider knows the bytes; it does not know which
turn or which phase it is serving, so both arrive by `ContextVar`: the
interaction's ids from `progress` (the same variable the progress events
read, carried into worker threads by the copied context), and the phase from
`model_phase`, which the brain wraps around each of its call sites.

The rules are the tracer's. Off unless `CHESSAPP_CONTEXT_PATH` is set, and a
capture that fails is logged and dropped — losing a diagnostic record must never
cost the player their turn. Nothing is reformatted, trimmed or redacted. And it
only *records* thought blocks: history is still built from `ChatResult`, which
never carries them.

It is a debugging tool, not something to leave on: a planner call is ~3k
prompt tokens, and each is kept twice (the JSON and its rendered template).
`scripts/watch_context.py` tails the file during a live game.
"""

import contextvars
import json
import logging
import threading
from collections import OrderedDict
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from chessapp.brain import PHASE_UNKNOWN
from chessapp.progress import current_interaction

logger = logging.getLogger(__name__)

# What a line in the capture file is, and its layout's version. A reader keys
# off `schema`, as the trace's readers do; bump it on any change they must
# branch on.
KIND_MODEL_CALL = "model_call"
CAPTURE_SCHEMA = 1

# How many interactions' call counters are remembered. A counter is only
# needed while its interaction can still make calls, which is one turn plus a
# late narration; this is a bound on memory, not a lifetime anyone waits on.
_SEQ_MEMORY = 256

_PHASE: contextvars.ContextVar[str] = contextvars.ContextVar(
    "chessapp_model_phase", default=PHASE_UNKNOWN
)


@contextmanager
def model_phase(phase: str) -> Iterator[None]:
    """Name the phase of every model call made inside this block.

    The brain wraps each of its call sites in one; the provider reads it back
    when it captures the call. A `ContextVar` rather than a `chat` argument so
    the `ChatProvider` protocol and every test double stay as they are.
    """
    token = _PHASE.set(phase)
    try:
        yield
    finally:
        _PHASE.reset(token)


def current_phase() -> str:
    """The phase `model_phase` named for this context, or `unknown`."""
    return _PHASE.get()


@dataclass(frozen=True)
class CallStamp:
    """Who a call belongs to, fixed at the moment it was sent.

    `seq` is the call's place among the calls its interaction *sent*, counted
    by the capture: with `correlation_id` it names the attempt. Taken before
    the request goes out, so two calls that overlap (a reaction still running
    while the turn moved on) are numbered in the order they started.
    `correlation_id` and `turn_id` are `None` for a call made outside any
    interaction (an eval harness, a probe).
    """

    correlation_id: str | None
    turn_id: int | None
    phase: str
    seq: int
    started_at: str


class ContextCapture(Protocol):
    """Whatever keeps the captured calls. Must never raise into a turn."""

    def begin(self) -> CallStamp: ...

    def record(self, stamp: CallStamp, call: dict[str, Any]) -> None: ...


class JsonlContextCapture:
    """Appends one JSONL line per model call, newest last.

    Thread-safe: the reaction beat and a late closer run on their own threads
    and can finish together, and two half-lines interleaved in one file are a
    record nobody can read.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._seq: OrderedDict[str | None, int] = OrderedDict()

    def begin(self) -> CallStamp:
        interaction = current_interaction()
        correlation_id, turn_id = interaction if interaction else (None, None)
        with self._lock:
            seq = self._seq.pop(correlation_id, 0) + 1
            self._seq[correlation_id] = seq
            while len(self._seq) > _SEQ_MEMORY:
                self._seq.popitem(last=False)
        return CallStamp(
            correlation_id=correlation_id,
            turn_id=turn_id,
            phase=current_phase(),
            seq=seq,
            started_at=_now(),
        )

    def record(self, stamp: CallStamp, call: dict[str, Any]) -> None:
        try:
            line = json.dumps(
                {
                    "kind": KIND_MODEL_CALL,
                    "schema": CAPTURE_SCHEMA,
                    "turn_id": stamp.turn_id,
                    "correlation_id": stamp.correlation_id,
                    "phase": stamp.phase,
                    "seq": stamp.seq,
                    "started_at": stamp.started_at,
                    "ended_at": _now(),
                    **call,
                }
            )
            with self._lock:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a") as handle:
                    handle.write(line + "\n")
        except Exception:
            logger.warning("context_capture_failed", exc_info=True)


def _now() -> str:
    return datetime.now(UTC).isoformat()
