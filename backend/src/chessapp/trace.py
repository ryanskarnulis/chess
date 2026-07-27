"""Turn tracing: the per-command record of what the agent actually did.

The pipeline runs an utterance down one of three roads — a deterministic
confirmation, the deterministic fast path, or the brain's tool loop — and until
this module existed it recorded *which* nowhere. A turn that misfired left
nothing behind to review: no way to tell whether the model was even called, what
tool it picked, what arguments it passed, or whether it ran out of budget. This
writes one JSONL record per turn holding exactly that, so a bad turn becomes a
thing you can read (and, with its `fen_before` + `utterance`, replay as an eval
scenario).

A record also has to say *which* turn it was and how much it moved: the ids
(`turn_id`, `correlation_id`), the mutation count, and a latency per model call.
Those are the four things the audit's item 18 found missing, and each answers a
question the earlier record could not — did this move land twice, is this log
line from this turn, was the slow part the model or Stockfish.

Tracing is diagnostics and nothing more. It is off unless a path is configured
(`CHESSAPP_TRACE_PATH`), and a tracer that fails is swallowed by the pipeline —
losing a diagnostic record must never cost the player their turn.
"""

import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

# The four roads an utterance can take through `_run_command`. Three of them are
# deterministic; only `brain` involves the model in deciding what to do.
ROUTE_CONFIRMATION = "confirmation"
ROUTE_FAST_PATH = "fast_path"
ROUTE_RESIGN = "resign"
ROUTE_BRAIN = "brain"
# …and the fifth road, which is not an utterance at all: a move dragged on the
# board in agent mode. It runs the same beats the fast path runs, so it is traced
# like a turn — with the structured move (`e2e4`) standing in for the utterance
# it never had, which is exactly the distinction a reader wants to make.
ROUTE_BOARD = "board"
# …and the board UI's control buttons, which are an interaction with no utterance
# at all: a destructive op armed, confirmed or declined. Traced because it can
# change the board, and a board change with nothing to explain it is what makes a
# trace hard to read — the spoken road's answer to the same question has always
# left a record.
ROUTE_CONTROL = "control"


class Tracer(Protocol):
    """Whatever records a finished turn. Free to fail: the caller swallows it."""

    def record(self, turn: dict[str, Any]) -> None: ...


def new_correlation_id() -> str:
    """A fresh id for one user interaction, short enough to grep by eye.

    It is what makes the record findable *from outside itself*: the pipeline
    stamps the same id on the warnings a turn logs, so a "commentary leaked
    move advice" line and the record of the turn that leaked it are one search
    apart rather than a matter of comparing timestamps. Distinct from the turn
    id on purpose — one interaction can span two coordinator turns (an undo
    abandons the open one), and two interactions can share one (a turn left
    open by a route that stopped early).
    """
    return uuid4().hex[:12]


def turn_record(
    *,
    utterance: str,
    route: str,
    commentary: str,
    stop_reason: str,
    changed: bool,
    turn_id: int,
    correlation_id: str,
    mutations: int,
    fen_before: str,
    fen_after: str,
    tool_calls: list[dict[str, Any]],
    tool_results: list[dict[str, Any]],
    engine_reply: dict[str, Any] | None = None,
    guarded: bool = False,
    provider_failure: str = "",
    model_calls: int = 0,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    model_latencies_ms: Sequence[int] = (),
) -> dict[str, Any]:
    """One turn, as the flat record a reviewer (or a replay) reads.

    `tool_calls` and `tool_results` arrive as the pipeline's two parallel lists
    — args and `{"name", "result"}`, parallel by construction — and are zipped
    back into one entry per call, which is the shape a human actually wants:
    what it called, with what, and what came back.

    `engine_reply` (`{"san", "uci"}`, or None when none was owed) is recorded
    separately because the reply is no longer part of any tool result: the move
    tool applies the player's move and the pipeline collects the answer after the
    observation beat. Without it a traced move turn would show a move with no
    answer to it — and a missing or duplicated engine move is one of the main
    things a trace is read to find.

    `provider_failure` is the empty string on every turn that did not die, and
    on one that did (`stop_reason="provider_error"`) it names the kind —
    `provider.ProviderFailure`'s vocabulary. The stop reason alone cannot tell
    "llama-server is crash-looping again" from "this prompt no longer fits the
    context", and those want opposite fixes; it is also what the eval harness
    classifies a sample by, so a retryable death is retried and a refusal is
    reported rather than re-sent five times.

    `guarded` marks a turn whose commentary was suppressed by the honesty guard
    (it announced an ending no tool produced). `commentary` is what the player
    actually saw, so the lie itself is not kept — but the *event* is countable,
    which is what tells you whether the model is still trying to fake outcomes.

    `model_calls`/`prompt_tokens`/`completion_tokens` are the turn's total cost
    at the provider boundary — how many times the model was called and the tokens
    summed across those calls. They default to 0 so a deterministic route (a
    canned confirmation, a declined op) records a real, readable zero rather than
    a gap. This is the number every context-shrinking cut is measured against.

    `model_latencies_ms` is one reading per model call, in call order, and
    `model_ms` is their sum — derived here rather than passed, so the total and
    the parts cannot disagree in a record. Per-call rather than per-turn because
    a slow planner and a slow narrator are different problems, and the turn's
    total cannot tell them apart.

    The three fields that say *which* turn this was, and how much of the board
    it moved:

    - `turn_id` is the coordinator's turn counter — the id the interaction
      opened under.
    - `correlation_id` identifies this one interaction (see
      `new_correlation_id`), and is stamped on the log lines the turn emitted.
    - `mutations` is how many times the board actually changed, counted off
      `ToolContext.board_version` rather than by hand: the one chokepoint every
      mutation already passes through. A healthy agent-mode move turn is **2**
      — the player's move and the engine's answer — so the duplicated-move bug
      the coordinator exists to prevent reads as a third under one `turn_id`,
      and a bypass reads as a mutation on a route that should have had none.
    """
    return {
        "utterance": utterance,
        "route": route,
        "commentary": commentary,
        "stop_reason": stop_reason,
        "provider_failure": provider_failure,
        "changed": changed,
        "guarded": guarded,
        "turn_id": turn_id,
        "correlation_id": correlation_id,
        "mutations": mutations,
        "model_calls": model_calls,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "model_ms": sum(model_latencies_ms),
        "model_latencies_ms": list(model_latencies_ms),
        "fen_before": fen_before,
        "fen_after": fen_after,
        "engine_reply": engine_reply,
        "tools": [
            {"name": result["name"], "args": args, "result": result["result"]}
            for args, result in zip(tool_calls, tool_results, strict=True)
        ],
    }


@dataclass
class JsonlTracer:
    """Appends each turn to a JSONL file, newest last, one line per turn.

    JSONL because turns are appended forever and read back a few at a time: it
    survives a crash mid-write, `tail -f` works during a live game, and a run of
    turns loads with a one-line comprehension.
    """

    path: Path

    def record(self, turn: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps({"ts": datetime.now(UTC).isoformat(), **turn})
        with self.path.open("a") as handle:
            handle.write(line + "\n")
