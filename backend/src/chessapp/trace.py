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
    origin: str = "",
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
    guarded_claims: Sequence[str] = (),
    suppressed: str = "",
    rewrite: str = "",
    rewrite_claims: Sequence[str] = (),
    rewrite_suppressed: str = "",
    provider_failure: str = "",
    engine_failure: str = "",
    error: str = "",
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

    `engine_failure` is the same fact about the *other* half of a turn: empty
    unless Stockfish died on the reply the player's move had already earned,
    and then the exception's class and message (`api._failure_name`). The
    board says a move went unanswered; only this says why, and the turn is
    otherwise an ordinary `completed` one — the loop's stop reason stays the
    loop's, because a delegate caller reads it as the run's outcome and the
    engine is not the run.

    `error` is the last resort: an exception that escaped the turn entirely,
    named by class and message. Every other field is then whatever the turn had
    learned before it died, which is the point — the record a reviewer most
    wants is the one a half-finished turn used to leave nothing of.

    `guarded` marks a turn whose first commentary asserted something the honesty
    guard's facts did not back, `guarded_claims` names the classes, and
    `suppressed` is that first text. `commentary` stays what the player actually
    saw. `rewrite` says what became of the second try the guard then asked the
    narrator for: `""` when none ran (nothing was guarded), `"spoken"` when the
    rewrite passed and is the commentary, `"cut"` when it still asserted
    something — `rewrite_claims` names what, `rewrite_suppressed` keeps its
    text — and the player got the deterministic fallback, `"lost"` when the
    provider died on it. A guarded turn is still a model miss whatever the
    rewrite did, which is why `guarded` reads the first draft: the eval floor
    measures the model's own discipline, and the rewrite is what spares the
    player the miss.

    The lie used to be dropped on purpose — the event was countable and that
    read like enough. It is not. A false positive is now the guard's likelier
    failure than a false negative (the classes have grown from one to twelve),
    and when one fires there is nowhere left to read what tripped it: the
    commentary is replaced, the transcript deliberately keeps it out
    (`api._remembered_facts` — a canned correction fed back is a register the
    model imitates), and the log put the classes in `extra`, which the default
    formatter drops. Two live misfires were diagnosed by guessing at candidate
    phrasings because of that. A guard nobody can debug gets loosened blindly,
    which is the one way this one becomes worthless. The trace is opt-in and
    off by default, so this costs nothing until somebody is already debugging.

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

    `origin` is which surface the interaction came from — `tools.PANEL_ORIGIN`
    for the panel and its buttons, `tools.delegate_origin(id)` for one delegate
    conversation — empty only on a record built without one. It is the field a
    confirmation bug is read off: a "yes" that ran a question asked in another
    thread (#281) looks identical to a legitimate one until the two records name
    their origins.

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
        "origin": origin,
        "commentary": commentary,
        "stop_reason": stop_reason,
        "provider_failure": provider_failure,
        "engine_failure": engine_failure,
        "error": error,
        "changed": changed,
        "guarded": guarded,
        "guarded_claims": list(guarded_claims),
        "suppressed": suppressed,
        "rewrite": rewrite,
        "rewrite_claims": list(rewrite_claims),
        "rewrite_suppressed": rewrite_suppressed,
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
