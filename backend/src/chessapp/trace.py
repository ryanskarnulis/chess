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
    outcome: dict[str, Any] | None = None,
    guarded: bool = False,
    guarded_claims: Sequence[str] = (),
    suppressed: str = "",
    rewrite: str = "",
    rewrite_claims: Sequence[str] = (),
    rewrite_suppressed: str = "",
    provider_failure: str = "",
    engine_failure: str = "",
    reaction_late: bool = False,
    error: str = "",
    model_calls: int = 0,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    model_latencies_ms: Sequence[int] = (),
    unmetered_calls: int = 0,
    spans_ms: dict[str, int] | None = None,
    serving: dict[str, str] | None = None,
    state_refreshes: Sequence[int] = (),
    handoff: dict[str, Any] | None = None,
    budget: str = "",
    input_trimmed: int = 0,
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

    `outcome` is how the game stood when the record was written, from the
    player's side (`{"winner": "player" | "opponent" | None, "termination"}`),
    or None on a live board. `fen_after` already says whether a board-ended
    game ended, but not who the player was and not whether anyone resigned,
    and those are exactly what a reviewer needs to re-judge a finished game's
    commentary against the guard's winner and termination facts (#287).

    `engine_failure` is the same fact about the *other* half of a turn: empty
    unless Stockfish died on the reply the player's move had already earned,
    and then the exception's class and message (`api._failure_name`). The
    board says a move went unanswered; only this says why, and the turn is
    otherwise an ordinary `completed` one — the loop's stop reason stays the
    loop's, because a delegate caller reads it as the run's outcome and the
    engine is not the run.

    `reaction_late` marks a turn whose narration was still being written when
    the turn went on without it (`api._REACTION_BUDGET_S`). It is not a failure
    — the words are optional and the app said its own line instead — and that
    is exactly why it is recorded: the commentary of a cut reaction is
    indistinguishable from verbosity=low, from a dead provider and from a beat
    that never opened, so without this the one thing worth tuning (the budget,
    against how long the narrator really takes) is invisible.

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
    Every round trip the turn made is in `model_calls`, whichever phase made it
    and whether or not it raised (#290) — a reader that died, a reaction the
    budget cut, a lost rewrite. `unmetered_calls` is how many of them reported
    no token usage (they raised, or the server sent none): the token totals are
    what was measured, so a non-zero count marks them as a lower bound rather
    than a measured total.

    `model_latencies_ms` is one reading per model call, in call order, and
    `model_ms` is their sum — derived here rather than passed, so the total and
    the parts cannot disagree in a record. Per-call rather than per-turn because
    a slow planner and a slow narrator are different problems, and the turn's
    total cannot tell them apart.

    `spans_ms` is where the turn's wall clock went outside the model (#290),
    whole milliseconds per phase, a key present only for a phase that ran:
    `queue` (waiting for the mutation lock behind another request), `tool`
    (tool handlers, analysis included — Stockfish asked *by a tool* is tool
    time), `engine` (collecting the engine's reply to the player's move),
    `guard` (the honesty guard, less its rewrite's model time, which is already
    in `model_ms`), and `total` (from asking for the lock to writing this
    record). Model time is `model_ms` and is not repeated here. Speech is not a
    span: text-to-speech is its own request, after the turn. `None` on a record
    built outside a request.

    `serving` is what served the turn, so two baselines can be tied to a
    configuration: short hashes of the planner prompt, the narrator prompt and
    the offered tool schemas, as they resolve when the record is written, beside
    the model name and the server URL. `None` when the app was not assembled
    with a model behind it (direct mode, an injected test brain).

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
    - `state_refreshes` is the board versions the planner was *re-shown* inside
      the command, in order (#282) — its opening state block ages the moment
      one of the turn's own tools mutates, and this is the record of it being
      told. Empty on a turn that mutated nothing, and equally on one whose
      mutation left the board mid-exchange, where the refresh is withheld
      because the legal moves would be the engine's; `mutations` beside it is
      what tells those two apart. `mutations: 2, state_refreshes: []` is a
      healthy move turn, and a mutating turn that goes on to decide again with
      nothing here is the bug this field exists to make visible.

    `budget` names the per-turn budget that ended the brain route's planning
    phase (#288) — `iterations`, `corrections`, `tool_calls`,
    `analysis_calls`, `wall_time` or `input` — and is empty on every turn none
    did, which is every route but that one. `stop_reason` says the phase ended
    early; this says on what, which is the number worth tuning when it fires.
    `input_trimmed` beside it counts the conversation's oldest exchanges the
    input budget dropped to fit the prompt; any non-zero reading is a prompt
    ten times larger than anything measured, and worth a look.

    `handoff` is what the brain route's narrator was told the turn did (#289,
    `handoff.Handoff.trace`): the kind, the tools that were done, refused and
    looked up, and whether the engine's reply was still owed as it spoke.
    `None` on every other route, and on a brain turn no narrator closed. A
    narration that announced something is re-judged against this, not against
    the planner's note: "took it back" under `performed: []` is the miss.
    """
    return {
        "utterance": utterance,
        "route": route,
        "origin": origin,
        "commentary": commentary,
        "stop_reason": stop_reason,
        "provider_failure": provider_failure,
        "engine_failure": engine_failure,
        "reaction_late": reaction_late,
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
        "state_refreshes": list(state_refreshes),
        "handoff": handoff,
        "budget": budget,
        "input_trimmed": input_trimmed,
        "model_calls": model_calls,
        "unmetered_calls": unmetered_calls,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "model_ms": sum(model_latencies_ms),
        "model_latencies_ms": list(model_latencies_ms),
        "spans_ms": spans_ms,
        "serving": serving,
        "fen_before": fen_before,
        "fen_after": fen_after,
        "engine_reply": engine_reply,
        "outcome": outcome,
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
