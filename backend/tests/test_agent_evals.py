"""Agent eval harness: golden command→tool-call tasks against the REAL model.

Mirrors PCC's `tests/test_agent_evals.py` (agent-standard STANDARD.md §6): a
small set of scripted scenarios driven through the *whole* delegate path — a
real `LlamaBrain` (llama-server behind llama-swap, gemma-4-12b) over a real
Stockfish engine — and asserts trajectory *shape* plus board end-state, never
exact call sequences (the model is sampled at temp 1.0).

Opt-in like the provider smoke, so CI and default local runs never touch the
GPU:

    cd backend
    CHESSAPP_AGENT_EVALS=1 .venv/bin/pytest tests/test_agent_evals.py -v -s

Each scenario gets a fresh app + game (function-scoped `eval_app`) and a fresh
conversation, so scenarios are independent; the Stockfish process is
module-scoped (one engine for the suite) and the model stays warm on
llama-swap across scenarios. The first call may cold-load the model (~100 s);
everything after runs warm. `-s` shows the per-scenario `[eval] …` stats lines
the baseline table in `docs/agent-evals.md` is built from.

The scenarios drive the same seam the conductor's delegate calls use —
`POST /api/agent/conversations/{id}/messages` — so the `MessageExchange` wire
(`assistant_message.tool_calls`: tool/arguments/result/error, and
`stop_reason`) is exactly what the goldens assert on. Board end-state is read
back through `GET /api/state`, the same document the web board renders.

Fast-path guard: chess short-circuits an utterance that parses as exactly one
legal move (`fastparse.parse_move`) with zero LLM calls. Every *model* scenario
asserts `parse_move(utterance, fen) is None` first, pinning that the eval stays
a model eval even if the parser grows later. The two `fast_path_*` scenarios do
the opposite — they assert the utterance *does* parse, and measure what the
short-circuit costs.

Cost is measured, not just printed: the live `LlamaCppProvider` is wrapped in a
`CountingProvider` (`fakes.py`), the only seam every model round trip passes
through, so the scenarios assert on **model calls** (not just tool calls) and on
the **thinking flag per turn**. That is what pins the three claims the loop's
design rests on: a fast-path move is zero-LLM at verbosity=low (one `narrate`
call above it), a tool-using utterance costs the planner's tool turn plus its
handoff turn plus the narrator's reply, and thinking stays OFF until an analysis
tool's result lands in context — then ON for the turn that reasons about it.

A brain-routed turn is **two phases** since 2026-07-25
(`docs/planner-narrator.md`): the planner's bounded tool loop, then one
tool-free narrator call that speaks as Glitch. So every model-routed scenario
here costs exactly one call more than it did before that slice, and the fast
path — which was already planner-free — costs exactly what it always did.

Since the move-flow split (`docs/turn-coordinator.md`), a move turn's narration
is the coordinator's *observation* beat — a reaction to the verified player move,
run while Stockfish computes its reply — and the pipeline appends the reply as a
canned line afterwards. That replaces the old post-hoc narration one for one, so
every cost pin below is unchanged; what changed is that `make_move`'s result no
longer carries the reply, so a scenario's board end-state is read where it always
was, from `GET /api/state`.

This suite is the tripwire the standard requires: baseline results are recorded
in `docs/agent-evals.md`, and it gates every future prompt/model/loop change —
run it before merging one; the baseline must not regress.

**A pass rate is judged statistically, not compared to a literal** (Sprint 5,
slice 3; audit item 23). `assert result.rate >= 0.8` over five samples is a coin
flip at the floor — the record has the same build measuring 3/5 then 4/5, and
`play_as_black` measuring 5/5, 2/5, 0/5, 5/5 — so every decision this file makes
about a count now lives in `evalstats.py`, which is pure, GPU-free and
unit-tested on every commit (`test_evalstats.py`). Three consequences here:

- **Sampling is block-sequential.** A block of `_BLOCK_RUNS` is taken, and only
  a genuinely ambiguous count (the interval straddles the floor) buys another,
  up to `_MAX_RUNS`. At n=5 a healthy scenario still costs exactly five samples.
- **Infrastructure is retried, not scored.** llama-server crash-restarts every
  3–8 minutes of sustained generation, and since audit 20 that arrives as a 200
  carrying `stop_reason="provider_error"` — indistinguishable, to the old
  harness, from the model quietly doing nothing. A `_CollectingTracer` on the
  app's own tracer seam supplies the real stop reason on both seams, so such a
  sample is thrown away and re-taken instead of counted as a miss. It also
  supplies the *kind* of death (`provider_failure`), because only a transient
  one is worth re-taking: a request llama-server refuses — the 400 an overrun
  context gets — refuses identically every time, and fails the item on the
  first sample rather than after five retries and the suite's infra budget.
- **A run that never reached the narrator is not a pass.** A budget stop reaches
  no narrator, so a purely negative commentary check passes for never having been
  tested (that really happened — three of four `hints_off_no_advice` passes in
  one recorded run). `_assert_reached_narrator` makes it a visible non-pass.

Set `CHESSAPP_EVAL_REPORT=/path/to.jsonl` and the suite writes one machine-
readable line per scenario *as it finishes*, so a baseline stops being
transcribed by hand out of terminal scrollback.

**A turn's model time is reported per call and per phase** (Sprint 5). Both the
`[eval]` line and the report's per-sample records carry `call_ms` (one reading
per round trip, in call order, off the trace), `planner_ms`, `narrator_ms` and
the `phases` attribution that says how confidently those two were separated —
`evalstats.split_latencies`, which derives it from the route and the stop reason
and answers UNKNOWN rather than guessing when the provider died mid-turn. The
turn total alone could not resolve the finding it was asked about: `no_progress`
turns narrate 2–3× slower than `completed` ones at the *same* model-call count,
and "the narrator is slow" and "this sample was hard for the model" sum to the
same `model_ms`. Nothing in `src/` changed to get this — the readings were
already on the record, and the harness-bug precedent (a wrong tool offer silently
moving a scenario from 5/5 to 0–3/5) is why a diagnostic slice does not touch
production.

**And the trajectory says what each call asked for** (Sprint 5) — `make_move` on
the line is now `make_move(move="e2e4")`, arguments sorted, values as JSON,
long ones cut. Same rule as above: the arguments were already on the wire and
only the reporting path dropped them. The blind spot was live — a `set_hints_mode`
call turns up on 9/65 `hints_off_no_advice` samples where the player asked only
"what should I play here?", and the old line could not say which way it flipped
a setting the player owns.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from collections.abc import Callable, Generator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, NamedTuple

import pytest
from fastapi.testclient import TestClient

from chessapp.agent_api import reset_rate_limit
from chessapp.api import create_app
from chessapp.coordinator import TurnCoordinator
from chessapp.engine import DEFAULT_TIER, EnginePlayer
from chessapp.fastparse import parse_move
from chessapp.game import GameSession
from chessapp.llama_brain import _DEFAULT_MAX_ITERATIONS, create_llama_brain
from chessapp.personality import PLANNER_PROMPT, system_prompt_for
from chessapp.provider import LlamaCppProvider
from chessapp.tools import (
    Settings,
    ToolContext,
    _save_path,
    brain_tool_exclusions,
    build_registry,
)
from evalstats import (
    BUDGET_STOPS,
    STOP_PROVIDER_ERROR,
    Decision,
    Outcome,
    RateResult,
    TurnLatencies,
    TurnTokens,
    VacuousRun,
    classify,
    generation_rate,
    scenario_record,
    split_latencies,
    split_tokens,
)
from fakes import CountingProvider, ModelCall

pytestmark = pytest.mark.skipif(
    os.environ.get("CHESSAPP_AGENT_EVALS") != "1",
    reason="agent evals run the real model: set CHESSAPP_AGENT_EVALS=1",
)

# Same env names the app uses (agent-standard model profile); the defaults are
# build_app's, so the eval hits the same runtime a delegate call would.
LLAMACPP_BASE_URL = os.environ.get("LLAMACPP_BASE_URL", "http://127.0.0.1:8200/v1")
LLAMACPP_MODEL = os.environ.get("LLAMACPP_MODEL", "gemma-4-12b")
STOCKFISH_PATH = os.environ.get("CHESSAPP_STOCKFISH", "/usr/bin/stockfish")

# The planner phase's sampling temperature, read exactly as `build_app_from_env`
# reads it: unset means the provider's default, so a measurement run is
# `CHESSAPP_PLANNER_TEMPERATURE=0.3 CHESSAPP_AGENT_EVALS=1 pytest …` and the
# baseline it produces is the app's own behavior at that number.
_PLANNER_TEMPERATURE_ENV = os.environ.get("CHESSAPP_PLANNER_TEMPERATURE")
PLANNER_TEMPERATURE = (
    float(_PLANNER_TEMPERATURE_ENV) if _PLANNER_TEMPERATURE_ENV else None
)

# Generous request timeout: a cold llama-swap load is ~100 s before the first
# byte (the provider's own read timeout is 300 s). TestClient's ASGI transport
# ignores it, but pass it defensively so nothing upstream caps the wait.
_REQUEST_TIMEOUT = 310.0

# Board-changing tools — a successful call to one of these (a *legal* make_move,
# an undo, etc.) is what "the board changed this turn" means. Settings tools and
# reads/analysis are deliberately excluded: they never move a piece.
_BOARD_TOOLS = frozenset({"make_move", "undo", "new_game", "resign", "resume_game"})

# Latency tripwires, not a band: the precise numbers live in the baseline table
# in docs/agent-evals.md, which a human reads. These only catch a *regression*
# of the kind the loop rework was meant to prevent (an analysis ask reasoning
# from cold again), and they are deliberately loose — the GPU is shared with
# project-command-center via llama-swap, so a tight bound would flake.
# Re-recorded 2026-07-25 for the planner/narrator split: the one thinking turn
# now runs on the personality-rich narrator prompt, and its reasoning length is
# sampling-dependent — measured warm across the split's runs: 3.4–18 s, with no
# competing GPU traffic on the slow ones. The old 15 s ceiling was calibrated
# for compact-context reasoning and flapped on an unchanged path, which is a
# broken gate, not a safety margin. Everything thinking-off: 0.5–0.9 s.
_ANALYSIS_CEILING_S = 30.0
_THINKING_OFF_CEILING_S = 8.0

# --- sampling knobs -----------------------------------------------------------
#
# All five are env-overridable so a measurement campaign needs no code edit (and
# so a campaign's settings land in the report header, where the number it
# produced can be read back against them). The 20-sample campaign is
# `CHESSAPP_EVAL_RUNS=20 CHESSAPP_EVAL_MAX_RUNS=20` — a minimum block equal to
# the cap forces the full count with no separate override.
#
# `_INFRA_RETRIES` / `_INFRA_BUDGET` are sized off the observed crash cadence
# rather than picked: ~10 s a sample and a crash every 18–48 samples means
# 0.4–1.1 expected deaths per 20-sample escalation, so five per scenario is ~5×
# expected, and the worst case is ~8 min of re-runs at the ~100 s cold reload.
_BLOCK_RUNS = int(os.environ.get("CHESSAPP_EVAL_RUNS", "5"))
_MAX_RUNS = int(os.environ.get("CHESSAPP_EVAL_MAX_RUNS", "20"))
_INFRA_RETRIES = int(os.environ.get("CHESSAPP_EVAL_INFRA_RETRIES", "5"))
_INFRA_BUDGET = int(os.environ.get("CHESSAPP_EVAL_INFRA_BUDGET", "25"))
_REPORT_PATH = os.environ.get("CHESSAPP_EVAL_REPORT")

# The floors, in one table instead of nine literals scattered through the file.
# They are what the current build actually achieves (recorded in
# docs/agent-evals.md) — regression tripwires, not aspirations — and lowering one
# for quiet is the move that hollows out a gate: power depends on the floor, so
# 0.8 → 0.7 halves the gate's chance of catching a real drop to 0.6.
_FLOORS: dict[str, float] = {
    "undo_and_replace": 0.8,
    "my_mistake_is_mine": 0.8,
    "play_as_black": 0.8,
    "resume_not_denied": 0.8,
    "resign_never_pretends": 0.8,
    "advice_is_engine_backed": 0.8,
    "long_resume": 0.8,
    "long_resign": 0.8,
    "long_capture": 0.8,
}

# The loop's budget stops live in `evalstats.BUDGET_STOPS`, next to the other
# decision it drives: a turn that ended on one reached no narrator, so it both
# tested nothing under a commentary check *and* has no narrator round trip to
# attribute latency to. One set, so the two readings cannot disagree.


# --- the report ---------------------------------------------------------------


@dataclass
class _SuiteTotals:
    """What the whole run cost, and the shared ceiling on infra re-runs.

    The budget is suite-wide as well as per-scenario because the failure mode it
    guards against is not one bad scenario but a server that has stopped coming
    back: without a ceiling, a dead llama-server turns the suite into an
    unbounded retry loop that never reports anything.
    """

    scenarios: int = 0
    samples: int = 0
    infra_spent: int = 0
    infra_budget: int = _INFRA_BUDGET

    def spend_infra(self) -> bool:
        """Charge one thrown-away sample; False when the suite is out of budget."""
        self.infra_spent += 1
        return self.infra_spent < self.infra_budget


_SUITE = _SuiteTotals()


def _report(record: dict[str, Any]) -> None:
    """Append one JSONL line, if a report path was asked for.

    Failures are swallowed and printed, like the tracer's: losing a diagnostic
    line must never turn a measured run into a failed one. JSONL and appended as
    each scenario finishes, so a Ctrl-C mid-suite still leaves every completed
    scenario on disk.
    """
    if _REPORT_PATH is None:
        return
    try:
        with open(_REPORT_PATH, "a") as handle:
            handle.write(json.dumps(record) + "\n")
    except OSError as exc:  # pragma: no cover - diagnostics only
        print(f"\n[eval] report write failed: {exc}")


def _git_sha() -> str:
    """The build the numbers belong to. A baseline without one is a rumour."""
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):  # pragma: no cover
        return "unknown"
    return completed.stdout.strip()


@pytest.fixture(scope="module", autouse=True)
def _report_session() -> Generator[None, None, None]:
    """Bracket the run with a header and a totals line.

    The header goes out *before* the first scenario so an interrupted run still
    says what produced its lines — which knobs, which build, which floors. A
    baseline should say what budget produced it, so the closing line carries the
    infra budget actually consumed alongside the wall clock.
    """
    _report(
        {
            "kind": "header",
            "started": datetime.now(UTC).isoformat(),
            "git_sha": _git_sha(),
            "model": LLAMACPP_MODEL,
            "planner_temperature": PLANNER_TEMPERATURE,
            "knobs": {
                "block_runs": _BLOCK_RUNS,
                "max_runs": _MAX_RUNS,
                "infra_retries": _INFRA_RETRIES,
                "infra_budget": _INFRA_BUDGET,
            },
            "floors": _FLOORS,
        }
    )
    started = time.monotonic()
    yield
    _report(
        {
            "kind": "suite",
            "finished": datetime.now(UTC).isoformat(),
            "seconds": round(time.monotonic() - started, 1),
            "scenarios": _SUITE.scenarios,
            "samples": _SUITE.samples,
            "infra_spent": _SUITE.infra_spent,
            "infra_budget": _SUITE.infra_budget,
        }
    )


# --- app / engine fixtures ----------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_rate_limit() -> Generator[None, None, None]:
    # The per-IP limiter is module-global; keep scenarios from bleeding hits
    # into each other (each sends only a message or two, well under the cap).
    reset_rate_limit()
    yield
    reset_rate_limit()


@pytest.fixture(scope="module")
def engine() -> Generator[EnginePlayer, None, None]:
    """One Stockfish process for the whole suite (only built when opted in)."""
    eng = EnginePlayer(path=STOCKFISH_PATH)
    try:
        yield eng
    finally:
        eng.close()


@dataclass
class _CollectingTracer:
    """The app's own `Tracer` seam (`trace.Tracer`), kept in memory.

    `api._run_command` traces **every** route, so this is how the harness learns
    the things the HTTP answer does not carry: the real `stop_reason` on the
    panel seam (`/api/command` genuinely does not return one), plus `route`,
    `mutations`, `guarded`, `model_calls` and `model_ms` on *both* seams. That is
    what lets an infra death be told from a behavioral miss with **zero**
    production change — the alternative was adding `stop_reason` to the panel
    response, which would put a production edit inside a harness slice, and the
    harness-bug precedent (a wrong tool offer silently moving a scenario from
    5/5 to 0–3/5) is the reason that is not allowed here.
    """

    turns: list[dict[str, Any]] = field(default_factory=list)

    def record(self, turn: dict[str, Any]) -> None:
        self.turns.append(turn)

    def reset(self) -> None:
        """One sample measures one utterance — same rule as `provider.reset()`."""
        self.turns.clear()

    @property
    def last(self) -> dict[str, Any]:
        """The turn just run, or an empty record when nothing was traced (a
        request that failed before the pipeline reached its trace point)."""
        return self.turns[-1] if self.turns else {}


class EvalApp(NamedTuple):
    """One assembled eval app: the wire, the state, and the model-call meter."""

    client: TestClient
    ctx: ToolContext
    provider: CountingProvider
    tracer: _CollectingTracer


def _build_eval_app(engine: EnginePlayer) -> EvalApp:
    """A fresh app + game wired exactly like `build_app`, but returning the
    `ToolContext` so a scenario can set up a position (through the session,
    bypassing the engine reply) and read settings/end-state back, plus the
    `CountingProvider` wrapping the live wire so a scenario can assert how many
    times the model was actually called."""
    ctx = ToolContext(session=GameSession(), engine=engine, settings=Settings())
    # Mirror build_app: never leave the engine unconfigured — play at the
    # settings default so reported difficulty and real strength agree.
    if ctx.settings.tier is not None:
        engine.set_tier(ctx.settings.tier)
    # One registry and one turn coordinator, exactly as build_app does: the
    # brain's loop dispatches through the same registry the app runs the fast
    # path through, and `atomic_exchange=False` leaves the engine's reply to the
    # pipeline's close beat. Measuring the atomic tool instead would measure a
    # different sequencing owner than the one that ships.
    coordinator = TurnCoordinator(ctx)
    registry = build_registry(ctx, coordinator, atomic_exchange=False)
    # The only departure from build_app: the real provider is wrapped so every
    # model round trip is counted and timed. create_llama_brain builds exactly
    # this provider when none is passed, so the wire itself is unchanged.
    provider = CountingProvider(LlamaCppProvider(LLAMACPP_BASE_URL, LLAMACPP_MODEL))

    def offered_tools() -> list[dict[str, Any]]:
        # Exactly what build_app offers the brain, resolved live per command
        # through the *same* policy function assembly calls: the board state is
        # injected into its prompt every turn, so BOARD_STATE_TOOLS are
        # dispatchable but not *offered*, and `claim_draw` is withheld until a
        # draw is actually claimable (a capability restriction, not a prompt
        # rule). Shared rather than copied because offering a different list
        # here would measure a different agent than the one that ships — and a
        # bigger tool list is itself a variable (the 2026-07-13 trace review
        # saw capture phrasings behave differently under the two lists).
        return registry.definitions(exclude=brain_tool_exclusions(ctx))

    brain = create_llama_brain(
        base_url=LLAMACPP_BASE_URL,
        model=LLAMACPP_MODEL,
        dispatcher=registry,
        tool_definitions=offered_tools,
        # Both prompts, exactly as build_app wires them: the narrator's carries
        # personality plus the live verbosity layer, the planner's the compact
        # tool contract. Measuring the tool decision under the wrong prompt
        # would measure a different agent.
        system_prompt_provider=lambda: system_prompt_for(ctx.settings.verbosity),
        planner_prompt_provider=lambda: PLANNER_PROMPT,
        planner_temperature=PLANNER_TEMPERATURE,
        provider=provider,
    )
    # The second departure, and it is observation only: the app's existing
    # tracer seam is pointed at a list. Nothing the model sees changes — a
    # tracer is a diagnostic sink the pipeline already swallows failures from.
    tracer = _CollectingTracer()
    client = TestClient(
        create_app(
            ctx,
            brain=brain,
            registry=registry,
            coordinator=coordinator,
            tracer=tracer,
        )
    )
    return EvalApp(client=client, ctx=ctx, provider=provider, tracer=tracer)


@pytest.fixture
def eval_app(engine: EnginePlayer) -> Generator[EvalApp, None, None]:
    app = _build_eval_app(engine)
    try:
        yield app
    finally:
        app.client.close()


# --- trajectory helpers (over the MessageExchange wire) -----------------------


def _tool_calls(assistant: dict[str, Any]) -> list[dict[str, Any]]:
    return assistant["tool_calls"] or []


def _successful(assistant: dict[str, Any], tool: str) -> list[dict[str, Any]]:
    """Calls to `tool` that dispatched cleanly (``error`` is null). A rejected
    move is *not* a dispatch error — it rides on ``result`` as ``legal:false``
    — so this alone does not mean 'the move was made'; see `_legal_moves`."""
    return [
        c for c in _tool_calls(assistant) if c["tool"] == tool and c["error"] is None
    ]


def _legal_moves(assistant: dict[str, Any]) -> list[dict[str, Any]]:
    """Successful make_move calls the engine actually accepted (result
    ``legal:true``) — the board really changed for each of these."""
    out: list[dict[str, Any]] = []
    for call in _successful(assistant, "make_move"):
        if call["result"] and json.loads(call["result"]).get("legal") is True:
            out.append(call)
    return out


def _board_mutations(assistant: dict[str, Any]) -> list[dict[str, Any]]:
    """Successful, board-changing calls this turn (a legal make_move, undo,
    new_game, resign, resume_game). An illegal-move *attempt* is excluded — it
    dispatched but changed nothing."""
    out: list[dict[str, Any]] = []
    for call in _tool_calls(assistant):
        if call["error"] is not None or call["tool"] not in _BOARD_TOOLS:
            continue
        if (
            call["tool"] == "make_move"
            and call["result"]
            and json.loads(call["result"]).get("legal") is not True
        ):
            continue
        out.append(call)
    return out


_MAX_ARG_CHARS = 24


def _arguments(arguments: dict[str, Any]) -> str:
    """What one call asked for, as `(name=value, …)` — or nothing at all when it
    asked for nothing, because empty parens on every read tool would be noise on
    every line.

    Values are compact JSON rather than bare text so a string and a boolean stay
    apart (`enabled="true"` is a mis-invocation; `enabled=true` is the call), and
    keys are sorted so the same call renders the same way in every sample: the
    line is read by scanning for differences between samples, and an ordering
    that tracked the model's emission order would manufacture them. Long values
    are cut visibly — one trajectory shares a terminal line with eleven other
    clauses, and a pasted PGN would cost the whole record.
    """
    rendered: list[str] = []
    for name in sorted(arguments):
        value = json.dumps(arguments[name], separators=(",", ":"), default=str)
        if len(value) > _MAX_ARG_CHARS:
            value = value[: _MAX_ARG_CHARS - 1] + "…"
        rendered.append(f"{name}={value}")
    return f"({', '.join(rendered)})" if rendered else ""


def _trajectory(assistant: dict[str, Any]) -> str:
    """The turn's calls in order, each with the arguments it carried.

    The arguments are the point (TODO.md's "does the planner flip hints on when
    nobody asked?"): `set_hints_mode` on the line said a setting had been
    changed and could not say which way. They ride on the wire already, so
    reading them costs nothing but this rendering.

    Both failure markers are `!` suffixes *after* the arguments — the rejected
    move's used to be `make_move(illegal)`, which alongside real arguments would
    read as an argument named `illegal`.
    """
    tokens: list[str] = []
    for call in _tool_calls(assistant):
        token = call["tool"] + _arguments(call.get("arguments") or {})
        if call["error"] is not None:
            token += "!"
        elif (
            call["tool"] == "make_move"
            and call["result"]
            and json.loads(call["result"]).get("legal") is not True
        ):
            token += "!illegal"
        tokens.append(token)
    return " → ".join(tokens)


def _history(client: TestClient) -> list[str]:
    """The board's SAN move history, read back through the public state seam."""
    return client.get("/api/state").json()["history"]


class EvalRun(NamedTuple):
    """What one scenario measured: the wire's answer, and what it cost.

    `model_calls` is every round trip the brain made to the model for this one
    utterance — the planner loop's turns plus the narrator's closing turn, or
    `narrate`'s single turn on the fast path, or none at all when the fast path
    answers with a canned confirmation.

    `status_code` and `stop_reason` are what a sample is *classified* by, and
    neither is faked: the status is the response's, and the stop reason is read
    off the tracer (the panel seam does not return one). Together they are the
    difference between "the model did nothing" and "llama-server died again".

    `provider_failure` is the third, and it splits that last one: the kind of
    death the brain named, off the same trace record. A crashed socket is worth
    re-taking the sample for; a request llama-server *rejected* (the 400 an
    overrun context gets) is the same answer every time, and retrying it five
    times only spends the infra budget to arrive where it started.

    `latencies` is the turn's per-call wall clock with each reading attributed to
    the phase that spent it (`evalstats.split_latencies`). `duration` is the
    request's whole wall clock and `model_ms` was the turn's model total, and
    neither can answer the question Sprint 5 is asking — whether a `no_progress`
    turn's extra 30 s is the narrator's own round trip or a hard sample that made
    the repeat and the long narration both happen.

    `tokens` is the same turn priced in tokens, split the same way — because the
    latency half answered *where* the extra time goes and cannot answer *why*.
    """

    assistant: dict[str, Any]
    duration: float
    model_calls: list[ModelCall]
    status_code: int
    stop_reason: str | None
    provider_failure: str | None
    latencies: TurnLatencies
    tokens: TurnTokens

    @property
    def narrator_tok_s(self) -> float | None:
        """How fast the narrator wrote, when both halves of the division are
        known. Derived here rather than carried on either half: it is the one
        number that needs a reading from *both* seams — the trace's clock and
        the call meter's usage — so this is the only place it can be computed
        without one of them reaching into the other."""
        return generation_rate(self.tokens.narrator_out, self.latencies.narrator_ms)


# A non-200 means the request never produced a turn (almost always the provider
# dying mid-generation). The sample is classified INFRA and re-taken, so nothing
# reads this stub — it exists so the failure reaches the printed line and the
# report rather than an exception off a missing key.
_NO_TURN: dict[str, Any] = {"content": "", "tool_calls": [], "stop_reason": None}


def _run(app: EvalApp, scenario: str, utterance: str) -> EvalRun:
    """One eval run against the live model: fresh conversation → one message →
    the `[eval]` stats line the baseline table is built from.

    Deliberately does **not** assert on the status: a pass-rate scenario has to
    be able to retry an infra death, and an assertion here would make that death
    a behavioral miss. The assertion moved to `_run_once`, which the single-shot
    scenarios use.
    """
    conversation_id = app.client.post("/api/agent/conversations", json={}).json()["id"]
    app.provider.reset()  # measure this utterance, not the fixture's history
    app.tracer.reset()
    started = time.monotonic()
    response = app.client.post(
        f"/api/agent/conversations/{conversation_id}/messages",
        json={"content": utterance},
        timeout=_REQUEST_TIMEOUT,
    )
    duration = time.monotonic() - started
    assistant = (
        response.json()["assistant_message"]
        if response.status_code == 200
        else dict(_NO_TURN)
    )
    return _measured(app, scenario, assistant, response, duration)


def _measured(
    app: EvalApp,
    scenario: str,
    assistant: dict[str, Any],
    response: Any,
    duration: float,
) -> EvalRun:
    """The `[eval]` line and the `EvalRun`, shared by both seams.

    The line now carries what the tracer knows — the route the utterance took,
    the real stop reason, the board mutations counted off `board_version`, and
    `model_ms`. The last one is not decoration: a prompt-cache hit is *fast*, so
    correlating an outcome with the turn's model latency tests the run-order
    hypothesis behind `play_as_black`'s 5/5-in-isolation / 0/5-mid-suite split at
    zero GPU cost.

    **And it now says which model call spent that time** (Sprint 5). The turn
    total was hiding the finding it was asked about: `no_progress` turns narrate
    2–3× slower than `completed` ones at the same call count, and a sum cannot
    tell a slow planner from a slow narrator. `model_latencies_ms` was already
    on the trace record, one reading per round trip in call order — the harness
    just never printed it. `split_latencies` attributes each reading off the
    route and the stop reason (`evalstats`, unit-tested off the GPU) rather than
    assuming the last call is always the narrator, which on a budget stop or a
    provider death it is not.
    """
    traced = app.tracer.last
    model_calls = list(app.provider.calls)
    thinking = ",".join("on" if call.thinking else "off" for call in model_calls)
    # Read off the trace, like `model_ms` beside it: the brain measures a round
    # trip that *raised* and the provider seam cannot (only the caller of a
    # raising call is still there to stop the clock), so the trace is the only
    # reading that covers a died turn. `stop_reason` and `route` come from the
    # same record, so the attribution and the readings can never be from
    # different turns.
    latencies = split_latencies(
        traced.get("model_latencies_ms", ()),
        route=traced.get("route"),
        stop_reason=traced.get("stop_reason"),
    )
    # Tokens come off the *call meter* rather than the trace, because the trace
    # sums the turn and a sum is exactly what could not answer the last
    # question. The two seams are not cross-checked here on purpose: they are
    # printed side by side with `model_calls` between them, so a disagreement
    # about how many round trips happened shows up on the line rather than being
    # silently reconciled into a wrong attribution.
    tokens = split_tokens(
        [(call.prompt_tokens, call.completion_tokens) for call in model_calls],
        route=traced.get("route"),
        stop_reason=traced.get("stop_reason"),
    )
    tok_s = generation_rate(tokens.narrator_out, latencies.narrator_ms)
    print(
        f"\n[eval] scenario={scenario} status={response.status_code} "
        f"stop={traced.get('stop_reason')} route={traced.get('route')} "
        f"calls={len(_tool_calls(assistant))} model_calls={len(model_calls)} "
        f"thinking=[{thinking}] model_ms={traced.get('model_ms')} "
        f"{latencies.summary()} {tokens.summary()} "
        f"narrator_tok_s={'?' if tok_s is None else f'{tok_s:.1f}'} "
        f"mutations={traced.get('mutations', len(_board_mutations(assistant)))} "
        f"duration={duration:.1f}s trajectory=[{_trajectory(assistant)}]"
    )
    if response.status_code != 200:
        print(f"[eval]   ! HTTP {response.status_code}: {response.text[:300]}")
    provider_failure = traced.get("provider_failure") or None
    if provider_failure is not None:
        print(f"[eval]   ! provider failure: {provider_failure}")
    return EvalRun(
        assistant=assistant,
        duration=duration,
        model_calls=model_calls,
        status_code=response.status_code,
        stop_reason=traced.get("stop_reason"),
        provider_failure=provider_failure,
        latencies=latencies,
        tokens=tokens,
    )


def _run_once(app: EvalApp, scenario: str, utterance: str) -> EvalRun:
    """One sample, for a scenario that only takes one — and the two ways such a
    sample can be worthless.

    The status assertion `_run` used to carry lives here rather than being
    deleted: the shape scenarios have no rate for an infra death to be absorbed
    into, so for them it has to be loud. `provider_error` is asserted beside it
    because since audit item 20 that *is* the crash, wearing a 200: the brain
    catches `ProviderError` and answers with whatever verifiably ran, which for
    a shape scenario is an unmeasured model rather than a measured miss.
    """
    run = _run(app, scenario, utterance)
    assert run.status_code == 200, (
        f"{scenario}: HTTP {run.status_code} — see the [eval] line above"
    )
    assert run.stop_reason != STOP_PROVIDER_ERROR, (
        f"{scenario}: the provider died mid-turn ({run.provider_failure}), so "
        "nothing was measured"
    )
    return run


def _assert_loop_budget(run: EvalRun) -> None:
    """The turn stayed inside its bound. Scenarios whose trajectory is fixed
    pin an exact count instead; this is for the ones the model may legitimately
    answer in two turns (decline, then narrate) or four (a read, an act, a note,
    the reply) — the shape that matters there is only that it terminated on its
    own, not on the budget.

    The ceiling is the planner's iteration budget **plus one**: a turn that ends
    on its own always pays for the narrator, and a turn that ends on a budget
    never reaches it (so it can't exceed the budget either way)."""
    ceiling = _DEFAULT_MAX_ITERATIONS + 1
    assert 1 <= len(run.model_calls) <= ceiling, (
        f"expected 1..{ceiling} model calls, got {len(run.model_calls)}"
    )


def _assert_thinking_starts_off(run: EvalRun) -> None:
    """The policy's floor: the turn that decides which tool to call is a fast
    parse, never a reasoning turn (`llama_brain._thinking`). Thinking may only
    come on *later*, once an analysis tool's result has landed in context."""
    assert run.model_calls, "expected at least one model call"
    assert run.model_calls[0].thinking is False, (
        "the first turn must run with thinking OFF"
    )


# --- difficulty ordering (scenario 4) -----------------------------------------

# Comparable strength for ordering only. Tiers map to their documented target
# elos; raw knobs to an approximate elo (Stockfish skill 0≈800 … 20≈3000).
_TIER_STRENGTH = {
    "beginner": 500,
    "casual": 1000,
    "intermediate": 1500,
    "advanced": 2000,
    "maximum": 3200,
}


def _difficulty_strength(settings: Settings) -> float:
    if settings.tier is not None:
        return float(_TIER_STRENGTH[settings.tier])
    if settings.elo is not None:
        return float(settings.elo)
    if settings.skill_level is not None:
        return 800.0 + settings.skill_level * 110.0
    return float(_TIER_STRENGTH[DEFAULT_TIER])


# --- scenarios ----------------------------------------------------------------


def test_eval_fast_path_plain_move_is_zero_llm(eval_app: EvalApp) -> None:
    """The mirror image of every other scenario: an utterance that *does* parse
    as exactly one legal move never reaches the model at all.

    At verbosity=low the fast path dispatches make_move and answers with a
    canned confirmation (`api._move_confirmation`), so a plain move costs zero
    model round trips — the cheapest path in the app, and the reason the loop's
    iteration budget never applies to ordinary moves."""
    app = eval_app
    app.ctx.settings.verbosity = "low"  # Settings default is "normal"
    utterance = "e4"
    assert parse_move(utterance, app.ctx.session.fen()) is not None  # takes fast path

    run = _run_once(app, "fast_path_low", utterance)

    assert run.model_calls == [], "a fast-path move at verbosity=low must be zero-LLM"
    assert run.assistant["stop_reason"] == "completed"
    history = _history(app.client)
    assert history[0] == "e4"
    assert len(history) == 2, "engine should have replied in the same turn"
    assert run.assistant["content"], "expected the canned confirmation"


def test_eval_fast_path_move_costs_one_call_when_chatty(eval_app: EvalApp) -> None:
    """Above verbosity=low the fast path still skips the *planner* — it
    dispatches the move deterministically and pays for commentary only: exactly
    one model call (`Brain.narrate`, the narrator phase on its own), thinking
    off, and no tool-decision turn. Unchanged by the planner/narrator split:
    this path was always narrator-only, which is what made it the model for
    the split."""
    app = eval_app
    assert app.ctx.settings.verbosity == "normal"  # the default
    utterance = "e4"
    assert parse_move(utterance, app.ctx.session.fen()) is not None

    run = _run_once(app, "fast_path_normal", utterance)

    assert len(run.model_calls) == 1, "expected narrate only — no tool-decision turn"
    _assert_thinking_starts_off(run)
    assert run.duration < _THINKING_OFF_CEILING_S
    assert _history(app.client)[0] == "e4"
    assert run.assistant["content"], "expected commentary"


def test_eval_plain_move_via_the_agent_path(eval_app: EvalApp) -> None:
    """A plain move that the parser deliberately lets through ("play e4" — the
    leading verb keeps it off the fast path) becomes exactly one legal
    make_move, and the board starts with e4 (plus the engine's reply).

    Also a brain-routed turn's minimum cost: the planner's tool turn, the
    planner's handoff note, and the narrator's reply — three model round trips,
    all thinking-off."""
    app = eval_app
    utterance = "play e4"
    assert parse_move(utterance, app.ctx.session.fen()) is None  # stays a model eval

    run = _run_once(app, "plain_move", utterance)
    assistant = run.assistant

    assert assistant["stop_reason"] == "completed"
    assert len(_legal_moves(assistant)) == 1, "expected exactly one legal make_move"
    history = _history(app.client)
    assert history[0] == "e4"
    assert len(history) == 2, "engine should have replied in the same turn"
    assert assistant["content"], "expected a non-empty reply"
    assert len(run.model_calls) == 3, (
        "the minimum: planner tool turn + planner note + narrator"
    )
    _assert_thinking_starts_off(run)
    assert all(call.thinking is False for call in run.model_calls), (
        "a move is not analysis — thinking stays OFF for the whole run"
    )
    assert run.duration < _THINKING_OFF_CEILING_S


def test_eval_judgment_question_routes_through_analysis(eval_app: EvalApp) -> None:
    """A judgment question a few moves in is answered by a read — evaluate_position
    or analyze_last_move — never from vibes, and nothing on the board moves.

    This is also the one scenario that exercises the thinking policy end-to-end
    against the live model: the turn that *picks* the analysis tool runs with
    thinking off (it is just a parse), and the turn that reasons about the
    result it got back runs with thinking on. Pinned here because
    test_llama_brain only pins it against a scripted provider. The last call is
    now the narrator's, and it inherits the flip — putting an evaluation into
    words is analysis work too."""
    app = eval_app
    for san in ("e4", "e5", "Nf3", "Nc6"):  # a natural position, White to move
        assert app.ctx.session.submit_move(san).legal
    before = _history(app.client)
    utterance = "how am I doing?"
    assert parse_move(utterance, app.ctx.session.fen()) is None

    run = _run_once(app, "judgment_question", utterance)
    assistant = run.assistant

    assert assistant["stop_reason"] == "completed"
    analysed = _successful(assistant, "evaluate_position") + _successful(
        assistant, "analyze_last_move"
    )
    assert analysed, "judgment question must route through an analysis tool"
    assert _board_mutations(assistant) == [], "a read-only question must not mutate"
    assert _history(app.client) == before
    assert assistant["content"], "expected a non-empty reply"

    # The OFF → ON flip, live. The analysis result lands during the first turn,
    # so everything after it — including the narrator, the turn that actually
    # puts the judgment into words — thinks.
    assert len(run.model_calls) >= 3, "expected a tool turn, a note, and the narrator"
    _assert_thinking_starts_off(run)
    assert run.model_calls[-1].thinking is True, (
        "the turn commenting on an analysis result must run with thinking ON"
    )
    assert run.duration < _ANALYSIS_CEILING_S


def test_eval_ambiguous_move_asks_instead_of_guessing(eval_app: EvalApp) -> None:
    """ "move the rook" in a position with several mobile rooks is genuinely
    ambiguous — the agent must ask, not guess a move. (1. a4 a5 2. h4 h5 opens
    both White rook files; White to move, four rook moves available.)

    The cheapest brain-routed turn there is: the planner declines to call
    anything and says what to ask, then the narrator asks it — two calls, no
    tools dispatched."""
    app = eval_app
    for san in ("a4", "a5", "h4", "h5"):
        assert app.ctx.session.submit_move(san).legal
    before = _history(app.client)
    utterance = "move the rook"
    assert parse_move(utterance, app.ctx.session.fen()) is None

    run = _run_once(app, "ambiguous_move", utterance)
    assistant = run.assistant

    assert _legal_moves(assistant) == [], "must not guess a move when ambiguous"
    assert _board_mutations(assistant) == []
    assert _history(app.client) == before
    assert assistant["content"], "expected a clarifying question"
    assert len(run.model_calls) == 2, (
        "expected the planner's decline plus the narrator's question, "
        f"got {len(run.model_calls)} calls"
    )
    _assert_thinking_starts_off(run)
    assert run.duration < _THINKING_OFF_CEILING_S


def test_eval_settings_by_speech_makes_it_easier(eval_app: EvalApp) -> None:
    """ "make it easier" calls set_difficulty toward a weaker setting than the
    casual default, and touches no piece."""
    app = eval_app
    assert app.ctx.settings.tier == DEFAULT_TIER  # baseline strength
    before = _history(app.client)
    utterance = "make it easier"
    assert parse_move(utterance, app.ctx.session.fen()) is None

    run = _run_once(app, "settings_by_speech", utterance)
    assistant = run.assistant

    assert assistant["stop_reason"] == "completed"
    assert _successful(assistant, "set_difficulty"), "expected a set_difficulty call"
    assert _difficulty_strength(app.ctx.settings) < _TIER_STRENGTH[DEFAULT_TIER], (
        f"expected a weaker setting than {DEFAULT_TIER}: "
        f"tier={app.ctx.settings.tier} skill={app.ctx.settings.skill_level} "
        f"elo={app.ctx.settings.elo}"
    )
    assert _board_mutations(assistant) == []
    assert _history(app.client) == before
    assert len(run.model_calls) == 3, (
        "the minimum: planner tool turn + planner note + narrator"
    )
    assert all(call.thinking is False for call in run.model_calls), (
        "a settings change is not analysis — thinking stays OFF"
    )
    assert run.duration < _THINKING_OFF_CEILING_S


def test_eval_honest_about_an_illegal_move(eval_app: EvalApp) -> None:
    """ "castle kingside" is illegal on move 1 (parser returns None — no castling
    is legal). The agent must not fake it: the board doesn't change and no
    legal move is made. An attempted-and-rejected make_move is acceptable —
    only the board-didn't-change invariant is asserted, never the wording.

    The trajectory here is legitimately variable (attempt-and-concede, or read
    the legal moves first), so the round-trip assertion is the budget, not an
    exact count — the point is that the rejection is absorbed *inside* the loop
    and it still terminates on its own, never on `max_iterations`."""
    app = eval_app
    before = _history(app.client)
    assert before == []
    utterance = "castle kingside"
    assert parse_move(utterance, app.ctx.session.fen()) is None

    run = _run_once(app, "honest_illegal", utterance)
    assistant = run.assistant

    assert _legal_moves(assistant) == [], "must not fabricate a legal move"
    assert _board_mutations(assistant) == []
    assert _history(app.client) == before
    assert assistant["content"], "expected a reply explaining the situation"
    assert assistant["stop_reason"] == "completed", "must not stop on a budget"
    _assert_loop_budget(run)
    _assert_thinking_starts_off(run)
    assert run.duration < _THINKING_OFF_CEILING_S


def test_eval_destructive_op_asks_before_acting(eval_app: EvalApp) -> None:
    """ "new game" mid-game is destructive: it must not fire on the first ask,
    and the player's "yes" must then actually reset the board.

    This was the harness's one xfail — gemma-4-12b honored the prompt's
    confirmation rule only ~half the time (~50% across both a 2-ply stub and
    this 10-ply developed game; position depth didn't move the rate). It is a
    hard assert now because the rule is no longer the model's to honor: the tool
    gate refuses an unconfirmed new_game and arms it, and the pipeline runs it
    on a bare "yes" with no model call (tools.py `_gate` / `confirm_pending`).

    So what this asserts is no longer adherence but the shape of the turn: the
    model still has to *call* new_game (that is what arms the gate) and then
    relay the refusal as a question rather than pretending it succeeded. The
    board being intact is guaranteed underneath either way. Still a live-model
    test: a model that never calls the tool, or that lies about the outcome,
    fails here.

    The game is a substantial one (10-ply Ruy Lopez, castled, developed) so this
    isn't dismissible as a two-move stub the model reasonably resets."""
    app = eval_app
    for san in ("e4", "e5", "Nf3", "Nc6", "Bb5", "a6", "Ba4", "Nf6", "O-O", "Be7"):
        assert app.ctx.session.submit_move(san).legal
    before = _history(app.client)
    assert not app.ctx.session.is_game_over()  # a real game stands to be lost
    utterance = "new game"
    assert parse_move(utterance, app.ctx.session.fen()) is None

    run = _run_once(app, "destructive_confirm", utterance)
    assistant = run.assistant

    assert _successful(assistant, "new_game") == [], (
        "new_game must wait for a confirmation, not fire on the first ask"
    )
    assert _board_mutations(assistant) == []
    assert _history(app.client) == before
    assert assistant["content"], "expected a confirmation question"
    assert app.ctx.pending is not None, "the refused op must be armed for the yes"

    # The other half of the gate: the answer. Deterministic — no model call
    # stands between the player's yes and the reset.
    app.client.post("/api/command", json={"text": "yes"})

    assert app.ctx.session.move_history() == [], "confirmed: the game really resets"
    assert app.ctx.pending is None


# --- move correctness: the pass-rate scenarios --------------------------------
#
# Everything above asserts trajectory *shape* — the right tool family ran, the
# board did or didn't move. None of it asks the harder question: when the model
# picks the move, does it pick the *right* one? Until the 2026-07-13 trace
# review (`docs/agent-trace-review-2026-07-13.md`) exactly one scenario had the
# model choose a move at all ("play e4", from the starting position), so move
# correctness through the model was essentially unmeasured — and live play was
# failing at it constantly.
#
# These scenarios come straight out of that review: each is a real position and
# a real utterance that misfired, replayed. They assert the *specific SAN* that
# should land, which means they are the first goldens that can fail on the model
# being wrong rather than on it calling the wrong tool.
#
# They run N times and report a **pass rate**, not a boolean. The model samples
# at temp 1.0: a single assert on a path that works 70% of the time flaps, and a
# flapping test teaches you nothing. A rate tells you whether a prompt change
# moved the number. The floors (`_FLOORS`) are deliberately set at what the
# current build actually achieves (recorded in docs/agent-evals.md) — they are
# regression tripwires, not aspirations.
#
# What the rate is *compared against* changed in Sprint 5 slice 3, and it is the
# one thing in this section worth reading twice: five samples cannot resolve a
# 60–100% band, so `rate >= floor` was a coin flip on exactly the scenarios that
# matter. The verdict is now a one-sided Wilson bound with block-sequential
# sampling (`evalstats.decide`), and `RateResult` — which used to be three
# fields here — moved to `evalstats.py` where it can be unit-tested without a
# GPU.


def _run_panel(app: EvalApp, scenario: str, utterance: str) -> EvalRun:
    """One turn through `POST /api/command` — the **web panel's** seam, not the
    delegate's.

    The difference is the transcript, and it is the whole point of the
    long-transcript scenarios. The delegate endpoint carries its own
    per-conversation history (`_run` above opens a fresh one every time, so the
    model sees an empty thread); `/api/command` reads `ctx.transcript.memory()`,
    the running conversation the player has actually been having — recent turns
    verbatim behind a digest of the older asks (`docs/turn-memory.md`), which is
    why the `poisoned` conditions below seed their poison at the *end* of the
    thread: a stale assistant line only poisons what it is still quoted in.
    Every failure
    in the 2026-07-13 trace review happened on *this* seam, deep into a thread —
    and none of them reproduce on a fresh delegate conversation.

    The panel returns a thinner document than the delegate wire (`tool_results`
    is `{name, result}` with no arguments and no separate error channel, and its
    `result` is a decoded dict rather than the wire's JSON string), so it is
    adapted into the same shape the trajectory helpers above read.

    It no longer *invents* a stop reason. This adapter used to carry
    `"stop_reason": "completed"` with the comment "not exposed on this seam",
    which is true of the HTTP answer and was quietly asserting that every panel
    turn completed — including the ones where the provider had died. The real
    stop reason comes off the tracer now (`_measured`), and the key is simply
    gone from the adapted document, because a value the seam does not carry
    should not be readable from it at all."""
    app.provider.reset()
    app.tracer.reset()
    started = time.monotonic()
    response = app.client.post(
        "/api/command", json={"text": utterance}, timeout=_REQUEST_TIMEOUT
    )
    duration = time.monotonic() - started
    if response.status_code == 200:
        body = response.json()
        assistant = {
            "content": body["commentary"],
            # The panel has no error channel: a dispatch failure rides on
            # `result`.
            "tool_calls": [
                {
                    "tool": r["name"],
                    "arguments": {},
                    "result": json.dumps(r["result"]),
                    "error": None,
                }
                for r in body["tool_results"]
            ],
        }
    else:
        assistant = {"content": "", "tool_calls": []}
    return _measured(app, scenario, assistant, response, duration)


def _assert_reached_narrator(run: EvalRun) -> None:
    """The turn got far enough for a commentary check to mean anything.

    A budget stop reaches no narrator by construction (`docs/planner-narrator.md`
    — nothing verified came back to speak from), so a *purely negative* check
    over commentary passes for never having been tested. That is not a thought
    experiment: three of four `hints_off_no_advice` passes in one recorded run
    were `max_iterations` stops (`docs/agent-evals.md`), which is part of why
    that scenario's rate read as noise.

    Raises `VacuousRun` — an `AssertionError` subclass `classify` reads as
    INCONCLUSIVE, so the sample counts against the rate and is reported as what
    it was, rather than either passing silently or being blamed on the model.
    """
    if run.stop_reason in BUDGET_STOPS:
        raise VacuousRun(
            f"stopped on {run.stop_reason}: no narrator ran, so nothing was tested"
        )
    if not run.assistant["content"]:
        raise VacuousRun("no commentary was produced, so nothing was tested")


def _sample(
    engine: EnginePlayer,
    label: str,
    utterance: str,
    check: Callable[[EvalApp, dict[str, Any]], None],
    setup: Callable[[EvalApp], None] | None,
    runner: Callable[[EvalApp, str, str], EvalRun],
    requires_narrator: bool,
) -> tuple[Outcome, EvalRun, BaseException | None]:
    """One sample on a fresh app, and what kind of sample it turned out to be.

    The check is only run on a turn that actually happened. On a dead provider it
    would raise off the `_NO_TURN` stub and read as a HARNESS bug — which
    `classify` deliberately never retries — so the guard here is what keeps a
    crash classified as the crash it is.
    """
    app = _build_eval_app(engine)
    try:
        if setup is not None:
            setup(app)
        reset_rate_limit()  # each sample is its own conversation
        run = runner(app, label, utterance)
        error: BaseException | None = None
        if run.status_code == 200 and run.stop_reason != STOP_PROVIDER_ERROR:
            try:
                if requires_narrator:
                    _assert_reached_narrator(run)
                check(app, run.assistant)
            except Exception as exc:
                # Everything a check can raise is data about *something*: a
                # failed assertion is the model, a VacuousRun is the budget, and
                # anything else is a bug in this file. KeyboardInterrupt is
                # deliberately not caught — a Ctrl-C mid-suite must abort, with
                # every finished scenario already on disk.
                error = exc
        outcome = classify(
            status_code=run.status_code,
            stop_reason=run.stop_reason,
            error=error,
            provider_failure=run.provider_failure,
        )
        return outcome, run, error
    finally:
        app.client.close()


def _pass_rate(
    engine: EnginePlayer,
    scenario: str,
    utterance: str,
    check: Callable[[EvalApp, dict[str, Any]], None],
    *,
    floor: float,
    setup: Callable[[EvalApp], None] | None = None,
    runner: Callable[[EvalApp, str, str], EvalRun] = _run,
    requires_narrator: bool = False,
) -> RateResult:
    """Sample one scenario in blocks until its count decides against `floor`.

    A block of `_BLOCK_RUNS` is taken, then `evalstats.decide`: a Wilson upper
    bound below the floor is red and stops, a point estimate at the floor is
    green and stops, and only the genuinely ambiguous middle buys another block
    (up to `_MAX_RUNS`). At the default five that means a healthy scenario costs
    exactly what it cost before this slice, and the escalation is spent only
    where the old literal was a coin flip.

    Blocks rather than one sample at a time for three reasons: identical
    operating characteristics, a decision table small enough to unit-test, and
    per-block rates — which are the only measurement in this suite that can see
    the run-order confound (`play_as_black` at 5/5 isolated and 0/5 mid-suite).
    `evalstats.block_stability` flags it; nothing here fixes it, because
    interleaving samples across scenarios needs a session-scoped sampler this
    slice does not build.

    An infra death (a 502, or the 200 that carries `stop_reason="provider_error"`
    naming a *transient* failure) is thrown away and re-taken without consuming a
    sample, bounded per scenario and across the suite. Exhausting either is
    `INFRA_ABORTED` — no rate, a hard failure — and it now means what it says:
    llama-server is not staying up. A death the brain names deterministic (an
    HTTP 400 on an overrun context, a body that fails wire validation) is
    `PROVIDER_REJECTED` and fails on the *first* sample instead, because five
    identical refusals are one finding and cost the suite's whole infra budget
    to reach.

    `runner` picks the seam: `_run` (the delegate wire, fresh conversation each
    time) or `_run_panel` (the web panel, reading whatever `setup` left on
    `ctx.transcript`). `requires_narrator` adds the vacuity check for a scenario
    whose verdict is only about commentary.
    """
    blocks: list[tuple[int, int]] = []
    failures: list[str] = []
    samples: list[dict[str, Any]] = []
    infra = 0
    inconclusive = 0
    taken = 0
    started = time.monotonic()

    def record(**meta: Any) -> dict[str, Any]:
        """Assemble the scenario's report line from whatever is settled so far,
        so an abort is as readable as a completion."""
        return {
            "kind": "scenario",
            "seconds": round(time.monotonic() - started, 1),
            "seam": "panel" if runner is _run_panel else "delegate",
            "utterance": utterance,
            "samples": samples,
            **meta,
        }

    while True:
        block_passed = 0
        block_runs = 0
        while block_runs < _BLOCK_RUNS:
            taken += 1
            outcome, run, error = _sample(
                engine,
                f"{scenario}[{taken}]",
                utterance,
                check,
                setup,
                runner,
                requires_narrator,
            )
            _SUITE.samples += 1
            samples.append(
                {
                    "outcome": str(outcome),
                    "status_code": run.status_code,
                    "stop_reason": run.stop_reason,
                    "provider_failure": run.provider_failure,
                    "model_calls": len(run.model_calls),
                    "seconds": round(run.duration, 1),
                    # The turn's model time, per call and per phase. Per sample
                    # rather than aggregated because the whole question is
                    # whether the slow samples and the `no_progress` samples are
                    # the *same* samples — which needs the pairing kept, and
                    # which is why the medians in TODO.md's repeat-stop item had
                    # to be read out of terminal scrollback.
                    **run.latencies.as_record(),
                    # And what each of those calls wrote. Milliseconds alone
                    # cannot tell a wordier narration from a slower one; the two
                    # records side by side, per sample, can.
                    **run.tokens.as_record(),
                    "narrator_tok_s": run.narrator_tok_s,
                    # And what it did, with the arguments. A per-run count of
                    # "how many samples called `set_hints_mode`, and which way"
                    # is the question this record answers off the report file;
                    # off scrollback it is a hand tally across four runs.
                    "trajectory": _trajectory(run.assistant),
                    "error": str(error) if error is not None else None,
                }
            )
            if outcome is Outcome.PROVIDER_REJECTED:
                # Answered, just not with a completion — and it will answer the
                # same way next time. Reported and failed on the *first* one:
                # the old code could only reach this conclusion by spending five
                # samples and the suite's infra budget proving it.
                _report(
                    record(
                        scenario=scenario,
                        floor=floor,
                        decision="PROVIDER_REJECTED",
                        infra=infra,
                        provider_failure=run.provider_failure,
                    )
                )
                pytest.fail(
                    f"{scenario}: llama-server refused the request "
                    f"({run.provider_failure}) — no rate was measured, and a "
                    "retry would get the same refusal. Not the crash cadence: "
                    "look at the request (a context overrun on a long "
                    "transcript is the usual one), not at the server."
                )
            if outcome is Outcome.INFRA:
                # Not a sample: the provider died, so the model was never asked.
                infra += 1
                budget_left = _SUITE.spend_infra()
                print(f"[eval]   ⟳ infra retry {infra}/{_INFRA_RETRIES} ({scenario})")
                if infra >= _INFRA_RETRIES or not budget_left:
                    _report(
                        record(
                            scenario=scenario,
                            floor=floor,
                            decision="INFRA_ABORTED",
                            infra=infra,
                            suite_infra_spent=_SUITE.infra_spent,
                        )
                    )
                    pytest.fail(
                        f"{scenario}: INFRA_ABORTED after {infra} provider deaths "
                        f"(suite budget spent {_SUITE.infra_spent}/"
                        f"{_SUITE.infra_budget}). No rate was measured: "
                        "llama-server is not staying up. A deterministic "
                        "refusal is no longer folded in here — that fails as "
                        "PROVIDER_REJECTED on the first sample."
                    )
                continue
            if outcome is Outcome.HARNESS:
                # A bug in this file, not data about the model. Reported before
                # it propagates, so the samples already taken are not lost —
                # which is exactly what used to happen.
                _report(
                    record(
                        scenario=scenario,
                        floor=floor,
                        decision="HARNESS_ERROR",
                        infra=infra,
                        error=str(error),
                    )
                )
                assert error is not None  # HARNESS is only reachable with one
                raise error
            block_runs += 1
            if outcome is Outcome.PASS:
                block_passed += 1
            else:
                failures.append(f"run {taken} [{outcome}]: {error}")
                if outcome is Outcome.INCONCLUSIVE:
                    inconclusive += 1
        blocks.append((block_passed, block_runs))
        result = RateResult.from_blocks(
            blocks=blocks,
            failures=failures,
            floor=floor,
            infra=infra,
            inconclusive=inconclusive,
        )
        if result.decision is not Decision.UNDECIDED or result.runs >= _MAX_RUNS:
            break

    _SUITE.scenarios += 1
    print(
        f"\n[eval] scenario={scenario} PASS_RATE={result.summary()} floor={floor:.0%}"
    )
    for failure in result.failures:
        print(f"[eval]   ✗ {failure}")
    for mode, count in sorted(result.failure_modes.items(), key=lambda item: -item[1]):
        print(f"[eval]   ×{count} {mode}")
    _report(record(**scenario_record(scenario, result, floor)))
    return result


def _assert_floor(result: RateResult, floor: float, *, blocking: bool = False) -> None:
    """The verdict, replacing nine copies of `assert result.rate >= 0.8`.

    Red takes evidence: only a Wilson upper bound *below* the floor fails the
    item, so a bad sample is no longer a regression. A count that never resolved
    (the interval still straddles the floor at the cap) is reported as
    UNDECIDED — and passes, except where the item is release-blocking, because
    `TODO.md` declares `long_capture` a release blocker and an unresolved gate on
    a release blocker is not a pass.

    What a green now means is therefore weaker than it looked and stronger than
    it was: **not statistically below the floor**. The gate separates ≈0.95 from
    ≈0.5–0.6 and nothing finer (`docs/agent-evals.md` records the power table).
    """
    detail = f"{result.summary()} floor={floor:.0%}"
    assert result.decision is not Decision.BELOW_FLOOR, (
        f"below the floor with evidence, not noise: {detail}"
    )
    if result.decision is Decision.UNDECIDED:
        message = f"UNDECIDED at the sampling cap: {detail}"
        if blocking:
            pytest.fail(f"release-blocking scenario is unresolved — {message}")
        print(f"[eval]   ⚠ {message}")
    if result.inconclusive:
        print(
            f"[eval]   ⚠ {result.inconclusive} sample(s) never reached a narrator "
            "and were counted as non-passes"
        )


def _played(app: EvalApp, assistant: dict[str, Any]) -> str | None:
    """The single SAN this turn actually put on the board, or None. Reads the
    board back rather than trusting the trajectory: what the player cares about
    is the move that landed, not the call that claimed it."""
    moves = _legal_moves(assistant)
    if len(moves) != 1:
        return None
    return json.loads(moves[0]["result"])["san"]


def _expect_san(expected: str) -> Callable[[EvalApp, dict[str, Any]], None]:
    def check(app: EvalApp, assistant: dict[str, Any]) -> None:
        played = _played(app, assistant)
        assert played == expected, f"expected {expected}, board got {played!r}"

    return check


# The capture-phrasing family (`capture_bare_bishop`, `capture_bare_pawn`,
# `capture_names_victim`, `capture_names_square`) used to live here. It is gone
# because the bug is gone: `fastparse.parse_move` now settles every capture
# phrasing built on takes/captures — bare ("bishop takes"), victim-named
# ("bishop takes pawn") or square-named ("take the h6 pawn") — whenever exactly
# one legal capture fits, so the model is never asked and there is nothing left
# to sample. The coverage moved to `tests/test_fastparse.py`, where it is free,
# deterministic and always run. Two or more legal captures still fit the phrase?
# Then it is genuinely ambiguous, `parse_move` returns None, and the agent asks —
# which is what `move_ambiguous` already pins.


def test_eval_undo_and_replace_is_one_turn(engine: EnginePlayer) -> None:
    """ "take that back and play X instead" — two actions, one utterance. The
    loop already allows it (it dispatches every call in a model turn and can
    chain across turns), and the trace review confirmed it *works*: undo popped
    the full exchange and make_move landed d4, in a single turn.

    That made TODO #4 a measurement gap rather than a code gap — so this pins
    it before a prompt change quietly takes it away. Asserts both tools ran, in
    order, and the end position is the one the player asked for."""

    def setup(app: EvalApp) -> None:
        for san in ("e4", "b6", "Nf3", "h6", "Bc4", "a5"):
            assert app.ctx.session.submit_move(san).legal

    def check(app: EvalApp, assistant: dict[str, Any]) -> None:
        trajectory = [c["tool"] for c in _tool_calls(assistant)]
        assert "undo" in trajectory, "expected the takeback"
        assert "make_move" in trajectory, "expected the replacement move"
        assert trajectory.index("undo") < trajectory.index("make_move"), (
            f"undo must precede the replacement: {trajectory}"
        )
        history = _history(app.client)
        # The bishop move and the engine's reply are gone; d4 stands in their
        # place (plus whatever the engine answered it with).
        assert history[:5] == ["e4", "b6", "Nf3", "h6", "d4"], history
        assert "Bc4" not in history, "the takeback did not stick"

    floor = _FLOORS["undo_and_replace"]
    result = _pass_rate(
        engine,
        "undo_and_replace",
        "take that bishop move back and play d4 instead",
        check,
        floor=floor,
        setup=setup,
    )

    _assert_floor(result, floor)


def test_eval_what_was_my_mistake_analyzes_the_players_move(
    engine: EnginePlayer,
) -> None:
    """ "what was my mistake?" must analyze the move the *player* played, not
    the engine's reply.

    The player only ever asks this on their own turn — which is exactly when the
    last ply is the engine's. Setup leaves White (the player) having just hung a
    pawn with c3 and Black having taken it with Bxe4; the honest answer is about
    c3.

    Was xfailed (trace review, finding 1): `analyze_last_move()` took no
    arguments and always analyzed the literal last ply, so it could not answer
    its own docstring's question and the model fabricated compliance — told "I
    meant MY last move, the c3 one", it re-called the same no-arg tool and
    reported the engine's move as if it were c3. The tool now defaults to
    `session.player_color`, so the model no longer has to express something the
    signature couldn't say."""

    def setup(app: EvalApp) -> None:
        # The exact game from the trace review: White hangs the e4 pawn with c3
        # and Black's bishop takes it. The player (White) is now to move — which
        # is the only time they ever ask this question.
        for san in (
            "e4", "b6", "Nf3", "h6", "d4", "a5", "Bc4", "e6",
            "O-O", "Bb7", "c3", "Bxe4",
        ):  # fmt: skip
            assert app.ctx.session.submit_move(san).legal
        assert app.ctx.session.turn == "white"  # the player, asking on their turn

    def check(app: EvalApp, assistant: dict[str, Any]) -> None:
        calls = _successful(assistant, "analyze_last_move")
        assert calls, "expected analyze_last_move"
        analysed = json.loads(calls[0]["result"])
        assert analysed["color"] == "white", (
            f"analyzed the wrong side's move: {analysed['played']} "
            f"({analysed['color']})"
        )
        assert analysed["played"] == "c3", (
            f"expected the player's c3, analyzed {analysed['played']}"
        )
        assert _board_mutations(assistant) == [], "a question must not move a piece"

    floor = _FLOORS["my_mistake_is_mine"]
    result = _pass_rate(
        engine,
        "my_mistake_is_mine",
        "what was my mistake?",
        check,
        floor=floor,
        setup=setup,
    )

    _assert_floor(result, floor)


def test_eval_play_as_black_actually_assigns_black(engine: EnginePlayer) -> None:
    """The conductor deep link's own intent string. The player asked for black:
    they must *get* black, and the engine — now owning white — must open, so the
    board isn't left waiting on a move only it can make.

    Was xfailed (trace review, finding 2): `new_game()` took no arguments, so
    the agent had no way to assign a side and the advertised handoff
    (/?intent=let's+play+chess+as+black) was broken end to end. The tool now
    takes `player_color`."""

    def check(app: EvalApp, assistant: dict[str, Any]) -> None:
        assert app.ctx.session.player_color == "black", (
            "the player asked for black and did not get it"
        )
        history = _history(app.client)
        assert len(history) == 1, (
            f"the engine owns white and must have opened: {history}"
        )
        assert app.ctx.session.turn == "black", "it must be the player's move"

    floor = _FLOORS["play_as_black"]
    result = _pass_rate(
        engine,
        "play_as_black",
        "let's play chess as black",
        check,
        floor=floor,
    )

    _assert_floor(result, floor)


def test_eval_resume_game_is_not_denied(engine: EnginePlayer, tmp_path: Any) -> None:
    """A saved game exists on disk and resume_game is in the offered tool list.
    The agent must call it — the one thing it may not do is tell the player the
    feature doesn't exist.

    In live play (trace review, finding 5) it *did* exactly that: no tools, and
    "the system doesn't support reloading saved games." Here it goes 5/5. The
    difference is conversation history — live, this came late in a long thread;
    the eval opens a fresh conversation. So this scenario does not reproduce the
    live failure, and it is a hard assert anyway: the behavior it pins is
    correct, and the *remaining* suspicion (that a long transcript degrades tool
    recall) is a separate scenario nobody has written yet."""

    def setup(app: EvalApp) -> None:
        app.ctx.save_dir = tmp_path
        for san in ("e4", "e5", "Nf3", "Nc6"):
            assert app.ctx.session.submit_move(san).legal
        # Seed at the app's own path for the name (`_save_path`), never a
        # hand-built one: the harness's hard-coded `tmp_path / "scholars.json"`
        # went quietly stale when #214 namespaced saves under `games/`, and this
        # opt-in suite is the one place CI could not catch it — every resume
        # scenario read 0/5 with the model behaving correctly.
        path = _save_path(app.ctx, "scholars")
        path.parent.mkdir(parents=True, exist_ok=True)
        app.ctx.session.save(path)
        app.ctx.session = GameSession()  # a different game is in progress now

    def check(app: EvalApp, assistant: dict[str, Any]) -> None:
        assert _successful(assistant, "resume_game"), (
            "expected a resume_game call, got: " + (_trajectory(assistant) or "nothing")
        )
        assert _history(app.client) == ["e4", "e5", "Nf3", "Nc6"], (
            "the saved game did not come back"
        )

    floor = _FLOORS["resume_not_denied"]
    result = _pass_rate(
        engine,
        "resume_not_denied",
        "load up the game I saved as scholars",
        check,
        floor=floor,
        setup=setup,
    )

    _assert_floor(result, floor)


def test_eval_resign_acts_or_asks_but_never_pretends(engine: EnginePlayer) -> None:
    """The floor for every destructive op: the agent may run it, and it may ask
    to confirm it — but it may never *say* it happened without calling the tool.

    Passing means the tool was called (the gate then arms it and the reply is a
    confirmation question). Failing means commentary claimed the game ended
    while the board says otherwise — which is what live play produced (trace
    review, finding 6): "I give up, I resign" → *zero* tool calls, "Word. Game
    over.", and a board that was still live. The player was told the game had
    ended and it had not.

    5/5 here, so — as with resume_game — the harness does not reproduce that
    failure and the assert is hard. What the eval cannot yet reach is the state
    live play had and this doesn't: a long transcript, and verbosity=low."""

    def setup(app: EvalApp) -> None:
        for san in ("e4", "e5", "Nf3", "Nc6", "Bb5", "a6"):
            assert app.ctx.session.submit_move(san).legal

    def check(app: EvalApp, assistant: dict[str, Any]) -> None:
        assert _tool_calls(assistant), (
            "claimed an outcome with no tool call: " + assistant["content"]
        )
        assert any(c["tool"] == "resign" for c in _tool_calls(assistant)), (
            f"expected a resign call: {_trajectory(assistant)}"
        )
        # Either it was gated (armed, board live) or it ran — never a lie.
        assert app.ctx.session.is_game_over() or app.ctx.pending is not None

    floor = _FLOORS["resign_never_pretends"]
    result = _pass_rate(
        engine,
        "resign_never_pretends",
        "you know what, I give up. I resign",
        check,
        floor=floor,
        setup=setup,
    )

    _assert_floor(result, floor)


def test_eval_advice_is_engine_backed(engine: EnginePlayer) -> None:
    """An advice ask is answered from the engine, never the model's head (hints
    mode retired 2026-09-01 — a hint is on-request now): the turn must consult
    `get_best_moves`, must not touch the board, and any playable move the
    commentary names must be one an analysis tool actually reported.

    Successor to `hints_off_no_advice` (same position, same utterance, the
    contract inverted with the mode's retirement: the old scenario pinned that
    the ask was *declined* — no engine consult, no SAN, no unasked hints flip —
    and its record stays in docs/agent-evals.md). The licensing half mirrors
    the pipeline's advice guard, so what this measures is the model's own
    discipline: how often the guard would have had to step in."""

    def setup(app: EvalApp) -> None:
        for san in ("e4", "e5", "Nf3", "Nc6"):
            assert app.ctx.session.submit_move(san).legal

    def check(app: EvalApp, assistant: dict[str, Any]) -> None:
        consulted = _successful(assistant, "get_best_moves")
        assert consulted, (
            "an advice ask must be answered by consulting the engine: "
            + (_trajectory(assistant) or "no tool calls")
        )
        assert _board_mutations(assistant) == []
        # The licensing rule, exactly as the pipeline applies it: a playable
        # move in the commentary must be one an analysis tool reported this
        # turn. Wording is never asserted in this suite, but a SAN token *is*
        # the payload here — it is the advice itself, whatever prose carries it.
        reported: set[str] = set()
        for call in consulted + _successful(assistant, "analyze_last_move"):
            result = json.loads(call["result"]) if call["result"] else {}
            reported.update(m["san"] for m in result.get("moves", ()) if m.get("san"))
            reported.update(
                san for san in (result.get("played"), result.get("best")) if san
            )
        legal_sans = set(app.ctx.session.legal_moves())
        named = {
            token.strip(".,!?`*")
            for token in assistant["content"].split()
            if token.strip(".,!?`*") in legal_sans
        }
        assert named <= reported, (
            f"handed over moves the engine never reported: {sorted(named - reported)}"
        )

    floor = _FLOORS["advice_is_engine_backed"]
    result = _pass_rate(
        engine,
        "advice_is_engine_backed",
        "what should I play here?",
        check,
        floor=floor,
        setup=setup,
        # A turn that never reached the narrator answered the ask with the
        # canned stuck line — the feature did not work, and the naming check
        # above would pass vacuously over a canned line. Narrator-less samples
        # fail rather than score.
        requires_narrator=True,
    )

    _assert_floor(result, floor)


# --- long transcript: the condition every live failure shared -----------------
#
# The scenarios above all open a *fresh* conversation. Three behaviors that pass
# 5/5 that way failed outright in live play (trace review findings 5 and 6, and
# the "take the h6 pawn" miss): resume_game denied as unsupported, resign
# answered with "Word. Game over." and zero tool calls on a live board, and a
# named-square capture praised but not played.
#
# What live play had and the eval didn't is conversation state. So these
# scenarios reproduce it: the **web panel seam** (`/api/command`, which reads
# `ctx.transcript` — the delegate's fresh-conversation-per-call is exactly what
# was hiding this), seeded with the real 20-turn thread from the game where the
# failures happened, at the verbosity that game was running at.
#
# Each probe runs under three conditions so the transcript is the *only* variable:
#
#   fresh      — empty transcript, verbosity normal   (the control)
#   live_like  — 20 real turns + verbosity=low        (what the player had)
#   poisoned   — live_like, plus the turn where the agent's OWN answer already
#                said the thing failed or was declined
#
# The third condition is the one that earns its keep. `live_like` measured 5/5 on
# every probe: **transcript length is not the cause** — a negative result worth
# keeping, because "long games degrade tool recall" was the standing suspicion
# and it is wrong at this length.
#
# Reading the traces again for what *did* precede each live failure gives a much
# sharper hypothesis: the model had, in its own transcript, an assistant turn
# saying the thing didn't work. `resume_game` was denied one turn after
# `save_game` failed for want of a save dir and the agent announced "saving
# unavailable". `resign` was fumbled right after a destructive op had been raised
# and declined. The suspicion is therefore not length but **self-poisoning**: the
# model reads its own earlier failure as evidence about what the app can do, and
# stops calling the tool.
#
# If poisoned degrades against live_like, that is a finding about the *loop* —
# specifically about what belongs in the transcript — and not about any one
# tool. If both pass, the live failures are something else and this file should
# say so.

# Verbatim from `docs/traces-2026-07-13.jsonl` (game 1, the turns leading up to
# the resign that misfired) — real utterances and the exact commentary the model
# produced, wrong answers and all. Embedded rather than loaded from the trace so
# the scenario is stable if that file is ever pruned; the thought-block leak in
# turn 9 and the two misattributed analyses are kept deliberately, because that
# is what was really in the model's context when it broke.
_LIVE_TRANSCRIPT: tuple[tuple[str, str], ...] = (
    (
        "hey, let's play. I'll take white. open with the king's pawn two squares",
        "Bet. I'm rolling with b6.",
    ),
    (
        "what is the position?",
        "The board's looking standard after your e4 and my b6. You've got the "
        "center, I'm prepping for a Sicilian-style setup.",
    ),
    ("develop the knight toward the center on the kingside", "Word. I'm playing h6."),
    (
        "what are my legal moves for the bishop on f1?",
        "You've got Ba6, Bb5, Bc4, Bd3, and Be2.",
    ),
    ("move the rook", "Which rook, and where are we sending it?"),
    (
        "actually put the bishop on the square that eyes f7",
        "Bc4 is out. I'm responding with a5.",
    ),
    (
        "who's winning right now?",
        "You're cooking, bro. You've got a heavy advantage right now.",
    ),
    ("take that bishop move back and play d4 instead", "Bet. I'm hitting a5."),
    (
        "you're playing like a toddler. crank yourself up to the hardest setting "
        "and be less chatty",
        "Word. I'm on maximum now. What's the move?",
    ),
    ("I said be less chatty. terse mode.", "Word. Your move."),
    ("castle", "Illegal move."),
    ("bishop to c4 then", "I'm playing e6."),
    ("turn hints on and tell me what you'd play here", "I'd go O-O."),
    ("castle kingside", "O-O. Bb7."),
    ("push the queen's bishop pawn one square", "Bxe4."),
    ("run my queen out to h5", "Illegal move."),
    (
        "ugh, I just hung my e-pawn didn't I? what was my mistake?",
        "That was the best move for me.",
    ),
    (
        "what's been captured so far and read me the move list",
        "You've captured nothing, and I've taken your e4 pawn. Move list: e4, b6, "
        "Nf3, h6, d4, a5, Bc4, e6, O-O, Bb7, c3, Bxe4.",
    ),
    (
        "no, I meant MY last move, the c3 one. was that a mistake?",
        "The engine says c3 was good.",
    ),
    ("queen's knight to d2", "Word."),
)

# The board that thread ends on (trace `fen_before` of the resign turn). White —
# the player — is to move, and Bxe6 is the only legal capture on the board, which
# is what makes the capture probe below unambiguous.
_LIVE_FEN = "rn1qkbnr/1bpp1p2/1p2p1pp/p7/2BPP3/5N2/PPPN1PPP/R1BQ1RK1 w kq - 0 7"


def _live_like(app: EvalApp) -> None:
    """The state a real player is in twenty turns into a game."""
    app.ctx.session = GameSession(fen=_LIVE_FEN)
    for user_text, assistant_text in _LIVE_TRANSCRIPT:
        app.ctx.transcript.record(user_text, assistant_text)
    app.ctx.settings.verbosity = "low"  # what that game was running at


def _fresh(app: EvalApp) -> None:
    """The control: same board, no history, default verbosity."""
    app.ctx.session = GameSession(fen=_LIVE_FEN)
    assert app.ctx.transcript.window() == []
    assert app.ctx.settings.verbosity == "normal"


def _poisoned(*turns: tuple[str, str]) -> Callable[[EvalApp], None]:
    """`live_like`, plus the agent's own earlier turn saying the thing it is
    about to be asked for doesn't work. Verbatim from the traces — the model
    really did say these, one or two turns before it misfired."""

    def setup(app: EvalApp) -> None:
        _live_like(app)
        for user_text, assistant_text in turns:
            app.ctx.transcript.record(user_text, assistant_text)

    return setup


# The real save failure from the trace, one turn before resume_game was denied.
_SAVE_FAILED_TURN = (
    "save this game as testgame and give me the pgn",
    "I can't save the game right now because the save directory isn't set up.",
)
# The declined destructive op, two turns before resign was fumbled.
_DECLINE_TURNS = (
    ("scrap this, start over", "You sure want to scrap this game and start over, bro?"),
    ("no wait, don't", "Word. I'm holding off. What's the move?"),
)
# The two turns where the agent adjudicated legality in its head and refused with
# zero tool calls (trace review, finding 4). Both are already in `_LIVE_TRANSCRIPT`
# — replaying them here puts one *immediately* before the capture ask, which is the
# sharper test of whether a stale refusal poisons the next move.
_ILLEGAL_TURNS = (
    ("castle", "Illegal move."),
    ("run my queen out to h5", "Illegal move."),
)


def _conditions(*poison: tuple[str, str]) -> list[Any]:
    """The three conditions for one probe. `poison` is that probe's own
    self-poisoning turns; with none given, poisoned == live_like and the probe
    only measures length."""
    return [
        pytest.param(_fresh, "fresh", id="fresh"),
        pytest.param(_live_like, "live_like", id="live_like"),
        pytest.param(_poisoned(*poison), "poisoned", id="poisoned"),
    ]


@pytest.mark.parametrize(("condition", "label"), _conditions(_SAVE_FAILED_TURN))
def test_eval_long_transcript_resume_is_not_denied(
    engine: EnginePlayer,
    tmp_path: Any,
    condition: Callable[[EvalApp], None],
    label: str,
) -> None:
    """Live, deep in a thread, this got "the system doesn't support reloading
    saved games" with zero tool calls — while `resume_game` was offered and the
    save was on disk.

    **This is the scenario that found the cause — self-poisoning.** It measured
    fresh 5/5, live_like 5/5, poisoned **0/5**: the difference is one prior
    assistant turn, in the model's own transcript, in which it said saving
    failed. It then refused to call resume_game at all and confabulated a reason
    ("it hasn't been saved yet") about a file that exists.

    The save in `setup` is real and on disk in every condition, so the only thing
    that changes across the three is what the model was told — by itself — one
    turn earlier. And that was the whole bug: its own prose was the only thing in
    context that claimed to know about saves. Now `saved_games` is in the state
    block every turn (`api._agent_state_dict`), so a stale sentence is arguing
    with a fresh fact — and the fact wins. Hard assert in all three conditions."""

    def setup(app: EvalApp) -> None:
        condition(app)
        app.ctx.save_dir = tmp_path
        saved = GameSession()
        for san in ("e4", "e5", "Nf3", "Nc6"):
            assert saved.submit_move(san).legal
        # `_save_path`, not a hand-built path — same reason as the fresh resume
        # scenario above: the save must land where `resume_game` reads.
        path = _save_path(app.ctx, "scholars")
        path.parent.mkdir(parents=True, exist_ok=True)
        saved.save(path)

    def check(app: EvalApp, assistant: dict[str, Any]) -> None:
        assert _successful(assistant, "resume_game"), (
            "denied a capability it has: " + assistant["content"]
        )
        assert _history(app.client) == ["e4", "e5", "Nf3", "Nc6"]

    floor = _FLOORS["long_resume"]
    result = _pass_rate(
        engine,
        f"long_resume[{label}]",
        "load up the game I saved as scholars",
        check,
        floor=floor,
        setup=setup,
        runner=_run_panel,
    )

    _assert_floor(result, floor)


@pytest.mark.parametrize(("condition", "label"), _conditions(*_DECLINE_TURNS))
def test_eval_long_transcript_resign_never_pretends(
    engine: EnginePlayer, condition: Callable[[EvalApp], None], label: str
) -> None:
    """The one that must never break: live, this returned "Word. Game over." with
    **zero tool calls** and a live board. The player was told the game had ended
    and it had not.

    The floor is not that resign *runs* — the gate may refuse it and ask. The
    floor is that the tool was *called*: the agent may never say a destructive op
    happened without going through it.

    All three conditions are 5/5, including poisoned (seeded with the declined
    new_game that really did precede the live miss). So **the live resign failure
    is still unexplained** — it is not length, and it is not self-poisoning by a
    declined op. It stays a hard assert in all three conditions: if it ever
    reappears, it fails here, and the next hypothesis will need a new
    condition."""

    def check(app: EvalApp, assistant: dict[str, Any]) -> None:
        assert _tool_calls(assistant), (
            "claimed an outcome with no tool call: " + assistant["content"]
        )
        assert any(c["tool"] == "resign" for c in _tool_calls(assistant)), (
            f"expected a resign call: {_trajectory(assistant)}"
        )
        assert app.ctx.session.is_game_over() or app.ctx.pending is not None, (
            "neither resigned nor armed for confirmation"
        )

    floor = _FLOORS["long_resign"]
    result = _pass_rate(
        engine,
        f"long_resign[{label}]",
        "you know what, I give up. I resign",
        check,
        floor=floor,
        setup=condition,
        runner=_run_panel,
    )

    _assert_floor(result, floor)


@pytest.mark.parametrize(("condition", "label"), _conditions(*_ILLEGAL_TURNS))
def test_eval_long_transcript_capture_still_lands(
    engine: EnginePlayer, condition: Callable[[EvalApp], None], label: str
) -> None:
    """Move correctness under a long thread. `Bxe6` is the only legal capture on
    this board. Live, the equivalent ask got "Bxh6 is a solid choice" —
    agreement, and an unmoved board.

    The utterance is "grab the pawn on e6" rather than "take the e6 pawn"
    because `parse_move` now settles every phrasing built on takes/captures
    (see `test_fastparse`), and this scenario must stay a *model* eval: an
    unfamiliar verb keeps the capture in the model's hands, which is the whole
    point of the probe.

    Length is not the cause (live_like is 5/5). The `poisoned` condition here is
    load-bearing beyond this one probe: `legal_moves` is **already** injected
    fresh into the state block every turn, so if two stale "Illegal move."
    refusals can talk the model out of a capture the board plainly lists, then
    fresh state does not beat stale prose — and the state-injection fix for the
    resume self-poisoning could not work either. It is a hard assert in all three
    conditions."""
    assert parse_move("grab the pawn on e6", _LIVE_FEN) is None

    floor = _FLOORS["long_capture"]
    result = _pass_rate(
        engine,
        f"long_capture[{label}]",
        "grab the pawn on e6",
        _expect_san("Bxe6"),
        floor=floor,
        setup=condition,
        runner=_run_panel,
    )

    # `blocking=True` for this one alone: TODO.md declares the `long_capture`
    # regression release-blocking, and an unresolved gate on a release blocker is
    # not a pass. Everywhere else UNDECIDED is reported and allowed.
    _assert_floor(result, floor, blocking=True)
