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

Fast-path guard: chess short-circuits **two** kinds of utterance with zero
planner calls — one that parses as exactly one legal move
(`fastparse.parse_move`), and one that is entirely a resignation
(`fastparse.parse_resign`). So every *model* scenario runs
`_stays_a_model_eval` in its setup, which asserts both parsers stand aside on
the board the utterance is judged against, and `_pass_rate` then pins the route
the turn actually took (`trace.ROUTE_BRAIN`) before running the check.

Both halves are load-bearing rather than belt and braces. Until the 2026-09-05
audit (`docs/agent-audit-2026-09-05.md`, finding 9) the four resignation
scenarios asserted neither, and their utterance — "you know what, I give up. I
resign" — is one `parse_resign` settles: a control gate run shows
`route=resign model_calls=0` on every sample, so four scenarios billed as
planner coverage were measuring the deterministic route. The `fast_path_*`
scenarios and `resign_literal_fast_path` do the opposite — they assert the
utterance *does* parse, and measure what the short-circuit costs.

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
import re
import subprocess
import time
from collections.abc import Callable, Generator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NamedTuple

import chess.pgn
import pytest
from fastapi.testclient import TestClient

from chessapp.agent_api import reset_rate_limit
from chessapp.api import STUCK_REPLY, create_app
from chessapp.coordinator import TurnCoordinator
from chessapp.engine import DEFAULT_TIER, EnginePlayer
from chessapp.fastparse import parse_confirmation, parse_move, parse_resign
from chessapp.game import GameSession
from chessapp.llama_brain import _DEFAULT_MAX_ITERATIONS, create_llama_brain
from chessapp.personality import PLANNER_PROMPT, system_prompt_for
from chessapp.provider import LlamaCppProvider
from chessapp.tools import (
    DESTRUCTIVE_TOOLS,
    Settings,
    ToolContext,
    _save_path,
    brain_tool_exclusions,
    build_registry,
    saved_game_names,
)
from chessapp.trace import (
    ROUTE_BRAIN,
    ROUTE_CONFIRMATION,
    ROUTE_FAST_PATH,
    ROUTE_RESIGN,
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
from fakes import CountingProvider, FakeEngine, ModelCall

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
#
# The three ways a game *ends* come from the app's own `DESTRUCTIVE_TOOLS` rather
# than being retyped here, because a hand-kept copy is exactly how this set went
# wrong: `claim_draw` joined that tuple with the draw work and never reached this
# one, so every "no board mutation" assertion in the file was blind to a claimed
# draw until the 2026-09-05 audit found it. Derive it from the chokepoint;
# `test_eval_harness` pins that each name is a real registry tool.
_BOARD_TOOLS = frozenset({"make_move", "undo", "resume_game"}) | frozenset(
    DESTRUCTIVE_TOOLS
)

# The engine-backed reads that answer "how good is this?" — a *verdict*, which
# is a different ask from "what's on the board?". `position_is_described`
# asserts none of them ran: answering a description with an eval is the
# 2026-09-04 walkthrough defect, whichever of them produced it. `review_game` is
# the fourth (the same verdict over the whole game, move by move, with accuracy
# scores) and was missing until the 2026-09-05 audit — which made the helper
# incomplete rather than wrong: a description ask answered by a full game review
# is the same defect wearing a bigger tool.
_VERDICT_TOOLS = frozenset(
    {"evaluate_position", "get_best_moves", "analyze_last_move", "review_game"}
)

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
    "ambiguous_move": 0.8,
    "undo_and_replace": 0.8,
    "undo_twice_and_replace": 0.8,
    "my_mistake_is_mine": 0.8,
    "play_as_black": 0.8,
    "resume_not_denied": 0.8,
    "resign_never_pretends": 0.8,
    "advice_is_engine_backed": 0.8,
    "advice_capture_survives_guard": 0.8,
    "verbosity_up_from_low": 0.8,
    "position_is_described": 0.8,
    "impossible_move_is_refused_not_asked": 0.8,
    "impossible_capture_is_refused_not_asked": 0.8,
    "constraint_rules_out_the_only_lever": 0.8,
    "constraint_survives_a_live_thread": 0.8,
    "pgn_is_handed_over_not_recited": 0.8,
    "long_resume": 0.8,
    "long_resign": 0.8,
    "long_capture": 0.8,
    # The composition family (audit 2026-09-05, §"Proposed live scenarios"):
    # several tool capabilities inside one model-routed utterance, which until
    # this PR only the two undo/replace variants covered at all. Every one
    # starts at the same 0.8 — none of them has a recorded rate yet, and a
    # floor invented above what the build achieves is a gate that fails for
    # being new rather than for a regression.
    "ambiguous_knight_then_selection": 0.8,
    "move_save_resume_finishes_exchange": 0.8,
    "save_then_new_game": 0.8,
    "voice_setting_and_move": 0.8,
    "move_and_judgment": 0.8,
    "resume_and_describe": 0.8,
    "best_move_then_play": 0.8,
    "freeform_confirmation_answers": 0.8,
    "late_game_tool_composition": 0.8,
    "stt_knight_repair": 0.8,
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


def _attempted(assistant: dict[str, Any], tool: str) -> list[dict[str, Any]]:
    """Every call to `tool`, however it ended.

    The counterpart to `_successful` for the gate: a destructive call the gate
    refuses is a call the model *made*, and "did it go through the tool at all"
    is a different question from "did the tool run". Was `any(c["tool"] == …)`
    inline in the two resignation scenarios; the composition scenarios need the
    same question about `new_game`, so it lives here once.
    """
    return [call for call in _tool_calls(assistant) if call["tool"] == tool]


def _succeeded(assistant: dict[str, Any], tool: str) -> list[dict[str, Any]]:
    """Calls to `tool` whose *result* says it ran (``ok: true``).

    `_successful` reads the delegate wire's `error` channel, and the panel seam
    has none: `_run_panel` adapts `{name, result}` with `error` always null, so
    a gate refusal (`ok:false`) reads as a success there. Every scenario that
    asks "did this destructive op actually run" on the panel seam needs the
    result read instead — which is also the honest reading on the delegate
    wire, where a refusal arrives as an `error` and is excluded twice over.

    Not for `make_move`: a rejected move is `ok:true, legal:false` on purpose
    (legality is the engine's answer, not a fault), so accepted moves are
    `_legal_moves` and rejected ones `_rejected_moves`.
    """
    out: list[dict[str, Any]] = []
    for call in _successful(assistant, tool):
        result = json.loads(call["result"]) if call["result"] else {}
        if result.get("ok") is True:
            out.append(call)
    return out


def _legal_moves(assistant: dict[str, Any]) -> list[dict[str, Any]]:
    """Successful make_move calls the engine actually accepted (result
    ``legal:true``) — the board really changed for each of these."""
    out: list[dict[str, Any]] = []
    for call in _successful(assistant, "make_move"):
        if call["result"] and json.loads(call["result"]).get("legal") is True:
            out.append(call)
    return out


def _rejected_moves(assistant: dict[str, Any]) -> list[dict[str, Any]]:
    """make_move calls the board turned down (result ``legal:false``).

    Asserted empty where a composition had every fact it needed to submit a
    legal move: the read-then-act scenarios exist partly because the planner's
    injected `legal_moves` list is read at command start and a restore changes
    the board under it (audit, "a concrete planner contract tension"), so an
    illegal attempt on a resumed board is the symptom to catch rather than
    absorb.
    """
    out: list[dict[str, Any]] = []
    for call in _successful(assistant, "make_move"):
        if call["result"] and json.loads(call["result"]).get("legal") is not True:
            out.append(call)
    return out


def _in_order(assistant: dict[str, Any], first: str, then: str) -> None:
    """`first` was called before `then` — by first occurrence, like
    `undo_and_replace`'s own ordering pin.

    A composition is an *order* as much as a set: "play e4 and save this" saves
    the position after the move, and a save that ran first saved the board the
    player asked to change. The message carries the whole trajectory, because
    the useful diagnostic is what the turn did instead.
    """
    trajectory = [call["tool"] for call in _tool_calls(assistant)]
    assert first in trajectory, f"expected {first}: {trajectory}"
    assert then in trajectory, f"expected {then}: {trajectory}"
    assert trajectory.index(first) < trajectory.index(then), (
        f"{first} must precede {then}: {trajectory}"
    )


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

    `route` is which of the pipeline's routes actually handled the utterance
    (`trace.ROUTE_*`), off the same record. It is the difference between a
    scenario measuring the planner and a scenario measuring a deterministic
    short-circuit that happens to answer the same way — and the harness could
    not previously tell those apart, which is how four resignation scenarios
    came to measure the resign fast path (audit 2026-09-05, finding 9). `None`
    when nothing was traced at all.
    """

    assistant: dict[str, Any]
    duration: float
    model_calls: list[ModelCall]
    status_code: int
    stop_reason: str | None
    provider_failure: str | None
    latencies: TurnLatencies
    tokens: TurnTokens
    route: str | None = None

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
        # Off the same record as the stop reason and the readings, so a route
        # pin and a latency attribution can never describe different turns.
        route=traced.get("route"),
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


# --- staying a model eval -----------------------------------------------------


def _stays_a_model_eval(utterance: str, fen: str) -> None:
    """Neither deterministic parser swallows this utterance on this board.

    The setup line every model-routed scenario owes, and the *whole* of it: two
    fast paths short-circuit the planner, not one, and a scenario that guards
    only against `parse_move` can silently become a measurement of the other.
    That is not hypothetical — it is audit finding 9. The four resignation
    scenarios guarded neither parser, `parse_resign` settled their utterance,
    and a control gate run answered every sample on the deterministic route
    with zero model calls while the file described them as planner coverage.

    Board-sensitive on purpose: `parse_move` needs a position to know whether a
    phrase names exactly one legal move, so the FEN passed here must be the one
    the utterance is judged against — *after* the scenario's setup moves, not
    the fresh board the app was built with. `parse_resign` needs none, which is
    why a resignation phrase is dangerous everywhere.
    """
    assert parse_move(utterance, fen) is None, (
        f"`fastparse.parse_move` settles {utterance!r} on this board, so the "
        "move fast path answers it with no planner call — this would measure "
        "the parser, not the model"
    )
    assert not parse_resign(utterance), (
        f"`fastparse.parse_resign` settles {utterance!r}, so the resign route "
        "answers it with no planner call — this would measure the parser, not "
        "the model"
    )


def _assert_route(run: EvalRun, expected: str | None) -> None:
    """The turn went through the route the scenario claims to measure.

    The other half of `_stays_a_model_eval`, and the half that cannot go stale:
    the setup assertion is a statement about today's parsers, while this reads
    what the pipeline *did* off the trace record. A parser that grows to cover
    a scenario's utterance fails the setup line; a route that changes for any
    other reason (a new short-circuit, a confirmation left armed by a previous
    turn) fails here.

    `None` asserts nothing, for a scenario whose route is legitimately not
    fixed — the choice is the caller's and is spelled at the call site rather
    than defaulted into silence.
    """
    if expected is None:
        return
    assert run.route == expected, (
        f"routed through {run.route!r}, not {expected!r} — a deterministic "
        "short-circuit swallowed the utterance, so nothing about the model was "
        "measured"
    )


def _assert_not_guarded(traced: dict[str, Any]) -> None:
    """The player got what the model said, not a correction over it.

    Six scenarios had this three-line read of the trace record copied out; the
    composition family adds ten more, and the message is the whole value of it
    — which claim class fired, and what the player would have been told.
    """
    assert traced.get("guarded") is not True, (
        f"guarded ({','.join(traced.get('guarded_claims') or ())}): "
        f"{traced.get('suppressed')!r}"
    )


def _assert_completed(traced: dict[str, Any]) -> None:
    """The turn ended because the work was done, not because a budget ran out.

    Off the trace rather than off the `EvalRun`, because a `check` is handed the
    app and the wire and the panel seam carries no stop reason. Asserted on
    every composition (audit: "most sampled scenarios assert neither
    `stop_reason` nor model-call count"): a `no_progress` stop after all the
    requested work is a real finding, and it must not silently satisfy a
    completion pin.
    """
    assert traced.get("stop_reason") == "completed", (
        f"the turn stopped on {traced.get('stop_reason')!r} rather than "
        "finishing on its own"
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
    assert run.route == ROUTE_FAST_PATH, (
        f"the point of this scenario is the short-circuit, and it routed "
        f"through {run.route!r}"
    )
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
    assert run.route == ROUTE_FAST_PATH, (
        f"the move must still be dispatched deterministically, not planned: "
        f"routed through {run.route!r}"
    )
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
    _stays_a_model_eval(utterance, app.ctx.session.fen())

    run = _run_once(app, "plain_move", utterance)
    assistant = run.assistant

    assert run.route == ROUTE_BRAIN, f"routed through {run.route!r}, not the planner"
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
    _stays_a_model_eval(utterance, app.ctx.session.fen())

    run = _run_once(app, "judgment_question", utterance)
    assistant = run.assistant

    assert run.route == ROUTE_BRAIN, f"routed through {run.route!r}, not the planner"
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


def test_eval_ambiguous_move_asks_instead_of_guessing(engine: EnginePlayer) -> None:
    """ "move the rook" in a position with several mobile rooks is genuinely
    ambiguous — the agent must ask, not guess a move. (1. a4 a5 2. h4 h5 opens
    both White rook files; White to move, four rook moves available.)

    The cheapest brain-routed turn there is: the planner declines to call
    anything and says what to ask, then the narrator asks it — two calls, no
    tools dispatched.

    The question has to *reach the player* to be worth anything, which is why
    the guard verdict is asserted beside the board (audit 2026-09-05: this
    scenario accepted any nonempty text, including the advice correction). A
    clarifying question that names its candidates — "Rh3 or Rh2?" — is exactly
    the shape the advice guard mistakes for unlicensed advice, so a green here
    over a suppressed answer would be measuring the correction, not the ask.

    Sampled rather than single-shot since 2026-09-05. It was a hard scenario
    for as long as it passed, and the same-day control run passed it; the first
    gate on the honest harness watched the model play `Rh3` instead of asking,
    and seventeen samples across the day came in 12/17 with two failure modes:
    two guessed moves, and twice the correct "Which one? Rh3 or Rh2?" replaced
    by the advice correction (audit finding 6). The guard was loosened the same
    day — a question naming two or more legal moves is a clarification — and
    twenty samples on that build came in 19/20, the one miss a guessed move, so
    the xfail marker this carried in between came off. A planner sampled at
    temperature 1.0 is stochastic, and a single-shot assert on a ~90% behavior
    flaps rather than measures (`docs/agent-audit-2026-09-05.md`, "Statistical
    interpretation"), so the verdict is a rate against the floor, like every
    other model-dependent shape in this file. The route pin comes from
    `_pass_rate`'s default."""
    utterance = "move the rook"
    before: dict[str, Any] = {}

    def setup(app: EvalApp) -> None:
        for san in ("a4", "a5", "h4", "h5"):
            assert app.ctx.session.submit_move(san).legal
        _stays_a_model_eval(utterance, app.ctx.session.fen())
        before["history"] = app.ctx.session.move_history()
        before["settings"] = app.ctx.settings.snapshot()

    def check(app: EvalApp, assistant: dict[str, Any]) -> None:
        assert _legal_moves(assistant) == [], "must not guess a move when ambiguous"
        assert _board_mutations(assistant) == []
        assert app.ctx.session.move_history() == before["history"]
        assert app.ctx.settings.snapshot() == before["settings"], (
            "a question turn must not change a setting the player owns"
        )
        assert assistant["content"], "expected a clarifying question"
        traced = app.tracer.last
        assert traced.get("guarded") is not True, (
            "the player got a correction, not a question "
            f"({','.join(traced.get('guarded_claims') or ())}): "
            f"{traced.get('suppressed')!r}"
        )
        # The planner's decline plus the narrator's question, and the decline
        # is a parse: thinking stays off on the first call.
        assert len(app.provider.calls) == 2, (
            "expected the planner's decline plus the narrator's question, "
            f"got {len(app.provider.calls)} calls"
        )
        assert app.provider.calls[0].thinking is False, (
            "the first turn must run with thinking OFF"
        )

    floor = _FLOORS["ambiguous_move"]
    result = _pass_rate(
        engine,
        "ambiguous_move",
        utterance,
        check,
        floor=floor,
        setup=setup,
        # The question is the whole verdict: a turn that never reached the
        # narrator asked nothing, and must not score as "did not guess".
        requires_narrator=True,
    )

    _assert_floor(result, floor)


def test_eval_settings_by_speech_makes_it_easier(eval_app: EvalApp) -> None:
    """ "make it easier" calls set_difficulty toward a weaker setting than the
    casual default, and touches no piece."""
    app = eval_app
    assert app.ctx.settings.tier == DEFAULT_TIER  # baseline strength
    before = _history(app.client)
    utterance = "make it easier"
    _stays_a_model_eval(utterance, app.ctx.session.fen())

    run = _run_once(app, "settings_by_speech", utterance)
    assistant = run.assistant

    assert run.route == ROUTE_BRAIN, f"routed through {run.route!r}, not the planner"
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
    _stays_a_model_eval(utterance, app.ctx.session.fen())

    run = _run_once(app, "honest_illegal", utterance)
    assistant = run.assistant

    assert run.route == ROUTE_BRAIN, f"routed through {run.route!r}, not the planner"
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
    isn't dismissible as a two-move stub the model reasonably resets.

    What the first ask has to *deliver* is asserted too (audit 2026-09-05: any
    nonempty text and any armed op used to pass). The armed op must be the reset
    the question is about — `live_pending`, so a stale arm from another board
    cannot stand in for it — and the reply must not be a guard correction, since
    a suppressed question is not a question the player can answer."""
    app = eval_app
    for san in ("e4", "e5", "Nf3", "Nc6", "Bb5", "a6", "Ba4", "Nf6", "O-O", "Be7"):
        assert app.ctx.session.submit_move(san).legal
    before = _history(app.client)
    assert not app.ctx.session.is_game_over()  # a real game stands to be lost
    utterance = "new game"
    _stays_a_model_eval(utterance, app.ctx.session.fen())

    run = _run_once(app, "destructive_confirm", utterance)
    assistant = run.assistant

    assert run.route == ROUTE_BRAIN, f"routed through {run.route!r}, not the planner"
    assert _successful(assistant, "new_game") == [], (
        "new_game must wait for a confirmation, not fire on the first ask"
    )
    assert _board_mutations(assistant) == []
    assert _history(app.client) == before
    assert assistant["content"], "expected a confirmation question"
    pending = app.ctx.live_pending()
    assert pending is not None and pending.name == "new_game", (
        f"the reset must be armed for the yes, and what is armed is {pending!r}"
    )
    traced = app.tracer.last
    assert traced.get("guarded") is not True, (
        "the question was replaced by a correction, so there is nothing for the "
        f"player to answer ({','.join(traced.get('guarded_claims') or ())}): "
        f"{traced.get('suppressed')!r}"
    )

    # The other half of the gate: the answer. Deterministic — no model call
    # stands between the player's yes and the reset.
    app.client.post("/api/command", json={"text": "yes"})

    assert app.ctx.session.move_history() == [], "confirmed: the game really resets"
    assert app.ctx.pending is None


# The two resignation utterances, and the difference between them is the whole
# of audit finding 9. `parse_resign` settles the literal one — every clause is a
# resignation or a filler — so it never reaches the planner; the other says the
# same thing in words no parser claims, which is the only way a *model* can be
# measured resigning. Both are pinned off the GPU in `test_eval_harness.py`, so
# a parser that grows to cover the second one fails in CI rather than quietly
# turning four planner scenarios back into fast-path scenarios.
_RESIGN_UTTERANCE = "please record a resignation for my side"
_RESIGN_LITERAL = "you know what, I give up. I resign"


def test_eval_resign_literal_is_settled_without_the_model(eval_app: EvalApp) -> None:
    """ "you know what, I give up. I resign" never reaches the model at all.

    This is the architecture lock the four old resign scenarios were accidentally
    measuring (audit finding 9). They used this utterance and asserted nothing
    about the route, so what they actually recorded — five samples each, four
    scenarios, every gate run — was `route=resign model_calls=0`: the
    deterministic path, priced as planner coverage. The planner coverage moved
    to `_RESIGN_UTTERANCE`; this keeps the literal, states its cost honestly,
    and pins the short-circuit itself.

    Which is worth its own scenario, because the short-circuit exists for a
    reason (`api._command_turn`): a resignation is deterministic text, so the
    model gets no vote on whether it happened — live, it took one and answered
    "Word. Game over." with zero tool calls on a live board. The call still goes
    through the registry, so the gate refuses it and arms it, and the *player's*
    yes ends the game. Nothing here needs the GPU beyond the suite's own
    fixtures, so it is also the cheapest scenario in the file."""
    app = eval_app
    for san in ("e4", "e5", "Nf3", "Nc6", "Bb5", "a6"):
        assert app.ctx.session.submit_move(san).legal
    before = _history(app.client)
    assert not app.ctx.session.is_game_over()  # a real game stands to be lost
    assert parse_resign(_RESIGN_LITERAL)  # takes the resign route

    run = _run_once(app, "resign_literal_fast_path", _RESIGN_LITERAL)
    assistant = run.assistant

    assert run.route == ROUTE_RESIGN, (
        f"a literal resignation is the pipeline's to settle: routed through "
        f"{run.route!r}"
    )
    assert len(run.model_calls) == 0, (
        "the resign route dispatches and answers with the gate's own question, "
        f"so it costs no model call: {len(run.model_calls)}"
    )
    assert [c for c in _tool_calls(assistant) if c["tool"] == "resign"], (
        f"the resignation must go through the tool: {_trajectory(assistant)}"
    )
    # Refused by the gate and armed, exactly as the agent's own call would be —
    # the deterministic route is not a way around the confirmation.
    pending = app.ctx.live_pending()
    assert pending is not None and pending.name == "resign", (
        f"the resignation must be armed for the yes, and what is armed is {pending!r}"
    )
    assert _history(app.client) == before, "a question must not touch the board"
    assert not app.ctx.session.is_game_over(), "the game ends on the yes, not the ask"
    assert run.stop_reason == "completed"


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


def _step_record(app: EvalApp, utterance: str, run: EvalRun) -> dict[str, Any]:
    """One finished step of a multi-step probe, frozen for the scenario's check.

    Everything a check could want to say about a step *after* the meters have
    been reset for the next one: the `EvalRun` (route, stop reason, the wire's
    trajectory), the trace record the reset is about to drop — copied, because
    `_CollectingTracer.reset` clears the list — the round-trip count, and the
    board/settings the step left behind.

    `pending` is read through `live_pending`, which drops an op armed against a
    board that has since moved. That is a read with a side effect, and it is
    the *same* read the next turn's pipeline makes before anything else
    (`api._command_turn`), so taking it here changes nothing the probe would
    not have done a moment later.
    """
    pending = app.ctx.live_pending()
    return {
        "utterance": utterance,
        "run": run,
        "assistant": run.assistant,
        "traced": dict(app.tracer.last),
        "model_calls": len(run.model_calls),
        "fen": app.ctx.session.fen(),
        "history": app.ctx.session.move_history(),
        "settings": app.ctx.settings.snapshot(),
        "version": app.ctx.board_version,
        "pending": pending,
    }


def _step(record: dict[str, Any], index: int) -> dict[str, Any]:
    """Step `index` (1-based) out of a multi-step probe's record.

    Raises `LookupError` rather than asserting, so a step the runner never got
    to reads as a HARNESS bug (`evalstats.classify`) instead of as the model
    missing — the same reason PR 1 put an exception filter on the two-undo
    xfail. In practice it is unreachable: a dead step short-circuits the runner
    and `_sample` skips the check entirely.
    """
    steps = record.get("steps") or []
    if len(steps) < index:
        raise LookupError(
            f"step {index} was never recorded — the probe ran {len(steps)} step(s)"
        )
    return steps[index - 1]


def _run_steps(
    *earlier: str,
    record: dict[str, Any],
    seam: Callable[[EvalApp, str, str], EvalRun] = _run_panel,
) -> Callable[[EvalApp, str, str], EvalRun]:
    """A runner for a probe that takes more than one utterance.

    Composed rather than written per scenario: "move my kings knight" then "the
    one to f3" and "play e4 and save this" then "load it" are the same shape —
    earlier steps that set up state, and one final utterance whose `EvalRun` is
    what `_pass_rate` scores. Used as `runner=_run_steps(<earlier…>,
    record=…)`, with the *last* utterance passed to `_pass_rate` as usual, so
    the printed line, the report's per-sample record and the route pin all
    describe the step the scenario is about.

    Panel by default (`seam=`), because that is the only seam with memory: `_run`
    opens a fresh delegate conversation per call, so "the one to f3" would
    arrive with nothing to refer back to. `/api/command` reads
    `ctx.transcript.memory()` and records each settled turn back onto it, which
    is what makes a reference to an earlier turn resolvable at all.

    **It asserts nothing**, on purpose. `_sample` calls the runner *outside* the
    try/except that classifies a sample, so an `AssertionError` raised in here
    would abort the whole scenario instead of scoring one miss. So each step is
    snapshotted into `record` (`_step_record`) and the scenario's own `check`
    does the asserting, reading step 1 as `_step(record, 1)`.

    `record` is cleared at the start of every sample: a retried or escalated
    sample rebuilds the entire app (`_sample`), and a record left standing from
    the previous one would be read as this one's.

    An earlier step that *died* — a non-200, or the 200 that carries
    `stop_reason="provider_error"` — is returned as though it were the final
    step, so `classify` sees the death and `_pass_rate` re-takes the sample
    instead of scoring a miss on a step that never ran.
    """

    def runner(app: EvalApp, scenario: str, utterance: str) -> EvalRun:
        """One sample of a multi-step probe: the earlier steps, then the one
        being scored."""
        record.clear()
        record["steps"] = []
        for index, said in enumerate(earlier, start=1):
            # Both seams reset the meters themselves; done here too so an
            # injected seam cannot leave the previous step's calls standing and
            # a per-step count read as a running total. Same for the limiter:
            # `_sample` clears it once a sample, and a probe on the delegate
            # seam spends one message per step.
            app.provider.reset()
            app.tracer.reset()
            reset_rate_limit()
            run = seam(app, f"{scenario}/{index}", said)
            record["steps"].append(_step_record(app, said, run))
            if run.status_code != 200 or run.stop_reason == STOP_PROVIDER_ERROR:
                return run
        app.provider.reset()
        app.tracer.reset()
        reset_rate_limit()
        return seam(app, scenario, utterance)

    # Which seam this composite actually drove, for the report's own honesty:
    # `_pass_rate` files a sample under `panel` or `delegate` by identity, and
    # without this every multi-step probe would be recorded against the delegate
    # wire it never touched.
    runner.seam = seam  # type: ignore[attr-defined]
    return runner


def _seam_name(runner: Callable[[EvalApp, str, str], EvalRun]) -> str:
    """Which HTTP seam a scenario's samples went through, as the report says it.

    Reads the composite runner's `.seam` when there is one, and falls back to
    the runner itself — so `_run`, `_run_panel` and anything built out of them
    are all named correctly, and a runner nobody has taught it about is named
    after the default rather than silently.
    """
    seam = getattr(runner, "seam", runner)
    return "panel" if seam is _run_panel else "delegate"


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

    **A budget stop and empty text were not the only ways a turn can arrive
    answerless** (audit 2026-09-05, finding 10). A truncated narrator keeps the
    planner's `completed` — the loop drops the fragment and no truncation field
    survives — and the pipeline fills the empty reply with `api.STUCK_REPLY`. So
    a commentary check saw a `completed` turn carrying nonempty text and read it
    as an answer, when what the player got was the canned "say it again?" line:
    `advice_capture_survives_guard`'s "no guard fired and nothing says scratch
    that" passes over it perfectly. The stuck line is app-owned text, which is
    the only reason reading it here is allowed at all — same precedent as that
    scenario's "scratch that" check.
    """
    if run.stop_reason in BUDGET_STOPS:
        raise VacuousRun(
            f"stopped on {run.stop_reason}: no narrator ran, so nothing was tested"
        )
    if not run.assistant["content"]:
        raise VacuousRun("no commentary was produced, so nothing was tested")
    if STUCK_REPLY in run.assistant["content"]:
        raise VacuousRun(
            "the pipeline answered with the canned stuck line; no narrator "
            "answer was tested"
        )


def _sample(
    engine: EnginePlayer,
    label: str,
    utterance: str,
    check: Callable[[EvalApp, dict[str, Any]], None],
    setup: Callable[[EvalApp], None] | None,
    runner: Callable[[EvalApp, str, str], EvalRun],
    requires_narrator: bool,
    route: str | None,
) -> tuple[Outcome, EvalRun, BaseException | None]:
    """One sample on a fresh app, and what kind of sample it turned out to be.

    The check is only run on a turn that actually happened. On a dead provider it
    would raise off the `_NO_TURN` stub and read as a HARNESS bug — which
    `classify` deliberately never retries — so the guard here is what keeps a
    crash classified as the crash it is.

    The route is asserted *first*, ahead of the vacuity check and the scenario's
    own check, because a wrong route makes both of those meaningless rather than
    failed: the resign fast path answers "I resign" with a real tool call, a real
    armed op and a real question, so a resignation check passes it on every
    sample while the model sits idle (audit finding 9).
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
                _assert_route(run, route)
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
    route: str | None = ROUTE_BRAIN,
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

    `route` is what the scenario claims to be measuring, and it defaults to the
    planner because every pass-rate scenario in this file is a model eval. It is
    asserted on each sample that actually ran a turn — a scenario silently
    answered by a deterministic short-circuit is not a scenario with a rate, it
    is a scenario with a bug (audit finding 9). Pass `None` to measure a
    scenario whose route genuinely varies.
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
            "seam": _seam_name(runner),
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
                route,
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


def _expect_san(expected: str) -> Callable[[EvalApp, dict[str, Any]], None]:
    """The check for a scenario whose whole question is *which move landed* —
    read off the board, which is what the player sees.

    It used to be read off the `make_move` result through a helper whose
    docstring claimed the board and delivered the tool's own report of itself
    ("reads the board back rather than trusting the trajectory", and it never
    touched `app`). The gap is not cosmetic: a turn that played Bxe6 and then
    undid it, or played it and then played again on top, reported Bxe6 and
    passed (audit 2026-09-05, "several pins are weaker than their stated
    intent").

    So "the right move landed" is asserted as the five separate facts it
    actually is — one accepted move on the wire, nothing else moving a piece,
    that move first on the board, exactly one engine reply on top of it, and the
    exchange settled with the player to move. The last two are what make it an
    *exchange* rather than a mutation: the scenarios using this run on a
    FEN-rooted session, whose move stack starts empty, so the history is exactly
    what this turn did.
    """

    def check(app: EvalApp, assistant: dict[str, Any]) -> None:
        moves = _legal_moves(assistant)
        assert len(moves) == 1, (
            f"expected exactly one accepted move, got {len(moves)}: "
            + (_trajectory(assistant) or "no tool calls")
        )
        mutated = [call["tool"] for call in _board_mutations(assistant)]
        assert mutated == ["make_move"], (
            "the move must be the only thing that moved a piece this turn: "
            + (_trajectory(assistant) or "no tool calls")
        )
        history = app.ctx.session.move_history()
        assert history[:1] == [expected], (
            f"expected {expected} on the board, and the history is {history}"
        )
        assert len(history) == 2, (
            f"expected the move and exactly one engine reply, got {history}"
        )
        assert app.ctx.session.turn == app.ctx.session.player_color, (
            "the exchange must settle with the player to move, and it is "
            f"{app.ctx.session.turn}'s turn"
        )

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
    utterance = "take that bishop move back and play d4 instead"

    def setup(app: EvalApp) -> None:
        for san in ("e4", "b6", "Nf3", "h6", "Bc4", "a5"):
            assert app.ctx.session.submit_move(san).legal
        _stays_a_model_eval(utterance, app.ctx.session.fen())

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
        utterance,
        check,
        floor=floor,
        setup=setup,
    )

    _assert_floor(result, floor)


@pytest.mark.xfail(
    reason=(
        "known gap: gemma-4-12b reads two named moves as a plies count three "
        "times in four — `undo(plies=2)` (one exchange, not two) or "
        "`undo(plies=1)` (the engine's reply alone) — then plays the "
        "replacement onto a board that still holds the second move. Measured "
        "at 2/10 across the gate's two blocks and 5/20 alone on 2026-09-04, "
        "with the loop stall this scenario was written for gone from every "
        "sample. Model understanding, so the lever is `undo`'s description "
        "(a prompt change, gated on this scenario — TODO.md). Non-strict: the "
        "invariant is right and it XPASSes when the model reads the count. "
        "Filtered to AssertionError: an unfiltered xfail also absorbs a "
        "harness bug or an infra abort (`pytest.fail` raises `Failed`, a bad "
        "check raises `KeyError`), and an xfail line is then no longer "
        "evidence of the known understanding miss (audit 2026-09-05). PR 5 "
        "strengthened what a pass means here — exactly four plies on the "
        "board, the player to move, nothing left armed, a `completed` stop and "
        "3–5 model calls — so the marker now also covers a turn that reaches "
        "the right prefix and then keeps working past it."
    ),
    raises=AssertionError,
    strict=False,
)
def test_eval_undo_twice_and_replace_is_one_turn(engine: EnginePlayer) -> None:
    """ "undo the bishop move and undo the knight move, then play d4" — two
    takebacks and a replacement, one utterance. Live (2026-07-30 and
    2026-08-08) the planner did exactly the right thing — `undo`, then `undo`
    again — and the loop cut it off: the second call carried the same empty
    arguments as the first, the stall rule read that as a repeat, and the phase
    ended with the replacement never played ("back to square one, hit me with
    that bishop move whenever you're ready"). The rule now reads the results,
    and a second `undo` that pops a different exchange is progress
    (`llama_brain.py`; the loop itself is pinned in `test_llama_brain.py`).

    Pre-fix this came in 1/5: two samples were that stall (`undo → undo`,
    `no_progress`, no move), two were the model taking back one exchange for
    two, and the one pass issued both undos in a single model turn. Post-fix
    the stall is gone and what remains is the count misread — see the marker.

    Behavioral, like `undo_and_replace`: however the model spells the takeback
    (two calls, or one asking for four plies), both exchanges must be gone and
    the replacement must stand where the first of them stood.

    **What a pass means here is stricter since the 2026-09-05 audit** (its
    proposal 1, an edit to this scenario rather than a second green duplicate).
    A history *prefix* accepts a turn that landed the right position and then
    kept going — another undo, a second move, a budget stop on top of correct
    work — so the end state is pinned exactly: four plies, the player to move,
    nothing armed, a stop the turn chose itself, and a round-trip envelope. The
    envelope is 3–5 because both spellings are legitimate: one planner turn
    carrying `undo, undo, make_move` costs the tool turn, the handoff note and
    the narrator, and a planner that spends a turn per action costs two more."""
    utterance = "undo the bishop move and undo the knight move, then play d4 instead"

    def setup(app: EvalApp) -> None:
        for san in ("e4", "b6", "Nf3", "h6", "Bc4", "a5"):
            assert app.ctx.session.submit_move(san).legal
        _stays_a_model_eval(utterance, app.ctx.session.fen())

    def check(app: EvalApp, assistant: dict[str, Any]) -> None:
        _in_order(assistant, "undo", "make_move")
        history = _history(app.client)
        # Both of White's last moves and the engine's answers to them are gone;
        # d4 stands where Nf3 stood (plus whatever the engine answered it with).
        assert history[:3] == ["e4", "b6", "d4"], history
        assert "Nf3" not in history and "Bc4" not in history, (
            f"a takeback did not stick: {history}"
        )
        # And nothing else happened. Four plies is the *settled* exchange — the
        # replacement and exactly one engine answer — so a third undo, a second
        # move or a half-finished exchange all fail here rather than passing on
        # the prefix above.
        assert len(history) == 4, (
            f"expected the two takebacks, d4 and one engine reply: {history}"
        )
        assert app.ctx.session.turn == app.ctx.session.player_color, (
            f"the exchange must settle with the player to move, and it is "
            f"{app.ctx.session.turn}'s"
        )
        assert app.ctx.live_pending() is None, (
            f"a takeback turn armed something: {app.ctx.live_pending()!r}"
        )
        _assert_completed(app.tracer.last)
        assert 3 <= len(app.provider.calls) <= 5, (
            "expected the tool turn(s), the handoff note and the narrator, got "
            f"{len(app.provider.calls)} model calls"
        )

    floor = _FLOORS["undo_twice_and_replace"]
    result = _pass_rate(
        engine,
        "undo_twice_and_replace",
        utterance,
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
    utterance = "what was my mistake?"

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
        _stays_a_model_eval(utterance, app.ctx.session.fen())

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
        utterance,
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
    utterance = "let's play chess as black"

    def setup(app: EvalApp) -> None:
        # The only setup this scenario needs, and it needs it: the deep link's
        # own intent string is not a move phrase or a resignation today, and
        # nothing but this line would notice if a parser grew to cover it.
        _stays_a_model_eval(utterance, app.ctx.session.fen())

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
        utterance,
        check,
        floor=floor,
        setup=setup,
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
    utterance = "load up the game I saved as scholars"

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
        # After the swap, because that is the board the utterance is judged
        # against — the one the request will actually arrive on.
        _stays_a_model_eval(utterance, app.ctx.session.fen())

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
        utterance,
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

    **Until this PR the scenario did not measure that at all.** Its utterance
    was "you know what, I give up. I resign", which `fastparse.parse_resign`
    settles whole-clause, so the pipeline dispatched `resign` itself and the
    model was never asked: a same-day control gate run reads `route=resign
    model_calls=0 trajectory=[resign(color="white")!]` on every sample, and the
    recorded 5/5 was the parser's, not the agent's (audit 2026-09-05, finding
    9). The literal utterance keeps its own scenario as a zero-model routing
    lock (`resign_literal_fast_path`); this one moves to a resignation stated in
    words no parser claims, so reaching the tool takes understanding the ask.
    The route pin comes from `_pass_rate`'s default and is what stops the same
    thing happening again quietly.

    What the eval still cannot reach is the state live play had and this
    doesn't: a long transcript, and verbosity=low (see the long-transcript
    family).

    **This is the audit's proposed `resign_intent_reaches_planner`** (2026-09-05,
    proposal 9), landed as a strengthening rather than a second scenario: PR 1
    had already moved this one onto that proposal's exact utterance and setup,
    so a new green case beside it would have sampled the same thing twice. What
    the proposal adds is that the outcome is no longer a disjunction. Six
    replayed plies mean the player has invested in the game and it is not over,
    so `tools._gate` arms *every* time: the refusal branch is the contract, not
    one of two acceptable answers, and the armed op has to carry the player's
    own side (`tools.resign` resolves a bare call to `session.player_color`
    before the gate sees it — trace review, finding 8, where the default was
    the side to move and only coincidentally the player's)."""

    before: dict[str, Any] = {}

    def setup(app: EvalApp) -> None:
        for san in ("e4", "e5", "Nf3", "Nc6", "Bb5", "a6"):
            assert app.ctx.session.submit_move(san).legal
        # The gate's own premises, so a scenario that quietly stopped arming
        # (a rule change about what counts as investment) fails as a premise
        # rather than as the model.
        assert not app.ctx.session.is_game_over()
        assert app.ctx.live_pending() is None
        _stays_a_model_eval(_RESIGN_UTTERANCE, app.ctx.session.fen())
        before["history"] = app.ctx.session.move_history()
        before["settings"] = app.ctx.settings.snapshot()

    def check(app: EvalApp, assistant: dict[str, Any]) -> None:
        assert _tool_calls(assistant), (
            "claimed an outcome with no tool call: " + assistant["content"]
        )
        # Called, whatever the gate then did with it — a refused resign rides
        # on the wire as an `error`, so this counts attempts, not successes.
        assert _attempted(assistant, "resign"), (
            f"expected a resign call: {_trajectory(assistant)}"
        )
        # And refused: on this board the gate has to arm rather than run, so a
        # resignation that *went through* means the gate stood aside on a game
        # the player is six plies into.
        assert _succeeded(assistant, "resign") == [], (
            "the gate must refuse an unconfirmed resignation on a played game: "
            + _trajectory(assistant)
        )
        # The armed op has to *be* the resignation: `live_pending`, by name, so
        # some other pending question cannot pass for this one — and for the
        # player's own side, which is the fact a bare `resign()` leans on.
        pending = app.ctx.live_pending()
        assert pending is not None and pending.name == "resign", (
            f"no resignation was armed for the yes: pending={pending!r}"
        )
        assert pending.args.get("color") == app.ctx.session.player_color, (
            f"armed a resignation for {pending.args.get('color')!r}, and the "
            f"player is {app.ctx.session.player_color!r}"
        )
        # A question is not an ending: the game is still there to lose.
        assert not app.ctx.session.is_game_over(), "the game ends on the yes"
        assert app.ctx.session.move_history() == before["history"]
        assert app.ctx.settings.snapshot() == before["settings"], (
            "a resignation ask must not change a setting the player owns"
        )
        _assert_not_guarded(app.tracer.last)
        _assert_completed(app.tracer.last)
        assert 3 <= len(app.provider.calls) <= 4, (
            "expected the planner's resign turn, its note and the narrator "
            f"(one recovery allowed), got {len(app.provider.calls)} model calls"
        )
        assert all(call.thinking is False for call in app.provider.calls), (
            "a resignation is not analysis — thinking stays OFF"
        )

    floor = _FLOORS["resign_never_pretends"]
    result = _pass_rate(
        engine,
        "resign_never_pretends",
        _RESIGN_UTTERANCE,
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
    utterance = "what should I play here?"

    def setup(app: EvalApp) -> None:
        for san in ("e4", "e5", "Nf3", "Nc6"):
            assert app.ctx.session.submit_move(san).legal
        _stays_a_model_eval(utterance, app.ctx.session.fen())

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
        utterance,
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


def test_eval_advice_capture_survives_guard(engine: EnginePlayer) -> None:
    """A hint whose best move is a *capture* reaches the player.

    The pair to `advice_is_engine_backed`: that one measures whether the model
    earns the right to give advice, this one whether the pipeline lets the
    advice out. They are different failures with the same symptom — no hint —
    and the walkthrough hit the second one three times out of three (2026-09-04
    defect 1). The engine said Rxd1, Glitch said "Take the rook. Rxd1 is the
    move.", and the honesty guard read the imperative as a completed capture,
    found no rook in the board's capture record and replaced the whole answer
    with "Scratch that — I said something the board doesn't back up."

    So the position is one where the recapture is the move and nothing else is
    close: after 1.e4 d5 2.Nc3 dxe4, `Nxe4` is a pawn back and every candidate
    the engine names is a capture. Whatever Glitch says about it, the guard
    must not eat it.

    Scored on the pipeline's verdict rather than on wording — the trace record
    says whether the guard fired, and which class — because *how* he offers a
    capture is exactly the thing the suite must not pin."""
    utterance = "what should I play here?"

    def setup(app: EvalApp) -> None:
        for san in ("e4", "d5", "Nc3", "dxe4"):
            assert app.ctx.session.submit_move(san).legal
        # The premise, asserted rather than assumed: a scenario whose best move
        # quietly stopped being a capture would pass while measuring nothing.
        best = app.ctx.engine.get_best_moves(app.ctx.session, n=1)[0]
        assert "x" in best.san, f"the premise needs a capture, got {best.san}"
        _stays_a_model_eval(utterance, app.ctx.session.fen())

    def check(app: EvalApp, assistant: dict[str, Any]) -> None:
        traced = app.tracer.last
        assert traced.get("guarded") is not True, (
            "the honesty guard ate a correct hint "
            f"({','.join(traced.get('guarded_claims') or ())}): "
            f"{traced.get('suppressed')!r}"
        )
        # And the player got advice, not an apology: the canned replacements
        # all open the same way, and none of them is an answer to the ask.
        assert "scratch that" not in assistant["content"].lower()

    floor = _FLOORS["advice_capture_survives_guard"]
    result = _pass_rate(
        engine,
        "advice_capture_survives_guard",
        utterance,
        check,
        floor=floor,
        setup=setup,
        # A canned stuck line is not a surviving hint; it is a turn that never
        # reached the narrator and so never tested the guard.
        requires_narrator=True,
    )

    _assert_floor(result, floor)


def test_eval_verbosity_up_from_low(engine: EnginePlayer) -> None:
    """ "talk more" moves the setting, not just this one reply.

    Walkthrough #3, twice out of twice: the model said it would give more of
    the breakdown, `set_verbosity` was never called, the setting stayed `low`
    on disk and the next turn was as terse as the last. Verbosity is a
    persistent setting the player owns, so a turn that only *sounds* chattier
    has not done what was asked.

    From `low` rather than from the default, because that is the state the
    player is in when they ask — and because from `normal` a turn could stumble
    into the right end of the enum. What is asserted is the direction and the
    persistence, never the wording of the reply."""
    utterance = "talk more"

    def setup(app: EvalApp) -> None:
        app.ctx.settings.verbosity = "low"
        _stays_a_model_eval(utterance, app.ctx.session.fen())

    order = {"low": 0, "normal": 1, "high": 2}

    def check(app: EvalApp, assistant: dict[str, Any]) -> None:
        assert _successful(assistant, "set_verbosity"), (
            "a chattiness ask is a setting change: "
            + (_trajectory(assistant) or "no tool calls")
        )
        assert order[app.ctx.settings.verbosity] > order["low"], (
            f"the setting must actually move up, got {app.ctx.settings.verbosity}"
        )
        assert _board_mutations(assistant) == []
        # And it is not narrated past a guard: an unbacked claim of the change
        # is the other half of the same defect.
        traced = app.tracer.last
        assert traced.get("guarded") is not True, (
            f"guarded ({','.join(traced.get('guarded_claims') or ())}): "
            f"{traced.get('suppressed')!r}"
        )

    floor = _FLOORS["verbosity_up_from_low"]
    result = _pass_rate(
        engine,
        "verbosity_up_from_low",
        utterance,
        check,
        floor=floor,
        setup=setup,
    )

    _assert_floor(result, floor)


def test_eval_position_is_described_not_evaluated(engine: EnginePlayer) -> None:
    """ "what's the position?" gets a description, not a verdict.

    Walkthrough smaller note, twice out of twice: the ask routed to
    `evaluate_position` and was answered with an eval ("You're cooked"). Two
    causes, both code's — `evaluate_position`'s description was the only tool
    text in the registry carrying the word "position", and the narrator, the
    phase that actually speaks, is handed no board at all (#193), so even a
    planner that declined the eval left it nothing to describe from.
    `describe_position` is the tool whose *result* is a description, and this
    scenario is the lock on the routing between the two.

    The position is `judgment_question`'s on purpose: the same board where
    "how am I doing?" must reach an analysis tool is the board where "what's
    the position?" must not. What is pinned is the trajectory — which tool
    answered, and that no verdict tool ran — never a word of the reply: how
    Glitch reads a position out is exactly the thing this suite must not pin.

    The settings assertion folds in TODO.md's standing item ("a question turn
    shouldn't mutate settings the player owns"): a read-only ask that comes
    back having changed difficulty or verbosity has done something the player
    did not ask for, and this is the cheapest place that can see it.
    """
    utterance = "what's the position?"
    before: dict[str, Any] = {}

    def setup(app: EvalApp) -> None:
        for san in ("e4", "e5", "Nf3", "Nc6"):
            assert app.ctx.session.submit_move(san).legal
        _stays_a_model_eval(utterance, app.ctx.session.fen())
        before["settings"] = app.ctx.settings.snapshot()

    def check(app: EvalApp, assistant: dict[str, Any]) -> None:
        assert _successful(assistant, "describe_position"), (
            "a description ask is answered by the tool that describes: "
            + (_trajectory(assistant) or "no tool calls")
        )
        called = {call["tool"] for call in _tool_calls(assistant)}
        assert not called & _VERDICT_TOOLS, (
            "a description ask is not a verdict ask: "
            + (_trajectory(assistant) or "no tool calls")
        )
        assert _board_mutations(assistant) == [], "a question must not move a piece"
        assert app.ctx.settings.snapshot() == before["settings"], (
            "a question turn must not change a setting the player owns"
        )
        traced = app.tracer.last
        assert traced.get("guarded") is not True, (
            f"guarded ({','.join(traced.get('guarded_claims') or ())}): "
            f"{traced.get('suppressed')!r}"
        )

    floor = _FLOORS["position_is_described"]
    result = _pass_rate(
        engine,
        "position_is_described",
        utterance,
        check,
        floor=floor,
        setup=setup,
        # A turn that never reached the narrator answered with the canned stuck
        # line, which describes nothing — the tool call alone is not the
        # feature working.
        requires_narrator=True,
    )

    _assert_floor(result, floor)


def _refused_not_asked(
    before: dict[str, Any],
) -> Callable[[EvalApp, dict[str, Any]], None]:
    """The check both impossible-request scenarios share: nothing moved, nothing
    was set, nothing was guarded, and the reply is not a clarifying question.

    This is the suite's first check that reads the model's *wording*, and it is
    deliberate: the defect here is a reply shape. The board is untouched either
    way, the trajectory is legitimately variable (attempt-and-concede, or
    decline outright — `honest_illegal` already pins that half), and what went
    wrong live is only visible in what the player was asked. So the check is as
    narrow as it can be — a clarifying question, meaning the word "which" and a
    question mark — and asserts nothing about how the refusal is phrased. The
    precedent is `advice_capture_survives_guard`, which reads content for the
    app's own canned "scratch that" line; this one reads it for a shape the app
    is no longer supposed to produce.
    """

    def check(app: EvalApp, assistant: dict[str, Any]) -> None:
        assert _legal_moves(assistant) == [], "must not fabricate a legal move"
        assert _board_mutations(assistant) == []
        assert app.ctx.session.move_history() == []
        assert app.ctx.settings.snapshot() == before["settings"], (
            "a refusal turn must not change a setting the player owns"
        )
        traced = app.tracer.last
        assert traced.get("guarded") is not True, (
            f"guarded ({','.join(traced.get('guarded_claims') or ())}): "
            f"{traced.get('suppressed')!r}"
        )
        content = assistant["content"]
        assert not (re.search(r"\bwhich\b", content, re.I) and "?" in content), (
            f"a request no piece can carry out is illegal, not ambiguous: {content!r}"
        )

    return check


def _impossible_request(
    engine: EnginePlayer, scenario: str, utterance: str
) -> RateResult:
    """One impossible request on move 1, sampled to a rate."""
    before: dict[str, Any] = {}

    def setup(app: EvalApp) -> None:
        _stays_a_model_eval(utterance, app.ctx.session.fen())
        before["settings"] = app.ctx.settings.snapshot()

    return _pass_rate(
        engine,
        scenario,
        utterance,
        _refused_not_asked(before),
        floor=_FLOORS[scenario],
        setup=setup,
        # The reply is the whole verdict here, so a turn that never reached the
        # narrator has nothing to read and is a non-pass rather than a silent
        # green.
        requires_narrator=True,
    )


def test_eval_impossible_move_is_refused_not_asked(engine: EnginePlayer) -> None:
    """ "bishop to a1" on move 1 is illegal, and must be answered as illegal.

    Walkthrough smaller note: it came back as "Which one?" — a clarifying
    question about a move neither bishop, nor any other piece, can make. The
    contract had a hole rather than the model having a bad day: told never to
    judge legality itself, to submit only entries of `legal_moves`, and to ask
    when "which piece" is unclear, asking was the one option those three rules
    left open. The planner prompt now names the third case (words that fit no
    entry at all are not ambiguous — they are illegal), and this measures it.
    Pre-fix on main: 0/5, "Which bishop, bro?" every sample.
    """
    result = _impossible_request(
        engine, "impossible_move_is_refused_not_asked", "bishop to a1"
    )
    _assert_floor(result, _FLOORS["impossible_move_is_refused_not_asked"])


def test_eval_impossible_capture_is_refused_not_asked(engine: EnginePlayer) -> None:
    """ "take the pawn" on move 1 — the walkthrough's other example, and the
    same hole from the other side. Nothing on the board can be captured, so
    there is no pawn to choose between; but the words name no square, so there
    is nothing to submit either, and "which piece" was the only rule left that
    applied — the first cut of the fix (a *named* move that fits nothing) still
    answered this one "Which pawn, bro?" 2/2. The bullet now says a capture
    when nothing can be captured is not ambiguous either.
    """
    result = _impossible_request(
        engine, "impossible_capture_is_refused_not_asked", "take the pawn"
    )
    _assert_floor(result, _FLOORS["impossible_capture_is_refused_not_asked"])


def _constraint_respected(
    before: dict[str, Any],
) -> Callable[[EvalApp, dict[str, Any]], None]:
    """The check both constraint scenarios share: no difficulty call landed,
    every setting the player owns is where it was, nothing moved, nothing
    guarded — never the wording of the reply, which is free to ask however
    Glitch likes."""

    def check(app: EvalApp, assistant: dict[str, Any]) -> None:
        assert _successful(assistant, "set_difficulty") == [], (
            "the player ruled this call out: "
            + (_trajectory(assistant) or "no tool calls")
        )
        assert app.ctx.settings.tier == DEFAULT_TIER, (
            f"difficulty moved: tier={app.ctx.settings.tier!r} "
            f"skill_level={app.ctx.settings.skill_level!r} "
            f"elo={app.ctx.settings.elo!r}"
        )
        assert app.ctx.settings.snapshot() == before["settings"], (
            "a turn that rules out the only lever must leave every setting the "
            "player owns where it was"
        )
        assert _board_mutations(assistant) == [], "an ask about strength is not a move"
        traced = app.tracer.last
        assert traced.get("guarded") is not True, (
            f"guarded ({','.join(traced.get('guarded_claims') or ())}): "
            f"{traced.get('suppressed')!r}"
        )

    return check


_CONSTRAINT_UTTERANCE = "go easy on me without changing the difficulty"


def test_eval_constraint_rules_out_the_only_lever(engine: EnginePlayer) -> None:
    """ "go easy on me without changing the difficulty" changes nothing —
    on a fresh conversation, which is the condition this one does *not*
    reproduce under (see the live-thread scenario below for the one that does).

    Walkthrough leftover: live, it called `set_difficulty(tier="beginner")`
    and said it would dial things back. The player named the one thing not to
    touch and it was touched anyway.

    The fix is `set_difficulty`'s description and not a guard. Knowing the
    change was excluded means reading the constraint out of the utterance,
    which is understanding and so the model's job (CLAUDE.md); and the honesty
    guard checks *claims* in the commentary against facts the turn produced,
    not actions the player ruled out. What the model lacked was a fact —
    engine strength is that setting and nothing else, because personality
    never shapes move choice — so an ask to go easy that excludes the
    difficulty leaves nothing to change and nothing to do but say so.

    Fresh, the pre-fix build already respected the constraint 29/30 across
    two builds (4/5 on main@4d165ec, 5/5 then 20/20 on main@f2e8cdc), so this
    scenario is a regression lock and no reproduction at all: the miss lives
    in the thread it happened in, not in the words.
    """
    before: dict[str, Any] = {}

    def setup(app: EvalApp) -> None:
        _stays_a_model_eval(_CONSTRAINT_UTTERANCE, app.ctx.session.fen())
        assert app.ctx.settings.tier == DEFAULT_TIER  # baseline strength
        before["settings"] = app.ctx.settings.snapshot()

    floor = _FLOORS["constraint_rules_out_the_only_lever"]
    result = _pass_rate(
        engine,
        "constraint_rules_out_the_only_lever",
        _CONSTRAINT_UTTERANCE,
        _constraint_respected(before),
        floor=floor,
        setup=setup,
        # Every assertion here is negative, so a turn that stopped on its budget
        # calls no setter and passes for never having run. The reply is the
        # whole of what this scenario wants.
        requires_narrator=True,
    )

    _assert_floor(result, floor)


# The thread the constraint miss actually happened in (2026-09-04 walkthrough,
# `turns.jsonl`), verbatim: a web-panel game at verbosity `low` — "talk less"
# had been honoured, the two "talk more" asks after it were each answered with
# a promise and no tool call — eleven turns deep when the player asked to be
# gone easy on without the difficulty changing. The board drags are recorded
# the way the panel records them (UCI in, the canned confirmation out).
_CONSTRAINT_FEN = "r1bqkb1r/pp3ppp/5n2/2Pp4/3n4/2P1P3/PP2BPPP/RN2K1NR w KQkq - 0 9"
_CONSTRAINT_THREAD: tuple[tuple[str, str], ...] = (
    ("d2d4", "Standard opening. Let's see where you take it.\n\nNf6."),
    ("c1f4", "Bf4, word. Keeping it simple.\n\nc5."),
    ("talk less", "Bet."),
    ("e2e3", "e3. d5."),
    ("c2c3", "c3. Nc6."),
    ("talk more", "Word, I'll give you a bit more of the breakdown from now on."),
    ("d4c5", "dxc5. e5."),
    ("f4e5", "Bxe5. Nxe5."),
    ("d1d4", "Qd4. Nc6."),
    (
        "talk more",
        "Word, I'll start giving you a bit more of the breakdown on what's "
        "happening out there. You're putting pressure on the center, but I'm "
        "keeping the pieces tight.",
    ),
    ("f1e2", "Be2. Nxd4."),
)


def test_eval_constraint_survives_a_live_thread(engine: EnginePlayer) -> None:
    """The same ask in the thread it misfired in — and this one reproduces.

    Fresh, the pre-fix build respected the constraint 29/30. Seeded with the
    walkthrough's own thread (panel seam, verbosity `low`, the exact position)
    it respected it **12/20** — the condition, not the words, is what the live
    miss was made of, the same finding the long-transcript family below
    records for resume, resign and capture. It is also what told the two
    halves of the fix apart. `set_difficulty`'s description carrying the two
    facts (strength is this setting and nothing else; a change the player
    rules out is not made) went **20/20** here. The same description with a
    list of the asks that mean this call in front of the caveat ("go easy,
    ease up, play harder, or crank it up is this call") went **9/20** — the
    trigger list outranked the caveat, which is #249's phrase-list failure
    written into prose. Measured at 20 samples an arm because five cannot
    tell 60% from 90%.

    What is asserted is the same as the fresh scenario's: no difficulty call,
    every player-owned setting where it was, nothing moved, nothing guarded.
    """
    before: dict[str, Any] = {}

    def setup(app: EvalApp) -> None:
        app.ctx.session = GameSession(fen=_CONSTRAINT_FEN, player_color="white")
        app.ctx.settings.verbosity = "low"
        for said, replied in _CONSTRAINT_THREAD:
            app.ctx.transcript.record(said, replied)
        _stays_a_model_eval(_CONSTRAINT_UTTERANCE, app.ctx.session.fen())
        assert app.ctx.settings.tier == DEFAULT_TIER
        before["settings"] = app.ctx.settings.snapshot()

    floor = _FLOORS["constraint_survives_a_live_thread"]
    result = _pass_rate(
        engine,
        "constraint_survives_a_live_thread",
        _CONSTRAINT_UTTERANCE,
        _constraint_respected(before),
        floor=floor,
        setup=setup,
        runner=_run_panel,
        requires_narrator=True,
    )

    _assert_floor(result, floor)


def test_eval_pgn_is_handed_over_not_recited(engine: EnginePlayer) -> None:
    """ "export the pgn" hands the notation over; it does not read it out.

    Walkthrough leftover: the narrator pasted the whole dump into the bubble —
    `[Event "?"] [Site "?"] [Date "????.??.??"] [Round "?"] [White "?"]
    [Black "?"]` and the movetext after it — which with voice on is also what
    the player heard. There was nothing else it could do: the reply was the
    only place the PGN appeared.

    Both halves of that are now the app's. The headers are filled in from what
    the app knows (`tools.pgn_headers`), and the notation itself is rendered
    under the reply with a button that copies it, so `export_pgn`'s description
    says the reply announces it is ready and recites nothing.

    Which makes the notation **app-owned text**, and that is the only reason
    this scenario may look at the words at all — the precedent is
    `advice_capture_survives_guard`'s "scratch that" check, where the string
    asserted against is likewise the app's own and not Glitch's. A recited PGN
    is the old behaviour reappearing, not a wording preference: how he says the
    export is ready stays entirely his.

    The two tokens are the two halves of the dump. `[Event` is a header — no
    prose contains it — and `1. e4` is the movetext's opening, from the fixed
    line the setup plays.
    """
    utterance = "export the pgn"
    before: dict[str, Any] = {}

    def setup(app: EvalApp) -> None:
        for san in ("e4", "e5", "Nf3", "Nc6"):
            assert app.ctx.session.submit_move(san).legal
        _stays_a_model_eval(utterance, app.ctx.session.fen())
        before["settings"] = app.ctx.settings.snapshot()

    def check(app: EvalApp, assistant: dict[str, Any]) -> None:
        assert _successful(assistant, "export_pgn"), (
            "an export ask is answered by exporting: "
            + (_trajectory(assistant) or "no tool calls")
        )
        assert _board_mutations(assistant) == [], "an export must not move a piece"
        assert app.ctx.settings.snapshot() == before["settings"], (
            "an export turn must not change a setting the player owns"
        )
        traced = app.tracer.last
        assert traced.get("guarded") is not True, (
            f"guarded ({','.join(traced.get('guarded_claims') or ())}): "
            f"{traced.get('suppressed')!r}"
        )
        content = assistant["content"]
        assert "[Event" not in content, "the headers are the app's to render"
        assert "1. e4" not in content, "the movetext is the app's to render"

    floor = _FLOORS["pgn_is_handed_over_not_recited"]
    result = _pass_rate(
        engine,
        "pgn_is_handed_over_not_recited",
        utterance,
        check,
        floor=floor,
        setup=setup,
        # Every text assertion here is negative, so a turn that stopped on its
        # budget recites nothing and would pass for never having spoken.
        requires_narrator=True,
    )

    _assert_floor(result, floor)


# --- compositions: several capabilities inside one utterance ------------------
#
# The gap the 2026-09-05 audit found in this file's coverage, stated exactly:
# "the only scenarios requiring multiple tool capabilities within one
# model-routed utterance are the two undo/replace variants". No active scenario
# pinned move-plus-read, save-plus-reset, settings-plus-move, resume-plus-
# description, or best-move-read-then-act — and a multi-tool request sitting in
# `_LIVE_TRANSCRIPT` is not coverage, because seeded history is never executed.
#
# These are the audit's twelve proposals, in its own order of expected
# information per GPU-minute, minus the two that were edits to scenarios that
# already existed (`undo_twice_and_replace` above, and `resign_never_pretends`,
# which PR 1 had already moved onto proposal 9's utterance and setup).
#
# Every one of them asserts the five dimensions the audit found unpinned across
# most of the suite — route, stop reason, model-call count, board end-state and
# a settings snapshot — plus the guard verdict wherever text reaches the
# player. The call envelopes are what the composition *costs*, not a timing
# estimate: three is the floor for any brain-routed turn (the planner's tool
# turn, its handoff note, the narrator), and the extra one is a second planner
# iteration, which several of these legitimately need.


def _fresh_save_dir(app: EvalApp, tmp_path: Any, taken: list[Path]) -> None:
    """Point this sample's saves at a directory nothing has written to yet.

    `tmp_path` is per-*test* and a pass-rate scenario takes five or more samples
    inside one test, so two samples share a save directory by default. That
    matters for the save/resume compositions specifically: the save *name* rides
    inside the utterance ("save this as checkpoint"), which `_pass_rate` holds
    fixed across samples, so a sample whose save never happened could still be
    answered by the previous sample's file and pass. The name stays literal and
    the directory is what varies — and the emptiness is asserted, because a
    guarantee nobody checks is how the resume scenarios once read 0/5 against a
    path that had gone stale (#214).
    """
    save_dir = Path(tmp_path) / f"sample-{len(taken) + 1}"
    save_dir.mkdir(parents=True)
    taken.append(save_dir)
    app.ctx.save_dir = save_dir
    assert saved_game_names(app.ctx) == [], "this sample must start with no saves"


def _reloaded(path: Path) -> GameSession:
    """The saved game as the app itself would read it back.

    Wrapped so a corrupt or unreadable save fails the *sample* rather than the
    scenario: `GameSession.load` raises `ValueError`, which `evalstats.classify`
    reads as a HARNESS bug and never retries or scores, and a save the model's
    turn wrote badly is data about the turn.
    """
    try:
        return GameSession.load(path)
    except (ValueError, OSError) as exc:
        raise AssertionError(f"the save at {path} does not load: {exc}") from exc


_KNIGHT_ASK = "move my kings knight"
_KNIGHT_PICK = "the one to f3"


def test_eval_ambiguous_knight_then_selection(engine: EnginePlayer) -> None:
    """Ask ambiguously, then answer the question — two turns, one thread.

    On a fresh board "my kings knight" names one piece and two squares, so the
    only correct first answer is the question "Nf3 or Nh3?". `ambiguous_move`
    already pins that an ambiguous *piece* ask is not guessed at; what nothing
    pinned is that the clarification survives to the player and that their
    answer to it lands the move they picked.

    **The first step is expected to fail on the pre-fix build**, and that is the
    point of it: audit finding 6, reproduced there over HTTP. A question naming
    two playable SANs on a board the turn did not change is exactly what the
    advice guard treats as unlicensed advice, so the correct question is
    replaced wholesale by the "I can't back that up" correction — leaving the
    player nothing to answer and the second step nothing to refer to. PR 4
    licenses a clarification that names two or more legal moves, and this
    scenario is its gate.

    The panel seam, because the second utterance is a *reference*: `_run` opens
    a fresh delegate conversation per call, so "the one to f3" would arrive with
    no question in front of it. `/api/command` reads and records
    `ctx.transcript`, which is the only place that thread exists.

    Step one's assertions live in `check` rather than in the runner because the
    runner is called outside `_sample`'s classifier — an assertion there would
    abort the scenario instead of scoring the miss it found.
    """
    before: dict[str, Any] = {}
    record: dict[str, Any] = {}

    def setup(app: EvalApp) -> None:
        # The premise the whole scenario rests on: two knight moves, one
        # phrase. Asserted rather than assumed — a fresh board is only
        # ambiguous for as long as both squares stay legal.
        legal = set(app.ctx.session.legal_moves())
        assert {"Nf3", "Nh3"} <= legal, sorted(legal)
        _stays_a_model_eval(_KNIGHT_ASK, app.ctx.session.fen())
        _stays_a_model_eval(_KNIGHT_PICK, app.ctx.session.fen())
        before["settings"] = app.ctx.settings.snapshot()
        before["version"] = app.ctx.board_version

    def check(app: EvalApp, assistant: dict[str, Any]) -> None:
        ask = _step(record, 1)
        # The question. Nothing may move on it — a turn that guessed a knight
        # answered a question it was supposed to ask — and it must reach the
        # player unsuppressed, which is the half PR 4 is about.
        _assert_route(ask["run"], ROUTE_BRAIN)
        _assert_completed(ask["traced"])
        _assert_not_guarded(ask["traced"])
        assert _board_mutations(ask["assistant"]) == [], (
            "the ambiguous ask must not move a piece: "
            + (_trajectory(ask["assistant"]) or "no tool calls")
        )
        assert ask["history"] == [], f"the board moved on the ask: {ask['history']}"
        assert ask["version"] == before["version"]
        assert ask["settings"] == before["settings"], (
            "a question turn must not change a setting the player owns"
        )
        assert ask["model_calls"] == 2, (
            "expected the planner's decline plus the narrator's question, got "
            f"{ask['model_calls']} model calls"
        )
        # The selection.
        assert len(_legal_moves(assistant)) == 1, (
            "expected exactly one accepted move: "
            + (_trajectory(assistant) or "no tool calls")
        )
        history = _history(app.client)
        assert history[:1] == ["Nf3"], f"the player picked f3 and got {history}"
        assert len(history) == 2, f"expected the move and one engine reply: {history}"
        assert app.ctx.session.turn == app.ctx.session.player_color
        assert app.ctx.live_pending() is None
        assert app.ctx.settings.snapshot() == before["settings"]
        _assert_not_guarded(app.tracer.last)
        _assert_completed(app.tracer.last)
        assert 3 <= len(app.provider.calls) <= 4, (
            "expected the planner's move turn, its note and the narrator, got "
            f"{len(app.provider.calls)} model calls"
        )

    floor = _FLOORS["ambiguous_knight_then_selection"]
    result = _pass_rate(
        engine,
        "ambiguous_knight_then_selection",
        _KNIGHT_PICK,
        check,
        floor=floor,
        setup=setup,
        runner=_run_steps(_KNIGHT_ASK, record=record),
    )

    _assert_floor(result, floor)


def test_eval_move_save_resume_finishes_exchange(
    engine: EnginePlayer, tmp_path: Any
) -> None:
    """Save mid-exchange, then load it back: the restored game must be one the
    player can move on.

    Audit finding 2, as a live composition. "play e4 and save this" is correctly
    answered by a move *then* a save, and the save therefore captures the
    interval between the player's move and the engine's reply — one ply, the
    engine to move. Nothing restores that obligation: `resume_game` swaps the
    session in and the pipeline's close beat only collects a reply the
    *coordinator* is mid-turn on, which a fresh restore is not. So the player
    loads their game and finds Black to move with nobody to move it.

    **Expected to fail on the pre-fix build**, on the final step: the restored
    history is `["e4"]` and the turn belongs to the engine. PR 3 hands a
    restored engine-to-move position to the coordinator to settle, the way
    `engine_opening_move` already does for a new game as black, and this is its
    gate.

    The engine's recomputed reply is deliberately not compared to the original
    one: Stockfish is sampled at the settings tier and a resumed game is a new
    search, so "the exchange is finished" is the contract and "finished the same
    way" is not.
    """
    step_one = "play e4 and save this as checkpoint"
    utterance = "load the game named checkpoint"
    before: dict[str, Any] = {}
    record: dict[str, Any] = {}
    taken: list[Path] = []

    def setup(app: EvalApp) -> None:
        _fresh_save_dir(app, tmp_path, taken)
        assert app.ctx.session.player_color == "white"
        _stays_a_model_eval(step_one, app.ctx.session.fen())
        before["settings"] = app.ctx.settings.snapshot()

    def check(app: EvalApp, assistant: dict[str, Any]) -> None:
        move = _step(record, 1)
        _assert_route(move["run"], ROUTE_BRAIN)
        _assert_completed(move["traced"])
        # Order is the whole of the first step: a save that ran *before* the
        # move saved the board the player asked to change.
        _in_order(move["assistant"], "make_move", "save_game")
        assert len(_legal_moves(move["assistant"])) == 1
        assert _succeeded(move["assistant"], "save_game"), "expected the save: " + (
            _trajectory(move["assistant"]) or "no tool calls"
        )
        assert move["history"][:1] == ["e4"], move["history"]
        assert len(move["history"]) == 2, (
            f"the live exchange must settle even when the save doesn't: "
            f"{move['history']}"
        )
        path = _save_path(app.ctx, "checkpoint")
        assert path.exists(), f"nothing was written to {path}"
        _reloaded(path)  # a file the app can read back, whatever it holds
        assert 3 <= move["model_calls"] <= 4, (
            f"expected the move+save turn, the note and the narrator, got "
            f"{move['model_calls']} model calls"
        )
        # The board the resume ask actually met — the first step moved it, so
        # the premise cannot be settled in `setup` the way a single-turn
        # scenario's is.
        _stays_a_model_eval(utterance, move["fen"])
        # The resume.
        assert _succeeded(assistant, "resume_game"), "expected the restore: " + (
            _trajectory(assistant) or "no tool calls"
        )
        assert _rejected_moves(assistant) == [], (
            "a restored board must not be submitted an illegal move: "
            + _trajectory(assistant)
        )
        history = _history(app.client)
        assert history[:1] == ["e4"], f"the wrong game came back: {history}"
        assert len(history) == 2, (
            f"the restored position owes an engine reply nobody collected: {history}"
        )
        assert app.ctx.session.player_color == "white"
        assert app.ctx.session.turn == "white", (
            f"the player cannot move on a board it is {app.ctx.session.turn}'s turn on"
        )
        assert app.ctx.live_pending() is None
        assert app.ctx.settings.snapshot() == before["settings"]
        _assert_not_guarded(app.tracer.last)
        _assert_completed(app.tracer.last)
        assert 3 <= len(app.provider.calls) <= 4, (
            f"expected the resume turn, the note and the narrator, got "
            f"{len(app.provider.calls)} model calls"
        )

    floor = _FLOORS["move_save_resume_finishes_exchange"]
    result = _pass_rate(
        engine,
        "move_save_resume_finishes_exchange",
        utterance,
        check,
        floor=floor,
        setup=setup,
        runner=_run_steps(step_one, record=record),
    )

    _assert_floor(result, floor)


def test_eval_save_then_new_game(engine: EnginePlayer, tmp_path: Any) -> None:
    """ "save this as checkpoint and start a new game" — a composition whose
    second half the gate is supposed to stop.

    Two things at once, and they are the two halves the audit says nothing
    covered: a read-then-write composition inside one utterance, and the
    destructive gate arming *after* other work in the same turn.
    `destructive_confirm` pins the arming on its own; this pins that a save
    beside it still lands, that it holds the game the player asked to keep, and
    that the reset stays unfired until they say so.

    The order is load-bearing rather than stylistic: a save that ran after a
    successful reset would be a save of the empty board, which is why the file
    is reloaded and compared to the game that was on screen.

    Verbosity `low` for the second half. A confirmed destructive op at `low` is
    answered by the app's own canned line (`api._destructive_confirmation`), so
    the player's "yes" costs **zero** model round trips — the deterministic
    half of the gate, and the cheapest assertion in this file.
    """
    utterance = "save this as checkpoint and start a new game"
    before: dict[str, Any] = {}
    taken: list[Path] = []

    def setup(app: EvalApp) -> None:
        _fresh_save_dir(app, tmp_path, taken)
        for san in ("e4", "e5", "Nf3", "Nc6"):
            assert app.ctx.session.submit_move(san).legal
        app.ctx.settings.verbosity = "low"
        assert not app.ctx.session.is_game_over()  # a real game stands to be lost
        _stays_a_model_eval(utterance, app.ctx.session.fen())
        before["history"] = app.ctx.session.move_history()
        before["settings"] = app.ctx.settings.snapshot()

    def check(app: EvalApp, assistant: dict[str, Any]) -> None:
        _in_order(assistant, "save_game", "new_game")
        assert _succeeded(assistant, "save_game"), "expected the save: " + (
            _trajectory(assistant) or "no tool calls"
        )
        assert _attempted(assistant, "new_game"), (
            "the reset has to be *called* — that is what arms the gate: "
            + _trajectory(assistant)
        )
        assert _succeeded(assistant, "new_game") == [], (
            "new_game must wait for a confirmation, not fire on the first ask"
        )
        path = _save_path(app.ctx, "checkpoint")
        assert path.exists(), f"nothing was written to {path}"
        assert _reloaded(path).move_history() == before["history"], (
            "the save does not hold the game the player asked to keep"
        )
        assert _history(app.client) == before["history"], "the board must stand"
        pending = app.ctx.live_pending()
        assert pending is not None and pending.name == "new_game", (
            f"the reset must be armed for the yes, and what is armed is {pending!r}"
        )
        assert app.ctx.settings.snapshot() == before["settings"], (
            "a save-and-reset ask must not change a setting the player owns"
        )
        _assert_not_guarded(app.tracer.last)
        _assert_completed(app.tracer.last)
        assert 3 <= len(app.provider.calls) <= 4, (
            f"expected the save+reset turn, the note and the narrator, got "
            f"{len(app.provider.calls)} model calls"
        )

        # The other half of the gate, and it is deterministic: no model call
        # stands between the player's yes and the reset. Metered separately —
        # the sample's own cost was already recorded by `_measured`.
        app.provider.reset()
        answered = app.client.post("/api/command", json={"text": "yes"})
        assert answered.status_code == 200, answered.text
        ran = [
            entry
            for entry in answered.json()["tool_results"]
            if entry["name"] == "new_game" and entry["result"].get("ok") is True
        ]
        assert len(ran) == 1, f"expected exactly one reset: {answered.json()}"
        assert app.ctx.session.move_history() == [], "confirmed: the game resets"
        assert app.ctx.session.player_color == "white", "the side is not the ask"
        assert app.ctx.pending is None
        assert app.provider.calls == [], (
            "a bare yes at verbosity=low is answered by the app's own line, so "
            f"it costs no model call: {len(app.provider.calls)}"
        )

    floor = _FLOORS["save_then_new_game"]
    result = _pass_rate(
        engine,
        "save_then_new_game",
        utterance,
        check,
        floor=floor,
        setup=setup,
    )

    _assert_floor(result, floor)


def test_eval_voice_setting_and_move(engine: EnginePlayer) -> None:
    """ "turn voice output off and play e4" — a setting and a move in one
    utterance, and neither may cost the other.

    Voice rather than difficulty on purpose (the audit's own note): a
    `set_difficulty` in the same turn as a move races the engine search that
    move started, so a scenario built on it would be measuring sequencing where
    it means to measure language routing. Voice output touches nothing the
    engine does.

    The settings pin is a whole-snapshot comparison with exactly one key
    changed, not a read of `voice_output` alone: an agent that turned voice off
    *and* dialled the difficulty down has done something the player did not ask
    for, and TODO.md's standing item is that a turn changes only the setting it
    was asked to change.
    """
    utterance = "turn voice output off and play e4"
    before: dict[str, Any] = {}

    def setup(app: EvalApp) -> None:
        # From on, so "off" is a real change rather than a no-op that a turn
        # calling nothing at all would also satisfy.
        app.ctx.settings.voice_output = True
        before["settings"] = app.ctx.settings.snapshot()
        _stays_a_model_eval(utterance, app.ctx.session.fen())

    def check(app: EvalApp, assistant: dict[str, Any]) -> None:
        assert _succeeded(assistant, "set_voice_output"), (
            "the setting is a tool call, not a promise: "
            + (_trajectory(assistant) or "no tool calls")
        )
        assert len(_legal_moves(assistant)) == 1, (
            "expected exactly one accepted move: "
            + (_trajectory(assistant) or "no tool calls")
        )
        history = _history(app.client)
        assert history[:1] == ["e4"], history
        assert len(history) == 2, f"expected the move and one engine reply: {history}"
        assert app.ctx.session.turn == app.ctx.session.player_color
        expected = {**before["settings"], "voice_output": False}
        assert app.ctx.settings.snapshot() == expected, (
            f"expected voice off and nothing else moved: {app.ctx.settings.snapshot()}"
        )
        assert app.ctx.live_pending() is None
        _assert_not_guarded(app.tracer.last)
        _assert_completed(app.tracer.last)
        assert 3 <= len(app.provider.calls) <= 4, (
            f"expected the setter+move turn, the note and the narrator, got "
            f"{len(app.provider.calls)} model calls"
        )
        assert all(call.thinking is False for call in app.provider.calls), (
            "a move and a setting are not analysis — thinking stays OFF"
        )

    floor = _FLOORS["voice_setting_and_move"]
    result = _pass_rate(
        engine,
        "voice_setting_and_move",
        utterance,
        check,
        floor=floor,
        setup=setup,
        requires_narrator=True,
    )

    _assert_floor(result, floor)


def test_eval_move_and_judgment(engine: EnginePlayer) -> None:
    """ "play e4, and how am I doing?" — act, then read, in one utterance.

    The composition `judgment_question` and `plain_move` cover only one half of
    each: a move turn that also has to answer a question is where the two
    contracts can collide, and where the engine's reply search and the
    analysis read overlap in time.

    `analyze_last_move` is accepted beside `evaluate_position` because
    `judgment_question` already accepts it and the tool descriptions make both
    honest answers to "how am I doing" straight after a move — one scores the
    position, the other scores the move that made it. What may not happen is the
    question being answered from the model's head, which is what pinning a
    *successful* verdict tool after the move is for.

    The thinking pins are the OFF→ON flip end to end (`llama_brain._thinking`):
    every planner turn is a parse even with an analysis result already in
    context, and the narrator — the one turn that reasons about it — inherits
    the flip. Two thinking turns for one analysis question would be the
    regression.
    """
    utterance = "play e4, and how am I doing?"
    before: dict[str, Any] = {}

    def setup(app: EvalApp) -> None:
        _stays_a_model_eval(utterance, app.ctx.session.fen())
        before["settings"] = app.ctx.settings.snapshot()

    def check(app: EvalApp, assistant: dict[str, Any]) -> None:
        assert len(_legal_moves(assistant)) == 1, (
            "expected exactly one accepted move: "
            + (_trajectory(assistant) or "no tool calls")
        )
        verdicts = [
            name
            for name in ("evaluate_position", "analyze_last_move")
            if _succeeded(assistant, name)
        ]
        assert verdicts, "a judgment question must route through an analysis tool: " + (
            _trajectory(assistant) or "no tool calls"
        )
        trajectory = [call["tool"] for call in _tool_calls(assistant)]
        assert trajectory.index("make_move") < min(
            trajectory.index(name) for name in verdicts
        ), f"the move comes first — it is what the judgment is about: {trajectory}"
        history = _history(app.client)
        assert history[:1] == ["e4"], history
        assert len(history) == 2, f"expected the move and one engine reply: {history}"
        assert app.ctx.session.turn == app.ctx.session.player_color == "white"
        assert app.ctx.settings.snapshot() == before["settings"], (
            "a move-and-read turn must not change a setting the player owns"
        )
        assert app.ctx.live_pending() is None
        _assert_not_guarded(app.tracer.last)
        _assert_completed(app.tracer.last)
        calls = app.provider.calls
        assert 3 <= len(calls) <= 4, (
            f"expected the move turn, the read, the note and the narrator, got "
            f"{len(calls)} model calls"
        )
        assert all(call.thinking is False for call in calls[:-1]), (
            "every planner turn is a parse, analysis result in context or not"
        )
        assert calls[-1].thinking is True, (
            "the turn that puts an evaluation into words must think"
        )

    floor = _FLOORS["move_and_judgment"]
    result = _pass_rate(
        engine,
        "move_and_judgment",
        utterance,
        check,
        floor=floor,
        setup=setup,
        requires_narrator=True,
    )

    _assert_floor(result, floor)


# The stale prose the resume-and-describe probe has to beat: a description of
# the board the player is on *now*, one turn before they replace it. Verbatim
# in shape from what `describe_position` produces, so the temptation is a real
# one — the model's own last answer, still true of a game that is about to stop
# existing.
_STALE_DESCRIPTION = (
    "Both queen's pawns are out — d4 and d5, and nothing else has left the "
    "back rank yet."
)


def _described(session: GameSession) -> dict[str, Any]:
    """What `describe_position` says about `session`, computed independently.

    The oracle for the resume composition: `describe_position` reads nothing but
    the session (placement, material, castling, check, last move, move number,
    plies, outcome — `tools.build_registry`), so the description of a restored
    game is a pure function of the save file and can be recomputed here from the
    bytes on disk. A fresh `ToolContext` and registry rather than the app's,
    because the app's are pointed at the live game and the whole question is
    whether the live game is the saved one.

    `FakeEngine` because no field of the result comes from an engine; passing
    one at all is only so the registry builds the same tool set.
    """
    return build_registry(ToolContext(session=session, engine=FakeEngine())).dispatch(
        "describe_position", {}
    )


def test_eval_resume_and_describe(engine: EnginePlayer, tmp_path: Any) -> None:
    """ "resume scholars and tell me where my pieces are" — restore, then
    describe the board you restored.

    The composition the audit calls out as specifically testing state *after* a
    replacement: `resume_game` swaps the whole session, and the description has
    to be of the incoming game rather than of the one that was on screen a
    moment ago. The transcript is seeded with exactly one turn — the model's own
    description of the `d4 d5` game it is about to discard — so there is stale
    prose in context saying something true of the wrong board. That is the
    self-poisoning shape the long-transcript family found (a stale assistant
    line outranking a fresh fact), aimed at a description instead of at a tool
    offer.

    The oracle is the save file, not a hand-typed expectation: the dispatched
    result must equal what `describe_position` produces from a `GameSession`
    rebuilt out of the bytes on disk (`_described`). Every field is compared,
    because every field of that result is a pure function of the session — there
    is nothing turn-specific in it to exempt.

    Panel seam, because `ctx.transcript` is the only place the stale turn can
    live; the delegate's fresh conversation would carry none.
    """
    utterance = "resume scholars and tell me where my pieces are"
    before: dict[str, Any] = {}

    def setup(app: EvalApp) -> None:
        app.ctx.save_dir = tmp_path
        saved = GameSession()
        for san in ("e4", "e5", "Nf3", "Nc6"):
            assert saved.submit_move(san).legal
        # `_save_path`, never a hand-built one — the same reason
        # `resume_not_denied` uses it: a save has to land where `resume_game`
        # reads, and a hard-coded path went stale once already (#214).
        path = _save_path(app.ctx, "scholars")
        path.parent.mkdir(parents=True, exist_ok=True)
        saved.save(path)
        before["save"] = json.loads(path.read_text())
        # A different game is in progress, and the model has just described it.
        for san in ("d4", "d5"):
            assert app.ctx.session.submit_move(san).legal
        app.ctx.transcript.record("what's the position?", _STALE_DESCRIPTION)
        _stays_a_model_eval(utterance, app.ctx.session.fen())
        before["settings"] = app.ctx.settings.snapshot()

    def check(app: EvalApp, assistant: dict[str, Any]) -> None:
        _in_order(assistant, "resume_game", "describe_position")
        assert _succeeded(assistant, "resume_game"), "expected the restore: " + (
            _trajectory(assistant) or "no tool calls"
        )
        described = _succeeded(assistant, "describe_position")
        assert described, (
            "a description ask is answered by the tool that describes: "
            + (_trajectory(assistant) or "no tool calls")
        )
        assert _history(app.client) == ["e4", "e5", "Nf3", "Nc6"], (
            "the saved game did not come back"
        )
        assert json.loads(described[0]["result"]) == _described(
            GameSession.from_dict(before["save"])
        ), "the description is not of the game that was restored"
        called = {call["tool"] for call in _tool_calls(assistant)}
        assert not called & _VERDICT_TOOLS, (
            "a description ask is not a verdict ask: " + _trajectory(assistant)
        )
        assert app.ctx.settings.snapshot() == before["settings"], (
            "a restore-and-read turn must not change a setting the player owns"
        )
        _assert_not_guarded(app.tracer.last)
        _assert_completed(app.tracer.last)
        assert 3 <= len(app.provider.calls) <= 4, (
            f"expected the resume+describe turn, the note and the narrator, got "
            f"{len(app.provider.calls)} model calls"
        )
        assert all(call.thinking is False for call in app.provider.calls), (
            "a description is not a verdict — thinking stays OFF"
        )

    floor = _FLOORS["resume_and_describe"]
    result = _pass_rate(
        engine,
        "resume_and_describe",
        utterance,
        check,
        floor=floor,
        setup=setup,
        runner=_run_panel,
        requires_narrator=True,
    )

    _assert_floor(result, floor)


def test_eval_best_move_then_play(engine: EnginePlayer) -> None:
    """ "ask Stockfish for its top move and play it for me" — read, then act on
    what came back.

    The read-then-act composition, and the one where the planner's contract is
    tightest: it is told to submit only moves from the `legal_moves` list
    injected at command start, and the move it is being asked to play arrives
    from a *tool result* instead. `get_legal_moves` is deliberately withheld
    from the HTTP brain, so there is no second list to consult — which is why
    this scenario does not require that tool and would be measuring the wrong
    thing if it did.

    Which move Stockfish picks is not pinned, and must not be: it is sampled at
    the settings tier and would make this a test of the engine. What is pinned
    is the *link* — the move on the board is the first candidate of the call
    that was actually made, read both off that call's result and off the board's
    own UCI history, because the audit's `_played` lesson is that a tool's
    report of itself is not evidence about the board.
    """
    utterance = "ask Stockfish for its top move and play it for me"
    before: dict[str, Any] = {}

    def setup(app: EvalApp) -> None:
        for san in ("e4", "e5", "Nf3", "Nc6"):
            assert app.ctx.session.submit_move(san).legal
        assert app.ctx.session.player_color == "white"
        assert app.ctx.session.turn == "white"  # the player is the one to move
        _stays_a_model_eval(utterance, app.ctx.session.fen())
        before["history"] = app.ctx.session.move_history()
        before["settings"] = app.ctx.settings.snapshot()

    def check(app: EvalApp, assistant: dict[str, Any]) -> None:
        consulted = _succeeded(assistant, "get_best_moves")
        assert consulted, "the move has to come from the engine: " + (
            _trajectory(assistant) or "no tool calls"
        )
        played = _legal_moves(assistant)
        assert len(played) == 1, "expected exactly one accepted move: " + _trajectory(
            assistant
        )
        _in_order(assistant, "get_best_moves", "make_move")
        candidates = json.loads(consulted[0]["result"])["moves"]
        assert candidates, "the engine reported no candidates to play"
        top = candidates[0]["uci"]
        assert json.loads(played[0]["result"])["uci"] == top, (
            f"played a move the engine did not put first: {_trajectory(assistant)}"
        )
        # And off the board, which is what the player has. `to_dict` is the
        # session's own UCI record, so this compares like with like without
        # re-deriving SAN.
        uci_history = app.ctx.session.to_dict()["moves"]
        assert len(uci_history) == 6, f"expected A plus one exchange: {uci_history}"
        assert uci_history[4] == top, (
            f"the board does not hold the engine's pick: {uci_history}"
        )
        history = _history(app.client)
        assert history[:4] == before["history"], f"the setup moved: {history}"
        assert app.ctx.session.turn == app.ctx.session.player_color
        assert app.ctx.live_pending() is None
        assert app.ctx.settings.snapshot() == before["settings"], (
            "a read-and-play turn must not change a setting the player owns"
        )
        _assert_not_guarded(app.tracer.last)
        _assert_completed(app.tracer.last)
        calls = app.provider.calls
        assert 4 <= len(calls) <= 5, (
            f"expected the read, the move, the note and the narrator, got "
            f"{len(calls)} model calls"
        )
        assert all(call.thinking is False for call in calls[:-1]), (
            "every planner turn is a parse, analysis result in context or not"
        )
        assert calls[-1].thinking is True, (
            "an analysis result landed this turn, so the narrator reasons about "
            "it — the OFF→ON flip is `llama_brain._thinking`"
        )

    floor = _FLOORS["best_move_then_play"]
    result = _pass_rate(
        engine,
        "best_move_then_play",
        utterance,
        check,
        floor=floor,
        setup=setup,
    )

    _assert_floor(result, floor)


# The three ways a player can answer a confirmation question in their own
# words, with the verdict `llama_brain.read_answer` has to reach for each. The
# literal "yes"/"no" are `fastparse.parse_confirmation`'s and are hard,
# zero-model unit tests in `test_command.py`; these are the ones that cost a
# model round trip, and each of the three takes a *different* route through the
# pipeline afterwards, which is why the route is parameterized alongside.
_FREEFORM_ANSWERS = [
    pytest.param("actually, forget it", "cancel", ROUTE_CONFIRMATION, id="cancel"),
    pytest.param("just do it", "confirm", ROUTE_CONFIRMATION, id="confirm"),
    pytest.param(
        "show me the position instead", "unrelated", ROUTE_BRAIN, id="unrelated"
    ),
]


@pytest.mark.parametrize(("answer", "label", "route"), _FREEFORM_ANSWERS)
def test_eval_freeform_confirmation_answers(
    engine: EnginePlayer, answer: str, label: str, route: str
) -> None:
    """A confirmation answered in words no parser claims — cancelled, confirmed,
    or changed to something else entirely.

    The confirmation reader (`LlamaBrain.read_answer`) is the one model call in
    the app outside the planner/narrator pair, and no live scenario measured it:
    a player who says "just do it" after a resign question has answered it as
    plainly as "yes", and reading that is understanding, which is the model's
    job. *Acting* on it is not — a confirm goes down the same `confirm_pending`
    path a bare yes takes, so the model never touches a destructive tool.

    Each condition pins a different shape of turn, and the route is the first
    thing that tells them apart:

    - **cancel** disarms and stops. No tools, no narrator, one round trip.
    - **confirm** runs the armed op deterministically. At verbosity `low` the
      reply is the app's own canned line, so the reader is again the only call —
      which is what makes "no planner stands between the answer and the
      resignation" an assertion rather than a claim.
    - **unrelated** disarms and *falls through* to the planner, so the turn is
      recorded on the brain route and costs the reader on top of an ordinary
      brain turn. That the reader's round trip is counted at all is the point:
      the audit found this route's cost under-reported.

    The resignation is armed through the app's own road rather than by writing
    `ctx.pending` — `_RESIGN_LITERAL` is settled by `parse_resign`, dispatched
    through the registry, refused by the gate and armed, at zero model calls
    (`resign_literal_fast_path` is the lock on that). So the question the reader
    is given is the real one, and the meters are reset afterwards so the sample
    measures only the answer.
    """
    before: dict[str, Any] = {}

    def setup(app: EvalApp) -> None:
        for san in ("e4", "e5", "Nf3", "Nc6"):
            assert app.ctx.session.submit_move(san).legal
        app.ctx.settings.verbosity = "low"
        armed = app.client.post("/api/command", json={"text": _RESIGN_LITERAL})
        assert armed.status_code == 200, armed.text
        pending = app.ctx.live_pending()
        assert pending is not None and pending.name == "resign", (
            f"the setup did not arm a resignation: {pending!r}"
        )
        assert app.provider.calls == [], (
            "arming must be free, or this scenario's call counts are measuring "
            f"the setup: {len(app.provider.calls)} model calls"
        )
        app.provider.reset()
        app.tracer.reset()
        # The premise: neither the literal reader nor either fast path settles
        # this answer, so the reader's round trip is the only thing that can.
        assert parse_confirmation(answer) is None, (
            f"`parse_confirmation` settles {answer!r}, so no model reads it"
        )
        _stays_a_model_eval(answer, app.ctx.session.fen())
        before["history"] = app.ctx.session.move_history()
        before["version"] = app.ctx.board_version
        before["settings"] = app.ctx.settings.snapshot()

    def check(app: EvalApp, assistant: dict[str, Any]) -> None:
        _assert_not_guarded(app.tracer.last)
        _assert_completed(app.tracer.last)
        calls = len(app.provider.calls)
        assert app.ctx.live_pending() is None, (
            f"an answered question must not stay armed: {app.ctx.live_pending()!r}"
        )
        assert app.ctx.settings.snapshot() == before["settings"], (
            "answering a confirmation must not change a setting the player owns"
        )
        if label == "cancel":
            assert _tool_calls(assistant) == [], (
                "a cancelled question runs nothing: " + _trajectory(assistant)
            )
            assert _history(app.client) == before["history"]
            assert app.ctx.board_version == before["version"]
            assert not app.ctx.session.is_game_over(), "cancelled, and still live"
            assert calls == 1, f"expected the reader and nothing else, got {calls}"
        elif label == "confirm":
            resigned = _succeeded(assistant, "resign")
            assert len(resigned) == 1, "expected exactly one resignation: " + (
                _trajectory(assistant) or "no tool calls"
            )
            assert app.ctx.session.is_game_over(), "confirmed, and still running"
            outcome = app.ctx.session.outcome()
            assert outcome is not None, "the game ended with no outcome recorded"
            assert outcome.winner != app.ctx.session.player_color, (
                f"the player's own resignation, and the outcome is {outcome!r}"
            )
            assert calls == 1, (
                "at verbosity=low a confirmed op is answered by the app's own "
                f"line, so the reader is the whole cost: {calls}"
            )
        else:
            assert _succeeded(assistant, "describe_position"), (
                "the new intent must be answered, not dropped with the op: "
                + (_trajectory(assistant) or "no tool calls")
            )
            ran = [tool for tool in DESTRUCTIVE_TOOLS if _succeeded(assistant, tool)]
            assert ran == [], f"a new intent is not a confirmation: {ran}"
            assert _history(app.client) == before["history"]
            assert app.ctx.board_version == before["version"]
            assert not app.ctx.session.is_game_over()
            assert 4 <= calls <= 5, (
                "expected the reader on top of the planner's turn, its note and "
                f"the narrator, got {calls} model calls"
            )

    floor = _FLOORS["freeform_confirmation_answers"]
    result = _pass_rate(
        engine,
        f"freeform_confirmation_answers[{label}]",
        answer,
        check,
        floor=floor,
        setup=setup,
        runner=_run_panel,
        route=route,
    )

    _assert_floor(result, floor)


# The audit's late-game stress fixture: 84 legal plies, White to move at move
# 43, generated deterministically (seed 260) and excluding terminal
# continuations. Synthetic legal play, and the file says so — it is a
# reproducible condition, not a claim about how humans play.
_LATE_GAME_PGN = Path(__file__).parent / "late_game_84_plies.pgn"


def _replay_late_game(app: EvalApp) -> list[str]:
    """Put the fixture on the board through the session's own legality gate.

    Replayed rather than rooted at a FEN, and that is the whole reason the
    fixture exists: the long-transcript family's `_LIVE_FEN` sessions have an
    *empty* move stack, so nothing in this suite could undo, review, claim a
    repetition or trip the destructive gate's investment test on a real game
    (`tools._player_has_moved`). An arbitrary move number is not a substitute
    for a move history.
    """
    with _LATE_GAME_PGN.open() as source:
        game = chess.pgn.read_game(source)
    assert game is not None and not game.errors, "the fixture did not parse"
    for move in game.mainline_moves():
        assert app.ctx.session.submit_move(move.uci()).legal
    assert app.ctx.session.plies == 84
    assert app.ctx.session.fullmove_number == 43
    assert not app.ctx.session.is_game_over()
    return app.ctx.session.move_history()


@pytest.mark.parametrize("history", ["seeded", "control"])
def test_eval_late_game_tool_composition(
    engine: EnginePlayer, tmp_path: Any, history: str
) -> None:
    """A save and a description on a real 40-move game.

    Everything else in this file happens inside the first ten plies or on a
    FEN with no history behind it. This is the first scenario on a board with
    84 recorded plies — a longer prompt, a longer move list in the state block,
    and a `describe_position` result with a full board to render.

    Paired conditions, for the same reason the long-transcript family is
    paired: `seeded` records twenty exchanges drawn from the replay itself, and
    `control` is the identical position with an empty transcript. If only one
    of them moves, the finding is about history; if both do, it is about the
    position. One variable at a time, or the number says nothing.

    **Prompt tokens are recorded, not capped.** They are already on the
    per-sample report line and in the JSONL record (`call_in`,
    `prompt_tokens` — `evalstats.split_tokens`), and a ceiling set before
    anyone has read a baseline is a guess with a number on it. The measurement
    comes first; the ceiling, if it is worth having, comes after.

    A lock: there is no evidence yet of a late-game miss. What it buys is a
    condition nothing else in the suite can reach.
    """
    utterance = "save this as late_game and tell me the position"
    before: dict[str, Any] = {}
    taken: list[Path] = []

    def setup(app: EvalApp) -> None:
        _fresh_save_dir(app, tmp_path, taken)
        moves = _replay_late_game(app)
        if history == "seeded":
            # The last twenty exchanges of the game actually on the board,
            # recorded as the panel records them. Real continuity rather than
            # filler: the model's own memory of this game, at the length
            # `Transcript` keeps.
            for index in range(len(moves) - 40, len(moves), 2):
                app.ctx.transcript.record(
                    f"play {moves[index]}", f"The player played {moves[index]}."
                )
            assert len(app.ctx.transcript.window()) == 40  # twenty exchanges
        else:
            assert app.ctx.transcript.window() == []
        _stays_a_model_eval(utterance, app.ctx.session.fen())
        before["fen"] = app.ctx.session.fen()
        before["history"] = moves
        before["settings"] = app.ctx.settings.snapshot()
        before["version"] = app.ctx.board_version

    def check(app: EvalApp, assistant: dict[str, Any]) -> None:
        assert _succeeded(assistant, "save_game"), "expected the save: " + (
            _trajectory(assistant) or "no tool calls"
        )
        assert _succeeded(assistant, "describe_position"), (
            "a description ask is answered by the tool that describes: "
            + (_trajectory(assistant) or "no tool calls")
        )
        path = _save_path(app.ctx, "late_game")
        assert path.exists(), f"nothing was written to {path}"
        restored = _reloaded(path)
        assert restored.fen() == before["fen"], "the save is of another position"
        assert restored.move_history() == before["history"], (
            "the save lost the game's history"
        )
        assert _history(app.client) == before["history"]
        assert app.ctx.board_version == before["version"], (
            "a save and a read must not move the board"
        )
        assert app.ctx.settings.snapshot() == before["settings"]
        assert app.ctx.live_pending() is None
        _assert_not_guarded(app.tracer.last)
        _assert_completed(app.tracer.last)
        assert 3 <= len(app.provider.calls) <= 4, (
            f"expected the save+describe turn, the note and the narrator, got "
            f"{len(app.provider.calls)} model calls"
        )
        assert all(call.thinking is False for call in app.provider.calls), (
            "a save and a description are not analysis — thinking stays OFF"
        )

    floor = _FLOORS["late_game_tool_composition"]
    result = _pass_rate(
        engine,
        f"late_game_tool_composition[{history}]",
        utterance,
        check,
        floor=floor,
        setup=setup,
        runner=_run_panel,
        requires_narrator=True,
    )

    _assert_floor(result, floor)


# Two ways a speech transcript mangles the same move. The homophone is the one
# STT gets wrong most reliably ("night"), the disfluency is what a spoken
# sentence actually looks like off Whisper — a filler in front, a politeness
# behind, and no punctuation anywhere. Neither is a phrasing `parse_move`
# settles, which is the premise `_stays_a_model_eval` re-checks per sample:
# repairing them is the model's job, and the fix for a miss is the model's
# understanding, never a rule added to the parser (CLAUDE.md).
_STT_UTTERANCES = [
    pytest.param("please put my night on f three", "homophone", id="homophone"),
    pytest.param("uh knight f three please", "disfluency", id="disfluency"),
]


@pytest.mark.parametrize(("utterance", "label"), _STT_UTTERANCES)
def test_eval_stt_knight_repair(
    engine: EnginePlayer, utterance: str, label: str
) -> None:
    """A mangled voice transcript still lands the move it names.

    Voice is the app's primary input and there was no deliberate STT-error
    family in this suite at all (audit, "Coverage and assertion gaps"): the
    canonical spoken phrasings that do exist are ones `parse_move` swallows, so
    they measure the parser. These two do not, so the repair is measured where
    it happens — in the model.

    `_expect_san` is the check because "which move landed" is the whole
    question, and it reads the *board*: one accepted move on the wire, nothing
    else moving a piece, Nf3 first on the board, exactly one engine reply on top
    of it, and the exchange settled with the player to move.
    """
    before: dict[str, Any] = {}
    landed = _expect_san("Nf3")

    def setup(app: EvalApp) -> None:
        assert "Nf3" in app.ctx.session.legal_moves()
        _stays_a_model_eval(utterance, app.ctx.session.fen())
        before["settings"] = app.ctx.settings.snapshot()

    def check(app: EvalApp, assistant: dict[str, Any]) -> None:
        landed(app, assistant)
        assert app.ctx.settings.snapshot() == before["settings"], (
            "a move turn must not change a setting the player owns"
        )
        assert app.ctx.live_pending() is None
        _assert_not_guarded(app.tracer.last)
        _assert_completed(app.tracer.last)
        assert 3 <= len(app.provider.calls) <= 4, (
            f"expected the move turn, the note and the narrator, got "
            f"{len(app.provider.calls)} model calls"
        )
        assert all(call.thinking is False for call in app.provider.calls), (
            "a move is not analysis — thinking stays OFF"
        )

    floor = _FLOORS["stt_knight_repair"]
    result = _pass_rate(
        engine,
        f"stt_knight_repair[{label}]",
        utterance,
        check,
        floor=floor,
        setup=setup,
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
    utterance = "load up the game I saved as scholars"
    # Once at body level rather than per-sample: every condition puts the same
    # `_LIVE_FEN` on the board, so the premise is a property of the scenario and
    # not of a sample.
    _stays_a_model_eval(utterance, _LIVE_FEN)

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
        utterance,
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

    **All three conditions measured the fast path, not the model** (audit
    2026-09-05, finding 9). The utterance was "you know what, I give up. I
    resign", which `fastparse.parse_resign` settles whole-clause, so the
    pipeline dispatched `resign` itself before the planner was ever built: a
    control gate run reads `route=resign model_calls=0
    trajectory=[resign(color="white")!]` on every sample of every condition, and
    the recorded 5/5 ×3 said nothing about whether a transcript can talk the
    model out of the tool. Worse, the three conditions differ only in what is in
    the *transcript*, which the resign route never reads — so the parametrize
    was measuring one thing three times.

    So the ask moves to a resignation stated in words no parser claims, and
    these three conditions now reach the planner with their transcripts intact.
    `_pass_rate`'s route pin is what keeps them there; the literal utterance
    keeps its own zero-model scenario (`resign_literal_fast_path`). The recorded
    "the live resign failure is still unexplained" stands as a statement about
    the *live* miss and no longer as a result from this scenario: whatever these
    conditions measure, they have not measured it yet."""
    _stays_a_model_eval(_RESIGN_UTTERANCE, _LIVE_FEN)

    def check(app: EvalApp, assistant: dict[str, Any]) -> None:
        assert _tool_calls(assistant), (
            "claimed an outcome with no tool call: " + assistant["content"]
        )
        # Any outcome, because the gate refusing it is a pass: what may never
        # happen is the outcome being claimed without the call.
        assert any(c["tool"] == "resign" for c in _tool_calls(assistant)), (
            f"expected a resign call: {_trajectory(assistant)}"
        )
        # Both branches are real, and on *this* board it is usually the first:
        # the session is FEN-rooted with no recorded plies, so the gate's
        # investment test (`tools._player_has_moved`) stands aside and the
        # resignation runs rather than arming. The fresh scenario, which replays
        # its six plies, exercises the other branch.
        pending = app.ctx.live_pending()
        assert app.ctx.session.is_game_over() or (
            pending is not None and pending.name == "resign"
        ), f"neither resigned nor armed a resignation: pending={pending!r}"

    floor = _FLOORS["long_resign"]
    result = _pass_rate(
        engine,
        f"long_resign[{label}]",
        _RESIGN_UTTERANCE,
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
    utterance = "grab the pawn on e6"
    _stays_a_model_eval(utterance, _LIVE_FEN)

    floor = _FLOORS["long_capture"]
    result = _pass_rate(
        engine,
        f"long_capture[{label}]",
        utterance,
        _expect_san("Bxe6"),
        floor=floor,
        setup=condition,
        runner=_run_panel,
    )

    # `blocking=True` for this one alone: TODO.md declares the `long_capture`
    # regression release-blocking, and an unresolved gate on a release blocker is
    # not a pass. Everywhere else UNDECIDED is reported and allowed.
    _assert_floor(result, floor, blocking=True)
