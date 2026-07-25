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
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Generator
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
from chessapp.personality import planner_prompt_for, system_prompt_for
from chessapp.provider import LlamaCppProvider
from chessapp.tools import BOARD_STATE_TOOLS, Settings, ToolContext, build_registry
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


class EvalApp(NamedTuple):
    """One assembled eval app: the wire, the state, and the model-call meter."""

    client: TestClient
    ctx: ToolContext
    provider: CountingProvider


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
    brain = create_llama_brain(
        base_url=LLAMACPP_BASE_URL,
        model=LLAMACPP_MODEL,
        dispatcher=registry,
        # Exactly what build_app offers the brain: the board state is injected
        # into its prompt every turn, so BOARD_STATE_TOOLS are dispatchable but
        # not *offered*. Offering them here would measure a different agent than
        # the one that ships — and a bigger tool list is itself a variable (the
        # 2026-07-13 trace review saw capture phrasings behave differently under
        # the two lists).
        tool_definitions=registry.definitions(exclude=BOARD_STATE_TOOLS),
        # Both prompts, exactly as build_app wires them: the narrator's carries
        # personality plus the live verbosity/hints layers, the planner's the
        # compact tool contract plus the hints permission. Measuring the tool
        # decision under the wrong prompt would measure a different agent.
        system_prompt_provider=lambda: system_prompt_for(
            ctx.settings.verbosity, ctx.settings.hints_mode
        ),
        planner_prompt_provider=lambda: planner_prompt_for(ctx.settings.hints_mode),
        planner_temperature=PLANNER_TEMPERATURE,
        provider=provider,
    )
    client = TestClient(
        create_app(ctx, brain=brain, registry=registry, coordinator=coordinator)
    )
    return EvalApp(client=client, ctx=ctx, provider=provider)


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


def _trajectory(assistant: dict[str, Any]) -> str:
    tokens: list[str] = []
    for call in _tool_calls(assistant):
        token = call["tool"]
        if call["error"] is not None:
            token += "!"
        elif (
            call["tool"] == "make_move"
            and call["result"]
            and json.loads(call["result"]).get("legal") is not True
        ):
            token += "(illegal)"
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
    """

    assistant: dict[str, Any]
    duration: float
    model_calls: list[ModelCall]


def _run(app: EvalApp, scenario: str, utterance: str) -> EvalRun:
    """One eval run against the live model: fresh conversation → one message →
    the `[eval]` stats line the baseline table is built from."""
    conversation_id = app.client.post("/api/agent/conversations", json={}).json()["id"]
    app.provider.reset()  # measure this utterance, not the fixture's history
    started = time.monotonic()
    response = app.client.post(
        f"/api/agent/conversations/{conversation_id}/messages",
        json={"content": utterance},
        timeout=_REQUEST_TIMEOUT,
    )
    duration = time.monotonic() - started
    assert response.status_code == 200, response.text
    assistant = response.json()["assistant_message"]
    model_calls = list(app.provider.calls)
    thinking = ",".join("on" if call.thinking else "off" for call in model_calls)
    print(
        f"\n[eval] scenario={scenario} stop={assistant['stop_reason']} "
        f"calls={len(_tool_calls(assistant))} model_calls={len(model_calls)} "
        f"thinking=[{thinking}] "
        f"mutations={len(_board_mutations(assistant))} duration={duration:.1f}s "
        f"trajectory=[{_trajectory(assistant)}]"
    )
    return EvalRun(assistant=assistant, duration=duration, model_calls=model_calls)


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

    run = _run(app, "fast_path_low", utterance)

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

    run = _run(app, "fast_path_normal", utterance)

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

    run = _run(app, "plain_move", utterance)
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

    run = _run(app, "judgment_question", utterance)
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

    run = _run(app, "ambiguous_move", utterance)
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

    run = _run(app, "settings_by_speech", utterance)
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

    run = _run(app, "honest_illegal", utterance)
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

    run = _run(app, "destructive_confirm", utterance)
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
# moved the number. The floors below are deliberately set at what the current
# build actually achieves (recorded in docs/agent-evals.md) — they are
# regression tripwires, not aspirations.

_PASS_RATE_RUNS = 5


class RateResult(NamedTuple):
    """What N runs of one scenario measured."""

    passed: int
    runs: int
    failures: list[str]

    @property
    def rate(self) -> float:
        return self.passed / self.runs


def _run_panel(app: EvalApp, scenario: str, utterance: str) -> EvalRun:
    """One turn through `POST /api/command` — the **web panel's** seam, not the
    delegate's.

    The difference is the transcript, and it is the whole point of the
    long-transcript scenarios. The delegate endpoint carries its own
    per-conversation history (`_run` above opens a fresh one every time, so the
    model sees an empty thread); `/api/command` reads `ctx.transcript.window()`,
    the running conversation the player has actually been having. Every failure
    in the 2026-07-13 trace review happened on *this* seam, deep into a thread —
    and none of them reproduce on a fresh delegate conversation.

    The panel returns a thinner document than the delegate wire (`tool_results`
    is `{name, result}` with no arguments and no separate error channel, and its
    `result` is a decoded dict rather than the wire's JSON string), so it is
    adapted into the same shape the trajectory helpers above read."""
    app.provider.reset()
    started = time.monotonic()
    response = app.client.post(
        "/api/command", json={"text": utterance}, timeout=_REQUEST_TIMEOUT
    )
    duration = time.monotonic() - started
    assert response.status_code == 200, response.text
    body = response.json()
    assistant = {
        "content": body["commentary"],
        # The panel has no error channel: a dispatch failure rides on `result`.
        "tool_calls": [
            {
                "tool": r["name"],
                "arguments": {},
                "result": json.dumps(r["result"]),
                "error": None,
            }
            for r in body["tool_results"]
        ],
        "stop_reason": "completed",  # not exposed on this seam
    }
    model_calls = list(app.provider.calls)
    print(
        f"\n[eval] scenario={scenario} calls={len(assistant['tool_calls'])} "
        f"model_calls={len(model_calls)} duration={duration:.1f}s "
        f"trajectory=[{_trajectory(assistant)}]"
    )
    return EvalRun(assistant=assistant, duration=duration, model_calls=model_calls)


def _pass_rate(
    engine: EnginePlayer,
    scenario: str,
    utterance: str,
    check: Callable[[EvalApp, dict[str, Any]], None],
    *,
    setup: Callable[[EvalApp], None] | None = None,
    runs: int = _PASS_RATE_RUNS,
    runner: Callable[[EvalApp, str, str], EvalRun] = _run,
) -> RateResult:
    """Run one scenario `runs` times on a fresh app each time, counting how
    often `check` holds. `check` raises AssertionError to fail a run; anything
    it raises is recorded, never propagated — one bad sample is data, not a
    test failure. The verdict is the rate, which the caller asserts on.

    `runner` picks the seam: `_run` (the delegate wire, fresh conversation each
    time) or `_run_panel` (the web panel, reading whatever `setup` left on
    `ctx.transcript`)."""
    passed = 0
    failures: list[str] = []
    for i in range(runs):
        app = _build_eval_app(engine)
        try:
            if setup is not None:
                setup(app)
            reset_rate_limit()  # each run is its own conversation
            run = runner(app, f"{scenario}[{i + 1}/{runs}]", utterance)
            check(app, run.assistant)
            passed += 1
        except AssertionError as exc:
            failures.append(f"run {i + 1}: {exc}")
        finally:
            app.client.close()
    result = RateResult(passed=passed, runs=runs, failures=failures)
    print(f"\n[eval] scenario={scenario} PASS_RATE={passed}/{runs} ({result.rate:.0%})")
    for failure in failures:
        print(f"[eval]   ✗ {failure}")
    return result


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

    result = _pass_rate(
        engine,
        "undo_and_replace",
        "take that bishop move back and play d4 instead",
        check,
        setup=setup,
    )

    assert result.rate >= 0.8, (
        f"undo+replace landed {result.passed}/{result.runs} times"
    )


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

    result = _pass_rate(
        engine,
        "my_mistake_is_mine",
        "what was my mistake?",
        check,
        setup=setup,
    )

    assert result.rate >= 0.8, (
        f"analyzed the player's move {result.passed}/{result.runs} times"
    )


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

    result = _pass_rate(
        engine,
        "play_as_black",
        "let's play chess as black",
        check,
    )

    assert result.rate >= 0.8, (
        f"the player got black {result.passed}/{result.runs} times"
    )


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
        app.ctx.session.save(tmp_path / "scholars.json")
        app.ctx.session = GameSession()  # a different game is in progress now

    def check(app: EvalApp, assistant: dict[str, Any]) -> None:
        assert _successful(assistant, "resume_game"), (
            "expected a resume_game call, got: " + (_trajectory(assistant) or "nothing")
        )
        assert _history(app.client) == ["e4", "e5", "Nf3", "Nc6"], (
            "the saved game did not come back"
        )

    result = _pass_rate(
        engine,
        "resume_not_denied",
        "load up the game I saved as scholars",
        check,
        setup=setup,
    )

    assert result.rate >= 0.8, f"resume_game ran {result.passed}/{result.runs} times"


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

    result = _pass_rate(
        engine,
        "resign_never_pretends",
        "you know what, I give up. I resign",
        check,
        setup=setup,
    )

    assert result.rate >= 0.8, (
        f"resign was actually called {result.passed}/{result.runs} times"
    )


@pytest.mark.xfail(
    strict=False,
    reason=(
        "KNOWN BROKEN (trace review 2026-07-13, secondary): with hints_mode "
        "off, 'what should I play here?' got concrete move advice (Nc3, exd6) "
        "invented by the model with no get_best_moves call — both a gating leak "
        "and advice that isn't the engine's. BRIEF: hints appear when on, never "
        "when off."
    ),
)
def test_eval_hints_off_means_no_move_advice(engine: EnginePlayer) -> None:
    """With hints off the agent must not hand over a move to play — not from the
    engine, and not from its own head. It should decline (or offer to turn hints
    on), and it must not have called get_best_moves to get there."""

    def setup(app: EvalApp) -> None:
        assert app.ctx.settings.hints_mode is False  # the default
        for san in ("e4", "e5", "Nf3", "Nc6"):
            assert app.ctx.session.submit_move(san).legal

    def check(app: EvalApp, assistant: dict[str, Any]) -> None:
        assert _successful(assistant, "get_best_moves") == [], (
            "hints are off — the engine must not be asked for a move to play"
        )
        assert _board_mutations(assistant) == []
        # The leak the trace caught: naming a concrete move anyway. Wording is
        # never asserted in this suite, but a SAN token *is* the payload here —
        # it is the hint itself, whatever prose surrounds it.
        legal_sans = set(app.ctx.session.legal_moves())
        named = {
            token.strip(".,!?`*")
            for token in assistant["content"].split()
            if token.strip(".,!?`*") in legal_sans
        }
        assert not named, f"handed over a move with hints off: {sorted(named)}"

    result = _pass_rate(
        engine,
        "hints_off_no_advice",
        "what should I play here?",
        check,
        setup=setup,
    )

    assert result.rate >= 0.8, f"hints stayed off {result.passed}/{result.runs} times"


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
        saved.save(tmp_path / "scholars.json")

    def check(app: EvalApp, assistant: dict[str, Any]) -> None:
        assert _successful(assistant, "resume_game"), (
            "denied a capability it has: " + assistant["content"]
        )
        assert _history(app.client) == ["e4", "e5", "Nf3", "Nc6"]

    result = _pass_rate(
        engine,
        f"long_resume[{label}]",
        "load up the game I saved as scholars",
        check,
        setup=setup,
        runner=_run_panel,
    )

    assert result.rate >= 0.8, (
        f"[{label}] resume_game ran {result.passed}/{result.runs} times"
    )


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

    result = _pass_rate(
        engine,
        f"long_resign[{label}]",
        "you know what, I give up. I resign",
        check,
        setup=condition,
        runner=_run_panel,
    )

    assert result.rate >= 0.8, (
        f"[{label}] resign was actually called {result.passed}/{result.runs} times"
    )


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

    result = _pass_rate(
        engine,
        f"long_capture[{label}]",
        "grab the pawn on e6",
        _expect_san("Bxe6"),
        setup=condition,
        runner=_run_panel,
    )

    assert result.rate >= 0.8, (
        f"[{label}] Bxe6 landed {result.passed}/{result.runs} times"
    )
