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
call above it), a tool-using utterance costs the tool turn plus the loop's
closing turn, and thinking stays OFF until an analysis tool's result lands in
context — then ON for the turn that reasons about it.

This suite is the tripwire the standard requires: baseline results are recorded
in `docs/agent-evals.md`, and it gates every future prompt/model/loop change —
run it before merging one; the baseline must not regress.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Generator
from typing import Any, NamedTuple

import pytest
from fastapi.testclient import TestClient

from chessapp.agent_api import reset_rate_limit
from chessapp.api import create_app
from chessapp.engine import DEFAULT_TIER, EnginePlayer
from chessapp.fastparse import parse_move
from chessapp.game import GameSession
from chessapp.llama_brain import _DEFAULT_MAX_ITERATIONS, create_llama_brain
from chessapp.personality import system_prompt_for
from chessapp.provider import LlamaCppProvider
from chessapp.tools import Settings, ToolContext, build_registry
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
# project-command-center via llama-swap, so a tight bound would flake. Recorded
# warm: analysis 1.9–5.4 s, everything else 0.5–0.9 s.
_ANALYSIS_CEILING_S = 15.0
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
    # One registry, exactly as build_app does: the brain's loop dispatches
    # through the same registry the app runs the fast path through.
    registry = build_registry(ctx)
    # The only departure from build_app: the real provider is wrapped so every
    # model round trip is counted and timed. create_llama_brain builds exactly
    # this provider when none is passed, so the wire itself is unchanged.
    provider = CountingProvider(LlamaCppProvider(LLAMACPP_BASE_URL, LLAMACPP_MODEL))
    brain = create_llama_brain(
        base_url=LLAMACPP_BASE_URL,
        model=LLAMACPP_MODEL,
        dispatcher=registry,
        tool_definitions=registry.definitions(),
        system_prompt_provider=lambda: system_prompt_for(
            ctx.settings.verbosity, ctx.settings.hints_mode
        ),
        provider=provider,
    )
    client = TestClient(create_app(ctx, brain=brain, registry=registry))
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
    utterance — the loop's turns, or `narrate`'s single turn on the fast path,
    or none at all when the fast path answers with a canned confirmation.
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
    """The loop stayed inside its bound. Scenarios whose trajectory is fixed
    pin an exact count instead; this is for the ones the model may legitimately
    answer in one turn (no tool) or three (a read, then an act) — the shape
    that matters there is only that it terminated on its own, not on the
    budget."""
    assert 1 <= len(run.model_calls) <= _DEFAULT_MAX_ITERATIONS, (
        f"expected 1..{_DEFAULT_MAX_ITERATIONS} model calls, got {len(run.model_calls)}"
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
    """Above verbosity=low the fast path still skips the *loop* — it dispatches
    the move deterministically and pays for commentary only: exactly one model
    call (`Brain.narrate`, the loop's closing turn on its own), thinking off,
    and no tool-decision turn."""
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

    Also the loop's minimum cost: the tool turn plus the closing turn that
    comments on its result — two model round trips, both thinking-off."""
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
    assert len(run.model_calls) == 2, "the loop's minimum: tool turn + closing turn"
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
    test_llama_brain only pins it against a scripted provider."""
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
    # so the closing turn — the one that actually judges the position — thinks.
    assert len(run.model_calls) >= 2, "expected a tool turn and a closing turn"
    _assert_thinking_starts_off(run)
    assert run.model_calls[-1].thinking is True, (
        "the turn commenting on an analysis result must run with thinking ON"
    )
    assert run.duration < _ANALYSIS_CEILING_S


def test_eval_ambiguous_move_asks_instead_of_guessing(eval_app: EvalApp) -> None:
    """ "move the rook" in a position with several mobile rooks is genuinely
    ambiguous — the agent must ask, not guess a move. (1. a4 a5 2. h4 h5 opens
    both White rook files; White to move, four rook moves available.)

    The cheapest loop run there is: one turn, no tools, straight to the
    clarifying question."""
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
    _assert_loop_budget(run)
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
    assert len(run.model_calls) == 2, "the loop's minimum: tool turn + closing turn"
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


@pytest.mark.xfail(
    reason=(
        "known gap: gemma-4-12b at temp 1.0 honors the new_game confirmation "
        "rule only ~half the time (measured ~50% across a 2-ply stub and a "
        "10-ply developed game — position depth didn't move the rate), so it "
        "often just complies and resets. The prompt carries the rule and "
        "test_personality pins that; the model doesn't reliably follow it. "
        "Real prompt-adherence gap, not scenario flakiness — see "
        "docs/agent-evals.md and TODO.md. Left as a non-strict xfail: the "
        "invariant is correct and it XPASSes when the model behaves."
    ),
    strict=False,
)
def test_eval_destructive_op_asks_before_acting(eval_app: EvalApp) -> None:
    """ "new game" mid-game is destructive: the prompt requires a confirmation
    question first, so no new_game should fire this turn and the game should be
    intact.

    Set up a substantial game (a 10-ply Ruy Lopez, castled, developed) so this
    isn't dismissible as a two-move stub the model reasonably resets — the
    probes showed the confirmation rate is ~50% regardless of how much game is
    on the board, which is why this scenario is xfail (see the marker)."""
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
