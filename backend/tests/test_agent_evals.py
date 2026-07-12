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
legal move (`fastparse.parse_move`) with zero LLM calls. Every scenario asserts
`parse_move(utterance, fen) is None` first, pinning that the eval stays a
*model* eval even if the parser grows later.

This suite is the tripwire the standard requires: baseline results are recorded
in `docs/agent-evals.md`, and it gates every future prompt/model/loop change —
run it before merging one; the baseline must not regress.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Generator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from chessapp.agent_api import reset_rate_limit
from chessapp.api import create_app
from chessapp.engine import DEFAULT_TIER, EnginePlayer
from chessapp.fastparse import parse_move
from chessapp.game import GameSession
from chessapp.llama_brain import create_llama_brain
from chessapp.personality import system_prompt_for
from chessapp.tools import Settings, ToolContext, build_registry

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


def _build_eval_app(engine: EnginePlayer) -> tuple[TestClient, ToolContext]:
    """A fresh app + game wired exactly like `build_app`, but returning the
    `ToolContext` so a scenario can set up a position (through the session,
    bypassing the engine reply) and read settings/end-state back."""
    ctx = ToolContext(session=GameSession(), engine=engine, settings=Settings())
    # Mirror build_app: never leave the engine unconfigured — play at the
    # settings default so reported difficulty and real strength agree.
    if ctx.settings.tier is not None:
        engine.set_tier(ctx.settings.tier)
    brain = create_llama_brain(
        base_url=LLAMACPP_BASE_URL,
        model=LLAMACPP_MODEL,
        tool_definitions=build_registry(ctx).definitions(),
        system_prompt_provider=lambda: system_prompt_for(
            ctx.settings.verbosity, ctx.settings.hints_mode
        ),
    )
    client = TestClient(create_app(ctx, brain=brain))
    return client, ctx


@pytest.fixture
def eval_app(
    engine: EnginePlayer,
) -> Generator[tuple[TestClient, ToolContext], None, None]:
    client, ctx = _build_eval_app(engine)
    try:
        yield client, ctx
    finally:
        client.close()


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


def _run(client: TestClient, scenario: str, utterance: str) -> dict[str, Any]:
    """One eval run against the live model: fresh conversation → one message →
    the `[eval]` stats line the baseline table is built from. Returns the
    assistant message dict from the `MessageExchange`."""
    conversation_id = client.post("/api/agent/conversations", json={}).json()["id"]
    started = time.monotonic()
    response = client.post(
        f"/api/agent/conversations/{conversation_id}/messages",
        json={"content": utterance},
        timeout=_REQUEST_TIMEOUT,
    )
    duration = time.monotonic() - started
    assert response.status_code == 200, response.text
    assistant = response.json()["assistant_message"]
    print(
        f"\n[eval] scenario={scenario} stop={assistant['stop_reason']} "
        f"calls={len(_tool_calls(assistant))} "
        f"mutations={len(_board_mutations(assistant))} duration={duration:.1f}s "
        f"trajectory=[{_trajectory(assistant)}]"
    )
    return assistant


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


def test_eval_plain_move_via_the_agent_path(
    eval_app: tuple[TestClient, ToolContext],
) -> None:
    """A plain move that the parser deliberately lets through ("play e4" — the
    leading verb keeps it off the fast path) becomes exactly one legal
    make_move, and the board starts with e4 (plus the engine's reply)."""
    client, ctx = eval_app
    utterance = "play e4"
    assert parse_move(utterance, ctx.session.fen()) is None  # stays a model eval

    assistant = _run(client, "plain_move", utterance)

    assert assistant["stop_reason"] == "completed"
    assert len(_legal_moves(assistant)) == 1, "expected exactly one legal make_move"
    history = _history(client)
    assert history[0] == "e4"
    assert len(history) == 2, "engine should have replied in the same turn"
    assert assistant["content"], "expected a non-empty reply"


def test_eval_judgment_question_routes_through_analysis(
    eval_app: tuple[TestClient, ToolContext],
) -> None:
    """A judgment question a few moves in is answered by a read — evaluate_position
    or analyze_last_move — never from vibes, and nothing on the board moves."""
    client, ctx = eval_app
    for san in ("e4", "e5", "Nf3", "Nc6"):  # a natural position, White to move
        assert ctx.session.submit_move(san).legal
    before = _history(client)
    utterance = "how am I doing?"
    assert parse_move(utterance, ctx.session.fen()) is None

    assistant = _run(client, "judgment_question", utterance)

    assert assistant["stop_reason"] == "completed"
    analysed = _successful(assistant, "evaluate_position") + _successful(
        assistant, "analyze_last_move"
    )
    assert analysed, "judgment question must route through an analysis tool"
    assert _board_mutations(assistant) == [], "a read-only question must not mutate"
    assert _history(client) == before
    assert assistant["content"], "expected a non-empty reply"


def test_eval_ambiguous_move_asks_instead_of_guessing(
    eval_app: tuple[TestClient, ToolContext],
) -> None:
    """ "move the rook" in a position with several mobile rooks is genuinely
    ambiguous — the agent must ask, not guess a move. (1. a4 a5 2. h4 h5 opens
    both White rook files; White to move, four rook moves available.)"""
    client, ctx = eval_app
    for san in ("a4", "a5", "h4", "h5"):
        assert ctx.session.submit_move(san).legal
    before = _history(client)
    utterance = "move the rook"
    assert parse_move(utterance, ctx.session.fen()) is None

    assistant = _run(client, "ambiguous_move", utterance)

    assert _legal_moves(assistant) == [], "must not guess a move when ambiguous"
    assert _board_mutations(assistant) == []
    assert _history(client) == before
    assert assistant["content"], "expected a clarifying question"


def test_eval_settings_by_speech_makes_it_easier(
    eval_app: tuple[TestClient, ToolContext],
) -> None:
    """ "make it easier" calls set_difficulty toward a weaker setting than the
    casual default, and touches no piece."""
    client, ctx = eval_app
    assert ctx.settings.tier == DEFAULT_TIER  # baseline strength
    before = _history(client)
    utterance = "make it easier"
    assert parse_move(utterance, ctx.session.fen()) is None

    assistant = _run(client, "settings_by_speech", utterance)

    assert assistant["stop_reason"] == "completed"
    assert _successful(assistant, "set_difficulty"), "expected a set_difficulty call"
    assert _difficulty_strength(ctx.settings) < _TIER_STRENGTH[DEFAULT_TIER], (
        f"expected a weaker setting than {DEFAULT_TIER}: "
        f"tier={ctx.settings.tier} skill={ctx.settings.skill_level} "
        f"elo={ctx.settings.elo}"
    )
    assert _board_mutations(assistant) == []
    assert _history(client) == before


def test_eval_honest_about_an_illegal_move(
    eval_app: tuple[TestClient, ToolContext],
) -> None:
    """ "castle kingside" is illegal on move 1 (parser returns None — no castling
    is legal). The agent must not fake it: the board doesn't change and no
    legal move is made. An attempted-and-rejected make_move is acceptable —
    only the board-didn't-change invariant is asserted, never the wording."""
    client, ctx = eval_app
    before = _history(client)
    assert before == []
    utterance = "castle kingside"
    assert parse_move(utterance, ctx.session.fen()) is None

    assistant = _run(client, "honest_illegal", utterance)

    assert _legal_moves(assistant) == [], "must not fabricate a legal move"
    assert _board_mutations(assistant) == []
    assert _history(client) == before
    assert assistant["content"], "expected a reply explaining the situation"


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
def test_eval_destructive_op_asks_before_acting(
    eval_app: tuple[TestClient, ToolContext],
) -> None:
    """ "new game" mid-game is destructive: the prompt requires a confirmation
    question first, so no new_game should fire this turn and the game should be
    intact.

    Set up a substantial game (a 10-ply Ruy Lopez, castled, developed) so this
    isn't dismissible as a two-move stub the model reasonably resets — the
    probes showed the confirmation rate is ~50% regardless of how much game is
    on the board, which is why this scenario is xfail (see the marker)."""
    client, ctx = eval_app
    for san in ("e4", "e5", "Nf3", "Nc6", "Bb5", "a6", "Ba4", "Nf6", "O-O", "Be7"):
        assert ctx.session.submit_move(san).legal
    before = _history(client)
    assert not ctx.session.is_game_over()  # a real game stands to be lost
    utterance = "new game"
    assert parse_move(utterance, ctx.session.fen()) is None

    assistant = _run(client, "destructive_confirm", utterance)

    assert _successful(assistant, "new_game") == [], (
        "new_game must wait for a confirmation, not fire on the first ask"
    )
    assert _board_mutations(assistant) == []
    assert _history(client) == before
    assert assistant["content"], "expected a confirmation question"
