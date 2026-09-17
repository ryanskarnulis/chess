"""The engine dies after the player's move has already committed (#284).

Half a turn is not a failed request. By the time Stockfish is asked for its
reply the player's move has been through the legality gate, is on the board and
has been broadcast to every client — so an exception from there may not reach
the player as a bare 500 that says nothing about the move it already played.
The coordinator's half of the recovery is `test_coordinator.py`'s (the phase
goes back to `player_move_applied`: the move stands, the reply is still owed);
this file owns the pipeline above it. What it pins: a 200 carrying the
committed results and the app's own line about the reply that never came, a
board holding exactly the one move, a turn the next command settles, and the
same answers from every surface that can play a move — the spoken fast path,
the brain's loop, a drag in agent mode, a drag with no brain at all, and the
delegate endpoint.

The engine is the `DyingEngine` double and the brain is scripted: nothing here
touches a live model or a real Stockfish. The app's line is imported rather
than typed out, so these tests pin the *substitution*, never a wording.
"""

import pytest
from fastapi.testclient import TestClient

from chessapp.agent_api import reset_rate_limit
from chessapp.api import ENGINE_LOST_REPLY_OWED, create_app
from chessapp.brain import AgentResponse, ToolCall
from chessapp.game import GameSession
from chessapp.tools import ToolContext
from fakes import DyingEngine, FakeEngine, ScriptedBrain, scripted_app


@pytest.fixture(autouse=True)
def _reset_rate_limit():
    """The delegate limiter is module-global; one test's posts must not count
    against another's."""
    reset_rate_limit()
    yield
    reset_rate_limit()


def make_client(*responses: AgentResponse, narrations=(), verbosity="normal"):
    """An agent-mode app whose engine is already dead."""
    ctx = ToolContext(session=GameSession(), engine=DyingEngine())
    ctx.settings.verbosity = verbosity
    brain = ScriptedBrain(*responses, narrations=narrations)
    app, _ = scripted_app(ctx, brain=brain)
    return TestClient(app), ctx


def move(san: str, text: str = "on it") -> AgentResponse:
    return AgentResponse(
        text=text, tool_calls=(ToolCall(name="make_move", args={"move": san}),)
    )


# --- the spoken routes ------------------------------------------------------


def test_the_fast_path_answers_with_the_turn_rather_than_a_500():
    client, ctx = make_client()

    response = client.post("/api/command", json={"text": "e4"})

    assert response.status_code == 200, "the move landed; this is not a failure"
    body = response.json()
    assert ctx.session.move_history() == ["e4"], "the move stands, unanswered"
    assert body["state"]["history"] == ["e4"]
    assert ENGINE_LOST_REPLY_OWED in body["commentary"]
    played = [r for r in body["tool_results"] if r["name"] == "make_move"]
    assert played and played[0]["result"]["legal"] is True, "the results came back"


def test_the_brains_route_answers_the_same_way():
    """The convergence collect, not the fast path's: the loop played the move
    through the registry and the pipeline closes the turn afterwards."""
    client, ctx = make_client(move("e4"))

    body = client.post("/api/command", json={"text": "push the king pawn"}).json()

    assert ctx.session.move_history() == ["e4"]
    assert ENGINE_LOST_REPLY_OWED in body["commentary"]


def test_the_apps_line_is_composed_around_glitchs_own_words():
    """Composed after the reaction, where the reply announcement would have
    been — the app reporting what answered the move, and here what did not.
    Glitch is never handed the line to say."""
    client, _ = make_client(narrations=("Classic opener.",))

    body = client.post("/api/command", json={"text": "e4"}).json()

    assert body["commentary"] == f"Classic opener.\n\n{ENGINE_LOST_REPLY_OWED}"


def test_a_silent_turn_still_says_the_move_stands():
    """verbosity=low never narrates, so the deterministic move confirmation is
    all there is to compose around — and the player still hears why no answer
    came."""
    client, _ = make_client(verbosity="low")

    body = client.post("/api/command", json={"text": "e4"}).json()

    assert body["commentary"] == f"e4.\n\n{ENGINE_LOST_REPLY_OWED}"


def test_the_turn_is_remembered_by_what_was_said_not_by_the_apps_line():
    """The transcript's half of the honesty rule: an app line recorded as
    Glitch's own words is a register he imitates, and this one would have him
    reporting engine failures that never happened."""
    client, ctx = make_client(narrations=("Classic opener.",))

    client.post("/api/command", json={"text": "e4"})

    assistant = ctx.transcript.window()[-1]
    assert assistant["content"] == "Classic opener."
    assert ENGINE_LOST_REPLY_OWED not in assistant["content"]


def test_a_silent_turn_remembers_the_facts():
    client, ctx = make_client(verbosity="low")

    client.post("/api/command", json={"text": "e4"})

    assert ctx.transcript.window()[-1]["content"] == "e4."


def test_the_board_holds_only_the_committed_move():
    client, _ = make_client()

    client.post("/api/command", json={"text": "e4"})

    state = client.get("/api/state").json()
    assert state["history"] == ["e4"], "one move, and no phantom reply"
    assert state["turn"] == "black", "the reply is still owed"


# --- recovery ---------------------------------------------------------------


def test_the_next_command_settles_the_owed_reply_and_the_game_goes_on():
    """The healing branch the coordinator's restored phase exists for: the new
    move is refused as mid-turn, the owed reply is played, and the turn after
    that is ordinary. One utterance is the whole cost of a dead engine."""
    client, ctx = make_client(move("d4"))
    client.post("/api/command", json={"text": "e4"})

    ctx.engine = FakeEngine("e7e5")  # Stockfish is back
    refused = client.post("/api/command", json={"text": "play d4"}).json()

    (attempt,) = [r for r in refused["tool_results"] if r["name"] == "make_move"]
    assert attempt["result"]["ok"] is False, "a move mid-turn is still refused"
    assert "player_move_applied" in attempt["result"]["error"]
    assert ctx.session.move_history() == ["e4", "e5"], "exactly one owed reply"

    ctx.engine.reply_uci = "d7d6"
    played = client.post("/api/command", json={"text": "d4"}).json()

    assert played["state"]["history"] == ["e4", "e5", "d4", "d6"]
    assert ENGINE_LOST_REPLY_OWED not in played["commentary"]


# --- a dragged move ---------------------------------------------------------


def test_a_drag_in_agent_mode_stands_with_the_apps_line():
    client, ctx = make_client()

    response = client.post("/api/game/move", json={"move": "e2e4"})

    assert response.status_code == 200
    body = response.json()
    assert body["legal"] is True and body["san"] == "e4"
    assert body["engine_move"] is None, "there is no reply to report"
    assert ENGINE_LOST_REPLY_OWED in body["commentary"]
    assert ctx.session.move_history() == ["e4"]
    assert body["state"]["history"] == ["e4"]


def direct_client(engine=None):
    """Direct mode: no brain at all, the LLM-off invariant's own path."""
    ctx = ToolContext(session=GameSession(), engine=engine or DyingEngine())
    return TestClient(create_app(ctx)), ctx


def test_a_drag_with_no_brain_stands_too():
    """Direct mode shares the collect (the coordinator's atomic exchange), so
    it shares the answer: the move is on the board, so the request did not
    fail. The one key it gains is the app's line — a board that moved once with
    nothing said about the answer that never came is a board the player cannot
    follow."""
    client, ctx = direct_client()

    response = client.post("/api/game/move", json={"move": "e2e4"})

    assert response.status_code == 200
    body = response.json()
    assert body["legal"] is True and body["san"] == "e4"
    assert body["engine_move"] is None
    assert body["commentary"] == ENGINE_LOST_REPLY_OWED
    assert ctx.session.move_history() == ["e4"]


def test_a_healthy_drag_with_no_brain_gains_no_new_key():
    """The line is the dead engine's, not direct mode's: an ordinary drag still
    answers exactly what it always answered."""
    client, _ = direct_client(engine=FakeEngine("e7e5"))

    body = client.post("/api/game/move", json={"move": "e2e4"}).json()

    assert body["engine_move"]["san"] == "e5"
    assert "commentary" not in body


def test_the_next_drag_with_no_brain_settles_the_owed_reply_too():
    """The same recovery as the spoken road, spelled out for the path that has
    no close beat to run it: refused move, owed reply played, game on."""
    client, ctx = direct_client()
    client.post("/api/game/move", json={"move": "e2e4"})

    ctx.engine = FakeEngine("e7e5")
    refused = client.post("/api/game/move", json={"move": "d2d4"})

    assert refused.status_code == 409, "a move mid-turn is refused, as it always was"
    assert ctx.session.move_history() == ["e4", "e5"], "exactly one owed reply"

    ctx.engine.reply_uci = "d7d6"
    played = client.post("/api/game/move", json={"move": "d2d4"}).json()

    assert played["state"]["history"] == ["e4", "e5", "d4", "d6"]


# --- the delegate endpoint --------------------------------------------------


def test_the_delegate_endpoint_answers_with_the_turn():
    """It catches `ProviderError` and nothing else, which is all it needs to:
    an engine failure is no longer an exception by the time it gets here, so
    the conductor is told what happened instead of being handed a 500 and
    deciding for itself whether the move landed."""
    client, ctx = make_client(move("e4"))
    conversation = client.post("/api/agent/conversations", json={}).json()["id"]

    response = client.post(
        f"/api/agent/conversations/{conversation}/messages",
        json={"content": "push the king pawn"},
    )

    assert response.status_code == 200
    assistant = response.json()["assistant_message"]
    assert ENGINE_LOST_REPLY_OWED in assistant["content"]
    assert assistant["stop_reason"] == "completed", "the loop itself finished"
    assert ctx.session.move_history() == ["e4"]
