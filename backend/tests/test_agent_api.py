"""Delegate REST API: conversation persistence + the messages round trip.

The workspace delegate contract (`../agent-standard/delegate-api.md`) so a
conductor agent can drive chess over HTTP. Mirrors PCC's `test_agent_api.py`
against chess's fixtures — the `ScriptedBrain` double (never a live LLM) and
the shared command pipeline extracted from `/api/command`. The conversation
store is in-memory (chess's documented divergence from PCC's SQLite), so a
fresh `create_app` gives each test a fresh store; the per-IP rate limiter is
module-global, so it is reset around every test.
"""

import logging

import pytest
from fastapi.testclient import TestClient

from chessapp.agent_api import LOOP_ACTOR, reset_rate_limit
from chessapp.api import MAX_TOOL_RETRY_ROUNDS, create_app
from chessapp.brain import AgentResponse, ToolCall
from chessapp.game import GameSession
from chessapp.provider import ProviderRequestError
from chessapp.tools import ToolContext
from fakes import ScriptedBrain


@pytest.fixture(autouse=True)
def _reset_rate_limit():
    reset_rate_limit()
    yield
    reset_rate_limit()


def make_client(*responses, reactions=(), brain=None, verbosity="normal"):
    brain = (
        brain if brain is not None else ScriptedBrain(*responses, reactions=reactions)
    )
    ctx = ToolContext(session=GameSession())
    ctx.settings.verbosity = verbosity
    return TestClient(create_app(ctx, brain=brain)), brain, ctx


def move(san):
    return AgentResponse(
        text="on it", tool_calls=(ToolCall(name="make_move", args={"move": san}),)
    )


def new_conversation(client, **body):
    return client.post("/api/agent/conversations", json=body).json()["id"]


def send(client, conversation_id, content, **kwargs):
    return client.post(
        f"/api/agent/conversations/{conversation_id}/messages",
        json={"content": content},
        **kwargs,
    )


# --- CRUD ---------------------------------------------------------------------


def test_create_conversation_is_201_with_null_title():
    client, _, _ = make_client()
    created = client.post("/api/agent/conversations", json={})
    assert created.status_code == 201
    body = created.json()
    assert body["title"] is None
    assert isinstance(body["id"], int)
    assert "created_at" in body and "updated_at" in body


def test_create_conversation_keeps_a_given_title():
    client, _, _ = make_client()
    body = client.post(
        "/api/agent/conversations", json={"title": "Weekly triage"}
    ).json()
    assert body["title"] == "Weekly triage"


def test_list_conversations_most_recent_first():
    client, _, _ = make_client()
    first = new_conversation(client)
    second = new_conversation(client)
    listed = client.get("/api/agent/conversations").json()
    assert [c["id"] for c in listed] == [second, first]


def test_detail_of_a_fresh_conversation_has_no_messages():
    client, _, _ = make_client()
    conversation_id = new_conversation(client, title="Endgame drill")
    detail = client.get(f"/api/agent/conversations/{conversation_id}")
    assert detail.status_code == 200
    body = detail.json()
    assert body["title"] == "Endgame drill"
    assert body["messages"] == []


def test_missing_and_soft_deleted_threads_404_across_endpoints():
    """A never-existed id and a soft-deleted thread must 404 identically on
    GET, POST-messages, and DELETE — the rule conductor's recreate-once retry
    depends on (a pruned thread is indistinguishable from one that never was)."""
    client, _, _ = make_client()

    missing = 999_999
    assert client.get(f"/api/agent/conversations/{missing}").status_code == 404
    assert send(client, missing, "hi").status_code == 404
    assert client.delete(f"/api/agent/conversations/{missing}").status_code == 404

    conversation_id = new_conversation(client)
    assert (
        client.delete(f"/api/agent/conversations/{conversation_id}").status_code == 204
    )
    assert client.get(f"/api/agent/conversations/{conversation_id}").status_code == 404
    assert send(client, conversation_id, "hi").status_code == 404
    assert (
        client.delete(f"/api/agent/conversations/{conversation_id}").status_code == 404
    )


# --- message flow -------------------------------------------------------------


def test_post_message_runs_pipeline_and_persists_the_exchange():
    client, _, ctx = make_client(move("e4"), reactions=("Pawn to e4. Your move.",))
    conversation_id = new_conversation(client)

    response = send(client, conversation_id, "open with the king's pawn")
    assert response.status_code == 200
    exchange = response.json()

    user = exchange["user_message"]
    assert user["role"] == "user"
    assert user["content"] == "open with the king's pawn"
    assert user["tool_calls"] is None
    assert user["stop_reason"] is None

    assistant = exchange["assistant_message"]
    assert assistant["role"] == "assistant"
    assert assistant["content"] == "Pawn to e4. Your move."
    assert assistant["stop_reason"] == "completed"
    calls = assistant["tool_calls"]
    assert [c["tool"] for c in calls] == ["make_move"]
    assert calls[0]["arguments"] == {"move": "e4"}
    assert calls[0]["result"] is not None and calls[0]["error"] is None

    # The move landed on the single shared game session.
    assert ctx.session.move_history() == ["e4"]

    # History survives a "reload" as an ordered thread.
    detail = client.get(f"/api/agent/conversations/{conversation_id}").json()
    assert detail["title"] == "open with the king's pawn"
    assert [m["role"] for m in detail["messages"]] == ["user", "assistant"]


def test_tool_calls_is_null_when_no_tools_ran():
    client, _, _ = make_client(AgentResponse(text="Looks balanced to me."))
    conversation_id = new_conversation(client)
    assistant = send(client, conversation_id, "how does the board look?").json()[
        "assistant_message"
    ]
    assert assistant["content"] == "Looks balanced to me."
    assert assistant["tool_calls"] is None
    assert assistant["stop_reason"] == "completed"


def test_ok_false_result_maps_to_error_not_result():
    client, _, _ = make_client(
        AgentResponse(
            text="odd", tool_calls=(ToolCall(name="launch_rocket", args={}),)
        ),
        AgentResponse(text="I can't do that."),
    )
    conversation_id = new_conversation(client)
    assistant = send(client, conversation_id, "do a barrel roll").json()[
        "assistant_message"
    ]
    call = assistant["tool_calls"][0]
    assert call["tool"] == "launch_rocket"
    assert call["result"] is None
    assert "unknown tool" in call["error"]
    # A wordy concession during the retry loop is a normal completion.
    assert assistant["content"] == "I can't do that."
    assert assistant["stop_reason"] == "completed"


def test_rejected_move_is_a_result_and_exhaustion_is_correction_limit():
    """A `legal: false` move rejection is a legitimate domain outcome — it
    rides on `result`, never `error`. A retry loop that never resolves the
    failures stops with `correction_limit`."""
    client, _, ctx = make_client(
        move("Nf6"),  # illegal for White from the start
        *[move("Nf6") for _ in range(MAX_TOOL_RETRY_ROUNDS)],
        reactions=("Couldn't make that one work.",),
    )
    conversation_id = new_conversation(client)
    assistant = send(client, conversation_id, "knight to f6").json()[
        "assistant_message"
    ]

    calls = assistant["tool_calls"]
    assert len(calls) == 1 + MAX_TOOL_RETRY_ROUNDS
    assert all(c["result"] is not None and c["error"] is None for c in calls)
    assert assistant["content"] == "Couldn't make that one work."
    assert assistant["stop_reason"] == "correction_limit"
    assert ctx.session.move_history() == []


def test_title_is_derived_from_the_first_user_message():
    client, _, _ = make_client(AgentResponse(text="ok"))
    conversation_id = new_conversation(client)
    assert (
        client.get(f"/api/agent/conversations/{conversation_id}").json()["title"]
        is None
    )
    send(client, conversation_id, "what's my best plan here")
    assert (
        client.get(f"/api/agent/conversations/{conversation_id}").json()["title"]
        == "what's my best plan here"
    )


def test_history_replays_text_turns_only_no_tool_payloads():
    """The second message's brain call sees only prior *text* turns — the
    reaction the caller saw, never the persisted tool trajectory."""
    brain = ScriptedBrain(
        move("e4"),  # turn 1, phase one
        AgentResponse(text="I opened with e4."),  # turn 2, phase one
        reactions=("Pawn to e4.",),  # turn 1, react
    )
    client, _, _ = make_client(brain=brain)
    conversation_id = new_conversation(client)

    send(client, conversation_id, "start with the king's pawn")
    send(client, conversation_id, "what did you play?")

    replayed = brain.transcripts[-1]
    assert replayed == [
        {"role": "user", "content": "start with the king's pawn"},
        {"role": "assistant", "content": "Pawn to e4."},
    ]
    # Text turns only: no tool payloads smuggled into model context.
    assert all(set(turn) == {"role", "content"} for turn in replayed)


def test_fast_path_move_skips_the_phase_one_brain_call():
    # A ScriptedBrain with no scripted phase-one responses raises if consulted.
    client, brain, ctx = make_client(reactions=("Classic opener.",))
    conversation_id = new_conversation(client)
    assistant = send(client, conversation_id, "e4").json()["assistant_message"]

    assert brain.calls == []  # phase one never consulted
    assert assistant["content"] == "Classic opener."
    assert assistant["tool_calls"][0]["tool"] == "make_move"
    assert ctx.session.move_history() == ["e4"]


def test_delegate_move_broadcasts_to_the_web_board():
    client, _, _ = make_client(reactions=("ok",))
    conversation_id = new_conversation(client)
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()  # connect snapshot
        send(client, conversation_id, "e4")
        message = ws.receive_json()
    assert message["state"]["history"] == ["e4"]


def test_provider_failure_is_502_and_keeps_the_user_message():
    class BoomBrain:
        def get_agent_response(self, board_state, command, transcript=()):
            raise ProviderRequestError("connect timeout")

        def react(self, board_state, changes, transcript=()):  # pragma: no cover
            raise AssertionError("react should not run")

    client, _, _ = make_client(brain=BoomBrain())
    conversation_id = new_conversation(client)

    response = send(client, conversation_id, "how am I doing?")
    assert response.status_code == 502

    detail = client.get(f"/api/agent/conversations/{conversation_id}").json()
    assert [m["role"] for m in detail["messages"]] == ["user"]
    assert detail["messages"][0]["content"] == "how am I doing?"


@pytest.mark.parametrize(
    "content", ["", "   ", "x" * 8001], ids=["empty", "whitespace", "too-long"]
)
def test_message_content_validation_is_422(content):
    client, _, _ = make_client()
    conversation_id = new_conversation(client)
    assert send(client, conversation_id, content).status_code == 422


def test_missing_content_field_is_422():
    client, _, _ = make_client()
    conversation_id = new_conversation(client)
    assert (
        client.post(
            f"/api/agent/conversations/{conversation_id}/messages", json={}
        ).status_code
        == 422
    )


def test_post_message_is_rate_limited(monkeypatch):
    monkeypatch.setenv("CHESSAPP_AGENT_MESSAGES_PER_MIN", "1")
    client, _, _ = make_client(reactions=("ok", "ok"))
    conversation_id = new_conversation(client)

    assert send(client, conversation_id, "e4").status_code == 200
    second = send(client, conversation_id, "e5")
    assert second.status_code == 429
    assert "Retry-After" in second.headers


def test_recognized_actor_binds_the_conductor(caplog):
    client, _, _ = make_client(reactions=("ok",))
    conversation_id = new_conversation(client)
    with caplog.at_level(logging.INFO, logger="chessapp.agent_api"):
        send(
            client,
            conversation_id,
            "e4",
            headers={"X-Agent-Actor": "agent:conductor"},
        )
    assert any("actor=agent:conductor" in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize(
    "headers",
    [{}, {"X-Agent-Actor": "agent:bogus"}],
    ids=["absent", "unrecognized"],
)
def test_actor_falls_back_to_the_default_when_absent_or_unrecognized(caplog, headers):
    client, _, _ = make_client(reactions=("ok",))
    conversation_id = new_conversation(client)
    with caplog.at_level(logging.INFO, logger="chessapp.agent_api"):
        send(client, conversation_id, "e4", headers=headers)
    assert any(f"actor={LOOP_ACTOR}" in r.getMessage() for r in caplog.records)
