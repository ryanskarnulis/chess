"""The panel's latest reply and the settings reach every client (#458).

The commentary bubble used to learn Glitch's words only from the response to
the request that produced them, so a reload showed "Your move." and a second
tab never heard him at all; settings had no broadcast either, so a second
tab's difficulty select kept the value it loaded with. What this file pins, at
the API boundary: the reply rides the socket's connect snapshot and its own
broadcast, as a frame of its own (the board document stays the board's); it is
the panel's alone, and belongs to one game; and a settings change is broadcast
once, whoever made it.
"""

from collections.abc import Iterator
from contextlib import contextmanager

from fastapi.testclient import TestClient

from chessapp.brain import AgentResponse, ToolCall
from chessapp.game import GameSession
from chessapp.tools import ToolContext
from fakes import FakeEngine, ScriptedBrain, receive_state, scripted_app


@contextmanager
def make_client(
    *responses: AgentResponse, narrations: tuple = ()
) -> Iterator[tuple[TestClient, ToolContext]]:
    """A client **entered**, so every request and socket shares one event loop.
    These tests open more than one socket per app, and an unentered client gives
    each socket session a loop of its own — the broadcaster stays bound to the
    first, and a later socket hears nothing (see `test_progress_stream`)."""
    ctx = ToolContext(session=GameSession(), engine=FakeEngine("e7e5"))
    brain = ScriptedBrain(*responses, narrations=narrations)
    app, _ = scripted_app(ctx, brain=brain)
    with TestClient(app) as client:
        yield client, ctx


def receive(ws, kind: str) -> dict:
    """The next message of one `type`, skipping the others (progress frames,
    state frames) that share the channel."""
    while True:
        message = ws.receive_json()
        if message["type"] == kind:
            return message


def restored_reply(client: TestClient) -> dict | None:
    """What a tab that loads now is told Glitch last said: the reply frame
    behind its connect snapshot, or None when none came. Absence is proven by
    what arrives instead — a settings frame the test provokes by flipping the
    voice setting, sent after any snapshot frame."""
    voice = client.get("/api/settings").json()["voice_output"]
    with client.websocket_connect("/ws") as ws:
        receive_state(ws)
        client.post("/api/settings/voice", json={"enabled": not voice})
        message = ws.receive_json()
    return message["reply"] if message["type"] == "reply" else None


def test_a_fresh_board_has_no_reply():
    with make_client() as (client, _):
        assert restored_reply(client) is None


def test_a_command_reply_is_restored_to_a_tab_that_loads_later():
    with make_client(AgentResponse(text="The Sicilian fights for d4.")) as (
        client,
        ctx,
    ):
        client.post("/api/command", json={"text": "what's the Sicilian about?"})
        assert restored_reply(client) == {
            "text": "The Sicilian fights for d4.",
            "game_id": ctx.session.game_id,
            "seq": 1,
        }


def test_the_board_document_stays_the_boards():
    """The reply is a frame of its own, never a key on the state document: a
    mutation's response *is* that document, and so is `GET /api/state`."""
    with make_client(AgentResponse(text="Hi.")) as (client, _):
        client.post("/api/command", json={"text": "hello"})
        assert "reply" not in client.get("/api/state").json()


def test_a_second_tab_hears_the_reply_after_the_board_it_is_about():
    played = AgentResponse(
        text="e4 it is.",
        tool_calls=(ToolCall(name="make_move", args={"move": "e4"}),),
    )
    with make_client(played, narrations=("e4 it is.",)) as (client, _):
        with client.websocket_connect("/ws") as ws:
            receive_state(ws)  # connect snapshot
            body = client.post("/api/command", json={"text": "play e4"}).json()
            kinds = []
            while "reply" not in kinds:
                message = ws.receive_json()
                kinds.append(message["type"])
                if message["type"] == "reply":
                    reply = message["reply"]
    assert reply["text"] == body["commentary"]
    assert reply["game_id"] == body["state"]["game_id"]
    # The board is on the other tab before the words about it are.
    assert "state" in kinds[: kinds.index("reply")]


def test_replies_are_numbered_so_a_resent_one_reads_as_old():
    with make_client(AgentResponse(text="One."), AgentResponse(text="Two.")) as (
        client,
        _,
    ):
        client.post("/api/command", json={"text": "first"})
        client.post("/api/command", json={"text": "second"})
        reply = restored_reply(client)
    assert reply is not None
    assert (reply["text"], reply["seq"]) == ("Two.", 2)


def test_a_dragged_moves_reaction_is_the_panels_reply():
    with make_client(narrations=("Bold.",)) as (client, _):
        with client.websocket_connect("/ws") as ws:
            receive_state(ws)
            body = client.post("/api/game/move", json={"move": "e2e4"}).json()
            reply = receive(ws, "reply")["reply"]
        restored = restored_reply(client)
    assert reply["text"] == body["commentary"]
    assert restored is not None
    assert restored["text"] == body["commentary"]


def test_a_delegate_threads_words_are_not_the_panels():
    with make_client(AgentResponse(text="Hello, delegate.")) as (client, _):
        conversation = client.post("/api/agent/conversations", json={}).json()
        client.post(
            f"/api/agent/conversations/{conversation['id']}/messages",
            json={"content": "hi"},
        )
        assert restored_reply(client) is None


def test_a_new_game_leaves_the_old_games_reply_unsaid():
    with make_client(AgentResponse(text="Ask me anything.")) as (client, _):
        client.post("/api/command", json={"text": "hello"})
        assert restored_reply(client) is not None
        # Nothing played yet, so the new game runs without asking.
        new = client.post("/api/game/new", json={"color": "white"})
        assert new.status_code == 200
        assert restored_reply(client) is None


# --- settings -----------------------------------------------------------------


def test_a_difficulty_change_is_broadcast_to_every_tab():
    with make_client() as (client, _):
        with client.websocket_connect("/ws") as ws:
            receive_state(ws)
            client.post("/api/game/difficulty", json={"tier": "advanced"})
            settings = receive(ws, "settings")["settings"]
        assert settings["tier"] == "advanced"
        assert settings == client.get("/api/settings").json()


def test_a_difficulty_set_by_chat_is_broadcast_too():
    easier = AgentResponse(
        text="Easier it is.",
        tool_calls=(ToolCall(name="set_difficulty", args={"tier": "beginner"}),),
    )
    with make_client(easier) as (client, _):
        with client.websocket_connect("/ws") as ws:
            receive_state(ws)
            client.post("/api/command", json={"text": "go easy on me"})
            settings = receive(ws, "settings")["settings"]
    assert settings["tier"] == "beginner"


def test_a_turn_that_changes_no_setting_broadcasts_none():
    with make_client(AgentResponse(text="Hi.")) as (client, _):
        with client.websocket_connect("/ws") as ws:
            receive_state(ws)
            client.post("/api/command", json={"text": "hello"})
            client.post("/api/settings/voice", json={"enabled": True})
            # The first settings frame is the voice toggle's, not the chat turn's.
            settings = receive(ws, "settings")["settings"]
    assert settings["voice_output"] is True
