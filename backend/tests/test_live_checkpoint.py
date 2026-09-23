"""The live game survives a restart, and a client can bind to one game (#291).

The board, the panel transcript and the game's identity are checkpointed to
`live.json` in the save dir on every change and restored at startup. The
pending confirmation is not: a question dies with the process that asked it.
A restored board is a *new* board version (clients that held the old number
are stale) but the *same* game (`game_id`), and a request may carry the
`game_id` it means as a precondition.
"""

import json

from fastapi.testclient import TestClient

from chessapp.app import build_app
from chessapp.brain import AgentResponse, ToolCall
from chessapp.game import GameSession
from chessapp.tools import (
    LIVE_CHECKPOINT_FILENAME,
    ToolContext,
    _save_path,
    live_checkpoint,
    restore_live_checkpoint,
    saved_game_names,
    write_live_checkpoint,
)
from fakes import FakeEngine, scripted_app


def _direct(save_dir, engine=None) -> TestClient:
    """The app as it starts on the box, LLM off: restore included."""
    return TestClient(
        build_app(
            agent_enabled=False,
            engine=engine if engine is not None else FakeEngine(),
            save_dir=save_dir,
        )
    )


def _state(client: TestClient) -> dict:
    return client.get("/api/state").json()


# --- restart --------------------------------------------------------------------


def test_a_restart_brings_the_game_back(tmp_path):
    before = _direct(tmp_path)
    assert before.post("/api/game/move", json={"move": "e4"}).status_code == 200
    was = _state(before)
    assert was["history"] == ["e4", "e5"]

    after = _state(_direct(tmp_path))

    assert after["history"] == ["e4", "e5"]
    assert after["fen"] == was["fen"]
    assert after["game_id"] == was["game_id"], "the same game, restored"
    assert after["version"] > was["version"], "and a new board version"


def test_a_version_held_across_a_restart_is_stale(tmp_path):
    before = _direct(tmp_path)
    before.post("/api/game/move", json={"move": "e4"})
    held = _state(before)["version"]

    client = _direct(tmp_path)
    response = client.post("/api/game/move", json={"move": "d4", "version": held})

    assert response.status_code == 409
    assert response.json()["stale"] is True
    assert _state(client)["history"] == ["e4", "e5"]


def test_a_checkpoint_taken_mid_exchange_is_settled_on_restore(tmp_path):
    """The process stopped between the player's move and the reply: nothing
    is left to collect that reply, so startup settles it, as a resume does."""
    ctx = ToolContext(session=GameSession(), save_dir=tmp_path)
    assert ctx.session.submit_move("e4").legal
    write_live_checkpoint(ctx, live_checkpoint(ctx))

    state = _state(_direct(tmp_path, FakeEngine("e7e5")))

    assert state["history"] == ["e4", "e5"]
    assert state["turn"] == "white"


def test_no_checkpoint_is_a_fresh_board(tmp_path):
    state = _state(_direct(tmp_path))
    assert state["history"] == []


def test_a_corrupt_checkpoint_is_a_fresh_board_not_a_failed_start(tmp_path):
    (tmp_path / LIVE_CHECKPOINT_FILENAME).write_text("{half a document")
    assert _state(_direct(tmp_path))["history"] == []


def test_a_tampered_checkpoint_cannot_put_an_illegal_board_up(tmp_path):
    ctx = ToolContext(session=GameSession(), save_dir=tmp_path)
    data = live_checkpoint(ctx)
    data["session"]["moves"] = ["e2e4", "e2e4"]
    (tmp_path / LIVE_CHECKPOINT_FILENAME).write_text(json.dumps(data))

    fresh = ToolContext(session=GameSession(), save_dir=tmp_path)
    assert restore_live_checkpoint(fresh) is False
    assert fresh.session.move_history() == []


def test_the_transcript_is_restored_with_the_board(tmp_path):
    ctx = ToolContext(session=GameSession(), save_dir=tmp_path)
    app, _ = scripted_app(
        ctx,
        AgentResponse(
            text="e4 it is.",
            tool_calls=(ToolCall(name="make_move", args={"move": "e4"}),),
        ),
    )
    TestClient(app).post("/api/command", json={"text": "go ahead and play e4"})
    assert ctx.transcript.to_dict(), "the exchange was recorded"

    fresh = ToolContext(session=GameSession(), save_dir=tmp_path)
    assert restore_live_checkpoint(fresh) is True

    assert fresh.transcript.to_dict() == ctx.transcript.to_dict()
    assert fresh.session.move_history() == ctx.session.move_history()
    assert fresh.board_version == ctx.board_version + 1


def test_a_pending_question_does_not_survive_a_restart(tmp_path):
    before = _direct(tmp_path)
    before.post("/api/game/move", json={"move": "e4"})
    assert before.post("/api/game/new", json={"color": "white"}).status_code == 409

    client = _direct(tmp_path)
    response = client.post("/api/game/confirm", json={"confirm": True})

    assert response.status_code == 409, "nothing is armed after a restart"
    assert _state(client)["history"] == ["e4", "e5"]


def test_the_checkpoint_is_never_offered_as_a_save(tmp_path):
    """Root-level and nested, so neither `saved_game_names` nor the legacy-save
    migration (which promotes root JSON that loads as a game) touches it."""
    _direct(tmp_path).post("/api/game/move", json={"move": "e4"})
    _direct(tmp_path)  # a second start runs the migration over it

    ctx = ToolContext(session=GameSession(), save_dir=tmp_path)
    assert saved_game_names(ctx) == []
    assert (tmp_path / LIVE_CHECKPOINT_FILENAME).exists()


def test_no_save_dir_writes_nothing(tmp_path):
    client = TestClient(build_app(agent_enabled=False, engine=FakeEngine()))
    assert client.post("/api/game/move", json={"move": "e4"}).status_code == 200
    assert list(tmp_path.iterdir()) == []


# --- game identity --------------------------------------------------------------


def test_a_new_game_is_a_new_game_id(tmp_path):
    client = _direct(tmp_path)
    first = _state(client)["game_id"]

    client.post("/api/game/new", json={"color": "white"})

    assert _state(client)["game_id"] != first


def test_a_move_keeps_the_game_id(tmp_path):
    client = _direct(tmp_path)
    first = _state(client)["game_id"]
    client.post("/api/game/move", json={"move": "e4"})
    assert _state(client)["game_id"] == first


def test_the_right_game_id_is_accepted(tmp_path):
    client = _direct(tmp_path)
    game_id = _state(client)["game_id"]

    response = client.post("/api/game/move", json={"move": "e4", "game_id": game_id})

    assert response.status_code == 200


def test_a_request_about_another_game_is_refused_untouched(tmp_path):
    client = _direct(tmp_path)
    client.post("/api/game/move", json={"move": "e4"})
    old = _state(client)["game_id"]
    assert client.post("/api/game/new", json={"color": "white"}).status_code == 409
    client.post("/api/game/confirm", json={"confirm": True})
    current = _state(client)

    response = client.post("/api/game/move", json={"move": "d4", "game_id": old})

    assert response.status_code == 409
    body = response.json()
    assert body["stale"] is True
    assert body["game_id"] == current["game_id"]
    assert "no longer on the board" in body["detail"]
    assert _state(client)["history"] == []


def test_a_resumed_save_is_a_game_of_its_own(tmp_path):
    """Resuming one save twice gives two games; a client holding the first
    one's id cannot act on the second."""
    ctx = ToolContext(session=GameSession(), save_dir=tmp_path)
    saved = GameSession()
    assert saved.submit_move("e4").legal
    path = _save_path(ctx, "scholars")
    path.parent.mkdir(parents=True)
    saved.save(path)
    app, _ = scripted_app(
        ctx,
        AgentResponse(
            text="Loaded.",
            tool_calls=(ToolCall(name="resume_game", args={"name": "scholars"}),),
        ),
    )
    TestClient(app).post("/api/command", json={"text": "load up scholars please"})

    assert ctx.session.move_history()[0] == "e4"
    assert ctx.session.game_id != saved.game_id


def test_the_game_id_round_trips_and_old_saves_get_one():
    session = GameSession()
    assert GameSession.from_dict(session.to_dict()).game_id == session.game_id
    legacy = session.to_dict()
    del legacy["game_id"]
    assert len(GameSession.from_dict(legacy).game_id) == 32


def test_a_malformed_game_id_is_refused():
    data = GameSession().to_dict()
    data["game_id"] = "../../etc"
    try:
        GameSession.from_dict(data)
    except ValueError:
        return
    raise AssertionError("a malformed game_id loaded")


def test_the_delegate_can_bind_a_message_to_a_game(tmp_path):
    ctx = ToolContext(session=GameSession(), save_dir=tmp_path)
    app, brain = scripted_app(
        ctx,
        AgentResponse(
            text="ok", tool_calls=(ToolCall(name="make_move", args={"move": "d4"}),)
        ),
    )
    client = TestClient(app)
    thread = client.post("/api/agent/conversations", json={}).json()["id"]

    response = client.post(
        f"/api/agent/conversations/{thread}/messages",
        json={"content": "play d4", "game_id": "0" * 32},
    )

    assert response.status_code == 409
    assert response.json()["game_id"] == ctx.session.game_id
    assert ctx.session.move_history() == []
