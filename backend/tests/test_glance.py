"""The glance endpoint: `GET /api/agent/glance` (#536).

The workspace glance contract (`../agent-standard/app-yaml-glance-block.md`):
a small, read-only summary conductor's `look(chess)` reads with no model call
and no engine search. Driven through the real app with a `FakeEngine` that
fails the test if anything asks it to search.
"""

import json

from fastapi.testclient import TestClient

from chessapp.api import create_app
from chessapp.game import GameSession
from chessapp.tools import ToolContext
from fakes import FakeEngine

FOOLS_MATE = ("f3", "e5", "g4", "Qh4#")
RUY_LOPEZ = ("e4", "e5", "Nf3", "Nc6", "Bb5")


class NoSearchEngine(FakeEngine):
    """A glance never searches: any engine call is a failure."""

    def __getattribute__(self, name):
        if name in {"evaluate_position", "analyze", "best_move", "move_for"}:
            raise AssertionError(f"glance called the engine: {name}")
        return super().__getattribute__(name)


def _client(tmp_path=None, **session_kwargs):
    ctx = ToolContext(
        session=GameSession(**session_kwargs),
        engine=NoSearchEngine(),
        save_dir=tmp_path,
    )
    return TestClient(create_app(ctx)), ctx


def _play(ctx, *sans):
    for san in sans:
        assert ctx.session.submit_move(san).legal, san
        ctx.observe_ledger()


def _glance(client):
    response = client.get("/api/agent/glance")
    assert response.status_code == 200
    return response.json()


def test_no_game_has_neither_current_nor_last():
    client, _ = _client()
    body = _glance(client)
    assert body["current"] is None
    assert body["last_finished"] is None
    assert body["as_of"]


def test_game_in_progress_says_whose_move_and_how_far():
    client, ctx = _client()
    _play(ctx, *RUY_LOPEZ)
    body = _glance(client)
    current = body["current"]
    assert current["app"] == "chess" and current["kind"] == "game"
    assert current["id"] == ctx.session.game_id
    assert current["to_move"] == "Glitch"
    assert current["moves"] == 3
    assert current["opening"] == "Ruy Lopez"
    assert body["last_finished"] is None


def test_it_is_the_players_move_when_it_is():
    client, ctx = _client()
    _play(ctx, "e4", "e5")
    assert _glance(client)["current"]["to_move"] == "you"


def test_material_is_said_in_words_and_no_evaluation_is_invented():
    client, ctx = _client()
    _play(ctx, "e4", "d5", "exd5")
    current = _glance(client)["current"]
    assert current["material"] == "you are up a pawn"
    assert "evaluation" not in current


def test_last_finished_game_has_result_opening_and_date():
    client, ctx = _client()
    _play(ctx, *FOOLS_MATE)  # white player loses to black's mate
    body = _glance(client)
    assert body["current"] is None
    last = body["last_finished"]
    assert last["app"] == "chess" and last["kind"] == "game"
    assert last["id"] == ctx.session.game_id
    assert last["result"] == "you lost"
    assert last["termination"] == "checkmate"
    assert last["score"] == "0-1"
    assert last["date"] == last["date"][:10] and len(last["date"]) == 10
    assert "title" in last


def test_last_finished_is_the_latest_and_names_its_opening(tmp_path):
    client, ctx = _client(tmp_path)
    _play(ctx, *FOOLS_MATE)
    ctx.session.new_game()
    _play(ctx, "e4", "e5", "Qh5", "Nc6", "Bc4", "Nf6", "Qxf7#")
    body = _glance(client)
    last = body["last_finished"]
    assert last["result"] == "you won"
    assert last["id"] == ctx.session.game_id
    assert last["opening"]


def test_a_taken_back_ending_is_no_longer_the_last_game():
    client, ctx = _client()
    _play(ctx, *FOOLS_MATE)
    ctx.session.undo(1)
    ctx.observe_ledger()
    body = _glance(client)
    assert body["last_finished"] is None
    assert body["current"] is not None


def test_it_is_small_read_only_and_argument_free():
    client, ctx = _client()
    _play(ctx, *FOOLS_MATE)
    revision = ctx.session.revision
    seq = len(ctx.ledger.since(0))
    first = client.get("/api/agent/glance")
    second = client.get("/api/agent/glance")
    assert len(first.text) <= 600
    assert {k: v for k, v in first.json().items() if k != "as_of"} == {
        k: v for k, v in second.json().items() if k != "as_of"
    }
    assert ctx.session.revision == revision
    assert len(ctx.ledger.since(0)) == seq
    assert client.post("/api/agent/glance").status_code == 405
    json.loads(first.text)
