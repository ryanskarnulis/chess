"""API layer: game lifecycle + state fetch over HTTP.

The API is trusted code, so it talks to `GameSession` directly (through the
shared `ToolContext`) — the tool registry stays the LLM's boundary. Illegal
moves are data (`legal: false`), not HTTP errors; domain failures on
mutations (undo with nothing to undo, resigning a finished game) are 409s.
Engine-reply tests need a live Stockfish and are skipped without one.
"""

import shutil

import pytest
from fastapi.testclient import TestClient

from chessapp.api import create_app
from chessapp.engine import EnginePlayer
from chessapp.game import GameSession
from chessapp.tools import ToolContext

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

requires_stockfish = pytest.mark.skipif(
    shutil.which("stockfish") is None, reason="stockfish binary not installed"
)


@pytest.fixture
def ctx():
    return ToolContext(session=GameSession())


@pytest.fixture
def client(ctx):
    return TestClient(create_app(ctx))


# --- state fetch -----------------------------------------------------------


def test_state_of_fresh_game(client):
    body = client.get("/api/state").json()
    assert body["fen"] == START_FEN
    assert body["turn"] == "white"
    assert body["game_over"] is False
    assert body["outcome"] is None
    assert body["history"] == []
    assert body["captured"] == {"white": [], "black": []}
    assert "e4" in body["legal_moves"]
    # Move hints for the board UI, grouped by origin square.
    assert set(body["dests"]["e2"]) == {"e3", "e4"}


def test_state_reflects_moves_and_captures(client):
    for move in ["e4", "d5", "exd5"]:
        client.post("/api/game/move", json={"move": move})
    body = client.get("/api/state").json()
    assert body["history"] == ["e4", "d5", "exd5"]
    assert body["captured"] == {"white": ["p"], "black": []}
    assert body["turn"] == "black"


# --- moves -----------------------------------------------------------------


def test_legal_move_returns_move_and_new_state(client):
    body = client.post("/api/game/move", json={"move": "e4"}).json()
    assert body["legal"] is True
    assert body["san"] == "e4"
    assert body["uci"] == "e2e4"
    assert body["engine_move"] is None
    assert body["state"]["turn"] == "black"


def test_illegal_move_is_data_not_an_error(client):
    response = client.post("/api/game/move", json={"move": "e5"})
    assert response.status_code == 200
    body = response.json()
    assert body["legal"] is False
    assert "e5" in body["reason"]
    assert body["state"]["fen"] == START_FEN


def test_move_accepts_uci(client):
    body = client.post("/api/game/move", json={"move": "g1f3"}).json()
    assert body["legal"] is True
    assert body["san"] == "Nf3"


def test_missing_move_field_is_422(client):
    assert client.post("/api/game/move", json={}).status_code == 422


# --- lifecycle -------------------------------------------------------------


def test_new_game_resets_position(client):
    client.post("/api/game/move", json={"move": "e4"})
    body = client.post("/api/game/new").json()
    assert body["state"]["fen"] == START_FEN
    assert body["state"]["history"] == []


def test_undo_takes_back_plies(client):
    client.post("/api/game/move", json={"move": "e4"})
    client.post("/api/game/move", json={"move": "e5"})
    body = client.post("/api/game/undo", json={"plies": 2}).json()
    assert body["undone"] == ["e5", "e4"]
    assert body["state"]["fen"] == START_FEN


def test_undo_defaults_to_one_ply(client):
    client.post("/api/game/move", json={"move": "e4"})
    body = client.post("/api/game/undo", json={}).json()
    assert body["undone"] == ["e4"]


def test_undo_with_nothing_played_is_409(client):
    response = client.post("/api/game/undo", json={})
    assert response.status_code == 409
    assert "cannot undo" in response.json()["detail"]


def test_resign_ends_game(client):
    body = client.post("/api/game/resign", json={}).json()
    assert body["outcome"] == {
        "termination": "resignation",
        "winner": "black",
        "result": "0-1",
    }
    assert body["state"]["game_over"] is True


def test_resign_specific_color(client):
    body = client.post("/api/game/resign", json={"color": "black"}).json()
    assert body["outcome"]["winner"] == "white"


def test_resign_when_game_over_is_409(client):
    client.post("/api/game/resign", json={})
    assert client.post("/api/game/resign", json={}).status_code == 409


def test_move_after_game_over_is_rejected_as_illegal(client):
    client.post("/api/game/resign", json={})
    body = client.post("/api/game/move", json={"move": "e4"}).json()
    assert body["legal"] is False


def test_export_pgn(client):
    client.post("/api/game/move", json={"move": "e4"})
    client.post("/api/game/resign", json={})
    body = client.get("/api/game/pgn").json()
    assert "1. e4" in body["pgn"]
    assert "1-0" in body["pgn"]


# --- difficulty ------------------------------------------------------------


def test_set_difficulty_by_skill_level(client):
    body = client.post("/api/game/difficulty", json={"skill_level": 5}).json()
    assert body["skill_level"] == 5
    assert body["elo"] is None


def test_set_difficulty_by_elo(client):
    body = client.post("/api/game/difficulty", json={"elo": 1500}).json()
    assert body["elo"] == 1500
    assert body["skill_level"] is None


def test_set_difficulty_updates_settings(ctx):
    client = TestClient(create_app(ctx))
    client.post("/api/game/difficulty", json={"skill_level": 7})
    assert ctx.settings.skill_level == 7
    assert ctx.settings.elo is None
    # The last one set wins; the other is cleared.
    client.post("/api/game/difficulty", json={"elo": 1600})
    assert ctx.settings.elo == 1600
    assert ctx.settings.skill_level is None


def test_set_difficulty_requires_exactly_one(client):
    assert client.post("/api/game/difficulty", json={}).status_code == 422
    both = client.post("/api/game/difficulty", json={"skill_level": 5, "elo": 1500})
    assert both.status_code == 422


def test_set_difficulty_out_of_range_is_409(client):
    high = client.post("/api/game/difficulty", json={"skill_level": 99})
    assert high.status_code == 409
    assert client.post("/api/game/difficulty", json={"elo": 100}).status_code == 409


# --- engine reply (LLM-off vs-Stockfish mode) ------------------------------


@requires_stockfish
def test_engine_replies_to_player_move():
    with EnginePlayer() as engine:
        ctx = ToolContext(session=GameSession(), engine=engine)
        client = TestClient(create_app(ctx))
        body = client.post("/api/game/move", json={"move": "e4"}).json()
        assert body["legal"] is True
        assert body["engine_move"]["legal"] is True
        assert body["state"]["turn"] == "white"
        assert len(body["state"]["history"]) == 2


@requires_stockfish
def test_engine_does_not_reply_to_illegal_move():
    with EnginePlayer() as engine:
        ctx = ToolContext(session=GameSession(), engine=engine)
        client = TestClient(create_app(ctx))
        body = client.post("/api/game/move", json={"move": "e5"}).json()
        assert body["legal"] is False
        assert body["engine_move"] is None
        assert body["state"]["history"] == []


# --- websocket state channel ------------------------------------------------


def test_ws_sends_current_state_on_connect(client):
    client.post("/api/game/move", json={"move": "e4"})
    with client.websocket_connect("/ws") as ws:
        message = ws.receive_json()
    assert message["type"] == "state"
    assert message["state"]["history"] == ["e4"]


def test_ws_broadcasts_state_after_move(client):
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()  # connect snapshot
        client.post("/api/game/move", json={"move": "e4"})
        message = ws.receive_json()
    assert message["state"]["turn"] == "black"
    assert message["state"]["history"] == ["e4"]


def test_ws_broadcasts_lifecycle_mutations(client):
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()
        client.post("/api/game/move", json={"move": "e4"})
        ws.receive_json()
        client.post("/api/game/undo", json={})
        assert ws.receive_json()["state"]["history"] == []
        client.post("/api/game/resign", json={})
        assert ws.receive_json()["state"]["game_over"] is True
        client.post("/api/game/new")
        message = ws.receive_json()
    assert message["state"]["fen"] == START_FEN
    assert message["state"]["game_over"] is False


def test_ws_no_broadcast_for_illegal_move_or_failed_mutation(client):
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()
        client.post("/api/game/move", json={"move": "e5"})  # illegal
        client.post("/api/game/undo", json={})  # 409: nothing played
        client.post("/api/game/move", json={"move": "e4"})  # legal
        message = ws.receive_json()
    assert message["state"]["history"] == ["e4"]


def test_ws_multiple_clients_all_receive(client):
    with client.websocket_connect("/ws") as ws1, client.websocket_connect("/ws") as ws2:
        ws1.receive_json()
        ws2.receive_json()
        client.post("/api/game/move", json={"move": "e4"})
        assert ws1.receive_json()["state"]["history"] == ["e4"]
        assert ws2.receive_json()["state"]["history"] == ["e4"]


def test_disconnected_client_does_not_break_broadcast(client):
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()
    response = client.post("/api/game/move", json={"move": "e4"})
    assert response.status_code == 200
    assert response.json()["legal"] is True


# --- settings ----------------------------------------------------------------


def test_get_settings_returns_the_full_settings_document(client):
    body = client.get("/api/settings").json()
    assert body == {
        "personality": "friendly_rival",
        "verbosity": "normal",
        "hints_mode": False,
        "voice_output": False,
        "skill_level": None,
        "elo": None,
    }


def test_set_voice_output_toggles_the_setting(ctx, client):
    body = client.post("/api/settings/voice", json={"enabled": True}).json()
    assert body == {"voice_output": True}
    assert ctx.settings.voice_output is True
    assert client.get("/api/settings").json()["voice_output"] is True
    client.post("/api/settings/voice", json={"enabled": False})
    assert ctx.settings.voice_output is False
