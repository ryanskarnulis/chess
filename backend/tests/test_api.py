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
