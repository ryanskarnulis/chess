"""API layer: game lifecycle + state fetch over HTTP.

The API is trusted code, so it talks to `GameSession` directly (through the
shared `ToolContext`) — the tool registry stays the LLM's boundary. Illegal
moves are data (`legal: false`), not HTTP errors; domain failures on
mutations (undo with nothing to undo, resigning a finished game) are 409s.
Engine-reply tests need a live Stockfish and are skipped without one.

Every app here is built with no brain, so this file is the **direct mode**
spec: the board route answers the exact document it always answered, with no
agent beats and no new keys. Agent mode, and the destructive-op gate the
lifecycle endpoints now share with the spoken road, live in
test_board_controls.py.
"""

import shutil

import pytest
from fastapi.testclient import TestClient

from chessapp.api import create_app
from chessapp.coordinator import TurnCoordinator, TurnPhase
from chessapp.engine import DEFAULT_TIER, CandidateMove, EnginePlayer
from chessapp.game import GameSession
from chessapp.tools import ToolContext, build_registry
from fakes import FakeEngine

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


def test_state_reports_no_claimable_draw_in_a_fresh_game(client):
    assert client.get("/api/state").json()["claimable_draws"] == []


def test_state_reports_a_claimable_threefold_repetition(client):
    """The issue's reproduction, as the document a client reads: the claim is
    available, the game is not over, and the rule is named — so a UI can offer
    the claim instead of the player having to know it exists."""
    for san in ("Nf3", "Nf6", "Ng1", "Ng8") * 2:
        assert client.post("/api/game/move", json={"move": san}).json()["legal"]

    body = client.get("/api/state").json()

    assert body["claimable_draws"] == ["threefold_repetition"]
    assert body["game_over"] is False
    assert body["outcome"] is None


def test_state_includes_per_ply_fens(client):
    assert client.get("/api/state").json()["fens"] == [START_FEN]
    for move in ["e4", "d5", "exd5"]:
        client.post("/api/game/move", json={"move": move})
    body = client.get("/api/state").json()
    assert len(body["fens"]) == len(body["history"]) + 1
    assert body["fens"][0] == START_FEN
    assert body["fens"][-1] == body["fen"]


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


# --- the turn coordinator on the trusted path -------------------------------


def test_board_move_advances_the_coordinator(ctx):
    ctx.engine = FakeEngine()
    coordinator = TurnCoordinator(ctx)
    client = TestClient(create_app(ctx, coordinator=coordinator))
    body = client.post("/api/game/move", json={"move": "e4"}).json()
    assert body["engine_move"]["uci"] == "e7e5"
    assert coordinator.turn_id == 2
    assert coordinator.phase == TurnPhase.AWAITING_PLAYER


def test_board_move_out_of_turn_is_409(ctx):
    """A turn-state rejection is a domain failure on the trusted path, the same
    as an impossible undo — the agent sees the same refusal as result data."""
    ctx.engine = FakeEngine()
    coordinator = TurnCoordinator(ctx)
    client = TestClient(create_app(ctx, coordinator=coordinator))
    coordinator.apply_player_move("e4")  # a turn is open, mid-sequence
    response = client.post("/api/game/move", json={"move": "d4"})
    assert response.status_code == 409
    assert ctx.session.move_history() == ["e4"]


def test_new_game_opening_move_comes_from_the_coordinator(ctx):
    ctx.engine = FakeEngine(reply_uci="e2e4")
    coordinator = TurnCoordinator(ctx)
    client = TestClient(create_app(ctx, coordinator=coordinator))
    body = client.post("/api/game/new", json={"color": "black"}).json()
    assert body["state"]["history"] == ["e4"]
    # The engine's opening move is not a turn: the player's first is still due.
    assert coordinator.turn_id == 1
    assert coordinator.phase == TurnPhase.AWAITING_PLAYER


def test_tool_path_and_board_path_share_one_turn_machine(ctx):
    """App assembly passes one coordinator to both `build_registry` and
    `create_app`, so a move typed at the agent and a move dragged on the board
    advance the same turn counter — not two machines that can disagree."""
    ctx.engine = FakeEngine()
    coordinator = TurnCoordinator(ctx)
    registry = build_registry(ctx, coordinator)
    client = TestClient(create_app(ctx, registry=registry, coordinator=coordinator))
    registry.dispatch("make_move", {"move": "e4"})
    assert coordinator.turn_id == 2
    client.post("/api/game/move", json={"move": "Nf3"})
    assert coordinator.turn_id == 3


# --- lifecycle -------------------------------------------------------------


def test_new_game_resets_position(client):
    client.post("/api/game/move", json={"move": "e4"})
    # A game in progress is not thrown away unanswered: the endpoint dispatches
    # through the same gate the agent's `new_game` hits, so mid-game it asks
    # (409 + the question) and the answer is what resets. The gate's contract
    # lives in test_board_controls.py; this pins that the reset still happens.
    assert client.post("/api/game/new").status_code == 409
    body = client.post("/api/game/confirm", json={"confirm": True}).json()
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


# --- claim-draw button ------------------------------------------------------
#
# The endpoint half of a claim the state document already advertises under
# `claimable_draws`. Until it existed the `claim_draw` tool was reachable only
# through the brain, so in direct mode a claim was unreachable (#220 follow-up).

REPETITION = ("Nf3", "Nf6", "Ng1", "Ng8")

# King and rook against a bare king with the fifty-move count complete: a claim
# is available on a board nobody has moved on, so the gate stands aside.
FIFTY_MOVE_FEN = "8/8/8/4k3/8/8/4K3/6R1 w - - 100 80"


def repeated(ctx):
    for san in REPETITION * 2:
        assert ctx.session.submit_move(san).legal
    return ctx


def test_claim_draw_with_nothing_to_claim_is_409(client):
    response = client.post("/api/game/claim-draw", json={})
    assert response.status_code == 409
    assert "no draw" in response.json()["detail"]
    assert "confirm" not in response.json(), "nothing to claim, so nothing armed"


def test_claim_draw_asks_then_draws_the_game(client, ctx):
    repeated(ctx)
    assert client.get("/api/state").json()["claimable_draws"] == [
        "threefold_repetition"
    ]

    asked = client.post("/api/game/claim-draw", json={})
    assert asked.status_code == 409
    assert asked.json()["confirm"] is True
    assert asked.json()["op"] == "claim_draw"

    body = client.post("/api/game/confirm", json={"confirm": True}).json()
    assert body["op"] == "claim_draw"
    assert body["state"]["game_over"] is True
    assert body["state"]["outcome"] == {
        "termination": "threefold_repetition",
        "winner": None,
        "result": "1/2-1/2",
    }
    assert body["state"]["claimable_draws"] == [], "nothing left to claim"


def test_claim_draw_runs_outright_where_the_gate_stands_aside():
    """A fifty-move claim on a board the player never moved on: no investment to
    guard, so the endpoint answers like `resign` does on a fresh board."""
    ctx = ToolContext(session=GameSession(fen=FIFTY_MOVE_FEN))
    client = TestClient(create_app(ctx))
    body = client.post("/api/game/claim-draw", json={}).json()
    assert body["outcome"] == {
        "termination": "fifty_moves",
        "winner": None,
        "result": "1/2-1/2",
    }
    assert body["state"]["game_over"] is True


def test_claim_draw_when_game_over_is_409(client):
    client.post("/api/game/resign", json={})
    assert client.post("/api/game/claim-draw", json={}).status_code == 409


def test_export_pgn_after_a_claimed_draw(client, ctx):
    repeated(ctx)
    client.post("/api/game/claim-draw", json={})  # gated mid-game: it asks
    client.post("/api/game/confirm", json={"confirm": True})
    body = client.get("/api/game/pgn").json()
    assert "1/2-1/2" in body["pgn"]


def test_move_after_game_over_is_rejected_as_illegal(client):
    client.post("/api/game/resign", json={})
    body = client.post("/api/game/move", json={"move": "e4"}).json()
    assert body["legal"] is False


def test_export_pgn(client):
    client.post("/api/game/move", json={"move": "e4"})
    client.post("/api/game/resign", json={})  # gated mid-game: it asks
    client.post("/api/game/confirm", json={"confirm": True})
    body = client.get("/api/game/pgn").json()
    assert "1. e4" in body["pgn"]
    # 0-1: the resign button is the *player* conceding (white here), not the
    # side to move — the rule the `resign` tool derives, now that the endpoint
    # dispatches through it. The side to move was only ever coincidentally the
    # player (trace review, finding 8).
    assert "0-1" in body["pgn"]


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


def test_set_difficulty_by_tier(ctx):
    engine = FakeEngine()
    ctx.engine = engine
    client = TestClient(create_app(ctx))
    body = client.post("/api/game/difficulty", json={"tier": "beginner"}).json()
    assert body["tier"] == "beginner"
    assert body["skill_level"] is None
    assert body["elo"] is None
    assert ctx.settings.tier == "beginner"
    assert engine.tiers == ["beginner"]
    # A raw knob replaces the tier, and vice versa.
    client.post("/api/game/difficulty", json={"skill_level": 7})
    assert ctx.settings.tier is None
    assert ctx.settings.skill_level == 7


def test_set_difficulty_unknown_tier_is_409(client):
    assert (
        client.post("/api/game/difficulty", json={"tier": "impossible"}).status_code
        == 409
    )


def test_settings_report_tier(client):
    client.post("/api/game/difficulty", json={"tier": "advanced"})
    settings = client.get("/api/settings").json()
    assert settings["tier"] == "advanced"


def test_set_difficulty_requires_exactly_one(client):
    assert client.post("/api/game/difficulty", json={}).status_code == 422
    both = client.post("/api/game/difficulty", json={"skill_level": 5, "elo": 1500})
    assert both.status_code == 422
    tier_and_skill = client.post(
        "/api/game/difficulty", json={"tier": "beginner", "skill_level": 5}
    )
    assert tier_and_skill.status_code == 422


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


def test_engine_reply_never_detours_through_multipv():
    # The reply is the engine's own move at the configured strength — never a
    # MultiPV detour that would bypass the difficulty setting (a personality
    # move-bias layer was tried in 2026-07 and removed for exactly this).
    engine = FakeEngine()
    ctx = ToolContext(session=GameSession(), engine=engine)
    client = TestClient(create_app(ctx))
    body = client.post("/api/game/move", json={"move": "e4"}).json()
    assert body["engine_move"]["legal"] is True
    assert body["engine_move"]["uci"] == "e7e5"
    assert engine.multipv_requests == []


# --- game review ------------------------------------------------------------


def test_review_without_engine_is_503(client):
    assert client.get("/api/game/review").status_code == 503


@requires_stockfish
def test_review_of_an_empty_game_is_409():
    with EnginePlayer() as engine:
        ctx = ToolContext(session=GameSession(), engine=engine)
        client = TestClient(create_app(ctx))
        assert client.get("/api/game/review").status_code == 409


@requires_stockfish
def test_review_returns_moves_and_accuracy():
    with EnginePlayer() as engine:
        ctx = ToolContext(session=GameSession(), engine=engine)
        client = TestClient(create_app(ctx))
        client.post("/api/game/move", json={"move": "e4"})
        response = client.get("/api/game/review")
        assert response.status_code == 200
        body = response.json()
        assert len(body["moves"]) == len(ctx.session.move_history())
        assert body["moves"][0]["san"] == "e4"
        assert "accuracy" in body and "counts" in body


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
        "verbosity": "normal",
        "voice_output": False,
        "tier": DEFAULT_TIER,
        "skill_level": None,
        "elo": None,
        # These tests build the app without a brain: direct mode, and the UI
        # renders that as a visible state rather than discovering it on a 503.
        "agent_available": False,
    }


def test_set_voice_output_toggles_the_setting(ctx, client):
    body = client.post("/api/settings/voice", json={"enabled": True}).json()
    assert body == {"voice_output": True}
    assert ctx.settings.voice_output is True
    assert client.get("/api/settings").json()["voice_output"] is True
    client.post("/api/settings/voice", json={"enabled": False})
    assert ctx.settings.voice_output is False


# --- hint --------------------------------------------------------------


def _hint_client(ctx, best_moves=()):
    ctx.engine = FakeEngine(best_moves=best_moves)
    return TestClient(create_app(ctx))


def test_hint_without_engine_is_503(client):
    assert client.get("/api/game/hint").status_code == 503


def test_hint_returns_best_move(ctx):
    best = CandidateMove(uci="e2e4", san="e4", score_cp=30, mate_in=None)
    client = _hint_client(ctx, best_moves=(best,))
    body = client.get("/api/game/hint").json()
    # Exact shape, so a stray key cannot appear unnoticed. `version` is the
    # board it analyzed (0 for an untouched game) — see the test below.
    assert body == {"uci": "e2e4", "san": "e4", "from": "e2", "to": "e4", "version": 0}


def test_hint_destination_is_correct_for_promotion(ctx):
    best = CandidateMove(uci="e7e8q", san="e8=Q", score_cp=900, mate_in=None)
    client = _hint_client(ctx, best_moves=(best,))
    body = client.get("/api/game/hint").json()
    assert body["from"] == "e7"
    assert body["to"] == "e8"


def test_hint_names_the_board_version_it_analyzed(ctx):
    """A hint is an answer about one position, so it says which one (#218).

    The search takes real time, and nothing stops the board moving while it
    runs — another client, or this one dragging a piece. Without the version in
    the payload a client that asked has no way to tell an answer about its board
    from an answer about the board before it, and the arrow lands on whichever
    position happens to be on screen when the response arrives. Same counter the
    state document publishes (`ToolContext.board_version`), so the two agree
    read back-to-back and a client can compare them directly.
    """
    best = CandidateMove(uci="e2e4", san="e4", score_cp=30, mate_in=None)
    client = _hint_client(ctx, best_moves=(best,))
    fresh = client.get("/api/game/hint").json()
    assert fresh["version"] == client.get("/api/state").json()["version"]
    # And it moves with the board: two hints across a move are distinguishable,
    # which is the whole point of carrying the number.
    client.post("/api/game/move", json={"move": "e4"})
    later = client.get("/api/game/hint").json()
    assert later["version"] > fresh["version"]
    assert later["version"] == client.get("/api/state").json()["version"]


def test_hint_when_game_over_is_409(ctx):
    client = _hint_client(
        ctx, best_moves=(CandidateMove(uci="e2e4", san="e4", score_cp=0, mate_in=None),)
    )
    ctx.session.resign()
    assert client.get("/api/game/hint").status_code == 409


def test_hint_with_no_candidates_is_409(ctx):
    client = _hint_client(ctx)
    assert client.get("/api/game/hint").status_code == 409


# --- player color ------------------------------------------------------------


def test_state_includes_player_color(client):
    assert client.get("/api/state").json()["player_color"] == "white"


def test_new_game_accepts_an_explicit_color(client):
    body = client.post("/api/game/new", json={"color": "black"}).json()
    assert body["state"]["player_color"] == "black"


def test_new_game_defaults_to_a_random_color(client, monkeypatch):
    # The endpoint rolls; pin the roll so the test is deterministic.
    monkeypatch.setattr("chessapp.api.random.choice", lambda options: "black")
    body = client.post("/api/game/new").json()
    assert body["state"]["player_color"] == "black"


def test_new_game_with_invalid_color_is_422(client):
    response = client.post("/api/game/new", json={"color": "green"})
    assert response.status_code == 422


def test_new_game_as_black_gets_the_engine_opening():
    ctx = ToolContext(session=GameSession(), engine=FakeEngine(reply_uci="e2e4"))
    client = TestClient(create_app(ctx))
    body = client.post("/api/game/new", json={"color": "black"}).json()
    state = body["state"]
    assert state["player_color"] == "black"
    assert state["history"] == ["e4"]
    assert state["turn"] == "black"


def test_new_game_as_white_gets_no_engine_opening():
    ctx = ToolContext(session=GameSession(), engine=FakeEngine(reply_uci="e2e4"))
    client = TestClient(create_app(ctx))
    body = client.post("/api/game/new", json={"color": "white"}).json()
    assert body["state"]["history"] == []
    assert body["state"]["turn"] == "white"


def test_new_game_as_black_without_engine_leaves_white_to_move(client):
    # Engine-free sandbox: nobody owns white, so nothing moves.
    body = client.post("/api/game/new", json={"color": "black"}).json()
    assert body["state"]["player_color"] == "black"
    assert body["state"]["history"] == []
    assert body["state"]["turn"] == "white"


# --- smart undo (vs engine, a takeback is the full exchange) -----------------


def test_undo_default_takes_back_the_full_exchange_vs_engine():
    ctx = ToolContext(session=GameSession(), engine=FakeEngine())
    client = TestClient(create_app(ctx))
    client.post("/api/game/move", json={"move": "e4"})  # engine replies e5
    body = client.post("/api/game/undo", json={}).json()
    assert body["undone"] == ["e5", "e4"]
    assert body["state"]["fen"] == START_FEN


def test_undo_explicit_plies_is_honored_and_the_board_comes_back_settled():
    """A client may ask for its own count, and an odd one pops the engine's
    reply alone — leaving the engine to move on a board no turn is open over. A
    position waiting on a side the player does not own is not a state to
    publish, so the coordinator settles it before the response goes back: the
    same rule the `undo` tool follows, on the same board (audit 2026-09-05,
    finding 2)."""
    ctx = ToolContext(session=GameSession(), engine=FakeEngine())
    client = TestClient(create_app(ctx))
    client.post("/api/game/move", json={"move": "e4"})  # engine replies e5
    body = client.post("/api/game/undo", json={"plies": 1}).json()
    assert body["undone"] == ["e5"], "exactly the count that was asked for"
    assert body["state"]["history"] == ["e4", "e5"], "and the engine answered again"
    assert body["state"]["turn"] == "white", "the player is to move"


def test_undo_default_pair_settles_nothing():
    """The takeback the button sends leaves the player to move, so nothing is
    owed and the engine is not asked."""
    ctx = ToolContext(session=GameSession(), engine=FakeEngine())
    client = TestClient(create_app(ctx))
    client.post("/api/game/move", json={"move": "e4"})
    body = client.post("/api/game/undo", json={}).json()
    assert body["state"]["history"] == []
    assert body["state"]["turn"] == "white"


def test_a_refused_undo_leaves_an_open_turn_alone():
    """The route used to abandon the turn before finding out whether it could
    take anything back — the `undo` tool's own bug, on the other road. A
    takeback that cannot happen replaces no position, so the turn that is still
    owed an engine reply is not this request's to throw away."""
    ctx = ToolContext(session=GameSession(), engine=FakeEngine())
    coordinator = TurnCoordinator(ctx)
    client = TestClient(create_app(ctx, coordinator=coordinator))
    coordinator.apply_player_move("e4")  # a turn left open mid-sequence

    response = client.post("/api/game/undo", json={"plies": 100})

    assert response.status_code == 409
    assert ctx.session.move_history() == ["e4"], "a refusal moves nothing"
    assert coordinator.phase == TurnPhase.PLAYER_MOVE_APPLIED
    assert coordinator.collect_engine_reply().san == "e5", "the reply is still there"


def test_undo_default_as_black_with_only_the_engine_opening_is_409():
    # The engine's own first move is not the player's to take back.
    ctx = ToolContext(session=GameSession(), engine=FakeEngine(reply_uci="e2e4"))
    client = TestClient(create_app(ctx))
    client.post("/api/game/new", json={"color": "black"})
    response = client.post("/api/game/undo", json={})
    assert response.status_code == 409


def test_undo_default_is_one_ply_when_the_engine_did_not_reply():
    # Player (white) mates: the game ends before any engine reply, so the
    # default takeback is just the player's own move.
    session = GameSession(fen="6k1/5ppp/8/8/8/8/8/4R2K w - - 0 1")
    ctx = ToolContext(session=session, engine=FakeEngine())
    client = TestClient(create_app(ctx))
    client.post("/api/game/move", json={"move": "Re8#"})
    body = client.post("/api/game/undo", json={}).json()
    assert body["undone"] == ["Re8#"]
