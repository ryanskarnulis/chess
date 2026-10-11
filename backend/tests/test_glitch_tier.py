"""The `glitch` difficulty tier (#473, docs/glitch-difficulty.md).

In it the mover chooses Glitch's reply, on every route, and Stockfish at
`beginner` plays when the mover fails — marked in the PGN, the trace and the
reply's result, never silent. With no mover (no brain) the tier cannot be
chosen. Tested at the boundary with a scripted mover: no model runs here.
"""

import io
import json
from dataclasses import replace

import chess
import chess.pgn
import pytest
from fastapi.testclient import TestClient

from chessapp.api import create_app
from chessapp.app import build_app
from chessapp.brain import ModelCall
from chessapp.coordinator import TurnCoordinator, TurnPhase
from chessapp.engine import DIFFICULTY_TIERS, MOVER_TIERS, STEPPED_TIERS
from chessapp.game import GameSession
from chessapp.move_source import (
    CHOOSER_FALLBACK,
    CHOOSER_GLITCH,
    NO_MOVER,
    GlitchSource,
)
from chessapp.mover import MoverChoice, MoverFailed, MoverFailureKind
from chessapp.tools import (
    GLITCH_NEEDS_MOVER,
    SETTINGS_FILENAME,
    ToolContext,
    build_registry,
    pgn_headers,
)
from chessapp.trace import JsonlTracer
from fakes import FakeEngine, ScriptedBrain, scripted_app

GLITCH = "glitch"


class StubMover:
    """Plays `outcomes` in order: a UCI string is a choice, a
    `MoverFailureKind` raises that failure. Records the moves it was shown."""

    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.seen: list[list[str]] = []

    def choose(self, session):
        self.seen.append(session.move_history())
        outcome = self.outcomes.pop(0)
        call = ModelCall(phase="mover", ms=5)
        if isinstance(outcome, MoverFailureKind):
            raise MoverFailed(
                outcome, "scripted", model="m", attempts=(), calls=(call,)
            )
        board = chess.Board(session.fen())
        san = board.san(chess.Move.from_uci(outcome))
        return MoverChoice(
            uci=outcome,
            san=san,
            intent="hitting the centre",
            model="m",
            attempts=(),
            calls=(call,),
        )


class DeadEngine(FakeEngine):
    def choose_move(self, session):
        raise ValueError("engine died")


def glitch_ctx(mover=None, *, engine=None, session=None, with_mover=True):
    ctx = ToolContext(
        session=session or GameSession(), engine=engine or FakeEngine("e7e5")
    )
    ctx.settings.tier = GLITCH
    if with_mover:
        ctx.glitch_mover = lambda: mover
    return ctx


# --- the tier itself ------------------------------------------------------------


def test_glitch_is_a_tier_with_the_beginner_engine_underneath():
    glitch = DIFFICULTY_TIERS[GLITCH]
    assert glitch.mover is True
    # The fallback plays exactly what beginner plays (decision 4).
    assert replace(glitch, name="beginner", mover=False) == DIFFICULTY_TIERS["beginner"]
    assert MOVER_TIERS == {GLITCH}


def test_glitch_is_not_on_the_difficulty_ladder():
    assert GLITCH not in STEPPED_TIERS
    assert STEPPED_TIERS[-1] == "maximum"


# --- the mover chooses, on every route ----------------------------------------


def test_the_reply_is_the_movers_and_is_not_computed_in_the_background():
    mover = StubMover("c7c5")
    ctx = glitch_ctx(mover)
    coordinator = TurnCoordinator(ctx)

    coordinator.apply_player_move("e4")
    # Nothing asked yet: a background mover call would compete with the
    # planner for the one llama-server.
    assert mover.seen == []

    reply = coordinator.collect_engine_reply()

    assert reply is not None and reply.san == "c5"
    assert mover.seen == [["e4"]]
    sourced = coordinator.sourced_for(reply)
    assert sourced.chooser == CHOOSER_GLITCH
    assert sourced.mover["intent"] == "hitting the centre"
    assert ctx.session.move_notes == {}


def test_settle_asks_the_mover_for_a_game_taken_as_black():
    session = GameSession()
    session.new_game("black")
    mover = StubMover("d2d4")
    coordinator = TurnCoordinator(glitch_ctx(mover, session=session))

    reply = coordinator.settle_engine_turn()

    assert reply is not None and reply.san == "d4"
    assert coordinator.sourced_for(reply).chooser == CHOOSER_GLITCH


def test_play_exchange_asks_the_mover():
    mover = StubMover("c7c5")
    ctx = glitch_ctx(mover)
    coordinator = TurnCoordinator(ctx)

    _, reply = coordinator.play_exchange("e4")

    assert reply.san == "c5"
    assert ctx.session.move_history() == ["e4", "c5"]


def test_the_glitch_tier_is_recorded_as_the_strength_played():
    ctx = glitch_ctx(StubMover("c7c5"))
    TurnCoordinator(ctx).play_exchange("e4")
    assert ctx.session.difficulty == {"tier": GLITCH, "skill_level": None, "elo": None}
    assert pgn_headers(ctx)["Black"] == "Glitch (own moves)"


def test_other_tiers_still_play_stockfish_and_never_ask_the_mover():
    mover = StubMover()
    ctx = glitch_ctx(mover)
    ctx.settings.tier = "casual"
    coordinator = TurnCoordinator(ctx)

    _, reply = coordinator.play_exchange("e4")

    assert reply.san == "e5"
    assert mover.seen == []
    assert coordinator.sourced_for(reply).chooser == "stockfish"


# --- the fallback: Stockfish at beginner, marked -------------------------------


@pytest.mark.parametrize("kind", list(MoverFailureKind))
def test_a_failed_mover_falls_back_to_the_engine_and_says_so(kind):
    ctx = glitch_ctx(StubMover(kind))
    coordinator = TurnCoordinator(ctx)

    _, reply = coordinator.play_exchange("e4")

    assert reply.san == "e5"  # the engine's move, configured as beginner
    sourced = coordinator.sourced_for(reply)
    assert sourced.chooser == CHOOSER_FALLBACK
    assert sourced.fallback_reason == str(kind)
    assert sourced.mover["fallback"] is True
    assert sourced.mover["outcome"] == str(kind)
    # The PGN carries it on that move.
    assert ctx.session.move_notes == {
        1: f"Glitch's mover failed ({kind}); Stockfish played this move at beginner"
    }
    game = chess.pgn.read_game(io.StringIO(ctx.session.export_pgn()))
    moves = list(game.mainline())
    assert moves[1].comment.startswith("Glitch's mover failed")
    assert moves[0].comment == ""


def test_with_no_mover_at_all_the_fallback_is_still_marked():
    ctx = glitch_ctx(None)
    coordinator = TurnCoordinator(ctx)

    _, reply = coordinator.play_exchange("e4")

    sourced = coordinator.sourced_for(reply)
    assert sourced.chooser == CHOOSER_FALLBACK
    assert sourced.fallback_reason == NO_MOVER


def test_a_failed_mover_over_a_dead_engine_leaves_the_reply_owed():
    ctx = glitch_ctx(StubMover(MoverFailureKind.PROVIDER), engine=DeadEngine())
    coordinator = TurnCoordinator(ctx)
    coordinator.apply_player_move("e4")

    settled = coordinator.settle_owed_reply()

    assert settled.reply is None and settled.failure == "ValueError: engine died"
    assert coordinator.phase == TurnPhase.PLAYER_MOVE_APPLIED
    assert ctx.session.move_history() == ["e4"]


def test_the_glitch_source_falls_back_on_its_own():
    session = GameSession()
    session.submit_move("e4")
    sourced = GlitchSource(None, FakeEngine("e7e5")).choose(session)
    assert (sourced.uci, sourced.chooser) == ("e7e5", CHOOSER_FALLBACK)


# --- the mover's record waits for its turn's trace -------------------------------


def test_the_mover_record_belongs_to_the_turn_that_played_it():
    ctx = glitch_ctx(StubMover("c7c5"))
    coordinator = TurnCoordinator(ctx)
    before = ctx.session.fen()
    coordinator.play_exchange("e4")
    after = ctx.session.fen()

    sourced = coordinator.take_mover_record(ctx.session.game_id, before, after)

    assert sourced is not None and sourced.mover["san"] == "c5"
    assert len(sourced.calls) == 1
    # Taken once.
    assert coordinator.take_mover_record(ctx.session.game_id, before, after) is None


def test_a_mover_record_from_an_earlier_position_is_not_this_turns():
    ctx = glitch_ctx(StubMover("c7c5"))
    coordinator = TurnCoordinator(ctx)
    coordinator.play_exchange("e4")
    later = ctx.session.fen()

    assert coordinator.take_mover_record(ctx.session.game_id, later, later) is None


def test_a_mover_record_from_another_game_is_not_this_turns():
    ctx = glitch_ctx(StubMover("c7c5"))
    coordinator = TurnCoordinator(ctx)
    before = ctx.session.fen()
    coordinator.play_exchange("e4")

    assert coordinator.take_mover_record("other", before, ctx.session.fen()) is None


# --- choosing the tier ----------------------------------------------------------


def test_set_difficulty_refuses_glitch_with_no_mover():
    ctx = ToolContext(session=GameSession(), engine=FakeEngine())
    result = build_registry(ctx).dispatch("set_difficulty", {"tier": GLITCH})
    assert result["ok"] is False
    assert GLITCH_NEEDS_MOVER in result["error"]
    assert ctx.settings.tier == "casual"


def test_set_difficulty_takes_glitch_with_a_mover():
    engine = FakeEngine()
    ctx = ToolContext(session=GameSession(), engine=engine)
    ctx.glitch_mover = lambda: StubMover()
    result = build_registry(ctx).dispatch("set_difficulty", {"tier": GLITCH})
    assert result["ok"] is True
    assert ctx.settings.tier == GLITCH
    assert engine.tiers[-1] == GLITCH


def test_harder_from_maximum_does_not_step_into_glitch():
    ctx = ToolContext(session=GameSession(), engine=FakeEngine())
    ctx.glitch_mover = lambda: StubMover()
    ctx.settings.tier = "maximum"
    result = build_registry(ctx).dispatch("set_difficulty", {"step": "harder"})
    assert result["ok"] is False
    assert "already at maximum" in result["error"]


def test_a_step_from_glitch_is_refused_with_the_fix():
    ctx = glitch_ctx(StubMover())
    result = build_registry(ctx).dispatch("set_difficulty", {"step": "easier"})
    assert result["ok"] is False
    assert "not on the easier/harder ladder" in result["error"]
    assert result["retry"] == "different_args"


def test_the_difficulty_endpoint_refuses_glitch_with_no_mover():
    ctx = ToolContext(session=GameSession(), engine=FakeEngine())
    client = TestClient(create_app(ctx))
    response = client.post("/api/game/difficulty", json={"tier": GLITCH})
    assert response.status_code == 409
    assert GLITCH_NEEDS_MOVER in response.json()["detail"]
    assert ctx.settings.tier == "casual"


def test_the_difficulty_endpoint_takes_glitch_with_a_mover():
    ctx = ToolContext(session=GameSession(), engine=FakeEngine())
    ctx.glitch_mover = lambda: StubMover()
    client = TestClient(create_app(ctx))
    response = client.post("/api/game/difficulty", json={"tier": GLITCH})
    assert response.status_code == 200
    assert response.json()["tier"] == GLITCH


def test_a_saved_glitch_tier_resets_when_the_app_starts_with_no_mover(tmp_path):
    settings_file = tmp_path / SETTINGS_FILENAME
    settings_file.write_text('{"tier": "glitch"}')
    engine = FakeEngine()

    build_app(engine=engine, save_dir=tmp_path, brain=ScriptedBrain())

    assert '"tier": "casual"' in settings_file.read_text()
    assert engine.tiers[-1] == "casual"


def test_direct_mode_cannot_choose_glitch(tmp_path):
    client = TestClient(build_app(engine=FakeEngine(), agent_enabled=False))
    response = client.post("/api/game/difficulty", json={"tier": GLITCH})
    assert response.status_code == 409


# --- every route reports who chose the reply ---------------------------------------


def test_the_atomic_make_move_result_says_who_chose():
    ctx = glitch_ctx(StubMover(MoverFailureKind.DEADLINE))
    result = build_registry(ctx).dispatch("make_move", {"move": "e4"})
    assert result["engine_move"]["chosen_by"] == CHOOSER_FALLBACK
    assert result["engine_move"]["fallback_reason"] == "deadline"


def test_a_stockfish_tier_result_has_no_chooser_key():
    ctx = ToolContext(session=GameSession(), engine=FakeEngine())
    result = build_registry(ctx).dispatch("make_move", {"move": "e4"})
    assert "chosen_by" not in result["engine_move"]


def test_a_restore_that_settles_says_who_chose():
    ctx = glitch_ctx(StubMover("d2d4"))
    result = build_registry(ctx).dispatch("new_game", {"player_color": "black"})
    assert result["engine_move"]["san"] == "d4"
    assert result["engine_move"]["chosen_by"] == CHOOSER_GLITCH


def test_a_dragged_move_in_the_tier_is_the_movers_and_is_traced(tmp_path):
    trace_path = tmp_path / "turns.jsonl"
    ctx = glitch_ctx(StubMover("c7c5"))
    app, _ = scripted_app(ctx, tracer=JsonlTracer(trace_path))
    client = TestClient(app)

    body = client.post("/api/game/move", json={"move": "e2e4"}).json()

    assert body["engine_move"]["san"] == "c5"
    assert body["engine_move"]["chosen_by"] == CHOOSER_GLITCH
    (record,) = [json.loads(line) for line in trace_path.read_text().splitlines()]
    assert record["mover"]["san"] == "c5"
    assert [call["phase"] for call in record["calls"]][-1] == "mover"


# --- move notes in the save ---------------------------------------------------------


def test_move_notes_survive_a_save_and_are_cut_by_undo():
    session = GameSession()
    session.submit_move("e4")
    session.submit_move("e5")
    session.note_last_move("not his own")

    restored = GameSession.from_dict(session.to_dict())
    assert restored.move_notes == {1: "not his own"}

    restored.undo(1)
    assert restored.move_notes == {}


def test_a_new_game_clears_the_notes():
    session = GameSession()
    session.submit_move("e4")
    session.note_last_move("x")
    session.new_game()
    assert session.move_notes == {}


def test_a_save_from_before_notes_loads():
    session = GameSession()
    session.submit_move("e4")
    data = session.to_dict()
    del data["notes"]
    assert GameSession.from_dict(data).move_notes == {}


@pytest.mark.parametrize("notes", [["x"], {"5": "x"}, {"a": "x"}, {"0": 3}])
def test_a_save_with_bad_notes_is_refused(notes):
    session = GameSession()
    session.submit_move("e4")
    data = session.to_dict()
    data["notes"] = notes
    with pytest.raises(ValueError):
        GameSession.from_dict(data)


def test_noting_with_no_move_is_refused():
    with pytest.raises(ValueError):
        GameSession().note_last_move("x")


def test_the_pgn_header_for_a_game_played_in_the_tier():
    ctx = ToolContext(session=GameSession(), engine=FakeEngine())
    ctx.settings.tier = GLITCH
    assert pgn_headers(ctx)["Black"] == "Glitch (own moves)"
    ctx.settings.tier = "casual"
    assert pgn_headers(ctx)["Black"] == "Glitch (Stockfish, casual)"


def test_settings_accept_the_glitch_tier(tmp_path):
    (tmp_path / SETTINGS_FILENAME).write_text('{"tier": "glitch"}')
    ctx = ToolContext(session=GameSession(), save_dir=tmp_path)
    assert ctx.settings.tier == GLITCH


# --- MCP: the mover is the one model call the server makes -----------------------


def test_the_mcp_server_has_no_mover_without_a_model(monkeypatch):
    from chessapp import mcp_server

    monkeypatch.delenv("LLAMACPP_MODEL", raising=False)
    monkeypatch.delenv("CHESSAPP_MOVER_MODEL", raising=False)
    monkeypatch.setattr(mcp_server, "_engine_from_env", lambda: FakeEngine())
    ctx = mcp_server.build_mcp_context()
    assert ctx.mover_available() is False
    result = build_registry(ctx).dispatch("set_difficulty", {"tier": GLITCH})
    assert result["ok"] is False


def test_the_mcp_server_gets_a_mover_on_the_named_model(monkeypatch):
    from chessapp import mcp_server

    monkeypatch.setenv("LLAMACPP_MODEL", "gemma-4-26b-a4b")
    monkeypatch.delenv("CHESSAPP_MOVER_MODEL", raising=False)
    monkeypatch.delenv("CHESSAPP_AGENT", raising=False)
    monkeypatch.setattr(mcp_server, "_engine_from_env", lambda: FakeEngine())
    ctx = mcp_server.build_mcp_context()
    assert ctx.mover_available() is True
    assert ctx.glitch_mover().model == "gemma-4-26b-a4b"


def test_the_mcp_server_has_no_mover_with_the_agent_off(monkeypatch):
    from chessapp import mcp_server

    monkeypatch.setenv("LLAMACPP_MODEL", "gemma-4-26b-a4b")
    monkeypatch.setenv("CHESSAPP_AGENT", "off")
    assert mcp_server._mover_from_env() is None
