"""The ledger: what happened in the game, as code saw it (#372).

Pure code, heavily tested: the story of the game is written from these events
and scored against them, so a missed takeback or a move credited to the wrong
side here is a false fact everywhere downstream.
"""

import json

import pytest
from fastapi.testclient import TestClient

from chessapp.api import create_app
from chessapp.app import build_app
from chessapp.coordinator import TurnCoordinator
from chessapp.game import GameSession
from chessapp.ledger import (
    DRAW_OFFER,
    GAME_END,
    MOVE,
    NEW_GAME,
    OFFER,
    RESUMED,
    TAKEBACK,
    Ledger,
    LedgerEvent,
    difficulty_label,
    followed_settings,
    from_session,
    render_record,
)
from chessapp.tools import (
    LIVE_CHECKPOINT_FILENAME,
    Settings,
    ToolContext,
    build_registry,
    live_checkpoint,
    restore_live_checkpoint,
)
from fakes import FakeEngine
from test_draw_offer import DEAD_DRAWN_ROOK_ENDGAME, FLAT_MIDDLEGAME, tooled

SETTINGS = Settings().snapshot()


def _kinds(ledger: Ledger) -> list[str]:
    return [e.kind for e in ledger.current()]


def _played(session: GameSession, *sans: str) -> GameSession:
    for san in sans:
        assert session.submit_move(san).legal, san
    return session


def _observed(*sans: str, session: GameSession | None = None) -> Ledger:
    """A ledger that watched each move land, one observation per move."""
    session = session or GameSession()
    ledger = Ledger()
    ledger.observe(session, SETTINGS)
    for san in sans:
        _played(session, san)
        ledger.observe(session, SETTINGS)
    return ledger


# --- moves ------------------------------------------------------------------------


def test_a_fresh_game_opens_with_one_new_game_event():
    ledger = _observed()
    [event] = ledger.current()
    assert event.kind == NEW_GAME
    assert event.ply == 0
    assert event.details == {"player_color": "white"}


def test_moves_are_credited_numbered_and_keyed_to_the_move_list():
    ledger = _observed("e4", "e5", "Nf3")
    moves = ledger.moves()
    assert [m.details["san"] for m in moves] == ["e4", "e5", "Nf3"]
    assert [m.details["by"] for m in moves] == ["player", "engine", "player"]
    assert [m.details["color"] for m in moves] == ["white", "black", "white"]
    assert [m.details["move_number"] for m in moves] == [1, 1, 2]
    assert [m.ply for m in moves] == [1, 2, 3]
    assert [m.details["uci"] for m in moves] == ["e2e4", "e7e5", "g1f3"]


def test_the_side_is_the_players_colour_not_white():
    session = GameSession(player_color="black")
    ledger = _observed("e4", "e5", session=session)
    assert [m.details["by"] for m in ledger.moves()] == ["engine", "player"]


def test_several_moves_between_observations_are_each_recorded_in_order():
    session = GameSession()
    ledger = Ledger()
    ledger.observe(session, SETTINGS)
    _played(session, "e4", "e5", "Nf3", "Nc6")
    ledger.observe(session, SETTINGS)
    assert [m.details["san"] for m in ledger.moves()] == ["e4", "e5", "Nf3", "Nc6"]


def test_observing_an_unchanged_game_adds_nothing():
    session = _played(GameSession(), "e4")
    ledger = Ledger()
    ledger.observe(session, SETTINGS)
    before = ledger.next_seq
    ledger.observe(session, SETTINGS)
    ledger.observe(session, dict(SETTINGS))
    assert ledger.next_seq == before


def test_a_capture_names_its_victim_and_the_material_follows():
    ledger = _observed("e4", "d5", "exd5")
    capture = ledger.moves()[-1]
    assert capture.details["capture"] == "pawn"
    assert capture.details["material"] == 1
    assert [m.details["san"] for m in ledger.captures()] == ["exd5"]
    assert ledger.material_by_ply() == {1: 0, 2: 0, 3: 1}


def test_material_is_the_players_side():
    session = GameSession(player_color="black")
    ledger = _observed("e4", "d5", "exd5", session=session)
    assert ledger.moves()[-1].details["material"] == -1


def test_en_passant_takes_a_pawn():
    ledger = _observed("e4", "a6", "e5", "d5", "exd6")
    assert ledger.moves()[-1].details["capture"] == "pawn"


def test_a_quiet_move_takes_nothing_and_check_is_recorded():
    ledger = _observed("e4", "f5", "Qh5+")
    check = ledger.moves()[-1]
    assert check.details["capture"] is None
    assert check.details["check"] is True
    assert ledger.moves()[0].details["check"] is False


def test_castling_and_promotion_keep_their_san():
    session = GameSession(fen="4k3/1P6/8/8/8/8/8/4K2R w K - 0 1")
    ledger = _observed("O-O", "Kd7", "b8=Q", session=session)
    sans = [m.details["san"] for m in ledger.moves()]
    assert sans == ["O-O", "Kd7", "b8=Q"]
    assert ledger.moves()[-1].details["material"] == 5 + 9


def test_a_game_rooted_on_a_fen_records_its_root_and_starts_with_black():
    fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
    session = GameSession(fen=fen)
    ledger = _observed("e5", "Nf3", session=session)
    assert ledger.current()[0].details["root_fen"] == fen
    first, second = ledger.moves()
    assert (first.details["color"], first.details["by"]) == ("black", "engine")
    assert first.details["move_number"] == 1
    assert (second.details["color"], second.details["move_number"]) == ("white", 2)


# --- takebacks --------------------------------------------------------------------


def test_a_takeback_is_an_event_and_the_moves_stay_in_the_record():
    session = GameSession()
    ledger = _observed("e4", "e5", "Qh5", session=session)
    session.undo(1)
    ledger.observe(session, SETTINGS)
    takeback = ledger.current()[-1]
    assert takeback.kind == TAKEBACK
    assert takeback.details == {"undone": ["Qh5"], "plies": 1}
    assert takeback.ply == 2
    assert [m.details["san"] for m in ledger.moves()] == ["e4", "e5", "Qh5"]
    assert [m.details["san"] for m in ledger.line()] == ["e4", "e5"]


def test_a_two_ply_takeback_lists_the_last_move_first():
    session = GameSession()
    ledger = _observed("e4", "e5", "Nf3", "Nc6", session=session)
    session.undo(2)
    ledger.observe(session, SETTINGS)
    assert ledger.current()[-1].details == {"undone": ["Nc6", "Nf3"], "plies": 2}


def test_undo_then_play_between_two_observations_is_a_takeback_then_a_move():
    session = GameSession()
    ledger = _observed("e4", "e5", "Qh5", session=session)
    session.undo(1)
    _played(session, "Nf3")
    ledger.observe(session, SETTINGS)
    last_two = ledger.current()[-2:]
    assert [e.kind for e in last_two] == [TAKEBACK, MOVE]
    assert last_two[0].details["undone"] == ["Qh5"]
    assert last_two[1].details["san"] == "Nf3"


def test_the_same_move_replayed_after_a_takeback_is_seen_when_observed_between():
    session = GameSession()
    ledger = _observed("e4", "e5", session=session)
    session.undo(1)
    ledger.observe(session, SETTINGS)
    _played(session, "e5")
    ledger.observe(session, SETTINGS)
    assert [e.kind for e in ledger.current()[-3:]] == [MOVE, TAKEBACK, MOVE]


def test_the_standing_line_always_matches_the_session():
    session = GameSession()
    ledger = _observed("e4", "e5", "Nf3", "Nc6", "Bc4", session=session)
    for plies, then in ((2, ("d6",)), (1, ()), (3, ("d4", "d5"))):
        session.undo(plies)
        _played(session, *then)
        ledger.observe(session, SETTINGS)
        line = [m.details["san"] for m in ledger.line()]
        assert line == session.move_history()


# --- settings ---------------------------------------------------------------------


def test_difficulty_is_one_value_however_it_was_set():
    assert difficulty_label({"tier": "hard"}) == "hard"
    assert difficulty_label({"tier": None, "skill_level": 7}) == "skill 7"
    assert difficulty_label({"tier": None, "elo": 1500}) == "elo 1500"
    assert followed_settings(SETTINGS)["voice"] == "off"


def test_a_setting_change_is_an_event_with_both_values():
    session = GameSession()
    settings = Settings()
    ledger = Ledger()
    ledger.observe(session, settings.snapshot())
    settings.verbosity = "high"
    settings.voice_output = True
    ledger.observe(session, settings.snapshot())
    changes = ledger.setting_changes()
    seen = [
        (e.details["name"], e.details["before"], e.details["after"]) for e in changes
    ]
    assert seen == [
        ("verbosity", "normal", "high"),
        ("voice", "off", "on"),
    ]


def test_the_first_observation_records_no_setting_change():
    ledger = Ledger()
    ledger.observe(GameSession(), {**SETTINGS, "verbosity": "low"})
    assert ledger.setting_changes() == []


def test_a_setting_tool_is_recorded_through_dispatch():
    ctx = ToolContext(session=GameSession(), engine=FakeEngine())
    registry = build_registry(ctx, TurnCoordinator(ctx), atomic_exchange=False)
    registry.dispatch("set_verbosity", {"verbosity": "low"})
    [change] = ctx.ledger.setting_changes()
    assert change.details == {"name": "verbosity", "before": "normal", "after": "low"}


def test_the_ui_settings_endpoints_are_recorded():
    ctx = ToolContext(session=GameSession(), engine=FakeEngine())
    client = TestClient(create_app(ctx))
    client.post("/api/settings/voice", json={"enabled": True})
    client.post("/api/game/difficulty", json={"skill_level": 5})
    changes = [
        (e.details["name"], e.details["after"]) for e in ctx.ledger.setting_changes()
    ]
    assert changes == [("voice", "on"), ("difficulty", "skill 5")]


# --- moves through the app --------------------------------------------------------


def test_the_engines_reply_and_the_undo_button_are_recorded(tmp_path):
    client = TestClient(
        build_app(agent_enabled=False, engine=FakeEngine(), save_dir=tmp_path)
    )
    client.post("/api/game/move", json={"move": "e4"})
    client.post("/api/game/undo", json={})
    events = json.loads((tmp_path / LIVE_CHECKPOINT_FILENAME).read_text())["ledger"]
    kinds = [e["kind"] for e in events]
    assert kinds == [NEW_GAME, MOVE, MOVE, TAKEBACK]
    assert [e["by"] for e in events if e["kind"] == MOVE] == ["player", "engine"]
    assert events[-1]["undone"] == ["e5", "e4"]


# --- draw offers and endings ------------------------------------------------------


def test_a_declined_offer_is_recorded_though_nothing_moved():
    ctx, _, registry = tooled(FLAT_MIDDLEGAME)
    registry.dispatch("offer_draw", {})
    offer = ctx.ledger.current()[-1]
    assert offer.kind == DRAW_OFFER
    assert offer.details == {"accepted": False, "reason": "not_an_endgame"}


def test_an_accepted_offer_reads_before_the_ending_it_caused():
    ctx, _, registry = tooled(DEAD_DRAWN_ROOK_ENDGAME)
    registry.dispatch("offer_draw", {})
    offer, end = ctx.ledger.current()[-2:]
    assert (offer.kind, offer.details["accepted"]) == (DRAW_OFFER, True)
    assert end.kind == GAME_END
    assert end.details == {
        "termination": "agreement",
        "result": "1/2-1/2",
        "winner": None,
    }


def test_checkmate_and_resignation_are_endings_from_the_players_side():
    session = GameSession()
    ledger = _observed("f3", "e5", "g4", "Qh4#", session=session)
    end = ledger.current()[-1]
    assert end.kind == GAME_END
    assert end.details["termination"] == "checkmate"
    assert end.details["winner"] == "opponent"

    other = GameSession()
    watched = _observed("e4", session=other)
    other.resign("black")
    watched.observe(other, SETTINGS)
    assert watched.current()[-1].details["winner"] == "player"
    watched.observe(other, SETTINGS)
    assert [e.kind for e in watched.current()].count(GAME_END) == 1


def test_a_takeback_out_of_mate_reopens_the_game():
    session = GameSession()
    ledger = _observed("f3", "e5", "g4", "Qh4#", session=session)
    session.undo(1)
    ledger.observe(session, SETTINGS)
    _played(session, "Qe7")
    ledger.observe(session, SETTINGS)
    assert _kinds(ledger)[-3:] == [GAME_END, TAKEBACK, MOVE]


# --- games ------------------------------------------------------------------------


def test_a_new_game_starts_a_new_game_and_keeps_the_last_ones_events():
    session = GameSession()
    ledger = _observed("e4", "e5", session=session)
    old_id = session.game_id
    session.new_game(player_color="black")
    ledger.observe(session, SETTINGS)
    [opening] = ledger.current()
    assert opening.kind == NEW_GAME
    assert opening.details == {"player_color": "black"}
    assert len(ledger.moves(old_id)) == 2
    assert [e.kind for e in ledger.since(0)][:3] == [NEW_GAME, MOVE, MOVE]


def test_a_resumed_save_is_a_resume_with_its_line_replayed(tmp_path):
    ctx = ToolContext(session=GameSession(), engine=FakeEngine(), save_dir=tmp_path)
    registry = build_registry(ctx, TurnCoordinator(ctx), atomic_exchange=False)
    _played(ctx.session, "e4", "e5", "Nf3")
    registry.dispatch("save_game", {"name": "keep"})
    # A fresh board, so the resume has no game to ask about throwing away.
    ctx.session = GameSession()
    ctx.observe_ledger()
    assert registry.dispatch("resume_game", {"name": "keep"})["ok"] is True
    events = ctx.ledger.current()
    assert events[0].kind == RESUMED
    assert events[0].details["name"] == "keep"
    restored = [e for e in events if e.kind == MOVE]
    assert [e.details["san"] for e in restored] == ["e4", "e5", "Nf3"]
    assert all(e.details["restored"] for e in restored)


def test_since_is_a_cursor_across_games():
    session = GameSession()
    ledger = _observed("e4", session=session)
    cursor = ledger.next_seq
    _played(session, "e5")
    ledger.observe(session, SETTINGS)
    session.new_game()
    ledger.observe(session, SETTINGS)
    assert [e.kind for e in ledger.since(cursor)] == [MOVE, NEW_GAME]


def test_earlier_games_are_bounded_in_memory(monkeypatch):
    monkeypatch.setattr("chessapp.ledger.EARLIER_GAMES_KEPT", 3)
    session = GameSession()
    ledger = _observed("e4", "e5", "Nf3", "Nc6", session=session)
    session.new_game()
    ledger.observe(session, SETTINGS)
    assert len(ledger.events()) == 3 + 1


# --- persistence ------------------------------------------------------------------


def test_events_round_trip_through_json():
    session = GameSession()
    ledger = _observed("e4", "d5", "exd5", session=session)
    session.undo(1)
    ledger.observe(session, SETTINGS)
    data = json.loads(json.dumps(ledger.to_dict()))
    restored = Ledger.restore(data, session, SETTINGS)
    assert [e.to_dict() for e in restored.current()] == data
    _played(session, "Nf3")
    restored.observe(session, SETTINGS)
    assert restored.current()[-1].details["san"] == "Nf3"
    assert restored.current()[-1].seq == data[-1]["seq"] + 1


def test_a_restored_ledger_records_nothing_new_on_its_first_observation():
    session = GameSession()
    ledger = _observed("f3", "e5", "g4", "Qh4#", session=session)
    restored = Ledger.restore(ledger.to_dict(), session, SETTINGS)
    before = restored.next_seq
    restored.observe(session, SETTINGS)
    assert restored.next_seq == before


def test_a_ledger_that_does_not_replay_to_the_session_is_rebuilt():
    session = _played(GameSession(), "e4", "e5")
    other = _observed("d4")
    rebuilt = Ledger.restore(other.to_dict(), session, SETTINGS)
    assert [e.kind for e in rebuilt.current()] == [NEW_GAME, MOVE, MOVE]
    assert all(m.details["restored"] for m in rebuilt.moves())


@pytest.mark.parametrize(
    "data",
    [
        None,
        "not a list",
        [{"seq": 0, "kind": "made_up", "game_id": "x", "ply": 0}],
        [{"seq": -1, "kind": "new_game", "game_id": "x", "ply": 0}],
        [{"seq": 0, "kind": "move", "game_id": "x", "ply": 1}],
    ],
)
def test_a_tampered_or_missing_ledger_is_rebuilt_from_the_session(data):
    session = _played(GameSession(), "e4")
    rebuilt = Ledger.restore(data, session, SETTINGS)
    assert [e.kind for e in rebuilt.current()] == [NEW_GAME, MOVE]


def test_from_session_is_moves_only():
    session = _played(GameSession(), "e4", "e5")
    ledger = from_session(session, SETTINGS)
    assert [e.kind for e in ledger.current()] == [NEW_GAME, MOVE, MOVE]


def test_event_parsing_rejects_a_non_object():
    with pytest.raises(ValueError):
        LedgerEvent.from_dict(["seq", 0])


def test_the_live_checkpoint_carries_the_ledger_and_a_restart_keeps_it(tmp_path):
    ctx = ToolContext(session=GameSession(), engine=FakeEngine(), save_dir=tmp_path)
    registry = build_registry(ctx, TurnCoordinator(ctx), atomic_exchange=False)
    _played(ctx.session, "e4", "e5", "Qh5")
    ctx.observe_ledger()
    registry.dispatch("undo", {})
    checkpoint = live_checkpoint(ctx)
    assert [e["kind"] for e in checkpoint["ledger"]][-1] == TAKEBACK
    (tmp_path / LIVE_CHECKPOINT_FILENAME).write_text(json.dumps(checkpoint))

    fresh = ToolContext(session=GameSession(), engine=FakeEngine(), save_dir=tmp_path)
    assert restore_live_checkpoint(fresh)
    kinds = [e.kind for e in fresh.ledger.current()]
    assert TAKEBACK in kinds, "the takeback survived the restart"
    assert [m.details["san"] for m in fresh.ledger.line()] == (
        fresh.session.move_history()
    )


def test_a_checkpoint_from_before_the_ledger_restores_moves_only(tmp_path):
    session = _played(GameSession(), "e4", "e5")
    (tmp_path / LIVE_CHECKPOINT_FILENAME).write_text(
        json.dumps(
            {
                "checkpoint": 1,
                "board_version": 2,
                "session": session.to_dict(),
                "transcript": [],
            }
        )
    )
    ctx = ToolContext(session=GameSession(), save_dir=tmp_path)
    assert restore_live_checkpoint(ctx)
    assert [e.kind for e in ctx.ledger.current()] == [NEW_GAME, MOVE, MOVE]


def test_a_ledger_that_cannot_observe_never_fails_the_call(monkeypatch):
    ctx = ToolContext(session=GameSession(), engine=FakeEngine())
    registry = build_registry(ctx, TurnCoordinator(ctx), atomic_exchange=False)

    def boom(*_args, **_kwargs):
        raise RuntimeError("ledger broke")

    monkeypatch.setattr(ctx.ledger, "observe", boom)
    result = registry.dispatch("make_move", {"move": "e4"})
    assert result["legal"] is True


# --- what was offered (#372) ------------------------------------------------------


def test_a_hint_a_question_and_a_refusals_alternatives_are_offers():
    engine = FakeEngine()
    ctx = ToolContext(session=GameSession(), engine=engine)
    registry = build_registry(ctx, TurnCoordinator(ctx), atomic_exchange=False)
    ctx.note_offer(
        "get_best_moves", {"ok": True, "moves": [{"san": "e4"}, {"san": "d4"}]}
    )
    registry.dispatch("ask_player", {"piece": "knight"})
    registry.dispatch("make_move", {"move": "Ke2"})
    offers = [e.details for e in ctx.ledger.current() if e.kind == OFFER]
    assert offers[0] == {"source": "hint", "moves": ["e4", "d4"]}
    assert offers[1]["source"] == "question"
    assert sorted(offers[1]["moves"]) == ["Na3", "Nc3", "Nf3", "Nh3"]
    assert offers[2]["source"] == "alternatives" and offers[2]["moves"]


def test_a_result_that_offered_nothing_records_nothing():
    ctx = ToolContext(session=GameSession(), engine=FakeEngine())
    registry = build_registry(ctx, TurnCoordinator(ctx), atomic_exchange=False)
    registry.dispatch("make_move", {"move": "e4"})
    ctx.note_offer("get_best_moves", {"ok": False, "error": "no engine"})
    assert not [e for e in ctx.ledger.current() if e.kind == OFFER]


# --- the record -------------------------------------------------------------------


def test_the_record_keys_what_the_move_list_cannot_show_to_it():
    session = GameSession()
    ledger = _observed("e4", "e5", "Qh5", session=session)
    session.undo(1)
    ledger.observe(session, SETTINGS)
    low = {**SETTINGS, "verbosity": "low"}
    ledger.observe(session, low)
    ledger.note_offer(
        session, low, "get_best_moves", {"ok": True, "moves": [{"san": "Nf3"}]}
    )
    _played(session, "Nf3", "Nc6")
    ledger.observe(session, low)
    assert render_record(ledger.current()) == [
        "after 2. Qh5: took back 2. Qh5.",
        "after 1... e5: verbosity changed from normal to low.",
        "after 1... e5: a hint offered Nf3.",
    ]


def test_the_record_ends_a_game_and_offers_a_draw_in_words():
    ctx, _, registry = tooled(DEAD_DRAWN_ROOK_ENDGAME)
    registry.dispatch("offer_draw", {})
    lines = render_record(ctx.ledger.current())
    assert lines[-2].endswith("the player offered a draw; the engine accepted.")
    assert lines[-1].endswith("the game ended by agreement; a draw (1/2-1/2).")


def test_the_record_names_a_resume_and_lists_no_moves():
    session = _played(GameSession(), "e4", "e5")
    ledger = Ledger()
    ledger.expect_resume("keep")
    ledger.observe(session, SETTINGS)
    assert render_record(ledger.current()) == [
        "The saved game 'keep' was resumed; the player has white."
    ]


def test_a_long_record_keeps_the_newest_and_says_how_many_went():
    session = GameSession()
    ledger = _observed(session=session)
    for level in ("low", "high") * 5:
        ledger.observe(session, {**SETTINGS, "verbosity": level})
    lines = render_record(ledger.current(), max_lines=4)
    assert lines[0] == "(6 earlier events not listed)"
    assert len(lines) == 5 and lines[-1].endswith("to high.")
