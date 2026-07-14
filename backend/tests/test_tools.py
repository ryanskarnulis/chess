"""Tool layer: registry + validated dispatch + read tools.

This is the boundary the LLM talks through. Dispatch must be un-crashable:
unknown tools, malformed args, and domain errors all come back as
`{"ok": False, "error": ...}` — never an exception, never corrupted state.
Analysis tools need a live stockfish and skip without one (CI installs it).
"""

import json
import shutil

import pytest

from chessapp.engine import DEFAULT_TIER
from chessapp.game import GameSession
from chessapp.tools import (
    BOARD_STATE_TOOLS,
    Settings,
    Tool,
    ToolContext,
    ToolRegistry,
    build_registry,
    confirm_pending,
    saved_game_names,
)
from fakes import FakeEngine

requires_stockfish = pytest.mark.skipif(
    shutil.which("stockfish") is None, reason="stockfish binary not installed"
)

# White to move, Qxf7# available (scholar's mate pattern).
WHITE_MATE_IN_1 = "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 0 1"


@pytest.fixture
def session():
    return GameSession()


@pytest.fixture
def registry(session):
    return build_registry(ToolContext(session=session))


@pytest.fixture(scope="module")
def live_engine():
    if shutil.which("stockfish") is None:
        pytest.skip("stockfish binary not installed")
    from chessapp.engine import EnginePlayer

    with EnginePlayer() as player:
        yield player


# --- registry ---------------------------------------------------------------


def test_registry_lists_all_read_tools(registry):
    names = {d["function"]["name"] for d in registry.definitions()}
    assert names >= {
        "get_board_state",
        "get_legal_moves",
        "get_move_history",
        "get_captured_pieces",
        "evaluate_position",
        "get_best_moves",
    }


def test_definitions_can_exclude_tools(registry):
    """The brain is offered a subset; the registry still holds everything.

    `BOARD_STATE_TOOLS` answer with strict subsets of the board state the brain
    is handed in its prompt every turn, so offering them to the brain only buys
    a wasted round trip out of a 4-iteration budget. Other callers (MCP, the
    delegate wire) have no such injection, so the tools stay registered and
    runnable — only the brain's *offer* narrows.
    """
    names = {
        d["function"]["name"] for d in registry.definitions(exclude=BOARD_STATE_TOOLS)
    }
    assert not names & set(BOARD_STATE_TOOLS)
    assert {"make_move", "undo", "evaluate_position"} <= names


def test_excluded_tools_are_still_dispatchable(registry):
    assert registry.dispatch("get_legal_moves", {})["ok"] is True


def test_definitions_are_openai_style_and_json_serializable(registry):
    definitions = registry.definitions()
    json.dumps(definitions)  # must not raise
    for definition in definitions:
        assert definition["type"] == "function"
        fn = definition["function"]
        assert fn["name"]
        assert fn["description"]
        assert fn["parameters"]["type"] == "object"
        # Closed schemas: the LLM cannot smuggle extra args past validation.
        assert fn["parameters"]["additionalProperties"] is False


def test_register_duplicate_name_raises():
    registry = ToolRegistry()
    tool = Tool(
        name="t",
        description="d",
        parameters={"type": "object", "properties": {}, "additionalProperties": False},
        handler=lambda: {"ok": True},
    )
    registry.register(tool)
    with pytest.raises(ValueError):
        registry.register(tool)


# --- dispatch boundary ------------------------------------------------------


def test_dispatch_unknown_tool_is_error_not_exception(registry):
    result = registry.dispatch("no_such_tool", {})
    assert result["ok"] is False
    assert "no_such_tool" in result["error"]


def test_dispatch_rejects_non_dict_args(registry):
    result = registry.dispatch("get_board_state", "not a dict")
    assert result["ok"] is False


def test_dispatch_rejects_extra_properties(registry):
    result = registry.dispatch("get_board_state", {"bogus": 1})
    assert result["ok"] is False


def test_dispatch_rejects_wrong_arg_type(registry):
    result = registry.dispatch("get_best_moves", {"n": "three"})
    assert result["ok"] is False


def test_dispatch_rejects_out_of_range_args(registry):
    assert registry.dispatch("get_best_moves", {"n": 0})["ok"] is False
    assert registry.dispatch("get_best_moves", {"n": 11})["ok"] is False


def test_legal_moves_after_game_over_is_empty(registry, session):
    session.resign("white")
    result = registry.dispatch("get_legal_moves", {})
    assert result["ok"] is True
    assert result["moves"] == []


def test_all_dispatch_results_are_json_serializable(registry):
    json.dumps(registry.dispatch("get_board_state", {}))
    json.dumps(registry.dispatch("get_legal_moves", {}))
    json.dumps(registry.dispatch("nope", {}))
    json.dumps(registry.dispatch("get_best_moves", {"n": -1}))


def test_dispatch_never_mutates_session(registry, session):
    before = session.fen()
    registry.dispatch("get_board_state", {})
    registry.dispatch("get_legal_moves", {})
    registry.dispatch("get_move_history", {})
    registry.dispatch("get_captured_pieces", {})
    registry.dispatch("unknown", {"x": 1})
    assert session.fen() == before
    assert session.move_history() == []


# --- read tools -------------------------------------------------------------


def test_get_board_state_fresh_game(registry, session):
    result = registry.dispatch("get_board_state", {})
    assert result["ok"] is True
    assert result["fen"] == session.fen()
    assert result["turn"] == "white"
    assert result["game_over"] is False
    assert result["outcome"] is None


def test_get_board_state_after_checkmate(session):
    for move in ["e4", "e5", "Bc4", "Nc6", "Qf3", "d6", "Qxf7#"]:
        assert session.submit_move(move).legal
    registry = build_registry(ToolContext(session=session))
    result = registry.dispatch("get_board_state", {})
    assert result["game_over"] is True
    assert result["outcome"] == {
        "termination": "checkmate",
        "winner": "white",
        "result": "1-0",
    }


def test_get_legal_moves_start_position(registry):
    result = registry.dispatch("get_legal_moves", {})
    assert result["ok"] is True
    assert len(result["moves"]) == 20
    assert "e4" in result["moves"]
    assert "Nf3" in result["moves"]


def test_get_move_history_and_captures(session):
    for move in ["e4", "d5", "exd5", "Qxd5"]:
        assert session.submit_move(move).legal
    registry = build_registry(ToolContext(session=session))
    history = registry.dispatch("get_move_history", {})
    assert history["ok"] is True
    assert history["moves"] == ["e4", "d5", "exd5", "Qxd5"]
    captures = registry.dispatch("get_captured_pieces", {})
    assert captures["ok"] is True
    assert captures["white"] == ["p"]
    assert captures["black"] == ["p"]


# --- write tools ------------------------------------------------------------


def test_registry_lists_all_write_tools(registry):
    names = {d["function"]["name"] for d in registry.definitions()}
    assert names >= {
        "make_move",
        "undo",
        "new_game",
        "resign",
        "save_game",
        "resume_game",
        "export_pgn",
    }


def test_make_move_legal(registry, session):
    result = registry.dispatch("make_move", {"move": "e4"})
    assert result["ok"] is True
    assert result["legal"] is True
    assert result["san"] == "e4"
    assert result["uci"] == "e2e4"
    assert result["game_over"] is False
    assert result["fen"] == session.fen()
    assert result["turn"] == "black"


def test_make_move_illegal_is_ok_result_with_legal_false(registry, session):
    before = session.fen()
    result = registry.dispatch("make_move", {"move": "e5"})
    assert result["ok"] is True
    assert result["legal"] is False
    assert result["reason"]
    assert session.fen() == before


def test_make_move_requires_move_arg(registry):
    assert registry.dispatch("make_move", {})["ok"] is False
    assert registry.dispatch("make_move", {"move": 4})["ok"] is False


def test_make_move_with_engine_triggers_reply(session):
    """The agent path mirrors the UI path: a legal player move gets the
    engine's reply in the same tool call, so texting 'e4' never leaves the
    player to move for both sides."""
    registry = build_registry(ToolContext(session=session, engine=FakeEngine()))
    result = registry.dispatch("make_move", {"move": "e4"})
    assert result["legal"] is True
    assert result["engine_move"] == {"san": "e5", "uci": "e7e5"}
    assert result["turn"] == "white"
    assert session.move_history() == ["e4", "e5"]
    assert result["fen"] == session.fen()


def test_make_move_reply_never_detours_through_multipv(session):
    # The reply is the engine's own move at the configured strength, never a
    # MultiPV detour around the difficulty (a personality move-bias layer was
    # tried in 2026-07 and removed for exactly this).
    engine = FakeEngine()
    ctx = ToolContext(session=session, engine=engine)
    registry = build_registry(ctx)
    result = registry.dispatch("make_move", {"move": "e4"})
    assert result["engine_move"]["uci"] == "e7e5"
    assert engine.multipv_requests == []


def test_make_move_illegal_gets_no_engine_reply(session):
    registry = build_registry(ToolContext(session=session, engine=FakeEngine()))
    result = registry.dispatch("make_move", {"move": "e5"})
    assert result["legal"] is False
    assert "engine_move" not in result
    assert session.move_history() == []


def test_make_move_no_engine_reply_once_game_is_over():
    class MustNotPlay:
        def play_move(self, session):
            raise AssertionError("engine must not reply to a game-ending move")

    session = GameSession(WHITE_MATE_IN_1)
    registry = build_registry(ToolContext(session=session, engine=MustNotPlay()))
    result = registry.dispatch("make_move", {"move": "Qxf7#"})
    assert result["game_over"] is True
    assert result["engine_move"] is None


def test_make_move_without_engine_has_no_reply(registry):
    result = registry.dispatch("make_move", {"move": "e4"})
    assert result["engine_move"] is None
    assert result["turn"] == "black"


def test_make_move_reports_checkmate(registry):
    for move in ["f3", "e5", "g4"]:
        assert registry.dispatch("make_move", {"move": move})["legal"]
    result = registry.dispatch("make_move", {"move": "Qh4"})
    assert result["legal"] is True
    assert result["game_over"] is True


def test_undo_reverts_last_ply(registry, session):
    registry.dispatch("make_move", {"move": "e4"})
    result = registry.dispatch("undo", {})
    assert result["ok"] is True
    assert result["undone"] == ["e4"]
    assert session.move_history() == []


def test_undo_two_plies_for_engine_pair(registry, session):
    registry.dispatch("make_move", {"move": "e4"})
    registry.dispatch("make_move", {"move": "e5"})
    result = registry.dispatch("undo", {"plies": 2})
    assert result["ok"] is True
    assert result["undone"] == ["e5", "e4"]
    assert session.move_history() == []


def test_undo_defaults_to_the_whole_exchange_vs_engine(session):
    """A bare `undo()` vs the engine takes back the pair, not the lone reply.

    The pairing rule is the REST endpoint's, and it belongs to the caller of
    `GameSession.undo` — not to the model. Popping one ply here would leave the
    player's move on the board with the engine to move and nothing to move it.
    """
    ctx = ToolContext(session=session, engine=FakeEngine(reply_uci="e7e5"))
    registry = build_registry(ctx)
    registry.dispatch("make_move", {"move": "e4"})  # engine replies e5
    result = registry.dispatch("undo", {})
    assert result["ok"] is True
    assert result["undone"] == ["e5", "e4"]
    assert session.move_history() == []
    assert session.turn == session.player_color


def test_undo_defaults_to_one_ply_without_an_engine(registry, session):
    registry.dispatch("make_move", {"move": "e4"})
    assert registry.dispatch("undo", {})["undone"] == ["e4"]
    assert session.move_history() == []


def test_undo_default_never_takes_back_the_engines_lone_opening(session):
    """Player is black: the engine's opening move is not theirs to take back."""
    session.new_game(player_color="black")
    ctx = ToolContext(session=session, engine=FakeEngine(reply_uci="e2e4"))
    registry = build_registry(ctx)
    ctx.engine.play_move(session)  # engine (white) opens
    result = registry.dispatch("undo", {})
    assert result["ok"] is False
    assert session.move_history() == ["e4"]


def test_undo_honors_an_explicit_ply_count(session):
    ctx = ToolContext(session=session, engine=FakeEngine(reply_uci="e7e5"))
    registry = build_registry(ctx)
    registry.dispatch("make_move", {"move": "e4"})
    result = registry.dispatch("undo", {"plies": 1})
    assert result["undone"] == ["e5"]


def test_undo_with_nothing_to_undo_is_error(registry):
    result = registry.dispatch("undo", {})
    assert result["ok"] is False


def test_undo_rejects_bad_plies(registry):
    assert registry.dispatch("undo", {"plies": 0})["ok"] is False
    assert registry.dispatch("undo", {"plies": "two"})["ok"] is False


def test_new_game_resets(session):
    # A game is under way, so this goes through the gate: the call arms it, the
    # confirmation runs it. (The refusal itself is pinned further down.)
    ctx = ToolContext(session=session)
    registry = build_registry(ctx)
    registry.dispatch("make_move", {"move": "e4"})

    registry.dispatch("new_game", {})
    _, result = confirm_pending(registry, ctx)

    assert result["ok"] is True
    assert session.move_history() == []
    assert session.turn == "white"


def test_resign_defaults_to_the_player_not_the_side_to_move(session):
    """An unqualified "I resign" is the *player's* resignation, and the session
    knows which side they are. The old default — the side to move — was only
    coincidentally the player (trace review, finding 8): here the player is
    white, it is black's move, and the game must still end 0-1."""
    ctx = ToolContext(session=session)
    registry = build_registry(ctx)
    registry.dispatch("make_move", {"move": "e4"})
    assert session.turn == "black" and session.player_color == "white"

    registry.dispatch("resign", {})
    _, result = confirm_pending(registry, ctx)

    assert result["ok"] is True
    assert result["outcome"] == {
        "termination": "resignation",
        "winner": "black",
        "result": "0-1",
    }
    assert session.is_game_over()


def test_resign_explicit_color(registry, session):
    result = registry.dispatch("resign", {"color": "black"})
    assert result["ok"] is True
    assert result["outcome"]["winner"] == "white"


def test_resign_rejects_bad_color_and_finished_game(registry, session):
    assert registry.dispatch("resign", {"color": "green"})["ok"] is False
    registry.dispatch("resign", {})
    assert registry.dispatch("resign", {})["ok"] is False


def test_export_pgn(registry):
    registry.dispatch("make_move", {"move": "e4"})
    registry.dispatch("make_move", {"move": "e5"})
    result = registry.dispatch("export_pgn", {})
    assert result["ok"] is True
    assert "1. e4 e5" in result["pgn"]


def test_save_and_resume_round_trip(tmp_path, session):
    registry = build_registry(ToolContext(session=session, save_dir=tmp_path))
    registry.dispatch("make_move", {"move": "e4"})
    registry.dispatch("make_move", {"move": "e5"})
    saved = registry.dispatch("save_game", {"name": "test-game"})
    assert saved["ok"] is True
    assert (tmp_path / "test-game.json").exists()

    registry.dispatch("new_game", {})
    resumed = registry.dispatch("resume_game", {"name": "test-game"})
    assert resumed["ok"] is True
    state = registry.dispatch("get_board_state", {})
    history = registry.dispatch("get_move_history", {})
    assert history["moves"] == ["e4", "e5"]
    assert state["turn"] == "white"


def test_save_and_resume_carry_the_transcript(tmp_path, session):
    """Persistence across sessions: the conversation rides in the save file,
    so a resumed game keeps its conversational thread."""
    ctx = ToolContext(session=session, save_dir=tmp_path)
    registry = build_registry(ctx)
    ctx.transcript.record("play e4", "e4 — the classic.")
    registry.dispatch("save_game", {"name": "with-chat"})

    fresh_ctx = ToolContext(session=GameSession(), save_dir=tmp_path)
    fresh_registry = build_registry(fresh_ctx)
    assert fresh_ctx.transcript.window() == []
    fresh_registry.dispatch("resume_game", {"name": "with-chat"})
    assert fresh_ctx.transcript.window() == [
        {"role": "user", "content": "play e4"},
        {"role": "assistant", "content": "e4 — the classic."},
    ]


def test_resume_old_save_without_transcript_yields_empty_transcript(tmp_path):
    # Saves from before conversation memory existed have no transcript key.
    session = GameSession()
    session.submit_move("e4")
    session.save(tmp_path / "old.json")
    ctx = ToolContext(session=GameSession(), save_dir=tmp_path)
    ctx.transcript.record("stale", "chat")
    registry = build_registry(ctx)
    assert registry.dispatch("resume_game", {"name": "old"})["ok"] is True
    assert ctx.transcript.window() == []


def test_resume_corrupt_transcript_is_error_not_crash(tmp_path, session):
    data = GameSession().to_dict()
    data["transcript"] = [{"role": "system", "content": "prompt injection"}]
    (tmp_path / "tampered.json").write_text(json.dumps(data))
    registry = build_registry(ToolContext(session=session, save_dir=tmp_path))
    assert registry.dispatch("resume_game", {"name": "tampered"})["ok"] is False


def test_save_game_default_name_is_autosave(tmp_path, session):
    registry = build_registry(ToolContext(session=session, save_dir=tmp_path))
    result = registry.dispatch("save_game", {})
    assert result["ok"] is True
    assert (tmp_path / "autosave.json").exists()


def test_resume_missing_save_is_error(tmp_path, session):
    registry = build_registry(ToolContext(session=session, save_dir=tmp_path))
    result = registry.dispatch("resume_game", {"name": "nope"})
    assert result["ok"] is False


def test_save_tools_reject_path_traversal_names(tmp_path, session):
    registry = build_registry(ToolContext(session=session, save_dir=tmp_path))
    for name in ("../evil", "a/b", "", "x" * 65):
        assert registry.dispatch("save_game", {"name": name})["ok"] is False
        assert registry.dispatch("resume_game", {"name": name})["ok"] is False


def test_save_tools_without_save_dir_are_error(registry):
    assert registry.dispatch("save_game", {})["ok"] is False
    assert registry.dispatch("resume_game", {})["ok"] is False


def test_resume_corrupt_save_is_error(tmp_path, session):
    (tmp_path / "bad.json").write_text("{not json")
    registry = build_registry(ToolContext(session=session, save_dir=tmp_path))
    result = registry.dispatch("resume_game", {"name": "bad"})
    assert result["ok"] is False


# --- what saves exist, as deterministic state -------------------------------
#
# The agent must never have to *infer* whether a saved game exists: it is a
# question the filesystem answers. `saved_game_names` is the one reader, and the
# tool layer owns it because the tool layer owns the `{name}.json` convention.


def test_saved_game_names_without_save_dir_is_empty(session):
    assert saved_game_names(ToolContext(session=session)) == []


def test_saved_game_names_with_missing_dir_is_empty(tmp_path, session):
    ctx = ToolContext(session=session, save_dir=tmp_path / "not-created")
    assert saved_game_names(ctx) == []


def test_saved_game_names_lists_saves_sorted(tmp_path, session):
    ctx = ToolContext(session=session, save_dir=tmp_path)
    registry = build_registry(ctx)
    registry.dispatch("save_game", {"name": "scholars"})
    registry.dispatch("save_game", {"name": "blitz"})
    assert saved_game_names(ctx) == ["blitz", "scholars"]


def test_saved_game_names_ignores_non_saves(tmp_path, session):
    (tmp_path / "notes.txt").write_text("not a save")
    ctx = ToolContext(session=session, save_dir=tmp_path)
    build_registry(ctx).dispatch("save_game", {"name": "real"})
    assert saved_game_names(ctx) == ["real"]


# --- settings tools ---------------------------------------------------------


def test_registry_lists_all_settings_tools(registry):
    names = {d["function"]["name"] for d in registry.definitions()}
    assert names >= {
        "set_difficulty",
        "set_verbosity",
        "set_hints_mode",
        "set_voice_output",
    }
    # The personality is fixed (Glitch); there is no set_personality tool.
    assert "set_personality" not in names


def test_settings_defaults(session):
    ctx = ToolContext(session=session)
    assert ctx.settings == Settings()
    assert ctx.settings.verbosity == "normal"
    assert ctx.settings.hints_mode is False
    assert ctx.settings.voice_output is False
    # A real default strength, not None: without one the engine silently
    # plays at Stockfish's full-strength default.
    assert ctx.settings.tier == DEFAULT_TIER
    assert ctx.settings.skill_level is None
    assert ctx.settings.elo is None


def test_set_difficulty_skill_level_recorded(session):
    ctx = ToolContext(session=session)
    registry = build_registry(ctx)
    result = registry.dispatch("set_difficulty", {"skill_level": 5})
    assert result["ok"] is True
    assert ctx.settings.skill_level == 5
    assert ctx.settings.elo is None
    # A raw knob replaces the named tier.
    assert ctx.settings.tier is None


def test_set_difficulty_by_tier_reaches_engine_and_clears_raw_knobs(session):
    engine = FakeEngine()
    ctx = ToolContext(session=session, engine=engine)
    registry = build_registry(ctx)
    registry.dispatch("set_difficulty", {"skill_level": 5})
    result = registry.dispatch("set_difficulty", {"tier": "beginner"})
    assert result["ok"] is True
    assert result["tier"] == "beginner"
    assert ctx.settings.tier == "beginner"
    assert ctx.settings.skill_level is None
    assert ctx.settings.elo is None
    assert engine.tiers == ["beginner"]


def test_set_difficulty_rejects_unknown_tier(registry):
    assert registry.dispatch("set_difficulty", {"tier": "impossible"})["ok"] is False


def test_set_difficulty_elo_recorded_and_clears_skill(session):
    ctx = ToolContext(session=session)
    registry = build_registry(ctx)
    registry.dispatch("set_difficulty", {"skill_level": 5})
    result = registry.dispatch("set_difficulty", {"elo": 1500})
    assert result["ok"] is True
    assert ctx.settings.elo == 1500
    assert ctx.settings.skill_level is None


def test_set_difficulty_requires_exactly_one_of_skill_or_elo(registry):
    assert registry.dispatch("set_difficulty", {})["ok"] is False
    assert (
        registry.dispatch("set_difficulty", {"skill_level": 5, "elo": 1500})["ok"]
        is False
    )


def test_set_difficulty_rejects_out_of_range(registry):
    assert registry.dispatch("set_difficulty", {"skill_level": 21})["ok"] is False
    assert registry.dispatch("set_difficulty", {"skill_level": -1})["ok"] is False
    assert registry.dispatch("set_difficulty", {"elo": 100})["ok"] is False
    assert registry.dispatch("set_difficulty", {"elo": 4000})["ok"] is False


@requires_stockfish
def test_set_difficulty_configures_live_engine(session, live_engine):
    ctx = ToolContext(session=session, engine=live_engine)
    registry = build_registry(ctx)
    assert registry.dispatch("set_difficulty", {"skill_level": 3})["ok"] is True
    assert registry.dispatch("set_difficulty", {"elo": 1400})["ok"] is True


def test_set_verbosity(session):
    ctx = ToolContext(session=session)
    registry = build_registry(ctx)
    result = registry.dispatch("set_verbosity", {"verbosity": "low"})
    assert result["ok"] is True
    assert ctx.settings.verbosity == "low"
    assert registry.dispatch("set_verbosity", {"verbosity": "shouty"})["ok"] is False


def test_set_hints_mode_and_voice_output(session):
    ctx = ToolContext(session=session)
    registry = build_registry(ctx)
    assert registry.dispatch("set_hints_mode", {"enabled": True})["ok"] is True
    assert ctx.settings.hints_mode is True
    assert registry.dispatch("set_voice_output", {"enabled": True})["ok"] is True
    assert ctx.settings.voice_output is True
    assert registry.dispatch("set_hints_mode", {})["ok"] is False
    assert registry.dispatch("set_voice_output", {"enabled": "yes"})["ok"] is False


def test_settings_results_are_json_serializable(registry):
    json.dumps(registry.dispatch("set_difficulty", {"skill_level": 5}))
    json.dumps(registry.dispatch("set_verbosity", {"verbosity": "high"}))
    json.dumps(registry.dispatch("set_hints_mode", {"enabled": False}))
    json.dumps(registry.dispatch("set_voice_output", {"enabled": False}))


# --- analysis tools (engine-backed) ----------------------------------------


def test_analysis_tools_without_engine_return_error(registry):
    for name in ("evaluate_position", "get_best_moves"):
        result = registry.dispatch(name, {})
        assert result["ok"] is False
        assert "engine" in result["error"]


@requires_stockfish
def test_evaluate_position_start(session, live_engine):
    registry = build_registry(ToolContext(session=session, engine=live_engine))
    result = registry.dispatch("evaluate_position", {})
    assert result["ok"] is True
    assert result["mate_in"] is None
    assert abs(result["score_cp"]) < 150


@requires_stockfish
def test_get_best_moves_mate_position(live_engine):
    session = GameSession(fen=WHITE_MATE_IN_1)
    registry = build_registry(ToolContext(session=session, engine=live_engine))
    result = registry.dispatch("get_best_moves", {"n": 2})
    assert result["ok"] is True
    best = result["moves"][0]
    assert best == {"uci": "f3f7", "san": "Qxf7#", "score_cp": None, "mate_in": 1}
    json.dumps(result)


@requires_stockfish
def test_analysis_on_finished_game_is_error_result(session, live_engine):
    session.resign("black")
    registry = build_registry(ToolContext(session=session, engine=live_engine))
    for name in ("evaluate_position", "get_best_moves"):
        result = registry.dispatch(name, {})
        assert result["ok"] is False
        assert "over" in result["error"]


def test_new_game_plays_engine_opening_when_player_is_black():
    # A voice/text "new game" while playing black must not leave the board
    # stuck waiting for white: the engine opens, mirroring the UI path.
    session = GameSession(player_color="black")
    engine = FakeEngine(reply_uci="e2e4")
    registry = build_registry(ToolContext(session=session, engine=engine))
    result = registry.dispatch("new_game", {})
    assert result["ok"] is True
    assert result["engine_move"] == {"san": "e4", "uci": "e2e4"}
    assert session.move_history() == ["e4"]
    assert session.turn == "black"


def test_new_game_as_white_has_no_engine_opening():
    session = GameSession()
    engine = FakeEngine(reply_uci="e2e4")
    registry = build_registry(ToolContext(session=session, engine=engine))
    result = registry.dispatch("new_game", {})
    assert result["ok"] is True
    assert result["engine_move"] is None
    assert session.move_history() == []


# --- "let's play chess as black": new_game has to be able to say it.
#
# `new_game()` took no arguments, so the agent had no way to assign a side and
# the model fabricated compliance instead (trace review, finding 2). This is
# also the exact intent string in the advertised conductor deep link
# (/?intent=let's+play+chess+as+black), so the handoff was broken end to end.


def test_new_game_assigns_the_requested_side():
    session = GameSession()  # the player is white today
    engine = FakeEngine(reply_uci="e2e4")
    registry = build_registry(ToolContext(session=session, engine=engine))

    result = registry.dispatch("new_game", {"player_color": "black"})

    assert result["ok"] is True
    assert session.player_color == "black"
    # Owning white, the engine must open — or the board sits waiting on a move
    # only it can make.
    assert result["engine_move"] == {"san": "e4", "uci": "e2e4"}
    assert session.move_history() == ["e4"]
    assert session.turn == "black"


def test_new_game_can_switch_back_to_white():
    session = GameSession(player_color="black")
    engine = FakeEngine(reply_uci="e2e4")
    registry = build_registry(ToolContext(session=session, engine=engine))

    result = registry.dispatch("new_game", {"player_color": "white"})

    assert result["ok"] is True
    assert session.player_color == "white"
    assert result["engine_move"] is None
    assert session.move_history() == []


def test_new_game_without_a_color_keeps_the_current_side():
    session = GameSession(player_color="black")
    engine = FakeEngine(reply_uci="e2e4")
    registry = build_registry(ToolContext(session=session, engine=engine))

    registry.dispatch("new_game", {})

    assert session.player_color == "black"


def test_new_game_rejects_a_color_that_is_not_a_side():
    session = GameSession()
    registry = build_registry(ToolContext(session=session))

    result = registry.dispatch("new_game", {"player_color": "red"})

    assert result["ok"] is False


def test_the_requested_side_survives_the_confirmation_gate():
    """The gate arms the op and the player's "yes" replays it later. If the
    requested color didn't ride along in the pending args, "new game, I'll take
    black" would be confirmed into a game as *white* — the gate would silently
    drop the only thing the player asked for."""
    session = GameSession(player_color="white")
    engine = FakeEngine(reply_uci="e2e4")
    ctx = ToolContext(session=session, engine=engine)
    registry = build_registry(ctx)
    played(session, "e4", "e5")  # a real game stands to be lost

    refused = registry.dispatch("new_game", {"player_color": "black"})
    assert refused["ok"] is False
    assert session.player_color == "white", "the gate must not mutate"

    name, result = confirm_pending(registry, ctx)

    assert name == "new_game"
    assert result["ok"] is True
    assert session.player_color == "black", "the player's yes lost their side"
    assert session.move_history() == ["e4"], "the engine owns white and must open"


# --- The destructive-op confirmation gate.
#
# `new_game` and `resign` throw a real game away. The prompt asks the agent to
# confirm first, but gemma-4-12b honors that only ~half the time (docs/agent-evals
# .md), so the rule is enforced here instead: an unconfirmed call does not mutate,
# it arms a pending op and comes back as a rejection *result* the agent reads and
# asks from. The model cannot talk its way past this — confirmation is not a tool
# argument (see test_confirmation_is_not_a_tool_argument), it is pipeline-owned.


def played(session, *sans):
    """Put a real game on the board — something that stands to be lost."""
    for san in sans:
        session.submit_move(san)
    return session


def test_unconfirmed_new_game_does_not_reset(session, registry):
    played(session, "e4", "e5")
    fen_before = session.fen()

    result = registry.dispatch("new_game", {})

    assert result["ok"] is False
    assert "confirm" in result["error"].lower()
    assert session.fen() == fen_before, "the board must not move on an unconfirmed call"


def test_unconfirmed_resign_does_not_end_the_game(session, registry):
    played(session, "e4", "e5")

    result = registry.dispatch("resign", {})

    assert result["ok"] is False
    assert "confirm" in result["error"].lower()
    assert not session.is_game_over()


def test_unconfirmed_call_arms_the_pending_op(session):
    ctx = ToolContext(session=played(GameSession(), "e4", "e5"))
    registry = build_registry(ctx)
    assert ctx.pending is None

    registry.dispatch("resign", {"color": "white"})

    assert ctx.pending is not None
    assert ctx.pending.name == "resign"
    assert ctx.pending.args == {"color": "white"}


def test_confirm_pending_executes_the_armed_op(session):
    ctx = ToolContext(session=played(GameSession(), "e4", "e5"))
    registry = build_registry(ctx)
    registry.dispatch("new_game", {})

    name, result = confirm_pending(registry, ctx)

    assert name == "new_game"
    assert result["ok"] is True
    assert ctx.session.fen() == GameSession().fen(), "confirmed: the board resets"
    assert ctx.pending is None, "the op is spent"


def test_confirm_pending_with_nothing_armed_is_a_no_op(session, registry):
    ctx = ToolContext(session=session)
    assert confirm_pending(build_registry(ctx), ctx) is None


def test_confirmation_is_not_a_tool_argument(session, registry):
    """The gate would be worthless if the model could open it: `confirm` is not
    in either schema, and the schemas are closed, so a model that invents the
    argument is rejected on args — the board still does not move."""
    played(session, "e4", "e5")
    fen_before = session.fen()

    for name in ("new_game", "resign"):
        result = registry.dispatch(name, {"confirm": True})
        assert result["ok"] is False
        assert "invalid args" in result["error"]
        assert session.fen() == fen_before


def test_a_second_unconfirmed_call_still_does_not_fire(session):
    """The agent retrying inside one turn must not self-confirm: re-arming is
    not confirming. Only the pipeline, on a new user turn, can open the gate."""
    ctx = ToolContext(session=played(GameSession(), "e4", "e5"))
    registry = build_registry(ctx)
    fen_before = ctx.session.fen()

    assert registry.dispatch("new_game", {})["ok"] is False
    assert registry.dispatch("new_game", {})["ok"] is False

    assert ctx.session.fen() == fen_before


def test_new_game_after_game_over_needs_no_confirmation(session, registry):
    """The standing exception: game_over means there is no game left to lose."""
    # Fool's mate — game over, nothing to protect.
    played(session, "f3", "e5", "g4", "Qh4")
    assert session.is_game_over()

    result = registry.dispatch("new_game", {})

    assert result["ok"] is True
    assert session.fen() == GameSession().fen()
