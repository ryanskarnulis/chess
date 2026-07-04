"""Tool layer: registry + validated dispatch + read tools.

This is the boundary the LLM talks through. Dispatch must be un-crashable:
unknown tools, malformed args, and domain errors all come back as
`{"ok": False, "error": ...}` — never an exception, never corrupted state.
Analysis tools need a live stockfish and skip without one (CI installs it).
"""

import json
import shutil

import pytest

from chessapp.game import GameSession
from chessapp.tools import Tool, ToolContext, ToolRegistry, build_registry

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
