""" "What was my mistake?" — deterministic last-move analysis.

`analyze_last_move` compares the move actually played against Stockfish's
best in the same position and reports the centipawn loss with a
classification; the agent narrates this data, it never judges the move
itself. Classification thresholds are pure and always tested; the
end-to-end analyses need a live Stockfish and skip without one.
"""

import shutil

import pytest

from chessapp.analysis import MoveAnalysis, analyze_last_move, classify_cp_loss
from chessapp.engine import EnginePlayer
from chessapp.game import GameSession
from chessapp.tools import ToolContext, build_registry

requires_stockfish = pytest.mark.skipif(
    shutil.which("stockfish") is None, reason="stockfish binary not installed"
)


@pytest.fixture(scope="module")
def engine():
    if shutil.which("stockfish") is None:
        pytest.skip("stockfish binary not installed")
    with EnginePlayer() as player:
        yield player


def play(session, *moves):
    for move in moves:
        assert session.submit_move(move).legal, move
    return session


# --- classification (pure) --------------------------------------------------


@pytest.mark.parametrize("cp_loss", [0, 10, 49])
def test_small_loss_is_good(cp_loss):
    assert classify_cp_loss(cp_loss) == "good"


@pytest.mark.parametrize("cp_loss", [50, 99])
def test_medium_loss_is_inaccuracy(cp_loss):
    assert classify_cp_loss(cp_loss) == "inaccuracy"


@pytest.mark.parametrize("cp_loss", [100, 299])
def test_large_loss_is_mistake(cp_loss):
    assert classify_cp_loss(cp_loss) == "mistake"


@pytest.mark.parametrize("cp_loss", [300, 1000, 99_999])
def test_huge_loss_is_blunder(cp_loss):
    assert classify_cp_loss(cp_loss) == "blunder"


def test_negative_loss_rejected():
    with pytest.raises(ValueError):
        classify_cp_loss(-1)


# --- analyze_last_move ------------------------------------------------------


@requires_stockfish
def test_no_moves_yet_raises(engine):
    with pytest.raises(ValueError):
        analyze_last_move(engine, GameSession())


@requires_stockfish
def test_hanging_mate_is_a_blunder(engine):
    # 3...Nf6?? walks into Qxf7# — the canonical scholar's-mate blunder.
    session = play(GameSession(), "e4", "e5", "Bc4", "Bc5", "Qh5", "Nf6")
    analysis = analyze_last_move(engine, session)
    assert isinstance(analysis, MoveAnalysis)
    assert analysis.played_san == "Nf6"
    assert analysis.color == "black"
    assert analysis.classification == "blunder"
    assert analysis.cp_loss >= 300
    assert analysis.best_san  # there was a better option


@requires_stockfish
def test_a_normal_opening_move_is_good(engine):
    session = play(GameSession(), "e4", "e5")
    analysis = analyze_last_move(engine, session)
    assert analysis.played_san == "e5"
    assert analysis.color == "black"
    assert analysis.classification in ("good", "inaccuracy")
    assert analysis.cp_loss < 100


@requires_stockfish
def test_delivering_mate_is_the_best_move(engine):
    # Fool's mate: 2...Qh4# ends the game; the move that mates loses nothing.
    session = play(GameSession(), "f3", "e5", "g4", "Qh4")
    analysis = analyze_last_move(engine, session)
    assert analysis.played_san == "Qh4#"
    assert analysis.cp_loss == 0
    assert analysis.classification == "good"


@requires_stockfish
def test_analysis_does_not_mutate_the_session(engine):
    session = play(GameSession(), "e4", "e5", "Bc4", "Bc5", "Qh5", "Nf6")
    fen_before = session.fen()
    history_before = session.move_history()
    analyze_last_move(engine, session)
    assert session.fen() == fen_before
    assert session.move_history() == history_before


# --- the tool ---------------------------------------------------------------


def test_tool_without_engine_is_an_error():
    registry = build_registry(ToolContext(session=GameSession()))
    result = registry.dispatch("analyze_last_move", {})
    assert result["ok"] is False
    assert "engine" in result["error"]


@requires_stockfish
def test_tool_reports_the_blunder(engine):
    session = play(GameSession(), "e4", "e5", "Bc4", "Bc5", "Qh5", "Nf6")
    registry = build_registry(ToolContext(session=session, engine=engine))
    result = registry.dispatch("analyze_last_move", {})
    assert result["ok"] is True
    assert result["played"] == "Nf6"
    assert result["color"] == "black"
    assert result["classification"] == "blunder"
    assert result["cp_loss"] >= 300
    assert result["best"]


@requires_stockfish
def test_tool_with_no_moves_is_an_error(engine):
    registry = build_registry(ToolContext(session=GameSession(), engine=engine))
    result = registry.dispatch("analyze_last_move", {})
    assert result["ok"] is False
