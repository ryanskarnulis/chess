"""Game review: per-move classification + accuracy for a whole game.

The review method is adapted from lichess's published approach (win-percent
conversion and per-move accuracy curve) as glue over our own Stockfish
bridge — deterministic code produces every number; the agent only narrates.
Pure math is always tested; whole-game reviews need a live Stockfish and
skip without one.
"""

import shutil

import pytest

from chessapp.analysis import (
    GameReview,
    move_accuracy,
    review_game,
    win_percent,
)
from chessapp.engine import EnginePlayer
from chessapp.game import GameSession
from chessapp.tools import ToolContext, build_registry

requires_stockfish = pytest.mark.skipif(
    shutil.which("stockfish") is None, reason="stockfish binary not installed"
)

# 1.e4 e5 2.Bc4 Bc5 3.Qh5 Nf6?? 4.Qxf7# — scholar's mate, one huge black
# blunder, White finishing with mate.
SCHOLARS_MATE = ("e4", "e5", "Bc4", "Bc5", "Qh5", "Nf6", "Qxf7#")


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


# --- win percent (pure) -----------------------------------------------------


def test_equal_position_is_fifty_percent():
    assert win_percent(0) == pytest.approx(50.0)


def test_win_percent_is_monotonic_and_bounded():
    values = [win_percent(cp) for cp in (-100_000, -300, 0, 300, 100_000)]
    assert values == sorted(values)
    assert 0.0 <= values[0] < 50.0 < values[-1] <= 100.0


def test_mate_scale_saturates():
    assert win_percent(100_000) > 99.9
    assert win_percent(-100_000) < 0.1


# --- move accuracy (pure) ---------------------------------------------------


def test_no_loss_is_perfect_accuracy():
    assert move_accuracy(55.0, 55.0) == 100.0


def test_improvement_is_perfect_accuracy():
    # The engine can under-promise; a move that improves the win chance is
    # never penalized.
    assert move_accuracy(50.0, 60.0) == 100.0


def test_a_huge_drop_scores_near_zero():
    assert move_accuracy(90.0, 1.0) < 10.0


def test_accuracy_is_clamped_to_0_100():
    assert 0.0 <= move_accuracy(100.0, 0.0) <= 100.0


def test_bigger_drops_score_lower():
    small = move_accuracy(60.0, 55.0)
    large = move_accuracy(60.0, 20.0)
    assert large < small <= 100.0


# --- review_game ------------------------------------------------------------


def test_reviewing_an_empty_game_raises():
    class NeverCalledEngine:
        def get_best_moves(self, session, n=1):  # pragma: no cover
            raise AssertionError("no analysis should happen")

    with pytest.raises(ValueError):
        review_game(NeverCalledEngine(), GameSession())


@requires_stockfish
def test_review_covers_every_move_in_order(engine):
    session = play(GameSession(), *SCHOLARS_MATE)
    review = review_game(engine, session)
    assert isinstance(review, GameReview)
    assert [m.san for m in review.moves] == list(SCHOLARS_MATE)
    assert [m.color for m in review.moves] == [
        "white",
        "black",
    ] * 3 + ["white"]


@requires_stockfish
def test_the_blunder_is_found_and_counted(engine):
    session = play(GameSession(), *SCHOLARS_MATE)
    review = review_game(engine, session)
    nf6 = review.moves[5]
    assert nf6.san == "Nf6"
    assert nf6.classification == "blunder"
    assert nf6.cp_loss >= 300
    assert review.counts["black"]["blunder"] >= 1


@requires_stockfish
def test_delivering_mate_costs_nothing(engine):
    session = play(GameSession(), *SCHOLARS_MATE)
    review = review_game(engine, session)
    mate = review.moves[-1]
    assert mate.san == "Qxf7#"
    assert mate.cp_loss == 0
    assert mate.classification == "good"


@requires_stockfish
def test_the_blundering_side_scores_lower_accuracy(engine):
    session = play(GameSession(), *SCHOLARS_MATE)
    review = review_game(engine, session)
    assert 0.0 <= review.accuracy["black"] < review.accuracy["white"] <= 100.0


@requires_stockfish
def test_review_does_not_mutate_the_session(engine):
    session = play(GameSession(), *SCHOLARS_MATE)
    fen_before = session.fen()
    review_game(engine, session)
    assert session.fen() == fen_before


# --- the tool ---------------------------------------------------------------


def test_tool_without_engine_is_an_error():
    registry = build_registry(ToolContext(session=GameSession()))
    result = registry.dispatch("review_game", {})
    assert result["ok"] is False
    assert "engine" in result["error"]


@requires_stockfish
def test_tool_reviews_the_game(engine):
    session = play(GameSession(), *SCHOLARS_MATE)
    registry = build_registry(ToolContext(session=session, engine=engine))
    result = registry.dispatch("review_game", {})
    assert result["ok"] is True
    assert len(result["moves"]) == len(SCHOLARS_MATE)
    assert result["moves"][5]["classification"] == "blunder"
    assert set(result["accuracy"]) == {"white", "black"}
    assert set(result["counts"]) == {"white", "black"}


@requires_stockfish
def test_tool_with_no_moves_is_an_error(engine):
    registry = build_registry(ToolContext(session=GameSession(), engine=engine))
    assert registry.dispatch("review_game", {})["ok"] is False
