"""Stockfish analysis tools: evaluate_position + get_best_moves (MultiPV).

Scores are reported from White's point of view. Live-engine tests skip
without a stockfish binary (CI installs one).
"""

import shutil

import pytest

from chessapp.engine import CandidateMove, EnginePlayer, Evaluation
from chessapp.game import GameSession

requires_stockfish = pytest.mark.skipif(
    shutil.which("stockfish") is None, reason="stockfish binary not installed"
)

# White to move, Qxf7# available (scholar's mate pattern).
WHITE_MATE_IN_1 = "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 0 1"
# Black to move, Qh4# available (fool's mate).
BLACK_MATE_IN_1 = "rnbqkbnr/pppp1ppp/8/4p3/6P1/5P2/PPPPP2P/RNBQKBNR b KQkq - 0 2"
# White is a queen up.
QUEEN_UP = "rnb1kbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1"


@pytest.fixture(scope="module")
def engine():
    if shutil.which("stockfish") is None:
        pytest.skip("stockfish binary not installed")
    with EnginePlayer() as player:
        yield player


# --- evaluate_position ----------------------------------------------------


@requires_stockfish
def test_start_position_is_roughly_equal(engine):
    evaluation = engine.evaluate_position(GameSession())
    assert isinstance(evaluation, Evaluation)
    assert evaluation.mate_in is None
    assert abs(evaluation.score_cp) < 150


@requires_stockfish
def test_white_mate_in_one_reported(engine):
    evaluation = engine.evaluate_position(GameSession(fen=WHITE_MATE_IN_1))
    assert evaluation.mate_in == 1
    assert evaluation.score_cp is None


@requires_stockfish
def test_black_mate_in_one_is_negative(engine):
    evaluation = engine.evaluate_position(GameSession(fen=BLACK_MATE_IN_1))
    assert evaluation.mate_in == -1


@requires_stockfish
def test_material_advantage_shows_positive_score(engine):
    evaluation = engine.evaluate_position(GameSession(fen=QUEEN_UP))
    assert evaluation.mate_in is None or evaluation.mate_in > 0
    if evaluation.score_cp is not None:
        assert evaluation.score_cp > 300


@requires_stockfish
def test_evaluate_finished_game_raises(engine):
    session = GameSession()
    session.resign("white")
    with pytest.raises(ValueError):
        engine.evaluate_position(session)


# --- get_best_moves -------------------------------------------------------


@requires_stockfish
def test_best_moves_returns_n_distinct_legal_candidates(engine):
    session = GameSession()
    candidates = engine.get_best_moves(session, n=3)
    assert len(candidates) == 3
    ucis = [c.uci for c in candidates]
    assert len(set(ucis)) == 3
    for candidate in candidates:
        assert isinstance(candidate, CandidateMove)
        probe = GameSession()
        assert probe.submit_move(candidate.uci).legal


@requires_stockfish
def test_best_move_in_mate_position_is_the_mate(engine):
    candidates = engine.get_best_moves(GameSession(fen=WHITE_MATE_IN_1), n=2)
    assert candidates[0].uci == "f3f7"
    assert candidates[0].mate_in == 1
    assert candidates[0].san == "Qxf7#"


@requires_stockfish
def test_n_capped_at_legal_move_count(engine):
    # King in the corner with only two legal moves.
    session = GameSession(fen="7k/8/5K2/8/8/8/8/1Q6 b - - 0 1")
    legal_count = 0
    import chess

    legal_count = chess.Board(session.fen()).legal_moves.count()
    candidates = engine.get_best_moves(session, n=10)
    assert len(candidates) == legal_count


@requires_stockfish
def test_best_moves_rejects_bad_n(engine):
    with pytest.raises(ValueError):
        engine.get_best_moves(GameSession(), n=0)


@requires_stockfish
def test_best_moves_finished_game_raises(engine):
    session = GameSession()
    session.resign("black")
    with pytest.raises(ValueError):
        engine.get_best_moves(session)


@requires_stockfish
def test_analysis_does_not_mutate_session(engine):
    session = GameSession()
    before = session.fen()
    engine.evaluate_position(session)
    engine.get_best_moves(session, n=2)
    assert session.fen() == before
    assert session.move_history() == []
