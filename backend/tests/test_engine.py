"""Stockfish bridge: engine moves at configurable difficulty.

Tests that need a live Stockfish are skipped when the binary is absent;
CI installs it and runs them. Difficulty validation is pure and always
tested.
"""

import shutil

import pytest

from chessapp.engine import (
    ELO_MAX,
    ELO_MIN,
    EnginePlayer,
    validate_elo,
    validate_skill_level,
)
from chessapp.game import GameSession

requires_stockfish = pytest.mark.skipif(
    shutil.which("stockfish") is None, reason="stockfish binary not installed"
)

# --- difficulty validation (pure, no binary) ------------------------------


def test_skill_level_bounds_accepted():
    validate_skill_level(0)
    validate_skill_level(20)


@pytest.mark.parametrize("level", [-1, 21, 3.5, "ten", None])
def test_bad_skill_level_rejected(level):
    with pytest.raises(ValueError):
        validate_skill_level(level)


def test_elo_bounds_accepted():
    validate_elo(ELO_MIN)
    validate_elo(ELO_MAX)


@pytest.mark.parametrize("elo", [ELO_MIN - 1, ELO_MAX + 1, 1500.5, "gm", None])
def test_bad_elo_rejected(elo):
    with pytest.raises(ValueError):
        validate_elo(elo)


# --- live engine ----------------------------------------------------------


@pytest.fixture(scope="module")
def engine():
    if shutil.which("stockfish") is None:
        pytest.skip("stockfish binary not installed")
    with EnginePlayer() as player:
        yield player


@requires_stockfish
def test_engine_choose_move_is_legal(engine):
    session = GameSession()
    uci = engine.choose_move(session)
    assert session.submit_move(uci).legal


@requires_stockfish
def test_engine_play_move_advances_session(engine):
    session = GameSession()
    session.submit_move("e4")
    result = engine.play_move(session)
    assert result.legal
    assert session.turn == "white"
    assert len(session.move_history()) == 2


@requires_stockfish
def test_engine_finds_mate_in_one(engine):
    # White mates with Qxf7#; full-strength engine must find it.
    session = GameSession(
        fen="r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 0 1"
    )
    engine.set_skill_level(20)
    result = engine.play_move(session)
    assert result.legal
    assert result.game_over
    assert session.outcome().termination == "checkmate"


@requires_stockfish
def test_engine_plays_at_low_skill(engine):
    session = GameSession()
    engine.set_skill_level(0)
    assert engine.play_move(session).legal


@requires_stockfish
def test_engine_plays_at_limited_elo(engine):
    session = GameSession()
    engine.set_elo(ELO_MIN)
    assert engine.play_move(session).legal


@requires_stockfish
def test_choose_move_on_finished_game_raises(engine):
    session = GameSession()
    session.resign("white")
    with pytest.raises(ValueError):
        engine.choose_move(session)


@requires_stockfish
def test_engine_does_not_mutate_session_on_choose(engine):
    session = GameSession()
    before = session.fen()
    engine.choose_move(session)
    assert session.fen() == before


@requires_stockfish
def test_context_manager_closes_engine():
    with EnginePlayer() as player:
        assert player.choose_move(GameSession())
    # after close, further use fails
    with pytest.raises(chess_engine_closed_errors()):
        player.choose_move(GameSession())


def chess_engine_closed_errors():
    import chess.engine

    return (chess.engine.EngineError, BrokenPipeError, ValueError)
