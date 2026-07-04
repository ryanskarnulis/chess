"""Acceptance test for the brief's core requirement: a full game against
Stockfish must work with the LLM turned off.

Only `GameSession` (board truth) and `EnginePlayer` (Stockfish bridge)
are used — no agent layer, no network, no LLM anywhere.
"""

import shutil

import pytest

from chessapp.engine import EnginePlayer
from chessapp.game import GameSession

requires_stockfish = pytest.mark.skipif(
    shutil.which("stockfish") is None, reason="stockfish binary not installed"
)

# Draw rules (75-move / fivefold) bound every game, but an engine-vs-engine
# game ending anywhere near that would indicate something is broken.
MAX_PLIES = 600


@requires_stockfish
def test_full_engine_vs_engine_game_reaches_a_result():
    session = GameSession()
    with EnginePlayer(move_time=0.01) as engine:
        plies = 0
        while not session.is_game_over():
            assert plies < MAX_PLIES, "game did not terminate in a sane length"
            result = engine.play_move(session)
            assert result.legal, f"engine produced a rejected move at ply {plies}"
            plies += 1

    outcome = session.outcome()
    assert outcome is not None
    assert outcome.result in {"1-0", "0-1", "1/2-1/2"}
    assert len(session.move_history()) == plies

    pgn = session.export_pgn()
    assert outcome.result in pgn


@requires_stockfish
def test_scripted_user_vs_engine_game():
    # A "user" (scripted moves, standing in for board clicks — still no LLM)
    # opens with the Italian; the engine answers each move.
    session = GameSession()
    with EnginePlayer(move_time=0.01) as engine:
        for user_move in ["e4", "Nf3", "Bc4", "O-O"]:
            result = session.submit_move(user_move)
            if not result.legal:
                # The engine's replies invalidated the scripted plan; the
                # "user" falls back to the engine's suggestion instead.
                result = session.submit_move(engine.choose_move(session))
            assert result.legal
            if session.is_game_over():
                break
            assert engine.play_move(session).legal
            if session.is_game_over():
                break

    # Mid-flight and consistent: user + engine plies both recorded.
    history = session.move_history()
    assert session.is_game_over() or len(history) == 8
    assert history[0] == "e4"


@requires_stockfish
def test_game_survives_save_resume_mid_flight(tmp_path):
    session = GameSession()
    with EnginePlayer(move_time=0.01) as engine:
        for _ in range(6):
            engine.play_move(session)
        session.save(tmp_path / "game.json")

        resumed = GameSession.load(tmp_path / "game.json")
        assert resumed.fen() == session.fen()
        # Play on from the resumed session until the game ends.
        plies = len(resumed.move_history())
        while not resumed.is_game_over():
            assert plies < MAX_PLIES
            assert engine.play_move(resumed).legal
            plies += 1
    assert resumed.outcome() is not None
