"""Resignation and result recording.

Resignation is a session-level fact (python-chess boards don't model it),
recorded by GameSession and folded into `outcome()` / `is_game_over()`.
"""

import pytest

from chessapp.game import GameSession


def test_resign_defaults_to_side_to_move():
    session = GameSession()
    session.submit_move("e4")  # black to move now
    outcome = session.resign()
    assert outcome.termination == "resignation"
    assert outcome.winner == "white"
    assert outcome.result == "1-0"


def test_white_resigns_black_wins():
    session = GameSession()
    outcome = session.resign("white")
    assert outcome.winner == "black"
    assert outcome.result == "0-1"


def test_black_resigns_white_wins():
    session = GameSession()
    outcome = session.resign("black")
    assert outcome.winner == "white"
    assert outcome.result == "1-0"


def test_game_is_over_after_resignation():
    session = GameSession()
    session.resign("white")
    assert session.is_game_over()
    assert session.outcome().termination == "resignation"


def test_moves_rejected_after_resignation():
    session = GameSession()
    session.resign("white")
    result = session.submit_move("e4")
    assert not result.legal
    assert result.reason


def test_undo_rejected_after_resignation():
    session = GameSession()
    session.submit_move("e4")
    session.resign("black")
    result = session.undo()
    assert not result.ok
    assert result.reason


def test_resign_invalid_color_raises():
    session = GameSession()
    with pytest.raises(ValueError):
        session.resign("green")


def test_resign_after_game_over_raises():
    session = GameSession()
    for move in ["f3", "e5", "g4", "Qh4"]:
        session.submit_move(move)
    with pytest.raises(ValueError):
        session.resign("white")


def test_double_resign_raises():
    session = GameSession()
    session.resign("white")
    with pytest.raises(ValueError):
        session.resign("black")


def test_new_game_clears_resignation():
    session = GameSession()
    session.resign("white")
    session.new_game()
    assert not session.is_game_over()
    assert session.outcome() is None
    assert session.submit_move("e4").legal


def test_history_preserved_after_resignation():
    session = GameSession()
    session.submit_move("e4")
    session.submit_move("e5")
    session.resign("black")
    assert session.move_history() == ["e4", "e5"]
