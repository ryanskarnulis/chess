"""Save/resume: serialize a session to disk and restore it.

The serialized form is root FEN + UCI move list + resignation flag —
resuming replays the moves through the same legality gate, so a
corrupted file can never produce an inconsistent board.
"""

import json

import pytest

from chessapp.game import GameSession

# --- dict round-trip ------------------------------------------------------


def test_fresh_game_round_trips():
    session = GameSession()
    restored = GameSession.from_dict(session.to_dict())
    assert restored.fen() == session.fen()
    assert restored.move_history() == []
    assert not restored.is_game_over()


def test_mid_game_round_trip_preserves_state():
    session = GameSession()
    for move in ["e4", "d5", "exd5", "Qxd5"]:
        session.submit_move(move)
    restored = GameSession.from_dict(session.to_dict())
    assert restored.fen() == session.fen()
    assert restored.turn == session.turn
    assert restored.move_history() == ["e4", "d5", "exd5", "Qxd5"]
    assert restored.captured_pieces() == session.captured_pieces()


def test_custom_fen_start_round_trips():
    session = GameSession(fen="4k3/8/8/8/8/8/8/4K2R w K - 0 1")
    session.submit_move("O-O")
    restored = GameSession.from_dict(session.to_dict())
    assert restored.fen() == session.fen()
    assert restored.move_history() == ["O-O"]


def test_resigned_game_round_trips():
    session = GameSession()
    session.submit_move("e4")
    session.resign("black")
    restored = GameSession.from_dict(session.to_dict())
    assert restored.is_game_over()
    outcome = restored.outcome()
    assert outcome.termination == "resignation"
    assert outcome.winner == "white"


def test_checkmated_game_round_trips():
    session = GameSession()
    for move in ["f3", "e5", "g4", "Qh4"]:
        session.submit_move(move)
    restored = GameSession.from_dict(session.to_dict())
    assert restored.is_game_over()
    assert restored.outcome().termination == "checkmate"


def test_undo_works_after_resume():
    session = GameSession()
    session.submit_move("e4")
    session.submit_move("e5")
    restored = GameSession.from_dict(session.to_dict())
    result = restored.undo()
    assert result.ok
    assert restored.move_history() == ["e4"]


def test_from_dict_rejects_missing_keys():
    with pytest.raises(ValueError):
        GameSession.from_dict({"version": 1})


def test_from_dict_rejects_illegal_move_in_file():
    data = GameSession().to_dict()
    data["moves"] = ["e2e5"]
    with pytest.raises(ValueError):
        GameSession.from_dict(data)


def test_from_dict_rejects_unknown_version():
    data = GameSession().to_dict()
    data["version"] = 999
    with pytest.raises(ValueError):
        GameSession.from_dict(data)


# --- disk round-trip ------------------------------------------------------


def test_save_and_load_from_disk(tmp_path):
    path = tmp_path / "game.json"
    session = GameSession()
    session.submit_move("e4")
    session.save(path)
    restored = GameSession.load(path)
    assert restored.fen() == session.fen()
    assert restored.move_history() == ["e4"]


def test_saved_file_is_json(tmp_path):
    path = tmp_path / "game.json"
    GameSession().save(path)
    data = json.loads(path.read_text())
    assert data["version"] == 1


def test_load_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        GameSession.load(tmp_path / "nope.json")
