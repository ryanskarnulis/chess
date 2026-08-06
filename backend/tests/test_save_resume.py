"""Save/resume: serialize a session to disk and restore it.

The serialized form is root FEN + UCI move list + the two session-level
endings a board cannot model (resignation, a claimed draw) — resuming replays
the moves through the same legality gate, so a corrupted file can never
produce an inconsistent board.
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


def test_claimed_draw_round_trips():
    """A claimed draw is session-level state like a resignation, so it has to
    survive the file: a saved-then-resumed drawn game is still drawn."""
    session = GameSession()
    for _ in range(2):
        for move in ["Nf3", "Nf6", "Ng1", "Ng8"]:
            session.submit_move(move)
    session.claim_draw()
    restored = GameSession.from_dict(session.to_dict())
    assert restored.is_game_over()
    assert restored.outcome() == session.outcome()
    assert restored.outcome().termination == "threefold_repetition"


def test_save_without_the_claim_flag_is_an_unclaimed_game():
    """Saves written before claims existed carry no flag; they load as the
    live games they were."""
    data = GameSession().to_dict()
    del data["draw_claimed"]
    restored = GameSession.from_dict(data)
    assert not restored.is_game_over()


def test_from_dict_rejects_a_claim_the_position_does_not_support():
    """Replayed through the same gate as the moves: a file claiming a draw in a
    position with no claim available cannot produce a drawn board."""
    data = GameSession().to_dict()
    data["draw_claimed"] = True
    with pytest.raises(ValueError):
        GameSession.from_dict(data)


def test_from_dict_rejects_a_non_bool_claim_flag():
    data = GameSession().to_dict()
    data["draw_claimed"] = "yes"
    with pytest.raises(ValueError):
        GameSession.from_dict(data)


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


# --- player color ----------------------------------------------------------


def test_player_color_round_trips():
    session = GameSession(player_color="black")
    restored = GameSession.from_dict(session.to_dict())
    assert restored.player_color == "black"


def test_save_without_player_color_defaults_to_white():
    # Saves that predate the player-color field must stay loadable.
    data = GameSession().to_dict()
    data.pop("player_color", None)
    assert GameSession.from_dict(data).player_color == "white"


def test_invalid_player_color_in_save_is_rejected():
    data = GameSession().to_dict()
    data["player_color"] = "green"
    with pytest.raises(ValueError):
        GameSession.from_dict(data)
