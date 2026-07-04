"""Undo: ply takeback on the deterministic core.

`undo(plies=1)` pops single plies; `undo(plies=2)` is the vs-engine pair
takeback (rewind the engine's reply and the user's move so it's the
user's turn again). Pairing policy lives in the caller — the core just
pops exactly what it's asked to, or refuses.
"""

from chessapp.game import GameSession, UndoResult

INITIAL_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def test_undo_single_ply_restores_position_and_turn():
    session = GameSession()
    session.submit_move("e4")
    result = session.undo()
    assert isinstance(result, UndoResult)
    assert result.ok
    assert session.fen() == INITIAL_FEN
    assert session.turn == "white"


def test_undo_reports_undone_moves_in_pop_order():
    session = GameSession()
    session.submit_move("e4")
    session.submit_move("e5")
    result = session.undo(plies=2)
    assert result.ok
    assert result.undone == ("e5", "e4")


def test_undo_pair_restores_turn_for_same_side():
    # vs-engine semantics: user (white) takes back their move plus the
    # engine's reply, landing back on white's turn.
    session = GameSession()
    session.submit_move("e4")
    session.submit_move("e5")
    session.submit_move("Nf3")
    session.submit_move("Nc6")
    result = session.undo(plies=2)
    assert result.ok
    assert session.turn == "white"
    assert session.move_history() == ["e4", "e5"]


def test_undo_with_no_moves_fails():
    session = GameSession()
    result = session.undo()
    assert not result.ok
    assert result.reason


def test_undo_more_plies_than_played_fails_and_leaves_board_unchanged():
    session = GameSession()
    session.submit_move("e4")
    before = session.fen()
    result = session.undo(plies=2)
    assert not result.ok
    assert result.reason
    assert session.fen() == before


def test_undo_zero_or_negative_plies_rejected():
    session = GameSession()
    session.submit_move("e4")
    assert not session.undo(plies=0).ok
    assert not session.undo(plies=-1).ok
    assert session.move_history() == ["e4"]


def test_undo_reopens_finished_game():
    session = GameSession()
    for move in ["f3", "e5", "g4", "Qh4"]:
        session.submit_move(move)
    assert session.is_game_over()
    result = session.undo()
    assert result.ok
    assert not session.is_game_over()
    assert session.outcome() is None
    # Black is back on the move and can play again (including re-mating).
    assert session.submit_move("Qh4").legal


def test_undo_restores_history_and_captures():
    session = GameSession()
    for move in ["e4", "d5", "exd5"]:
        session.submit_move(move)
    session.undo()
    assert session.move_history() == ["e4", "d5"]
    assert session.captured_pieces() == {"white": [], "black": []}


def test_move_after_undo_is_accepted():
    session = GameSession()
    session.submit_move("e4")
    session.undo()
    result = session.submit_move("d4")
    assert result.legal
    assert session.move_history() == ["d4"]
