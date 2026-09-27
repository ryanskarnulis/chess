"""`move_parts.fits`: the legal moves that fit a move described by its parts
(#371). Board terms only — nothing here reads the player's words."""

import chess
import pytest

from chessapp.game import GameSession
from chessapp.move_parts import PartsError, fits, starting_squares


def _board(*sans: str) -> chess.Board:
    board = chess.Board()
    for san in sans:
        board.push_san(san)
    return board


def test_the_kings_knight_is_the_g1_knights_two_moves():
    assert sorted(fits(chess.Board(), "knight", "kings")) == ["Nf3", "Nh3"]
    assert sorted(fits(chess.Board(), "knight", "queens")) == ["Na3", "Nc3"]


def test_the_kings_pawn_is_the_e_pawn():
    board = chess.Board()
    assert sorted(fits(board, "pawn", "kings")) == ["e3", "e4"]
    assert sorted(fits(board, "pawn", "queens")) == ["d3", "d4"]
    assert sorted(fits(board, "pawn", "e")) == ["e3", "e4"]
    assert sorted(fits(board, "pawn", "e2")) == ["e3", "e4"]


def test_a_bare_piece_is_every_move_it_has():
    assert len(fits(chess.Board(), "pawn")) == 16
    assert len(fits(chess.Board(), "knight")) == 4


def test_to_narrows_to_the_destination():
    assert fits(chess.Board(), "pawn", to="e4") == ["e4"]
    assert fits(chess.Board(), "knight", to="f3") == ["Nf3"]
    assert fits(chess.Board(), "knight", "queens", to="f3") == []


def test_a_bishop_by_its_square_colour():
    board = _board("e4", "e5")
    assert sorted(fits(board, "bishop", "light")) == [
        "Ba6",
        "Bb5",
        "Bc4",
        "Bd3",
        "Be2",
    ]
    assert fits(board, "bishop", "dark") == []


def test_castling_is_the_kings_move():
    board = _board("e4", "e5", "Nf3", "Nc6", "Bc4", "Bc5")
    assert "O-O" in fits(board, "king")
    assert fits(board, "king", to="g1") == ["O-O"]


def test_a_piece_is_followed_from_where_it_started():
    """After Nc3 and Nf3-d4, "the king's knight" is still the one from g1."""
    board = _board("Nc3", "e5", "Nf3", "a6", "Nd4", "h6")
    started = starting_squares(board)
    assert started[chess.D4] == chess.G1
    assert all(
        san.startswith("N") and board.parse_san(san).from_square == chess.D4
        for san in fits(board, "knight", "kings")
    )
    assert all(
        board.parse_san(san).from_square == chess.C3
        for san in fits(board, "knight", "queens")
    )


def test_castling_moves_the_rook_with_the_king():
    board = _board("e4", "e5", "Nf3", "Nc6", "Bc4", "Bc5", "O-O", "d6")
    started = starting_squares(board)
    assert started[chess.F1] == chess.H1
    assert started[chess.G1] == chess.E1
    assert all(
        board.parse_san(san).from_square == chess.F1
        for san in fits(board, "rook", "kings")
    )


def test_black_is_answered_for_black():
    board = _board("e4")
    assert sorted(fits(board, "knight", "kings")) == ["Nf6", "Nh6"]
    assert sorted(fits(board, "pawn", "kings")) == ["e5", "e6"]


def test_a_pinned_piece_has_no_moves():
    board = chess.Board("4k3/8/8/b7/8/8/3N4/4K3 w - - 0 1")
    assert fits(board, "knight") == []


def test_not_a_board_term_is_refused():
    with pytest.raises(PartsError):
        fits(chess.Board(), "dragon")
    with pytest.raises(PartsError):
        fits(chess.Board(), "knight", "the shiny one")
    with pytest.raises(PartsError):
        fits(chess.Board(), "knight", to="z9")


def test_the_session_answers_off_its_own_history():
    session = GameSession()
    for san in ("Nc3", "e5", "Nf3", "a6", "Nd4", "h6"):
        assert session.submit_move(san).legal
    assert all(san.startswith("N") for san in session.moves_fitting("knight", "kings"))
    assert "Nb5" not in session.moves_fitting("knight", "kings")
