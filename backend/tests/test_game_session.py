"""GameSession: the deterministic core owning board truth.

Everything here is pure python-chess — no engine, no LLM.
"""

import pytest

from chessapp.game import GameSession, MoveResult

# --- new game -----------------------------------------------------------


def test_new_session_starts_at_initial_position():
    session = GameSession()
    assert session.fen() == "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def test_new_session_is_whites_turn():
    session = GameSession()
    assert session.turn == "white"


def test_new_session_is_not_over():
    session = GameSession()
    assert not session.is_game_over()
    assert session.outcome() is None


def test_new_game_resets_position():
    session = GameSession()
    session.submit_move("e4")
    session.new_game()
    assert session.fen() == "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    assert session.turn == "white"


def test_session_can_start_from_fen():
    fen = "4k3/8/8/8/8/8/8/4K2R w K - 0 1"
    session = GameSession(fen=fen)
    assert session.fen() == fen


def test_session_rejects_invalid_fen():
    with pytest.raises(ValueError):
        GameSession(fen="not a fen")


# --- submit_move: accept ------------------------------------------------


def test_legal_san_move_is_accepted():
    session = GameSession()
    result = session.submit_move("e4")
    assert isinstance(result, MoveResult)
    assert result.legal
    assert result.san == "e4"
    assert result.uci == "e2e4"


def test_legal_uci_move_is_accepted():
    session = GameSession()
    result = session.submit_move("e2e4")
    assert result.legal
    assert result.san == "e4"
    assert result.uci == "e2e4"


def test_accepted_move_updates_board():
    session = GameSession()
    session.submit_move("e4")
    assert session.fen() == "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"


def test_san_capture_notation_is_accepted():
    session = GameSession()
    session.submit_move("e4")
    session.submit_move("d5")
    result = session.submit_move("exd5")
    assert result.legal
    assert result.san == "exd5"


def test_uci_promotion_move_is_accepted():
    session = GameSession(fen="8/4P3/8/8/8/8/2k5/K7 w - - 0 1")
    result = session.submit_move("e7e8q")
    assert result.legal
    assert result.san == "e8=Q"


def test_castling_san_is_accepted():
    session = GameSession(fen="4k3/8/8/8/8/8/8/4K2R w K - 0 1")
    result = session.submit_move("O-O")
    assert result.legal
    assert result.uci == "e1g1"


# --- submit_move: reject ------------------------------------------------


def test_illegal_move_is_rejected():
    session = GameSession()
    result = session.submit_move("e5")  # black's pawn push on white's turn
    assert not result.legal
    assert result.reason


def test_illegal_uci_move_is_rejected():
    session = GameSession()
    result = session.submit_move("e2e5")
    assert not result.legal


def test_garbage_input_is_rejected_not_raised():
    session = GameSession()
    result = session.submit_move("banana")
    assert not result.legal
    assert result.reason


def test_rejected_move_leaves_board_unchanged():
    session = GameSession()
    before = session.fen()
    session.submit_move("Ke2")  # illegal: blocked by own pawn
    assert session.fen() == before


def test_move_leaving_king_in_check_is_rejected():
    # White king pinned scenario: moving the pinned piece is illegal.
    session = GameSession(fen="4k3/8/8/8/8/4r3/4B3/4K3 w - - 0 1")
    result = session.submit_move("Bc4")
    assert not result.legal


def test_move_rejected_after_game_over():
    # Fool's mate, then try to keep playing.
    session = GameSession()
    for move in ["f3", "e5", "g4", "Qh4"]:
        assert session.submit_move(move).legal
    result = session.submit_move("a3")
    assert not result.legal
    assert result.reason


# --- legal moves ----------------------------------------------------------


def test_legal_moves_start_position():
    session = GameSession()
    moves = session.legal_moves()
    assert len(moves) == 20
    assert "e4" in moves
    assert "Nf3" in moves


def test_legal_moves_reflect_position():
    # Cornered king: Qb1 covers h7, so g8 is the only flight square.
    session = GameSession(fen="7k/8/5K2/8/8/8/8/1Q6 b - - 0 1")
    assert session.legal_moves() == ["Kg8"]


def test_legal_moves_empty_when_game_over():
    session = GameSession()
    session.resign("white")
    assert session.legal_moves() == []


# --- check ------------------------------------------------------------------


def test_is_check_false_at_start():
    assert GameSession().is_check() is False


def test_is_check_true_when_side_to_move_is_in_check():
    # Rook on e2 gives check down the open e-file.
    session = GameSession(fen="4k3/8/8/8/8/8/4R3/4K3 b - - 0 1")
    assert session.is_check() is True


def test_is_check_clears_after_the_check_is_answered():
    session = GameSession(fen="4k3/8/8/8/8/8/4R3/4K3 b - - 0 1")
    assert session.submit_move("Kd8").legal
    assert session.is_check() is False


# --- legal destinations (board-UI move hints) -----------------------------


def test_legal_destinations_start_position():
    dests = GameSession().legal_destinations()
    # Every pawn and the two knights can move; kings/queen/rooks/bishops can't.
    assert set(dests["e2"]) == {"e3", "e4"}
    assert set(dests["g1"]) == {"f3", "h3"}
    assert "e1" not in dests  # king is hemmed in at the start


def test_legal_destinations_grouped_by_origin():
    # Cornered king: g8 is the only flight square.
    session = GameSession(fen="7k/8/5K2/8/8/8/8/1Q6 b - - 0 1")
    assert session.legal_destinations() == {"h8": ["g8"]}


def test_legal_destinations_empty_when_game_over():
    session = GameSession()
    session.resign("white")
    assert session.legal_destinations() == {}


# --- turn tracking ------------------------------------------------------


def test_turn_alternates_after_moves():
    session = GameSession()
    session.submit_move("e4")
    assert session.turn == "black"
    session.submit_move("e5")
    assert session.turn == "white"


def test_rejected_move_does_not_change_turn():
    session = GameSession()
    session.submit_move("banana")
    assert session.turn == "white"


# --- game-over detection ------------------------------------------------


def test_checkmate_detected():
    session = GameSession()
    for move in ["f3", "e5", "g4", "Qh4"]:
        session.submit_move(move)
    assert session.is_game_over()
    outcome = session.outcome()
    assert outcome.termination == "checkmate"
    assert outcome.winner == "black"
    assert outcome.result == "0-1"


def test_move_result_flags_checkmate():
    session = GameSession()
    session.submit_move("f3")
    session.submit_move("e5")
    session.submit_move("g4")
    result = session.submit_move("Qh4")
    assert result.legal
    assert result.game_over


def test_stalemate_detected():
    # Black to move, no legal moves, not in check.
    session = GameSession(fen="7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
    assert session.is_game_over()
    outcome = session.outcome()
    assert outcome.termination == "stalemate"
    assert outcome.winner is None
    assert outcome.result == "1/2-1/2"


def test_insufficient_material_detected():
    session = GameSession(fen="4k3/8/8/8/8/8/8/4K3 w - - 0 1")
    assert session.is_game_over()
    assert session.outcome().termination == "insufficient_material"


def test_fivefold_repetition_detected():
    session = GameSession()
    for _ in range(4):
        session.submit_move("Nf3")
        session.submit_move("Nf6")
        session.submit_move("Ng1")
        session.submit_move("Ng8")
    assert session.is_game_over()
    assert session.outcome().termination == "fivefold_repetition"


def test_check_is_not_game_over():
    session = GameSession(fen="4k3/4R3/4K3/8/8/8/8/8 b - - 0 1")
    assert not session.is_game_over()
    assert session.outcome() is None


def test_ongoing_game_has_no_outcome():
    session = GameSession()
    session.submit_move("e4")
    assert session.outcome() is None


# --- position_fens: per-ply positions for history review -----------------


START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def test_position_fens_fresh_game():
    session = GameSession()
    assert session.position_fens() == [START_FEN]


def test_position_fens_after_moves():
    session = GameSession()
    session.submit_move("e4")
    session.submit_move("e5")
    fens = session.position_fens()
    assert len(fens) == 3
    assert fens[0] == START_FEN
    # python-chess omits the ep square when no en-passant capture is legal.
    assert fens[1] == "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
    assert fens[-1] == session.fen()


def test_position_fens_respects_undo():
    session = GameSession()
    session.submit_move("e4")
    session.submit_move("e5")
    session.undo(1)
    fens = session.position_fens()
    assert len(fens) == 2
    assert fens[-1] == session.fen()


def test_position_fens_from_custom_root():
    fen = "4k3/8/8/8/8/8/8/4K2R w K - 0 1"
    session = GameSession(fen=fen)
    session.submit_move("Rh8+")
    fens = session.position_fens()
    assert fens[0] == fen
    assert fens[-1] == session.fen()
