"""Move history and captured-pieces, derived from board state.

Both are derived from the board's move stack — no separate bookkeeping
the LLM could ever desync.
"""

from chessapp.game import GameSession

# --- move history ---------------------------------------------------------


def test_new_session_has_empty_history():
    session = GameSession()
    assert session.move_history() == []


def test_history_records_moves_in_san_order():
    session = GameSession()
    for move in ["e4", "e5", "Nf3"]:
        session.submit_move(move)
    assert session.move_history() == ["e4", "e5", "Nf3"]


def test_history_uses_san_even_for_uci_input():
    session = GameSession()
    session.submit_move("e2e4")
    session.submit_move("g8f6")
    assert session.move_history() == ["e4", "Nf6"]


def test_rejected_move_not_recorded():
    session = GameSession()
    session.submit_move("e4")
    session.submit_move("banana")
    session.submit_move("e2e5")
    assert session.move_history() == ["e4"]


def test_new_game_clears_history():
    session = GameSession()
    session.submit_move("e4")
    session.new_game()
    assert session.move_history() == []


def test_history_from_custom_fen_start():
    session = GameSession(fen="4k3/8/8/8/8/8/8/4K2R w K - 0 1")
    session.submit_move("O-O")
    assert session.move_history() == ["O-O"]


def test_history_disambiguates_san():
    # Two knights can reach d2; SAN must disambiguate.
    session = GameSession(fen="4k3/8/8/8/8/1N3N2/8/4K3 w - - 0 1")
    session.submit_move("f3d2")
    assert session.move_history() == ["Nfd2"]


# --- captured pieces ------------------------------------------------------


def test_new_session_has_no_captures():
    session = GameSession()
    assert session.captured_pieces() == {"white": [], "black": []}


def test_pawn_capture_recorded_for_capturing_color():
    session = GameSession()
    session.submit_move("e4")
    session.submit_move("d5")
    session.submit_move("exd5")
    assert session.captured_pieces() == {"white": ["p"], "black": []}


def test_captures_recorded_in_order_per_color():
    session = GameSession()
    for move in ["e4", "d5", "exd5", "Qxd5", "Nc3", "Qxg2", "Nf3", "Qxh1"]:
        assert session.submit_move(move).legal
    assert session.captured_pieces() == {
        "white": ["p"],
        "black": ["p", "p", "r"],
    }


def test_en_passant_capture_records_pawn():
    session = GameSession()
    for move in ["e4", "a6", "e5", "d5", "exd6"]:
        assert session.submit_move(move).legal
    assert session.captured_pieces() == {"white": ["p"], "black": []}


def test_capture_with_promotion_records_captured_piece():
    session = GameSession(fen="1r2k3/2P5/8/8/8/8/8/4K3 w - - 0 1")
    result = session.submit_move("cxb8=Q")
    assert result.legal
    assert session.captured_pieces() == {"white": ["r"], "black": []}


def test_non_capture_moves_record_nothing():
    session = GameSession()
    for move in ["e4", "e5", "Nf3", "Nc6"]:
        session.submit_move(move)
    assert session.captured_pieces() == {"white": [], "black": []}


def test_new_game_clears_captures():
    session = GameSession()
    session.submit_move("e4")
    session.submit_move("d5")
    session.submit_move("exd5")
    session.new_game()
    assert session.captured_pieces() == {"white": [], "black": []}


# --- material balance -----------------------------------------------------
#
# The count behind the honesty guard's material class: who is ahead, by how
# many pawns, from the *player's* side. Read off the board rather than from
# `captured_pieces()`, because a promotion adds material nobody captured.


def test_a_fresh_board_is_level():
    assert GameSession().material_balance() == 0


def test_a_capture_puts_the_capturing_side_ahead():
    session = GameSession()
    for move in ["e4", "d5", "exd5"]:
        assert session.submit_move(move).legal
    assert session.material_balance() == 1


def test_the_balance_is_read_from_the_players_side():
    session = GameSession(player_color="black")
    for move in ["e4", "d5", "exd5"]:
        assert session.submit_move(move).legal
    assert session.material_balance() == -1


def test_a_trade_of_equal_pieces_is_level_again():
    session = GameSession()
    for move in ["e4", "d5", "exd5", "Qxd5"]:
        assert session.submit_move(move).legal
    assert session.material_balance() == 0


def test_the_nominal_trade_counts_net():
    """A knight for a pawn is +2, not +3 — which is why the guard verifies
    material talk's magnitude to within a pawn."""
    session = GameSession()
    for move in ["e4", "Nf6", "Nc3", "Nxe4", "Nxe4"]:
        assert session.submit_move(move).legal
    assert session.material_balance() == 2


def test_a_promotion_lands_in_the_balance():
    """Material nobody captured: the pawn that walked in is a queen now."""
    session = GameSession(fen="4k3/2P5/8/8/8/8/8/4K3 w - - 0 1")
    assert session.submit_move("c8=Q").legal
    assert session.material_balance() == 9
