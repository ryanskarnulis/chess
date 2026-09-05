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


# --- what the move did: the facts the narrator reacts to --------------------
#
# The observe beat between the player's move and the engine's reply describes a
# *verified* move ("you took my bishop", "and that's check"), so those two facts
# come back with the move itself. They are board truth, so the session derives
# them — not the caller, and never the model.


def test_a_quiet_move_captured_nothing_and_gave_no_check():
    result = GameSession().submit_move("e4")
    assert result.capture is None
    assert result.check is False


def test_a_capture_reports_the_piece_it_took():
    session = GameSession()
    session.submit_move("e4")
    session.submit_move("d5")
    result = session.submit_move("exd5")
    assert result.capture == "p"


def test_a_capture_reports_the_piece_type_not_just_that_it_captured():
    session = GameSession(
        "rnbqkbnr/ppp1pppp/8/8/3n4/2P5/PP1PPPPP/RNBQKBNR w KQkq - 0 1"
    )
    result = session.submit_move("cxd4")
    assert result.capture == "n"


def test_en_passant_reports_the_pawn_it_took():
    """The captured piece is not on the destination square, which is the one
    case a naive `piece_at(to_square)` gets wrong (and `captured_pieces`
    already handles)."""
    session = GameSession(
        "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3"
    )
    result = session.submit_move("exf6")
    assert result.capture == "p"


def test_a_checking_move_reports_the_check():
    session = GameSession("4k3/8/8/8/8/8/8/4K2R w K - 0 1")
    result = session.submit_move("Rh8")
    assert result.check is True
    assert result.capture is None


def test_a_rejected_move_reports_neither_fact():
    result = GameSession().submit_move("e5")
    assert result.legal is False
    assert result.capture is None
    assert result.check is False


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


# --- claimable draws: threefold repetition and the fifty-move rule ---------
#
# The two draws the rules make *claimable* rather than automatic. python-chess
# owns both predicates (`can_claim_*`), including the part everybody forgets:
# a claim is available as soon as a legal move would complete the repetition or
# the count, not only once it has. The session inherits those semantics whole —
# it never re-derives the rule — and adds the one thing a board cannot model:
# whether anybody actually claimed.

REPETITION = ("Nf3", "Nf6", "Ng1", "Ng8")

# King and rook against a bare king, one half-move short of the fifty-move
# count: no automatic draw here, and no insufficient material either.
FIFTY_MOVE_FEN = "8/8/8/4k3/8/8/4K3/6R1 w - - 99 80"


def repeated(session: GameSession, cycles: int = 2) -> GameSession:
    for _ in range(cycles):
        for san in REPETITION:
            assert session.submit_move(san).legal
    return session


def test_no_draw_is_claimable_in_a_fresh_game():
    assert GameSession().claimable_draws() == ()


def test_a_repeated_position_makes_a_threefold_claim_available():
    session = repeated(GameSession())
    assert session.claimable_draws() == ("threefold_repetition",)


def test_a_claimable_repetition_does_not_end_the_game_by_itself():
    """The issue's reproduction: the claim is available and the game plays on.
    Only fivefold repetition is automatic; three is a claim somebody has to
    make."""
    session = repeated(GameSession())
    assert session.claimable_draws()
    assert not session.is_game_over()
    assert session.outcome() is None
    assert len(session.legal_moves()) == 20


def test_a_repetition_the_next_move_would_complete_is_already_claimable():
    """python-chess's semantics, inherited rather than re-implemented: after
    seven plies the position has occurred twice and Ng8 would make it three, so
    the side to move may claim now."""
    session = repeated(GameSession())
    session.undo(1)
    assert session.move_history() == list(REPETITION * 2)[:-1]
    assert session.claimable_draws() == ("threefold_repetition",)


def test_the_fifty_move_claim_follows_the_halfmove_clock():
    ninety_eight = FIFTY_MOVE_FEN.replace(" 99 ", " 98 ")
    assert GameSession(fen=ninety_eight).claimable_draws() == ()
    # 99 is the same "one legal move away" case as the repetition above; 100 is
    # the count reached.
    assert GameSession(fen=FIFTY_MOVE_FEN).claimable_draws() == ("fifty_moves",)
    hundred = FIFTY_MOVE_FEN.replace(" 99 ", " 100 ")
    assert GameSession(fen=hundred).claimable_draws() == ("fifty_moves",)


def test_claiming_a_repetition_draw_ends_the_game():
    session = repeated(GameSession())
    outcome = session.claim_draw()
    assert outcome.termination == "threefold_repetition"
    assert outcome.winner is None
    assert outcome.result == "1/2-1/2"
    assert session.is_game_over()
    assert session.outcome() == outcome


def test_claiming_a_fifty_move_draw_names_that_rule():
    session = GameSession(fen=FIFTY_MOVE_FEN)
    assert session.claim_draw().termination == "fifty_moves"
    assert session.outcome().result == "1/2-1/2"


def test_a_claim_bumps_the_revision():
    """The board version every client tracks rides on this counter, so a claim
    has to move it exactly once."""
    session = repeated(GameSession())
    before = session.revision
    session.claim_draw()
    assert session.revision == before + 1


def test_claiming_a_draw_with_nothing_to_claim_raises():
    session = GameSession()
    session.submit_move("e4")
    fen_before, revision_before = session.fen(), session.revision
    with pytest.raises(ValueError):
        session.claim_draw()
    assert session.fen() == fen_before
    assert session.revision == revision_before
    assert not session.is_game_over()


def test_claiming_a_draw_after_the_game_is_over_raises():
    session = GameSession()
    for move in ["f3", "e5", "g4", "Qh4"]:  # fool's mate
        session.submit_move(move)
    with pytest.raises(ValueError):
        session.claim_draw()
    assert session.outcome().termination == "checkmate", "the ending stands"


def test_claiming_a_draw_after_a_resignation_raises():
    session = repeated(GameSession())
    session.resign("white")
    with pytest.raises(ValueError):
        session.claim_draw()
    assert session.outcome().termination == "resignation"


def test_a_second_claim_raises():
    session = repeated(GameSession())
    session.claim_draw()
    with pytest.raises(ValueError):
        session.claim_draw()


def test_nothing_is_claimable_once_a_draw_has_been_claimed():
    session = repeated(GameSession())
    session.claim_draw()
    assert session.claimable_draws() == ()


def test_moves_are_rejected_after_a_claimed_draw():
    session = repeated(GameSession())
    session.claim_draw()
    result = session.submit_move("Nf3")
    assert not result.legal
    assert result.reason
    assert session.legal_moves() == []
    assert session.legal_destinations() == {}


def test_undo_rejected_after_a_claimed_draw():
    """The same rule resignation has: the claim is about *this* position, so
    popping plies out from under it would leave a game over by a claim the
    board no longer supports."""
    session = repeated(GameSession())
    session.claim_draw()
    result = session.undo()
    assert not result.ok
    assert result.reason


def test_new_game_clears_a_claimed_draw():
    session = repeated(GameSession())
    session.claim_draw()
    session.new_game()
    assert not session.is_game_over()
    assert session.outcome() is None
    assert session.claimable_draws() == ()
    assert session.submit_move("e4").legal


def test_history_survives_a_claimed_draw():
    session = repeated(GameSession())
    session.claim_draw()
    assert session.move_history() == list(REPETITION * 2)


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


# --- player color ---------------------------------------------------------


def test_player_color_defaults_to_white():
    assert GameSession().player_color == "white"


def test_session_can_start_with_a_player_color():
    assert GameSession(player_color="black").player_color == "black"


def test_session_rejects_invalid_player_color():
    with pytest.raises(ValueError):
        GameSession(player_color="green")


def test_new_game_can_reassign_player_color():
    session = GameSession()
    session.new_game(player_color="black")
    assert session.player_color == "black"


def test_new_game_keeps_player_color_when_not_given():
    session = GameSession(player_color="black")
    session.submit_move("e4")
    session.new_game()
    assert session.player_color == "black"


def test_new_game_rejects_invalid_player_color():
    session = GameSession()
    with pytest.raises(ValueError):
        session.new_game(player_color="green")


# --- piece placement and castling: the derivations a description reads ----
#
# Both feed `tools.describe_position`, which is the only route a description of
# the board takes to the phase that speaks (`api._narrator_state_dict` hands the
# narrator no FEN). So they are board truth like everything else here: read off
# the position, or replayed off the move stack — never tracked alongside it.


def test_piece_placement_start_position():
    placement = GameSession().piece_placement()
    assert placement["white"] == {
        "king": ["e1"],
        "queen": ["d1"],
        "rook": ["a1", "h1"],
        "bishop": ["c1", "f1"],
        "knight": ["b1", "g1"],
        "pawn": ["a2", "b2", "c2", "d2", "e2", "f2", "g2", "h2"],
    }
    assert placement["black"]["king"] == ["e8"]
    assert placement["black"]["pawn"] == [
        "a7",
        "b7",
        "c7",
        "d7",
        "e7",
        "f7",
        "g7",
        "h7",
    ]


def test_piece_placement_names_types_in_reading_order():
    """King first, pawns last — the order a player says a position out loud,
    and the order the composed sentence comes out in."""
    assert list(GameSession().piece_placement()["white"]) == [
        "king",
        "queen",
        "rook",
        "bishop",
        "knight",
        "pawn",
    ]


def test_piece_placement_follows_the_pieces():
    session = GameSession()
    for san in ("e4", "e5", "Nf3", "Nc6"):
        assert session.submit_move(san).legal
    placement = session.piece_placement()
    assert placement["white"]["knight"] == ["b1", "f3"]
    assert placement["black"]["knight"] == ["c6", "g8"]
    assert "e2" not in placement["white"]["pawn"]
    assert "e4" in placement["white"]["pawn"]


def test_a_captured_piece_is_gone_from_the_placement():
    session = GameSession()
    for san in ("e4", "d5", "exd5", "Qxd5", "Nc3", "Qxa2", "Rxa2"):
        assert session.submit_move(san).legal
    assert "queen" not in session.piece_placement()["black"]
    assert session.piece_placement()["white"]["rook"] == ["a2", "h1"]


def test_a_promoted_pawn_stands_there_as_a_queen():
    """Material nobody captured — the reason this is read off the board rather
    than replayed as "the pawn that was on c7"."""
    session = GameSession(fen="4k3/2P5/8/8/8/8/8/4K3 w - - 0 1")
    assert session.submit_move("c8=Q").legal
    placement = session.piece_placement()
    assert placement["white"]["queen"] == ["c8"]
    assert "pawn" not in placement["white"]


def test_castling_status_at_the_start():
    status = GameSession().castling_status()
    assert status == {
        "white": {"castled": None, "rights": ["kingside", "queenside"]},
        "black": {"castled": None, "rights": ["kingside", "queenside"]},
    }


def test_a_rook_move_takes_that_side_of_the_castling_rights():
    session = GameSession()
    for san in ("Nf3", "Nf6", "Rg1", "Ng8"):
        assert session.submit_move(san).legal
    assert session.castling_status()["white"]["rights"] == ["queenside"]
    assert session.castling_status()["black"]["rights"] == ["kingside", "queenside"]


def test_a_king_move_takes_both():
    session = GameSession()
    for san in ("e4", "e5", "Ke2", "Ke7"):
        assert session.submit_move(san).legal
    status = session.castling_status()
    assert status["white"]["rights"] == []
    assert status["black"]["rights"] == []
    assert status["white"]["castled"] is None  # a walking king has not castled


def test_castling_is_recorded_for_the_side_that_castled():
    session = GameSession()
    for san in ("e4", "e5", "Nf3", "Nc6", "Bc4", "Bc5", "O-O"):
        assert session.submit_move(san).legal
    status = session.castling_status()
    assert status["white"] == {"castled": "kingside", "rights": []}
    assert status["black"]["castled"] is None
    assert status["black"]["rights"] == ["kingside", "queenside"]


def test_queenside_castling_is_told_from_kingside():
    session = GameSession()
    for san in ("d4", "d5", "Nc3", "Nc6", "Bf4", "Bf5", "Qd2", "Qd7", "O-O-O", "O-O-O"):
        assert session.submit_move(san).legal
    status = session.castling_status()
    assert status["white"]["castled"] == "queenside"
    assert status["black"]["castled"] == "queenside"


def test_a_takeback_unmakes_the_castling_too():
    """Replayed off the move stack, not recorded when it happened — so popping
    the ply is all it takes to un-say it."""
    session = GameSession()
    for san in ("e4", "e5", "Nf3", "Nc6", "Bc4", "Bc5", "O-O"):
        assert session.submit_move(san).legal
    assert session.undo(1).ok
    status = session.castling_status()
    assert status["white"]["castled"] is None
    assert status["white"]["rights"] == ["kingside", "queenside"]


# --- legal captures: the victim SAN cannot name -----------------------------


def test_no_captures_on_a_fresh_board():
    assert GameSession().legal_captures() == {}


def test_legal_captures_name_what_each_move_takes():
    session = GameSession()
    for san in ("e4", "d5", "Nf3", "Bg4"):
        assert session.submit_move(san).legal
    assert session.legal_captures() == {"exd5": "pawn"}
    assert session.submit_move("exd5").legal
    assert session.legal_captures() == {"Bxf3": "knight", "Qxd5": "pawn"}


def test_en_passant_counts_as_taking_a_pawn():
    session = GameSession(fen="4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 2")
    assert session.legal_captures() == {"exd6": "pawn"}


def test_legal_captures_are_empty_once_the_game_is_over():
    session = GameSession(
        fen="r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 0 1"
    )
    assert session.legal_captures()  # the mating move is a capture
    assert session.submit_move("Qxf7#").legal
    assert session.legal_captures() == {}
