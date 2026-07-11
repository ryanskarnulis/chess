"""Deterministic fast-parse for plain move commands (the seam BRIEF reserves).

`parse_move` maps an utterance that is *entirely* one unambiguous legal move
to that move's SAN, and returns None for everything else — ambiguous, illegal,
or non-move text falls through to the agent unchanged. Pure function of
(text, fen): exhaustively unit-tested here, no LLM, no session.
"""

import pytest

from chessapp.fastparse import parse_move

START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
# 1. e4 d5 — white to move; exd5 is the only capture on d5.
CAPTURE = "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2"
# Both castlings legal for white.
CASTLE_BOTH = "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1"
# Only kingside rights for white.
CASTLE_KING_ONLY = "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w Kkq - 0 1"
# White pawn on e7, e8 empty: four promotion moves and nothing else to e8.
PROMOTION = "8/4P1k1/8/8/8/8/8/4K3 w - - 0 1"
# Knights on a1 and c1 both reach b3 — "knight to b3" is ambiguous.
TWO_KNIGHTS = "5k2/8/8/8/8/8/8/N1N2K2 w - - 0 1"
# Pawn b5 and bishop e4 can both take the c6 pawn: bxc6 vs Bxc6.
BXC6 = "7k/8/2p5/1P6/4B3/8/8/4K3 w - - 0 1"
# Lone white rook a1: Ra8 gives check, so its SAN carries the suffix.
ROOK_CHECK = "4k3/8/8/8/8/8/8/R3K3 w - - 0 1"
# Fool's mate delivered — game over, no legal moves.
GAME_OVER = "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"
# 1. d4 e5 — dxe5 is the pawn capture named by file.
DXE5 = "rnbqkbnr/pppp1ppp/8/4p3/3P4/8/PPP1PPPP/RNBQKBNR w KQkq e6 0 2"
# A knight on d3 can take e5, but no pawn can: "d takes e5" names a pawn
# capture (SAN dxe5 semantics), so nothing matches.
KNIGHT_NOT_PAWN = "4k3/8/8/4p3/8/3N4/8/4K3 w - - 0 1"


# --- notation: SAN and UCI, case- and suffix-forgiving ------------------------


@pytest.mark.parametrize(
    ("text", "san"),
    [
        ("e4", "e4"),
        ("E4", "e4"),
        ("e4.", "e4"),
        ("Nf3", "Nf3"),
        ("nf3", "Nf3"),
        ("e2e4", "e4"),
        ("g1f3", "Nf3"),
    ],
)
def test_notation_forms_from_the_start_position(text, san):
    assert parse_move(text, START) == san


def test_check_suffix_is_not_required():
    assert parse_move("ra8", ROOK_CHECK) == "Ra8+"


def test_exact_case_settles_a_case_insensitive_san_collision():
    # bxc6 (pawn) and Bxc6 (bishop) collide when lowercased; the typed case
    # picks one. Without it the collision is ambiguous.
    assert parse_move("bxc6", BXC6) == "bxc6"
    assert parse_move("Bxc6", BXC6) == "Bxc6"


# --- spoken phrases -----------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "san"),
    [
        ("pawn to e4", "e4"),
        ("knight to f3", "Nf3"),
        ("Knight to f3!", "Nf3"),
        ("knight f3", "Nf3"),
        ("night to f3", "Nf3"),  # STT homophone
    ],
)
def test_piece_phrases_from_the_start_position(text, san):
    assert parse_move(text, START) == san


@pytest.mark.parametrize(
    ("text", "san"),
    [
        ("takes on d5", "exd5"),
        ("take d5", "exd5"),
        ("captures on d5", "exd5"),
        ("pawn takes d5", "exd5"),
    ],
)
def test_capture_phrases(text, san):
    assert parse_move(text, CAPTURE) == san


def test_piece_word_settles_a_capture_collision():
    assert parse_move("pawn takes c6", BXC6) == "bxc6"
    assert parse_move("bishop takes on c6", BXC6) == "Bxc6"


@pytest.mark.parametrize(
    "text",
    [
        "d takes e5",
        "d takes on e5",
        "d captures e5",
        "d x e5",
        "d ex e5",  # how STT hears "dxe5"
        "D takes E5",
    ],
)
def test_file_source_capture_phrases(text):
    # "d takes e5" is how players pronounce dxe5: a pawn capture named by
    # its source file.
    assert parse_move(text, DXE5) == "dxe5"


def test_file_source_capture_settles_the_bxc6_collision():
    # "b takes c6" names the pawn capture, never the bishop's.
    assert parse_move("b takes c6", BXC6) == "bxc6"


def test_file_source_capture_means_a_pawn():
    # SAN dxe5 semantics: a piece capture is named by its piece ("knight
    # takes e5"), so when only a knight can take, the file phrase matches
    # nothing and falls through.
    assert parse_move("d takes e5", KNIGHT_NOT_PAWN) is None
    assert parse_move("knight takes e5", KNIGHT_NOT_PAWN) == "Nxe5"


def test_bare_square_fires_only_when_one_move_lands_there():
    # Ra8+ is the only move to a8 — "a8" names it. d5 from the start position
    # is reachable by nothing, and no phrase means no move.
    assert parse_move("a8", ROOK_CHECK) == "Ra8+"
    assert parse_move("d6", START) is None
    assert parse_move("rook to a8", ROOK_CHECK) == "Ra8+"


# --- castling -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "san"),
    [
        ("castle kingside", "O-O"),
        ("castle king side", "O-O"),
        ("kingside castle", "O-O"),
        ("castle short", "O-O"),
        ("castle queenside", "O-O-O"),
        ("long castle", "O-O-O"),
        ("o-o", "O-O"),
        ("0-0-0", "O-O-O"),
    ],
)
def test_castling_forms(text, san):
    assert parse_move(text, CASTLE_BOTH) == san


def test_bare_castle_is_ambiguous_only_when_both_sides_are_legal():
    assert parse_move("castle", CASTLE_BOTH) is None
    assert parse_move("castle", CASTLE_KING_ONLY) == "O-O"


def test_castling_when_illegal_falls_through():
    assert parse_move("castle kingside", START) is None


# --- promotion ------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "san"),
    [
        ("e8 promote to queen", "e8=Q"),
        ("pawn to e8 promoting to a knight", "e8=N+"),  # knight on e8 checks g7
        ("e8=q", "e8=Q"),
        ("e7e8q", "e8=Q"),
    ],
)
def test_promotion_forms(text, san):
    assert parse_move(text, PROMOTION) == san


def test_promotion_without_a_piece_is_ambiguous():
    # Four promotions land on e8 — which piece is the agent's question to ask.
    assert parse_move("pawn to e8", PROMOTION) is None


def test_file_source_capture_promotion():
    # White pawn g7, black rook h8: a capture-promotion named by file.
    fen = "7r/6P1/8/8/8/8/8/K3k3 w - - 0 1"
    assert parse_move("g takes h8 promote to queen", fen) == "gxh8=Q"


# --- everything else falls through ---------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "what are my legal moves?",
        "can I play e4?",
        "play e4",  # leading verbs stay the agent's job
        "resign",
        "undo",
        "move the knight",
        "options, then knight f6",
        "knight",
        "double push",
        "",
        "   ",
    ],
)
def test_non_move_text_falls_through(text):
    assert parse_move(text, START) is None


def test_illegal_moves_fall_through():
    assert parse_move("knight to f6", START) is None
    assert parse_move("e5", START) is None


def test_ambiguous_piece_phrase_falls_through():
    assert parse_move("knight to b3", TWO_KNIGHTS) is None


def test_ambiguous_capture_phrase_falls_through():
    assert parse_move("takes on c6", BXC6) is None


def test_game_over_position_parses_nothing():
    assert parse_move("e4", GAME_OVER) is None
