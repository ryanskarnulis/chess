"""Board facts (#496, docs/glitch-sight.md decisions 1 and 2): what a player
sees at a glance, and never a conclusion. Hand-built positions pin each rule
at its edge, and the four positions of game `927c238e` pin what the mover
would have seen when he misread d4."""

import ast
import json
import re
from pathlib import Path

import chess

from chessapp import board_facts as module
from chessapp.board_facts import board_facts, material

CORPUS = Path(__file__).parent / "fixtures" / "mover_blunder_corpus.json"


def _attacked(facts, side):
    return {piece.square: piece for piece in facts.attacked[side]}


def _corpus_board(position_id):
    positions = json.loads(CORPUS.read_text())["positions"]
    (position,) = [p for p in positions if p["id"] == position_id]
    board = chess.Board(position["root_fen"])
    for uci in position["moves"]:
        board.push_uci(uci)
    return board


# -- what it is built from ----------------------------------------------------


def test_the_module_never_imports_an_engine_or_anything_of_the_app():
    tree = ast.parse(Path(module.__file__).read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module)
    assert imported <= {"__future__", "dataclasses", "typing", "chess"}


def test_the_text_never_judges():
    # The words of a conclusion: the mover weighs attacks himself.
    facts = board_facts(_corpus_board("g927c-17b"))
    words = set(re.findall(r"[a-z]+", facts.text().lower()))
    judging = {"hanging", "safe", "unsafe", "mate", "checkmate", "best", "threat"}
    assert not words & (judging | {"winning", "losing", "free", "blunder"})


# -- pieces and material ------------------------------------------------------


def test_the_start_position():
    facts = board_facts(chess.Board())
    assert facts.turn == "white"
    assert facts.pieces["white"] == (
        "Ke1",
        "Qd1",
        "Ra1",
        "Rh1",
        "Bc1",
        "Bf1",
        "Nb1",
        "Ng1",
        *(f"pawn {f}2" for f in "abcdefgh"),
    )
    assert facts.material == {"white": 39, "black": 39}
    assert facts.attacked == {"white": (), "black": ()}
    assert facts.pins == ()
    assert facts.checkers == ()
    assert facts.text() == (
        "Board facts (what is on the board; nothing here judges it):\n"
        "White: Ke1, Qd1, Ra1, Rh1, Bc1, Bf1, Nb1, Ng1, "
        "pawns a2 b2 c2 d2 e2 f2 g2 h2.\n"
        "Black: Ke8, Qd8, Ra8, Rh8, Bc8, Bf8, Nb8, Ng8, "
        "pawns a7 b7 c7 d7 e7 f7 g7 h7.\n"
        "White's pieces under attack: none.\n"
        "Black's pieces under attack: none.\n"
        "Pinned to the king: none."
    )


def test_material_counts_1_3_3_5_9_and_not_the_king():
    board = chess.Board("4k3/8/8/8/8/8/8/QRBNP1K1 w - - 0 1")
    assert material(board, chess.WHITE) == 9 + 5 + 3 + 3 + 1
    assert material(board, chess.BLACK) == 0


def test_a_side_with_no_pawns_lists_none():
    facts = board_facts(chess.Board("4k3/8/8/8/8/8/8/R3K3 w - - 0 1"))
    assert "White: Ke1, Ra1.\n" in facts.text()


# -- attacks and defences -----------------------------------------------------


def test_attackers_and_defenders_are_listed_cheapest_first():
    # Equal values break by square, a1 first.
    board = chess.Board("3qk3/8/4p3/Q2n4/4P3/1B6/8/3RK3 b - - 0 1")
    knight = _attacked(board_facts(board), "black")["d5"]
    assert knight.label == "Nd5"
    assert knight.attackers == ("pawn e4", "Bb3", "Rd1", "Qa5")
    assert knight.defenders == ("pawn e6", "Qd8")


def test_an_undefended_piece_is_defended_by_nothing_and_never_called_hanging():
    board = chess.Board("4k3/8/8/3n4/8/8/8/3RK3 w - - 0 1")
    facts = board_facts(board)
    assert _attacked(facts, "black")["d5"].defenders == ()
    assert "  Nd5: attacked by Rd1; defended by nothing" in facts.text()


def test_attacked_pieces_run_most_valuable_first():
    # Black's queen and pawn both attacked by the rook and knight.
    board = chess.Board("4k3/8/8/3q4/8/2N5/3p4/3RK3 b - - 0 1")
    assert [p.label for p in board_facts(board).attacked["black"]] == [
        "Qd5",
        "pawn d2",
    ]


def test_the_king_is_never_an_attacked_piece_but_can_attack_and_defend():
    board = chess.Board("8/8/8/8/8/2k5/3p4/3K4 w - - 0 1")
    pawn = _attacked(board_facts(board), "black")["d2"]
    assert pawn.attackers == ("Kd1",)
    assert pawn.defenders == ("Kc3",)
    assert all(
        not p.label.startswith("K") for p in board_facts(board).attacked["white"]
    )


def test_x_rays_are_not_counted():
    # Rd1 sits behind Rd2: only Rd2 defends d4 directly. Black's Qd8 behind
    # Rd7 attacks nothing on d4 either.
    board = chess.Board("3qk3/3r4/8/8/3N4/8/3R4/3RK3 w - - 0 1")
    knight = _attacked(board_facts(board), "white")["d4"]
    assert knight.attackers == ("Rd7",)
    assert knight.defenders == ("Rd2",)


def test_a_pinned_defender_still_defends_and_the_pin_is_listed():
    # White's Nd2 attacks e4 though the bishop on b4 pins it to the king.
    board = chess.Board("4k3/8/8/8/1b2n3/8/3N4/4K3 w - - 0 1")
    facts = board_facts(board)
    assert _attacked(facts, "black")["e4"].attackers == ("Nd2",)
    assert [pin.to_dict() for pin in facts.pins] == [
        {"side": "white", "pinned": "Nd2", "by": "Bb4"}
    ]
    assert "Pinned to the king: White's Nd2, by Bb4." in facts.text()


def test_a_piece_beyond_the_pinner_on_the_same_line_is_not_the_pinner():
    # pin() returns the whole e-file; Re8, behind the pinning Re5, is not it.
    board = chess.Board("4r2k/8/8/4r3/8/8/4B3/4K3 w - - 0 1")
    (pin,) = board_facts(board).pins
    assert pin.to_dict() == {"side": "white", "pinned": "Be2", "by": "Re5"}


def test_a_promoting_pawn_attacks_like_any_pawn():
    board = chess.Board("2r1k3/3P4/8/8/8/8/8/4K3 w - - 0 1")
    rook = _attacked(board_facts(board), "black")["c8"]
    assert rook.attackers == ("pawn d7",)


def test_a_pawn_that_just_moved_two_is_attacked_by_the_en_passant_pawn():
    board = chess.Board("4k3/3p4/8/4P3/8/8/8/4K3 b - - 0 1")
    board.push_san("d5")
    pawn = _attacked(board_facts(board), "black")["d5"]
    assert pawn.attackers == ("pawn e5",)


def test_en_passant_ends_with_the_next_move():
    board = chess.Board("4k3/3p4/8/4P3/8/8/8/4K3 b - - 0 1")
    board.push_san("d5")
    board.push_san("Kd2")
    board.push_san("Ke7")
    assert "d5" not in _attacked(board_facts(board), "black")


def test_castling_rights_change_nothing():
    fen = "r3k2r/8/8/8/8/8/8/R3K2R w {} - 0 1"
    with_rights = board_facts(chess.Board(fen.format("KQkq")))
    without = board_facts(chess.Board(fen.format("-")))
    assert with_rights == without


# -- check --------------------------------------------------------------------


def test_check_names_the_checkers():
    board = chess.Board("4k3/8/8/8/1b6/8/8/R3K2r w - - 0 1")
    facts = board_facts(board)
    assert facts.checkers == ("Bb4", "Rh1")
    assert facts.text().endswith("White is in check from Bb4, Rh1.")


def test_checkmate_is_shown_as_the_check_it_is_and_never_called_mate():
    board = chess.Board()
    for san in ("f3", "e5", "g4", "Qh4#"):
        board.push_san(san)
    facts = board_facts(board)
    assert facts.checkers == ("Qh4",)
    assert not {"mate", "checkmate"} & set(re.findall(r"[a-z]+", facts.text().lower()))


# -- game 927c238e --------------------------------------------------------------


def test_before_13_nxd4_the_pawn_is_defended_by_two_pawns():
    facts = board_facts(_corpus_board("g927c-13b"))
    assert facts.turn == "black"
    pawn = _attacked(facts, "white")["d4"]
    assert pawn.attackers == ("Nc6",)
    assert pawn.defenders == ("pawn c3", "pawn e3")


def test_before_17_qxd4_the_pawn_is_defended_by_c3_and_the_rook():
    facts = board_facts(_corpus_board("g927c-17b"))
    assert facts.text() == (
        "Board facts (what is on the board; nothing here judges it):\n"
        "Black: Kg8, Qd7, Ra8, Rd8, Be7, pawns a7 b7 c6 e6 f7 g7 h7.\n"
        "White: Kc1, Qe2, Rd1, Re1, Bg3, Ne4, pawns a2 b2 c3 d4 f2 g2 h4.\n"
        "Black's pieces under attack: none.\n"
        "White's pieces under attack:\n"
        "  pawn d4: attacked by Qd7; defended by pawn c3, Rd1\n"
        "  pawn h4: attacked by Be7; defended by Bg3\n"
        "Pinned to the king: none."
    )


def test_before_7_o_o_the_bishop_on_d3_is_undefended():
    facts = board_facts(_corpus_board("g927c-7b"))
    bishop = _attacked(facts, "white")["d3"]
    assert bishop.attackers == ("Bf5",)
    assert bishop.defenders == ()


def test_before_11_bxe4_both_sides_pieces_are_attacked():
    facts = board_facts(_corpus_board("g927c-11b"))
    knight = _attacked(facts, "white")["e4"]
    assert knight.attackers == ("pawn d5", "Bf5")
    assert knight.defenders == ("Bd3",)
    assert _attacked(facts, "black")["d6"].attackers == ("Bg3", "Ne4")


def test_the_structured_form_carries_the_same_facts():
    facts = board_facts(_corpus_board("g927c-17b"))
    data = facts.to_dict()
    assert data["turn"] == "black"
    assert data["material"] == {"white": 32, "black": 29}
    assert data["attacked"]["white"][0] == {
        "square": "d4",
        "piece": "pawn d4",
        "attackers": ["Qd7"],
        "defenders": ["pawn c3", "Rd1"],
    }
    assert data["pins"] == [] and data["checkers"] == []
    assert json.loads(json.dumps(data)) == data
