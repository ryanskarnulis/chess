"""PGN export from the deterministic core."""

import io

import chess.pgn

from chessapp.game import GameSession


def parse(pgn: str) -> chess.pgn.Game:
    game = chess.pgn.read_game(io.StringIO(pgn))
    assert game is not None
    return game


def test_export_contains_movetext_in_san():
    session = GameSession()
    for move in ["e4", "e5", "Nf3"]:
        session.submit_move(move)
    pgn = session.export_pgn()
    assert "1. e4 e5 2. Nf3" in pgn


def test_ongoing_game_has_star_result():
    session = GameSession()
    session.submit_move("e4")
    game = parse(session.export_pgn())
    assert game.headers["Result"] == "*"


def test_checkmate_result_recorded():
    session = GameSession()
    for move in ["f3", "e5", "g4", "Qh4"]:
        session.submit_move(move)
    game = parse(session.export_pgn())
    assert game.headers["Result"] == "0-1"


def test_resignation_result_recorded():
    session = GameSession()
    session.submit_move("e4")
    session.resign("black")
    game = parse(session.export_pgn())
    assert game.headers["Result"] == "1-0"


def test_custom_fen_start_gets_setup_headers():
    fen = "4k3/8/8/8/8/8/8/4K2R w K - 0 1"
    session = GameSession(fen=fen)
    session.submit_move("O-O")
    game = parse(session.export_pgn())
    assert game.headers["FEN"] == fen
    assert game.headers["SetUp"] == "1"


def test_exported_pgn_round_trips_moves():
    session = GameSession()
    moves = ["e4", "d5", "exd5", "Qxd5", "Nc3"]
    for move in moves:
        session.submit_move(move)
    game = parse(session.export_pgn())
    board = game.board()
    sans = []
    for move in game.mainline_moves():
        sans.append(board.san(move))
        board.push(move)
    assert sans == moves
    assert board.fen() == session.fen()


def test_empty_game_exports_valid_pgn():
    game = parse(GameSession().export_pgn())
    assert game.headers["Result"] == "*"
