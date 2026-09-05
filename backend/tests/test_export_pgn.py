"""PGN export: the deterministic core, and the headers the app composes for it.

The core writes down whatever headers it is handed and owns `Result` alone.
Who played, where and when are facts a `GameSession` cannot know, so
`tools.pgn_headers` composes them — one composer for both exports, because a
player who copies the PGN out of the chat and out of the post-game screen must
get the same document.
"""

import io
from datetime import date

import chess.pgn
import pytest
from fastapi.testclient import TestClient

from chessapp.api import create_app
from chessapp.game import GameSession
from chessapp.tools import ToolContext, build_registry, pgn_headers
from fakes import FakeEngine

TODAY = date.today().isoformat().replace("-", ".")


def parse(pgn: str) -> chess.pgn.Game:
    game = chess.pgn.read_game(io.StringIO(pgn))
    assert game is not None
    return game


def exported(session: GameSession, **ctx_kwargs) -> chess.pgn.Game:
    """The game as the `export_pgn` tool hands it over: composed headers and
    all. Built through `ToolContext` rather than a literal header dict, because
    what is under test is what the app fills in, not that a dict is copied."""
    ctx = ToolContext(session=session, **ctx_kwargs)
    return parse(session.export_pgn(pgn_headers(ctx)))


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


def test_claimed_draw_result_recorded():
    session = GameSession()
    for _ in range(2):
        for move in ["Nf3", "Nf6", "Ng1", "Ng8"]:
            session.submit_move(move)
    session.claim_draw()
    game = parse(session.export_pgn())
    assert game.headers["Result"] == "1/2-1/2"


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


# --- the headers the app composes -------------------------------------------


def test_headers_name_the_player_on_their_own_side():
    """Live, every tag came back "?" — for facts the app was holding all along
    (2026-09-04 walkthrough). The human is "Player" on whichever side they took,
    and the engine's strength rides with Glitch's name."""
    game = exported(GameSession(), engine=FakeEngine())
    assert game.headers["Event"] == "Casual game"
    assert game.headers["Site"] == "Chess vs Glitch (home network)"
    assert game.headers["Date"] == TODAY
    assert game.headers["Round"] == "-"
    assert game.headers["White"] == "Player"
    assert game.headers["Black"] == "Glitch (Stockfish, casual)"
    assert "?" not in str(game)


def test_headers_follow_a_player_who_took_black():
    game = exported(GameSession(player_color="black"), engine=FakeEngine())
    assert game.headers["White"] == "Glitch (Stockfish, casual)"
    assert game.headers["Black"] == "Player"


@pytest.mark.parametrize(
    ("setting", "value", "expected"),
    [
        ("tier", "advanced", "Glitch (Stockfish, advanced)"),
        ("elo", 1400, "Glitch (Stockfish, 1400 Elo)"),
        ("skill_level", 5, "Glitch (Stockfish, skill 5)"),
    ],
)
def test_strength_is_spelled_the_way_it_was_set(setting, value, expected):
    """Difficulty is exactly one of the three, and the header says which — a
    reader of the PGN cannot tell 1400 Elo from skill 5 from a tier name, and
    guessing one spelling for all three would misreport two of them."""
    ctx = ToolContext(session=GameSession(), engine=FakeEngine())
    ctx.settings.tier = None
    setattr(ctx.settings, setting, value)
    assert pgn_headers(ctx)["Black"] == expected


def test_no_engine_claims_no_stockfish():
    """Direct mode off, brain-only, a game replayed from a save: whatever it
    was played against, it was not this deployment's engine."""
    assert pgn_headers(ToolContext(session=GameSession()))["Black"] == "Glitch"


def test_composed_headers_cannot_overwrite_the_result():
    """`Result` is board truth, so it is the one tag the composer may not set.
    Applied last, whatever a caller passes."""
    session = GameSession()
    session.submit_move("e4")
    session.resign("black")
    game = parse(session.export_pgn({"Result": "0-1", "Event": "Casual game"}))
    assert game.headers["Result"] == "1-0"


def test_export_without_headers_is_unchanged():
    """The core's own export still fills in nothing — what an offline export
    and the older tests here get."""
    game = parse(GameSession().export_pgn())
    assert game.headers["Event"] == "?"
    assert game.headers["Date"] == "????.??.??"


# --- the date rides with the game -------------------------------------------


def test_resumed_game_keeps_the_day_it_was_played(tmp_path):
    session = GameSession()
    session.submit_move("e4")
    session._started = "2026-07-04"
    session.save(tmp_path / "game.json")
    resumed = GameSession.load(tmp_path / "game.json")
    assert resumed.started == "2026-07-04"
    assert exported(resumed).headers["Date"] == "2026.07.04"


def test_save_from_before_the_date_existed_says_unknown():
    """A save written before games recorded a start date has none, and the PGN
    uses the standard's own spelling for it rather than claiming today."""
    legacy = GameSession().to_dict()
    del legacy["started"]
    restored = GameSession.from_dict(legacy)
    assert restored.started is None
    assert exported(restored).headers["Date"] == "????.??.??"


def test_a_start_date_that_is_not_a_date_is_refused():
    bad = GameSession().to_dict() | {"started": "sometime"}
    with pytest.raises(ValueError):
        GameSession.from_dict(bad)


# --- one PGN, two routes -----------------------------------------------------


def test_endpoint_and_tool_export_the_same_pgn():
    """The chat's copy button and the post-game screen's hand over the same
    bytes: both compose through `pgn_headers`, so neither can drift into a
    document the other would not produce."""
    ctx = ToolContext(session=GameSession(player_color="black"), engine=FakeEngine())
    for san in ("e4", "e5", "Nf3", "Nc6"):
        assert ctx.session.submit_move(san).legal
    registry = build_registry(ctx)
    client = TestClient(create_app(ctx, registry=registry))
    assert (
        client.get("/api/game/pgn").json()["pgn"]
        == registry.dispatch("export_pgn", {})["pgn"]
    )
