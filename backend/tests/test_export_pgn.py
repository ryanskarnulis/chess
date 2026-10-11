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
from chessapp.coordinator import TurnCoordinator
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


# --- how the game ended (#460) -----------------------------------------------


def _mated() -> GameSession:
    session = GameSession()
    for move in ["f3", "e5", "g4", "Qh4"]:
        session.submit_move(move)
    return session


def _resigned() -> GameSession:
    session = GameSession()
    session.submit_move("e4")
    session.resign("black")
    return session


def _agreed() -> GameSession:
    session = GameSession()
    session.submit_move("e4")
    session.agree_draw()
    return session


def _claimed() -> GameSession:
    session = GameSession()
    for _ in range(2):
        for move in ["Nf3", "Nf6", "Ng1", "Ng8"]:
            session.submit_move(move)
    session.claim_draw()
    return session


def _stalemated() -> GameSession:
    session = GameSession(fen="7k/8/6K1/8/8/8/8/5Q2 w - - 0 1")
    session.submit_move("Qf7")
    return session


@pytest.mark.parametrize(
    ("finished", "result"),
    [
        (_mated, "0-1"),
        (_resigned, "1-0"),
        (_agreed, "1/2-1/2"),
        (_claimed, "1/2-1/2"),
        (_stalemated, "1/2-1/2"),
    ],
)
def test_a_finished_game_says_it_ended_normally(finished, result):
    """A resigned or mated game exported only `Result` (#460). Every ending a
    game here can have is the standard's "Normal" — what lichess-style viewers
    read — and the export, composed headers and all, still parses and replays
    to the board it came from."""
    session = finished()
    game = exported(session, engine=FakeEngine())
    assert game.headers["Termination"] == "Normal"
    assert game.headers["Result"] == result
    assert game.headers["White"] == "Player"
    assert game.end().board().fen() == session.fen()


def test_a_game_still_going_has_no_termination():
    session = GameSession()
    session.submit_move("e4")
    assert "Termination" not in parse(session.export_pgn()).headers


def test_composed_headers_cannot_overwrite_the_termination():
    game = parse(_resigned().export_pgn({"Termination": "Time forfeit"}))
    assert game.headers["Termination"] == "Normal"


# --- the strength the game was played at (#460) --------------------------------

CASUAL = {"tier": "casual", "skill_level": None, "elo": None}
ELO_1400 = {"tier": None, "skill_level": None, "elo": 1400}


def _opened(**kwargs) -> GameSession:
    """e4 e5: one move each, so the engine (Black) has moved."""
    session = GameSession(**kwargs)
    session.submit_move("e4")
    session.submit_move("e5")
    return session


def test_difficulty_is_recorded_at_the_engines_first_move():
    """Not before: a game the engine has not moved in was played at no
    strength, so a call then records nothing. Only strength is kept."""
    session = GameSession()
    session.submit_move("e4")
    session.note_difficulty(CASUAL)
    assert session.difficulty is None
    session.submit_move("e5")
    session.note_difficulty({**CASUAL, "verbosity": "low"})
    assert session.difficulty == CASUAL


def test_a_later_setting_does_not_rename_the_opponent():
    """The rule for a game played at several strengths: the first one."""
    session = _opened()
    session.note_difficulty(CASUAL)
    session.submit_move("Nf3")
    session.submit_move("Nc6")
    session.note_difficulty(ELO_1400)
    assert session.difficulty == CASUAL


def test_engine_moving_first_records_on_its_opening_move():
    session = GameSession(player_color="black")
    session.submit_move("e4")
    session.note_difficulty(ELO_1400)
    assert session.difficulty == ELO_1400


def test_taking_back_every_engine_move_forgets_the_strength():
    session = _opened()
    session.note_difficulty(CASUAL)
    session.undo(1)
    assert session.difficulty is None
    session.submit_move("c5")
    session.note_difficulty(ELO_1400)
    assert session.difficulty == ELO_1400


def test_a_new_game_forgets_the_strength():
    session = _opened()
    session.note_difficulty(CASUAL)
    session.new_game()
    assert session.difficulty is None


def test_difficulty_survives_a_save_and_a_resume(tmp_path):
    session = _opened()
    session.note_difficulty(ELO_1400)
    session.save(tmp_path / "game.json")
    assert GameSession.load(tmp_path / "game.json").difficulty == ELO_1400


def test_save_from_before_the_difficulty_existed_loads():
    legacy = _opened().to_dict()
    del legacy["difficulty"]
    restored = GameSession.from_dict(legacy)
    assert restored.difficulty is None
    assert restored.fen() == _opened().fen()


@pytest.mark.parametrize(
    "bad",
    [
        "casual",
        {"tier": "casual"},
        {"tier": 3, "skill_level": None, "elo": None},
        {"tier": None, "skill_level": True, "elo": None},
        {**CASUAL, "voice": "on"},
    ],
)
def test_a_difficulty_the_app_could_not_have_written_is_refused(bad):
    with pytest.raises(ValueError):
        GameSession.from_dict(GameSession().to_dict() | {"difficulty": bad})


def test_the_engines_reply_records_the_strength_it_was_played_at():
    """Recorded where every engine move is played (`TurnCoordinator`), so the
    tool, a board drag and MCP all record it; changing the difficulty after
    the game leaves the game's record alone, and so does a save and resume."""
    ctx = ToolContext(session=GameSession(), engine=FakeEngine())
    ctx.settings.tier = None
    ctx.settings.elo = 1400
    TurnCoordinator(ctx).play_exchange("e4")
    ctx.settings.elo = None
    ctx.settings.tier = "advanced"
    assert ctx.session.difficulty == ELO_1400
    assert GameSession.from_dict(ctx.session.to_dict()).difficulty == ELO_1400


def test_the_opponent_is_named_by_the_strength_the_game_was_played_at():
    """Change difficulty after a game and its PGN misnamed the opponent
    (#460): the name tag was built from the setting at export time. Both
    exports name it alike, since both compose through `pgn_headers`."""
    ctx = ToolContext(session=GameSession(), engine=FakeEngine())
    ctx.settings.tier = None
    ctx.settings.elo = 1400
    registry = build_registry(ctx)
    client = TestClient(create_app(ctx, registry=registry))
    assert registry.dispatch("make_move", {"move": "e4"})["ok"]
    ctx.settings.elo = None
    ctx.settings.tier = "advanced"
    pgn = client.get("/api/game/pgn").json()["pgn"]
    assert parse(pgn).headers["Black"] == "Glitch (Stockfish, 1400 Elo)"
    assert registry.dispatch("export_pgn", {})["pgn"] == pgn


def test_the_engines_opening_move_records_the_strength():
    ctx = ToolContext(
        session=GameSession(player_color="black"), engine=FakeEngine("e2e4")
    )
    TurnCoordinator(ctx).settle_engine_turn()
    assert ctx.session.difficulty == CASUAL
