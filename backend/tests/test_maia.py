"""Maia-3 behind the tiers below `maximum` (#510, docs/maia-tiers.md).

No Maia process here: a scripted UCI double stands in for it, and another
for Stockfish, so these pin what the code decides. Which tier plays Maia and
at what settings, the game's moves handed over rather than a bare FEN, a dead
process started again once, and Stockfish at the tier's own settings
whenever Maia is missing or fails. How strong each setting really plays is
measured, not tested (scripts/calibrate_difficulty.py).
"""

from __future__ import annotations

import chess
import chess.engine
import pytest

from chessapp import maia as maia_module
from chessapp.engine import DIFFICULTY_TIERS, EnginePlayer, tier_plays_maia
from chessapp.game import GameSession
from chessapp.maia import MAIA_ARGS, MaiaEngine, MaiaSettings, game_board, maia_from_env


class FakeMaiaUci:
    """A Maia `SimpleEngine` stand-in: records each `play` and answers with
    `reply` (UCI) or the first legal move; dies on its first `dies` calls."""

    def __init__(self, reply: str | None = None, dies: int = 0):
        self.reply = reply
        self.dies = dies
        self.plays: list[tuple[chess.Board, dict]] = []
        self.closed = False

    def play(self, board, limit, *, options=None):
        if self.dies:
            self.dies -= 1
            raise chess.engine.EngineTerminatedError("maia died")
        self.plays.append((board.copy(), dict(options or {})))
        move = (
            chess.Move.from_uci(self.reply)
            if self.reply
            else next(iter(board.legal_moves))
        )
        return chess.engine.PlayResult(move, None)

    def close(self):
        self.closed = True

    def quit(self):
        self.closed = True


class FakeStockfish:
    """Stockfish's double: records configure calls and plays its first legal
    move, or for `analyse` (the sampler tiers) scores every move the same."""

    def __init__(self):
        self.configured: list[dict] = []
        self.played = 0
        self.closed = False

    def configure(self, options):
        self.configured.append(dict(options))

    def play(self, board, limit):
        self.played += 1
        return chess.engine.PlayResult(next(iter(board.legal_moves)), None)

    def analyse(self, board, limit, multipv=None):
        self.played += 1
        move = next(iter(board.legal_moves))
        return [
            {
                "pv": [move],
                "score": chess.engine.PovScore(chess.engine.Cp(0), board.turn),
            }
        ]

    def quit(self):
        self.closed = True


def maia_engine(*engines) -> tuple[MaiaEngine, list]:
    queue = list(engines)
    launched: list = []

    def open_engine(command):
        engine = queue.pop(0)
        if isinstance(engine, Exception):
            raise engine
        launched.append((command, engine))
        return engine

    return MaiaEngine("/opt/maia/bin/maia3-uci", open_engine=open_engine), launched


def player(maia: MaiaEngine | None) -> tuple[EnginePlayer, FakeStockfish]:
    stockfish = FakeStockfish()
    return EnginePlayer(open_engine=lambda path: stockfish, maia=maia), stockfish


# --- the tier map ------------------------------------------------------------


def test_the_tiers_below_maximum_play_maia_at_the_measured_ratings():
    ratings = {
        name: profile.maia.elo
        for name, profile in DIFFICULTY_TIERS.items()
        if profile.maia is not None
    }
    assert ratings == {
        "beginner": 600,
        "casual": 1000,
        "intermediate": 1400,
        "advanced": 2200,
        # The mover's stand-in is `beginner`, Maia included (decision 4).
        "glitch": 600,
    }
    assert DIFFICULTY_TIERS["maximum"].maia is None
    assert all(
        (p.maia.temperature, p.maia.top_p) == (1.0, 0.9)
        for p in DIFFICULTY_TIERS.values()
        if p.maia is not None
    )


def test_tier_plays_maia():
    assert tier_plays_maia("casual")
    assert not tier_plays_maia("maximum")
    assert not tier_plays_maia(None)
    assert not tier_plays_maia("no such tier")


def test_settings_become_maias_uci_options():
    assert MaiaSettings(1000).options() == {
        "Elo": 1000,
        "Temperature": "1.0",
        "TopP": "0.9",
    }


# --- the move ----------------------------------------------------------------


def test_maia_is_handed_the_games_moves_from_where_it_began():
    start = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
    session = GameSession(fen=start)
    session.submit_move("e5")
    session.submit_move("Nf3")
    board = game_board(session)
    assert board.root().fen() == start
    assert [m.uci() for m in board.move_stack] == ["e7e5", "g1f3"]
    assert board.fen() == session.fen()


def test_a_maia_tier_plays_maias_move_with_its_settings():
    uci = FakeMaiaUci(reply="e7e5")
    engine, launched = maia_engine(uci)
    play, stockfish = player(engine)
    play.set_tier("casual")
    session = GameSession()
    session.submit_move("e4")
    assert play.choose_move(session) == "e7e5"
    board, options = uci.plays[0]
    assert [m.uci() for m in board.move_stack] == ["e2e4"]
    assert options == {"Elo": 1000, "Temperature": "1.0", "TopP": "0.9"}
    assert stockfish.played == 0
    assert launched[0][0] == ["/opt/maia/bin/maia3-uci", *MAIA_ARGS]


def test_the_glitch_fallback_plays_maia_as_beginner():
    uci = FakeMaiaUci()
    play, _ = player(maia_engine(uci)[0])
    play.set_tier("glitch")
    play.choose_move(GameSession())
    assert uci.plays[0][1]["Elo"] == 600


@pytest.mark.parametrize(
    "set_strength",
    [
        lambda p: p.set_tier("maximum"),
        lambda p: p.set_elo(1600),
        lambda p: p.set_skill_level(5),
    ],
    ids=["maximum", "raw elo", "raw skill"],
)
def test_maximum_and_the_raw_knobs_play_stockfish(set_strength):
    uci = FakeMaiaUci()
    play, stockfish = player(maia_engine(uci)[0])
    play.set_tier("casual")
    set_strength(play)
    play.choose_move(GameSession())
    assert uci.plays == []
    assert stockfish.played == 1


def test_without_maia_a_tier_plays_its_stockfish_settings():
    play, stockfish = player(None)
    assert not play.has_maia
    play.set_tier("intermediate")
    play.choose_move(GameSession())
    assert stockfish.configured[-1] == {"UCI_LimitStrength": True, "UCI_Elo": 1500}
    assert stockfish.played == 1


def test_a_maia_tier_still_sets_stockfish_up_as_its_stand_in():
    play, stockfish = player(maia_engine(FakeMaiaUci())[0])
    assert play.has_maia
    play.set_tier("advanced")
    assert stockfish.configured[-1] == {"UCI_LimitStrength": True, "UCI_Elo": 2000}


# --- when Maia fails ---------------------------------------------------------


def test_a_dead_maia_is_started_again_once_and_the_move_repeated():
    first, second = FakeMaiaUci(dies=1), FakeMaiaUci(reply="d7d5")
    engine, launched = maia_engine(first, second)
    play, stockfish = player(engine)
    play.set_tier("beginner")
    session = GameSession()
    session.submit_move("d4")
    assert play.choose_move(session) == "d7d5"
    assert len(launched) == 2
    assert first.closed
    assert stockfish.played == 0


def test_a_maia_that_will_not_come_back_hands_the_move_to_stockfish():
    engine, _ = maia_engine(FakeMaiaUci(dies=1), FileNotFoundError("gone"))
    play, stockfish = player(engine)
    play.set_tier("intermediate")
    move = play.choose_move(GameSession())
    assert chess.Move.from_uci(move) in chess.Board().legal_moves
    assert stockfish.played == 1


def test_closing_the_player_closes_maia():
    uci = FakeMaiaUci()
    play, stockfish = player(maia_engine(uci)[0])
    play.close()
    assert uci.closed
    assert stockfish.closed


# --- attaching it ------------------------------------------------------------


def test_no_chessapp_maia_means_no_maia(monkeypatch):
    monkeypatch.delenv("CHESSAPP_MAIA", raising=False)
    assert maia_from_env() is None


def test_a_maia_that_will_not_start_means_no_maia(monkeypatch):
    def broken(path, checkpoint):
        raise FileNotFoundError(path)

    monkeypatch.setenv("CHESSAPP_MAIA", "/nowhere/maia3-uci")
    monkeypatch.setattr(maia_module, "MaiaEngine", broken)
    assert maia_from_env() is None


def test_chessapp_maia_attaches_it_with_its_local_weights(monkeypatch):
    made = []
    monkeypatch.setenv("CHESSAPP_MAIA", "/opt/maia/bin/maia3-uci")
    monkeypatch.setenv("CHESSAPP_MAIA_CHECKPOINT", "/opt/maia/maia3-79m.pt")
    monkeypatch.setattr(
        maia_module, "MaiaEngine", lambda *args: made.append(args) or args[0]
    )
    assert maia_from_env() == "/opt/maia/bin/maia3-uci"
    assert made == [("/opt/maia/bin/maia3-uci", "/opt/maia/maia3-79m.pt")]


def test_local_weights_are_passed_as_the_checkpoint():
    launched = []
    MaiaEngine(
        "maia3-uci",
        "/opt/maia/maia3-79m.pt",
        open_engine=lambda command: launched.append(command) or FakeMaiaUci(),
    )
    assert launched == [
        ["maia3-uci", *MAIA_ARGS, "--checkpoint-path", "/opt/maia/maia3-79m.pt"]
    ]
