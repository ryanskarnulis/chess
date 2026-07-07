"""Stockfish bridge: engine moves at configurable difficulty.

Tests that need a live Stockfish are skipped when the binary is absent;
CI installs it and runs them. Difficulty validation is pure and always
tested.
"""

import shutil

import chess
import pytest

from chessapp.engine import (
    DIFFICULTY_TIERS,
    ELO_MAX,
    ELO_MIN,
    EnginePlayer,
    validate_elo,
    validate_skill_level,
    validate_tier,
)
from chessapp.game import GameSession

requires_stockfish = pytest.mark.skipif(
    shutil.which("stockfish") is None, reason="stockfish binary not installed"
)

# --- difficulty validation (pure, no binary) ------------------------------


def test_skill_level_bounds_accepted():
    validate_skill_level(0)
    validate_skill_level(20)


@pytest.mark.parametrize("level", [-1, 21, 3.5, "ten", None])
def test_bad_skill_level_rejected(level):
    with pytest.raises(ValueError):
        validate_skill_level(level)


def test_elo_bounds_accepted():
    validate_elo(ELO_MIN)
    validate_elo(ELO_MAX)


@pytest.mark.parametrize("elo", [ELO_MIN - 1, ELO_MAX + 1, 1500.5, "gm", None])
def test_bad_elo_rejected(elo):
    with pytest.raises(ValueError):
        validate_elo(elo)


# --- difficulty tiers (pure, no binary) ------------------------------------


@pytest.mark.parametrize(
    "name", ["beginner", "casual", "intermediate", "advanced", "maximum"]
)
def test_every_tier_resolves(name):
    assert validate_tier(name).name == name


@pytest.mark.parametrize("name", ["impossible", "", "Beginner", None, 3])
def test_unknown_tier_rejected(name):
    with pytest.raises(ValueError):
        validate_tier(name)


def test_sub_floor_tiers_weaken_beyond_uci_options():
    # Stockfish's UCI_Elo floor is ~1320 and Skill Level 0 still plays
    # ~1300+, so the ~500/~1000 tiers must starve the search and inject
    # blunders — UCI knobs alone cannot produce them.
    beginner = DIFFICULTY_TIERS["beginner"]
    assert beginner.skill_level == 0
    assert beginner.max_nodes is not None
    assert beginner.blunder_chance > 0
    casual = DIFFICULTY_TIERS["casual"]
    assert casual.max_nodes is not None
    assert casual.blunder_chance > 0
    assert beginner.max_nodes < casual.max_nodes
    assert beginner.blunder_chance > casual.blunder_chance


def test_upper_tiers_map_to_plain_uci_strength():
    assert DIFFICULTY_TIERS["intermediate"].elo == 1500
    assert DIFFICULTY_TIERS["advanced"].elo == 2000
    maximum = DIFFICULTY_TIERS["maximum"]
    assert maximum.skill_level == 20
    assert maximum.max_nodes is None
    assert maximum.blunder_chance == 0


class ForcedRng:
    """RNG double: fixed `random()` roll; `choice` picks first or forbids."""

    def __init__(self, roll: float, allow_choice: bool = True):
        self._roll = roll
        self._allow_choice = allow_choice

    def random(self) -> float:
        return self._roll

    def choice(self, seq):
        assert self._allow_choice, "engine took the blunder path unexpectedly"
        return seq[0]


# --- live engine ----------------------------------------------------------


@pytest.fixture(scope="module")
def engine():
    if shutil.which("stockfish") is None:
        pytest.skip("stockfish binary not installed")
    with EnginePlayer() as player:
        yield player


@requires_stockfish
def test_engine_choose_move_is_legal(engine):
    session = GameSession()
    uci = engine.choose_move(session)
    assert session.submit_move(uci).legal


@requires_stockfish
def test_engine_play_move_advances_session(engine):
    session = GameSession()
    session.submit_move("e4")
    result = engine.play_move(session)
    assert result.legal
    assert session.turn == "white"
    assert len(session.move_history()) == 2


@requires_stockfish
def test_engine_finds_mate_in_one(engine):
    # White mates with Qxf7#; full-strength engine must find it.
    session = GameSession(
        fen="r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 0 1"
    )
    engine.set_skill_level(20)
    result = engine.play_move(session)
    assert result.legal
    assert result.game_over
    assert session.outcome().termination == "checkmate"


@requires_stockfish
def test_engine_plays_at_low_skill(engine):
    session = GameSession()
    engine.set_skill_level(0)
    assert engine.play_move(session).legal


@requires_stockfish
def test_engine_plays_at_limited_elo(engine):
    session = GameSession()
    engine.set_elo(ELO_MIN)
    assert engine.play_move(session).legal


@requires_stockfish
@pytest.mark.parametrize(
    "tier", ["beginner", "casual", "intermediate", "advanced", "maximum"]
)
def test_engine_plays_legal_moves_at_every_tier(engine, tier):
    session = GameSession()
    engine.set_tier(tier)
    assert engine.play_move(session).legal


@requires_stockfish
def test_blunder_roll_plays_a_random_legal_move():
    session = GameSession()
    with EnginePlayer(rng=ForcedRng(roll=0.0)) as player:
        player.set_tier("beginner")
        uci = player.choose_move(session)
    first_legal = next(iter(chess.Board().legal_moves)).uci()
    assert uci == first_legal


@requires_stockfish
def test_no_blunder_roll_consults_the_engine():
    session = GameSession()
    with EnginePlayer(rng=ForcedRng(roll=1.0, allow_choice=False)) as player:
        player.set_tier("beginner")
        assert session.submit_move(player.choose_move(session)).legal


@requires_stockfish
def test_raw_strength_setting_clears_tier_weakening():
    # A roll of 0.0 blunders whenever any blunder chance survives, so the
    # ForcedRng's forbidden `choice` proves set_skill_level cleared it.
    session = GameSession()
    with EnginePlayer(rng=ForcedRng(roll=0.0, allow_choice=False)) as player:
        player.set_tier("beginner")
        player.set_skill_level(20)
        assert session.submit_move(player.choose_move(session)).legal


@requires_stockfish
def test_choose_move_on_finished_game_raises(engine):
    session = GameSession()
    session.resign("white")
    with pytest.raises(ValueError):
        engine.choose_move(session)


@requires_stockfish
def test_engine_does_not_mutate_session_on_choose(engine):
    session = GameSession()
    before = session.fen()
    engine.choose_move(session)
    assert session.fen() == before


@requires_stockfish
def test_context_manager_closes_engine():
    with EnginePlayer() as player:
        assert player.choose_move(GameSession())
    # after close, further use fails
    with pytest.raises(chess_engine_closed_errors()):
        player.choose_move(GameSession())


def chess_engine_closed_errors():
    import chess.engine

    return (chess.engine.EngineError, BrokenPipeError, ValueError)
