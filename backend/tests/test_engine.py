"""Stockfish bridge: engine moves at configurable difficulty.

Tests that need a live Stockfish are skipped when the binary is absent;
CI installs it and runs them. Difficulty validation is pure and always
tested.
"""

import random
import shutil

import pytest

from chessapp.engine import (
    DIFFICULTY_TIERS,
    ELO_MAX,
    ELO_MIN,
    MAX_SAMPLE_LOSS,
    EnginePlayer,
    sample_weighted,
    validate_elo,
    validate_skill_level,
    validate_tier,
)
from chessapp.game import GameSession


class RollRng:
    """RNG double for the weighted sampler: `random()` returns a fixed roll
    in [0, 1). The sampler multiplies it by the total weight, so roll=0.0
    always lands on the best candidate and a roll just under 1.0 lands on
    the worst."""

    def __init__(self, roll: float):
        self._roll = roll

    def random(self) -> float:
        return self._roll


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


def test_sub_floor_tiers_weaken_by_weighted_sampling():
    # Stockfish's UCI_Elo floor is ~1320 and Skill Level 0 still plays
    # ~1300+, so the ~500/~1000 tiers can't come from UCI knobs. They pick a
    # move by sampling all legal moves weighted by how much each loses vs the
    # best (a higher `temperature` = flatter = sloppier). Beginner is the
    # sloppier of the two, so it carries the higher temperature.
    beginner = DIFFICULTY_TIERS["beginner"]
    casual = DIFFICULTY_TIERS["casual"]
    for tier in (beginner, casual):
        assert tier.temperature is not None
        assert tier.sample_depth is not None
        assert tier.elo is None and tier.skill_level is None
    assert beginner.temperature > casual.temperature


def test_upper_tiers_map_to_plain_uci_strength():
    assert DIFFICULTY_TIERS["intermediate"].elo == 1500
    assert DIFFICULTY_TIERS["advanced"].elo == 2000
    maximum = DIFFICULTY_TIERS["maximum"]
    assert maximum.skill_level == 20
    assert maximum.temperature is None


# --- weighted sampler (pure, no binary) -----------------------------------


def test_sample_weighted_low_roll_picks_best():
    # Candidates are best-first (loss 0 is the engine's top move); a roll of
    # 0.0 always lands in the first bucket.
    assert sample_weighted([0, 50, 300], temperature=200, rng=RollRng(0.0)) == 0


def test_sample_weighted_high_roll_reaches_worst():
    # A roll just under 1.0 exhausts every earlier bucket and lands on the
    # worst candidate — the sampler can play a genuine blunder.
    losses = [0, 50, 300]
    assert sample_weighted(losses, temperature=200, rng=RollRng(0.999)) == 2


def test_low_temperature_concentrates_on_the_best():
    # As temperature -> 0 the best move dominates, so even a large roll still
    # picks it: at T=1 a 300cp-worse move has weight e**-300 ~ 0.
    assert sample_weighted([0, 300], temperature=1, rng=RollRng(0.999)) == 0


def test_higher_temperature_spreads_weight_to_worse_moves():
    # The same middling roll that the best move still wins at low temperature
    # tips to a worse move once temperature flattens the distribution.
    losses = [0, 120]
    assert sample_weighted(losses, temperature=20, rng=RollRng(0.8)) == 0
    assert sample_weighted(losses, temperature=400, rng=RollRng(0.8)) == 1


def test_sample_weighted_caps_catastrophic_losses():
    # Losses past MAX_SAMPLE_LOSS (e.g. walking into mate, scored in the tens
    # of thousands) are clamped, so a hopeless move keeps a small but nonzero
    # weight instead of underflowing to exactly zero and never being played.
    huge = MAX_SAMPLE_LOSS * 100
    assert sample_weighted([0, huge], temperature=300, rng=RollRng(0.999)) == 1


def test_sample_weighted_rejects_nonpositive_temperature():
    with pytest.raises(ValueError):
        sample_weighted([0, 50], temperature=0, rng=RollRng(0.5))


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
def test_sampler_tier_is_deterministic_under_a_seeded_rng():
    # Same seed, same position => same sampled move, so a weak tier is
    # reproducible (and its blunders aren't a live coin flip).
    session = GameSession()
    moves = []
    for _ in range(2):
        with EnginePlayer(rng=random.Random(1234)) as player:
            player.set_tier("beginner")
            moves.append(player.choose_move(session))
    assert moves[0] == moves[1]
    assert session.submit_move(moves[0]).legal


@requires_stockfish
def test_raw_strength_setting_clears_the_sampler():
    # Switching to a raw skill/elo escape hatch must drop the tier's sampler
    # so choose_move consults the engine directly, not the weighted pool.
    with EnginePlayer(rng=random.Random(0)) as player:
        player.set_tier("beginner")
        assert player._sample_temperature is not None
        player.set_skill_level(20)
        assert player._sample_temperature is None
        player.set_tier("casual")
        player.set_elo(ELO_MIN)
        assert player._sample_temperature is None


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
