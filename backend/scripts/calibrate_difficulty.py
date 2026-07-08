"""Calibrate the difficulty tiers by playing them against each other.

The sub-floor tiers (`beginner`, `casual`) are weakened by the weighted
sampler in `engine.py`, whose `temperature` knobs can't be read off a UCI
Elo — they have to be measured. This script is that measurement: it runs a
head-to-head gauntlet between tiers (and the 1320 UCI floor) so you can
confirm the ordering `beginner < casual < intermediate` and re-tune the
temperatures in `DIFFICULTY_TIERS` against real games.

Run from `backend/` with a Stockfish binary on PATH:

    python scripts/calibrate_difficulty.py            # default gauntlet
    python scripts/calibrate_difficulty.py --games 12 # more games, less noise

It is a developer tool, not part of the app or the test suite.
"""

import argparse
import random

from chessapp.engine import EnginePlayer
from chessapp.game import GameSession


def _configure(player: EnginePlayer, spec: str) -> None:
    """A contender is either a tier name or ``elo:<n>`` for a raw UCI anchor."""
    if spec.startswith("elo:"):
        player.set_elo(int(spec.split(":", 1)[1]))
    else:
        player.set_tier(spec)


def play_game(white: EnginePlayer, black: EnginePlayer, max_plies: int = 160) -> str:
    """Play one game to completion; return the result string ("1-0"/"0-1"/"1/2-1/2")."""
    session = GameSession()
    for _ in range(max_plies):
        if session.is_game_over():
            break
        mover = white if session.turn == "white" else black
        mover.play_move(session)
    outcome = session.outcome()
    return outcome.result if outcome is not None else "1/2-1/2"


def match(a_spec: str, b_spec: str, games: int, seed: int) -> float:
    """A's score out of `games`, alternating colours. Shared seed per side so
    a sampler tier's blunders are reproducible across runs."""
    a_points = 0.0
    with (
        EnginePlayer(rng=random.Random(seed)) as a,
        EnginePlayer(rng=random.Random(seed + 1)) as b,
    ):
        _configure(a, a_spec)
        _configure(b, b_spec)
        for g in range(games):
            a_white = g % 2 == 0
            result = play_game(a, b) if a_white else play_game(b, a)
            if result == "1/2-1/2":
                a_points += 0.5
            elif (result == "1-0") == a_white:
                a_points += 1.0
    return a_points


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=6, help="games per pairing")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    # Each pairing should show the first contender clearly weaker (low score).
    pairings = [
        ("beginner", "casual"),
        ("casual", "elo:1320"),
        ("beginner", "elo:1320"),
    ]
    print(f"gauntlet: {args.games} games per pairing (alternating colours)\n")
    for i, (a, b) in enumerate(pairings):
        score = match(a, b, args.games, args.seed + 10 * i)
        print(f"  {a:>10s} vs {b:<10s}  {a} scored {score:g}/{args.games}")


if __name__ == "__main__":
    main()
