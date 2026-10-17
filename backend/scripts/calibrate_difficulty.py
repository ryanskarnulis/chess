"""Calibrate the difficulty tiers by playing them against each other.

The sub-floor tiers (`beginner`, `casual`) are weakened by the weighted
sampler in `engine.py`, whose `temperature` knobs can't be read off a UCI
Elo — they have to be measured. This script is that measurement: it runs a
head-to-head gauntlet between tiers (and the 1320 UCI floor) so you can
confirm the ordering `beginner < casual < intermediate` and re-tune the
temperatures in `DIFFICULTY_TIERS` against real games.

A contender is a tier name, ``elo:<n>`` for a raw Stockfish UCI anchor, or
``maia:<elo>[:<temperature>[:<top_p>]]`` for Maia-3 (#511), a human-move model run as
its own UCI process (``--maia``, default ``maia3-uci --model maia3-79m`` on
CPU). Maia is handed the game's moves, not a bare FEN, because it reads the
last few positions as well as the current one. Its temperature defaults to
1.0 (sampling from its policy); 0 is its single most likely move. `top_p`
(default 1.0, off) samples only from the most likely moves that together hold
that share of the policy, which cuts the long tail a flat low-rated policy has.

Run from `backend/` with a Stockfish binary on PATH:

    python scripts/calibrate_difficulty.py            # default gauntlet
    python scripts/calibrate_difficulty.py --games 12 # more games, less noise
    python scripts/calibrate_difficulty.py --pair maia:1100 casual \\
        --pair maia:1500 intermediate --jobs 4         # Maia against the tiers

It is a developer tool, not part of the app or the test suite.
"""

import argparse
import math
import os
import random
import shlex
from concurrent.futures import ProcessPoolExecutor
from typing import Protocol

import chess
import chess.engine

from chessapp.engine import EnginePlayer
from chessapp.game import GameSession

DEFAULT_MAIA = "maia3-uci --model maia3-79m --device cpu --no-use-amp --use-uci-history"


class Player(Protocol):
    def play_move(self, session: GameSession) -> object: ...

    def close(self) -> None: ...


class MaiaPlayer:
    """Maia-3 over UCI at one Elo, temperature and top-p. Its move goes
    through the session's legality gate like the engine's."""

    def __init__(
        self, command: str, elo: int, temperature: float, top_p: float, threads: int
    ):
        env = {**os.environ, "OMP_NUM_THREADS": str(threads)}
        self._engine = chess.engine.SimpleEngine.popen_uci(
            shlex.split(command), env=env, stderr=open(os.devnull, "w")
        )
        self._engine.configure(
            {"Elo": elo, "Temperature": str(temperature), "TopP": str(top_p)}
        )

    def play_move(self, session: GameSession) -> object:
        board = chess.Board(session.position_fens()[0])
        for san in session.move_history():
            board.push_san(san)
        result = self._engine.play(board, chess.engine.Limit(nodes=1))
        if result.move is None:
            raise ValueError("maia returned no move")
        return session.submit_move(result.move.uci())

    def close(self) -> None:
        self._engine.quit()


def _player(spec: str, seed: int, maia: str, threads: int) -> Player:
    """A contender is a tier name, ``elo:<n>`` for a raw UCI anchor, or
    ``maia:<elo>[:<temperature>[:<top_p>]]``."""
    if spec.startswith("maia:"):
        parts = spec.split(":")
        temperature = float(parts[2]) if len(parts) > 2 else 1.0
        top_p = float(parts[3]) if len(parts) > 3 else 1.0
        return MaiaPlayer(maia, int(parts[1]), temperature, top_p, threads)
    player = EnginePlayer(rng=random.Random(seed))
    if spec.startswith("elo:"):
        player.set_elo(int(spec.split(":", 1)[1]))
    else:
        player.set_tier(spec)
    return player


def play_game(white: Player, black: Player, max_plies: int = 160) -> str | None:
    """Play one game; return its result ("1-0"/"0-1"/"1/2-1/2"), or None if
    it was still going at `max_plies`."""
    session = GameSession()
    for _ in range(max_plies):
        if session.is_game_over():
            break
        mover = white if session.turn == "white" else black
        mover.play_move(session)
    outcome = session.outcome()
    return outcome.result if outcome is not None else None


def match(
    a_spec: str,
    b_spec: str,
    games: int,
    seed: int,
    maia: str = DEFAULT_MAIA,
    threads: int = 2,
    max_plies: int = 160,
) -> tuple[float, int]:
    """A's score out of `games`, alternating colours, and how many games were
    unfinished (scored as draws). Shared seed per side so a sampler tier's
    blunders are reproducible across runs."""
    a_points = 0.0
    unfinished = 0
    a = _player(a_spec, seed, maia, threads)
    b = _player(b_spec, seed + 1, maia, threads)
    try:
        for g in range(games):
            a_white = g % 2 == 0
            result = (
                play_game(a, b, max_plies) if a_white else play_game(b, a, max_plies)
            )
            if result is None:
                unfinished += 1
                a_points += 0.5
            elif result == "1/2-1/2":
                a_points += 0.5
            elif (result == "1-0") == a_white:
                a_points += 1.0
    finally:
        a.close()
        b.close()
    return a_points, unfinished


def elo_gap(points: float, games: int) -> str:
    """A's rating minus B's implied by the score, on the logistic Elo curve.
    A sweep or a whitewash has no point estimate."""
    if points <= 0:
        return "-inf"
    if points >= games:
        return "+inf"
    share = points / games
    return f"{400 * math.log10(share / (1 - share)):+.0f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=6, help="games per pairing")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--pair",
        nargs=2,
        action="append",
        metavar=("A", "B"),
        help="a pairing to play (repeatable); replaces the default gauntlet",
    )
    parser.add_argument("--jobs", type=int, default=1, help="pairings run at once")
    parser.add_argument("--maia", default=DEFAULT_MAIA, help="the Maia UCI command")
    parser.add_argument("--maia-threads", type=int, default=2)
    parser.add_argument("--max-plies", type=int, default=160)
    args = parser.parse_args()

    # Each pairing should show the first contender clearly weaker (low score).
    pairings = (
        [tuple(pair) for pair in args.pair]
        if args.pair
        else [
            ("beginner", "casual"),
            ("casual", "elo:1320"),
            ("beginner", "elo:1320"),
        ]
    )
    print(f"gauntlet: {args.games} games per pairing (alternating colours)\n")
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures = [
            pool.submit(
                match,
                a,
                b,
                args.games,
                args.seed + 10 * i,
                args.maia,
                args.maia_threads,
                args.max_plies,
            )
            for i, (a, b) in enumerate(pairings)
        ]
        for (a, b), future in zip(pairings, futures, strict=True):
            score, unfinished = future.result()
            note = f"  ({unfinished} unfinished, scored as draws)" if unfinished else ""
            print(
                f"  {a:>14s} vs {b:<14s}  {a} scored {score:g}/{args.games}"
                f"  gap {elo_gap(score, args.games)}{note}",
                flush=True,
            )


if __name__ == "__main__":
    main()
