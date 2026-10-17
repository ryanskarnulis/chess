"""`scripts/calibrate_difficulty.py` — the tier gauntlet (#511).

The gauntlet itself needs Stockfish and Maia; what it decides is tested here
without them: how a contender spec becomes a player, what an unfinished game
counts as, how a match is scored, and the Elo gap read from a score.
"""

from __future__ import annotations

import pytest

import calibrate_difficulty
from calibrate_difficulty import elo_gap, match, play_game
from chessapp.game import GameSession


class _FirstLegal:
    """Plays the first legal move, so a game is deterministic and quick."""

    def play_move(self, session: GameSession) -> object:
        return session.submit_move(session.legal_moves()[0])

    def close(self) -> None:
        pass


class _Moves:
    """Plays a fixed list of moves, one per call."""

    def __init__(self, moves: list[str]):
        self._moves = iter(moves)

    def play_move(self, session: GameSession) -> object:
        return session.submit_move(next(self._moves))

    def close(self) -> None:
        pass


def test_a_game_still_going_at_the_cap_is_unfinished() -> None:
    assert play_game(_FirstLegal(), _FirstLegal(), max_plies=4) is None


def test_a_finished_game_returns_its_result() -> None:
    # Fool's mate: Black mates on ply 4.
    white = _Moves(["f3", "g4"])
    black = _Moves(["e5", "Qh4#"])
    assert play_game(white, black) == "0-1"


@pytest.mark.parametrize(
    ("spec", "elo", "temperature", "top_p"),
    [
        ("maia:1500", 1500, 1.0, 1.0),
        ("maia:600:0", 600, 0.0, 1.0),
        ("maia:600:1:0.9", 600, 1.0, 0.9),
    ],
)
def test_a_maia_spec_sets_elo_temperature_and_top_p(
    monkeypatch: pytest.MonkeyPatch,
    spec: str,
    elo: int,
    temperature: float,
    top_p: float,
) -> None:
    made = []

    def fake(command, elo, temperature, top_p, threads):
        made.append((command, elo, temperature, top_p, threads))
        return _FirstLegal()

    monkeypatch.setattr(calibrate_difficulty, "MaiaPlayer", fake)
    calibrate_difficulty._player(spec, seed=1, maia="maia3-uci", threads=3)
    assert made == [("maia3-uci", elo, temperature, top_p, 3)]


def test_a_match_alternates_colours_and_scores_unfinished_as_draws(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    results = iter(["1-0", "1-0", None, "1/2-1/2"])
    seen = []

    def fake_game(white, black, max_plies):
        seen.append((white.name, black.name))
        return next(results)

    class _Named(_FirstLegal):
        def __init__(self, name: str):
            self.name = name

    monkeypatch.setattr(calibrate_difficulty, "_player", lambda spec, *_: _Named(spec))
    monkeypatch.setattr(calibrate_difficulty, "play_game", fake_game)
    points, unfinished = match("a", "b", games=4, seed=1)
    assert seen == [("a", "b"), ("b", "a"), ("a", "b"), ("b", "a")]
    # A won as White, lost as Black, then two half points.
    assert (points, unfinished) == (2.0, 1)


@pytest.mark.parametrize(
    ("points", "games", "gap"),
    [
        (10, 20, "+0"),
        (15, 20, "+191"),
        (5, 20, "-191"),
        (0, 20, "-inf"),
        (20, 20, "+inf"),
    ],
)
def test_the_elo_gap_reads_the_score_on_the_logistic_curve(
    points: float, games: int, gap: str
) -> None:
    assert elo_gap(points, games) == gap
