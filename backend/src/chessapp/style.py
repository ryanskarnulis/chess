"""Personality move style: deterministic bias over MultiPV candidates.

Each personality gets a `StyleProfile` describing *which* of Stockfish's
near-best moves it plays — flashier personalities lean toward captures and
checks, the beginner leans toward the weakest still-reasonable option, and
the precise ones just play the best move. This layer only chooses among
engine-vetted candidates and every choice still enters the game through
`GameSession.submit_move`, so personality can never produce an illegal or
truly bad move (`tolerance_cp` caps how much it may give up, and a forced
mate is never passed over).

Deliberately deterministic (no RNG): same position + personality = same
move, which keeps behavior reproducible and testable. Variety comes from
Stockfish's own candidate ordering changing as the game changes.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from chessapp.engine import pov_cp

if TYPE_CHECKING:
    from chessapp.engine import CandidateMove, EnginePlayer
    from chessapp.game import GameSession, MoveResult


@dataclass(frozen=True)
class StyleProfile:
    """How a personality picks among MultiPV candidates.

    `multipv` is how many candidates to consider (1 = always the engine's
    best, no analysis detour). `tolerance_cp` is the most a candidate may
    score below the best and still be eligible. `prefers` picks among the
    eligible: "best" (top candidate), "aggressive" (first capture/check,
    else best), "modest" (weakest eligible — the beginner's move).
    """

    multipv: int = 1
    tolerance_cp: int = 0
    prefers: str = "best"


DEFAULT_PROFILE = StyleProfile()

PROFILES: dict[str, StyleProfile] = {
    "friendly_rival": DEFAULT_PROFILE,
    "calm_coach": DEFAULT_PROFILE,
    "grandmaster": DEFAULT_PROFILE,
    "silent_assassin": DEFAULT_PROFILE,
    "trash_talker": StyleProfile(multipv=4, tolerance_cp=50, prefers="aggressive"),
    "villain": StyleProfile(multipv=4, tolerance_cp=60, prefers="aggressive"),
    "streamer": StyleProfile(multipv=4, tolerance_cp=80, prefers="aggressive"),
    "beginner_bot": StyleProfile(multipv=5, tolerance_cp=150, prefers="modest"),
}


def profile_for(personality: str) -> StyleProfile:
    """The move-style profile for `personality`; the safe default (engine's
    best move) for any unknown name."""
    return PROFILES.get(personality, DEFAULT_PROFILE)


def _pov_cp(candidate: "CandidateMove", turn: str) -> int:
    """Candidate score in centipawns from the side to move's point of view."""
    return pov_cp(candidate.score_cp, candidate.mate_in, turn)


def _is_forcing(san: str) -> bool:
    """Captures, checks, and mates — the moves flashy personalities crave."""
    return "x" in san or san.endswith("+") or san.endswith("#")


def choose_candidate(
    candidates: "list[CandidateMove]", profile: StyleProfile, turn: str
) -> "CandidateMove":
    """Pick one of `candidates` (best-first, as Stockfish returned them)
    according to `profile`, scoring from `turn`'s point of view."""
    if not candidates:
        raise ValueError("no candidates to choose from")
    best = candidates[0]
    best_cp = _pov_cp(best, turn)
    if best.mate_in is not None and best_cp > 0:
        # A forced mate in our favor is never passed over for style.
        return best
    eligible = [
        c for c in candidates if best_cp - _pov_cp(c, turn) <= profile.tolerance_cp
    ]
    if profile.prefers == "aggressive":
        forcing = [c for c in eligible if _is_forcing(c.san)]
        if forcing:
            return forcing[0]
    elif profile.prefers == "modest":
        return min(eligible, key=lambda c: _pov_cp(c, turn))
    return eligible[0]


def play_styled_move(
    engine: "EnginePlayer", session: "GameSession", profile: StyleProfile
) -> "MoveResult":
    """The engine's reply, biased by `profile`, submitted through the
    session's legality gate. A `multipv=1` profile (and the rare case of no
    candidates) is just the engine's own move — no analysis detour."""
    if profile.multipv <= 1:
        return engine.play_move(session)
    candidates = engine.get_best_moves(session, n=profile.multipv)
    if not candidates:
        return engine.play_move(session)
    chosen = choose_candidate(candidates, profile, session.turn)
    return session.submit_move(chosen.uci)
