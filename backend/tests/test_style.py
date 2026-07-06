"""Personality move style: deterministic bias over MultiPV candidates.

Pure selection logic — no engine, no LLM. Candidates come in best-first
(Stockfish order); `choose_candidate` may only pick among them, so every
choice is a legal, engine-vetted move. The personality decides *which* of
the near-best moves gets played, never whether a move is legal.
"""

import pytest

from chessapp.engine import CandidateMove
from chessapp.style import (
    DEFAULT_PROFILE,
    PROFILES,
    StyleProfile,
    choose_candidate,
    play_styled_move,
    profile_for,
)
from chessapp.tools import PERSONALITIES


def cand(san, uci="e2e4", score_cp=None, mate_in=None):
    return CandidateMove(uci=uci, san=san, score_cp=score_cp, mate_in=mate_in)


# --- profiles ---------------------------------------------------------------


def test_every_personality_has_a_profile():
    # No personality selectable without a move style (even if it's "best").
    assert set(PROFILES) == set(PERSONALITIES)


def test_unknown_personality_falls_back_to_default_profile():
    assert profile_for("nonexistent") == DEFAULT_PROFILE


def test_default_profile_plays_the_engines_best_move():
    assert DEFAULT_PROFILE.multipv == 1


def test_stronger_personalities_always_play_best():
    for name in ("grandmaster", "silent_assassin", "calm_coach"):
        assert profile_for(name).multipv == 1


def test_biased_personalities_consider_multiple_candidates():
    for name in ("trash_talker", "villain", "streamer", "beginner_bot"):
        profile = profile_for(name)
        assert profile.multipv > 1
        assert profile.tolerance_cp > 0


# --- choose_candidate -------------------------------------------------------


def test_no_candidates_is_an_error():
    with pytest.raises(ValueError):
        choose_candidate([], DEFAULT_PROFILE, "white")


def test_single_candidate_is_chosen():
    only = cand("e4", score_cp=30)
    assert choose_candidate([only], DEFAULT_PROFILE, "white") is only


def test_best_profile_picks_the_top_candidate():
    best = cand("e4", score_cp=30)
    other = cand("Nxf7", score_cp=28)
    profile = StyleProfile(multipv=2, tolerance_cp=50, prefers="best")
    assert choose_candidate([best, other], profile, "white") is best


def test_aggressive_prefers_a_capture_within_tolerance():
    quiet_best = cand("e4", score_cp=30)
    capture = cand("Nxe5", score_cp=10)
    profile = StyleProfile(multipv=2, tolerance_cp=50, prefers="aggressive")
    assert choose_candidate([quiet_best, capture], profile, "white") is capture


def test_aggressive_prefers_a_check_within_tolerance():
    quiet_best = cand("e4", score_cp=30)
    check = cand("Qh5+", score_cp=0)
    profile = StyleProfile(multipv=2, tolerance_cp=50, prefers="aggressive")
    assert choose_candidate([quiet_best, check], profile, "white") is check


def test_aggressive_ignores_a_capture_outside_tolerance():
    # The flashy move loses too much: personality never overrides sanity.
    quiet_best = cand("e4", score_cp=30)
    bad_capture = cand("Qxb7", score_cp=-200)
    profile = StyleProfile(multipv=2, tolerance_cp=50, prefers="aggressive")
    assert choose_candidate([quiet_best, bad_capture], profile, "white") is quiet_best


def test_aggressive_falls_back_to_best_when_nothing_is_forcing():
    best = cand("e4", score_cp=30)
    quiet = cand("d4", score_cp=25)
    profile = StyleProfile(multipv=2, tolerance_cp=50, prefers="aggressive")
    assert choose_candidate([best, quiet], profile, "white") is best


def test_modest_picks_the_weakest_eligible_candidate():
    best = cand("e4", score_cp=30)
    ok = cand("d4", score_cp=10)
    weakest = cand("a3", score_cp=-50)
    profile = StyleProfile(multipv=3, tolerance_cp=100, prefers="modest")
    assert choose_candidate([best, ok, weakest], profile, "white") is weakest


def test_modest_never_picks_below_tolerance():
    best = cand("e4", score_cp=30)
    blunder = cand("g4", score_cp=-300)
    profile = StyleProfile(multipv=2, tolerance_cp=100, prefers="modest")
    assert choose_candidate([best, blunder], profile, "white") is best


def test_no_profile_ever_spoils_a_forced_mate():
    mate = cand("Qh7#", score_cp=None, mate_in=1)
    tempting_capture = cand("Qxb7", score_cp=500)
    for prefers in ("best", "aggressive", "modest"):
        profile = StyleProfile(multipv=2, tolerance_cp=1000, prefers=prefers)
        assert choose_candidate([mate, tempting_capture], profile, "white") is mate


def test_scores_are_read_from_the_side_to_moves_point_of_view():
    # Scores are White-POV: for Black, -40 is *better* than +10. A modest
    # profile must treat the +10 (worse for Black) as the weakest eligible.
    best_for_black = cand("e5", score_cp=-40)
    worse_for_black = cand("a6", score_cp=10)
    profile = StyleProfile(multipv=2, tolerance_cp=100, prefers="modest")
    chosen = choose_candidate([best_for_black, worse_for_black], profile, "black")
    assert chosen is worse_for_black


def test_black_mate_is_favorable_for_black():
    mate_for_black = cand("Qh2#", score_cp=None, mate_in=-1)
    quiet = cand("e5", score_cp=-40)
    profile = StyleProfile(multipv=2, tolerance_cp=1000, prefers="modest")
    assert choose_candidate([mate_for_black, quiet], profile, "black") is mate_for_black


# --- play_styled_move -------------------------------------------------------


class FakeStyledEngine:
    """Engine double: scripted MultiPV candidates + best-move fallback."""

    def __init__(self, candidates, best_uci="g1f3"):
        self.candidates = candidates
        self.best_uci = best_uci
        self.multipv_requests = []

    def get_best_moves(self, session, n=3):
        self.multipv_requests.append(n)
        return self.candidates[:n]

    def choose_move(self, session):
        return self.best_uci

    def play_move(self, session):
        return session.submit_move(self.best_uci)


def make_session():
    from chessapp.game import GameSession

    return GameSession()


def test_play_styled_move_submits_the_chosen_candidate():
    session = make_session()
    candidates = [
        cand("e4", uci="e2e4", score_cp=30),
        cand("Nf3", uci="g1f3", score_cp=25),
    ]
    engine = FakeStyledEngine(candidates)
    profile = StyleProfile(multipv=2, tolerance_cp=100, prefers="modest")
    result = play_styled_move(engine, session, profile)
    assert result.legal
    assert result.uci == "g1f3"  # the modest pick, not the best move
    assert engine.multipv_requests == [2]


def test_play_styled_move_with_best_profile_skips_multipv():
    # multipv=1 means "just play the engine's move" — no analysis detour.
    session = make_session()
    engine = FakeStyledEngine([], best_uci="e2e4")
    result = play_styled_move(engine, session, DEFAULT_PROFILE)
    assert result.legal
    assert result.uci == "e2e4"
    assert engine.multipv_requests == []


def test_play_styled_move_falls_back_when_no_candidates():
    session = make_session()
    engine = FakeStyledEngine([], best_uci="e2e4")
    profile = StyleProfile(multipv=3, tolerance_cp=50, prefers="aggressive")
    result = play_styled_move(engine, session, profile)
    assert result.legal
    assert result.uci == "e2e4"
