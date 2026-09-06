"""The draw-offer rule: whether Glitch's side accepts a draw the player offers.

Code, not the model (`docs/draw-offer.md`). Whether an offer was *made* is
understanding and the planner's; whether it is *accepted* is a verdict about
the position, and a verdict the model gave could be talked into, misread off a
FEN, or coloured by the personality it is playing. So the answer is a pure
function of Stockfish's number and the material on the board — never the
model's judgment, never personality, and never the difficulty tier: the offer
is a question about the position, not about how hard Glitch is playing, and a
beginner-tier Glitch accepts exactly the draws a maximum-tier one does.

One function over `(session, evaluation)`, so every branch is pinned without an
engine; the constants live here and nowhere else.
"""

from dataclasses import dataclass
from typing import Any

from chessapp.engine import Evaluation, pov_cp
from chessapp.game import GameSession

# How far from level the engine-POV evaluation may sit for the position to
# count as drawn: half a pawn either way. Mates score far outside it on the
# `MATE_CP` scale `pov_cp` maps them onto, so "no forced mate either way" is the
# same check.
DRAW_OFFER_BAND_CP = 50

# The endgame threshold, per side, in pawns of non-pawn material: a rook and a
# minor piece, or two minors. Queens disqualify outright whatever the sum —
# a queen ending is a swindle waiting to happen, and Glitch plays those out.
ENDGAME_MAX_NON_PAWN_MATERIAL = 8

# The decline reasons, machine-readable, in the order they are checked. The
# evaluation reasons come before the material one on purpose: "you're ahead"
# is the more useful thing to hear about a middlegame the player is winning.
TOO_EARLY = "too_early"
ENGINE_AHEAD = "engine_ahead"
PLAYER_AHEAD = "player_ahead"
NOT_AN_ENDGAME = "not_an_endgame"


@dataclass(frozen=True)
class DrawOfferVerdict:
    """The answer to an offer, with every fact it was made from.

    `cp_engine_pov` is the evaluation from the engine's side of the board
    (positive: the engine is better), mates folded onto the `MATE_CP` scale;
    `mate_in` is the raw signed White-POV mate distance when there is one.
    `material` is `GameSession.material_profile()`. `reason` is None exactly
    when `accepted` is True.
    """

    accepted: bool
    reason: str | None
    cp_engine_pov: int
    mate_in: int | None
    material: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "reason": self.reason,
            "evaluation": {
                "cp_engine_pov": self.cp_engine_pov,
                "mate_in": self.mate_in,
            },
            "material": dict(self.material),
        }


def engine_color(session: GameSession) -> str:
    return "black" if session.player_color == "white" else "white"


def is_endgame(material: dict[str, Any]) -> bool:
    """The simple material rule: no queens, and neither side above the
    non-pawn threshold."""
    if material["queens"]:
        return False
    return all(
        value <= ENDGAME_MAX_NON_PAWN_MATERIAL
        for value in material["non_pawn"].values()
    )


def judge_draw_offer(
    session: GameSession, evaluation: Evaluation, *, player_has_moved: bool
) -> DrawOfferVerdict:
    """Accept iff the player has moved, the engine-POV evaluation is within
    `DRAW_OFFER_BAND_CP` of level (mates included), and the position is an
    endgame by `is_endgame`. Otherwise decline with the first reason that
    fails, in that order.

    `player_has_moved` is the destructive gate's own notion of investment
    (`tools._player_has_moved`), passed in rather than re-derived so the two
    cannot disagree about whose plies are whose.
    """
    material = session.material_profile()
    cp = pov_cp(evaluation.score_cp, evaluation.mate_in, engine_color(session))
    reason: str | None
    if not player_has_moved:
        reason = TOO_EARLY
    elif cp > DRAW_OFFER_BAND_CP:
        reason = ENGINE_AHEAD
    elif cp < -DRAW_OFFER_BAND_CP:
        reason = PLAYER_AHEAD
    elif not is_endgame(material):
        reason = NOT_AN_ENDGAME
    else:
        reason = None
    return DrawOfferVerdict(
        accepted=reason is None,
        reason=reason,
        cp_engine_pov=cp,
        mate_in=evaluation.mate_in,
        material=material,
    )
