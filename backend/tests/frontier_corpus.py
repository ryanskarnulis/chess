"""The frontier corpus (#318): scenarios built to be hard, measured not gated.

Each scenario says why it is hard, which tier it sits in, and how it is
graded: named checkpoints, each a predicate over the board and the tool
results (`frontier.Checkpoint`). `dev` variants are for iterating on prompts;
`heldout` variants are never tuned against.

Tiers, by what today's model is expected to manage:

1. **Stretch** — one utterance composing several intents.
2. **Multi-turn** — a thread whose later turns lean on earlier ones.
3. **Frontier** — long sessions and long games, expected near zero.

This file starts with one scenario in each of the first two tiers, enough to
exercise the harness end to end; the corpus proper is #318's PR 4.
"""

from __future__ import annotations

from frontier import VERDICT_TOOLS, Checkpoint, Episode, Say, Scenario, Variant
from test_agent_evals import EvalApp


def after_e4_e5(app: EvalApp) -> None:
    """One whole exchange on the board, the player (white) to move — set
    through the session, the gate's own way of placing a position."""
    for san in ("e4", "e5"):
        assert app.ctx.session.submit_move(san).legal


def _judged_after_the_move(episode: Episode) -> bool:
    """A verdict tool succeeded after the replacement move, in call order."""
    names = episode.turn(1).succeeded()
    if "make_move" not in names:
        return False
    after = names[len(names) - 1 - names[::-1].index("make_move") :]
    return any(name in VERDICT_TOOLS for name in after)


UNDO_REPLACE_AND_JUDGE = Scenario(
    name="undo_replace_and_judge",
    tier=1,
    why=(
        "Three intents in one breath — take back, replace, evaluate — in that "
        "order. The gate's undo_and_replace pins the first two; the verdict "
        "has to come off the new position, after the move."
    ),
    dev=(
        Variant(
            "take_back_d4_better",
            (Say("take that back, play d4 instead, and tell me if it's any better"),),
            after_e4_e5,
        ),
        Variant(
            "undo_go_d4_how_looks",
            (Say("undo my move, go d4, then tell me how the position looks"),),
            after_e4_e5,
        ),
    ),
    heldout=(
        Variant(
            "scratch_queen_pawn_read",
            (
                Say(
                    "scratch that move — push the queen pawn two instead and "
                    "give me your read on it"
                ),
            ),
            after_e4_e5,
        ),
    ),
    checkpoints=(
        Checkpoint("took_back", lambda e: bool(e.turn(1).ran("undo"))),
        Checkpoint("played_d4", lambda e: e.history()[:1] == ["d4"]),
        Checkpoint("one_exchange", lambda e: len(e.history()) == 2),
        Checkpoint("judged_after_the_move", _judged_after_the_move),
        Checkpoint("completed", lambda e: e.turn(1).stop_reason == "completed"),
    ),
)


KNIGHT_ASK_THEN_CHANGE_OF_MIND = Scenario(
    name="knight_ask_then_change_of_mind",
    tier=2,
    why=(
        "An ambiguous ask must be asked about; the player then drops it for a "
        "different move, and finally asks a judgment question that must move "
        "nothing. Three turns, each depending on the one before."
    ),
    dev=(
        Variant(
            "move_knight_then_d4",
            (
                Say("move my knight"),
                Say("actually forget the knight, push the queen's pawn two squares"),
                Say("so am I doing better than before?"),
            ),
        ),
    ),
    heldout=(
        Variant(
            "knight_out_then_d_pawn",
            (
                Say("bring a knight out"),
                Say("hmm, no — the d-pawn forward two instead"),
                Say("who's better right now?"),
            ),
        ),
    ),
    checkpoints=(
        Checkpoint("asked_first", lambda e: e.turn(1).asked and not e.turn(1).moved),
        Checkpoint(
            "no_knight_played",
            lambda e: not any(san.startswith("N") for san in e.history()[::2]),
        ),
        Checkpoint("played_d4", lambda e: e.history(after_turn=2)[:1] == ["d4"]),
        Checkpoint(
            "judged",
            lambda e: any(e.turn(3).ran(name) for name in VERDICT_TOOLS),
        ),
        Checkpoint("question_moved_nothing", lambda e: not e.turn(3).moved),
    ),
)


SCENARIOS: tuple[Scenario, ...] = (
    UNDO_REPLACE_AND_JUDGE,
    KNIGHT_ASK_THEN_CHANGE_OF_MIND,
)
