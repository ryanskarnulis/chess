"""The predicate behind the pipeline's honesty guard.

`claims_destructive_outcome` answers one question about a *string*: does this
commentary tell the player the game ended, or that a new one began? The pipeline
pairs that with what deterministically happened (board over? destructive tool
succeeded?) and refuses to emit the claim when the two disagree — see
`test_command.py`. This file is the predicate's spec.

The failure it exists for is trace-review finding 6: commentary that invents the
event it is supposedly reacting to — "Word. Game over." on a live board, "you
actually have me in checkmate" after a quiet rook move. The player is *told* the
thing happened, which is the worst thing the app can do.

It must not be trigger-happy: trash talk, questions and hypotheticals are the
agent's whole personality, and suppressing those would cost more than the lie.
"""

import pytest

from chessapp.honesty import claims_destructive_outcome, names_a_legal_move


@pytest.mark.parametrize(
    "text",
    [
        # The real ones, from the traces.
        "Word. Game over.",
        "I'm calling that. Game over, bro.",
        "You're done. Resigning now.",
        "oh shit. you actually have me in checkmate.",
        "Checkmate. You're actually cooked.",
        # Other assertions of the same events.
        "That's mate.",
        "Checkmate!",
        "You win.",
        "I win. Better luck next time.",
        "I resign.",
        "You resigned — I'll take it.",
        "Fresh board. New game.",
        "Alright, I've started a new game.",
        "the game is over",
    ],
)
def test_an_asserted_ending_is_a_claim(text):
    assert claims_destructive_outcome(text) is True


@pytest.mark.parametrize(
    "text",
    [
        # Ordinary commentary, including the trash talk that is the point of it.
        "Nf3. Your move.",
        "Bold. That hangs your knight, but bold.",
        "You're getting crushed here, just so you know.",
        # Hypothetical, conditional, negated, interrogative — none of them claim
        # anything happened, and all of them are things the agent legitimately says.
        "That's nearly mate.",
        "One more move and it's checkmate.",
        "If you resign now I'd understand.",
        "Do you want to resign?",
        "Want to start a new game?",
        "Not checkmate yet.",
        "This is not game over — you still have the exchange.",
        "Mate threats everywhere, but you're not dead.",
        "",
    ],
)
def test_ordinary_commentary_is_not_a_claim(text):
    assert claims_destructive_outcome(text) is False


# --- naming a playable move (the hints-off advice leak) ------------------------
#
# Audit item 11's second half: with hints off the model can no longer *call*
# get_best_moves (the offer withholds it), but it can still invent a move from
# its own head — the 2026-07-13 trace leak, still measured live after the
# capability cut (2/5–3/5). The payload of a hint is a SAN token the player
# could play right now, whatever prose surrounds it, so that is what the
# predicate matches. The pipeline pairs it with the settings and the turn's
# evidence: analysis the player explicitly asked for keeps its moves.

LEGAL = ["Nf3", "Nc3", "e4", "d4", "Bc4", "O-O"]


@pytest.mark.parametrize(
    "text",
    [
        "Try Nf3 here.",
        "I'd go with e4, obviously.",
        "Bc4 or Nc3 — both fine.",
        "Castle already: O-O!",
        "`d4` is the move.",
    ],
)
def test_naming_a_playable_move_is_advice(text):
    assert names_a_legal_move(text, LEGAL) is True


@pytest.mark.parametrize(
    "text",
    [
        # Declining, needling, or talking about the position without handing
        # over a move — the commentary hints-off is supposed to produce.
        "Figure it out yourself.",
        "Hints are off. You wanted a fair fight, remember?",
        "Your knight is hanging, just saying.",
        # A move that is not currently playable is not a hint.
        "That e5 push last game was rough.",
        "",
    ],
)
def test_commentary_without_a_playable_move_is_not_advice(text):
    assert names_a_legal_move(text, LEGAL) is False
