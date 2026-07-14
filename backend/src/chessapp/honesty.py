"""The honesty guard: commentary may not announce an event that never happened.

The house rule says the model never decides what deterministic state already
knows. Commentary is where that rule leaks: the loop's closing turn is produced
from a context ending in the new board, and it *still* invents the board it
wanted — "Word. Game over." with no tool call and a live game, "you actually
have me in checkmate" after a quiet rook move (trace review 2026-07-13, finding
6). That is the worst failure the app has, because the player is *told* the game
ended. A prompt rule is no defense: a 12B follows one about half the time.

So the pipeline checks the claim against the board before it emits it. This
module owns only the string half of that check — does this text *assert* that
the game ended or that a new one began? Whether it actually did is the session's
answer, and `api._run_command` puts the two together.

The bar is deliberately "asserts", not "mentions". Trash talk, threats,
questions and hypotheticals are the whole point of the commentary — "one more
move and it's checkmate", "want a new game?", "that's nearly mate" — and
suppressing those would cost far more than the lie does. So a claim must be an
unhedged assertion in its own sentence, and the tests in `test_honesty.py` are
the spec.
"""

import re

# Unhedged assertions that the game just ended, or that a new one just began.
# `resign` counts when it is inflected ("resigning now", "you resigned") or owned
# by somebody ("I resign") — all of those report an event. A bare, unowned
# "resign" is just the word, as in "want to resign?" or "never resign a won game".
_CLAIMS = re.compile(
    r"""
    \b(?: game \s+ over
        | game \s+ (?: is |'s ) \s+ over
        | checkmate
        | (?: it | that ) (?: 's | \s+ is ) \s+ mate
        | (?: i | you ) \s+ (?: win | won | lose | lost )
        | resign (?: s | ed | ing )
        | (?: i | you ) \s+ resign
        | new \s+ game
    )\b
    """,
    re.IGNORECASE | re.VERBOSE,
)

# What turns an assertion back into talk: a question, a condition, a negation, a
# near-miss, a threat. Any of these in the same sentence and the claim is not one.
_HEDGES = re.compile(
    r"""
    \? | n't
    | \b(?: if | unless | when | once | almost | nearly | close \s+ to
          | not | no | never | yet | maybe | might | could | would | should
          | want | wanna | threat | threats | threatening | one \s+ more
          | about \s+ to | next \s+ move | avoid | prevent )\b
    """,
    re.IGNORECASE | re.VERBOSE,
)

_SENTENCES = re.compile(r"(?<=[.!?])\s+")


def claims_destructive_outcome(text: str) -> bool:
    """True when the commentary asserts the game ended or a new one began.

    Sentence by sentence, so a claim is never rescued or invented by its
    neighbours: "Mate threats everywhere, but you're not dead" claims nothing,
    while "oh shit. you actually have me in checkmate." claims plenty.
    """
    return any(
        _CLAIMS.search(sentence) and not _HEDGES.search(sentence)
        for sentence in _SENTENCES.split(text)
    )
