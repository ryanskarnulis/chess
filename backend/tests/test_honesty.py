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

from chessapp.honesty import (
    VerifiedFacts,
    claims_destructive_outcome,
    names_a_legal_move,
    unverified_claims,
)


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


# --- verified facts: every operational claim, not just the ending -------------
#
# Audit item 13. The ending guard above proved the shape works, and the shape
# generalizes: build the turn's facts deterministically (tool results + board),
# then require the *operational* claims in commentary to derive from them.
# Personality varies the wording; it does not get to vary the facts.
#
# `unverified_claims` returns the claim classes the text asserts and the facts
# don't support — a list rather than a bool, so a guarded turn can say which
# class failed. Every class keeps the ending guard's bar: an unhedged assertion
# in its own sentence. Trash talk, threats, questions and hypotheticals are the
# whole point of the commentary and must keep surviving, which is why roughly
# half the cases below are the ones that must come back empty.

# A quiet turn on a live, level board: nothing happened, so nothing is claimable.
NOTHING = VerifiedFacts()


def test_a_turn_with_no_facts_supports_no_operational_claim():
    assert unverified_claims("I took your knight.", NOTHING) == ("capture",)


def test_ordinary_commentary_claims_nothing_to_verify():
    assert unverified_claims("Bold. That hangs your knight, but bold.", NOTHING) == ()


# --- captures ------------------------------------------------------------------
#
# Verified against the board's captured-piece record, per side: "I" is Glitch,
# "you" is the player. Existence, not tense — a capture that really happened
# ten moves ago is a true fact awkwardly placed, and guarding *that* would cost
# more than it saves. What the class is for is the piece that was never taken.

TOOK_A_KNIGHT = VerifiedFacts(captured_by_opponent=frozenset({"knight"}))


@pytest.mark.parametrize(
    "text",
    [
        "I took your knight.",
        "Grabbed your queen, thanks.",
        "That bishop is gone.",
        "Your rook is mine now.",
        "Snagged the pawn.",
    ],
)
def test_an_unbacked_capture_is_a_claim(text):
    assert "capture" in unverified_claims(text, NOTHING)


@pytest.mark.parametrize(
    "text",
    [
        "I took your knight.",
        "Knight's off the board.",
        "captured your knight, obviously",
    ],
)
def test_a_capture_that_happened_is_reportable(text):
    assert unverified_claims(text, TOOK_A_KNIGHT) == ()


def test_a_capture_is_verified_per_side():
    """Who took what is a fact too: Glitch took the knight, so the player did
    not, and saying they did is the same invention in the other direction."""
    assert "capture" in unverified_claims("You took my knight.", TOOK_A_KNIGHT)


def test_the_wrong_piece_is_still_an_invention():
    assert "capture" in unverified_claims("I took your queen.", TOOK_A_KNIGHT)


@pytest.mark.parametrize(
    "text",
    [
        # Threats and offers, which is most of what Glitch says about captures.
        "I'll take your knight next move.",
        "That knight is going to get taken.",
        "Want to trade queens?",
        "If you leave the rook there I'm taking it.",
        "Your bishop is not gone yet.",
        "Nothing gets taken this move.",
        # A takeback puts a piece back on the board; it never takes one off.
        "Took your knight back. Try again.",
    ],
)
def test_a_threatened_capture_is_not_a_claim(text):
    assert unverified_claims(text, NOTHING) == ()


# --- check, and the draw the ending class does not cover ------------------------


def test_an_unbacked_check_is_a_claim():
    assert "check" in unverified_claims("You're in check, by the way.", NOTHING)


def test_a_real_check_is_reportable():
    assert unverified_claims("You're in check.", VerifiedFacts(check=True)) == ()


@pytest.mark.parametrize(
    "text",
    [
        "One more move and you're in check.",
        "You're not in check, relax.",
        "Check this out.",
    ],
)
def test_talking_about_check_is_not_a_claim(text):
    assert unverified_claims(text, NOTHING) == ()


def test_an_unbacked_draw_is_a_claim():
    """The ending class knows game-over and checkmate; a draw is its own fact,
    because a game that ended in *mate* did not end in a draw either."""
    assert "draw" in unverified_claims("That's a draw.", NOTHING)
    assert "draw" in unverified_claims("Stalemate.", VerifiedFacts(ended=True))


def test_a_real_draw_is_reportable():
    facts = VerifiedFacts(ended=True, drawn=True)
    assert unverified_claims("Stalemate. We're splitting it.", facts) == ()


@pytest.mark.parametrize(
    "text",
    [
        "This is basically a draw.",
        "Heading for a draw unless one of us blunders.",
        "Take the draw?",
    ],
)
def test_calling_a_position_drawish_is_not_a_claim(text):
    assert unverified_claims(text, NOTHING) == ()


# --- moves that were never on the board ----------------------------------------
#
# Only unambiguous move notation counts: a bare pawn push is spelled like a
# square ("the pawn on e4"), and squares are discussed constantly, so `e4` alone
# is never read as a move claim. A move played this turn, reported by an
# analysis, or playable now or at the turn's start is all fair game — including
# the move the player missed, which is commentary, not invention.

PLAYED_NF3 = VerifiedFacts(moves=frozenset({"Nf3", "Nc6", "Bc4"}))


def test_a_move_that_was_never_playable_is_a_claim():
    assert "move" in unverified_claims("Nice, you took it with Bxc6.", PLAYED_NF3)


@pytest.mark.parametrize(
    "text",
    [
        "Nf3. Your move.",
        "Nc6 was the reply.",
        "Bc4 was right there and you missed it.",
        "The pawn on e4 is doing a lot of work.",  # a square, not a move claim
    ],
)
def test_a_move_the_turn_accounts_for_is_not_a_claim(text):
    assert unverified_claims(text, PLAYED_NF3) == ()


# --- saves ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "Saved it.",
        "Game saved.",
        "Saved as tuesday-night.",
        "Restored the game. Your move.",
    ],
)
def test_an_unbacked_save_is_a_claim(text):
    assert "save" in unverified_claims(text, NOTHING)


def test_a_real_save_is_reportable():
    assert (
        unverified_claims("Saved it as tuesday-night.", VerifiedFacts(saved=True)) == ()
    )


@pytest.mark.parametrize(
    "text",
    [
        "Want me to save this?",
        "That bishop saved you.",
        "I can save it if you want.",
    ],
)
def test_talking_about_saving_is_not_a_claim(text):
    assert unverified_claims(text, NOTHING) == ()


# --- settings ------------------------------------------------------------------
#
# Verified against the live settings rather than against a tool call, because
# the live settings are the truth either way: a setting the player changed three
# turns ago is as true as one changed this turn, and a value nothing ever set is
# a lie however confidently it is announced.

CASUAL = VerifiedFacts(
    settings={"difficulty": "casual", "hints": "off", "voice": "on", "verbosity": "low"}
)


@pytest.mark.parametrize(
    ("text", "claim"),
    [
        ("Difficulty's on advanced now.", "difficulty"),
        ("You're playing maximum strength.", "difficulty"),
        ("Hints are on.", "hints"),
        ("Turned off your voice.", "voice"),
        ("Verbosity is high now.", "verbosity"),
    ],
)
def test_a_setting_that_is_not_set_is_a_claim(text, claim):
    assert claim in unverified_claims(text, CASUAL)


@pytest.mark.parametrize(
    "text",
    [
        "Difficulty is casual.",
        "Hints are off, so you're on your own.",
        "Voice output is on.",
        "Verbosity is low. Keeping it short.",
    ],
)
def test_a_setting_that_is_set_is_reportable(text):
    assert unverified_claims(text, CASUAL) == ()


@pytest.mark.parametrize(
    "text",
    [
        "Want hints on?",
        "Say the word and I'll turn the difficulty up to advanced.",
        "Turn hints on if you want help.",
    ],
)
def test_offering_a_setting_change_is_not_a_claim(text):
    assert unverified_claims(text, CASUAL) == ()


# --- material: the count the board can do --------------------------------------
#
# The class the evaluation one below deliberately left out. "You're two pawns
# down" is operational — material balance is a piece count, board truth, no
# Stockfish involved — so it gets verified like every other fact, against
# `material`: the player's advantage in pawns, positive when they are ahead.
#
# The bar the class had to clear to ship: it must tell "you're getting crushed"
# (an opinion about the position, and the trash talk that makes Glitch worth
# playing) from "you're two pawns down" (a count). So only a *quantified* claim
# — a direction and a named amount — is read as one, and the arithmetic is
# side-aware: "I'm up a piece" is the same fact from the other end.
#
# Direction is the fact; magnitude is verified to within a pawn. Material talk
# names the nominal trade ("up a knight" after winning a knight for a pawn),
# which is a pawn off the net count and true as anybody plays it — while the
# lie the class exists for is being told you are ahead when you are behind.

LEVEL = VerifiedFacts(material=0)
UP_A_KNIGHT = VerifiedFacts(material=3)
DOWN_TWO_PAWNS = VerifiedFacts(material=-2)


@pytest.mark.parametrize(
    ("text", "facts"),
    [
        # Nothing has been traded at all, so no side is up anything.
        ("You're up a pawn.", LEVEL),
        ("You're two pawns down.", LEVEL),
        ("I'm up a rook.", LEVEL),
        # The direction is the lie, which is the one that matters most.
        ("You're down a piece.", UP_A_KNIGHT),
        ("I'm up a knight.", UP_A_KNIGHT),
        ("You're up the exchange.", DOWN_TWO_PAWNS),
        # Right direction, invented amount.
        ("You're up a queen.", UP_A_KNIGHT),
        ("You're four pawns down.", DOWN_TWO_PAWNS),
        # No material fact supplied is no material claim allowed — the guard
        # fails closed on evidence, exactly as every other class does.
        ("You're two pawns down.", NOTHING),
    ],
)
def test_an_unbacked_material_claim_is_a_claim(text, facts):
    assert "material" in unverified_claims(text, facts)


@pytest.mark.parametrize(
    ("text", "facts"),
    [
        ("You're up a piece.", UP_A_KNIGHT),
        ("You're a knight up.", UP_A_KNIGHT),
        ("You're up a knight, so stop panicking.", UP_A_KNIGHT),
        ("I'm up a piece.", VerifiedFacts(material=-3)),
        ("You're two pawns down.", DOWN_TWO_PAWNS),
        ("Two pawns behind and it shows.", DOWN_TWO_PAWNS),
        ("I'm two pawns ahead.", DOWN_TWO_PAWNS),
        ("You're up the exchange.", VerifiedFacts(material=2)),
        # No subject at all: ambiguous, so either reading verifying is enough.
        ("Up a knight. Cute.", UP_A_KNIGHT),
    ],
)
def test_a_material_count_the_board_backs_is_reportable(text, facts):
    assert unverified_claims(text, facts) == ()


def test_the_nominal_trade_is_reportable_against_the_net_count():
    """A knight taken for a pawn is "up a knight" to everybody who plays chess,
    and +2 to the board. Magnitude is verified to within a pawn precisely so
    that ordinary material talk survives its own guard."""
    knight_for_a_pawn = VerifiedFacts(material=2)
    assert unverified_claims("You're up a knight.", knight_for_a_pawn) == ()


@pytest.mark.parametrize(
    "text",
    [
        # The bar for the class: an opinion about the position is not a count,
        # however brutal it is. These are Glitch, and they must all survive.
        "You're getting crushed here, just so you know.",
        "You're winning, obviously.",
        "That's a losing position and you know it.",
        "Material's about level.",
        # Threats, conditions, offers, negations — the usual half of the spec.
        "One more trade and you're up a pawn.",
        "Take the knight and you'll be up a piece.",
        "You're not down a piece, relax.",
        "If I take that rook I'm up two pawns.",
        "Want to be a pawn up? Take it.",
        # Talk that reuses the words without counting anything.
        "Your knight is up on f3 doing nothing.",
        "This is not game over — you still have the exchange.",
    ],
)
def test_talking_about_the_position_is_not_a_material_claim(text):
    assert unverified_claims(text, LEVEL) == ()


# --- analysis numbers ----------------------------------------------------------
#
# Narrow on purpose: a signed or decimal score and a mate-in-N are shapes only
# an engine can produce, so they must match a number the turn's analysis
# actually reported. Material talk is the sibling class above: derivable from
# the board rather than from Stockfish, so it is counted, not quoted.

EVALUATED = VerifiedFacts(numbers=frozenset({"150", "1.5", "+1.5"}))


@pytest.mark.parametrize(
    "text",
    [
        "You're at -2.4 here.",
        "Mate in 4, by the way.",
        "That's 320 centipawns of damage.",
    ],
)
def test_an_unbacked_number_is_a_claim(text):
    assert "evaluation" in unverified_claims(text, EVALUATED)


@pytest.mark.parametrize(
    "text",
    [
        "+1.5 for me.",
        "Stockfish says 150 centipawns.",
        "1.5 and climbing.",
    ],
)
def test_a_reported_number_is_reportable(text):
    assert unverified_claims(text, EVALUATED) == ()


def test_a_pgn_read_out_loud_is_not_an_evaluation_claim():
    """From the recorded turns: "give me the pgn" gets the whole thing back,
    headers and move numbers included. A date is not a score and `1.` is not
    a decimal — the app's own exports must survive their own guard."""
    text = '[Date "2023.10.27"]\n[Result "*"]\n\n1. e4 b6 2. Nf3 h6 3. d4 a5'
    facts = VerifiedFacts(moves=frozenset({"e4", "b6", "Nf3", "h6", "d4", "a5"}))
    assert unverified_claims(text, facts) == ()


def test_a_game_result_is_not_an_evaluation_claim():
    """The app's own closing line carries a result score, and `1-0` must not
    read as the number `-0`. It is the pipeline's own deterministic text, so
    guarding it would suppress the truth on every game that ends."""
    facts = VerifiedFacts(ended=True)
    assert unverified_claims("Game over: 1-0 (checkmate).", facts) == ()
