"""The claim reading speech accuracy scores with (#367).

`claims_destructive_outcome` answers one question about a *string*: does this
commentary tell the player the game ended, or that a new one began? The scorer
pairs the reading with what deterministically happened (board over? destructive
tool succeeded?) and counts the claim unbacked when the two disagree. Until
#368 a live guard did the same before the player heard it. This file is the
reading's spec.

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
    claims,
    claims_destructive_outcome,
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
        # A result with its object still reads as one (#384 narrowed the verb).
        "You lost.",
        "you lost the game, bro",
    ],
)
def test_an_asserted_ending_is_a_claim(text):
    assert claims_destructive_outcome(text) is True


@pytest.mark.parametrize(
    "text",
    [
        # Live, 2026-09-06: a threat spoken as a distance, cut as the ending it
        # was forecasting. `close to` was a hedge; `close for` and a bare
        # `looking` were not.
        "Damn, checkmate's looking real close for me.",
        "Checkmate is close.",
        "Looking like mate soon, dude.",
    ],
)
def test_a_near_miss_spoken_as_a_distance_is_not_a_claim(text):
    assert claims_destructive_outcome(text) is False


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
        # The #340 frontier run, verbatim (#384): the reset gate's question, a
        # review's lost evaluation, and a trade.
        "Yo, starting a new game will end this one.",
        "you lost like 817 centipawns there",
        "I won a pawn, you lost the exchange.",
        "",
    ],
)
def test_ordinary_commentary_is_not_a_claim(text):
    assert claims_destructive_outcome(text) is False


# --- verified facts: every operational claim, not just the ending -------------
#
# Audit item 13. The ending guard above proved the shape works, and the shape
# generalizes: build the turn's facts deterministically (tool results + board),
# then require the *operational* claims in commentary to derive from them.
# Personality varies the wording; it does not get to vary the facts.
#
# `unverified_claims` returns the claim classes the text asserts and the facts
# don't support — a list rather than a bool, so a miss says which class
# failed. Every class keeps the ending class's bar: an unhedged assertion
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


# A subject pronoun is not the only way the direction gets pinned. Glitch's
# register is mostly subjectless — "Snagged your bishop.", "Your knight is
# gone." — and every one of those went to the union of both sides, so a capture
# announced in exactly the wrong direction read as verified. A possessive is not
# ambiguity: "your bishop" names whose piece left the board, and the speaker is
# always the player's opponent, so it names who took it too. What stays
# fail-permissive is the phrasing that really pins nothing ("that bishop is
# gone") — the class is for the piece that was never taken, or taken the other
# way round.

PLAYER_TOOK_A_BISHOP = VerifiedFacts(captured_by_player=frozenset({"bishop"}))


@pytest.mark.parametrize(
    "text",
    [
        "Snagged your bishop.",
        "Grabbed your bishop, obviously.",
        "Your bishop is gone.",
        "Your bishop is toast.",
        "Your bishop is mine.",
    ],
)
def test_a_possessive_pins_the_direction_of_a_subjectless_capture(text):
    """The player took Glitch's bishop, so Glitch did not take theirs."""
    assert "capture" in unverified_claims(text, PLAYER_TOOK_A_BISHOP)


@pytest.mark.parametrize(
    "text",
    [
        "Snagged my bishop.",
        "My bishop is gone.",
        "My bishop is history.",
        # No possessive at all: genuinely ambiguous about whose bishop, so
        # either side's record backing it is enough. Unchanged behavior.
        "That bishop is gone.",
        "The bishop is off the board.",
    ],
)
def test_a_possessive_that_matches_the_board_is_reportable(text):
    assert unverified_claims(text, PLAYER_TOOK_A_BISHOP) == ()


def test_the_possessive_reads_the_other_way_round_too():
    """Glitch took the knight, so "your knight is gone" is the report and "my
    knight is gone" is the invention."""
    assert unverified_claims("Your knight is gone.", TOOK_A_KNIGHT) == ()
    assert "capture" in unverified_claims("My knight is gone.", TOOK_A_KNIGHT)


def test_the_subject_outranks_the_possessive():
    """An explicit subject is the stronger evidence and keeps deciding alone.
    The two agree in every natural phrasing, so the pin has to be an unnatural
    one: "I took my knight" is Glitch claiming the capture, whatever the
    possessive says about whose piece it was."""
    player_took_a_knight = VerifiedFacts(captured_by_player=frozenset({"knight"}))
    assert "capture" in unverified_claims("I took your knight.", player_took_a_knight)
    assert unverified_claims("I took your knight.", TOOK_A_KNIGHT) == ()
    assert "capture" in unverified_claims("I took my knight.", player_took_a_knight)
    assert unverified_claims("I took my knight.", TOOK_A_KNIGHT) == ()


# Which piece — the half the capture record cannot settle. It spans the whole
# game, so a queen taken twenty moves ago backs "that queen" attached to any
# move at all. `captures_by_move` is per move and read off the board, so a
# sentence that names its move is held to what *that* move takes: live, the
# analysis said Qxe2 was the better move, Qxe2 takes a pawn, and the narration
# called it a queen (walkthrough #5). `""` is a move that takes nothing.

QXE2_TAKES_A_PAWN = VerifiedFacts(
    moves=frozenset({"Qxe2", "Re1", "Nf3"}),
    captured_by_player=frozenset({"queen"}),  # a real queen, from earlier
    captures_by_move={"Qxe2": "pawn", "Re1": "", "Nf3": ""},
)


@pytest.mark.parametrize(
    "text",
    [
        "You've taken that queen with Qxe2.",
        "Qxe2 takes the queen.",
        "Qxe2 and that knight is gone.",
        # A move that captures nothing cannot have taken anything.
        "Nf3 grabs the bishop.",
    ],
)
def test_a_named_move_is_held_to_what_it_actually_takes(text):
    assert "capture" in unverified_claims(text, QXE2_TAKES_A_PAWN)


@pytest.mark.parametrize(
    "text",
    [
        "You've taken that pawn with Qxe2.",
        "Qxe2 takes the pawn.",
        "Qxe2 and that pawn is gone.",
    ],
)
def test_the_right_piece_on_a_named_move_is_reportable(text):
    assert unverified_claims(text, QXE2_TAKES_A_PAWN) == ()


def test_the_named_move_outranks_the_game_wide_record():
    """The record really does hold a queen the player took, and it really does
    not back *this* sentence — that mismatch is the whole defect."""
    assert "queen" in QXE2_TAKES_A_PAWN.captured_by_player
    assert unverified_claims("You took my queen.", QXE2_TAKES_A_PAWN) == ()
    assert "capture" in unverified_claims(
        "You took my queen with Qxe2.", QXE2_TAKES_A_PAWN
    )


def test_a_move_the_board_cannot_place_falls_back_to_the_record():
    """No knowledge is not evidence of a lie: a SAN from a position the turn no
    longer holds leaves the coarser checks in charge."""
    facts = VerifiedFacts(
        moves=frozenset({"Bxc6"}), captured_by_player=frozenset({"knight"})
    )
    assert unverified_claims("You took my knight with Bxc6.", facts) == ()


# "Taken" is a claim only in the perfect. Bare, it is passive and predictive,
# which is the register a threat is actually spoken in.


@pytest.mark.parametrize(
    "text",
    [
        "I've taken your knight.",
        "You have taken my bishop.",
        "He's taken the rook.",
    ],
)
def test_the_perfect_reports_a_capture(text):
    assert "capture" in unverified_claims(text, NOTHING)


@pytest.mark.parametrize(
    "text",
    [
        "That knight is going to get taken.",
        "Your bishop gets taken either way.",
    ],
)
def test_a_bare_taken_is_still_not_a_claim(text):
    assert unverified_claims(text, NOTHING) == ()


@pytest.mark.parametrize(
    "text",
    [
        "I'll take your bishop next.",
        "Your bishop is not gone yet.",
        "Want me to take your bishop?",
        # A takeback is still not a capture, possessive or no possessive.
    ],
)
def test_a_possessive_does_not_turn_talk_into_a_claim(text):
    assert unverified_claims(text, NOTHING) == ()


# Advice, which is the other half of what Glitch says about captures and the
# half that used to die. The player asks "what should I play?", the engine says
# Rxd1, and every natural way to hand that over reads like a capture report to
# a class that cannot tell an imperative from a past tense. All three such
# turns in the 2026-09-04 walkthrough were replaced with "Scratch that — I said
# something the board doesn't back up."
#
# Two things tell advice from a report and both come off the board: the verb is
# a bare stem (an imperative or an infinitive reports nothing), or the sentence
# hangs on a move that is playable but that nobody has played.

ROOK_IS_HANGING = VerifiedFacts(moves=frozenset({"Rxd1", "Rd8", "Kf8"}))


@pytest.mark.parametrize(
    "text",
    [
        # The two strings the walkthrough actually lost, verbatim.
        "Take the rook. Rxd1 is the move, bro.",
        "Take that rook on d1. It's clean.",
        # The same advice in the tense the verb list cannot rule out.
        "Rxd1 takes the rook. Free material.",
        "Rxd1 and that rook is gone.",
        "Grab the queen while it's sitting there.",
    ],
)
def test_advice_about_a_capture_is_not_a_report(text):
    assert unverified_claims(text, ROOK_IS_HANGING) == ()


def test_a_played_move_still_reports_its_capture():
    """The unplayed-move reading is about advice, not about SAN: once the move
    is one somebody played, the sentence is a report again and the record
    decides."""
    played_rxd1 = VerifiedFacts(
        moves=frozenset({"Rxd1"}), moves_by_opponent=frozenset({"Rxd1"})
    )
    assert "capture" in unverified_claims("Rxd1 takes the rook.", played_rxd1)


def test_a_subject_outranks_the_advice_reading():
    """A person is not a line. "I took your queen with Qxe2" names a move that
    was never played and is still a claim that an event happened — which is
    exactly the shape of the mistake narration that invents a victim."""
    available = VerifiedFacts(moves=frozenset({"Qxe2"}))
    assert "capture" in unverified_claims("I took your queen with Qxe2.", available)


def test_a_false_past_tense_capture_survives_the_loosening():
    """The class still exists. Advice got quieter; the invention did not."""
    assert "capture" in unverified_claims("Took your rook.", ROOK_IS_HANGING)
    assert "capture" in unverified_claims("Your rook is gone.", ROOK_IS_HANGING)
    assert "capture" in unverified_claims("I grabbed the rook.", ROOK_IS_HANGING)


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
    ],
)
def test_a_threatened_capture_is_not_a_claim(text):
    assert unverified_claims(text, NOTHING) == ()


@pytest.mark.parametrize(
    "text",
    ["Took your knight back. Try again.", "Took your bishop back. Try again."],
)
def test_takeback_talk_is_the_takeback_class_and_never_a_capture(text):
    """The capture class exempts "back" because a takeback puts the piece back
    *on* the board. Since #289 the sentence is still read — by the class that
    owns it, against the turn's own `undo`."""
    assert unverified_claims(text, NOTHING) == ("takeback",)
    assert unverified_claims(text, VerifiedFacts(undone=True)) == ()


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
    facts = VerifiedFacts(ended=True, drawn=True, termination="stalemate")
    assert unverified_claims("Stalemate. We're splitting it.", facts) == ()


@pytest.mark.parametrize(
    "text",
    [
        "This is basically a draw.",
        "Heading for a draw unless one of us blunders.",
        "Take the draw?",
        # The gap that was left, and it is the same bar the material class
        # already holds: "drawn" is overwhelmingly an *assessment* of a level
        # position, not a report that the game ended in one. The hedges caught
        # "basically" and "dead drawn" and "looks like" — and let the plainest
        # phrasings through, on exactly the symmetrical positions that earn
        # them. A London against a London is where Glitch reaches for this
        # word, and a live turn there is what sent it to the guard.
        "This is drawn, bro.",
        "Symmetrical. kinda drawn already.",
        "Looks drawn to me.",
        "That's a drawn endgame if I ever saw one.",
        "Drawish. do something.",
    ],
)
def test_calling_a_position_drawish_is_not_a_claim(text):
    assert unverified_claims(text, NOTHING) == ()


def test_the_game_actually_ending_in_a_draw_is_still_a_report():
    """Loosening the assessment must not lose the event. What makes a draw a
    *claim* is naming the game as over, not the adjective."""
    assert "draw" in unverified_claims("Stalemate.", VerifiedFacts(ended=True))
    assert "draw" in unverified_claims("That's a draw.", NOTHING)
    assert "draw" in unverified_claims("We drew.", NOTHING)
    assert "draw" in unverified_claims("Game's drawn. gg.", NOTHING)


# The draw as a noun somebody offered, declined or called too early is not a
# report that the game came to one. Found in #303's gate (2026-09-22,
# `offer_draw_routes` 4/5): a declined offer explained correctly and cut,
# because `(a|the) draw` read every mention as the result. A probe the same day
# showed most natural ways of narrating a decline tripped it, so the class now
# reads the shapes that report a result, and none of these has one.
@pytest.mark.parametrize(
    "text",
    [
        "Yo, the engine says it's too early to call it a draw. Keep going.",
        # `main`'s three misses in the 2026-09-22 campaign, verbatim.
        "Word, the engine says it's way too early to call it a draw.",
        "Word, but the engine says it's too early for a draw. Keep going.",
        "yo, the engine says it's way too early to call it a draw. keep going.",
        "I turned down the draw.",
        "You offered a draw, I said nah.",
        "The draw offer got bounced.",
        "Engine declined the draw.",
        "Nobody's calling it a draw yet.",
        "Let's call it a draw.",  # a proposal, the player's to accept
        "Offering a draw in the Ruy? Cute.",
    ],
)
def test_a_draw_mentioned_is_not_a_draw_reported(text):
    assert "draw" not in unverified_claims(text, NOTHING)


DECLINED = VerifiedFacts(moves=frozenset({"a6"}))
AGREED = VerifiedFacts(ended=True, drawn=True, termination="agreement")


@pytest.mark.parametrize(
    "text",
    [
        "That's a draw.",
        "It's a draw by repetition.",
        "We drew.",
        "Game's drawn. gg.",
        "Game ends in a draw.",
        "The game ended in a draw, bro.",
        "We called it a draw.",
        # The lie on a declined offer, which the noun reading never caught.
        "Draw agreed.",
        "Draw accepted, gg.",
        "A draw it is.",
        "Split the point, man.",
    ],
)
def test_a_draw_reported_on_a_live_board_is_a_claim_and_true_when_drawn(text):
    assert "draw" in unverified_claims(text, DECLINED)
    assert "draw" not in unverified_claims(text, AGREED)


def test_a_draw_reported_over_a_checkmate_is_still_a_claim():
    mated = VerifiedFacts(ended=True, winner="player", termination="checkmate")
    assert "draw" in unverified_claims("That's a draw.", mated)
    assert "draw" in unverified_claims("Stalemate.", mated)


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


@pytest.mark.parametrize(
    "text",
    [
        # A threat names a move that is deliberately *not* playable yet — that
        # is what makes it a threat — so the class must read the tense or it
        # guards the one thing it was built to protect. `_FUTURE` had the
        # explicit futures ("I'll play Qh5", "Qh5 next") and missed the way
        # people actually threaten.
        "Qh5 is coming.",
        "Qh5 incoming, by the way.",
        "Rd8 is on the way.",
        "Qh5 looming. sleep on it.",
    ],
)
def test_a_threatened_move_is_not_a_claim(text):
    assert unverified_claims(text, PLAYED_NF3) == ()


# --- ...and a piece named on the square it stands on ---------------------------
#
# Found in #289's gates (2026-09-22; 1/20 on `main` and on the #289 tree,
# interleaved, and twice in #302's gate): asked to show the position, the
# narrator listed it in piece-letter notation and the move class read `Ke1` as
# a move nobody could play. A king cannot move to the square it stands on, so
# no move list ever holds it — the facts had to learn where the pieces are.
# Only the bare shape is placement: a capture, a check or a promotion mark
# names an event, and a square the piece is not on is still an invention.

AFTER_NF3_D5 = VerifiedFacts(
    moves=frozenset({"Nf3", "d5", "g3", "c4", "Nc3", "e3"}),
    placements=frozenset(
        {"Ke1", "Qd1", "Ra1", "Rh1", "Bc1", "Bf1", "Nb1", "Nf3"}
        | {"Ke8", "Qd8", "Ra8", "Rh8", "Bc8", "Bf8", "Nb8", "Nc6", "Ng8"}
    ),
)


@pytest.mark.parametrize(
    "text",
    [
        # The misfire verbatim, from `main`'s arm of the 2026-09-22 campaign
        # (3/20 there; the position after 1.e4 e5 2.Nf3 Nc6).
        "White: Ke1, Qd1, Ra1, Rh1, Bc1, Bf1, Nb1, Nf3, Pawns a2, b2, c2, d2, "
        "e4, f2, g2, h2.\nBlack: Ke8, Qd8, Ra8, Rh8, Bc8, Bf8, Nc6, Ng8, Pawns "
        "a7, b7, c7, d7, e5, f7, g7, h7.",
        "Here's where everything's at. White: Ke1, Qd1, Ra1, Rh1, Bc1, Bf1, "
        "Nb1, Nf3, pawns on a2 through h2 except the ones that moved; Black: "
        "Ke8, Qd8, Ra8, Rh8, Bc8, Bf8, Nb8, Ng8.",
        "White — Ke1 Qd1 Ra1 Rh1 Bc1 Bf1 Nb1 Nf3. Black — Ke8 Qd8 Ra8 Rh8.",
        "Your Nf3 is holding the whole kingside together.",
        "My Ng8 is still asleep. Patience.",
    ],
)
def test_a_piece_named_on_its_own_square_is_placement_not_a_move(text):
    assert unverified_claims(text, AFTER_NF3_D5) == ()


def test_placement_after_castling_and_promotion_is_still_placement():
    facts = VerifiedFacts(placements=frozenset({"Kg1", "Rf1", "Qa8"}))
    assert unverified_claims("White: Kg1, Rf1, and a fresh Qa8.", facts) == ()


@pytest.mark.parametrize(
    "text",
    [
        # The 2026-07-13 lie the move class exists for: the engine played Na6.
        "Bxa6. You're really just taking my pieces for free.",
        "Nd5, and your center's gone.",  # no knight stands on d5
        "Ke2. Bold.",  # the king is on e1
        "Qxf7#, gg.",  # a capture and a mate are events, never placement
        "Nxf3 was the move.",  # a capture onto a square a piece holds
    ],
)
def test_a_move_the_board_does_not_hold_is_still_a_claim(text):
    assert "move" in unverified_claims(text, AFTER_NF3_D5)


def test_a_placement_list_over_a_pending_reply_is_not_the_reply():
    """A quiet move cannot land on an occupied square, so a piece named on its
    own square is never a reply the engine could still play there."""
    facts = VerifiedFacts(
        moves=frozenset({"e4", "Nf6", "Nc6", "e5"}),
        placements=frozenset({"Ke8", "Ng8", "Nb8", "Ke1", "Ng1"}),
        unplayed_replies=frozenset({"Nf6", "Nc6", "e5"}),
    )
    assert unverified_claims("Black: Ke8, Nb8, Ng8. White: Ke1, Ng1.", facts) == ()


# --- ...and who played one -----------------------------------------------------
#
# `moves` is deliberately wide — played, reported, playable — which is what keeps
# "Bc4 was right there" sayable, and it is also why it can say nothing about
# *whose* move a move was. "I played Nf3" when the player played Nf3 derives from
# it perfectly. So the two sides' played moves are their own facts, and a
# sentence where a pronoun owns a move is checked against the side it credits.
#
# The wide set still answers every unattributed mention, and it is the fallback
# when a caller supplies no attribution at all: fail permissive on missing
# evidence, exactly as the ambiguous phrasings do.

OWNED = VerifiedFacts(
    moves=frozenset({"Nf3", "Nc6", "e5"}),
    moves_by_player=frozenset({"Nf3"}),
    moves_by_opponent=frozenset({"Nc6", "e5"}),
)


@pytest.mark.parametrize(
    "text",
    [
        "I played Nf3.",
        "I went Nf3, obviously.",
        "You played Nc6.",
        "You answered with Nc6.",
        "You moved Nc6 and here we are.",
    ],
)
def test_a_move_credited_to_the_wrong_side_is_a_claim(text):
    assert "owned_move" in unverified_claims(text, OWNED)


@pytest.mark.parametrize(
    "text",
    [
        "You played Nf3.",
        "You pushed Nf3 and I liked it.",
        "I played Nc6.",
        "I replied with Nc6.",
        # A bare pawn push is spelled like a square, so it is no more a move
        # claim here than in the class above — and this one is true anyway.
        "I played e5.",
    ],
)
def test_a_move_the_side_really_played_is_reportable(text):
    assert unverified_claims(text, OWNED) == ()


def test_an_unattributed_move_mention_is_still_the_wider_set():
    """Nobody owns the move, so nothing is credited: the generic class answers
    it off `moves`, exactly as it did before attribution existed."""
    assert unverified_claims("Nf3 was strong.", OWNED) == ()
    claims = unverified_claims("Bxc6 was strong.", OWNED)
    assert "move" in claims and "owned_move" not in claims


@pytest.mark.parametrize(
    "text",
    [
        # The usual half of the spec: a threat, a hypothetical, a question. Each
        # names a move the turn really holds, so only the attribution could
        # guard them — and a class that guards a threat is the one thing this
        # module must never ship.
        "I'll play Nf3.",
        "I'm going to play Nf3.",
        "You could have played Nc6.",
        "You should have played Nc6.",
        # The #340 frontier run, verbatim (#384).
        "you shoulda played Bxc4",
        "Did you play Nf3?",
        "If I play Nf3 you're in trouble.",
    ],
)
def test_an_owned_move_that_is_not_a_report_is_not_a_claim(text):
    assert unverified_claims(text, OWNED) == ()


def test_a_turn_that_supplied_no_attribution_falls_back_to_the_move_list():
    """A caller with no split to give — MCP, the delegate wire, anything older
    than this class — must not have every credited move guarded. With both sides
    empty the class reads the wider set, so it can still catch the invented move
    and never the true one."""
    unattributed = VerifiedFacts(moves=frozenset({"Nf3"}))
    assert unverified_claims("I played Nf3.", unattributed) == ()
    assert unverified_claims("You played Nf3.", unattributed) == ()
    assert "owned_move" in unverified_claims("I played Bxc6.", unattributed)


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
    settings={"difficulty": "casual", "voice": "on", "verbosity": "low"}
)


@pytest.mark.parametrize(
    ("text", "claim"),
    [
        ("Difficulty's on advanced now.", "difficulty"),
        ("You're playing maximum strength.", "difficulty"),
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
        "Voice output is on.",
        "Verbosity is low. Keeping it short.",
    ],
)
def test_a_setting_that_is_set_is_reportable(text):
    assert unverified_claims(text, CASUAL) == ()


# A *change* of how much gets said is its own claim, and the live value cannot
# settle it: "more detail from now on" names no level. What settles it is
# whether the turn moved the setting, so the fact is the narrower
# `settings_changed`. Twice in the 2026-09-04 walkthrough the model answered
# "talk more" by narrating the change and never calling `set_verbosity`; the
# setting stayed `low` on disk and the next turn was as terse as the last.

TALKED_MORE = VerifiedFacts(
    settings={"verbosity": "high"}, settings_changed=frozenset({"verbosity"})
)


@pytest.mark.parametrize(
    "text",
    [
        "Alright, more detail from now on.",
        "Talking more from here.",
        "Fewer words from me.",
        "Going chattier.",
        "You'll get more of the breakdown.",
    ],
)
def test_a_narrated_verbosity_change_needs_the_call(text):
    assert "verbosity_change" in unverified_claims(text, VerifiedFacts())
    assert unverified_claims(text, TALKED_MORE) == ()


@pytest.mark.parametrize(
    "text",
    [
        # A question and a condition — the shared hedges, doing their job.
        "Want me to talk more?",
        "If you want more detail, just ask.",
        # "More" of something that is not the talking.
        "e4 gives you more space.",
        "That rook is doing more work than your queen.",
        "Two more moves and this is over.",
    ],
)
def test_talk_that_is_not_a_verbosity_change_is_not_a_claim(text):
    assert unverified_claims(text, VerifiedFacts()) == ()


@pytest.mark.parametrize(
    "text",
    [
        "Want the difficulty up to advanced? Say the word.",
        "I can turn the voice off if it's annoying you.",
        # Hint talk is ordinary prose since the mode retired (2026-09-01):
        # there is no hints setting left to claim a value of.
        "Ask me for a hint if you want help.",
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

LEVEL = VerifiedFacts(material=(0,))
UP_A_KNIGHT = VerifiedFacts(material=(3,))
DOWN_TWO_PAWNS = VerifiedFacts(material=(-2,))


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
        ("I'm up a piece.", VerifiedFacts(material=(-3,))),
        ("You're two pawns down.", DOWN_TWO_PAWNS),
        ("Two pawns behind and it shows.", DOWN_TWO_PAWNS),
        ("I'm two pawns ahead.", DOWN_TWO_PAWNS),
        ("You're up the exchange.", VerifiedFacts(material=(2,))),
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
    knight_for_a_pawn = VerifiedFacts(material=(2,))
    assert unverified_claims("You're up a knight.", knight_for_a_pawn) == ()


# The count is plural because the turn has more than one board in it. The
# narrator reacts during the observation beat — after the player's move, while
# Stockfish is still computing its answer — so the position it counted is not
# the position the guard is standing in when it checks. Both are boards this
# turn really had, so a count either one backs is a count, and only a direction
# neither board supports is the invention the class exists for.


def test_a_count_from_the_board_the_narrator_saw_is_reportable():
    """The player takes a knight and the engine recaptures: +3 while the
    reaction is being written, 0 by the time it is checked. "You're up a piece"
    was true when it was said."""
    traded = VerifiedFacts(material=(3, 0))
    assert unverified_claims("Word, you're up a piece.", traded) == ()


def test_a_direction_no_board_this_turn_backs_is_still_a_claim():
    assert "material" in unverified_claims(
        "You're down a piece.", VerifiedFacts(material=(3, 0))
    )


def test_no_count_at_all_still_backs_nothing():
    """`()` is not `(0,)`: a turn that supplied no count licenses no count,
    while a level board genuinely backs "we're dead even". The class fails
    closed on evidence, never on a default that happens to read as one."""
    assert "material" in unverified_claims("You're two pawns down.", VerifiedFacts())
    assert "material" in unverified_claims("You're up a pawn.", VerifiedFacts())


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


# --- who is better, as a direction (#320) ------------------------------------------
#
# The number class checks a score was reported, never which way it was hung.
# With the player 1.5 down, "you're up 1.5" quotes a reported number and is
# the lie that matters, so the direction is its own class, held to the turn's
# engine verdicts from the player's side (`VerifiedFacts.advantages`).

PLAYER_BEHIND = VerifiedFacts(
    advantages=(-150,), numbers=frozenset({"150", "1.5", "-150", "-1.5"})
)
PLAYER_AHEAD = VerifiedFacts(advantages=(150,))


@pytest.mark.parametrize(
    "text",
    [
        "You're up 1.5, easy.",
        "You're winning, bro.",
        "You are clearly better here.",
        "I'm losing this one.",
        "I'm behind.",
        "I'm down 1.5 here.",
    ],
)
def test_a_direction_the_engine_contradicts_is_a_claim(text):
    assert unverified_claims(text, PLAYER_BEHIND) == ("advantage",)


@pytest.mark.parametrize(
    "text",
    [
        "You're down 1.5.",
        "You're losing, bro.",
        "I'm winning.",
        "I'm ahead, and it's not close.",
        "You're behind, but it's fixable.",
    ],
)
def test_a_direction_the_engine_backs_is_reportable(text):
    assert unverified_claims(text, PLAYER_BEHIND) == ()


@pytest.mark.parametrize(
    "text",
    [
        # Not verdicts: whose turn, advice, comparisons, material counts.
        "I'm up next.",
        "You're better off castling.",
        "I'm better than you and we both know it.",
        "You're up a knight.",
        "You're down to your last rook.",
        # Vibes are Glitch's to say.
        "You're cooked.",
        "You're doing okay, man.",
        # Futures, conditions and questions, the shared hedges.
        "You'll be winning after that trade.",
        "If you take it, you're ahead.",
        "Are you winning? Nah.",
    ],
)
def test_talk_that_is_not_a_verdict_is_no_advantage_claim(text):
    assert "advantage" not in unverified_claims(text, PLAYER_AHEAD)
    assert "advantage" not in unverified_claims(text, PLAYER_BEHIND) or (
        text == "You're up a knight."  # the material class's claim, not this one
    )


def test_a_near_level_verdict_backs_either_direction():
    """Inside `LEVEL_BAND_CP` a slight edge is talk either way."""
    nearly_level = VerifiedFacts(advantages=(-30,))
    assert unverified_claims("You're slightly better.", nearly_level) == ()
    assert unverified_claims("I'm slightly better.", nearly_level) == ()


def test_no_verdict_this_turn_licenses_the_direction_for_the_guard():
    """Nothing asked of the engine: the class passes, and the scorer drops the
    claim (`speech_accuracy._meaningful`) rather than count it backed."""
    assert unverified_claims("You're losing.", VerifiedFacts()) == ()


def test_a_forced_mate_is_the_strongest_verdict():
    mated = VerifiedFacts(advantages=(-(100_000 - 3),))
    assert unverified_claims("You're winning.", mated) == ("advantage",)
    assert unverified_claims("I'm winning.", mated) == ()


# The first `judgment_as_black` run (2026-09-28), verbatim: the player on Black
# a queen up (+654 their side), and the planner's note had read the score as
# White's. Every one is the lie #320 is about, and none named "you" or "I".
BLACK_A_QUEEN_UP = VerifiedFacts(advantages=(654,), player_color="black")


@pytest.mark.parametrize(
    "text",
    [
        "White's already up a ton, bro.",
        "You're cooked fr. White's way ahead.",
        "Word, you're good. White's got a slight edge.",
        "White's got a pretty big lead, fr.",
        "You're cooked, man. White's up big.",
    ],
)
def test_the_side_named_by_color_is_the_side_the_verdict_is_about(text):
    assert unverified_claims(text, BLACK_A_QUEEN_UP) == ("advantage",)


@pytest.mark.parametrize(
    "text",
    [
        # The same run once the result carried its own summary, verbatim.
        "Nasty. You're up about 6.3 pawns already.",
        "Word, you're up by about 6.5 pawns. That's clean.",
        "Black is winning.",
        "You've got a nice edge.",
        "I'm behind, no lie.",
    ],
)
def test_a_color_or_possession_the_verdict_backs_is_reportable(text):
    assert "advantage" not in unverified_claims(text, BLACK_A_QUEEN_UP)


def test_a_color_with_no_side_to_map_it_to_is_not_judged():
    unknown = VerifiedFacts(advantages=(654,))
    assert unverified_claims("White's way ahead.", unknown) == ()


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
    facts = VerifiedFacts(ended=True, winner="player", termination="checkmate")
    assert unverified_claims("Game over: 1-0 (checkmate).", facts) == ()


# --- the winner and the termination (astra audit F7, #287) ----------------------
#
# `ended` licenses the ending words; it cannot tell "you win by checkmate" from
# the resignation the player actually lost by. The outcome class reads the same
# words against the session's outcome — but only over a finished game, because
# on a live board the lie is that the game ended at all and that is the ending
# class's line. The bar is the file's: no new words, and the tables below are
# the labeled corpus the class is measured against — every line of finished-game
# commentary the deployed trace held on 2026-09-18 is in the must-not-fire table.

PLAYER_MATED = VerifiedFacts(ended=True, winner="player", termination="checkmate")
PLAYER_RESIGNED = VerifiedFacts(
    ended=True, winner="opponent", termination="resignation"
)
GLITCH_RESIGNED = VerifiedFacts(ended=True, winner="player", termination="resignation")
AGREED_DRAW = VerifiedFacts(ended=True, drawn=True, termination="agreement")
# A `new_game` ran: ended, no outcome behind it, and the reset itself backed.
FRESH_BOARD = VerifiedFacts(ended=True, restarted=True)


@pytest.mark.parametrize(
    "text, facts",
    [
        # The wrong winner.
        ("You win by checkmate.", PLAYER_RESIGNED),
        ("I win. Better luck next time.", PLAYER_MATED),
        ("You lost that one.", PLAYER_MATED),
        ("I lost. GG.", PLAYER_RESIGNED),
        ("You resigned, so I win.", PLAYER_MATED),
        ("I resign.", PLAYER_RESIGNED),
        # A winner where there was none.
        ("I win.", AGREED_DRAW),
        ("You lose.", AGREED_DRAW),
        # The wrong termination.
        ("Checkmate.", PLAYER_RESIGNED),
        ("That's mate.", GLITCH_RESIGNED),
        ("Stalemate.", AGREED_DRAW),
        ("Resigning now.", PLAYER_MATED),
        # A result on a board that has none: `ended` is true because a new game
        # ran, and the termination is unknown, which fails closed.
        ("Checkmate!", FRESH_BOARD),
    ],
)
def test_a_wrong_winner_or_termination_is_a_claim(text, facts):
    assert unverified_claims(text, facts) == ("outcome",)


@pytest.mark.parametrize(
    "text, facts",
    [
        # Every finished-game commentary the deployed trace held (2026-09-18).
        (
            "Damn, you actually did that. That's some filthy finishing move, bro.",
            PLAYER_MATED,
        ),
        ("GG. You actually cooked me.", PLAYER_MATED),
        ("Checkmate. That was nasty, bro.", PLAYER_MATED),
        ("clean. that was a nasty finish, bro.", PLAYER_MATED),
        ("Fr, that was a rough one.", PLAYER_RESIGNED),
        ("Word. GG.", PLAYER_RESIGNED),
        # The true report, in the words the class reads.
        ("Checkmate, you win.", PLAYER_MATED),
        ("You won, fair and square.", PLAYER_MATED),
        ("I lost that one.", PLAYER_MATED),
        ("You resigned. I'll take it.", PLAYER_RESIGNED),
        ("You lose. Resigning was the right call, though.", PLAYER_RESIGNED),
        ("I resigned. You had me.", GLITCH_RESIGNED),
        (
            "Stalemate. We're splitting it.",
            VerifiedFacts(ended=True, drawn=True, termination="stalemate"),
        ),
        ("Game over.", PLAYER_MATED),
        ("Fresh board. New game.", FRESH_BOARD),
        # Post-mortem talk: hypothetical, conditional, hedged, a question.
        ("You'd have won with Rxe5.", PLAYER_RESIGNED),
        ("If you hadn't resigned I was losing.", PLAYER_RESIGNED),
        ("I should've won that.", PLAYER_MATED),
        ("You almost lost that one.", PLAYER_MATED),
        ("I'll win the rematch.", PLAYER_MATED),
        ("Want a new game?", PLAYER_MATED),
        ("Rematch? I win next time.", PLAYER_MATED),
        ("Not checkmate — you resigned.", PLAYER_RESIGNED),
        # Trash talk the class has no words for, by design.
        ("You're cooked.", PLAYER_RESIGNED),
        ("I'm the winner here.", PLAYER_MATED),
        ("That's a win for the good guys.", PLAYER_MATED),
        ("Total domination.", PLAYER_RESIGNED),
    ],
)
def test_finished_game_commentary_that_tells_the_truth_is_not_a_claim(text, facts):
    assert unverified_claims(text, facts) == ()


@pytest.mark.parametrize(
    "text", ["I win.", "Checkmate!", "You resigned.", "That's mate, you lose."]
)
def test_on_a_live_board_only_the_ending_class_speaks(text):
    """One correction per sentence: the outcome class defers to `ended`."""
    assert unverified_claims(text, NOTHING) == ("ending",)


# A level middlegame with a little history, for the whole-reading checks below.
LEVEL = VerifiedFacts(
    material=(0,),
    settings={"voice": "on", "verbosity": "low", "difficulty": "casual"},
    moves=frozenset({"e4", "e5", "Nf3", "Nc6"}),
    moves_by_player=frozenset({"e4", "Nf3"}),
    moves_by_opponent=frozenset({"e5", "Nc6"}),
    captured_by_player=frozenset({"pawn"}),
)


@pytest.mark.parametrize(
    "text, claim",
    [
        ("Game over.", "ending"),
        ("We drew that one.", "draw"),
        ("You're in check.", "check"),
        ("Snagged your bishop.", "capture"),
        ("Rxe5 wins on the spot.", "move"),
        ("I played Nf3, obviously.", "owned_move"),
        ("You played Nc6 there.", "owned_move"),
        ("Saved it as scholars.", "save"),
        ("Voice is off now.", "voice"),
        ("Difficulty is maximum now.", "difficulty"),
        ("Verbosity is high.", "verbosity"),
        ("Alright, more detail from now on.", "verbosity_change"),
        ("You're at -3.5 here.", "evaluation"),
        ("You're up a knight.", "material"),
        ("Classic Sicilian Defence.", "opening"),
        ("I've won 3 games against you.", "results"),
    ],
)
def test_every_claim_class_reads_an_unbacked_line(text, claim):
    """One unbacked line per class, on a level board."""
    assert unverified_claims(text, LEVEL) == (claim,)


def test_unverified_claims_is_the_same_reading_by_class_name():
    text = "Word. Game over. Snagged your bishop. Rxe5 wins."
    assert unverified_claims(text, LEVEL) == ("ending", "capture", "move")
    assert unverified_claims("Nf3. Your move.", LEVEL) == ()


# --- the actions no board fact backs, and the reply that has not happened (#289)
#
# The labeled corpus the three classes are measured against, per the #287 rule:
# false positives measured before the regex ships. Must-fire lines are the
# false reports a narrator repeating a false planner note would make; must-not
# lines are banter, offers, clarifications, idioms that share the words, and
# the deployed trace's real takeback and reset narrations (2026-09-04..18),
# which pass once the turn's own `undo` / `new_game` backs them. The deployed
# sweep over all 219 recorded drafts is in docs/agent-evals.md.

UNDONE = VerifiedFacts(undone=True)
RESTARTED = VerifiedFacts(restarted=True)


@pytest.mark.parametrize(
    ("text", "claim"),
    [
        ("Done, taken back.", "takeback"),
        ("Took it back.", "takeback"),
        ("Undid your last move.", "takeback"),
        ("Took your knight back for you.", "takeback"),
        ("Took back your bishop move.", "takeback"),
        ("Move undone.", "takeback"),
        ("That's undone. Go again.", "takeback"),
        ("Rolled it back.", "takeback"),
        ("Takeback done.", "takeback"),
        ("bet. back to where we were.", "takeback"),
        ("The board's back to what it was a few turns ago.", "takeback"),
        ("Fresh board. Your move.", "restart"),
        ("yo, fresh start. your move, man.", "restart"),
        ("Starting over.", "restart"),
        ("Board's reset.", "restart"),
        ("I reset the board.", "restart"),
    ],
)
def test_an_action_nothing_did_is_a_claim(text, claim):
    assert claim in unverified_claims(text, NOTHING)


@pytest.mark.parametrize(
    ("text", "facts"),
    [
        # Offers, questions and threats: the hedges' territory.
        ("Want me to undo that?", NOTHING),
        ("Undo it and try again?", NOTHING),
        ("I'll take it back if you ask.", NOTHING),
        ("Let me take that back.", NOTHING),
        ("Say the word and I'll roll it back.", NOTHING),
        ("Take it back?", NOTHING),
        ("You can start over any time.", NOTHING),
        ("No takebacks.", NOTHING),
        # Idioms and chess talk that share the words.
        ("You undid all your good work.", NOTHING),
        ("You took the lead back there.", NOTHING),
        ("You took back control of the center.", NOTHING),
        ("Your position's coming undone.", NOTHING),
        ("Your defence came undone.", NOTHING),
        ("You got taken back to school.", NOTHING),
        ("Fresh ideas, same blunders.", NOTHING),
        ("Back to the drawing board. Your turn, man.", NOTHING),
        ("Back where you left it.", NOTHING),  # a resumed save, not a takeback
        # The deployed trace's real reports, with the evidence that backs them.
        ("Word. The board's back to what it was a few turns ago.", UNDONE),
        ("bet. back to where we were.", UNDONE),
        ("Done, taken back.", UNDONE),
        ("yo, fresh start. your move, man.", RESTARTED),
        ("Fresh board. Your move.", RESTARTED),
    ],
)
def test_talk_and_true_reports_are_not_action_claims(text, facts):
    claims = unverified_claims(text, facts)
    assert "takeback" not in claims and "restart" not in claims


PENDING = VerifiedFacts(
    moves=frozenset({"Nf6", "Nc6", "e4"}),
    moves_by_player=frozenset({"e4"}),
    moves_by_opponent=frozenset({"Nf6"}),
    unplayed_replies=frozenset({"Nf6", "Nc6"}),
)


@pytest.mark.parametrize(
    "text",
    [
        "My turn. Nf6.",
        "King's pawn. ...Nf6.",
        "I played Nf6.",
        "Nc6, obviously.",
    ],
)
def test_naming_the_reply_before_it_exists_is_a_claim(text):
    assert "unplayed_reply" in unverified_claims(text, PENDING)


@pytest.mark.parametrize(
    "text",
    [
        "I'll hit you with Nf6.",  # a threat
        "Nf6 is coming.",
        "Should I go Nf6?",
        "e4, classic.",  # the player's own move
        "Bold.",
    ],
)
def test_threats_and_the_players_move_are_not_the_reply(text):
    assert "unplayed_reply" not in unverified_claims(text, PENDING)


def test_the_reply_class_is_silent_when_no_reply_was_pending():
    facts = VerifiedFacts(
        moves=frozenset({"Nf6"}), moves_by_opponent=frozenset({"Nf6"})
    )
    assert unverified_claims("I played Nf6.", facts) == ()


# --- the whole reading, for the scorer (#367) -----------------------------------
#
# The scorer needs the backed claims as well as the unbacked ones;
# `unverified_claims` is the same reading with the backed ones left out.


def test_claims_keep_the_backed_reading_beside_the_unbacked_one():
    found = claims("I took your knight. You took my queen.", TOOK_A_KNIGHT)

    assert [(c.claim, c.backed) for c in found] == [
        ("capture", True),
        ("capture", False),
    ]
    assert [(c.claim, c.sentence) for c in found if not c.backed] == [
        ("capture", "You took my queen.")
    ]


def test_a_sentence_is_one_claim_per_class_unbacked_if_any_part_is():
    facts = VerifiedFacts(moves=frozenset({"Nf3"}))

    both = claims("Nf3 and Qh5.", facts)
    one = claims("Nf3 and Nf3 again.", facts)

    assert [(c.claim, c.said, c.backed) for c in both if c.claim == "move"] == [
        ("move", "Qh5", False)
    ]
    assert [(c.claim, c.said, c.backed) for c in one if c.claim == "move"] == [
        ("move", "Nf3", True)
    ]


def test_hedged_talk_is_no_claim_at_all():
    assert claims("One more move and it's checkmate.", NOTHING) == ()
    assert claims("I'll take your knight next.", NOTHING) == ()


# --- the opening (#373) ---------------------------------------------------------

# A game at 1. e4 e5 2. Nf3 Nc6 3. Bb5 a6: every name its line passed through.
IN_THE_MORPHY = VerifiedFacts(
    openings=frozenset(
        {
            "king's pawn game",
            "king's knight opening",
            "king's knight opening: normal variation",
            "ruy lopez",
            "ruy lopez: morphy defense",
        }
    )
)


@pytest.mark.parametrize(
    "text",
    [
        "Classic Ruy Lopez.",
        "We're deep in the Ruy Lopez: Morphy Defense.",
        "the ruy lopez, my favourite.",
        "King’s Pawn Game from move one.",
    ],
)
def test_an_opening_the_line_passed_through_is_backed(text):
    assert [(c.claim, c.backed) for c in claims(text, IN_THE_MORPHY)] == [
        ("opening", True)
    ]


@pytest.mark.parametrize(
    "text",
    [
        "This is the Sicilian.",
        "Textbook Italian Game.",
        "We're in the Grünfeld Defence.",
        "Ah, the Caro Kann.",
    ],
)
def test_an_opening_the_line_never_reached_is_unbacked(text):
    assert unverified_claims(text, IN_THE_MORPHY) == ("opening",)


@pytest.mark.parametrize(
    "text",
    [
        # Pieces and ordinary words the vocabulary leaves out.
        "Your king's knight is loose.",
        "The four knights on the board are staring at each other.",
        "Say it in English.",
        # A head with no article is not read.
        "Italian food after this?",
        # Questions, conditions and plans stay talk.
        "Want to try the Sicilian next game?",
        "If you play c5 it's the Sicilian.",
        "I'll steer this into a French Defense.",
    ],
)
def test_opening_talk_that_is_not_a_claim(text):
    assert "opening" not in unverified_claims(text, IN_THE_MORPHY)


def test_no_opening_is_sayable_without_one():
    assert unverified_claims("Classic Ruy Lopez.", LEVEL) == ("opening",)


# --- the results tally (#373) ----------------------------------------------------

# Five games: Glitch won three, the player one, one drawn; all at casual.
FIVE_PLAYED = VerifiedFacts(
    results=frozenset(
        {("games", 5), ("engine_won", 3), ("player_won", 1), ("drawn", 1)}
    )
)


@pytest.mark.parametrize(
    "text",
    [
        "I've won 3 games against you.",
        "I have won three games.",
        "You've lost three times.",
        "You beat me 1 time.",
        "We've played five games.",
        "I've got 3 wins.",
        "You've got 1 win.",
        "You have 3 losses.",
        "One win, three losses, one draw.",
        # Glitch's own ways of saying it, from the first live run.
        "I've taken three of your games, bro.",
        "I've taken three of 'em.",
        "I've got you three times, bro.",
        "I've taken three from you so far, bro.",
        "Yo, we're at five games so far.",
        "Yo, we've had 5 games so far.",
        "We've done five games so far.",
        "We've clocked five games so far, bro.",
        "I've taken three of 'em from you, bro.",
    ],
)
def test_a_count_the_tally_holds_is_backed(text):
    assert [(c.claim, c.backed) for c in claims(text, FIVE_PLAYED)] == [
        ("results", True)
    ]


@pytest.mark.parametrize(
    "text",
    [
        "You've won 3 games.",
        "I won 4 games.",
        "You beat me twice.",
        "We played 4 games.",
        "You have 2 losses.",
        "Two wins apiece.",
        "You've taken four of my games.",
    ],
)
def test_a_count_the_tally_does_not_hold_is_unbacked(text):
    assert unverified_claims(text, FIVE_PLAYED) == ("results",)


@pytest.mark.parametrize(
    "text",
    [
        "You lost 3 pawns in that trade.",
        "I won that game.",
        "Want to play 5 games?",
        "If you win 3 games I'll be impressed.",
        "I'll win ten games in a row.",
        "I've got three pawns on the kingside.",
        "You took 2 of my pawns.",
    ],
)
def test_talk_that_is_not_a_record(text):
    assert "results" not in unverified_claims(text, FIVE_PLAYED)
