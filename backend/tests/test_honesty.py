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


# --- naming a playable move (the invented-advice leak) -------------------------
#
# Audit item 11's second half: the model can invent a move from its own head —
# the 2026-07-13 trace leak, still measured live after the capability cut
# (2/5–3/5). The payload of a hint is a SAN token the player could play right
# now, whatever prose surrounds it, so that is what the predicate matches. The
# pipeline pairs it with the turn's evidence: a move is licensed exactly when
# an analysis tool reported it this turn (hints mode is gone, 2026-09-01 — the
# license is evidence now, never a setting).

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
        # over a move — commentary with no engine consult behind it is
        # supposed to look like this.
        "Figure it out yourself.",
        "Ask me for a hint if you actually want help.",
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
        "Took your bishop back. Try again.",
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
