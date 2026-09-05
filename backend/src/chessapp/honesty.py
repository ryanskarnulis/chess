"""The honesty guard: commentary may not announce an event that never happened.

The house rule says the model never decides what deterministic state already
knows. Commentary is where that rule leaks: the loop's closing turn is produced
from a context ending in the new board, and it *still* invents the board it
wanted — "Word. Game over." with no tool call and a live game, "you actually
have me in checkmate" after a quiet rook move (trace review 2026-07-13, finding
6). That is the worst failure the app has, because the player is *told* the game
ended. A prompt rule is no defense: a 12B follows one about half the time.

So the pipeline checks the claim against the board before it emits it. This
module owns only the string half of that check — does this text *assert* an
event? Whether it happened is the session's answer, and `api._run_command` puts
the two together.

The ending is where the rule started and it is not where it stops (audit item
13). `unverified_claims` takes the same shape to every operational fact a turn
produces — captures, check, draws, moves, saves, settings, engine numbers, the
material count — against a `VerifiedFacts` set the pipeline assembles from the
tool results, the engine's reply and the board. Personality varies the wording;
it does not get to vary the facts.

The bar is deliberately "asserts", not "mentions". Trash talk, threats,
questions and hypotheticals are the whole point of the commentary — "one more
move and it's checkmate", "want a new game?", "I'll take that knight" — and
suppressing those would cost far more than the lie does. So a claim must be an
unhedged assertion in its own sentence, a class that cannot tell a threat from
a report does not belong here, and the tests in `test_honesty.py` are the spec.
"""

import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field

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
          | about \s+ to | next \s+ move | avoid | prevent
          # Evaluative talk about how a position *looks*. A live position
          # called "basically a draw" is trash talk about the board, not a
          # report that the game ended in one.
          | basically | practically | essentially | pretty \s+ much
          | heading | headed | toward | towards | probably | likely
          | looks \s+ like | looked \s+ like | feels \s+ like | dead \s+ drawn )\b
    """,
    re.IGNORECASE | re.VERBOSE,
)

# The extra hedges every class *except* the ending one carries: intention,
# prediction, permission. "I'll take your knight" is the threat that makes
# Glitch worth playing; "I took your knight" is the claim. The ending class is
# deliberately left out — it is pinned to today's behavior, where "You
# resigned — I'll take it." is an assertion about an event that already
# happened, and reading `'ll` there as a hedge would let the worst lie through.
# `coming`/`incoming`/`on the way`/`looming` are how a threat is actually
# spoken. The explicit futures were all here and the idiomatic ones were not,
# so "Qh5 is coming" — a move that is deliberately *not* playable yet, which is
# the entire point of a threat — read as a claim that it had been played.
_FUTURE = re.compile(
    r"""
    '(?: ll | d )
    | \b(?: will | gonna | going \s+ to | plan | planning | plans
          | next | soon | after | then | let \s+ me | can | say \s+ the \s+ word
          | coming | incoming | looming | on \s+ the \s+ way )\b
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


# What a token sheds before it is compared against the legal list: the prose
# punctuation and markdown a SAN move arrives wrapped in ("Nf3.", "`e4`").
_TOKEN_WRAPPING = ".,!?`*()[]{}:;\"'—"


def names_a_legal_move(text: str, legal_moves: Iterable[str]) -> bool:
    """True when the text names a move playable in the current position.

    The other honesty predicate's sibling, for the invented-advice leak
    (audit item 11): the model can hand over a move it never checked with the
    engine — and the payload of a hint is a SAN token from the current
    `legal_moves`, whatever prose surrounds it. No hedge analysis here: there
    is no sentence shape that makes handing over an unbacked playable move
    fine. Whether the turn's evidence *licenses* the move (an analysis tool
    reported it) is the pipeline's half, exactly as with
    `claims_destructive_outcome`.
    """
    legal = set(legal_moves)
    return any(token.strip(_TOKEN_WRAPPING) in legal for token in text.split())


# --- the verified facts, and the claims that must derive from them -------------
#
# Audit item 13. The ending guard proved the shape and the shape generalizes:
# assemble what deterministically happened this turn, then require the
# *operational* claims in the commentary to derive from it. Personality varies
# the wording; it does not get to vary the facts.
#
# The evidence is assembled by the pipeline (`api._verified_facts`), because
# that is where the tool results, the engine's reply and the board all are.
# This module still owns only the reading of the string.


@dataclass(frozen=True)
class VerifiedFacts:
    """What the turn can honestly say, as the board and the tools left it.

    Every field defaults to "nothing happened", so a fact the pipeline does not
    supply is a fact the commentary may not assert — the guard fails closed on
    evidence, exactly as the ending check does.

    Captures are split by side because who took what is a fact too, and they
    hold the *game's* captured pieces rather than only this turn's: a capture
    that really happened ten moves ago is a true fact awkwardly placed, and
    guarding tense would cost far more than it saves. What the class is for is
    the piece that was never taken at all.

    `moves` is every SAN the turn accounts for — played, reported by an
    analysis, or playable now or at the turn's start. The last of those is what
    keeps "Bc4 was right there and you missed it" sayable.

    `moves_by_player` and `moves_by_opponent` are the narrower fact that width
    cannot hold: the moves each side really *played*, game-spanning like the
    captured sets. "I played Nf3" derives from `moves` perfectly when the
    player is the one who played Nf3, so crediting a move was unguardable
    until the sides were told apart. A caller that supplies neither set is
    supplying no attribution at all, and the owned-move class reads `moves`
    instead — missing evidence is the one thing this guard fails permissive on,
    since the alternative is guarding every credited move on the routes that
    have no pipeline to assemble them.

    `material` is the player's advantage in pawns, positive when they are
    ahead — and a *tuple* of them, because one turn holds more than one board.
    The narrator reacts during the observation beat, after the player's move
    and while Stockfish is still computing its answer, so the position it
    counted is not the position the guard is standing in when it checks: a
    recapture is +3 to the reaction and 0 to the check. Every board this turn
    really had contributes its count and any of them backing a claim is enough,
    on the same reasoning that lets `captured_by_player` span the whole game —
    staleness is not invention, and the class exists for the direction no board
    supports. Empty rather than `(0,)` for a turn that supplied no count,
    because a level board is itself a claimable fact ("we're dead even") and
    the guard fails closed on evidence, never on a default that happens to
    read as one.

    `settings` is the live value of each setting; `settings_changed` is the
    narrower fact of which ones a tool actually moved *this turn*. They answer
    different claims and only one of them can answer the second: "verbosity is
    high" is true whenever it is high, whoever set it and whenever, while
    "alright, more detail from now on" names no level at all and is true only
    if the turn really changed one. Announcing a change that never happened is
    the whole of walkthrough #3.

    `captures_by_move` is what each move the turn knows about takes, read off
    the board it is legal on — the piece's name, or `""` for a quiet move. It
    is the one fact that can settle *which* piece a capture claim names, which
    the game-wide capture record cannot: a queen taken twenty moves ago backs
    "taken that queen" perfectly. Live it backed exactly nothing — the engine
    said Qxe2 was the better move, Qxe2 takes a pawn, and the narration called
    it a queen (walkthrough #5).
    """

    ended: bool = False
    drawn: bool = False
    check: bool = False
    captured_by_player: frozenset[str] = frozenset()
    captured_by_opponent: frozenset[str] = frozenset()
    moves: frozenset[str] = frozenset()
    moves_by_player: frozenset[str] = frozenset()
    moves_by_opponent: frozenset[str] = frozenset()
    saved: bool = False
    settings: Mapping[str, str] = field(default_factory=dict)
    settings_changed: frozenset[str] = frozenset()
    captures_by_move: Mapping[str, str] = field(default_factory=dict)
    numbers: frozenset[str] = frozenset()
    material: tuple[int, ...] = ()


_PIECE_WORDS = r"(?: pawn | knight | bishop | rook | queen | king | horse )"

# The verbs that report a capture as done — inflected forms only. The bare
# stem is missing on purpose, and so is "taken": neither reports anything. A
# bare "take" is an imperative ("Take the rook."), an infinitive ("try to take
# the rook") or a subjunctive, which is how every piece of *advice* about a
# capture is phrased — and advice is what the player asked for. "Taken" is
# nearly always passive and predictive ("that knight is going to get taken").
#
# Most bare stems were already absent ("capture", "snatch", "nab", "eat"); the
# two that were not cost the app every hint whose best move was a capture. All
# three "what should I play?" turns of the 2026-09-04 walkthrough answered
# "Take the rook. Rxd1 is the move." and all three were suppressed.
#
# "Taken" is here only with a perfect auxiliary. Bare, it is passive and
# predictive — "that knight is going to get taken" — but "you've taken that
# queen" reports a completed capture as flatly as "you took" does, and that is
# the tense the mistake narration invents its victims in. Not `'d`: the
# contraction is "would" as often as "had", and the capture class already
# reads it as a future.
_TAKE_VERBS = (
    r"(?: took | takes | taking | grabbed | grabs | grabbing "
    r"| captured | captures | capturing | snagged | snags | snatched | nabbed "
    r"| ate | eats | eating "
    r"| (?: have | has | had | ' (?: ve | s ) ) \s+ taken )"
)

# The possessive is direction evidence, and it is the direction Glitch's own
# register actually carries: he says "Snagged your bishop." and "Your knight is
# gone.", not "I took your bishop." Every subjectless phrasing used to fall
# through to the union of both sides' captures, so a capture announced in
# exactly the wrong direction read as verified. Whose piece it was is not
# ambiguous, and the speaker is always the player's opponent, so whose piece it
# was says who took it.
_POSSESSIVE = r"(?: your | my )"

_CAPTURE = re.compile(
    rf"""
    (?: (?P<subject> \b (?: i | you ) \b ) [^.!?]{{0,24}}? )?
    \b {_TAKE_VERBS} \b [^.!?]{{0,24}}?
    (?: \b (?P<owner> {_POSSESSIVE} ) \s+ )?
    \b (?P<piece> {_PIECE_WORDS} ) s? \b
    |
    (?: \b (?P<gone_owner> {_POSSESSIVE} ) \s+ )?
    \b (?P<gone_piece> {_PIECE_WORDS} ) s? \b \s* (?: is | 's | are ) \s+
    (?: gone | dead | history | mine | yours | toast | off \s+ the \s+ board )\b
    """,
    re.IGNORECASE | re.VERBOSE,
)

_CHECK = re.compile(
    r"""
    \b in \s+ check \b
    | \b (?: that | this | it ) (?: 's | \s+ is ) \s+ check \b
    | \b check \s+ on \s+ (?: your | the | my ) \s+ king \b
    """,
    re.IGNORECASE | re.VERBOSE,
)

# A draw is its own fact, not a flavour of the ending class: a game that ended
# in mate did not end in a draw either, and the ending class would wave that
# through.
#
# "Drawn" is the wrinkle, and it is the material class's distinction again:
# almost every use of it is an *assessment* of a level position ("this is
# drawn", "a drawn endgame"), which is opinion and the trash talk the guard
# exists to protect — while the report is about *the game* ("the game's
# drawn"). Reading the bare adjective as a claim guarded ordinary commentary on
# exactly the symmetrical positions that earn the word, so it is now read only
# with the game as its subject. The other spellings are unambiguous reports
# already: nobody says "we drew" about a position.
_DRAW = re.compile(
    r"""
    \b(?: stalemate
        | (?: a | the ) \s+ draw
        | (?: game | match ) \s* (?: is | 's | was ) \s+ drawn
        | (?: we | it ) \s+ (?: drew | tied )
        | split (?: ting )? \s+ (?: it | the \s+ point )
    )\b
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Unambiguous move notation only. A bare pawn push is spelled exactly like a
# square ("the pawn on e4") and squares are discussed constantly, so `e4` alone
# is never read as a claim that a move was played — the class exists for the
# invented piece move ("you took it with Bxc6"), not for square talk. Case
# matters: SAN piece letters are capitals.
_SAN = r"""
    (?<! [\w-] )
    (?: O-O (?: -O )?
      | [KQRBN] [a-h]? [1-8]? x? [a-h] [1-8]
      | [a-h] x [a-h] [1-8]
    ) (?: = [QRBN] )? [+#]?
    (?! [\w-] )
"""

_SAN_CLAIM = re.compile(_SAN, re.VERBOSE)

# Who played it, when the sentence says. `moves` is wide by design and so it
# verifies "I played Nf3" off the player's own Nf3 — the credit was the one part
# of a move claim nothing checked. Only a *played* move can be credited, so the
# verbs are the ones that report a move as made; "I castled" names no SAN and is
# not this class's business.
_PLAY_VERBS = (
    r"(?: played | play | went | pushed | moved | dropped | slid | swung "
    r"| replied \s+ with | answered \s+ with | met \s+ it \s+ with )"
)

# Case-sensitive on the SAN half and not on the prose half: SAN piece letters
# are capitals, which is what keeps ordinary words out of the move classes.
_OWNED_MOVE = re.compile(
    rf"""
    (?i: \b (?P<subject> i | you ) \b [^.!?]{{0,16}}? \b {_PLAY_VERBS} \b )
    [^.!?]{{0,16}}? (?P<san> {_SAN} )
    """,
    re.VERBOSE,
)

_SAVE = re.compile(
    r"""
    \b sav (?: ed | ing ) \s+
        (?: it | that | this | the \s+ game | your \s+ game | the \s+ position )\b
    | \b (?: game | position | progress ) \s+ (?: is \s+ )? saved \b
    | \b saved \s+ (?: it \s+ )? (?: as | under )\b
    | \b (?: resumed | restored | reloaded | loaded ) \s+
        (?: it | that | this | the \s+ game | your \s+ game )\b
    | \b (?: game | position ) \s+ (?: is \s+ )?
        (?: resumed | restored | reloaded | loaded )\b
    """,
    re.IGNORECASE | re.VERBOSE,
)

# The settings classes. Each match names a value; the fact it is checked
# against is the *live* setting, not a tool call, because the live setting is
# the truth either way — one the player changed three turns ago is as true as
# one changed this turn, and a value nothing ever set is a lie however
# confidently it is announced. (Hints had a class here until the mode was
# retired 2026-09-01 — with no setting to be right or wrong about, hint talk
# is ordinary prose, and unbacked move advice is the pipeline's guard's job.)
_VOICE = re.compile(
    r"""
    \b voice (?: \s+ output )? \s+ (?: is | 's | are ) \s+ (?: now \s+ )?
        (?P<value> on | off | muted | unmuted | enabled | disabled )\b
    | \b (?: turn | turned | turning | switch | switched | switching ) \s+
        (?P<value_b> on | off ) \s+ (?: your | the | my ) \s+
        voice (?: \s+ output )? \b
    | \b (?: turned | switched ) \s+ (?: your | the | my ) \s+
        voice (?: \s+ output )? \s+ (?P<value_c> on | off )\b
    """,
    re.IGNORECASE | re.VERBOSE,
)

_TIERS = r"(?: beginner | casual | intermediate | advanced | maximum )"

_DIFFICULTY = re.compile(
    rf"""
    \b (?: difficulty | strength | tier ) \b [^.!?]{{0,20}}? \b (?P<value> {_TIERS} )\b
    | \b (?P<value_b> {_TIERS} ) \b [^.!?]{{0,20}}?
        \b (?: difficulty | strength | tier )\b
    | \b (?: playing | set \s+ to ) \s+ (?P<value_c> {_TIERS} )\b
    """,
    re.IGNORECASE | re.VERBOSE,
)

_VERBOSITY = re.compile(
    r"""
    \b verbosity \b [^.!?]{0,16}? \b (?P<value> low | normal | high )\b
    | \b (?P<value_b> low | normal | high ) \s+ verbosity \b
    """,
    re.IGNORECASE | re.VERBOSE,
)

# The *change* the value class cannot see. Nobody asks for "verbosity high" —
# they say "talk more", and the answer that came back twice in the 2026-09-04
# walkthrough narrated the change without calling the tool, leaving the
# setting `low` on disk and the next turn as terse as the one before. Naming
# no level, the sentence has nothing to check against the live value; what
# settles it is whether the turn actually moved the setting.
#
# Two halves in either order, built the way the material class is: a
# direction, and the thing being measured, which is the talking. Neither half
# alone is a claim — "more space" is a position and "the breakdown" is a noun
# — and the few words carrying both halves at once get their own branch.
_TALK = (
    r"(?: talk | talking | chat | chatter | say | saying | detail | details "
    r"| breakdown | commentary | words | rambling | explanation )"
)
_MORE_OR_LESS = r"(?: more | less | fewer | longer | shorter )"

_VERBOSITY_CHANGE = re.compile(
    rf"""
    \b (?: chattier | quieter | wordier | briefer | terser
          | (?: more | less ) \s+ talkative )\b
    | \b {_MORE_OR_LESS} \b [^.!?]{{0,20}}? \b {_TALK} \b
    | \b {_TALK} \b [^.!?]{{0,20}}? \b {_MORE_OR_LESS} \b
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Material: a count, not an opinion. The class ships only because it can tell
# "you're getting crushed" (an opinion about the position, and the trash talk
# that makes Glitch worth playing) from "you're two pawns down" (a claim the
# board settles) — so it reads *quantified* material talk only, a direction
# plus a named amount, in either order and from either side's mouth.
#
# "The exchange" is the one amount with no fixed value in ordinary use — a rook
# for a minor piece to some players, "won some material" to others — so it
# claims a direction and nothing more.
_MATERIAL_UNIT = r"(?: pawn | knight | bishop | piece | rook | queen )"
_MATERIAL_COUNT = r"(?: a | an | one | two | three | four | five | \d )"
_AHEAD = r"(?: up | ahead )"
_BEHIND = r"(?: down | behind )"

_MATERIAL = re.compile(
    rf"""
    (?: (?P<subject> \b (?: i | you ) \b ) [^.!?]{{0,16}}? )?
    (?: \b (?P<direction> {_AHEAD} | {_BEHIND} ) \s+
          (?: (?P<count> {_MATERIAL_COUNT} ) \s+ (?P<unit> {_MATERIAL_UNIT} ) s?
            | the \s+ exchange ) \b
      | \b (?P<count_b> {_MATERIAL_COUNT} ) \s+ (?P<unit_b> {_MATERIAL_UNIT} ) s? \s+
          (?P<direction_b> {_AHEAD} | {_BEHIND} ) \b
      | \b the \s+ exchange \s+ (?P<direction_c> {_AHEAD} | {_BEHIND} ) \b
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# What each named amount is worth. "A piece" is a minor piece by convention.
_MATERIAL_VALUES = {
    "pawn": 1,
    "knight": 3,
    "bishop": 3,
    "piece": 3,
    "rook": 5,
    "queen": 9,
}

_MATERIAL_COUNTS = {
    "a": 1,
    "an": 1,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
}

# Numbers only an engine can produce: a signed or decimal score, a centipawn
# count, a mate-in-N. Material talk is the class above rather than a number to
# quote: the board counts it, so it is checked against the count.
#
# The bounds and the lookarounds are what keep the app's own text out. A game
# result must not read as a score: in "1-0" the minus is preceded by a digit.
# Neither must a PGN, which Glitch legitimately reads out whole — a scoreless
# `1.` move number has no digit after the point, and a `2023.10.27` date is too
# long on both sides of it (a real evaluation is one or two digits either way).
_EVALUATION = re.compile(
    r"""
    (?<! [\w/.-] ) (?P<score> [+-] \d{1,2} (?: \. \d{1,2} )? ) (?! [\w/-] )
    | (?<! [\w/.-] ) (?P<decimal> \d{1,2} \. \d{1,2} ) (?! [\w/-] )
    | \b (?P<centipawns> \d+ ) \s* (?: centipawns? | cp )\b
    | \b mate \s+ in \s+ (?P<mate> \d+ )\b
    """,
    re.IGNORECASE | re.VERBOSE,
)

# How a matched setting value is spelled in `VerifiedFacts.settings`.
_SETTING_SYNONYMS = {
    "enabled": "on",
    "disabled": "off",
    "unmuted": "on",
    "muted": "off",
}


def _matched_value(match: re.Match[str]) -> str:
    """The value a settings match named, whichever alternative caught it."""
    for name in ("value", "value_b", "value_c"):
        if (value := match.group(name)) is not None:
            return _SETTING_SYNONYMS.get(value.lower(), value.lower())
    return ""


def _setting_is(key: str) -> Callable[[re.Match[str], VerifiedFacts], bool]:
    return lambda match, facts: facts.settings.get(key) == _matched_value(match)


def _victims_of_named_moves(sentence: str, facts: VerifiedFacts) -> set[str] | None:
    """What the moves this sentence names take, when the board knows.

    The most precise evidence a capture claim can be held to, and the only one
    that can catch the wrong *piece*: the capture record spans the whole game,
    so a queen taken twenty moves ago backs "that queen" no matter which move
    the sentence hangs it on. `captures_by_move` is per move and read off the
    board — `""` when the move takes nothing at all.

    Fail-permissive on ambiguity, as everywhere else here: with two known
    moves named, any one of them backing the piece is enough. `None` when the
    sentence names no move the board could resolve, which hands the claim back
    to the coarser evidence.
    """
    victims = {
        facts.captures_by_move[san]
        for match in _SAN_CLAIM.finditer(sentence)
        for san in (match.group(0).rstrip("+#"),)
        if san in facts.captures_by_move
    }
    return victims or None


def _is_advice(sentence: str, facts: VerifiedFacts) -> bool:
    """Whether the sentence hangs what it says on a move nobody has played.

    The other half of the advice problem, for the tense the verb list cannot
    rule out: "Rxd1 takes the rook" and "Rxd1 and that rook is gone" describe
    what a *recommended* move would do, and the present tense is how anybody
    describes a line. The move itself is the tell — it is one the turn knows
    about (legal now, or reported by an analysis) and one that no side has
    actually played — and the board already holds both halves of that.

    Weaker than an explicit subject, which is why `_capture_happened` reads
    this only after "I"/"you" have had their say: a person is not a line, and
    "I took your queen with Qxe2" stays a claim about an event.
    """
    played = facts.moves_by_player | facts.moves_by_opponent
    return any(
        _names(match.group(0), facts.moves) and not _names(match.group(0), played)
        for match in _SAN_CLAIM.finditer(sentence)
    )


def _capture_happened(match: re.Match[str], facts: VerifiedFacts) -> bool:
    """Whether the board's capture record backs the capture this sentence says.

    By the strongest evidence the sentence carries. A named move whose victim
    the board knows decides outright — it is the only evidence about *this*
    capture rather than about some capture. Failing that, an explicit subject
    decides the direction, then advice about an unplayed move, and only after
    those does the possessive get a say — "took *your* bishop" is Glitch
    taking the player's piece, "*my* bishop is gone" is the player taking his.
    With none of them, the text pins nothing ("a knight came off") and either
    side's record is enough, because the guard fails permissive on real
    ambiguity.
    """
    piece = match.group("piece") or match.group("gone_piece")
    # "horse" is a knight; the board only knows the one word for it.
    name = "knight" if piece.lower() == "horse" else piece.lower()
    victims = _victims_of_named_moves(match.string, facts)
    if victims is not None:
        # The sentence hangs its capture on a move, and the board knows what
        # that move takes. Outranks the subject and the record alike: it is
        # the only evidence about *this* capture rather than about some
        # capture, and it is the evidence the invented victim needs.
        return name in victims
    subject = (match.group("subject") or "").lower()
    if subject == "i":  # Glitch is the player's opponent
        return name in facts.captured_by_opponent
    if subject == "you":
        return name in facts.captured_by_player
    if _is_advice(match.string, facts):
        return True
    owner = (match.group("owner") or match.group("gone_owner") or "").lower()
    if owner == "your":  # the speaker is the opponent, so the piece was his to take
        return name in facts.captured_by_opponent
    if owner == "my":
        return name in facts.captured_by_player
    return name in (facts.captured_by_player | facts.captured_by_opponent)


def _names(claimed: str, sans: Iterable[str]) -> bool:
    """Whether `sans` holds the move named, check and mate marks aside."""
    bare = claimed.rstrip("+#")
    return any(san.rstrip("+#") == bare for san in sans)


def _move_happened(match: re.Match[str], facts: VerifiedFacts) -> bool:
    return _names(match.group(0), facts.moves)


def _owned_move_happened(match: re.Match[str], facts: VerifiedFacts) -> bool:
    """Whether the side the sentence credits is the side that played the move.

    With neither side's moves supplied there is no attribution to check against
    — an older call site, or a caller with no pipeline to assemble one — so the
    class falls back to the wider set and behaves exactly like the generic one.
    Guarding every credited move on the strength of absent evidence would cost
    far more than the miss does.
    """
    claimed = match.group("san")
    if not facts.moves_by_player and not facts.moves_by_opponent:
        return _names(claimed, facts.moves)
    if match.group("subject").lower() == "i":  # Glitch is the opponent
        return _names(claimed, facts.moves_by_opponent)
    return _names(claimed, facts.moves_by_player)


def _material_claimed(match: re.Match[str]) -> tuple[int | None, bool]:
    """What a material match asserts: how many pawns, and which way.

    The amount is `None` for "the exchange", which names no fixed number.
    """
    unit = match.group("unit") or match.group("unit_b")
    amount = None
    if unit is not None:
        count = (match.group("count") or match.group("count_b")).lower()
        amount = _MATERIAL_COUNTS.get(count, 0) or int(count)
        amount *= _MATERIAL_VALUES[unit.lower()]
    for name in ("direction", "direction_b", "direction_c"):
        if (direction := match.group(name)) is not None:
            return amount, direction.lower() in ("down", "behind")
    return amount, False  # unreachable; every branch names a direction


def _material_holds(amount: int | None, behind: bool, balance: int) -> bool:
    """Whether a claim of being `amount` pawns up (or down) fits `balance`.

    Direction is the fact and magnitude is checked to within a pawn. Material
    talk names the *nominal* trade — "up a knight" after winning a knight for a
    pawn, which the board counts as two — and that is how everybody who plays
    chess says it, so a class that called it a lie would be guarding wording.
    Being told you are ahead when you are behind is the invention that matters.
    """
    advantage = -balance if behind else balance
    if advantage <= 0:
        return False
    return amount is None or abs(advantage - amount) <= 1


def _material_matches(match: re.Match[str], facts: VerifiedFacts) -> bool:
    """Whether any board this turn held backs the count the text names.

    Any of them, because the narrator counted one board and the guard checks
    from another (see `VerifiedFacts.material`), and both were real.
    """
    amount, behind = _material_claimed(match)
    subject = (match.group("subject") or "").lower()

    def holds(balance: int) -> bool:
        if subject == "i":  # Glitch is the player's opponent: the count flips
            return _material_holds(amount, behind, -balance)
        if subject == "you":
            return _material_holds(amount, behind, balance)
        # Nobody named: "Up a knight. Cute." is genuinely ambiguous about whose
        # knight, so either reading verifying is enough — the guard fails
        # permissive on ambiguity, and a level board still backs neither.
        return _material_holds(amount, behind, balance) or _material_holds(
            amount, behind, -balance
        )

    return any(holds(balance) for balance in facts.material)


def _number_reported(match: re.Match[str], facts: VerifiedFacts) -> bool:
    number = match.group(0) if match.group("score") else None
    for name in ("score", "decimal", "centipawns", "mate"):
        if (value := match.group(name)) is not None:
            number = value
            break
    if number is None:  # unreachable; the pattern always fills one group
        return True
    # The sign is part of the fact — who is winning is exactly what it says —
    # so only a redundant leading "+" is negotiable.
    return number in facts.numbers or number.lstrip("+") in facts.numbers


# A takeback is not a capture, however much it sounds like one: "took your
# knight back" is undo talk, and the piece it names is going back *on* the
# board. Only the capture class needs to know that.
_CAPTURE_HEDGES = re.compile(
    _FUTURE.pattern
    + r"""
    | \b(?: undo | undid | undone | takeback | back )\b
    """,
    re.IGNORECASE | re.VERBOSE,
)


@dataclass(frozen=True)
class _ClaimClass:
    """One kind of operational claim: how to spot it, and what makes it true.

    `hedges` is per class rather than global: the ending class is pinned to
    the behavior the traces demanded of it and reads no tense at all (see
    `_FUTURE`), and the capture class has a wrinkle of its own. The shared
    `_HEDGES` apply to every class on top of these.
    """

    name: str
    pattern: re.Pattern[str]
    verified: Callable[[re.Match[str], VerifiedFacts], bool]
    hedges: re.Pattern[str] | None = _FUTURE


_CLAIM_CLASSES = (
    _ClaimClass("ending", _CLAIMS, lambda match, facts: facts.ended, hedges=None),
    _ClaimClass("draw", _DRAW, lambda match, facts: facts.drawn),
    _ClaimClass("capture", _CAPTURE, _capture_happened, hedges=_CAPTURE_HEDGES),
    _ClaimClass("check", _CHECK, lambda match, facts: facts.check),
    _ClaimClass("move", _SAN_CLAIM, _move_happened),
    _ClaimClass("owned_move", _OWNED_MOVE, _owned_move_happened),
    _ClaimClass("save", _SAVE, lambda match, facts: facts.saved),
    _ClaimClass("voice", _VOICE, _setting_is("voice")),
    _ClaimClass("difficulty", _DIFFICULTY, _setting_is("difficulty")),
    _ClaimClass("verbosity", _VERBOSITY, _setting_is("verbosity")),
    # Reads no tense, for the ending class's reason: a promise about how much
    # will be said from here *is* the claim that the setting moved, and the
    # only thing that can make it true is the call. "Want me to talk more?"
    # and "if you want more detail" are a question and a condition, which the
    # shared hedges take out already.
    _ClaimClass(
        "verbosity_change",
        _VERBOSITY_CHANGE,
        lambda match, facts: "verbosity" in facts.settings_changed,
        hedges=None,
    ),
    _ClaimClass("evaluation", _EVALUATION, _number_reported),
    _ClaimClass("material", _MATERIAL, _material_matches),
)


def unverified_claims(text: str, facts: VerifiedFacts) -> tuple[str, ...]:
    """The claim classes this commentary asserts that the facts don't support.

    Empty for commentary that claims nothing operational, which is most of it.
    Sentence by sentence and hedge by hedge, on the same bar the ending class
    set: an assertion in its own sentence, never a mention. Trash talk, threats,
    questions and hypotheticals are the whole point of the commentary, so a
    class that cannot tell them from a report does not belong here.

    Returned as an ordered list of class names rather than a bool so the
    pipeline can log *which* fact the model invented — that is the thing worth
    knowing when a guarded turn shows up in a trace.
    """
    found: list[str] = []
    for sentence in _SENTENCES.split(text):
        if _HEDGES.search(sentence):
            continue
        for claim in _CLAIM_CLASSES:
            if claim.name in found:
                continue
            if claim.hedges is not None and claim.hedges.search(sentence):
                continue
            if any(
                not claim.verified(match, facts)
                for match in claim.pattern.finditer(sentence)
            ):
                found.append(claim.name)
    return tuple(found)
