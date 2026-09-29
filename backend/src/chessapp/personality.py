"""The agent's prompts: Glitch's voice, and the planner's contract.

Two model phases, two prompts (`docs/planner-narrator.md`). `system_prompt_for`
is the **narrator's**: the full personality, used for the turn that speaks to
the player and is offered no tools. `PLANNER_PROMPT` is the **planner's**: the
compact, persona-free contract the bounded tool loop runs under, because on
a 12B a page of tone competes with the tool decision for attention. Everything
below the planner section is the narrator's prompt, layered as follows.

The personality *is* a system prompt, and there is exactly one — Glitch
(decided 2026-07: the selectable eight-personality roster was collapsed into
one dialed-in character). The prompt is composed in layers per
`agent-standard/STANDARD.md` §5:

1. `_BASE` — chess's own app base prompt: the narrator's contract, which is a
   speaking contract (#289). The narrator holds no tools and acts on nothing;
   by the time it speaks, what was done is settled and handed to it as a
   record. So the rules are about what it may *say*: it is never the referee,
   a move made at the player's request is the player's, only what the record
   shows done may be said to be done, and a choice the player owes is asked
   with the options named.
2. The global Glitch personality — a vendored, verbatim copy of
   `agent-standard/personality-global.md` (`personality-global.md` next to
   this module). The house has one character shared by every app agent; fix
   drift by re-copying (`agent-standard/check-sync.sh`), never by editing the
   copy in place.
3. `_CHESS_FLAVOR` — chess-specific tone on top of the global character (the
   competitive trolling contract). Flavor is tone only; personality never
   shapes move choice, difficulty, or any other setting.
"""

from pathlib import Path

# --- the planner's prompt -----------------------------------------------------
#
# A short contract (#370): turn the player's words into tool calls; the tools
# enforce the rules and their results say how to fix a bad call; when the words
# are unclear, ask. What each tool does, and which requests trigger it, lives in
# that tool's own description, and every rule a tool can enforce lives in the
# tool: legality in `make_move`'s refusal (which says whether a corrected call
# can still be the player's move), confirmation in `_gate`, a pick by position
# in `source`, the moves a question offers in `ask_player`, which works them
# out from the parts the planner names (#371).
#
# The contract it replaced was a page of rules, and some fought each other: it
# said "never decide whether a move is legal", then that a move matching no
# `legal_moves` entry "is not legal … the answer is to say so". Live
# (2026-09-26 capture) the planner judged "knight to f6" illegal in prose and
# no tool ran. Now the planner submits what was asked and `make_move` answers.
#
# What survived, and why — each line is one a screen showed the 12B needs
# (paired gate campaigns vs main, 2026-09-27; per-arm history in
# docs/agent-evals.md and docs/knight-ask-campaign.md):
# - "match against `legal_moves`: two or more fit → `ask_player` with every
#   entry that fits (#289: an ask left to the note named none of the moves);
#   exactly one fits → submit it". The match step is main's, and it is what
#   reads "knight to sea three" as Nc3 (replayed payload: 20/20 with it, 0/20
#   asked without). Asking first matters: "exactly one fits, submit" as the
#   opening clause played O-O on a bare "castle" with both sides legal 19/20;
# - "grab that pawn" and what `captures` says. Without them, "take the pawn"
#   on move 1 read as a pawn *push* and was asked about (0/20 → 18/20 with);
# - "nothing fits … is not a question … never ask which piece". The lean
#   contract without it answered "take the pawn" with "Which pawn, bro?" 20/20:
#   with no square in the words there is nothing to submit, so the planner
#   fell back on "call nothing and say what to ask". Where the words do name a
#   move, it is submitted and `make_move` refuses it — the "knight to f6" fix;
# - "several things → a tool for each, in their order". Without it "undo the
#   bishop move and undo the knight move, then play d4" undid once 11/20;
# - the closing note, which is what a turn with no tool call hands on.
PLANNER_PROMPT = """\
You turn a chess player's words into tool calls. The words are free-form and
often transcribed speech. You never speak to the player.

- Call the tools that do what the player asked. Each tool says what it does
  and the kinds of requests it answers.
- The tools enforce the rules, not you. Never judge whether a move is legal:
  submit the move the player asked for. Map loose phrasing ("grab that pawn")
  onto the `legal_moves` entry it names — `captures` says what each capturing
  move takes — and `make_move` says if it cannot be played.
- Nothing fits — a move no piece can make, a capture when nothing can be
  taken — is not a question: submit the move if their words name one, and
  otherwise say it cannot be made. Never ask which piece they meant.
- When they ask for several things, call a tool for each, in their order.
- Match their words against `legal_moves`. Two or more fit: do not guess —
  call `ask_player` with every entry that fits. Exactly one fits: submit it. \
When you cannot tell what they want at all, call
  nothing and reply with one short line saying what to ask.
- Omit optional arguments the player's words did not supply.
- A result that failed says how to fix it: `retry: different_args` means
  correct the call and repeat; `never` means stop.

When the work is done, or no tool is needed, reply with one short factual
line: what happened, or what the player should be asked or told. A separate
voice phrases what the player sees.
"""


# --- the summarizer's prompt (#372) --------------------------------------------

# The story of the game: a running record both phases will read in place of
# the quoted chat history (docs/story-and-ledger.md). Written from the ledger
# and the tool results, which code computed; Glitch's lines ride along labelled
# as his words and are never a source of facts, or one wrong line would become
# permanent canon. Neutral and third person so nothing in it is a voice to
# imitate. The word cap is the prompt's; `story.STORY_MAX_TOKENS` is the hard
# one the call is held to.
SUMMARIZER_PROMPT = """\
You keep the story of a chess game between a player and Glitch, the voice of
the chess engine the player is playing against. You are given the story so far
and the turns since it was written. Reply with the new story, in exactly this
shape and nothing else:

Standing requests: every request or preference the player stated that should
still guide later turns, in their own words, or "none".
Open: a question left unanswered, moves offered or suggested, or a choice the
player was asked to make, that a next remark could point back to, or "none".
The game so far: at most 120 words, in the past tense.

- Third person, plainly: "the player", "Glitch". Never write as Glitch and
  never address anyone. Do not quote Glitch unless the player may refer back
  to what he said.
- Facts come only from the "What happened" and "Tools" lines. What Glitch said
  is only what he said: if it is not in those lines, it did not happen.
- Name moves as the lines do ("12. Nf3", "12... e5"). Takebacks, setting
  changes, draw offers and results are events too.
- Sum up: fold old stretches of moves into one sentence ("after a quiet
  opening, Glitch won the a3 knight"), keep the last turn or two in detail, and
  drop what no later remark will need. Do not describe the current position.
"""


# --- the narrator's prompt ----------------------------------------------------

# It used to be the whole agent's contract — "you read the position and change
# the game only through your tools", "never claim to have done something you
# did not actually do with a tool" — from before the planner/narrator split,
# when one prompt both acted and spoke. The phase that reads it has had no
# tools since the split, and on a 12B every line that does not apply is one
# more thing to answer instead of the player (astra audit F9, #289). What it
# does is speak from a record the harness wrote, so that is what it is told.
_BASE = """\
You are the player's opponent in a chess game, and the voice they talk to. The
player talks to you in free-form text — often transcribed speech. By the time
you reply, whatever was done about it is settled, and you are handed the
record of it: your job is to say it.

Rules you must never break:
- You are not the referee. The board and engine own the truth: you never decide
  whether a move is legal, and you never track the position in your head.
- You are the player's opponent. A move carried out at the player's request is
  the player's move, not yours, and anything it captured was yours, taken off
  you — never claim it, or its capture, as something you did.
- Describe only what the record shows. Never say something was done that the
  record does not show done, and never invent a move, capture, or threat that
  is not on the board.
- When the player has to choose, ask them one short question, naming the
  options you were given.
"""

# Global layer — the vendored house personality (STANDARD.md §5). The body is
# canonical and must never be edited in place; re-vendor to change Glitch.
_PERSONALITY_PATH = Path(__file__).with_name("personality-global.md")


def _load_global_personality() -> str:
    """The vendored Glitch text, minus its one leading ``<!-- vendored -->`` line."""
    lines = _PERSONALITY_PATH.read_text(encoding="utf-8").splitlines()
    body = [line for line in lines if not line.startswith("<!-- vendored")]
    return "\n".join(body).strip()


_GLOBAL_PERSONALITY = _load_global_personality()

# App-flavor layer — chess-specific tone on top of the global character. Only
# the competitive/trolling contract lives here; generic tone (brevity, the
# slang whitelist, swearing permission, "help is always real") is now the
# global layer's job.
_CHESS_FLAVOR = """
You rarely troll — mostly you just play — but when the board earns it you drop
one dry, understated line ("Interesting." "Bold.") and move on, never piling on.
When the player genuinely gets you — a real move, material, a win — give them
props for a beat, then allow yourself one salty-but-obvious line of cope.
When you do help, a small jab on the way in is fine, but the wit rides on top of
real competence: never troll the player into worse chess, it never replaces it.
"""

SYSTEM_PROMPT = _BASE + "\n" + _GLOBAL_PERSONALITY + "\n" + _CHESS_FLAVOR

# "Talk more / talk less": verbosity layers an output-length instruction on
# top of the personality. `normal` adds nothing — the base prompt already
# says "keep your replies short".
_VERBOSITY_INSTRUCTIONS: dict[str, str] = {
    "low": (
        "\nThe player asked you to talk less: reply in one short sentence at "
        "most, no elaboration, unless they ask a direct question.\n"
    ),
    "normal": "",
    "high": (
        "\nThe player asked you to talk more: be chattier — add a remark "
        "about the position, their play, or the game so far when you reply.\n"
    ),
}


def system_prompt_for(verbosity: str = "normal") -> str:
    """The system prompt at `verbosity`.

    An unknown verbosity adds no extra instruction: the setting tool is
    enum-guarded so this shouldn't happen, but the lookup must never leave
    the agent without a valid prompt.

    Hints take no layer here (the mode was retired 2026-09-01): a hint exists
    only as the answer to an ask — the planner routes it to `get_best_moves` —
    so there is no state in which Glitch is told to volunteer one.
    """
    return SYSTEM_PROMPT + _VERBOSITY_INSTRUCTIONS.get(verbosity, "")
