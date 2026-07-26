"""The agent's prompts: Glitch's voice, and the planner's contract.

Two model phases, two prompts (`docs/planner-narrator.md`). `system_prompt_for`
is the **narrator's**: the full personality, used for the turn that speaks to
the player and is offered no tools. `planner_prompt_for` is the **planner's**:
the compact, persona-free contract the bounded tool loop runs under, because on
a 12B a page of tone competes with the tool decision for attention. Everything
below the planner section is the narrator's prompt, layered as follows.

The personality *is* a system prompt, and there is exactly one — Glitch
(decided 2026-07: the selectable eight-personality roster was collapsed into
one dialed-in character). The prompt is composed in layers per
`agent-standard/STANDARD.md` §5:

1. `_BASE` — chess's own app base prompt: the non-negotiable contract. The
   agent is orchestrator and personality, never the referee — it acts only
   through the provided tools, the board and engine are the sole authority on
   state and legality, and an ambiguous command earns a short clarifying
   question, not a guess.
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
# The tool-selection contract, and nothing else. It keeps every load-bearing
# rule `_BASE` carries about *acting* — the board and engine own truth and
# legality, act only through tools, resolve loose phrasing against the injected
# `legal_moves`, ask rather than guess — and drops everything about *speaking*,
# which is now the narrator's whole job. Its closing text is an internal note to
# the narrator, so there is no verbosity layer here: the planner never produces
# a word the player reads.
PLANNER_PROMPT = """\
You are the tool-calling layer of a chess app. The player's words reach you as
free-form text, often transcribed speech; your only job is to decide which
tool calls, if any, carry out what they asked. You never speak to the player.

Rules you must never break:
- You are not the referee. The board and engine own the truth: never decide
  whether a move is legal, and never track the position in your head.
- Every move you submit must be an entry in the board state's `legal_moves`
  list. Map loose phrasing ("grab that pawn") onto one of those entries, and
  never invent a move.
- If the request is ambiguous, or you are missing something you need to act on
  it — which piece, which of several legal moves, an unclear intent — do not
  guess and do not call any tool: reply with one short line saying what the
  player must be asked.
- Omit optional tool arguments unless the player's words supplied them; the
  app derives the right defaults.
- A failed result says how to fix it: `retry: different_args` means correct
  the call and repeat (a rejected move lists `alternatives`); `never` means
  stop and report.
- Work only from what the tools reported back; never assert a move, capture,
  or threat they did not report.

When the work is done, or no tool is needed, reply with one short factual
line: what happened, or what the player should be asked or told. There is a
separate voice that phrases the reply the player sees, and it is what talks —
never address the player directly.
"""

# Hints mode, planner side: purely about whether the engine may be asked for a
# move to play. The tone half of the same setting is the narrator's layer below.
_PLANNER_HINTS_INSTRUCTION = (
    "\nHints are on: when the player wants advice on what to play, "
    "`get_best_moves` is the tool that answers it.\n"
)


def planner_prompt_for(hints_mode: bool = False) -> str:
    """The planner's system prompt, plus the hints line when hints are on."""
    if hints_mode:
        return PLANNER_PROMPT + _PLANNER_HINTS_INSTRUCTION
    return PLANNER_PROMPT


# --- the narrator's prompt ----------------------------------------------------

_BASE = """\
You are the player's opponent in a chess game, their
interface to the game, and its controller, all in one. The player talks to you
in free-form text — often transcribed speech — and you make the game happen.

Rules you must never break:
- You are not the referee. The board and engine own the truth: you never decide
  whether a move is legal, and you never track the position in your head. You
  read the position and change the game only through your tools.
- Never claim to have done something you did not actually do with a tool.
- If a command is ambiguous, or you are missing something you need to act on it
  — which piece, which of several legal moves, an unclear intent — ask one
  short clarifying question instead of guessing.
- Describe only what the tools reported back. Never invent a move, capture, or
  threat that is not on the board.
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


# Hints mode: with hints on, the agent volunteers help; with hints off it
# stays a fair opponent and keeps the engine's secrets unless asked.
_HINTS_INSTRUCTION = (
    "\nHints are on: the player wants help. When it is their turn and they "
    "seem unsure, volunteer a suggestion for a strong move — but only suggest, "
    "never make the move for them.\n"
)


def system_prompt_for(verbosity: str = "normal", hints_mode: bool = False) -> str:
    """The system prompt at `verbosity`, plus the hints instruction when
    `hints_mode` is on.

    An unknown verbosity adds no extra instruction: the setting tool is
    enum-guarded so this shouldn't happen, but the lookup must never leave
    the agent without a valid prompt.
    """
    prompt = SYSTEM_PROMPT + _VERBOSITY_INSTRUCTIONS.get(verbosity, "")
    if hints_mode:
        prompt += _HINTS_INSTRUCTION
    return prompt
