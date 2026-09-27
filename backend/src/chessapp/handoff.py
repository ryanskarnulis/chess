"""The typed handoff: what the narrator is told a turn did, built by code.

The narrator closes a brain-routed turn, and until #289 it closed from two
things: the raw tool results and the planner's free-form note, with the brief
asking it to reply "based only on those results and that note". The results
are a record; the note is not. On a turn with no tool calls the note was the
*only* thing the narrator had, and nothing said that nothing had been done —
so a planner note reading "undid your last move" could come back from the
narrator as "Done, taken back." with no undo anywhere in the turn.

So the harness now says what happened, and the planner's note is demoted to
what it is: the planner's reading of what the player wants. `build` sorts the
turn's results into what was done (`performed`), what was refused
(`refused`) and what was only looked up (`consulted`), and derives the turn's
`kind` from those and the loop's stop reason alone — never from the note. The
narrator reads the sorted record with an explicit "Done this turn: nothing."
when nothing was, and the facts it may state arrive fresh from the app
(`facts`), not reconstructed from whichever results happened to carry a board.

Telling an answer from a clarification is language, so the harness does not
guess at it: the planner declares a clarification by calling `ask_player`
with the legal moves the player's words fit, and that turn is `clarify` with
those `candidates`. A turn that called nothing at all is `reply`.

`narrator_result_view` is the second half: one projection of a tool result for
every narrator brief. `undo`, `new_game` and `resume_game` answer with `fen`
and `turn` — right for the planner, which decides from the board — and a
narrator handed `turn` beside `player_color` treats its reaction as a
move-selection beat (#193). The narrator's state view already withheld those
keys (`api.narrator_facts`); the results did not, on either route.

Pure: no session, no I/O. Everything here is decided from its arguments.
"""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

# What the narrator is never handed, in a state view or in a tool result: every
# spelling of "it is your move". `turn` and `fen` name the side to move (the
# FEN string does it in its second field), and `legal_moves` / `captures` are
# the menu to pick from. `api.narrator_facts` carries none of the four, and
# a test pins that.
NARRATOR_HIDDEN_KEYS = ("fen", "turn", "legal_moves", "captures")

# The tools that only read. Anything else that answers `ok: true` changed
# something — the board, a setting, a save, an offer made — and is `performed`.
# Named as the reads rather than the writes so a tool added later without a
# classification fails towards "something was done", which a guard can check,
# rather than towards "nothing was", which a narrator would then repeat. A test
# pins that every registered tool is on one side or the other.
READ_TOOLS = frozenset(
    {
        "get_board_state",
        "get_legal_moves",
        "get_move_history",
        "get_captured_pieces",
        "describe_position",
        "evaluate_position",
        "get_best_moves",
        "analyze_last_move",
        "review_game",
        "export_pgn",
    }
)

# How the loop ending on its own reads here — a stall (`no_progress`) or a
# budget (#288): the planner never declared itself done, so a turn that also
# changed something is `partial`.
_UNFINISHED_STOPS = frozenset(
    {"no_progress", "length", "budget", "max_iterations", "correction_limit"}
)

Kind = Literal["completed", "partial", "declined", "reply", "clarify"]

# The planner's clarification (`tools.ASK_PLAYER`). Named here as a string, as
# the reads are, because this module imports nothing from the tool layer.
_ASK = "ask_player"


@dataclass(frozen=True)
class Entry:
    """One tool result, sorted: `ref` is its 1-based position in the turn's
    results (the `#n` the brief prints beside it), `tool` its name, and
    `reason` — refusals only — the refusal's own words."""

    ref: int
    tool: str
    reason: str = ""


@dataclass(frozen=True)
class Handoff:
    """What the turn did, as the harness read it off the results.

    `engine_reply` is the move the engine just played for the narrator in
    answer to the player's — `{"san", "capture", "check"}`, or None — and the
    narrator's to announce (#365): it speaks after the reply is on the board,
    and the player learns the move from what it says. `reply_owed` is whether
    the player's move is still waiting on that answer as the narrator speaks,
    which since #365 means the engine died on it. `facts` is the fresh,
    narrator-safe board view (no side to move, no history). `note` is the
    planner's closing line, kept because a `reply` turn has nothing else to
    answer from — and labelled as a reading, never a record.
    """

    kind: Kind
    performed: tuple[Entry, ...] = ()
    refused: tuple[Entry, ...] = ()
    consulted: tuple[Entry, ...] = ()
    reply_owed: bool = False
    engine_reply: Mapping[str, Any] | None = None
    facts: Mapping[str, Any] = field(default_factory=dict)
    note: str = ""
    # The legal moves the player must choose between, on a `clarify` turn.
    candidates: tuple[str, ...] = ()

    def trace(self) -> dict[str, Any]:
        """The handoff as the turn record keeps it: enough to re-judge a
        narration against what it was told, and no copy of the results (the
        record already holds them)."""
        return {
            "kind": self.kind,
            "performed": [entry.tool for entry in self.performed],
            "refused": [entry.tool for entry in self.refused],
            "consulted": [entry.tool for entry in self.consulted],
            "reply_owed": self.reply_owed,
            "engine_reply": self.engine_reply.get("san") if self.engine_reply else None,
            "candidates": list(self.candidates),
        }


def _refusal_reason(result: Mapping[str, Any]) -> str:
    return str(result.get("error") or result.get("reason") or "refused")


def _refused(result: Mapping[str, Any]) -> bool:
    """A refusal is `ok` not true, or a move the board rejected — which the
    move tool reports as `ok: true, legal: false`, because an illegal move is
    data and not a fault."""
    return result.get("ok") is not True or result.get("legal") is False


def build(
    tool_results: Sequence[Mapping[str, Any]],
    stop_reason: str = "completed",
    *,
    note: str = "",
    reply_owed: bool = False,
    engine_reply: Mapping[str, Any] | None = None,
    facts: Mapping[str, Any] | None = None,
) -> Handoff:
    """Sort a turn's results and derive its kind from them.

    - `reply`: no tool was called — the turn is an answer or a question.
    - `declined`: something was refused and nothing was done.
    - `partial`: something was done, and something was refused or the loop
      ended the phase itself (`no_progress`) before the planner said it was
      finished.
    - `completed`: every call that was made landed. A turn that only looked
      things up is `completed` too — it did everything it set out to.
    - `clarify`: the planner asked the player to choose (`ask_player`
      landed). Calls made before the ask stand and are sorted as usual; the
      loop answers every call after it unrun (#314), so they read as refused.
      The candidates are the tool's, validated against the board, never the
      note's.
    """
    performed: list[Entry] = []
    refused: list[Entry] = []
    consulted: list[Entry] = []
    candidates: list[str] = []
    for ref, entry in enumerate(tool_results, start=1):
        name, result = entry["name"], entry["result"]
        if name == _ASK and not _refused(result):
            candidates.extend(result.get("candidates", ()))
        elif _refused(result):
            refused.append(Entry(ref, name, _refusal_reason(result)))
        elif name in READ_TOOLS:
            consulted.append(Entry(ref, name))
        else:
            performed.append(Entry(ref, name))
    kind: Kind
    if candidates:
        kind = "clarify"
    elif not tool_results:
        kind = "reply"
    elif not performed:
        kind = "declined" if refused else "completed"
    elif refused or stop_reason in _UNFINISHED_STOPS:
        kind = "partial"
    else:
        kind = "completed"
    return Handoff(
        kind=kind,
        performed=tuple(performed),
        refused=tuple(refused),
        consulted=tuple(consulted),
        reply_owed=reply_owed,
        engine_reply=dict(engine_reply) if engine_reply else None,
        facts=dict(facts or {}),
        note=note,
        candidates=tuple(dict.fromkeys(candidates)),
    )


def narrator_result_view(entry: Mapping[str, Any]) -> dict[str, Any]:
    """One `{"name", "result"}` tool result as a narrator may read it: the
    same result minus `NARRATOR_HIDDEN_KEYS`. Everything else a result says —
    the move, what it took, what was undone, the engine's reply to a restore —
    is what the narrator speaks from, and stays."""
    result = {
        key: value
        for key, value in entry["result"].items()
        if key not in NARRATOR_HIDDEN_KEYS
    }
    return {"name": entry["name"], "result": result}


def _reply_words(reply: Mapping[str, Any]) -> str:
    """The engine's reply as the brief states it: "Nf6, taking their knight,
    check" — the move, then what it did, from the facts alone."""
    words = [str(reply.get("san"))]
    if reply.get("capture"):
        words.append(f"taking their {reply['capture']}")
    if reply.get("check"):
        words.append("check")
    return ", ".join(words)


def _refs(entries: Sequence[Entry]) -> str:
    return ", ".join(f"#{entry.ref} {entry.tool}" for entry in entries)


def render(
    handoff: Handoff, command: str, tool_results: Sequence[Mapping[str, Any]]
) -> str:
    """The narrator's brief, for every turn (#369): one the planner just
    finished, and one the loop never ran (a fast-path move, a board drag, a
    confirmed op), which has no note and may have no words.

    The results come first and carry ids, and the sorted lines point at them:
    "Done this turn" is the record of what changed, and when nothing did it
    says so in those words. The planner's note comes last and is labelled as
    what it is. The closing instruction speaks from the record and the facts;
    the note is context for understanding the ask, not a source of claims.
    """
    # A board drag has no words. Said outright, because "Done this turn" with
    # no ask above it reads to a 12B as "I did" — live, Glitch once narrated
    # the player's capture as his own (#193).
    parts = [
        f"The player said:\n{command}"
        if command
        else "The player acted on the board without saying anything: "
        "what was done this turn, the player did."
    ]
    if tool_results:
        listed = "\n".join(
            f"#{ref} {json.dumps(narrator_result_view(entry))}"
            for ref, entry in enumerate(tool_results, start=1)
        )
        parts.append(f"What the tools reported this turn:\n{listed}")
    else:
        parts.append("No tool was called this turn.")
    record = [
        "Done this turn: "
        + (_refs(handoff.performed) + "." if handoff.performed else "nothing.")
    ]
    if handoff.refused:
        record.append(
            "Refused: "
            + "; ".join(
                f"#{entry.ref} {entry.tool} ({entry.reason})"
                for entry in handoff.refused
            )
            + "."
        )
    if handoff.consulted:
        record.append(f"Looked up: {_refs(handoff.consulted)}.")
    if handoff.candidates:
        record.append(
            "The player has to choose between: "
            + ", ".join(handoff.candidates)
            + ". Ask them which one they mean, naming each."
        )
    if handoff.engine_reply:
        record.append(
            f"Your reply, already on the board: {_reply_words(handoff.engine_reply)}. "
            "The player learns your move only from what you say; say it "
            "however you like."
        )
    elif handoff.reply_owed:
        record.append(
            "Your reply to the player's move never came: the engine failed. "
            "The app tells the player so after you speak."
        )
    parts.append("\n".join(record))
    if handoff.facts:
        parts.append(f"The game now:\n{json.dumps(dict(handoff.facts))}")
    if handoff.note:
        parts.append(
            "The planner's reading of what the player wants (not a record of "
            f"what happened):\n{handoff.note}"
        )
    parts.append(
        "Reply to the player in character. Say only what the record above "
        "shows was done; if it shows nothing done, do not say anything was. "
        "When the player has to choose, ask them, naming the options."
    )
    return "\n\n".join(parts)
