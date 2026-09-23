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

Telling an answer from a clarification is language, so it is not attempted
here: both are `reply`, a turn that called no tool. That distinction is the
model's to declare (`docs/planner-narrator.md`, "The handoff").

`narrator_result_view` is the second half: one projection of a tool result for
every narrator brief. `undo`, `new_game` and `resume_game` answer with `fen`
and `turn` — right for the planner, which decides from the board — and a
narrator handed `turn` beside `player_color` treats its reaction as a
move-selection beat (#193). The fast path's state view already withheld those
keys (`api._narrator_state_dict`); the results did not, on either route.

Pure: no session, no I/O. Everything here is decided from its arguments.
"""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

# What the narrator is never handed, in a state view or in a tool result: every
# spelling of "it is your move". `turn` and `fen` name the side to move (the
# FEN string does it in its second field), and `legal_moves` / `captures` are
# the menu to pick from. `api._narrator_state_dict` deletes the same four, and
# a test pins that the two lists agree.
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

# How the loop ending on its own (`no_progress`) reads here: the planner never
# declared itself done, so a turn that also changed something is `partial`.
_UNFINISHED_STOPS = frozenset({"no_progress", "length"})

Kind = Literal["completed", "partial", "declined", "reply"]


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

    `reply_owed` is whether the player's move is still waiting on the engine's
    answer as the narrator speaks: the reply is being computed and the app
    announces it afterwards, so the narrator must not. `facts` is the fresh,
    narrator-safe board view (no side to move, no history). `note` is the
    planner's closing line, kept because a `reply` turn has nothing else to
    answer from — and labelled as a reading, never a record.
    """

    kind: Kind
    performed: tuple[Entry, ...] = ()
    refused: tuple[Entry, ...] = ()
    consulted: tuple[Entry, ...] = ()
    reply_owed: bool = False
    facts: Mapping[str, Any] = field(default_factory=dict)
    note: str = ""

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
    """
    performed: list[Entry] = []
    refused: list[Entry] = []
    consulted: list[Entry] = []
    for ref, entry in enumerate(tool_results, start=1):
        name, result = entry["name"], entry["result"]
        if _refused(result):
            refused.append(Entry(ref, name, _refusal_reason(result)))
        elif name in READ_TOOLS:
            consulted.append(Entry(ref, name))
        else:
            performed.append(Entry(ref, name))
    kind: Kind
    if not tool_results:
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
        facts=dict(facts or {}),
        note=note,
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


def _refs(entries: Sequence[Entry]) -> str:
    return ", ".join(f"#{entry.ref} {entry.tool}" for entry in entries)


def render(
    handoff: Handoff, command: str, tool_results: Sequence[Mapping[str, Any]]
) -> str:
    """The narrator's brief for a turn the planner just finished.

    The results come first and carry ids, and the sorted lines point at them:
    "Done this turn" is the record of what changed, and when nothing did it
    says so in those words. The planner's note comes last and is labelled as
    what it is. The closing instruction speaks from the record and the facts;
    the note is context for understanding the ask, not a source of claims.
    """
    parts = [f"The player said:\n{command}"]
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
    if handoff.reply_owed:
        record.append(
            "The engine has not played its reply to the player's move yet; "
            "the app announces it after you speak."
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
