"""The open clarification: a question the player was asked and has not answered.

`ask_player` (#289) validates the moves a player's words fit and hands them to
the narrator, who asks. Until #319 that was the end of them: the next turn had
to recover *which* moves were on offer from the narrator's wording in the
transcript — paraphrased, dropped from the verbatim window four turns later
(`conversation.condense` keeps the player's words, not Glitch's), and bound to
no position and no conversation. An aside, a long chat, another client's move
or a second delegate thread could each leave the planner without the question,
or holding one about a board that is gone.

So the harness keeps the question, the way it keeps an armed destructive op
(`tools.PendingOp`): who was asked (`origin`), about which game (`game_id`)
and which board (`board_version`), what they said, and the board-validated
`candidates`. The same two bindings for the same reason — a choice is an
answer to a position, asked in a conversation, and neither half can be trusted
to still hold when the answer arrives.

**Code owns whether the question is still open; the model owns what the
player meant.** Nothing here reads the player's words, and nothing here ever
plays a candidate: the answer is the planner's ordinary `make_move`, checked
against the live board like any other. What this decides is only whether a
question still stands (`staleness`) and, once the board moves, what became of
it (`settle`).

The lifecycle (`docs/planner-narrator.md`, "The open question"):

- **asked** — a turn in some origin ended on a landed `ask_player`. One record
  per origin; asking again replaces it (`superseded`, `asked_again`).
- **open** — while the game and the board are the ones it was asked about. A
  turn that changes neither (a read, a setting, chat) leaves it standing,
  however many come between: a question about a position nobody has touched
  is still the decision in front of the player.
- **answered** — the asking origin's next board change was a move among the
  candidates. The record goes with it, so it can be answered once.
- **superseded** — that origin changed the board some other way: a move
  outside the candidates, an undo, a reset.
- **invalidated** — the board or the game changed under it from anywhere
  else (another client, a button, a resume). Found at the next read, like
  `ToolContext.live_pending`'s stale op, and reported once as expired.

Not persisted: a restart ends the record the way it ends an armed op
(`tools.live_checkpoint`), and bumps the board version past it besides.

Pure: no session, no I/O, like `handoff`.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal
from uuid import uuid4

from chessapp.handoff import READ_TOOLS

# What became of a question that is no longer open.
ANSWERED = "answered"
SUPERSEDED = "superseded"
INVALIDATED = "invalidated"

Status = Literal["answered", "superseded", "invalidated"]

# Why, in the record's own words.
GAME_CHANGED = "game_changed"
BOARD_CHANGED = "board_changed"
ASKED_AGAIN = "asked_again"
OTHER_MOVE = "other_move"
TURN_CHANGED_BOARD = "turn_changed_board"


@dataclass(frozen=True)
class Clarification:
    """One open question. Frozen: a question is never edited, only replaced
    or closed."""

    id: str
    origin: str
    game_id: str
    board_version: int
    request: str
    candidates: tuple[str, ...]

    def trace(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "origin": self.origin,
            "game_id": self.game_id,
            "board_version": self.board_version,
            "candidates": list(self.candidates),
        }


@dataclass(frozen=True)
class Closed:
    """A question that stopped being open, and why. `move` is the move that
    closed it, when a move did."""

    record: Clarification
    status: Status
    reason: str
    move: str = ""

    def trace(self) -> dict[str, Any]:
        return {
            "id": self.record.id,
            "status": self.status,
            "reason": self.reason,
            "move": self.move,
            "candidates": list(self.record.candidates),
        }


def ask(
    *,
    origin: str,
    game_id: str,
    board_version: int,
    request: str,
    candidates: Sequence[str],
) -> Clarification:
    """A new open question. `candidates` are the handoff's — validated by
    `ask_player` against the board — and `board_version` is the board at the
    *end* of the asking turn, which is the one the player hears the question
    over (`ToolContext.restamp_pending`'s reason)."""
    return Clarification(
        id=uuid4().hex[:12],
        origin=origin,
        game_id=game_id,
        board_version=board_version,
        request=request,
        candidates=tuple(dict.fromkeys(candidates)),
    )


def staleness(record: Clarification, *, game_id: str, board_version: int) -> str:
    """Why `record` no longer stands on this board, or "" when it does. The
    game first: a new game or a resume moves the version too, and "a
    different game" is the more exact thing to say."""
    if record.game_id != game_id:
        return GAME_CHANGED
    if record.board_version != board_version:
        return BOARD_CHANGED
    return ""


def settle(record: Clarification, tool_results: Sequence[Mapping[str, Any]]) -> Closed:
    """What became of `record` on a turn *in its own origin* that changed the
    board. Called only then — a turn that moved nothing leaves it open.

    Answered when the turn's first landed board change is a move among the
    candidates — the first, because a candidate played after an undo is a
    move on a different position, not an answer to the question. Any other
    first change (a different move, an undo, a reset) supersedes it. The move
    is read off the results, never off the player's words.
    """
    for entry in tool_results:
        name = entry.get("name")
        result = entry.get("result") or {}
        # A refused call moved nothing, and neither did a declined draw offer.
        if (
            name in _BOARD_NEUTRAL
            or result.get("ok") is False
            or result.get("accepted") is False
        ):
            continue
        if name != "make_move":
            return Closed(record, SUPERSEDED, TURN_CHANGED_BOARD)
        if result.get("legal") is not True or not result.get("san"):
            continue
        san = str(result["san"])
        if san in record.candidates:
            return Closed(record, ANSWERED, "", san)
        return Closed(record, SUPERSEDED, OTHER_MOVE, san)
    return Closed(record, SUPERSEDED, TURN_CHANGED_BOARD)


# The tools that land without moving the board: the reads, the settings, a
# save, the ask itself. Named as the neutral ones, the way `handoff.READ_TOOLS`
# is, so a tool added later without a classification counts as a board change
# — which supersedes a question rather than answering it. A test pins that
# every registered tool is on one side or the other.
_BOARD_NEUTRAL = READ_TOOLS | {
    "ask_player",
    "save_game",
    "set_difficulty",
    "set_verbosity",
    "set_voice_output",
}
