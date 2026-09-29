"""Conversation memory: what was said, and what the model phases are shown of it.

One user command + the final commentary the user actually saw = one turn. The
transcript stores final answers only — never thought blocks, never raw tool
payloads (BRIEF: final answers only). The full transcript is kept, so the whole
conversation survives a save/resume round trip.

The model phases are never handed it as chat turns (#372,
`docs/turn-memory.md`). Each reads a `Recall` instead, as data inside its own
prompt: the game's record (the ledger's, code's), the player's requests in
their own words, and the last exchange. Glitch's older lines are not in it —
an old assistant turn is personality, and a false one stayed in history as
fact — and nothing here is written by a model: code copies words and reads
none of them.

No board facts, settings or saves are copied here either. Those are injected
fresh into the state block every turn (`api._agent_state_dict`), and a second
copy would be an ageing one.

Roles are restricted to user/assistant: the system prompt is owned by the
brain's personality layer, so a save file can never smuggle one in.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

# How many prior turns `window` returns by default.
DEFAULT_WINDOW_TURNS = 20

_ROLES = ("user", "assistant")

# A command that is nothing but a move: how a board drag records itself, and how
# a typed "e4" arrives. That turn's content is already in the state block's
# `history`, so quoting it back is noise wearing a fact's clothes. SAN or UCI,
# whole string only.
_BARE_MOVE = re.compile(
    r"""\A\s*
    (?: O-O (?: -O )?
      | [KQRBN]? [a-h]? [1-8]? x? [a-h] [1-8] (?: = [QRBN] )?
      | [a-h] [1-8] [a-h] [1-8] [qrbn]?
    ) [+#]? [.!]?
    \s*\Z""",
    re.VERBOSE,
)


class Transcript:
    """An ordered log of conversation turns, stored as chat messages."""

    def __init__(self) -> None:
        self._messages: list[dict[str, str]] = []

    def record(self, user_text: str, assistant_text: str) -> None:
        """Log one completed turn: the command and the commentary shown."""
        self._messages.append({"role": "user", "content": user_text})
        self._messages.append({"role": "assistant", "content": assistant_text})

    def window(self, max_turns: int = DEFAULT_WINDOW_TURNS) -> list[dict[str, str]]:
        """The most recent `max_turns` turns as chat messages, oldest first.
        Turns are recorded atomically, so slicing by message pairs never
        splits a turn."""
        return [dict(m) for m in self._messages[-2 * max_turns :]]

    def requests(self) -> list[str]:
        """What the player has asked for across the whole conversation, the
        last exchange aside (`player_requests`)."""
        return player_requests(self._messages)

    def last_exchange(self) -> tuple[str, str] | None:
        """The latest turn, as said (`last_exchange`)."""
        return last_exchange(self._messages)

    def to_dict(self) -> list[dict[str, str]]:
        """Serialized form: the full message list (not windowed)."""
        return [dict(m) for m in self._messages]

    @classmethod
    def from_dict(cls, data: Any) -> "Transcript":
        """Rebuild from serialized form, validating shape and roles so a
        corrupted or tampered save file can never inject arbitrary prompt
        content under an unexpected role."""
        if not isinstance(data, list):
            raise ValueError("transcript must be a list of messages")
        transcript = cls()
        for message in data:
            if not isinstance(message, dict):
                raise ValueError(f"transcript entry is not a message: {message!r}")
            role, content = message.get("role"), message.get("content")
            if role not in _ROLES:
                raise ValueError(f"transcript message has invalid role: {role!r}")
            if not isinstance(content, str):
                raise ValueError("transcript message content must be a string")
            transcript._messages.append({"role": role, "content": content})
        return transcript


def _truncate(text: str, limit: int) -> str:
    """Cut `text` to `limit` characters on a word boundary. Whole words only:
    half a move phrase is worse than a shorter one."""
    if len(text) <= limit:
        return text
    head = text[: limit + 1]
    cut = head.rfind(" ")
    return f"{text[:cut].rstrip() if cut > 0 else text[:limit]}…"


# --- what replaces the chat history (#372) ------------------------------------------

# The player's requests, as the model phases will read them: each cut to this
# many characters, and the newest kept until this many characters in all.
REQUEST_CHARS = 160
REQUESTS_MAX_CHARS = 1500


def player_requests(
    messages: list[dict[str, str]],
    *,
    request_chars: int = REQUEST_CHARS,
    max_chars: int = REQUESTS_MAX_CHARS,
) -> list[str]:
    """The player's own words from every turn but the latest, oldest first:
    what they asked for, where a standing ask ("only knights from now on")
    lives. Code copies words and reads none — which requests still stand is
    the model's to work out. A turn that was only a move is left out (the
    move list has it), and so is the latest turn (`last_exchange` shows it
    whole). Past `max_chars` the oldest go, and the first line says how many,
    because a memory that quietly forgets reads like one that never heard.
    Glitch's words are never here."""
    users = [m["content"] for m in messages if m["role"] == "user"]
    if messages and messages[-1]["role"] == "assistant" and users:
        users = users[:-1]
    requests = [
        _truncate(collapsed, request_chars)
        for text in users
        if (collapsed := " ".join(text.split())) and not _BARE_MOVE.match(collapsed)
    ]
    kept: list[str] = []
    total = 0
    for request in reversed(requests):
        total += len(request)
        if total > max_chars and kept:
            break
        kept.append(request)
    kept.reverse()
    dropped = len(requests) - len(kept)
    return ([f"({dropped} earlier requests not listed)"] if dropped else []) + kept


def last_exchange(messages: list[dict[str, str]]) -> tuple[str, str] | None:
    """The latest completed turn, `(the player's words, the reply as
    remembered)`, or None before there is one. Kept whole because "the other
    one" and "undo that" point at it. The reply is what the transcript
    remembers: Glitch's words, or the app's deterministic line for a turn he
    said nothing on (`api.CommandOutcome.memory`), or "" for neither."""
    for index in range(len(messages) - 1, 0, -1):
        if (
            messages[index]["role"] == "assistant"
            and messages[index - 1]["role"] == "user"
        ):
            return messages[index - 1]["content"], messages[index]["content"]
    return None


# How much of the last exchange is shown: whole in practice, cut only when a
# pasted wall of text would crowd out the prompt.
LAST_EXCHANGE_CHARS = 600

# How each phase's prompt names the last reply (#372): his own words, and not a
# record — what the tools did is, and a line said last turn is never a fact to
# repeat. The narrator is Glitch, so to him it is what *he* said.
NARRATOR_REPLY_LABEL = "What you said then (your words, not a record)"
PLANNER_REPLY_LABEL = "Glitch said then (his words, not a record)"


@dataclass(frozen=True)
class Recall:
    """What came before this turn, as data for a model phase (#372), in the
    three parts code keeps: the game's record (`ledger.render_record`), the
    player's requests in their own words, and the last exchange. Each phase
    renders it for itself (`render`), because the last reply is Glitch's own
    words to the narrator and his words to the planner."""

    record: tuple[str, ...] = ()
    requests: tuple[str, ...] = ()
    last: tuple[str, str] | None = None

    @classmethod
    def of(cls, messages: list[dict[str, str]], record: Sequence[str]) -> "Recall":
        return cls(
            tuple(record), tuple(player_requests(messages)), last_exchange(messages)
        )

    def render(self, reply_label: str) -> str:
        """The three parts under their labels, empty ones left out; "" when
        there is nothing before this turn."""
        parts = []
        if self.record:
            parts.append(
                "The game's record, kept by the app (what happened earlier in "
                "this game besides the moves themselves):\n"
                + "\n".join(f"- {line}" for line in self.record)
            )
        if self.requests:
            parts.append(
                "What the player asked for earlier, in their own words, oldest "
                "first:\n"
                + "\n".join(
                    f"- {r}" if r.startswith("(") else f'- "{r}"' for r in self.requests
                )
            )
        if self.last is not None:
            said, reply = (
                _truncate(" ".join(t.split()), LAST_EXCHANGE_CHARS) for t in self.last
            )
            parts.append(
                f'The last exchange:\nThe player said: "{said}"\n'
                + (f'{reply_label}: "{reply}"' if reply else f"{reply_label}: nothing.")
            )
        return "\n\n".join(parts)
