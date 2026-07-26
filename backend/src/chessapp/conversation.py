"""Conversation memory: what the agent remembers being said.

One user command + the final commentary the user actually saw = one turn. The
transcript stores final answers only — never thought blocks, never raw tool
payloads (BRIEF: final answers only). The full transcript is kept in memory so
the whole conversation survives a save/resume round trip; only the model's view
is reduced.

Two views, because they answer different questions:

- `window()` is the **raw record**: the most recent turns exactly as they were
  said. Serialization, tests, and anything asking "what was actually said" reads
  this.
- `memory()` is the **model's view** (`docs/turn-memory.md`): the last
  `RECENT_TURNS` turns verbatim, behind a deterministic digest of what the
  player asked for in the turns before them. Reference-following ("do the second
  one") only reaches back a turn or two, so the recent turns stay untouched;
  everything older collapses to the player's own words, and Glitch's side of it
  is dropped. That prose is the noise the digest exists to remove — an older
  assistant turn is personality, and personality competing with the tool
  decision is this project's measured failure mode (self-poisoning, trace review
  2026-07-13).

The digest carries **no board facts, no settings, no saves**. Those are injected
fresh into the state block every turn (`api._agent_state_dict`), and a summary
restating them would be a second, ageing copy of a fact the app holds — exactly
the bug that injection cured. What survives is what genuinely lives only in the
conversation: the player's standing asks, in their own words. No model writes
it; copying words needs no model, and an unguarded summary the player never sees
is the worst place to let one invent something.

Roles are restricted to user/assistant: the system prompt is owned by the
brain's personality layer, so a save file can never smuggle one in.
"""

import re
from typing import Any

# How many prior turns the raw window holds. ~20 turns keeps banter continuity
# without letting a long game grow the record unboundedly.
DEFAULT_WINDOW_TURNS = 20

# How many of those the model sees verbatim, and how much older history the
# digest in front of them may quote. The digest is one line per older request,
# so this pair is the whole prompt-size story: a turn's memory can never exceed
# a digest plus RECENT_TURNS turns, however long the game runs.
RECENT_TURNS = 4
DIGEST_MAX_REQUESTS = 12
DIGEST_REQUEST_CHARS = 100

_ROLES = ("user", "assistant")

# The digest's header. It names its own limits: the model is told, in the same
# breath, that the state block — not this — is where board truth lives.
_DIGEST_HEADER = (
    "Earlier in this conversation, condensed to what the player asked for. "
    "The board, the settings and the saved games are supplied fresh with every "
    "command — never take them from here."
)
# The digest rides as a user message; this keeps the user/assistant alternation
# the chat template expects. Deliberately inert — it becomes model context.
_DIGEST_ACK = "Noted."

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

    def memory(self, max_turns: int = DEFAULT_WINDOW_TURNS) -> list[dict[str, str]]:
        """What a brain is given: `condense` over the raw window. One bound in
        one place — the window caps how far back the digest may look, the digest
        caps how much of that reaches the prompt."""
        return condense(self.window(max_turns))

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


def condense(
    messages: list[dict[str, str]], *, recent_turns: int = RECENT_TURNS
) -> list[dict[str, str]]:
    """The model's view of a conversation: recent turns verbatim, older ones
    reduced to the player's requests.

    Pure and deterministic — the same messages always condense the same way, so
    this is unit-testable without a provider and cannot drift under sampling.
    The synthetic pair is built here and never recorded, so `Transcript`'s role
    whitelist stays the only thing that decides what a save file may contain.
    """
    split = len(messages) - 2 * recent_turns
    if split <= 0:
        return [dict(m) for m in messages]
    older, recent = messages[:split], [dict(m) for m in messages[split:]]

    requests = [
        collapsed
        for message in older
        if message["role"] == "user"
        and (collapsed := " ".join(message["content"].split()))
        and not _BARE_MOVE.match(collapsed)
    ]
    if not requests:
        # Those turns held nothing the state block doesn't already carry (a
        # stretch of board drags, say). Don't spend tokens saying so.
        return recent

    # The ack exists only to keep user/assistant alternating. When the recent
    # slice already opens on an assistant turn — the delegate store can drop a
    # contentless message and leave one there — it would be the thing that
    # breaks alternation, so it is left out.
    ack = [] if recent and recent[0]["role"] == "assistant" else [_ack()]
    return [{"role": "user", "content": _digest(requests)}, *ack, *recent]


def _ack() -> dict[str, str]:
    return {"role": "assistant", "content": _DIGEST_ACK}


def _digest(requests: list[str]) -> str:
    """The digest text: the header, then the newest `DIGEST_MAX_REQUESTS`
    requests, then — when older ones were dropped — how many. The count is
    explicit because a memory that quietly forgets reads like one that never
    heard."""
    kept = requests[-DIGEST_MAX_REQUESTS:]
    dropped = len(requests) - len(kept)
    lines = [_DIGEST_HEADER]
    lines += [f'- "{_truncate(request)}"' for request in kept]
    if dropped:
        lines.append(f"(+{dropped} earlier requests not listed)")
    return "\n".join(lines)


def _truncate(text: str, limit: int = DIGEST_REQUEST_CHARS) -> str:
    """Cut a request to `limit` characters on a word boundary. Whole words only:
    half a move phrase is worse than a shorter one."""
    if len(text) <= limit:
        return text
    head = text[: limit + 1]
    cut = head.rfind(" ")
    return f"{text[:cut].rstrip() if cut > 0 else text[:limit]}…"
