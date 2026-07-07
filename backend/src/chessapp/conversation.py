"""Conversation memory: what the agent remembers being said.

One user command + the final commentary the user actually saw = one turn.
The transcript stores final answers only — never thought blocks, never raw
tool payloads (BRIEF: final answers only) — and brains read it through
`window()`, which caps the turns sent so prompt/KV-cache growth stays
bounded. The full transcript is kept in memory so the whole conversation
survives a save/resume round trip; only the model's view is windowed.

Roles are restricted to user/assistant: the system prompt is owned by the
brain's personality layer, so a save file can never smuggle one in.
"""

from typing import Any

# How many prior turns a brain sees. ~20 turns keeps banter continuity
# without letting a long game grow the prompt unboundedly.
DEFAULT_WINDOW_TURNS = 20

_ROLES = ("user", "assistant")


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
