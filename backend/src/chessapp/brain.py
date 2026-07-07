"""The swappable-brain seam.

Everything model-specific lives behind `Brain.get_agent_response`; nothing
outside a Brain implementation may know which model or backend answers.
The brain only *names* tools — every call it returns goes through the
validated `ToolRegistry`, so no brain can corrupt game state.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class ToolCall:
    """One tool invocation the brain wants: a name and raw (unvalidated) args."""

    name: str
    args: dict[str, Any]


@dataclass(frozen=True)
class AgentResponse:
    """What the brain says (commentary) and wants done (tool calls, in order)."""

    text: str
    tool_calls: tuple[ToolCall, ...] = ()


class Brain(Protocol):
    def get_agent_response(
        self,
        board_state: dict[str, Any],
        command: str,
        transcript: Sequence[dict[str, str]] = (),
    ) -> AgentResponse:
        """Phase one of the loop: turn the user's utterance into tool calls
        (and/or a direct reply when nothing needs doing). `transcript` is the
        prior conversation as chat messages (a `Transcript` window: user
        commands + the commentary the user actually saw, final answers only)
        so the agent can follow references to earlier turns."""
        ...

    def react(
        self,
        board_state: dict[str, Any],
        changes: list[dict[str, Any]],
        transcript: Sequence[dict[str, str]] = (),
    ) -> str:
        """Phase two: after the tool calls have run, produce the user-facing
        commentary from the *new* game state and `changes` (each a
        ``{"name", "result"}`` tool result). Deliberately given no access to
        the raw utterance — a future deterministic fast-parse path can drive
        phase one without an LLM while the reaction still reads live state.
        `transcript` covers prior turns only, never the in-flight one."""
        ...
