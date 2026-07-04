"""The swappable-brain seam.

Everything model-specific lives behind `Brain.get_agent_response`; nothing
outside a Brain implementation may know which model or backend answers.
The brain only *names* tools — every call it returns goes through the
validated `ToolRegistry`, so no brain can corrupt game state.
"""

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
        self, board_state: dict[str, Any], command: str
    ) -> AgentResponse:
        """Phase one of the loop: turn the user's utterance into tool calls
        (and/or a direct reply when nothing needs doing)."""
        ...

    def react(self, board_state: dict[str, Any], changes: list[dict[str, Any]]) -> str:
        """Phase two: after the tool calls have run, produce the user-facing
        commentary from the *new* game state and `changes` (each a
        ``{"name", "result"}`` tool result). Deliberately given no access to
        the raw utterance — a future deterministic fast-parse path can drive
        phase one without an LLM while the reaction still reads live state."""
        ...
