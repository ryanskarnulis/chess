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
    ) -> AgentResponse: ...
