"""The swappable-brain seam.

Everything model-specific lives behind `Brain.get_agent_response`; nothing
outside a Brain implementation may know which model or backend answers.

The brain runs the agent loop — it calls the model, feeds each tool result
back, and keeps going until the model answers in words — but it never
*executes* anything itself: every call it decides on goes out through a
`ToolDispatcher` (the validated `ToolRegistry`), which is what makes it
impossible for a brain to corrupt game state. The loop is bounded, and the
stop reason says how it ended (`completed | max_iterations |
correction_limit`, the fleet's vocabulary — `../agent-standard/STANDARD.md`
§3).
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class ToolCall:
    """One tool invocation the brain wanted: a name and the args it passed."""

    name: str
    args: dict[str, Any]


class ToolDispatcher(Protocol):
    """Whatever executes a named tool call and answers with a result dict.

    `ToolRegistry` (`tools.py`) satisfies this structurally — the brain is
    handed one at assembly and never imports the tool layer. It never raises
    on an agent-caused fault: a bad call comes back as error *data* the model
    can read and correct from.
    """

    def dispatch(self, name: str, args: Any) -> dict[str, Any]: ...


@dataclass(frozen=True)
class AgentResponse:
    """One finished agent run.

    `text` is the user-facing commentary — the model's first turn that asked
    for no tools (empty when the loop stopped on a budget instead). Every call
    the loop made and ran is in `tool_calls`, with `tool_results` holding each
    one's `{"name", "result"}` in the *same order*: the two are parallel by
    construction, and the delegate wire zips them strictly.
    """

    text: str
    tool_calls: tuple[ToolCall, ...] = ()
    tool_results: tuple[dict[str, Any], ...] = ()
    stop_reason: str = "completed"
    # The run's cost at the provider boundary: how many times the model was
    # called and the tokens summed across those calls. Plain ints — the seam
    # stays model-agnostic, so the provider's `Usage` type never crosses it.
    model_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass(frozen=True)
class Narration:
    """One commentary turn made outside the loop (the fast path's stand-in for
    the closing turn). `text` is what the player sees; the cost fields mirror
    `AgentResponse`'s so a narrated turn reaches the trace with the same
    accounting a looped one does. It is always exactly one model call."""

    text: str
    model_calls: int = 1
    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass
class _RunState:
    """The loop's accumulator: what has been called, what came back, what it cost."""

    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_results: list[dict[str, Any]] = field(default_factory=list)
    model_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def record(self, name: str, args: dict[str, Any], result: dict[str, Any]) -> None:
        self.tool_calls.append(ToolCall(name=name, args=args))
        self.tool_results.append({"name": name, "result": result})

    def count_call(self, prompt_tokens: int = 0, completion_tokens: int = 0) -> None:
        """Tally one model round trip and its tokens. Called for every trip —
        including one that raised before returning a result (it still cost a
        call), which lands here with zeros."""
        self.model_calls += 1
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens

    def response(self, text: str, stop_reason: str) -> AgentResponse:
        return AgentResponse(
            text=text,
            tool_calls=tuple(self.tool_calls),
            tool_results=tuple(self.tool_results),
            stop_reason=stop_reason,
            model_calls=self.model_calls,
            prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens,
        )


class Brain(Protocol):
    def get_agent_response(
        self,
        board_state: dict[str, Any],
        command: str,
        transcript: Sequence[dict[str, str]] = (),
    ) -> AgentResponse:
        """Run the agent loop for one utterance: turn it into tool calls, run
        them through the dispatcher, feed the results back, and stop on the
        first turn that asks for no tools — that turn *is* the commentary, and
        because it is offered no tools it cannot act on the utterance a second
        time. `board_state` is the agent-facing view (fen, turn, player_color,
        in_check, SAN history, captured, legal_moves, game_over/outcome — not
        the UI state document), captured before the loop runs; the loop reads
        every later state change from the tool results themselves.
        `transcript` is the prior conversation as chat messages (a `Transcript`
        window: user commands + the commentary the user actually saw, final
        answers only) so the agent can follow references to earlier turns."""
        ...

    def narrate(
        self,
        board_state: dict[str, Any],
        changes: list[dict[str, Any]],
        transcript: Sequence[dict[str, str]] = (),
    ) -> Narration:
        """Commentary on a move the loop did not make: the deterministic fast
        path (`parse_move` → `make_move`) skips the model entirely, so there is
        no tool-less turn to take the commentary from. This is that turn on its
        own — the *new* board plus `changes` (each a `{"name", "result"}` tool
        result), no tools offered, no access to the raw utterance. It is the
        only commentary path outside the loop, and it exists because the fast
        path is deliberately outside the loop too; at verbosity=low even this
        is skipped for a canned confirmation, making a plain move zero-LLM."""
        ...
