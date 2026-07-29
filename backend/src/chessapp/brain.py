"""The swappable-brain seam.

Everything model-specific lives behind `Brain.get_agent_response`; nothing
outside a Brain implementation may know which model or backend answers.

The brain runs the agent loop — it calls the model, feeds each tool result
back, and keeps going until the model stops asking for tools — but it never
*executes* anything itself: every call it decides on goes out through a
`ToolDispatcher` (the validated `ToolRegistry`), which is what makes it
impossible for a brain to corrupt game state. The loop is bounded, and the
stop reason says how it ended (`completed | max_iterations |
correction_limit`, the fleet's vocabulary — `../agent-standard/STANDARD.md`
§3 — plus two chess additions: `provider_error` when the provider died
mid-turn (the response still carries every tool result that verifiably ran,
so the pipeline can close the turn and tell the truth instead of catching an
exception after the board changed, and `provider_failure` names *which*
death it was so a caller can tell one worth retrying from one that is not),
and `no_progress` when the loop ended a planning phase that had started
repeating itself. The two are opposites for the player: a budget stop or a
dead provider produces no commentary, while `no_progress` reaches the
narrator like `completed` does — results came back, so there is something
verified to speak from).

How the words get written is the implementation's business, and
`LlamaBrain`'s answer is a second, tool-free model phase
(`docs/planner-narrator.md`). This seam only promises that `text` is what the
player may be shown and that it was produced from verified results.
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

    `text` is the user-facing commentary — spoken by the narrator phase from
    the turn's verified results (empty when the loop stopped on a budget
    instead, in which case no narrator ran and the pipeline substitutes its
    stuck reply). Every call the loop made and ran is in `tool_calls`, with
    `tool_results` holding each
    one's `{"name", "result"}` in the *same order*: the two are parallel by
    construction, and the delegate wire zips them strictly.
    """

    text: str
    tool_calls: tuple[ToolCall, ...] = ()
    tool_results: tuple[dict[str, Any], ...] = ()
    stop_reason: str = "completed"
    # Only meaningful beside `stop_reason == "provider_error"`, and empty
    # otherwise: *which* kind of provider failure ended the turn. A dead socket
    # and a refused request stop a turn identically but want opposite handling —
    # one is worth asking again, the other is worth asking about — and the
    # implementation used to swallow the distinction with the exception. A
    # plain string for the same reason the token counts are plain ints: the
    # seam stays model-agnostic, so the provider's enum never crosses it (the
    # values are `provider.ProviderFailure`'s).
    provider_failure: str = ""
    # The run's cost at the provider boundary: how many times the model was
    # called and the tokens summed across those calls. Plain ints — the seam
    # stays model-agnostic, so the provider's `Usage` type never crosses it.
    model_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    # One wall-clock reading per model call, in call order — parallel to
    # `model_calls` by construction, including the calls that raised (a round
    # trip that died still spent the time). Empty from a brain that doesn't
    # measure: an unmeasured turn is not a fast one, so it records no readings
    # rather than zeros.
    model_latencies_ms: tuple[int, ...] = ()


@dataclass(frozen=True)
class Narration:
    """One narrator turn: commentary on work already done, with no tools on
    offer. `text` is what the player sees; the cost fields mirror
    `AgentResponse`'s so a narrated turn reaches the trace with the same
    accounting a looped one does. It is always exactly one model call."""

    text: str
    model_calls: int = 1
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: int = 0

    @property
    def model_latencies_ms(self) -> tuple[int, ...]:
        """The one call's latency, under the name `AgentResponse` uses — so
        whatever reads a turn's cost reads both shapes the same way."""
        return (self.latency_ms,)


@dataclass
class _RunState:
    """The loop's accumulator: what has been called, what came back, what it cost."""

    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_results: list[dict[str, Any]] = field(default_factory=list)
    model_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latencies_ms: list[int] = field(default_factory=list)

    def record(self, name: str, args: dict[str, Any], result: dict[str, Any]) -> None:
        self.tool_calls.append(ToolCall(name=name, args=args))
        self.tool_results.append({"name": name, "result": result})

    def count_call(
        self,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        latency_ms: int = 0,
    ) -> None:
        """Tally one model round trip, its tokens and its wall clock. Called for
        every trip — including one that raised before returning a result (it
        still cost a call, and usually the most time), which lands here with no
        tokens but a real latency."""
        self.model_calls += 1
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.latencies_ms.append(latency_ms)

    def response(
        self, text: str, stop_reason: str, provider_failure: str = ""
    ) -> AgentResponse:
        return AgentResponse(
            text=text,
            tool_calls=tuple(self.tool_calls),
            tool_results=tuple(self.tool_results),
            stop_reason=stop_reason,
            provider_failure=provider_failure,
            model_calls=self.model_calls,
            prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens,
            model_latencies_ms=tuple(self.latencies_ms),
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
        first turn that asks for no tools. The commentary is then written from
        the turn's results by a phase that holds no tools, so it cannot act on
        the utterance a second time. `board_state` is the agent-facing view
        (fen, turn, player_color,
        in_check, SAN history, captured, legal_moves, game_over/outcome — not
        the UI state document), captured before the loop runs; the loop reads
        every later state change from the tool results themselves.
        `transcript` is the prior conversation as chat messages (final answers
        only) so the agent can follow references to earlier turns. How far back
        it reaches and in what form is the app's memory policy, not the brain's:
        what actually arrives is `Transcript.memory()` — the last few turns
        verbatim behind a digest of the older asks (`docs/turn-memory.md`) — and
        a brain neither knows nor needs to know which of them were condensed."""
        ...

    def narrate(
        self,
        board_state: dict[str, Any],
        changes: list[dict[str, Any]],
        transcript: Sequence[dict[str, str]] = (),
    ) -> Narration:
        """Commentary on a move the loop did not make: the deterministic fast
        path (`parse_move` → `make_move`) skips the planner entirely, so there
        is no turn for the narrator to close. This is the narrator phase on its
        own — the *new* board plus `changes` (each a `{"name", "result"}` tool
        result), no tools offered, no access to the raw utterance. What the
        board view contains is the caller's policy, and the app deliberately
        hands a view with no side to play for — no turn, no legal moves, no
        FEN (`api._narrator_state_dict`): the beat runs while the engine's
        reply is still being computed, and a narrator that can see whose move
        it is announces one. It exists
        because the fast path is deliberately outside the loop; at verbosity=low
        even this is skipped for a canned confirmation, making a plain move
        zero-LLM."""
        ...
