"""Shared test doubles.

`ScriptedBrain` is the one canonical no-LLM `Brain` for the whole suite: it
stands in for a finished agent loop, dispatching its scripted tool calls for
real, so the *pipeline* can be exercised deterministically without a model.
`FakeEngine` is the canonical no-Stockfish engine double: it plays a scripted
reply and records every strength setting and MultiPV request, so tests can
pin that difficulty reaches the engine and that replies never take an
analysis detour. `ScriptedProvider` is the canonical no-LLM `ChatProvider`
double, one layer below `ScriptedBrain`: it returns scripted `ChatResult`s and
records the `chat()` requests, so the real `LlamaBrain` loop (tool messages,
the iteration and correction budgets, thinking toggles) can be exercised
without a model. `CountingProvider` is the odd one out — not a double at all
but a decorator: it wraps a *real* provider and records each round trip, so the
agent evals can assert how many times the live model was called and whether
thinking was on. Keep the doubles here — not copied into test files.
"""

import time
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

from chessapp.api import create_app
from chessapp.brain import AgentResponse, Narration
from chessapp.coordinator import TurnCoordinator
from chessapp.engine import Evaluation
from chessapp.provider import ChatResult, Usage
from chessapp.provider import ToolCall as ProviderToolCall
from chessapp.tools import build_registry

_DEAD_EVEN = Evaluation(score_cp=0, mate_in=None)


class FakeEngine:
    """Engine double: scripted reply + recorders for strength and MultiPV."""

    def __init__(
        self,
        reply_uci: str = "e7e5",
        best_moves: tuple = (),
        evaluation: Evaluation = _DEAD_EVEN,
    ):
        self.reply_uci = reply_uci
        self.best_moves = list(best_moves)
        self.evaluation = evaluation
        self.multipv_requests: list[int] = []
        self.skill_levels: list[int] = []
        self.elos: list[int] = []
        self.tiers: list[str] = []
        # A real engine holds a Stockfish process and the non-daemon thread
        # that drives it, so who closes it decides whether the process can
        # exit at all (`app.serve`). Recorded, not counted: closing twice is
        # harmless and closing never is the bug.
        self.closed = False

    def play_move(self, session):
        return session.submit_move(self.reply_uci)

    def choose_move(self, session):
        return self.reply_uci

    def get_best_moves(self, session, n=3):
        self.multipv_requests.append(n)
        return self.best_moves[:n]

    def evaluate_position(self, session):
        """One canned score, whatever the position. Enough for the tools that
        call it as a step (`analyze_last_move`) — the numbers themselves are
        Stockfish's business and are pinned in test_mistake_analysis.py."""
        return self.evaluation

    def close(self) -> None:
        self.closed = True

    def set_skill_level(self, level: int) -> None:
        self.skill_levels.append(level)

    def set_elo(self, elo: int) -> None:
        self.elos.append(elo)

    def set_tier(self, tier: str) -> None:
        self.tiers.append(tier)


class ScriptedBrain:
    """Scripted brain: a *finished* agent loop, canned.

    A real brain runs the loop itself — model turn, dispatch, feed the result
    back, answer in words — and hands the pipeline one `AgentResponse` holding
    everything that happened. `ScriptedBrain` skips the model half: each
    scripted `AgentResponse` says which tool calls the loop "decided on" and
    what it said afterwards, and `get_agent_response` runs those calls through
    the real `dispatcher` (the registry) to fill in `tool_results`. So api-level
    tests still exercise real dispatch and real state changes, without a model
    and without re-testing the loop (that lives in `test_llama_brain.py`).

    Under-scripting raises `IndexError` so a test that asks for more turns than
    it planned fails loudly instead of silently reusing stale output.

    `narrate` — the observation beat's commentary turn, the one commentary path
    outside the loop — pops the next scripted narration and records the new
    board plus what changed. When none is scripted it returns a placeholder, so
    tests that don't assert on commentary don't have to script one. A scripted
    `Exception` is raised instead of returned, which is how a provider failure
    during the observation beat is exercised (the beat is skippable; the engine's
    reply and the turn are not).
    """

    def __init__(
        self,
        *responses: AgentResponse,
        dispatcher=None,
        narrations: tuple[str | Narration | Exception, ...] = (),
    ) -> None:
        self._responses = list(responses)
        self._narrations = list(narrations)
        self.dispatcher = dispatcher
        self.calls: list[tuple[dict, str]] = []
        self.narrate_calls: list[tuple[dict, list]] = []
        self.transcripts: list[list] = []
        self.narrate_transcripts: list[list] = []

    def get_agent_response(
        self, board_state: dict, command: str, transcript=()
    ) -> AgentResponse:
        self.calls.append((board_state, command))
        self.transcripts.append(list(transcript))
        scripted = self._responses.pop(0)
        if not scripted.tool_calls or self.dispatcher is None:
            return scripted
        results = tuple(
            {
                "name": call.name,
                "result": self.dispatcher.dispatch(call.name, call.args),
            }
            for call in scripted.tool_calls
        )
        return replace(scripted, tool_results=results)

    def narrate(self, board_state: dict, changes: list, transcript=()) -> Narration:
        self.narrate_calls.append((board_state, changes))
        self.narrate_transcripts.append(list(transcript))
        scripted = self._narrations.pop(0) if self._narrations else "(commentary)"
        if isinstance(scripted, Exception):
            raise scripted
        # A bare string is the common case (a test that only cares about the
        # words); a full `Narration` lets a test script the cost fields too.
        return scripted if isinstance(scripted, Narration) else Narration(text=scripted)


def scripted_app(ctx, *responses: AgentResponse, brain=None, **create_kwargs):
    """`create_app` with a `ScriptedBrain` that dispatches through the app's own
    registry — the one wiring every api-level test needs.

    The brain and the app share a single registry — and one turn coordinator
    behind it — exactly as app assembly does, so a scripted tool call really runs
    against the real `ToolContext`, advances the real turn machine, and its
    result is real. `atomic_exchange=False` mirrors assembly too: a scripted
    `make_move` applies the player's move and leaves the coordinator mid-turn for
    the pipeline to run its beats and collect the engine's reply.
    Returns `(app, brain)`.
    """
    coordinator = TurnCoordinator(ctx)
    registry = build_registry(ctx, coordinator, atomic_exchange=False)
    if brain is None:
        brain = ScriptedBrain(*responses, dispatcher=registry)
    elif getattr(brain, "dispatcher", None) is None:
        brain.dispatcher = registry
    return (
        create_app(
            ctx,
            brain=brain,
            registry=registry,
            coordinator=coordinator,
            **create_kwargs,
        ),
        brain,
    )


def receive_state(ws) -> dict:
    """The next *state* document off the websocket, skipping progress frames.

    One channel carries two kinds of message (`api.StateBroadcaster`): the
    authoritative board document, and the live progress of the turn that is
    changing it. A test that is about the board says so by asking for the board;
    what it must never do is silently accept a progress frame as one, which is
    what a bare `receive_json` would now do. Blocks until a state arrives, so a
    test asserting a broadcast *happened* still fails (by hanging out to the
    suite's own limits) when it did not.
    """
    while True:
        message = ws.receive_json()
        if message["type"] == "state":
            return message


def text_turn(
    content: str | None,
    *,
    finish_reason: str = "stop",
    usage: Usage | None = None,
) -> ChatResult:
    """A plain-text `ChatResult` turn (no tool calls). `usage` scripts the token
    counts a real llama-server would return, for tests that assert on cost."""
    return ChatResult(
        content=content, tool_calls=[], finish_reason=finish_reason, usage=usage
    )


def tool_calls_turn(
    *calls: tuple[str, dict[str, Any]],
    content: str | None = None,
    usage: Usage | None = None,
) -> ChatResult:
    """A `ChatResult` turn carrying tool calls, args already parsed to dicts
    (the shape the provider hands the brain). Each `call` is `(name, args)`.
    `usage` scripts the token counts, for tests that assert on cost."""
    return ChatResult(
        content=content,
        tool_calls=[
            ProviderToolCall(id=f"call_{index}", name=name, arguments=args)
            for index, (name, args) in enumerate(calls)
        ],
        finish_reason="tool_calls",
        usage=usage,
    )


class ScriptedProvider:
    """`ChatProvider` double: records each `chat()` request, returns scripted
    turns.

    Given one turn it returns it for every call; given several it returns them
    in order and repeats the last — so a retry loop can be scripted "bad then
    good" (and a single bad turn repeated to exhaustion). A turn may be an
    `Exception` to raise instead of return (e.g. `ToolCallArgumentsError`),
    modelling a provider-level failure the brain must recover from.
    """

    def __init__(self, *turns: ChatResult | Exception) -> None:
        self._turns = list(turns)
        self.calls: list[dict[str, Any]] = []

    def chat(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        tools: Sequence[dict[str, Any]] | None = None,
        enable_thinking: bool = False,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> ChatResult:
        self.calls.append(
            {
                "messages": list(messages),
                "tools": list(tools) if tools is not None else None,
                "enable_thinking": enable_thinking,
                # Recorded, not resolved: the planner asks for its own
                # temperature and the narrator asks for none, and which of the
                # two made a call is exactly what the split's tests assert on.
                "temperature": temperature,
                # Same for the generation cap: which phase's ceiling a call
                # carried is the runaway-thought fix's whole assertion.
                "max_tokens": max_tokens,
            }
        )
        turn = self._turns[min(len(self.calls) - 1, len(self._turns) - 1)]
        if isinstance(turn, Exception):
            raise turn
        return turn


@dataclass(frozen=True)
class ModelCall:
    """One model round trip as the harness sees it: was thinking on, how long,
    and what it cost in tokens.

    The token counts are `None` when the round trip did not report any — it
    raised, or the provider returned no usage — never 0. An unmeasured call is
    not a free one, the same rule the phase splits follow for an unmeasured
    phase (`evalstats.TurnLatencies`).
    """

    thinking: bool
    seconds: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class CountingProvider:
    """`ChatProvider` decorator that counts and times model round trips.

    Unlike `ScriptedProvider`, which *replaces* the wire, this one *observes*
    it: it delegates every `chat()` to a real inner provider and records what
    went over. The agent evals wrap the live `LlamaCppProvider` in one, because
    `ChatProvider` is the only seam every round trip passes through and nothing
    in production counts them — that is how the evals can assert a fast-path
    move costs zero model calls, a tool-using utterance costs the planner's tool
    turn plus its handoff note plus the narrator's reply, and thinking is off
    until an analysis result lands.

    A raising round trip is still recorded: the model was called, and the loop
    pays a correction for it.

    It records **usage per call** for the same reason it records latency per
    call: the trace carries the turn's token totals, and a total cannot say
    whether a slow narration wrote more tokens or wrote them slower — which is
    the mechanism question left open by the repeat-stop finding
    (`docs/agent-evals.md`). Nothing in `src/` needs to change to answer it,
    because every round trip already passes through here.
    """

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.calls: list[ModelCall] = []

    def reset(self) -> None:
        """Drop the record — one eval scenario measures one utterance."""
        self.calls.clear()

    def chat(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        tools: Sequence[dict[str, Any]] | None = None,
        enable_thinking: bool = False,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> ChatResult:
        started = time.monotonic()
        # Bound before the call so the `finally` can read it after a raise, where
        # it stays None: a round trip that died reported no usage, and the loop
        # still paid for whatever it generated first.
        result: ChatResult | None = None
        try:
            result = self.inner.chat(
                messages,
                tools=tools,
                enable_thinking=enable_thinking,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return result
        finally:
            usage = result.usage if result is not None else None
            self.calls.append(
                ModelCall(
                    thinking=enable_thinking,
                    seconds=time.monotonic() - started,
                    prompt_tokens=usage.prompt_tokens if usage else None,
                    completion_tokens=usage.completion_tokens if usage else None,
                )
            )
