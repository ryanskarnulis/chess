"""Shared test doubles.

`ScriptedBrain` is the one canonical no-LLM `Brain` for the whole suite: it
pops canned responses in order and records what it was shown, so the agent
loop can be exercised deterministically without ever reaching a live model.
`FakeEngine` is the canonical no-Stockfish engine double: it plays a scripted
reply and records every strength setting and MultiPV request, so tests can
pin that difficulty reaches the engine and that replies never take an
analysis detour. `ScriptedProvider` is the canonical no-LLM `ChatProvider`
double, one layer below `ScriptedBrain`: it returns scripted `ChatResult`s and
records the `chat()` requests, so the real `LlamaBrain` orchestration (schema
validation, correction retries, thinking toggles) can be exercised without a
model. Keep the doubles here — not copied into test files.
"""

from collections.abc import Sequence
from typing import Any

from chessapp.brain import AgentResponse
from chessapp.provider import ChatResult
from chessapp.provider import ToolCall as ProviderToolCall


class FakeEngine:
    """Engine double: scripted reply + recorders for strength and MultiPV."""

    def __init__(self, reply_uci: str = "e7e5", best_moves: tuple = ()):
        self.reply_uci = reply_uci
        self.best_moves = list(best_moves)
        self.multipv_requests: list[int] = []
        self.skill_levels: list[int] = []
        self.elos: list[int] = []
        self.tiers: list[str] = []

    def play_move(self, session):
        return session.submit_move(self.reply_uci)

    def choose_move(self, session):
        return self.reply_uci

    def get_best_moves(self, session, n=3):
        self.multipv_requests.append(n)
        return self.best_moves[:n]

    def set_skill_level(self, level: int) -> None:
        self.skill_levels.append(level)

    def set_elo(self, elo: int) -> None:
        self.elos.append(elo)

    def set_tier(self, tier: str) -> None:
        self.tiers.append(tier)


class ScriptedBrain:
    """Scripted brain: pops canned responses in order, records prompts.

    Phase one (`get_agent_response`) pops the next scripted `AgentResponse`;
    under-scripting raises `IndexError` so a test that asks for more turns than
    it planned fails loudly instead of silently reusing stale output.

    Phase two (`react`) — the game loop's react-from-new-state step — pops the
    next scripted reaction text and records the new board plus what changed.
    When no reaction is scripted it returns a placeholder, so tests that don't
    assert on reaction text don't have to script one.
    """

    def __init__(
        self, *responses: AgentResponse, reactions: tuple[str, ...] = ()
    ) -> None:
        self._responses = list(responses)
        self._reactions = list(reactions)
        self.calls: list[tuple[dict, str]] = []
        self.react_calls: list[tuple[dict, list]] = []
        self.transcripts: list[list] = []
        self.react_transcripts: list[list] = []

    def get_agent_response(
        self, board_state: dict, command: str, transcript=()
    ) -> AgentResponse:
        self.calls.append((board_state, command))
        self.transcripts.append(list(transcript))
        return self._responses.pop(0)

    def react(self, board_state: dict, changes: list, transcript=()) -> str:
        self.react_calls.append((board_state, changes))
        self.react_transcripts.append(list(transcript))
        return self._reactions.pop(0) if self._reactions else "(reaction)"


def text_turn(content: str | None, *, finish_reason: str = "stop") -> ChatResult:
    """A plain-text `ChatResult` turn (no tool calls)."""
    return ChatResult(
        content=content, tool_calls=[], finish_reason=finish_reason, usage=None
    )


def tool_calls_turn(
    *calls: tuple[str, dict[str, Any]], content: str | None = None
) -> ChatResult:
    """A `ChatResult` turn carrying tool calls, args already parsed to dicts
    (the shape the provider hands the brain). Each `call` is `(name, args)`."""
    return ChatResult(
        content=content,
        tool_calls=[
            ProviderToolCall(id=f"call_{index}", name=name, arguments=args)
            for index, (name, args) in enumerate(calls)
        ],
        finish_reason="tool_calls",
        usage=None,
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
    ) -> ChatResult:
        self.calls.append(
            {
                "messages": list(messages),
                "tools": list(tools) if tools is not None else None,
                "enable_thinking": enable_thinking,
            }
        )
        turn = self._turns[min(len(self.calls) - 1, len(self._turns) - 1)]
        if isinstance(turn, Exception):
            raise turn
        return turn
