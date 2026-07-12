"""llama-server brain: the `Brain` implementation over a `ChatProvider`.

The brain owns orchestration — prompt assembly, the thinking-toggle policy,
schema validation, and the self-correction retry loop — and delegates the wire
to a `ChatProvider` (`provider.py`), which speaks the OpenAI chat API to
llama-server over plain httpx. This module is the only place that knows the
model is Gemma-4 behind llama.cpp; everything else sees the `Brain` protocol.

Model-specific quirks, split across the two layers:

- Gemma emits its chain-of-thought in a separate `reasoning_content` field.
  The provider drops it, so `ChatResult.content` is final answers only and
  thought blocks never leak into commentary or back into history (BRIEF).
- Thinking is toggled per request via the provider's `enable_thinking` flag.
  Thinking is OFF by default for fast move parsing; callers flip it ON for
  analysis (`react` to analysis-tool results).

Under quantization the model can occasionally emit a malformed tool call: args
that aren't valid JSON (the provider raises `ToolCallArgumentsError`), an
unknown tool, or args that violate the schema (caught here). The `ToolRegistry`
is the ultimate guard — it turns bad calls into error data, never a crash — but
a bad call there just wastes a turn. So the brain first validates each call
and, on failure, feeds the error back and retries a bounded number of times,
giving the model a chance to self-correct. If it never does, the invalid calls
are dropped.
"""

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import jsonschema

from chessapp.brain import AgentResponse, ToolCall
from chessapp.personality import system_prompt_for
from chessapp.provider import (
    ChatProvider,
    ChatResult,
    LlamaCppProvider,
    ToolCallArgumentsError,
)
from chessapp.provider import ToolCall as ProviderToolCall

_DEFAULT_MAX_RETRIES = 2

# Reacting to these tools' results is analysis work: thinking goes ON
# (BRIEF: thinking OFF for fast move parsing/reactions, ON for analysis).
_ANALYSIS_TOOLS = frozenset(
    {"evaluate_position", "get_best_moves", "analyze_last_move"}
)


@dataclass
class LlamaBrain:
    """A `Brain` backed by a `ChatProvider` (llama-server behind llama-swap).

    The provider is injected so tests exercise the mapping without a live LLM;
    `create_llama_brain` builds the real one. `tool_definitions` are the
    registry's OpenAI-style schemas — the single source of truth for what the
    agent may call, and what tool calls are validated against.
    """

    provider: ChatProvider
    tool_definitions: list[dict[str, Any]]
    system_prompt: str | Callable[[], str]
    enable_thinking: bool = False
    max_retries: int = _DEFAULT_MAX_RETRIES

    def _resolve_system_prompt(self) -> str:
        """The system prompt for this request. A callable is re-resolved every
        call, so a live settings change (verbosity/hints mutating what the
        provider reads) takes effect on the next command; a plain string is a
        fixed prompt."""
        prompt = self.system_prompt
        return prompt() if callable(prompt) else prompt

    def get_agent_response(
        self,
        board_state: dict[str, Any],
        command: str,
        transcript: Sequence[dict[str, str]] = (),
    ) -> AgentResponse:
        schemas = {
            d["function"]["name"]: d["function"]["parameters"]
            for d in self.tool_definitions
        }
        messages = self._messages(board_state, command, transcript)

        for attempt in range(self.max_retries + 1):
            try:
                result = self._complete(messages)
            except ToolCallArgumentsError as exc:
                # Arguments that aren't valid JSON: the provider raises before
                # returning a result, so the whole turn's calls are lost (it is
                # all-or-nothing on parsing). Treat it exactly like a schema
                # violation — a correctable failure, retried until the budget
                # runs out, then dropped. The error names the tool, so the
                # correction turn can too.
                text, valid, errors = "", [], [str(exc)]
            else:
                text = result.content or ""
                valid, errors = _validate_calls(result.tool_calls, schemas)

            if not errors or attempt == self.max_retries:
                # Clean, or out of retries: return what validated, drop the rest.
                return AgentResponse(text=text, tool_calls=tuple(valid))

            # Self-correction turn: tell the model exactly what was wrong.
            messages = messages + [{"role": "user", "content": _correction(errors)}]

        raise AssertionError("unreachable")  # pragma: no cover

    def react(
        self,
        board_state: dict[str, Any],
        changes: list[dict[str, Any]],
        transcript: Sequence[dict[str, str]] = (),
    ) -> str:
        # Phase two reads the *new* state and what changed, never the raw
        # utterance, and offers no tools — the reaction is commentary only, so
        # it cannot loop back into acting.
        prompt = (
            "You just acted on the player's behalf. Here is what happened "
            f"(each entry is a tool call and its result):\n{json.dumps(changes)}"
            f"\n\nNew board state:\n{json.dumps(board_state)}\n\n"
            "React with a short, in-character comment for the player, based "
            "only on these results and the new board. Do not call any tools."
        )
        messages = [
            {"role": "system", "content": self._resolve_system_prompt()},
            *transcript,
            {"role": "user", "content": prompt},
        ]
        thinking = any(change.get("name") in _ANALYSIS_TOOLS for change in changes)
        result = self._complete(messages, use_tools=False, thinking=thinking)
        return result.content or ""

    def _complete(
        self,
        messages: list[dict[str, str]],
        *,
        use_tools: bool = True,
        thinking: bool | None = None,
    ) -> ChatResult:
        # The provider owns the wire (model, sampling, top_k, the payload
        # shape); the brain owns only the two policy knobs — whether tools are
        # offered and whether the thinking channel is on for this call.
        return self.provider.chat(
            messages,
            tools=self.tool_definitions if use_tools else None,
            enable_thinking=(self.enable_thinking if thinking is None else thinking),
        )

    def _messages(
        self,
        board_state: dict[str, Any],
        command: str,
        transcript: Sequence[dict[str, str]] = (),
    ) -> list[dict[str, str]]:
        # Small prompt: personality/instructions, prior conversation (a
        # bounded Transcript window, final answers only), then board truth +
        # command. State comes from current game state, not the raw
        # utterance, so a future fast-parse path stays free to add.
        user = f"Board state:\n{json.dumps(board_state)}\n\nCommand: {command}"
        return [
            {"role": "system", "content": self._resolve_system_prompt()},
            *transcript,
            {"role": "user", "content": user},
        ]


def _validate_calls(
    calls: Sequence[ProviderToolCall], schemas: dict[str, dict[str, Any]]
) -> tuple[list[ToolCall], list[str]]:
    """Split parsed tool calls into (valid ToolCalls, human-readable errors).

    A call is valid when its name is known and its (already-parsed) arguments
    satisfy the tool's schema. The provider has already guaranteed the
    arguments are a JSON object — malformed JSON never reaches here, it raised
    `ToolCallArgumentsError` upstream.
    """
    valid: list[ToolCall] = []
    errors: list[str] = []
    for tc in calls:
        schema = schemas.get(tc.name)
        if schema is None:
            errors.append(f"{tc.name}: unknown tool")
            continue
        try:
            jsonschema.validate(tc.arguments, schema)
        except jsonschema.ValidationError as exc:
            errors.append(f"{tc.name}: invalid arguments ({exc.message})")
            continue
        valid.append(ToolCall(name=tc.name, args=tc.arguments))
    return valid, errors


def _correction(errors: list[str]) -> str:
    joined = "; ".join(errors)
    return (
        "Your previous tool call(s) were rejected: "
        f"{joined}. Call the tools again with corrected arguments."
    )


def create_llama_brain(
    *,
    base_url: str,
    model: str,
    tool_definitions: list[dict[str, Any]],
    system_prompt_provider: Callable[[], str] | None = None,
    enable_thinking: bool = False,
    max_retries: int = _DEFAULT_MAX_RETRIES,
    provider: ChatProvider | None = None,
) -> LlamaBrain:
    """Build a LlamaBrain against a real llama-server (e.g. localhost:8200/v1).

    The system prompt, two ways: by default it is resolved once to a fixed
    string; pass a `system_prompt_provider` — a zero-arg callable the brain
    calls per command — so live settings changes (verbosity/hints) take effect
    immediately (the app-assembly wires it to read `ctx.settings`). Either way
    the brain stays prompt-agnostic: it just carries a string or a callable.

    `provider` is injected in tests / alternate backends; otherwise the factory
    builds a real `LlamaCppProvider` against `base_url` + `model` (no API key —
    llama-server needs none).
    """
    if provider is None:
        provider = LlamaCppProvider(base_url, model)
    system_prompt: str | Callable[[], str] = (
        system_prompt_provider
        if system_prompt_provider is not None
        else system_prompt_for()
    )
    return LlamaBrain(
        provider=provider,
        tool_definitions=tool_definitions,
        system_prompt=system_prompt,
        enable_thinking=enable_thinking,
        max_retries=max_retries,
    )
