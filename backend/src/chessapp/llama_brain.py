"""llama-server brain: the OpenAI-compatible `Brain` implementation.

llama.cpp's server speaks the OpenAI chat API, so the brain is just that
client pointed at localhost with a `tools` array. This module is the only
place that knows the model is Gemma-4 behind llama.cpp; everything else sees
the `Brain` protocol.

Model-specific quirks, learned from a live run and handled here:

- Gemma emits its chain-of-thought in a separate ``reasoning_content`` field.
  We read ``content`` only, so thought blocks never leak into commentary or
  back into history (BRIEF: final answers only).
- Thinking is toggled via the non-standard ``chat_template_kwargs`` /
  ``top_k`` fields, passed through ``extra_body``. Thinking is OFF by default
  for fast move parsing; callers flip it ON for analysis.

Under quantization the model can occasionally emit a malformed tool call
(non-JSON arguments, an unknown tool, args that violate the schema). The
`ToolRegistry` is the ultimate guard — it turns such calls into error data,
never a crash — but a bad call there just wastes a turn. So the brain first
validates each call against the tool schemas and, on failure, feeds the
error back and retries a bounded number of times, giving the model a chance
to self-correct. If it never does, the invalid calls are dropped.
"""

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import jsonschema

from chessapp.brain import AgentResponse, ToolCall
from chessapp.personality import system_prompt_for

# BRIEF-mandated sampling for Gemma-4 tool calling.
_TEMPERATURE = 1.0
_TOP_P = 0.95
_TOP_K = 64
_DEFAULT_MAX_RETRIES = 2

# Reacting to these tools' results is analysis work: thinking goes ON
# (BRIEF: thinking OFF for fast move parsing/reactions, ON for analysis).
_ANALYSIS_TOOLS = frozenset(
    {"evaluate_position", "get_best_moves", "analyze_last_move"}
)


@dataclass
class LlamaBrain:
    """A `Brain` backed by an OpenAI-compatible client (llama-server).

    The client is injected so tests exercise the mapping without a live LLM;
    `create_llama_brain` builds the real one. `tool_definitions` are the
    registry's OpenAI-style schemas — the single source of truth for what the
    agent may call, and what tool calls are validated against.
    """

    client: Any
    model: str
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
            message = self._complete(messages)
            text = message.content or ""
            valid, errors = _validate_calls(message.tool_calls or (), schemas)

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
        message = self._complete(messages, use_tools=False, thinking=thinking)
        return message.content or ""

    def _complete(
        self,
        messages: list[dict[str, str]],
        *,
        use_tools: bool = True,
        thinking: bool | None = None,
    ) -> Any:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": _TEMPERATURE,
            "top_p": _TOP_P,
            "extra_body": {
                "top_k": _TOP_K,
                "chat_template_kwargs": {
                    "enable_thinking": (
                        self.enable_thinking if thinking is None else thinking
                    )
                },
            },
        }
        if use_tools:
            kwargs["tools"] = self.tool_definitions
            kwargs["tool_choice"] = "auto"
        completion = self.client.chat.completions.create(**kwargs)
        return completion.choices[0].message

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
    raw_calls: Any, schemas: dict[str, dict[str, Any]]
) -> tuple[list[ToolCall], list[str]]:
    """Split raw tool calls into (valid ToolCalls, human-readable errors).

    A call is valid when its name is known, its ``arguments`` parse as JSON,
    and those args satisfy the tool's schema. ``reasoning_content`` is never
    consulted.
    """
    valid: list[ToolCall] = []
    errors: list[str] = []
    for tc in raw_calls:
        name = tc.function.name
        schema = schemas.get(name)
        if schema is None:
            errors.append(f"{name}: unknown tool")
            continue
        try:
            args = _parse_args(tc.function.arguments)
        except json.JSONDecodeError as exc:
            errors.append(f"{name}: arguments are not valid JSON ({exc.msg})")
            continue
        try:
            jsonschema.validate(args, schema)
        except jsonschema.ValidationError as exc:
            errors.append(f"{name}: invalid arguments ({exc.message})")
            continue
        valid.append(ToolCall(name=name, args=args))
    return valid, errors


def _parse_args(raw: str | None) -> dict[str, Any]:
    if not raw or not raw.strip():
        return {}
    return json.loads(raw)


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
    api_key: str = "llama-server-needs-no-key",
    client: Any | None = None,
) -> LlamaBrain:
    """Build a LlamaBrain against a real llama-server (e.g. localhost:8080/v1).

    The system prompt, two ways: by default it is resolved once to a fixed
    string; pass a `system_prompt_provider` — a zero-arg callable the brain
    calls per command — so live settings changes (verbosity/hints) take effect
    immediately (the app-assembly wires it to read `ctx.settings`). Either way
    the brain stays prompt-agnostic: it just carries a string or a callable.

    `client` is injected in tests / alternate backends; otherwise the factory
    builds a real OpenAI client against `base_url`.
    """
    if client is None:
        from openai import OpenAI

        client = OpenAI(base_url=base_url, api_key=api_key)
    system_prompt: str | Callable[[], str] = (
        system_prompt_provider
        if system_prompt_provider is not None
        else system_prompt_for()
    )
    return LlamaBrain(
        client=client,
        model=model,
        tool_definitions=tool_definitions,
        system_prompt=system_prompt,
        enable_thinking=enable_thinking,
        max_retries=max_retries,
    )
