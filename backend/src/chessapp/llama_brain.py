"""llama-server brain: the OpenAI-compatible `Brain` implementation.

llama.cpp's server speaks the OpenAI chat API, so the brain is just that
client pointed at localhost with a `tools` array. This module is the only
place that knows the model is Gemma-4 behind llama.cpp; everything else sees
the `Brain` protocol.

Two model-specific quirks, learned from a live run and handled here:

- Gemma emits its chain-of-thought in a separate ``reasoning_content`` field.
  We read ``content`` only, so thought blocks never leak into commentary or
  back into history (BRIEF: final answers only).
- Thinking is toggled via the non-standard ``chat_template_kwargs`` /
  ``top_k`` fields, passed through ``extra_body``. Thinking is OFF by default
  for fast move parsing; callers flip it ON for analysis.
"""

import json
from dataclasses import dataclass
from typing import Any

from chessapp.brain import AgentResponse, ToolCall

# BRIEF-mandated sampling for Gemma-4 tool calling.
_TEMPERATURE = 1.0
_TOP_P = 0.95
_TOP_K = 64


@dataclass
class LlamaBrain:
    """A `Brain` backed by an OpenAI-compatible client (llama-server).

    The client is injected so tests exercise the mapping without a live LLM;
    `create_llama_brain` builds the real one. `tool_definitions` are the
    registry's OpenAI-style schemas — the single source of truth for what the
    agent may call.
    """

    client: Any
    model: str
    tool_definitions: list[dict[str, Any]]
    system_prompt: str
    enable_thinking: bool = False

    def get_agent_response(
        self, board_state: dict[str, Any], command: str
    ) -> AgentResponse:
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=self._messages(board_state, command),
            tools=self.tool_definitions,
            tool_choice="auto",
            temperature=_TEMPERATURE,
            top_p=_TOP_P,
            extra_body={
                "top_k": _TOP_K,
                "chat_template_kwargs": {"enable_thinking": self.enable_thinking},
            },
        )
        return _to_agent_response(completion)

    def _messages(
        self, board_state: dict[str, Any], command: str
    ) -> list[dict[str, str]]:
        # Small prompt: personality/instructions, then board truth + command.
        # State comes from current game state, not the raw utterance, so a
        # future fast-parse path stays free to add.
        user = f"Board state:\n{json.dumps(board_state)}\n\nCommand: {command}"
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user},
        ]


def _to_agent_response(completion: Any) -> AgentResponse:
    """Map an OpenAI chat completion to the seam's AgentResponse.

    Reads ``content`` only (``reasoning_content`` is deliberately ignored) and
    parses each tool call's JSON ``arguments`` string into a dict; blank args
    become ``{}``.
    """
    message = completion.choices[0].message
    text = message.content or ""
    tool_calls = tuple(
        ToolCall(name=tc.function.name, args=_parse_args(tc.function.arguments))
        for tc in (message.tool_calls or ())
    )
    return AgentResponse(text=text, tool_calls=tool_calls)


def _parse_args(raw: str | None) -> dict[str, Any]:
    if not raw or not raw.strip():
        return {}
    return json.loads(raw)


def create_llama_brain(
    *,
    base_url: str,
    model: str,
    tool_definitions: list[dict[str, Any]],
    system_prompt: str,
    enable_thinking: bool = False,
    api_key: str = "llama-server-needs-no-key",
) -> LlamaBrain:
    """Build a LlamaBrain against a real llama-server (e.g. localhost:8080/v1)."""
    from openai import OpenAI

    client = OpenAI(base_url=base_url, api_key=api_key)
    return LlamaBrain(
        client=client,
        model=model,
        tool_definitions=tool_definitions,
        system_prompt=system_prompt,
        enable_thinking=enable_thinking,
    )
