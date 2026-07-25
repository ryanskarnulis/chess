"""llama.cpp chat provider: the OpenAI wire format spoken over plain httpx.

llama.cpp's server speaks the OpenAI chat API, so a provider is just a POST to
`{base_url}/chat/completions` — no SDK. Every response is validated by Pydantic
wire models at the boundary: a malformed body, or tool-call arguments that
aren't a JSON object, raise a typed error rather than being best-effort parsed.

This is the one place that knows the model is Gemma-4 behind llama.cpp; the
`Brain` on top (`llama_brain.py`) sees only `ChatResult`. Two Gemma quirks are
handled here:

- Chain-of-thought arrives in a separate `reasoning_content` field. It is
  validated (so an unexpected shape fails loudly) but never surfaced as answer
  text and never serialized back into history — `to_message()` drops it (the
  binding invariant: final answers only, never thought blocks).
- Thinking is toggled per request via the non-standard `chat_template_kwargs`
  field (default OFF, for fast tool calls); llama-server also takes `top_k` as
  a plain body field. Both ride in the JSON payload directly — no SDK
  `extra_body` indirection.

Correction retries on bad tool calls live in the `Brain`, not here (that is
where the retry budget and schema validation already are), which is why
`ToolCallArgumentsError` carries the tool name for the correction message.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any, Protocol

import httpx
from pydantic import BaseModel, Field, ValidationError

# BRIEF-mandated sampling for Gemma-4 tool calling. The canonical set lives in
# ../agent-standard/model-profile.md; the provider always sets it per request
# so a drift in the shared server config never changes chess's behavior.
# `_TEMPERATURE` is the default a caller gets by saying nothing — `chat` takes a
# per-request override for the planner phase's cooler sampling.
_TEMPERATURE = 1.0
_TOP_P = 0.95
_TOP_K = 64

# One generous read timeout, not a special-cased first request: a cold model
# load through llama-swap is ~100s before the first byte, warm calls never get
# near it; the connect phase stays short so a dead server fails fast.
_READ_TIMEOUT = 300.0
_CONNECT_TIMEOUT = 10.0


class ProviderError(Exception):
    """Base for everything a completion attempt can raise."""


class ProviderRequestError(ProviderError):
    """No usable HTTP response: connect/timeout failure or a non-200 status."""


class ProviderResponseError(ProviderError):
    """The server answered 200 but the body failed validation."""


class ToolCallArgumentsError(ProviderResponseError):
    """The model emitted tool-call arguments that aren't a JSON object.

    Carries the tool name so the brain's correction turn can name the offender
    when it feeds the failure back for a retry.
    """

    def __init__(self, tool_name: str, detail: str) -> None:
        super().__init__(f"tool call {tool_name!r}: {detail}")
        self.tool_name = tool_name


class ToolCall(BaseModel):
    """A tool call with its arguments already parsed from the wire's JSON string."""

    id: str
    name: str
    arguments: dict[str, Any]


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class _WireFunction(BaseModel):
    name: str
    arguments: str | None = None


class _WireToolCall(BaseModel):
    id: str = ""
    function: _WireFunction


class _WireMessage(BaseModel):
    content: str | None = None
    # Gemma's chain-of-thought channel: validated so unexpected shapes fail
    # loudly, but never surfaced as answer text and never sent back in history.
    reasoning_content: str | None = None
    tool_calls: list[_WireToolCall] = []


class _WireChoice(BaseModel):
    message: _WireMessage
    finish_reason: str | None = None


class _WireCompletion(BaseModel):
    choices: list[_WireChoice] = Field(min_length=1)
    usage: Usage | None = None


class ChatResult(BaseModel):
    """One validated completion turn. `content` is answer text only."""

    content: str | None
    tool_calls: list[ToolCall]
    finish_reason: str | None
    usage: Usage | None

    def to_message(self) -> dict[str, Any]:
        """This turn as an assistant message for the next request's history.

        Tool arguments are re-serialized to the wire's JSON-string form;
        `reasoning_content` deliberately never round-trips (final answers only).
        """
        message: dict[str, Any] = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            message["tool_calls"] = [
                {
                    "type": "function",
                    "id": call.id,
                    "function": {
                        "name": call.name,
                        "arguments": json.dumps(call.arguments),
                    },
                }
                for call in self.tool_calls
            ]
        return message


class ChatProvider(Protocol):
    """What the brain needs from a provider — matched by `LlamaCppProvider`.

    `tools` are OpenAI-style function definitions (the registry's
    `definitions()` output) passed straight through; the brain owns schema
    validation of the parsed arguments, so the provider only guarantees they
    are a JSON object.

    `temperature` is the one sampling knob a caller may set per request, for
    the planner/narrator split's per-phase sampling; `None` means the module
    default, so a caller that does not care sends exactly what it always did.
    """

    def chat(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        tools: Sequence[dict[str, Any]] | None = None,
        enable_thinking: bool = False,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> ChatResult: ...


class LlamaCppProvider:
    """Chat-completions client for the shared llama-server (behind llama-swap).

    Synchronous by design, matching the sync tool/API layer; FastAPI runs sync
    callers in worker threads. `client` is injectable for tests
    (`httpx.MockTransport`); otherwise the provider owns one.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        timeout_seconds: float = _READ_TIMEOUT,
        client: httpx.Client | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._client = client or httpx.Client(
            timeout=httpx.Timeout(timeout_seconds, connect=_CONNECT_TIMEOUT)
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> LlamaCppProvider:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def chat(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        tools: Sequence[dict[str, Any]] | None = None,
        enable_thinking: bool = False,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> ChatResult:
        """One completion turn, optionally offering tools."""
        payload = self._payload(
            messages,
            tools=tools,
            enable_thinking=enable_thinking,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return self._result(self._post(payload))

    def _payload(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        tools: Sequence[dict[str, Any]] | None,
        enable_thinking: bool,
        max_tokens: int | None,
        temperature: float | None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": list(messages),
            # `None` resolves to the module default rather than omitting the
            # field, so a caller with no opinion sends the same bytes as ever.
            "temperature": _TEMPERATURE if temperature is None else temperature,
            "top_p": _TOP_P,
            # llama-server takes these OpenAI extensions as plain body fields.
            "top_k": _TOP_K,
            "chat_template_kwargs": {"enable_thinking": enable_thinking},
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if tools:
            # Already OpenAI-format dicts (registry `definitions()`) — verbatim.
            payload["tools"] = list(tools)
            payload["tool_choice"] = "auto"
        return payload

    def _post(self, payload: dict[str, Any]) -> _WireCompletion:
        try:
            response = self._client.post(
                f"{self._base_url}/chat/completions", json=payload
            )
        except httpx.HTTPError as exc:
            raise ProviderRequestError(f"llama-server request failed: {exc}") from exc
        if response.status_code != 200:
            raise ProviderRequestError(
                f"llama-server returned {response.status_code}: {response.text[:500]}"
            )
        try:
            return _WireCompletion.model_validate_json(response.text)
        except ValidationError as exc:
            raise ProviderResponseError(
                f"llama-server response failed wire validation: {exc}"
            ) from exc

    @staticmethod
    def _result(completion: _WireCompletion) -> ChatResult:
        choice = completion.choices[0]
        calls: list[ToolCall] = []
        for wire_call in choice.message.tool_calls:
            raw = wire_call.function.arguments
            if raw is None or not raw.strip():
                arguments: Any = {}
            else:
                try:
                    arguments = json.loads(raw)
                except json.JSONDecodeError as exc:
                    raise ToolCallArgumentsError(
                        wire_call.function.name,
                        f"arguments are not valid JSON ({exc.msg})",
                    ) from exc
            if not isinstance(arguments, dict):
                raise ToolCallArgumentsError(
                    wire_call.function.name, "arguments are not a JSON object"
                )
            calls.append(
                ToolCall(
                    id=wire_call.id,
                    name=wire_call.function.name,
                    arguments=arguments,
                )
            )
        return ChatResult(
            content=choice.message.content,
            tool_calls=calls,
            finish_reason=choice.finish_reason,
            usage=completion.usage,
        )
