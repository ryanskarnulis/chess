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

Every failure also carries a `ProviderFailure` — the machine-readable half of
the "no". The exception hierarchy says where the attempt broke; the field says
whether asking again could work, so a caller never has to read a message to
tell a crashed server from a request the server refuses identically forever.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from enum import StrEnum
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


class ProviderFailure(StrEnum):
    """*Which* kind of "no" a completion attempt got — a field, not a message.

    The exception type says where the failure happened (no usable body / a bad
    body); this says whether asking again could plausibly work, which is the
    only question a caller ever has. llama-server crash-restarts every few
    minutes of sustained generation and answers differently ten seconds later;
    an HTTP 400 (llama.cpp's answer to a context overrun) answers the same way
    forever. Both used to arrive as a bare `ProviderError` whose message the
    brain caught and discarded, so the eval harness retried the second one five
    times before aborting on it.

    Code owns the transient/not answer, the same rule the tool layer's `retry`
    field keeps (`tools.py`): a caller must not have to infer it from wording.
    """

    #: The socket never answered — connect refused, reset, timeout.
    UNREACHABLE = "unreachable"
    #: A 5xx. llama-swap's answer while the upstream model is restarting.
    SERVER_ERROR = "server_error"
    #: A 4xx: the server understood the request and refuses it. A context
    #: overrun is this, and it is deterministic — the same bytes get the same
    #: refusal.
    REJECTED = "rejected"
    #: 200, but the body failed wire validation. Version skew, not a crash.
    MALFORMED_RESPONSE = "malformed_response"
    #: The model's tool-call arguments weren't a JSON object. The brain answers
    #: this with a correction rather than ending the turn, so it never reaches a
    #: `provider_error` stop — it is named here to keep the vocabulary complete.
    BAD_TOOL_ARGUMENTS = "bad_tool_arguments"

    @property
    def transient(self) -> bool:
        """Whether re-sending the identical request could get a different
        answer. Only the two that mean "the server wasn't there"."""
        return self in _TRANSIENT_FAILURES


_TRANSIENT_FAILURES = frozenset(
    {ProviderFailure.UNREACHABLE, ProviderFailure.SERVER_ERROR}
)


class ProviderError(Exception):
    """Base for everything a completion attempt can raise.

    `failure` is the machine-readable half — see `ProviderFailure`. It defaults
    to UNREACHABLE so an unclassified failure (a test double, a future raise
    site that forgets) reads as a dead server, which is the cheap mistake:
    retrying a deterministic failure wastes a few samples, while calling a
    restarting server deterministic aborts a whole eval suite on the first
    crash.
    """

    def __init__(
        self,
        message: str,
        failure: ProviderFailure = ProviderFailure.UNREACHABLE,
    ) -> None:
        super().__init__(message)
        self.failure = failure


class ProviderRequestError(ProviderError):
    """No usable HTTP response: connect/timeout failure or a non-200 status."""


class ProviderResponseError(ProviderError):
    """The server answered 200 but the body failed validation."""

    def __init__(
        self,
        message: str,
        failure: ProviderFailure = ProviderFailure.MALFORMED_RESPONSE,
    ) -> None:
        super().__init__(message, failure)


class ToolCallArgumentsError(ProviderResponseError):
    """The model emitted tool-call arguments that aren't a JSON object.

    Carries the tool name so the brain's correction turn can name the offender
    when it feeds the failure back for a retry.
    """

    def __init__(self, tool_name: str, detail: str) -> None:
        super().__init__(
            f"tool call {tool_name!r}: {detail}", ProviderFailure.BAD_TOOL_ARGUMENTS
        )
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
            raise ProviderRequestError(
                f"llama-server request failed: {exc}", ProviderFailure.UNREACHABLE
            ) from exc
        if response.status_code != 200:
            # 5xx is the server having a bad moment (llama-swap mid-restart);
            # 4xx is the server having read the request and refusing it, which
            # it will do again identically.
            raise ProviderRequestError(
                f"llama-server returned {response.status_code}: {response.text[:500]}",
                ProviderFailure.SERVER_ERROR
                if response.status_code >= 500
                else ProviderFailure.REJECTED,
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
