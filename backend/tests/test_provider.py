"""The httpx llama.cpp provider, against faked wire responses.

`httpx.MockTransport` plays the server: each test hands the provider a client
whose transport returns a canned `/chat/completions` body (or raises), so the
whole build-request -> wire-validate -> ChatResult path runs without a live
model. These tests pin the exact wire payload (sampling, the non-standard
top_k / chat_template_kwargs plain fields, tools + tool_choice), the
response->ChatResult mapping (tool args parsed, reasoning_content dropped), and
the typed-error contract (HTTP/network vs invalid body vs bad tool-call JSON)
and the `ProviderFailure` each of those carries — which is what tells a dead
server from a request the server will refuse identically forever.
The live round trip against the real server is a scratch smoke test, not here.
"""

import json
from typing import Any

import httpx
import pytest

from chessapp.provider import (
    ChatResult,
    LlamaCppProvider,
    ProviderError,
    ProviderFailure,
    ProviderRequestError,
    ProviderResponseError,
    ToolCall,
    ToolCallArgumentsError,
)

_USER = [{"role": "user", "content": "play e4"}]

# Chess passes the registry's OpenAI-style tool dicts straight through — the
# provider needs no ToolSpec of its own (see tools.py `definitions()`).
_MAKE_MOVE = {
    "type": "function",
    "function": {
        "name": "make_move",
        "description": "Submit a move in SAN or UCI.",
        "parameters": {
            "type": "object",
            "properties": {"move": {"type": "string"}},
            "required": ["move"],
        },
    },
}


def _completion_body(
    message: dict[str, Any], finish_reason: str = "stop"
) -> dict[str, Any]:
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "model": "gemma-4-12b",
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


def _provider_returning(
    body: dict[str, Any],
    *,
    status_code: int = 200,
    captured: list[dict[str, Any]] | None = None,
) -> LlamaCppProvider:
    """A provider whose MockTransport answers every POST with `body`, capturing
    the request payload into `captured` when given one."""

    def handler(request: httpx.Request) -> httpx.Response:
        if captured is not None:
            captured.append(json.loads(request.content))
        return httpx.Response(status_code, json=body)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    return LlamaCppProvider("http://llm.test/v1", "gemma-4-12b", client=client)


# --- the request the provider sends ----------------------------------------


def test_chat_sets_sampling_and_thinking_off_per_request():
    captured: list[dict[str, Any]] = []
    body = _completion_body({"role": "assistant", "content": "e4, classic."})
    result = _provider_returning(body, captured=captured).chat(_USER)

    assert result.content == "e4, classic."
    assert result.tool_calls == []
    assert result.finish_reason == "stop"
    assert result.usage is not None and result.usage.total_tokens == 15

    payload = captured[0]
    assert payload["model"] == "gemma-4-12b"
    # Sampling is always set per request (server flags are defaults only); the
    # thinking channel is off by default for fast tool calls. top_k and
    # chat_template_kwargs ride as plain body fields — no SDK extra_body.
    assert (payload["temperature"], payload["top_p"], payload["top_k"]) == (
        1.0,
        0.95,
        64,
    )
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert "tools" not in payload


def test_chat_temperature_can_be_overridden_per_request():
    # The planner phase samples cooler than the narrator (audit 15): the caller
    # asks for a temperature and the wire carries exactly that, leaving the
    # rest of the BRIEF's sampling set alone.
    captured: list[dict[str, Any]] = []
    body = _completion_body({"role": "assistant", "content": "ok"})
    _provider_returning(body, captured=captured).chat(_USER, temperature=0.2)

    payload = captured[0]
    assert payload["temperature"] == 0.2
    assert (payload["top_p"], payload["top_k"]) == (0.95, 64)


def test_chat_without_a_temperature_sends_the_default_unchanged():
    # None is not "omit temperature" — it is "the module default", so an
    # untouched call is byte-identical to what it sent before the knob existed.
    captured: list[dict[str, Any]] = []
    body = _completion_body({"role": "assistant", "content": "ok"})
    _provider_returning(body, captured=captured).chat(_USER, temperature=None)
    assert captured[0]["temperature"] == 1.0


def test_chat_thinking_can_be_enabled():
    captured: list[dict[str, Any]] = []
    body = _completion_body({"role": "assistant", "content": "deep thoughts"})
    _provider_returning(body, captured=captured).chat(_USER, enable_thinking=True)
    assert captured[0]["chat_template_kwargs"] == {"enable_thinking": True}


def test_chat_with_tools_sends_them_verbatim_with_auto_choice():
    captured: list[dict[str, Any]] = []
    body = _completion_body({"role": "assistant", "content": "ok"})
    _provider_returning(body, captured=captured).chat(_USER, tools=[_MAKE_MOVE])

    payload = captured[0]
    assert payload["tool_choice"] == "auto"
    # The OpenAI-style dict is passed straight through, unwrapped.
    assert payload["tools"] == [_MAKE_MOVE]


# --- response -> ChatResult mapping ----------------------------------------


def test_tool_call_arguments_are_parsed_from_the_wire_json_string():
    body = _completion_body(
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {
                        "name": "make_move",
                        "arguments": '{"move": "e2e4"}',
                    },
                }
            ],
        },
        finish_reason="tool_calls",
    )
    result = _provider_returning(body).chat(_USER, tools=[_MAKE_MOVE])

    assert result.content is None
    call = result.tool_calls[0]
    assert (call.id, call.name) == ("call-1", "make_move")
    assert call.arguments == {"move": "e2e4"}  # a dict, already parsed


def test_empty_tool_arguments_become_empty_dict():
    # No-arg tools (new_game, undo) may arrive with "" or "{}" arguments.
    body = _completion_body(
        {
            "role": "assistant",
            "tool_calls": [
                {"id": "c", "function": {"name": "new_game", "arguments": ""}}
            ],
        }
    )
    result = _provider_returning(body).chat(_USER)
    assert result.tool_calls[0].arguments == {}


def test_malformed_tool_arguments_raise_not_best_effort():
    # A quant hiccup: arguments that aren't valid JSON. The provider never
    # best-effort parses — it raises a typed error naming the offending tool,
    # which the brain's correction loop treats as a retryable failure.
    body = _completion_body(
        {
            "role": "assistant",
            "tool_calls": [
                {"id": "c", "function": {"name": "make_move", "arguments": "{bad"}}
            ],
        }
    )
    with pytest.raises(ToolCallArgumentsError) as excinfo:
        _provider_returning(body).chat(_USER, tools=[_MAKE_MOVE])
    assert excinfo.value.tool_name == "make_move"
    assert "make_move" in str(excinfo.value)


def test_non_object_tool_arguments_raise():
    # Valid JSON, wrong shape: a list, not an arguments object.
    body = _completion_body(
        {
            "role": "assistant",
            "tool_calls": [
                {"id": "c", "function": {"name": "make_move", "arguments": "[1]"}}
            ],
        }
    )
    with pytest.raises(ToolCallArgumentsError):
        _provider_returning(body).chat(_USER, tools=[_MAKE_MOVE])


# --- the thought-block invariant, pinned at the provider layer -------------


def test_reasoning_content_is_never_answer_text_and_never_round_trips():
    # Gemma's chain-of-thought arrives in a separate field. It is validated
    # (so an unexpected shape fails loudly) but never surfaced as content and
    # never serialized back into history (BRIEF: final answers only).
    body = _completion_body(
        {
            "role": "assistant",
            "content": "e4 it is.",
            "reasoning_content": "The king's pawn opening is e2e4...",
        }
    )
    result = _provider_returning(body).chat(_USER)
    assert result.content == "e4 it is."
    assert "reasoning_content" not in result.to_message()


def test_to_message_re_serializes_tool_calls_for_history():
    result = ChatResult(
        content=None,
        tool_calls=[ToolCall(id="call-1", name="make_move", arguments={"move": "e4"})],
        finish_reason="tool_calls",
        usage=None,
    )
    message = result.to_message()
    assert message["role"] == "assistant"
    # Arguments go back to the wire's JSON-string form for the next request.
    assert message["tool_calls"][0]["function"]["arguments"] == '{"move": "e4"}'


# --- typed errors: no usable body ------------------------------------------


def test_http_error_status_raises_request_error():
    with pytest.raises(ProviderRequestError, match="500"):
        _provider_returning({"error": "boom"}, status_code=500).chat(_USER)


def test_network_failure_raises_request_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    provider = LlamaCppProvider("http://llm.test/v1", "gemma-4-12b", client=client)
    with pytest.raises(ProviderRequestError, match="request failed"):
        provider.chat(_USER)


def test_invalid_wire_body_raises_response_error():
    # 200 OK, but the body fails wire validation (no choices).
    with pytest.raises(ProviderResponseError, match="wire validation"):
        _provider_returning({"object": "chat.completion", "choices": []}).chat(_USER)


# --- the failure kind: which *sort* of "no" this was -------------------------
#
# The exception type only says where the failure happened (no body / a bad
# body). What a caller has to branch on is whether asking again could work: a
# crashed llama-server answers differently in ten seconds, an HTTP 400 answers
# the same way forever. Before this, both arrived as a bare `ProviderError` the
# brain caught and discarded, so the eval harness retried a context overrun five
# times before aborting on it.


def test_a_dead_socket_is_unreachable_and_transient():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    provider = LlamaCppProvider("http://llm.test/v1", "gemma-4-12b", client=client)
    with pytest.raises(ProviderRequestError) as raised:
        provider.chat(_USER)
    assert raised.value.failure is ProviderFailure.UNREACHABLE
    assert raised.value.failure.transient


def test_a_5xx_is_a_server_error_and_transient():
    # llama-swap answers 502/503 while the upstream model is restarting — the
    # crash cadence the eval harness exists to retry through.
    with pytest.raises(ProviderRequestError) as raised:
        _provider_returning({"error": "upstream gone"}, status_code=502).chat(_USER)
    assert raised.value.failure is ProviderFailure.SERVER_ERROR
    assert raised.value.failure.transient


def test_a_4xx_is_a_rejection_and_not_transient():
    # The 400 llama.cpp answers a context overrun with. Sending the same request
    # again gets the same 400, which is the whole distinction this field exists
    # to carry.
    with pytest.raises(ProviderRequestError) as raised:
        _provider_returning(
            {"error": {"message": "exceeds context size"}}, status_code=400
        ).chat(_USER)
    assert raised.value.failure is ProviderFailure.REJECTED
    assert not raised.value.failure.transient


def test_a_body_that_fails_wire_validation_is_not_transient():
    with pytest.raises(ProviderResponseError) as raised:
        _provider_returning({"object": "chat.completion", "choices": []}).chat(_USER)
    assert raised.value.failure is ProviderFailure.MALFORMED_RESPONSE
    assert not raised.value.failure.transient


def test_unparseable_tool_arguments_carry_their_own_kind():
    error = ToolCallArgumentsError("make_move", "arguments are not a JSON object")
    assert error.failure is ProviderFailure.BAD_TOOL_ARGUMENTS
    assert not error.failure.transient


def test_an_unclassified_provider_error_fails_toward_retry():
    """A bare `ProviderError` — what a fake raises, and what any future caller
    that forgets to classify will raise — reads as a dead server.

    The default is deliberate rather than incidental: retrying a deterministic
    failure costs a few wasted samples, while treating a dead server as
    deterministic aborts the whole eval suite on the first crash. The cheap
    mistake is the default."""
    unclassified = ProviderError("llama-server went away")
    assert unclassified.failure is ProviderFailure.UNREACHABLE
