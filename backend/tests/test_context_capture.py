"""Context capture at the provider seam (#359), against a faked llama-server.

`httpx.MockTransport` plays llama-swap: `/v1/chat/completions` answers with a
canned body and `/upstream/<model>/apply-template` with a rendered prompt. The
point of the capture is that it is exact, so the assertions compare bytes: what
the transport received is what the record holds, and what the transport served
is what the record holds. Everything else — ids, phase, seq, failures — is the
bookkeeping that lets `scripts/watch_context.py` put a call under its turn.
"""

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from chessapp.brain import PHASE_PLANNER, PHASE_UNKNOWN
from chessapp.context_capture import (
    CAPTURE_SCHEMA,
    KIND_MODEL_CALL,
    JsonlContextCapture,
    model_phase,
)
from chessapp.progress import ProgressReporter
from chessapp.provider import (
    LlamaCppProvider,
    ProviderFailure,
    ProviderRequestError,
)

_USER = [{"role": "user", "content": "play e4 — «quoted» ♞"}]
_TOOL = {
    "type": "function",
    "function": {
        "name": "make_move",
        "description": "Submit a move.",
        "parameters": {"type": "object", "properties": {"move": {"type": "string"}}},
    },
}
# A response the way llama-server writes it: its own spacing, a thought block,
# and tool-call arguments as a raw JSON string. The capture keeps it as sent.
_RESPONSE = (
    b'{"choices":[{"index":0,"finish_reason":"tool_calls","message":'
    b'{"role":"assistant","content":null,'
    b'"reasoning_content":"The knight on g1 \\u2014 hmm.",'
    b'"tool_calls":[{"id":"c0","type":"function","function":'
    b'{"name":"make_move","arguments":"{\\"move\\": \\"e4\\"}"}}]}}],'
    b' "usage":{"prompt_tokens":12,"completion_tokens":3,"total_tokens":15}}'
)
_PROMPT = "<|turn>user\nplay e4<turn|>\n<|turn>model\n"


class FakeServer:
    """A llama-swap stand-in that remembers every request it received."""

    def __init__(
        self,
        *,
        chat: httpx.Response | Exception | None = None,
        template: httpx.Response | Exception | None = None,
    ) -> None:
        self.requests: list[httpx.Request] = []
        self._chat = chat or httpx.Response(200, content=_RESPONSE)
        self._template = template or httpx.Response(200, json={"prompt": _PROMPT})

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        answer = (
            self._chat
            if request.url.path.endswith("/chat/completions")
            else self._template
        )
        if isinstance(answer, Exception):
            raise answer
        return answer

    def provider(self, capture: Any) -> LlamaCppProvider:
        client = httpx.Client(transport=httpx.MockTransport(self))
        return LlamaCppProvider(
            "http://llm.test/v1", "gemma-4-12b", client=client, capture=capture
        )


def records(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_the_record_holds_the_exact_bytes_in_and_out(tmp_path):
    server = FakeServer()
    path = tmp_path / "context.jsonl"
    provider = server.provider(JsonlContextCapture(path))

    result = provider.chat(_USER, tools=[_TOOL], max_tokens=64, temperature=0.3)

    assert result.tool_calls[0].arguments == {"move": "e4"}
    (record,) = records(path)
    sent = server.requests[0]
    assert sent.url.path == "/v1/chat/completions"
    assert record["request"].encode("utf-8") == sent.content
    assert record["response"].encode("utf-8") == _RESPONSE
    # Every sampling knob is in those bytes, because they are the request.
    body = json.loads(record["request"])
    assert body["temperature"] == 0.3 and body["max_tokens"] == 64
    assert body["tools"] == [_TOOL]
    assert record["kind"] == KIND_MODEL_CALL
    assert record["schema"] == CAPTURE_SCHEMA
    assert record["status_code"] == 200
    assert record["error"] == ""
    assert record["url"] == "http://llm.test/v1/chat/completions"


def test_the_rendered_template_is_asked_of_the_server_with_the_same_bytes(tmp_path):
    server = FakeServer()
    path = tmp_path / "context.jsonl"
    server.provider(JsonlContextCapture(path)).chat(_USER, tools=[_TOOL])

    chat, template = server.requests
    assert template.url.path == "/upstream/gemma-4-12b/apply-template"
    assert template.content == chat.content
    assert records(path)[0]["template"] == {"prompt": _PROMPT}


def test_capture_off_sends_one_request_and_the_same_bytes(tmp_path):
    captured = FakeServer()
    captured.provider(JsonlContextCapture(tmp_path / "c.jsonl")).chat(_USER)
    plain = FakeServer()
    plain.provider(None).chat(_USER)

    assert len(plain.requests) == 1
    assert plain.requests[0].content == captured.requests[0].content


def test_a_template_the_server_will_not_render_is_recorded_not_raised(tmp_path):
    server = FakeServer(template=httpx.Response(404, text="404 page not found"))
    path = tmp_path / "context.jsonl"
    result = server.provider(JsonlContextCapture(path)).chat(_USER)

    assert result.tool_calls, "the call's own result is untouched"
    assert records(path)[0]["template"] == {"error": "HTTP 404: 404 page not found"}


def test_a_template_request_that_dies_is_recorded_not_raised(tmp_path):
    server = FakeServer(template=httpx.ConnectError("gone"))
    path = tmp_path / "context.jsonl"
    server.provider(JsonlContextCapture(path)).chat(_USER)
    assert records(path)[0]["template"] == {"error": "ConnectError: gone"}


def test_a_refused_call_is_recorded_and_still_raises_its_kind(tmp_path):
    server = FakeServer(chat=httpx.Response(400, text="context overrun"))
    path = tmp_path / "context.jsonl"
    with pytest.raises(ProviderRequestError) as excinfo:
        server.provider(JsonlContextCapture(path)).chat(_USER)
    assert excinfo.value.failure is ProviderFailure.REJECTED

    (record,) = records(path)
    assert record["status_code"] == 400
    assert record["response"] == "context overrun"
    assert record["error"].startswith("ProviderRequestError: llama-server returned 400")
    # The request the server refused is exactly the one worth rendering.
    assert record["template"] == {"prompt": _PROMPT}


def test_a_dead_socket_is_recorded_without_asking_for_a_template(tmp_path):
    server = FakeServer(chat=httpx.ConnectError("refused"))
    path = tmp_path / "context.jsonl"
    with pytest.raises(ProviderRequestError):
        server.provider(JsonlContextCapture(path)).chat(_USER)

    assert len(server.requests) == 1, "no second request to a server that is down"
    (record,) = records(path)
    assert record["status_code"] is None
    assert record["response"] is None
    assert record["template"] is None
    assert (
        record["error"] == "ProviderRequestError: llama-server request failed: refused"
    )
    assert record["request"].encode("utf-8") == server.requests[0].content


def test_a_capture_that_fails_never_costs_the_turn(tmp_path):
    class Broken:
        def begin(self):
            raise OSError("disk full")

        def record(self, stamp, call):  # pragma: no cover - begin already failed
            raise OSError("disk full")

    result = FakeServer().provider(Broken()).chat(_USER)
    assert result.tool_calls


def test_a_file_that_cannot_be_written_never_costs_the_turn(tmp_path):
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("")
    result = FakeServer().provider(JsonlContextCapture(blocker / "c.jsonl")).chat(_USER)
    assert result.tool_calls


def test_a_call_carries_its_interaction_its_phase_and_its_place(tmp_path):
    path = tmp_path / "context.jsonl"
    provider = FakeServer().provider(JsonlContextCapture(path))
    reporter = ProgressReporter()

    provider.chat(_USER)  # outside any interaction
    with reporter.interaction("abc123", 7):
        with model_phase(PHASE_PLANNER):
            provider.chat(_USER)
            provider.chat(_USER)
        provider.chat(_USER)
    with reporter.interaction("def456", 8):
        provider.chat(_USER)

    stamps = [
        (r["correlation_id"], r["turn_id"], r["phase"], r["seq"]) for r in records(path)
    ]
    assert stamps == [
        (None, None, PHASE_UNKNOWN, 1),
        ("abc123", 7, PHASE_PLANNER, 1),
        ("abc123", 7, PHASE_PLANNER, 2),
        ("abc123", 7, PHASE_UNKNOWN, 3),
        ("def456", 8, PHASE_UNKNOWN, 1),
    ]
    first = records(path)[1]
    assert first["started_at"] <= first["ended_at"]
    assert isinstance(first["ms"], int)
