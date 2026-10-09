"""Today's config sends today's bytes (#375): the golden-request pin.

Model profiles moved the gemma-4-12b knowledge — sampling, the thinking
toggle's shape, the per-phase temperature and token caps — out of the code and
into `data/profiles/`. The issue's bar is that the default config's requests
are unchanged *byte for byte*, so this drives every phase the brain has
(planner, narrator with thinking off and on, the observe beat's narration, the
answer reader) through a real `LlamaCppProvider` over a faked llama-server and
compares the exact bodies it sent with the ones recorded from `main` before
the change (`fixtures/gemma_4_12b_requests.json`).

The default brain moved to gemma-4-26b-a4b (#433), so each pinned model has
its own recording (`fixtures/<model>_requests.json`, dashes as underscores):
the default's are the default config's bytes, and the 12B's stay pinned
because it is still a brain the app can run.

Re-record only for a change that is *meant* to move the bytes, and say so in
its PR: `CHESSAPP_RECORD_GOLDEN=1 pytest tests/test_profile_bytes.py`.
"""

import json
import os
from pathlib import Path
from typing import Any

import httpx
import pytest

from chessapp.app import DEFAULT_MODEL
from chessapp.game import GameSession
from chessapp.llama_brain import create_llama_brain
from chessapp.profiles import PLANNER, load_profile
from chessapp.provider import LlamaCppProvider
from chessapp.tools import ToolContext, brain_tool_definitions, build_registry
from fakes import FakeEngine

_PINNED = ("gemma-4-12b", "gemma-4-26b-a4b")


def _golden(model: str) -> Path:
    name = model.replace("-", "_") + "_requests.json"
    return Path(__file__).parent / "fixtures" / name


_BOARD = {
    "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "side_to_move": "white",
    "legal_moves": ["e4", "d4", "Nf3"],
}


def _completion(message: dict[str, Any], finish: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "choices": [{"index": 0, "finish_reason": finish, "message": message}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
        },
    )


def _tool_call(name: str, arguments: dict[str, Any]) -> httpx.Response:
    call = {
        "id": "c0",
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }
    return _completion(
        {"role": "assistant", "content": None, "tool_calls": [call]}, "tool_calls"
    )


def _text(content: str) -> httpx.Response:
    return _completion({"role": "assistant", "content": content}, "stop")


class _Server:
    """Answers each call by what it is: a planner call (tools offered) asks for
    the command's one tool until a tool result is in its history, then
    finishes with a note; a call with no tools is the narrator's or the answer
    reader's, and gets words."""

    def __init__(self) -> None:
        self.bodies: list[str] = []
        self.paths: list[str] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.bodies.append(request.content.decode("utf-8"))
        body = json.loads(request.content)
        self.paths.append(request.url.path)
        if "tools" not in body:
            return _text("confirm" if "Their reply" in str(body["messages"]) else "Ok.")
        if any(m["role"] == "tool" for m in body["messages"]):
            return _text("Done.")
        command = body["messages"][-1]["content"]
        if "how am I doing" in command:
            return _tool_call("evaluate_position", {})
        return _tool_call("make_move", {"move": "e4", "source": "said_the_move"})


def _requests(model: str) -> list[str]:
    server = _Server()
    client = httpx.Client(transport=httpx.MockTransport(server))
    ctx = ToolContext(session=GameSession(), engine=FakeEngine())
    registry = build_registry(ctx)
    brain = create_llama_brain(
        base_url="http://llm.test/v1",
        model=model,
        dispatcher=registry,
        tool_definitions=lambda: brain_tool_definitions(registry, ctx),
        provider=LlamaCppProvider("http://llm.test/v1", model, client=client),
    )
    brain.get_agent_response(_BOARD, "play e4")
    brain.get_agent_response(_BOARD, "how am I doing")
    brain.narrate(_BOARD, [{"name": "make_move", "result": {"ok": True, "san": "e4"}}])
    brain.read_answer("Start a new game?", "yes")
    assert set(server.paths) == {"/v1/chat/completions"}
    return server.bodies


def test_the_default_model_is_pinned():
    # A new default must come with its own recording, not ride on another's.
    assert DEFAULT_MODEL in _PINNED


@pytest.mark.parametrize("model", _PINNED)
def test_each_pinned_model_sends_the_recorded_bytes(model):
    sent = _requests(model)
    golden = _golden(model)
    if os.environ.get("CHESSAPP_RECORD_GOLDEN"):
        golden.parent.mkdir(exist_ok=True)
        golden.write_text(json.dumps(sent, indent=1) + "\n")
    recorded = json.loads(golden.read_text())
    assert len(sent) == len(recorded)
    for index, (now, then) in enumerate(zip(sent, recorded, strict=True)):
        assert now == then, f"request {index} changed"


@pytest.mark.parametrize("model", _PINNED)
def test_the_golden_covers_every_phase_and_both_thinking_settings(model):
    # What the pin is worth depends on what it drove: a fixture that never
    # reached a phase would pass any change to it.
    profile = load_profile(model)
    planning = profile.phase(PLANNER)
    recorded = [json.loads(body) for body in json.loads(_golden(model).read_text())]
    planner = [r for r in recorded if "tools" in r]
    words = [r for r in recorded if "tools" not in r]
    # An analysis tool's answer turns thinking on for the words about it.
    assert {r["chat_template_kwargs"]["enable_thinking"] for r in words} == {
        False,
        True,
    }
    assert {r["chat_template_kwargs"]["enable_thinking"] for r in planner} == {
        planning.thinking
    }
    assert {r["max_tokens"] for r in planner} == {planning.max_tokens}
    assert {r["max_tokens"] for r in words} == {4096, 16}
    assert {r["temperature"] for r in planner} == {planning.temperature}
