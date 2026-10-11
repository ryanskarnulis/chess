"""The mover sends the recorded bytes (#472): nothing leaks in later.

Decision 2 of docs/glitch-difficulty.md says the mover sees the board, the
game's moves, its side and the legal moves, and nothing else. This drives the
mover through a real `LlamaCppProvider` over a faked llama-server on fixed
positions, for every pinned model, and compares the exact request bodies with
the recorded ones (`fixtures/<model>_mover_requests.json`). A conversation, an
evaluation or a note that found its way into the request would change the
bytes and fail here.

The positions: Black's first move after 1.e4; a game from a set-up position
with a few moves played; and a retry, where the first answer is illegal and
the second request carries the refusal.

Re-record only for a change that is *meant* to move the bytes, and say so in
its PR: `CHESSAPP_RECORD_GOLDEN=1 pytest tests/test_mover_bytes.py`.
"""

import json
import os
from pathlib import Path
from typing import Any

import httpx
import pytest

from chessapp.brain import PHASE_MOVER
from chessapp.context_capture import JsonlContextCapture
from chessapp.game import GameSession
from chessapp.mover import PLAY_MOVE, Mover
from chessapp.profiles import MOVER, load_profile
from chessapp.provider import LlamaCppProvider
from test_profile_bytes import _PINNED

_BASE = "http://llm.test/v1"


def _golden(model: str) -> Path:
    name = model.replace("-", "_") + "_mover_requests.json"
    return Path(__file__).parent / "fixtures" / name


class _Server:
    """Plays the scripted moves in order, one per completion, and renders a
    template for the context capture."""

    def __init__(self, *moves: str) -> None:
        self.moves = list(moves)
        self.bodies: list[str] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/apply-template"):
            return httpx.Response(200, json={"prompt": "<rendered>"})
        self.bodies.append(request.content.decode("utf-8"))
        move = self.moves.pop(0)
        call = {
            "id": f"c{len(self.bodies)}",
            "type": "function",
            "function": {
                "name": PLAY_MOVE,
                "arguments": json.dumps({"move": move, "intent": "a plan"}),
            },
        }
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [call],
                        },
                    }
                ],
                "usage": {"prompt_tokens": 9, "completion_tokens": 3},
            },
        )


def _choose(model: str, session: GameSession, *moves: str, capture=None) -> list[str]:
    server = _Server(*moves)
    client = httpx.Client(transport=httpx.MockTransport(server))
    provider = LlamaCppProvider(_BASE, model, client=client, capture=capture)
    Mover(provider, load_profile(model)).choose(session)
    return server.bodies


def _after_e4() -> GameSession:
    session = GameSession()
    session.submit_move("e4")
    return session


def _set_up() -> GameSession:
    session = GameSession(
        fen="r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 2 3"
    )
    for move in ("Nf6", "Bc4", "Bc5", "O-O"):
        session.submit_move(move)
    return session


def _requests(model: str) -> list[str]:
    return [
        *_choose(model, _after_e4(), "e5"),
        *_choose(model, _set_up(), "O-O"),
        *_choose(model, _after_e4(), "Nf3", "Nf6"),
    ]


@pytest.mark.parametrize("model", _PINNED)
def test_each_pinned_model_sends_the_recorded_mover_bytes(model):
    sent = _requests(model)
    golden = _golden(model)
    if os.environ.get("CHESSAPP_RECORD_GOLDEN"):
        golden.parent.mkdir(exist_ok=True)
        golden.write_text(json.dumps(sent, indent=1) + "\n")
    recorded = json.loads(golden.read_text())
    assert len(sent) == len(recorded) == 4
    for index, (now, then) in enumerate(zip(sent, recorded, strict=True)):
        assert now == then, f"mover request {index} changed"


@pytest.mark.parametrize("model", _PINNED)
def test_the_recording_carries_the_mover_phase_settings(model):
    # What the pin is worth depends on what it drove.
    settings = load_profile(model).phase(MOVER)
    recorded = [json.loads(body) for body in json.loads(_golden(model).read_text())]
    for body in recorded:
        assert body["chat_template_kwargs"]["enable_thinking"] is settings.thinking
        assert body["max_tokens"] == settings.max_tokens
        assert [tool["function"]["name"] for tool in body["tools"]] == [PLAY_MOVE]
    # The retry carries the refusal, and nothing that was thought.
    retry = recorded[-1]["messages"]
    assert retry[-1]["role"] == "tool"
    assert "not a legal move" in retry[-1]["content"]
    assert all("reasoning_content" not in message for message in retry)


def test_a_mover_call_is_captured_under_its_phase(tmp_path):
    path = tmp_path / "context.jsonl"
    _choose(
        "gemma-4-26b-a4b",
        _after_e4(),
        "e5",
        capture=JsonlContextCapture(path),
    )
    records: list[dict[str, Any]] = [
        json.loads(line) for line in path.read_text().splitlines()
    ]
    assert [record["phase"] for record in records] == [PHASE_MOVER]
    assert records[0]["template"] == {"prompt": "<rendered>"}
