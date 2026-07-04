"""Text command endpoint: user string in → brain → tool calls → new state.

The brain is exercised only through its interface with a scripted fake —
never a live LLM. The endpoint dispatches whatever tool calls the brain
returns through the validated registry, so a misbehaving brain surfaces
as error results, never as corrupted state or a 500.
"""

from fastapi.testclient import TestClient

from chessapp.api import create_app
from chessapp.brain import AgentResponse, ToolCall
from chessapp.game import GameSession
from chessapp.tools import ToolContext

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


class FakeBrain:
    """Scripted brain: pops canned responses in order, records prompts."""

    def __init__(self, *responses: AgentResponse):
        self._responses = list(responses)
        self.calls: list[tuple[dict, str]] = []

    def get_agent_response(self, board_state: dict, command: str) -> AgentResponse:
        self.calls.append((board_state, command))
        return self._responses.pop(0)


def make_client(*responses: AgentResponse, brain: FakeBrain | None = None):
    brain = brain if brain is not None else FakeBrain(*responses)
    ctx = ToolContext(session=GameSession())
    return TestClient(create_app(ctx, brain=brain)), brain


def test_no_brain_is_503():
    client = TestClient(create_app(ToolContext(session=GameSession())))
    response = client.post("/api/command", json={"text": "play e4"})
    assert response.status_code == 503


def test_missing_text_is_422():
    client, _ = make_client(AgentResponse(text="hi"))
    assert client.post("/api/command", json={}).status_code == 422


def test_brain_receives_board_state_and_command():
    client, brain = make_client(AgentResponse(text="hello"))
    client.post("/api/command", json={"text": "how does it look?"})
    board_state, command = brain.calls[0]
    assert command == "how does it look?"
    assert board_state["fen"] == START_FEN
    assert board_state["turn"] == "white"


def test_command_with_no_tool_calls_returns_commentary_only():
    client, _ = make_client(AgentResponse(text="Which knight did you mean?"))
    body = client.post("/api/command", json={"text": "move the knight"}).json()
    assert body["commentary"] == "Which knight did you mean?"
    assert body["tool_results"] == []
    assert body["state"]["fen"] == START_FEN


def test_command_tool_calls_mutate_state_through_registry():
    client, _ = make_client(
        AgentResponse(
            text="e4, the classic.",
            tool_calls=(ToolCall(name="make_move", args={"move": "e4"}),),
        )
    )
    body = client.post("/api/command", json={"text": "play e4"}).json()
    assert body["commentary"] == "e4, the classic."
    assert body["tool_results"][0]["name"] == "make_move"
    assert body["tool_results"][0]["result"]["legal"] is True
    assert body["state"]["history"] == ["e4"]
    assert body["state"]["turn"] == "black"


def test_multiple_tool_calls_run_in_order():
    client, _ = make_client(
        AgentResponse(
            text="Fresh board, and I checked your options.",
            tool_calls=(
                ToolCall(name="new_game", args={}),
                ToolCall(name="get_legal_moves", args={}),
            ),
        )
    )
    body = client.post("/api/command", json={"text": "start over"}).json()
    names = [r["name"] for r in body["tool_results"]]
    assert names == ["new_game", "get_legal_moves"]
    assert "e4" in body["tool_results"][1]["result"]["moves"]


def test_unknown_tool_from_brain_is_error_result_not_500():
    client, _ = make_client(
        AgentResponse(
            text="doing something odd",
            tool_calls=(ToolCall(name="launch_rocket", args={}),),
        )
    )
    response = client.post("/api/command", json={"text": "do it"})
    assert response.status_code == 200
    result = response.json()["tool_results"][0]["result"]
    assert result["ok"] is False
    assert "unknown tool" in result["error"]


def test_invalid_args_from_brain_is_error_result_not_500():
    client, _ = make_client(
        AgentResponse(
            text="moving",
            tool_calls=(ToolCall(name="make_move", args={"move": 42}),),
        )
    )
    response = client.post("/api/command", json={"text": "play something"})
    assert response.status_code == 200
    assert response.json()["tool_results"][0]["result"]["ok"] is False


def test_command_mutation_broadcasts_state_to_ws():
    client, _ = make_client(
        AgentResponse(
            text="done",
            tool_calls=(ToolCall(name="make_move", args={"move": "e4"}),),
        )
    )
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()  # connect snapshot
        client.post("/api/command", json={"text": "play e4"})
        message = ws.receive_json()
    assert message["state"]["history"] == ["e4"]


def test_read_only_command_does_not_broadcast():
    client, _ = make_client(
        AgentResponse(
            text="You have 20 moves.",
            tool_calls=(ToolCall(name="get_legal_moves", args={}),),
        ),
        AgentResponse(
            text="done",
            tool_calls=(ToolCall(name="make_move", args={"move": "e4"}),),
        ),
    )
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()
        client.post("/api/command", json={"text": "what are my options?"})
        client.post("/api/command", json={"text": "play e4"})
        message = ws.receive_json()  # first broadcast is the move, not the read
    assert message["state"]["history"] == ["e4"]
