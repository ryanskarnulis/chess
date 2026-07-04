"""The llama-server brain: OpenAI-compatible client behind the `Brain` seam.

Exercised only through a fake OpenAI client that returns the exact wire
shapes captured from a live llama.cpp + Gemma-4 run — never a live LLM.
These tests pin the response->AgentResponse mapping and the request the
brain sends (tools, sampling, thinking flag, prompt contents).
"""

from types import SimpleNamespace

from chessapp.brain import AgentResponse
from chessapp.llama_brain import LlamaBrain

# --- fakes -----------------------------------------------------------------

TOOLS = [
    {
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
]


def _tool_call(name: str, arguments: str, call_id: str = "id0"):
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def _completion(
    *, content=None, tool_calls=None, reasoning_content=None, finish_reason="stop"
):
    """Mimic openai's ChatCompletion object shape (attribute access)."""
    message = SimpleNamespace(
        role="assistant",
        content=content,
        tool_calls=tool_calls,
        reasoning_content=reasoning_content,
    )
    return SimpleNamespace(
        choices=[SimpleNamespace(index=0, finish_reason=finish_reason, message=message)]
    )


class FakeClient:
    """Records the create() kwargs; returns a scripted completion."""

    def __init__(self, completion):
        self._completion = completion
        self.calls: list[dict] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return self._completion


def make_brain(completion) -> tuple[LlamaBrain, FakeClient]:
    client = FakeClient(completion)
    brain = LlamaBrain(
        client=client,
        model="gemma-test",
        tool_definitions=TOOLS,
        system_prompt="You are a chess opponent.",
    )
    return brain, client


# --- response mapping ------------------------------------------------------


def test_tool_call_maps_to_agentresponse():
    # Wire shape from live server: content empty, one tool call, args as JSON string.
    brain, _ = make_brain(
        _completion(
            content="",
            tool_calls=[_tool_call("make_move", '{"move":"e2e4"}')],
            finish_reason="tool_calls",
        )
    )
    resp = brain.get_agent_response(board_state={"fen": "startpos"}, command="play e4")
    assert isinstance(resp, AgentResponse)
    assert resp.text == ""
    assert len(resp.tool_calls) == 1
    assert resp.tool_calls[0].name == "make_move"
    assert resp.tool_calls[0].args == {"move": "e2e4"}


def test_plain_content_maps_to_text_no_tools():
    brain, _ = make_brain(_completion(content="I like the Ruy Lopez!", tool_calls=None))
    resp = brain.get_agent_response(board_state={}, command="favorite openings?")
    assert resp.text == "I like the Ruy Lopez!"
    assert resp.tool_calls == ()


def test_multiple_tool_calls_preserved_in_order():
    brain, _ = make_brain(
        _completion(
            content="",
            tool_calls=[
                _tool_call("make_move", '{"move":"e2e4"}', "a"),
                _tool_call("speak", '{"text":"your move"}', "b"),
            ],
            finish_reason="tool_calls",
        )
    )
    resp = brain.get_agent_response(board_state={}, command="play e4 and taunt me")
    assert [tc.name for tc in resp.tool_calls] == ["make_move", "speak"]
    assert resp.tool_calls[1].args == {"text": "your move"}


def test_reasoning_content_is_dropped():
    # Thinking-on responses carry reasoning_content separately; it must never
    # surface as commentary (BRIEF: final answers only, never thought blocks).
    brain, _ = make_brain(
        _completion(
            content="e4 it is.",
            reasoning_content="The king's pawn opening is e2e4...",
            tool_calls=[_tool_call("make_move", '{"move":"e2e4"}')],
            finish_reason="tool_calls",
        )
    )
    resp = brain.get_agent_response(board_state={}, command="play e4")
    assert "king's pawn" not in resp.text
    assert resp.text == "e4 it is."


def test_null_content_becomes_empty_string():
    brain, _ = make_brain(
        _completion(tool_calls=[_tool_call("make_move", '{"move":"e2e4"}')])
    )
    resp = brain.get_agent_response(board_state={}, command="play e4")
    assert resp.text == ""


def test_blank_arguments_become_empty_dict():
    # no-arg tools (new_game, undo) may arrive with "" or "{}" arguments.
    brain, _ = make_brain(
        _completion(tool_calls=[_tool_call("new_game", "")], finish_reason="tool_calls")
    )
    resp = brain.get_agent_response(board_state={}, command="new game")
    assert resp.tool_calls[0].args == {}


# --- request the brain sends ----------------------------------------------


def test_request_carries_tools_sampling_and_thinking_off_by_default():
    brain, client = make_brain(_completion(content="ok"))
    brain.get_agent_response(board_state={"fen": "startpos"}, command="play e4")
    kwargs = client.calls[0]
    assert kwargs["model"] == "gemma-test"
    assert kwargs["tools"] == TOOLS
    assert kwargs["temperature"] == 1.0
    assert kwargs["top_p"] == 0.95
    # top_k and the thinking toggle are non-OpenAI-standard -> extra_body.
    assert kwargs["extra_body"]["top_k"] == 64
    assert kwargs["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False


def test_prompt_includes_system_board_state_and_command():
    brain, client = make_brain(_completion(content="ok"))
    brain.get_agent_response(board_state={"fen": "8/8/8/8"}, command="play Nf3")
    messages = client.calls[0]["messages"]
    assert messages[0]["role"] == "system"
    assert "chess opponent" in messages[0]["content"]
    blob = " ".join(m["content"] for m in messages)
    assert "8/8/8/8" in blob  # board state reached the model
    assert "Nf3" in blob  # command reached the model


def test_thinking_can_be_enabled_for_analysis():
    client = FakeClient(_completion(content="analysis"))
    brain = LlamaBrain(
        client=client,
        model="gemma-test",
        tool_definitions=TOOLS,
        system_prompt="analyst",
        enable_thinking=True,
    )
    brain.get_agent_response(board_state={}, command="was that a blunder?")
    kwargs = client.calls[0]
    assert kwargs["extra_body"]["chat_template_kwargs"]["enable_thinking"] is True
