"""The llama-server brain: OpenAI-compatible client behind the `Brain` seam.

Exercised only through a fake OpenAI client that returns the exact wire
shapes captured from a live llama.cpp + Gemma-4 run — never a live LLM.
These tests pin the response->AgentResponse mapping and the request the
brain sends (tools, sampling, thinking flag, prompt contents).
"""

from types import SimpleNamespace

import pytest

from chessapp.brain import AgentResponse
from chessapp.llama_brain import LlamaBrain, create_llama_brain
from chessapp.personality import system_prompt_for

# --- fakes -----------------------------------------------------------------


def _fn(name, description, parameters):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters,
        },
    }


TOOLS = [
    _fn(
        "make_move",
        "Submit a move in SAN or UCI.",
        {
            "type": "object",
            "properties": {"move": {"type": "string"}},
            "required": ["move"],
            "additionalProperties": False,
        },
    ),
    _fn(
        "speak",
        "Say something aloud.",
        {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    ),
    _fn(
        "new_game",
        "Start a new game.",
        {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    ),
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
    """Records create() kwargs; returns scripted completions in sequence.

    Given one completion it returns it for every call; given several it pops
    them per call (repeating the last) so a retry loop can be scripted as
    "bad then good".
    """

    def __init__(self, *completions):
        self._completions = list(completions)
        self.calls: list[dict] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        i = min(len(self.calls), len(self._completions) - 1)
        self.calls.append(kwargs)
        return self._completions[i]


def make_brain(*completions, **kwargs) -> tuple[LlamaBrain, FakeClient]:
    client = FakeClient(*completions)
    brain = LlamaBrain(
        client=client,
        model="gemma-test",
        tool_definitions=TOOLS,
        system_prompt="You are a chess opponent.",
        **kwargs,
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


# --- conversation transcript ------------------------------------------------

TRANSCRIPT = [
    {"role": "user", "content": "play e4"},
    {"role": "assistant", "content": "e4 — the classic."},
]


def test_transcript_sits_between_system_and_current_command():
    brain, client = make_brain(_completion(content="ok"))
    brain.get_agent_response(
        board_state={"fen": "8/8/8/8"}, command="play Nf3", transcript=TRANSCRIPT
    )
    messages = client.calls[0]["messages"]
    assert messages[0]["role"] == "system"
    assert messages[1:3] == TRANSCRIPT
    assert messages[-1]["role"] == "user"
    assert "Nf3" in messages[-1]["content"]


def test_transcript_defaults_to_empty():
    brain, client = make_brain(_completion(content="ok"))
    brain.get_agent_response(board_state={}, command="hi")
    assert len(client.calls[0]["messages"]) == 2  # system + current turn only


def test_react_includes_the_transcript():
    brain, client = make_brain(_completion(content="ok"))
    brain.react(
        board_state={"fen": "8/8/8/8"},
        changes=[{"name": "make_move", "result": {"san": "Nf3"}}],
        transcript=TRANSCRIPT,
    )
    messages = client.calls[0]["messages"]
    assert messages[0]["role"] == "system"
    assert messages[1:3] == TRANSCRIPT
    assert "Nf3" in messages[-1]["content"]


def test_retry_correction_lands_after_the_transcript():
    brain, client = make_brain(_bad_json_call(), _good_move())
    brain.get_agent_response(board_state={}, command="play e4", transcript=TRANSCRIPT)
    retry_messages = client.calls[1]["messages"]
    assert retry_messages[1:3] == TRANSCRIPT  # history preserved on retry
    assert "make_move" in retry_messages[-1]["content"]


# --- reaction step (game-loop phase two) ----------------------------------


def test_react_returns_commentary_text():
    brain, _ = make_brain(_completion(content="Nice, e4! The classic."))
    text = brain.react(
        board_state={"fen": "after-e4", "turn": "black"},
        changes=[{"name": "make_move", "result": {"legal": True, "san": "e4"}}],
    )
    assert text == "Nice, e4! The classic."


def test_react_prompt_shows_new_state_and_changes_not_a_command():
    brain, client = make_brain(_completion(content="ok"))
    brain.react(
        board_state={"fen": "8/8/8/8", "turn": "black"},
        changes=[{"name": "make_move", "result": {"san": "Nf3"}}],
    )
    messages = client.calls[0]["messages"]
    assert messages[0]["role"] == "system"
    blob = " ".join(m["content"] for m in messages)
    assert "8/8/8/8" in blob  # new board reached the model
    assert "Nf3" in blob  # what changed reached the model


def test_react_sends_no_tools():
    # The reaction is commentary only; it must not be able to act again.
    brain, client = make_brain(_completion(content="ok"))
    brain.react(board_state={}, changes=[])
    assert "tools" not in client.calls[0]


def test_react_drops_reasoning_content():
    brain, _ = make_brain(
        _completion(
            content="Solid.",
            reasoning_content="Black is now slightly worse because...",
        )
    )
    text = brain.react(board_state={}, changes=[])
    assert text == "Solid."
    assert "slightly worse" not in text


def test_react_null_content_becomes_empty_string():
    brain, _ = make_brain(_completion(content=None))
    assert brain.react(board_state={}, changes=[]) == ""


# --- thinking mode: ON for analysis, OFF for speed -------------------------


@pytest.mark.parametrize(
    "tool", ["evaluate_position", "get_best_moves", "analyze_last_move"]
)
def test_react_to_analysis_results_thinks(tool):
    # BRIEF: thinking OFF for fast reactions, ON for analysis. Reacting to
    # analysis-tool results is exactly the analysis case.
    brain, client = make_brain(_completion(content="deep thoughts"))
    brain.react(board_state={}, changes=[{"name": tool, "result": {"ok": True}}])
    kwargs = client.calls[0]
    assert kwargs["extra_body"]["chat_template_kwargs"]["enable_thinking"] is True


def test_react_to_a_plain_move_does_not_think():
    brain, client = make_brain(_completion(content="nice move"))
    brain.react(
        board_state={},
        changes=[{"name": "make_move", "result": {"legal": True}}],
    )
    kwargs = client.calls[0]
    assert kwargs["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False


def test_react_mixed_changes_with_any_analysis_thinks():
    brain, client = make_brain(_completion(content="ok"))
    brain.react(
        board_state={},
        changes=[
            {"name": "make_move", "result": {"legal": True}},
            {"name": "evaluate_position", "result": {"ok": True, "score_cp": 30}},
        ],
    )
    kwargs = client.calls[0]
    assert kwargs["extra_body"]["chat_template_kwargs"]["enable_thinking"] is True


# --- factory wires the system prompt ---------------------------------------


def test_create_llama_brain_defaults_to_the_glitch_prompt():
    # The brain is model-specific but personality-agnostic: without a provider
    # the factory resolves the one system prompt once and carries the string.
    brain = create_llama_brain(
        base_url="http://localhost:8080/v1",
        model="gemma",
        tool_definitions=[],
    )
    assert brain.system_prompt == system_prompt_for()


# --- live prompt switching --------------------------------------------------


def test_callable_system_prompt_is_resolved_per_request():
    # Live switching: the brain may carry a provider instead of a frozen
    # string, so a settings change (verbosity/hints) between commands takes
    # effect on the very next command — the prompt is resolved fresh each
    # request.
    live = {"verbosity": "normal"}
    client = FakeClient(_completion(content="a"), _completion(content="b"))
    brain = LlamaBrain(
        client=client,
        model="m",
        tool_definitions=TOOLS,
        system_prompt=lambda: system_prompt_for(verbosity=live["verbosity"]),
    )
    brain.get_agent_response(board_state={}, command="hi")
    assert client.calls[0]["messages"][0]["content"] == system_prompt_for()
    live["verbosity"] = "low"
    brain.get_agent_response(board_state={}, command="hi again")
    assert client.calls[1]["messages"][0]["content"] == system_prompt_for(
        verbosity="low"
    )


def test_callable_system_prompt_also_drives_the_reaction():
    # The reaction (phase two) must use the same live prompt as phase one.
    client = FakeClient(_completion(content="ok"))
    brain = LlamaBrain(
        client=client,
        model="m",
        tool_definitions=TOOLS,
        system_prompt=lambda: system_prompt_for(verbosity="low"),
    )
    brain.react(board_state={}, changes=[])
    assert client.calls[0]["messages"][0]["content"] == system_prompt_for(
        verbosity="low"
    )


def test_create_llama_brain_accepts_a_live_prompt_provider():
    from chessapp.tools import Settings

    settings = Settings()  # verbosity defaults to normal
    brain = create_llama_brain(
        base_url="http://localhost:8080/v1",
        model="gemma",
        tool_definitions=[],
        system_prompt_provider=lambda: system_prompt_for(settings.verbosity),
        client=FakeClient(_completion(content="ok")),
    )
    assert brain.system_prompt() == system_prompt_for()
    settings.verbosity = "low"
    assert brain.system_prompt() == system_prompt_for(verbosity="low")


def test_create_llama_brain_accepts_an_injected_client():
    # The OpenAI client can be injected (tests, alternate backends) instead of
    # the factory building a real one against base_url.
    fake = FakeClient(_completion(content="ok"))
    brain = create_llama_brain(
        base_url="http://localhost:8080/v1",
        model="gemma",
        tool_definitions=[],
        client=fake,
    )
    assert brain.client is fake


# --- defensive parse + retry loop -----------------------------------------


def _bad_json_call():
    # arguments is not valid JSON (a quant hiccup): unquoted value.
    return _completion(
        tool_calls=[_tool_call("make_move", '{"move": e2e4}')],
        finish_reason="tool_calls",
    )


def _good_move():
    return _completion(
        tool_calls=[_tool_call("make_move", '{"move":"e2e4"}')],
        finish_reason="tool_calls",
    )


def test_valid_first_response_makes_one_call():
    brain, client = make_brain(_good_move())
    resp = brain.get_agent_response(board_state={}, command="play e4")
    assert resp.tool_calls[0].args == {"move": "e2e4"}
    assert len(client.calls) == 1  # no wasted retry on a clean call


def test_malformed_json_args_retries_then_succeeds():
    brain, client = make_brain(_bad_json_call(), _good_move())
    resp = brain.get_agent_response(board_state={}, command="play e4")
    assert len(client.calls) == 2
    assert resp.tool_calls[0].args == {"move": "e2e4"}


def test_unknown_tool_name_retries():
    brain, client = make_brain(
        _completion(
            tool_calls=[_tool_call("teleport_king", "{}")], finish_reason="tool_calls"
        ),
        _good_move(),
    )
    resp = brain.get_agent_response(board_state={}, command="play e4")
    assert len(client.calls) == 2
    assert [tc.name for tc in resp.tool_calls] == ["make_move"]


def test_schema_violation_retries():
    # make_move wants a string; the model sent a number.
    brain, client = make_brain(
        _completion(
            tool_calls=[_tool_call("make_move", '{"move": 5}')],
            finish_reason="tool_calls",
        ),
        _good_move(),
    )
    resp = brain.get_agent_response(board_state={}, command="play e4")
    assert len(client.calls) == 2
    assert resp.tool_calls[0].args == {"move": "e2e4"}


def test_retry_feeds_the_error_back_to_the_model():
    brain, client = make_brain(_bad_json_call(), _good_move())
    brain.get_agent_response(board_state={}, command="play e4")
    # The retry's prompt must have grown a correction turn naming the offender.
    retry_messages = client.calls[1]["messages"]
    assert len(retry_messages) > len(client.calls[0]["messages"])
    correction = retry_messages[-1]["content"]
    assert "make_move" in correction


def test_retries_are_bounded_and_drop_invalid_calls():
    # Every attempt is malformed; brain gives up without crashing.
    brain, client = make_brain(_bad_json_call(), max_retries=2)
    resp = brain.get_agent_response(board_state={}, command="play e4")
    assert len(client.calls) == 3  # initial + 2 retries
    assert resp.tool_calls == ()  # invalid calls dropped, no crash


def test_exhaustion_keeps_valid_calls_and_text():
    both = _completion(
        content="here you go",
        tool_calls=[
            _tool_call("make_move", '{"move":"e2e4"}', "ok"),
            _tool_call("make_move", "{bad", "bad"),
        ],
        finish_reason="tool_calls",
    )
    brain, client = make_brain(both, max_retries=1)
    resp = brain.get_agent_response(board_state={}, command="play e4")
    assert len(client.calls) == 2
    assert resp.text == "here you go"
    assert [tc.args for tc in resp.tool_calls] == [{"move": "e2e4"}]
