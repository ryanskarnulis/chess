"""The llama-server brain over a `ChatProvider`, behind the `Brain` seam.

Exercised only through a `ScriptedProvider` (tests/fakes.py) that returns canned
`ChatResult`s and records the `chat()` requests — never a live LLM. These tests
pin the brain's orchestration: the ChatResult->AgentResponse mapping, the
request it builds (tools offered, thinking flag, prompt contents), and the
schema-validation + self-correction retry loop. The wire itself (sampling,
top_k, chat_template_kwargs, tool-arg JSON parsing, reasoning_content drop) is
pinned one layer down in test_provider.py.
"""

import pytest

from chessapp.brain import AgentResponse
from chessapp.llama_brain import LlamaBrain, create_llama_brain
from chessapp.personality import system_prompt_for
from chessapp.provider import ToolCallArgumentsError
from fakes import ScriptedProvider, text_turn, tool_calls_turn

# --- tool definitions the brain validates against --------------------------


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


def make_brain(*turns, **kwargs) -> tuple[LlamaBrain, ScriptedProvider]:
    """A brain wired to a ScriptedProvider playing `turns` in order.

    One turn is returned for every call; several are returned in order and the
    last repeats (so a retry loop scripts as "bad then good"). A turn may be an
    Exception to raise, modelling a provider-level failure.
    """
    provider = ScriptedProvider(*turns)
    brain = LlamaBrain(
        provider=provider,
        tool_definitions=TOOLS,
        system_prompt="You are a chess opponent.",
        **kwargs,
    )
    return brain, provider


# --- response mapping ------------------------------------------------------


def test_tool_call_maps_to_agentresponse():
    brain, _ = make_brain(tool_calls_turn(("make_move", {"move": "e2e4"}), content=""))
    resp = brain.get_agent_response(board_state={"fen": "startpos"}, command="play e4")
    assert isinstance(resp, AgentResponse)
    assert resp.text == ""
    assert len(resp.tool_calls) == 1
    assert resp.tool_calls[0].name == "make_move"
    assert resp.tool_calls[0].args == {"move": "e2e4"}


def test_plain_content_maps_to_text_no_tools():
    brain, _ = make_brain(text_turn("I like the Ruy Lopez!"))
    resp = brain.get_agent_response(board_state={}, command="favorite openings?")
    assert resp.text == "I like the Ruy Lopez!"
    assert resp.tool_calls == ()


def test_multiple_tool_calls_preserved_in_order():
    brain, _ = make_brain(
        tool_calls_turn(
            ("make_move", {"move": "e2e4"}),
            ("speak", {"text": "your move"}),
            content="",
        )
    )
    resp = brain.get_agent_response(board_state={}, command="play e4 and taunt me")
    assert [tc.name for tc in resp.tool_calls] == ["make_move", "speak"]
    assert resp.tool_calls[1].args == {"text": "your move"}


def test_null_content_becomes_empty_string():
    brain, _ = make_brain(tool_calls_turn(("make_move", {"move": "e2e4"})))
    resp = brain.get_agent_response(board_state={}, command="play e4")
    assert resp.text == ""


def test_empty_args_pass_through_as_empty_dict():
    # No-arg tools (new_game, undo) arrive with {} args (the provider already
    # turned a "" wire string into {} — pinned in test_provider).
    brain, _ = make_brain(tool_calls_turn(("new_game", {})))
    resp = brain.get_agent_response(board_state={}, command="new game")
    assert resp.tool_calls[0].args == {}


# --- request the brain sends ----------------------------------------------


def test_request_offers_tools_with_thinking_off_by_default():
    # Sampling (temperature/top_p/top_k, model) is the provider's job now and
    # is pinned in test_provider; the brain owns which tools are offered and
    # whether the thinking channel is on.
    brain, provider = make_brain(text_turn("ok"))
    brain.get_agent_response(board_state={"fen": "startpos"}, command="play e4")
    call = provider.calls[0]
    assert call["tools"] == TOOLS
    assert call["enable_thinking"] is False


def test_prompt_includes_system_board_state_and_command():
    brain, provider = make_brain(text_turn("ok"))
    brain.get_agent_response(board_state={"fen": "8/8/8/8"}, command="play Nf3")
    messages = provider.calls[0]["messages"]
    assert messages[0]["role"] == "system"
    assert "chess opponent" in messages[0]["content"]
    blob = " ".join(m["content"] for m in messages)
    assert "8/8/8/8" in blob  # board state reached the model
    assert "Nf3" in blob  # command reached the model


def test_thinking_can_be_enabled_for_analysis():
    brain, provider = make_brain(text_turn("analysis"), enable_thinking=True)
    brain.get_agent_response(board_state={}, command="was that a blunder?")
    assert provider.calls[0]["enable_thinking"] is True


# --- conversation transcript ------------------------------------------------

TRANSCRIPT = [
    {"role": "user", "content": "play e4"},
    {"role": "assistant", "content": "e4 — the classic."},
]


def test_transcript_sits_between_system_and_current_command():
    brain, provider = make_brain(text_turn("ok"))
    brain.get_agent_response(
        board_state={"fen": "8/8/8/8"}, command="play Nf3", transcript=TRANSCRIPT
    )
    messages = provider.calls[0]["messages"]
    assert messages[0]["role"] == "system"
    assert messages[1:3] == TRANSCRIPT
    assert messages[-1]["role"] == "user"
    assert "Nf3" in messages[-1]["content"]


def test_transcript_defaults_to_empty():
    brain, provider = make_brain(text_turn("ok"))
    brain.get_agent_response(board_state={}, command="hi")
    assert len(provider.calls[0]["messages"]) == 2  # system + current turn only


def test_react_includes_the_transcript():
    brain, provider = make_brain(text_turn("ok"))
    brain.react(
        board_state={"fen": "8/8/8/8"},
        changes=[{"name": "make_move", "result": {"san": "Nf3"}}],
        transcript=TRANSCRIPT,
    )
    messages = provider.calls[0]["messages"]
    assert messages[0]["role"] == "system"
    assert messages[1:3] == TRANSCRIPT
    assert "Nf3" in messages[-1]["content"]


def test_retry_correction_lands_after_the_transcript():
    brain, provider = make_brain(_bad_json_call(), _good_move())
    brain.get_agent_response(board_state={}, command="play e4", transcript=TRANSCRIPT)
    retry_messages = provider.calls[1]["messages"]
    assert retry_messages[1:3] == TRANSCRIPT  # history preserved on retry
    assert "make_move" in retry_messages[-1]["content"]


# --- reaction step (game-loop phase two) ----------------------------------


def test_react_returns_commentary_text():
    brain, _ = make_brain(text_turn("Nice, e4! The classic."))
    text = brain.react(
        board_state={"fen": "after-e4", "turn": "black"},
        changes=[{"name": "make_move", "result": {"legal": True, "san": "e4"}}],
    )
    assert text == "Nice, e4! The classic."


def test_react_prompt_shows_new_state_and_changes_not_a_command():
    brain, provider = make_brain(text_turn("ok"))
    brain.react(
        board_state={"fen": "8/8/8/8", "turn": "black"},
        changes=[{"name": "make_move", "result": {"san": "Nf3"}}],
    )
    messages = provider.calls[0]["messages"]
    assert messages[0]["role"] == "system"
    blob = " ".join(m["content"] for m in messages)
    assert "8/8/8/8" in blob  # new board reached the model
    assert "Nf3" in blob  # what changed reached the model


def test_react_sends_no_tools():
    # The reaction is commentary only; it must not be able to act again.
    brain, provider = make_brain(text_turn("ok"))
    brain.react(board_state={}, changes=[])
    assert provider.calls[0]["tools"] is None


def test_react_null_content_becomes_empty_string():
    brain, _ = make_brain(text_turn(None))
    assert brain.react(board_state={}, changes=[]) == ""


# --- thinking mode: ON for analysis, OFF for speed -------------------------


@pytest.mark.parametrize(
    "tool", ["evaluate_position", "get_best_moves", "analyze_last_move"]
)
def test_react_to_analysis_results_thinks(tool):
    # BRIEF: thinking OFF for fast reactions, ON for analysis. Reacting to
    # analysis-tool results is exactly the analysis case.
    brain, provider = make_brain(text_turn("deep thoughts"))
    brain.react(board_state={}, changes=[{"name": tool, "result": {"ok": True}}])
    assert provider.calls[0]["enable_thinking"] is True


def test_react_to_a_plain_move_does_not_think():
    brain, provider = make_brain(text_turn("nice move"))
    brain.react(
        board_state={},
        changes=[{"name": "make_move", "result": {"legal": True}}],
    )
    assert provider.calls[0]["enable_thinking"] is False


def test_react_mixed_changes_with_any_analysis_thinks():
    brain, provider = make_brain(text_turn("ok"))
    brain.react(
        board_state={},
        changes=[
            {"name": "make_move", "result": {"legal": True}},
            {"name": "evaluate_position", "result": {"ok": True, "score_cp": 30}},
        ],
    )
    assert provider.calls[0]["enable_thinking"] is True


# --- factory wires the system prompt ---------------------------------------


def test_create_llama_brain_defaults_to_the_glitch_prompt():
    # The brain is model-specific but personality-agnostic: without a provider
    # the factory resolves the one system prompt once and carries the string.
    brain = create_llama_brain(
        base_url="http://localhost:8200/v1",
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
    provider = ScriptedProvider(text_turn("a"), text_turn("b"))
    brain = LlamaBrain(
        provider=provider,
        tool_definitions=TOOLS,
        system_prompt=lambda: system_prompt_for(verbosity=live["verbosity"]),
    )
    brain.get_agent_response(board_state={}, command="hi")
    assert provider.calls[0]["messages"][0]["content"] == system_prompt_for()
    live["verbosity"] = "low"
    brain.get_agent_response(board_state={}, command="hi again")
    assert provider.calls[1]["messages"][0]["content"] == system_prompt_for(
        verbosity="low"
    )


def test_callable_system_prompt_also_drives_the_reaction():
    # The reaction (phase two) must use the same live prompt as phase one.
    provider = ScriptedProvider(text_turn("ok"))
    brain = LlamaBrain(
        provider=provider,
        tool_definitions=TOOLS,
        system_prompt=lambda: system_prompt_for(verbosity="low"),
    )
    brain.react(board_state={}, changes=[])
    assert provider.calls[0]["messages"][0]["content"] == system_prompt_for(
        verbosity="low"
    )


def test_create_llama_brain_accepts_a_live_prompt_provider():
    from chessapp.tools import Settings

    settings = Settings()  # verbosity defaults to normal
    brain = create_llama_brain(
        base_url="http://localhost:8200/v1",
        model="gemma",
        tool_definitions=[],
        system_prompt_provider=lambda: system_prompt_for(settings.verbosity),
        provider=ScriptedProvider(text_turn("ok")),
    )
    assert brain.system_prompt() == system_prompt_for()
    settings.verbosity = "low"
    assert brain.system_prompt() == system_prompt_for(verbosity="low")


def test_create_llama_brain_accepts_an_injected_provider():
    # The provider can be injected (tests, alternate backends) instead of the
    # factory building a real LlamaCppProvider against base_url.
    fake = ScriptedProvider(text_turn("ok"))
    brain = create_llama_brain(
        base_url="http://localhost:8200/v1",
        model="gemma",
        tool_definitions=[],
        provider=fake,
    )
    assert brain.provider is fake


# --- defensive parse + retry loop -----------------------------------------


def _bad_json_call():
    # A quant hiccup: the provider could not parse the tool-call arguments as
    # JSON, so it raised before returning a result. The brain must treat this
    # as a correctable failure, not a crash — the error names the tool.
    return ToolCallArgumentsError(
        "make_move", "arguments are not valid JSON (Expecting value)"
    )


def _good_move():
    return tool_calls_turn(("make_move", {"move": "e2e4"}))


def test_valid_first_response_makes_one_call():
    brain, provider = make_brain(_good_move())
    resp = brain.get_agent_response(board_state={}, command="play e4")
    assert resp.tool_calls[0].args == {"move": "e2e4"}
    assert len(provider.calls) == 1  # no wasted retry on a clean call


def test_malformed_json_args_retries_then_succeeds():
    brain, provider = make_brain(_bad_json_call(), _good_move())
    resp = brain.get_agent_response(board_state={}, command="play e4")
    assert len(provider.calls) == 2
    assert resp.tool_calls[0].args == {"move": "e2e4"}


def test_unknown_tool_name_retries():
    brain, provider = make_brain(
        tool_calls_turn(("teleport_king", {})),
        _good_move(),
    )
    resp = brain.get_agent_response(board_state={}, command="play e4")
    assert len(provider.calls) == 2
    assert [tc.name for tc in resp.tool_calls] == ["make_move"]


def test_schema_violation_retries():
    # make_move wants a string; the model sent a number.
    brain, provider = make_brain(
        tool_calls_turn(("make_move", {"move": 5})),
        _good_move(),
    )
    resp = brain.get_agent_response(board_state={}, command="play e4")
    assert len(provider.calls) == 2
    assert resp.tool_calls[0].args == {"move": "e2e4"}


def test_retry_feeds_the_error_back_to_the_model():
    brain, provider = make_brain(_bad_json_call(), _good_move())
    brain.get_agent_response(board_state={}, command="play e4")
    # The retry's prompt must have grown a correction turn naming the offender.
    retry_messages = provider.calls[1]["messages"]
    assert len(retry_messages) > len(provider.calls[0]["messages"])
    correction = retry_messages[-1]["content"]
    assert "make_move" in correction


def test_retries_are_bounded_and_drop_invalid_calls():
    # Every attempt raises the provider's parse error; brain gives up cleanly.
    brain, provider = make_brain(_bad_json_call(), max_retries=2)
    resp = brain.get_agent_response(board_state={}, command="play e4")
    assert len(provider.calls) == 3  # initial + 2 retries
    assert resp.tool_calls == ()  # invalid calls dropped, no crash


def test_exhaustion_keeps_valid_calls_and_text():
    # One good call + one that survives parsing but violates the schema
    # ({"move": 5}). On exhaustion the valid call and the text stay; the
    # schema-invalid call is dropped. (A *malformed-JSON* call is all-or-
    # nothing at the provider, so this partial-validity case is exercised with
    # a schema violation, the failure that still returns a result.)
    both = tool_calls_turn(
        ("make_move", {"move": "e2e4"}),
        ("make_move", {"move": 5}),
        content="here you go",
    )
    brain, provider = make_brain(both, max_retries=1)
    resp = brain.get_agent_response(board_state={}, command="play e4")
    assert len(provider.calls) == 2
    assert resp.text == "here you go"
    assert [tc.args for tc in resp.tool_calls] == [{"move": "e2e4"}]
