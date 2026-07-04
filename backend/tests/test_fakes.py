"""The canonical no-LLM brain double lives in one place (`tests/fakes.py`).

`ScriptedBrain` is the sanctioned stand-in for a real `Brain` across the test
suite — it never touches a live model. These tests pin its contract so the
double stays trustworthy no matter which test file leans on it.
"""

import pytest

from chessapp.brain import AgentResponse, ToolCall
from fakes import ScriptedBrain


def test_pops_phase_one_responses_in_order():
    brain = ScriptedBrain(
        AgentResponse(text="first"),
        AgentResponse(text="second"),
    )
    assert brain.get_agent_response({}, "a").text == "first"
    assert brain.get_agent_response({}, "b").text == "second"


def test_records_phase_one_board_state_and_command():
    brain = ScriptedBrain(AgentResponse(text="ok"))
    brain.get_agent_response({"turn": "white"}, "play e4")
    assert brain.calls == [({"turn": "white"}, "play e4")]


def test_pops_scripted_reactions_in_order():
    brain = ScriptedBrain(reactions=("one", "two"))
    assert brain.react({}, []) == "one"
    assert brain.react({}, []) == "two"


def test_react_defaults_when_no_reaction_scripted():
    # Tests that don't care about reaction text shouldn't have to script one.
    brain = ScriptedBrain()
    assert brain.react({}, []) == "(reaction)"


def test_records_react_board_state_and_changes():
    brain = ScriptedBrain()
    changes = [{"name": "make_move", "result": {"legal": True}}]
    brain.react({"turn": "black"}, changes)
    assert brain.react_calls == [({"turn": "black"}, changes)]


def test_under_scripted_phase_one_raises():
    # Calling more than was scripted is a test bug — surface it loudly rather
    # than returning a stale or silent placeholder.
    brain = ScriptedBrain(AgentResponse(text="only one"))
    brain.get_agent_response({}, "a")
    with pytest.raises(IndexError):
        brain.get_agent_response({}, "b")


def test_carries_tool_calls_through_untouched():
    call = ToolCall(name="make_move", args={"move": "e4"})
    brain = ScriptedBrain(AgentResponse(text="on it", tool_calls=(call,)))
    assert brain.get_agent_response({}, "play e4").tool_calls == (call,)
