"""The canonical no-LLM brain double lives in one place (`tests/fakes.py`).

`ScriptedBrain` is the sanctioned stand-in for a real `Brain` across the test
suite — it never touches a live model. It stands in for a *finished* agent
loop: the scripted tool calls are dispatched for real, and the scripted text is
the loop's closing comment. These tests pin its contract so the double stays
trustworthy no matter which test file leans on it.
"""

import pytest

from chessapp.brain import AgentResponse, ToolCall
from chessapp.game import GameSession
from chessapp.tools import ToolContext, build_registry
from fakes import ScriptedBrain


def test_pops_responses_in_order():
    brain = ScriptedBrain(
        AgentResponse(text="first"),
        AgentResponse(text="second"),
    )
    assert brain.get_agent_response({}, "a").text == "first"
    assert brain.get_agent_response({}, "b").text == "second"


def test_records_board_state_and_command():
    brain = ScriptedBrain(AgentResponse(text="ok"))
    brain.get_agent_response({"turn": "white"}, "play e4")
    assert brain.calls == [({"turn": "white"}, "play e4")]


def test_pops_scripted_narrations_in_order():
    brain = ScriptedBrain(narrations=("one", "two"))
    assert brain.narrate({}, []).text == "one"
    assert brain.narrate({}, []).text == "two"


def test_narrate_defaults_when_none_scripted():
    # Tests that don't care about the fast path's commentary needn't script it.
    brain = ScriptedBrain()
    assert brain.narrate({}, []).text == "(commentary)"


def test_records_narrate_board_state_and_changes():
    brain = ScriptedBrain()
    changes = [{"name": "make_move", "result": {"legal": True}}]
    brain.narrate({"turn": "black"}, changes)
    assert brain.narrate_calls == [({"turn": "black"}, changes)]


def test_scripted_tool_calls_are_really_dispatched():
    # The double fakes the model, never the tools: a scripted call runs through
    # the real registry and its real result comes back in tool_results.
    session = GameSession()
    registry = build_registry(ToolContext(session=session))
    brain = ScriptedBrain(
        AgentResponse(
            text="e4.", tool_calls=(ToolCall(name="make_move", args={"move": "e4"}),)
        ),
        dispatcher=registry,
    )
    resp = brain.get_agent_response({}, "play e4")
    assert resp.tool_results[0]["name"] == "make_move"
    assert resp.tool_results[0]["result"]["legal"] is True
    assert session.move_history() == ["e4"]


def test_under_scripting_raises():
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
