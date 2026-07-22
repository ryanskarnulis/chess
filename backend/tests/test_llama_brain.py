"""The llama-server brain over a `ChatProvider`, behind the `Brain` seam.

Exercised only through a `ScriptedProvider` (tests/fakes.py) that returns canned
`ChatResult`s and records the `chat()` requests — never a live LLM. These tests
pin the brain's orchestration: the bounded tool loop (results fed back as `tool`
messages, a text turn ending the run, the iteration and correction budgets), the
request it builds (tools offered, thinking flag, prompt contents), and the
ChatResult->AgentResponse mapping. The wire itself (sampling, top_k,
chat_template_kwargs, tool-arg JSON parsing, reasoning_content drop) is pinned
one layer down in test_provider.py; the pipeline around the brain is pinned in
test_command.py.
"""

import json

import pytest

from chessapp.brain import AgentResponse
from chessapp.game import GameSession
from chessapp.llama_brain import LlamaBrain, create_llama_brain
from chessapp.personality import system_prompt_for
from chessapp.provider import ToolCallArgumentsError, Usage
from chessapp.tools import ToolContext, build_registry
from fakes import FakeEngine, ScriptedProvider, text_turn, tool_calls_turn

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
    _fn(
        "evaluate_position",
        "Evaluate the position.",
        {"type": "object", "properties": {}, "additionalProperties": False},
    ),
    _fn(
        "get_best_moves",
        "Candidate moves.",
        {"type": "object", "properties": {}, "additionalProperties": False},
    ),
    _fn(
        "analyze_last_move",
        "Was that a blunder?",
        {"type": "object", "properties": {}, "additionalProperties": False},
    ),
]


class FakeDispatcher:
    """`ToolDispatcher` double: records calls, returns canned results.

    Default result is a bland success, so a test only scripts the result it
    actually cares about (a rejected move, a domain error).
    """

    def __init__(self, results: dict[str, dict] | None = None) -> None:
        self.results = results or {}
        self.calls: list[tuple[str, dict]] = []

    def dispatch(self, name: str, args) -> dict:
        self.calls.append((name, args))
        return self.results.get(name, {"ok": True})


def make_brain(
    *turns, dispatcher=None, **kwargs
) -> tuple[LlamaBrain, ScriptedProvider]:
    """A brain wired to a ScriptedProvider playing `turns` in order.

    One turn is returned for every call; several are returned in order and the
    last repeats (so a loop scripts as "tool call, then the answer"). A turn may
    be an Exception to raise, modelling a provider-level failure.
    """
    provider = ScriptedProvider(*turns)
    brain = LlamaBrain(
        provider=provider,
        dispatcher=dispatcher if dispatcher is not None else FakeDispatcher(),
        system_prompt="You are a chess opponent.",
        **{"tool_definitions": TOOLS, **kwargs},
    )
    return brain, provider


def real_registry() -> tuple[object, GameSession]:
    """The actual registry over a real game — the brain's real executor."""
    session = GameSession()
    ctx = ToolContext(session=session, engine=FakeEngine())
    return build_registry(ctx), session


# --- the loop: a tool result reaches the model while it still holds tools ---


def test_the_agent_can_read_a_tool_result_and_then_act_on_it():
    # The case the old two-phase flow made structurally impossible: read
    # candidate moves, *then* play one. Three turns — call, call, comment.
    registry, session = real_registry()
    brain, provider = make_brain(
        tool_calls_turn(("get_best_moves", {"n": 3})),
        tool_calls_turn(("make_move", {"move": "e4"})),
        text_turn("Best line on the board. Your move."),
        dispatcher=registry,
        tool_definitions=registry.definitions(),
    )
    resp = brain.get_agent_response(board_state={}, command="play the best move")

    assert len(provider.calls) == 3
    assert [tc.name for tc in resp.tool_calls] == ["get_best_moves", "make_move"]
    assert [r["name"] for r in resp.tool_results] == ["get_best_moves", "make_move"]
    assert resp.text == "Best line on the board. Your move."
    assert resp.stop_reason == "completed"
    assert session.move_history()[0] == "e4"  # the move really happened


def test_tool_results_go_back_as_tool_messages_on_a_growing_prompt():
    # The structure the model was trained on: its own assistant(tool_calls)
    # turn, then a role:"tool" answer carrying the result — appended to the
    # same conversation, never a rebuilt prompt with a synthetic user message.
    dispatcher = FakeDispatcher({"make_move": {"legal": True, "san": "e4"}})
    brain, provider = make_brain(
        tool_calls_turn(("make_move", {"move": "e4"})),
        text_turn("e4 it is."),
        dispatcher=dispatcher,
    )
    brain.get_agent_response(board_state={}, command="play e4")

    first, second = provider.calls[0]["messages"], provider.calls[1]["messages"]
    assert second[: len(first)] == first  # the prompt grew; nothing was rebuilt
    assistant, tool = second[-2], second[-1]
    assert assistant["role"] == "assistant"
    assert assistant["tool_calls"][0]["function"]["name"] == "make_move"
    assert tool == {
        "role": "tool",
        "tool_call_id": "call_0",
        "content": json.dumps({"legal": True, "san": "e4"}),
    }


def test_every_turn_still_offers_tools():
    # The loop ends when the model declines to call a tool, never because the
    # brain took them away mid-run.
    brain, provider = make_brain(
        tool_calls_turn(("make_move", {"move": "e4"})),
        text_turn("done"),
    )
    brain.get_agent_response(board_state={}, command="play e4")
    assert all(call["tools"] == TOOLS for call in provider.calls)


def test_a_text_turn_ends_the_run_and_is_the_commentary():
    brain, provider = make_brain(text_turn("I like the Ruy Lopez!"))
    resp = brain.get_agent_response(board_state={}, command="favorite openings?")
    assert resp.text == "I like the Ruy Lopez!"
    assert resp.tool_calls == ()
    assert resp.tool_results == ()
    assert resp.stop_reason == "completed"
    assert len(provider.calls) == 1


def test_null_content_becomes_empty_string():
    brain, _ = make_brain(text_turn(None))
    assert brain.get_agent_response(board_state={}, command="hi").text == ""


def test_multiple_tool_calls_in_one_turn_run_in_order():
    dispatcher = FakeDispatcher()
    brain, _ = make_brain(
        tool_calls_turn(
            ("make_move", {"move": "e2e4"}),
            ("speak", {"text": "your move"}),
        ),
        text_turn("done"),
        dispatcher=dispatcher,
    )
    resp = brain.get_agent_response(board_state={}, command="play e4 and taunt me")
    assert [name for name, _ in dispatcher.calls] == ["make_move", "speak"]
    assert [tc.name for tc in resp.tool_calls] == ["make_move", "speak"]
    assert resp.tool_calls[1].args == {"text": "your move"}
    assert isinstance(resp, AgentResponse)


def test_empty_args_pass_through_as_empty_dict():
    # No-arg tools (new_game, undo) arrive with {} args (the provider already
    # turned a "" wire string into {} — pinned in test_provider).
    dispatcher = FakeDispatcher()
    brain, _ = make_brain(
        tool_calls_turn(("new_game", {})),
        text_turn("fresh board"),
        dispatcher=dispatcher,
    )
    resp = brain.get_agent_response(board_state={}, command="new game")
    assert resp.tool_calls[0].args == {}
    assert dispatcher.calls == [("new_game", {})]


# --- domain rejections are results, not corrections ------------------------


def test_an_illegal_move_is_a_result_the_model_corrects_from():
    # The heart of the reliability fix: an illegal-move guess no longer ends the
    # turn. The rejection goes back as a tool result, in the same conversation,
    # and the model's next call lands — inside the iteration budget, spending no
    # correction (an illegal move is a legitimate answer, not a malformed call).
    registry, session = real_registry()
    brain, provider = make_brain(
        tool_calls_turn(("make_move", {"move": "Nf6"})),  # illegal for White
        tool_calls_turn(("make_move", {"move": "Nf3"})),
        text_turn("Meant that one."),
        dispatcher=registry,
        tool_definitions=registry.definitions(),
        max_corrections=0,  # proves no correction was consumed
    )
    resp = brain.get_agent_response(board_state={}, command="knight to f6")

    assert resp.stop_reason == "completed"
    assert resp.tool_results[0]["result"]["legal"] is False
    assert session.move_history()[0] == "Nf3"
    rejection = json.loads(provider.calls[1]["messages"][-1]["content"])
    assert rejection["legal"] is False  # the model was shown the rejection


def test_a_domain_error_is_a_result_too():
    brain, _ = make_brain(
        tool_calls_turn(("make_move", {"move": "e4"})),
        text_turn("Can't do that one."),
        dispatcher=FakeDispatcher({"make_move": {"ok": False, "error": "no engine"}}),
        max_corrections=0,
    )
    resp = brain.get_agent_response(board_state={}, command="play e4")
    assert resp.stop_reason == "completed"
    assert resp.tool_results[0]["result"] == {"ok": False, "error": "no engine"}


# --- schema failures: a tool result *and* a correction ---------------------


def test_a_schema_violation_comes_back_as_an_error_result_and_is_corrected():
    # make_move wants a string; the model sent a number. Never dispatched, but
    # the model is shown exactly what was wrong and gets to fix it.
    dispatcher = FakeDispatcher()
    brain, provider = make_brain(
        tool_calls_turn(("make_move", {"move": 5})),
        tool_calls_turn(("make_move", {"move": "e4"})),
        text_turn("there we go"),
        dispatcher=dispatcher,
    )
    resp = brain.get_agent_response(board_state={}, command="play e4")

    assert dispatcher.calls == [("make_move", {"move": "e4"})]  # bad call not run
    assert resp.tool_results[0]["result"]["ok"] is False
    assert "invalid args" in resp.tool_results[0]["result"]["error"]
    assert resp.stop_reason == "completed"
    fed_back = json.loads(provider.calls[1]["messages"][-1]["content"])
    assert fed_back["ok"] is False


def test_an_unknown_tool_is_an_error_result_and_is_corrected():
    dispatcher = FakeDispatcher()
    brain, _ = make_brain(
        tool_calls_turn(("teleport_king", {})),
        tool_calls_turn(("make_move", {"move": "e4"})),
        text_turn("ok"),
        dispatcher=dispatcher,
    )
    resp = brain.get_agent_response(board_state={}, command="win instantly")
    assert dispatcher.calls == [("make_move", {"move": "e4"})]
    assert "unknown tool" in resp.tool_results[0]["result"]["error"]
    assert resp.stop_reason == "completed"


def test_repeated_schema_failures_exhaust_the_correction_budget():
    # A model that cannot form a valid call stops early, on its own budget —
    # without burning the whole iteration budget.
    brain, provider = make_brain(
        tool_calls_turn(("make_move", {"move": 5})),
        max_iterations=8,
        max_corrections=2,
    )
    resp = brain.get_agent_response(board_state={}, command="play e4")
    assert resp.stop_reason == "correction_limit"
    assert len(provider.calls) == 3  # the third correction is the one too many
    assert resp.text == ""


# --- unparseable arguments: the one case with nowhere to attach ------------


def _bad_json_call():
    # A quant hiccup: the provider could not parse the tool-call arguments as
    # JSON, so it raised before returning a result. There is no assistant turn
    # to append and so no tool message to answer it — the correction has to go
    # back as a user-role message.
    return ToolCallArgumentsError(
        "make_move", "arguments are not valid JSON (Expecting value)"
    )


def test_unparseable_args_correct_via_a_user_message_then_succeed():
    brain, provider = make_brain(
        _bad_json_call(),
        tool_calls_turn(("make_move", {"move": "e2e4"})),
        text_turn("e4."),
    )
    resp = brain.get_agent_response(board_state={}, command="play e4")

    assert resp.tool_calls[0].args == {"move": "e2e4"}
    assert resp.stop_reason == "completed"
    correction = provider.calls[1]["messages"][-1]
    assert correction["role"] == "user"  # nothing to attach a tool result to
    assert "make_move" in correction["content"]
    # The unusable turn was dropped, not appended.
    assert all(m["role"] != "assistant" for m in provider.calls[1]["messages"])


def test_unparseable_args_exhaust_the_correction_budget():
    brain, provider = make_brain(_bad_json_call(), max_iterations=8, max_corrections=2)
    resp = brain.get_agent_response(board_state={}, command="play e4")
    assert resp.stop_reason == "correction_limit"
    assert resp.tool_calls == ()
    assert len(provider.calls) == 3


# --- the iteration budget ---------------------------------------------------


def test_a_model_that_never_stops_calling_tools_hits_max_iterations():
    brain, provider = make_brain(
        tool_calls_turn(("make_move", {"move": "e4"})),  # repeats forever
        max_iterations=3,
    )
    resp = brain.get_agent_response(board_state={}, command="play e4")
    assert resp.stop_reason == "max_iterations"
    assert len(provider.calls) == 3
    assert resp.text == ""
    assert len(resp.tool_results) == 3  # everything it did is still reported


# --- cost accounting: model calls and tokens per turn ----------------------


def test_a_run_sums_tokens_and_counts_its_model_calls():
    # Two round trips — a tool call then the closing comment. The turn's cost is
    # their sum, and the call count is what every context cut is measured against.
    brain, _ = make_brain(
        tool_calls_turn(
            ("make_move", {"move": "e4"}),
            usage=Usage(prompt_tokens=7, completion_tokens=3),
        ),
        text_turn("e4 it is.", usage=Usage(prompt_tokens=5, completion_tokens=2)),
    )
    resp = brain.get_agent_response(board_state={}, command="play e4")
    assert resp.model_calls == 2
    assert resp.prompt_tokens == 12
    assert resp.completion_tokens == 5


def test_missing_usage_counts_the_call_but_adds_no_tokens():
    # llama-server may omit usage; a call with none still happened.
    brain, _ = make_brain(text_turn("hi", usage=None))
    resp = brain.get_agent_response(board_state={}, command="hi")
    assert resp.model_calls == 1
    assert resp.prompt_tokens == 0
    assert resp.completion_tokens == 0


def test_a_raised_round_trip_still_counts_as_a_call():
    # The provider raised before returning a result (unparseable args), but the
    # model was still called and the loop paid for it — the count must reflect
    # that, matching the eval harness's CountingProvider.
    brain, _ = make_brain(
        _bad_json_call(),
        tool_calls_turn(
            ("make_move", {"move": "e2e4"}),
            usage=Usage(prompt_tokens=4, completion_tokens=1),
        ),
        text_turn("e4.", usage=Usage(prompt_tokens=6, completion_tokens=2)),
    )
    resp = brain.get_agent_response(board_state={}, command="play e4")
    assert resp.model_calls == 3  # the raised call, the good call, the comment
    assert resp.prompt_tokens == 10  # only the two that returned usage
    assert resp.completion_tokens == 3


# --- thinking mode: OFF for speed, ON once analysis lands ------------------


def test_the_first_turn_does_not_think():
    brain, provider = make_brain(text_turn("ok"))
    brain.get_agent_response(board_state={"fen": "startpos"}, command="play e4")
    assert provider.calls[0]["enable_thinking"] is False


@pytest.mark.parametrize(
    "tool", ["evaluate_position", "get_best_moves", "analyze_last_move"]
)
def test_thinking_turns_on_once_an_analysis_result_lands(tool):
    # BRIEF: thinking OFF for fast move parsing, ON for analysis. In a loop the
    # rule is positional — off until an analysis tool has answered, on after.
    brain, provider = make_brain(
        tool_calls_turn((tool, {})),
        text_turn("deep thoughts"),
    )
    brain.get_agent_response(board_state={}, command="how am I doing?")
    assert provider.calls[0]["enable_thinking"] is False
    assert provider.calls[1]["enable_thinking"] is True


def test_a_plain_move_never_thinks():
    brain, provider = make_brain(
        tool_calls_turn(("make_move", {"move": "e4"})),
        text_turn("nice move"),
    )
    brain.get_agent_response(board_state={}, command="play e4")
    assert [c["enable_thinking"] for c in provider.calls] == [False, False]


def test_thinking_can_be_forced_on_for_the_whole_run():
    brain, provider = make_brain(text_turn("analysis"), enable_thinking=True)
    brain.get_agent_response(board_state={}, command="was that a blunder?")
    assert provider.calls[0]["enable_thinking"] is True


# --- request the brain sends ----------------------------------------------


def test_prompt_includes_system_board_state_and_command():
    brain, provider = make_brain(text_turn("ok"))
    brain.get_agent_response(board_state={"fen": "8/8/8/8"}, command="play Nf3")
    messages = provider.calls[0]["messages"]
    assert messages[0]["role"] == "system"
    assert "chess opponent" in messages[0]["content"]
    blob = " ".join(m["content"] for m in messages)
    assert "8/8/8/8" in blob  # board state reached the model
    assert "Nf3" in blob  # command reached the model


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


def test_the_transcript_survives_a_correction():
    brain, provider = make_brain(_bad_json_call(), text_turn("sorry"))
    brain.get_agent_response(board_state={}, command="play e4", transcript=TRANSCRIPT)
    assert provider.calls[1]["messages"][1:3] == TRANSCRIPT


# --- narrate: the fast path's commentary turn ------------------------------


def test_narrate_returns_commentary_from_the_new_state():
    brain, provider = make_brain(text_turn("Nice, e4! The classic."))
    narration = brain.narrate(
        board_state={"fen": "8/8/8/8", "turn": "black"},
        changes=[{"name": "make_move", "result": {"legal": True, "san": "e4"}}],
    )
    assert narration.text == "Nice, e4! The classic."
    blob = " ".join(m["content"] for m in provider.calls[0]["messages"])
    assert "8/8/8/8" in blob  # new board reached the model
    assert "e4" in blob  # what changed reached the model


def test_narrate_reports_its_one_model_call_and_tokens():
    # The fast path's stand-in for the closing turn is exactly one round trip;
    # its cost has to reach the trace like the loop's does.
    brain, _ = make_brain(
        text_turn("e4!", usage=Usage(prompt_tokens=9, completion_tokens=4))
    )
    narration = brain.narrate(board_state={}, changes=[])
    assert narration.model_calls == 1
    assert narration.prompt_tokens == 9
    assert narration.completion_tokens == 4


def test_narrate_sends_no_tools():
    # It stands in for the loop's closing turn, so like that turn it must not
    # be able to act.
    brain, provider = make_brain(text_turn("ok"))
    brain.narrate(board_state={}, changes=[])
    assert provider.calls[0]["tools"] is None


def test_narrate_includes_the_transcript():
    brain, provider = make_brain(text_turn("ok"))
    brain.narrate(
        board_state={"fen": "8/8/8/8"},
        changes=[{"name": "make_move", "result": {"san": "Nf3"}}],
        transcript=TRANSCRIPT,
    )
    assert provider.calls[0]["messages"][1:3] == TRANSCRIPT


def test_narrate_null_content_becomes_empty_string():
    brain, _ = make_brain(text_turn(None))
    assert brain.narrate(board_state={}, changes=[]).text == ""


# --- factory wires the system prompt ---------------------------------------


def test_create_llama_brain_defaults_to_the_glitch_prompt():
    # The brain is model-specific but personality-agnostic: without a provider
    # the factory resolves the one system prompt once and carries the string.
    brain = create_llama_brain(
        base_url="http://localhost:8200/v1",
        model="gemma",
        dispatcher=FakeDispatcher(),
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
        dispatcher=FakeDispatcher(),
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


def test_callable_system_prompt_also_drives_narration():
    provider = ScriptedProvider(text_turn("ok"))
    brain = LlamaBrain(
        provider=provider,
        dispatcher=FakeDispatcher(),
        tool_definitions=TOOLS,
        system_prompt=lambda: system_prompt_for(verbosity="low"),
    )
    brain.narrate(board_state={}, changes=[])
    assert provider.calls[0]["messages"][0]["content"] == system_prompt_for(
        verbosity="low"
    )


def test_create_llama_brain_accepts_a_live_prompt_provider():
    from chessapp.tools import Settings

    settings = Settings()  # verbosity defaults to normal
    brain = create_llama_brain(
        base_url="http://localhost:8200/v1",
        model="gemma",
        dispatcher=FakeDispatcher(),
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
        dispatcher=FakeDispatcher(),
        tool_definitions=[],
        provider=fake,
    )
    assert brain.provider is fake
