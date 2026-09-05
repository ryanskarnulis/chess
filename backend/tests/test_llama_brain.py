"""The llama-server brain over a `ChatProvider`, behind the `Brain` seam.

Exercised only through a `ScriptedProvider` (tests/fakes.py) that returns canned
`ChatResult`s and records the `chat()` requests — never a live LLM. These tests
pin the brain's orchestration: the bounded tool loop (results fed back as `tool`
messages, a text turn ending the loop, the iteration and correction budgets),
the two phases the loop sits inside (`docs/planner-narrator.md`), the requests
it builds (tools offered, thinking flag, temperature, prompt contents), and the
ChatResult->AgentResponse mapping. The wire itself (sampling, top_k,
chat_template_kwargs, tool-arg JSON parsing, reasoning_content drop) is pinned
one layer down in test_provider.py; the pipeline around the brain is pinned in
test_command.py.

One turn is two phases, so most scripted sequences here end with **two** text
turns: the planner's handoff note (which no player ever sees) and the narrator's
reply (which is the `AgentResponse.text`).
"""

import json

import pytest

from chessapp.brain import CANCEL, CONFIRM, UNRELATED, AgentResponse
from chessapp.game import GameSession
from chessapp.llama_brain import (
    _ANSWER_MAX_TOKENS,
    _NO_PROGRESS_NOTE,
    LlamaBrain,
    _fast_path_brief,
    create_llama_brain,
)
from chessapp.personality import PLANNER_PROMPT, system_prompt_for
from chessapp.provider import (
    ProviderError,
    ProviderFailure,
    ProviderRequestError,
    ToolCallArgumentsError,
    Usage,
)
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


# The two prompts, as recognizable stand-ins: which one a call carries is how a
# test tells a planner turn from the narrator's.
PLANNER = "Pick the tools."
PERSONA = "You are a chess opponent."


def make_brain(
    *turns, dispatcher=None, **kwargs
) -> tuple[LlamaBrain, ScriptedProvider]:
    """A brain wired to a ScriptedProvider playing `turns` in order.

    One turn is returned for every call; several are returned in order and the
    last repeats (so a loop scripts as "tool call, note, reply"). A turn may
    be an Exception to raise, modelling a provider-level failure.
    """
    provider = ScriptedProvider(*turns)
    brain = LlamaBrain(
        provider=provider,
        dispatcher=dispatcher if dispatcher is not None else FakeDispatcher(),
        system_prompt=PERSONA,
        **{"tool_definitions": TOOLS, "planner_prompt": PLANNER, **kwargs},
    )
    return brain, provider


def stub_clock(*seconds: float):
    """A monotonic clock returning each reading in turn — two per round trip
    (before and after), so a pair scripts one call's latency. Running out is a
    `StopIteration`: a test that mistimes its script fails loudly."""
    readings = iter(seconds)
    return lambda: next(readings)


def system_prompts(provider: ScriptedProvider) -> list[str]:
    """The system prompt every recorded call carried, in order."""
    return [call["messages"][0]["content"] for call in provider.calls]


def real_registry() -> tuple[object, GameSession]:
    """The actual registry over a real game — the brain's real executor."""
    session = GameSession()
    ctx = ToolContext(session=session, engine=FakeEngine())
    return build_registry(ctx), session


# --- the loop: a tool result reaches the model while it still holds tools ---


def test_the_agent_can_read_a_tool_result_and_then_act_on_it():
    # The case the old two-phase flow made structurally impossible: read
    # candidate moves, *then* play one. Four turns — call, call, note, reply.
    registry, session = real_registry()
    brain, provider = make_brain(
        tool_calls_turn(("get_best_moves", {"n": 3})),
        tool_calls_turn(("make_move", {"move": "e4"})),
        text_turn("read the candidates, played e4"),
        text_turn("Best line on the board. Your move."),
        dispatcher=registry,
        tool_definitions=registry.definitions(),
    )
    resp = brain.get_agent_response(board_state={}, command="play the best move")

    assert len(provider.calls) == 4
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


def test_every_planner_turn_still_offers_tools():
    # The loop ends when the model declines to call a tool, never because the
    # brain took them away mid-run. (The narrator is a different phase and is
    # offered none — pinned below.)
    brain, provider = make_brain(
        tool_calls_turn(("make_move", {"move": "e4"})),
        text_turn("played e4"),
        text_turn("done"),
    )
    brain.get_agent_response(board_state={}, command="play e4")
    assert [call["tools"] for call in provider.calls] == [TOOLS, TOOLS, None]


def test_a_planner_text_turn_ends_the_loop_and_hands_off():
    brain, provider = make_brain(
        text_turn("nothing to do; answer the opening question"),
        text_turn("I like the Ruy Lopez!"),
    )
    resp = brain.get_agent_response(board_state={}, command="favorite openings?")
    assert resp.text == "I like the Ruy Lopez!"
    assert resp.tool_calls == ()
    assert resp.tool_results == ()
    assert resp.stop_reason == "completed"
    assert len(provider.calls) == 2  # the decline, then the voice


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
    # without burning the whole iteration budget. A budget stop also skips the
    # narrator entirely: there is no verified outcome to speak from, so the
    # pipeline's canned stuck reply answers instead.
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
    # Distinct calls every turn: a model that keeps finding new work to do runs
    # out of budget. (A model that asks for the *same* work twice is stopped one
    # turn earlier by the repeat rule below — a different stop, on purpose.)
    brain, provider = make_brain(
        tool_calls_turn(("make_move", {"move": "e4"})),
        tool_calls_turn(("make_move", {"move": "e5"})),
        tool_calls_turn(("make_move", {"move": "e6"})),
        max_iterations=3,
    )
    resp = brain.get_agent_response(board_state={}, command="play e4")
    assert resp.stop_reason == "max_iterations"
    assert len(provider.calls) == 3  # the loop's turns only — no narrator
    assert resp.text == ""
    assert len(resp.tool_results) == 3  # everything it did is still reported


# --- the repeat rule: a turn that asks for nothing new is the planner's last --


def test_a_planner_turn_that_only_repeats_itself_ends_the_loop():
    # The recorded defect: asked "what should I play?" under the retired
    # hints-off mode, the planner re-ran the reads it had already run and
    # burned all four iterations, so the turn ended on a budget stop and the
    # player got the canned stuck line. A call it already made against this
    # turn cannot bring anything new back, so the second one is the planner's
    # last word.
    brain, provider = make_brain(
        tool_calls_turn(("evaluate_position", {})),
        tool_calls_turn(("evaluate_position", {})),
        text_turn("nothing new to add"),
        max_iterations=4,
    )
    resp = brain.get_agent_response(board_state={}, command="what should I play?")

    assert resp.stop_reason == "no_progress"
    # Two planner turns and the narrator — not the four the budget allowed.
    assert system_prompts(provider) == [PLANNER, PLANNER, PERSONA]
    assert resp.text == "nothing new to add"


def test_a_repeated_call_still_runs_and_is_still_reported():
    # The rule only ends the *loop*: whether a repeat is allowed to run is the
    # tool layer's call (the phase machine already refuses a second player
    # move), so the dispatch happens exactly as before and the turn reports it.
    dispatcher = FakeDispatcher()
    brain, _ = make_brain(
        tool_calls_turn(("evaluate_position", {})),
        tool_calls_turn(("evaluate_position", {})),
        text_turn("reply"),
        dispatcher=dispatcher,
    )
    resp = brain.get_agent_response(board_state={}, command="what should I play?")

    assert dispatcher.calls == [("evaluate_position", {}), ("evaluate_position", {})]
    assert [r["name"] for r in resp.tool_results] == [
        "evaluate_position",
        "evaluate_position",
    ]


def test_a_repeat_next_to_new_work_is_progress_and_the_loop_goes_on():
    # Only a turn that asks for *nothing* new stops the loop: a repeat riding
    # along beside a fresh call is a turn that still moved.
    brain, provider = make_brain(
        tool_calls_turn(("evaluate_position", {})),
        tool_calls_turn(("evaluate_position", {}), ("analyze_last_move", {})),
        text_turn("looked at both"),
        text_turn("here's the read"),
    )
    resp = brain.get_agent_response(board_state={}, command="how am I doing?")

    assert resp.stop_reason == "completed"
    assert len(provider.calls) == 4  # three planner turns and the narrator


def test_the_same_tool_with_different_args_is_not_a_repeat():
    brain, provider = make_brain(
        tool_calls_turn(("make_move", {"move": "e4"})),
        tool_calls_turn(("make_move", {"move": "e5"})),
        text_turn("played both"),
        text_turn("your move"),
    )
    resp = brain.get_agent_response(board_state={}, command="play e4 then e5")

    assert resp.stop_reason == "completed"
    assert len(provider.calls) == 4


def test_the_same_args_in_a_different_order_is_still_a_repeat():
    # The key is the call, not how the model happened to serialize it.
    brain, _ = make_brain(
        tool_calls_turn(("speak", {"text": "hi", "voice": "glitch"})),
        tool_calls_turn(("speak", {"voice": "glitch", "text": "hi"})),
        text_turn("said it once"),
    )
    resp = brain.get_agent_response(board_state={}, command="say hi")

    assert resp.stop_reason == "no_progress"


def test_a_repeat_stop_reaches_the_narrator_with_what_the_turn_verified():
    # The stop that is *not* a budget stop: results came back, so there is
    # something verified to speak from and the player gets an answer rather
    # than the pipeline's canned stuck line.
    dispatcher = FakeDispatcher({"analyze_last_move": {"ok": True, "verdict": "fine"}})
    brain, provider = make_brain(
        tool_calls_turn(("analyze_last_move", {})),
        tool_calls_turn(("analyze_last_move", {})),
        text_turn("that move was fine"),
        dispatcher=dispatcher,
    )
    resp = brain.get_agent_response(board_state={}, command="was that bad?")

    assert resp.text == "that move was fine"
    narrator = provider.calls[-1]
    assert narrator["tools"] is None  # the closing pass still holds no tools
    assert (
        json.dumps({"ok": True, "verdict": "fine"})
        in narrator["messages"][-1]["content"]
    )


def test_a_repeat_stop_hands_the_narrator_the_loops_own_note():
    # The planner never reached the turn that writes a note, so the loop
    # supplies one — a brief with no note at all measured slower live, the
    # narrator reasoning about what to do next instead of what to say. It says
    # the work is finished and nothing about the loop that finished it.
    brain, provider = make_brain(
        tool_calls_turn(("evaluate_position", {})),
        tool_calls_turn(("evaluate_position", {})),
        text_turn("reply"),
    )
    brain.get_agent_response(board_state={}, command="what should I play?")

    brief = provider.calls[-1]["messages"][-1]["content"]
    assert _NO_PROGRESS_NOTE in brief
    assert "repeat" not in brief.lower()  # the machinery stays in the trace


def test_an_empty_planner_note_leaves_the_heading_out_of_the_brief():
    # The planner *can* hand off with nothing to say. An empty heading would
    # read as a note that said nothing, which is a different claim.
    brain, provider = make_brain(
        tool_calls_turn(("evaluate_position", {})),
        text_turn(""),  # the planner's handoff, wordless
        text_turn("reply"),
    )
    brain.get_agent_response(board_state={}, command="how's it looking?")

    brief = provider.calls[-1]["messages"][-1]["content"]
    assert "Note from the layer that did it" not in brief


# --- cost accounting: model calls and tokens per turn ----------------------


def test_a_run_sums_tokens_and_counts_its_model_calls():
    # Three round trips — a tool call, the planner's note, the narrator's reply.
    # The turn's cost is their sum, and the call count is what every context cut
    # is measured against.
    brain, _ = make_brain(
        tool_calls_turn(
            ("make_move", {"move": "e4"}),
            usage=Usage(prompt_tokens=7, completion_tokens=3),
        ),
        text_turn("played e4", usage=Usage(prompt_tokens=5, completion_tokens=2)),
        text_turn("e4 it is.", usage=Usage(prompt_tokens=4, completion_tokens=6)),
    )
    resp = brain.get_agent_response(board_state={}, command="play e4")
    assert resp.model_calls == 3
    assert resp.prompt_tokens == 16
    assert resp.completion_tokens == 11


def test_the_narrator_round_trip_is_counted():
    # The extra call the split costs is *visible*: it is in the turn's own
    # accounting, so the trace and the eval baseline pay for it honestly.
    brain, _ = make_brain(
        text_turn("nothing to do", usage=Usage(prompt_tokens=3, completion_tokens=1)),
        text_turn(
            "Ruy Lopez, obviously.", usage=Usage(prompt_tokens=8, completion_tokens=5)
        ),
    )
    resp = brain.get_agent_response(board_state={}, command="favorite openings?")
    assert resp.model_calls == 2
    assert (resp.prompt_tokens, resp.completion_tokens) == (11, 6)


def test_missing_usage_counts_the_call_but_adds_no_tokens():
    # llama-server may omit usage; a call with none still happened.
    brain, _ = make_brain(text_turn("hi", usage=None))
    resp = brain.get_agent_response(board_state={}, command="hi")
    assert resp.model_calls == 2  # the planner's decline and the narrator
    assert resp.prompt_tokens == 0
    assert resp.completion_tokens == 0


def test_each_round_trip_is_timed_on_its_own():
    # Per-call latency, not just the turn's total: a slow narrator and a slow
    # planner are different problems, and one number cannot tell them apart.
    brain, _ = make_brain(
        tool_calls_turn(("make_move", {"move": "e4"})),
        text_turn("played e4"),
        text_turn("e4 it is."),
        clock=stub_clock(0.0, 0.4, 1.0, 1.25, 2.0, 2.9),
    )
    resp = brain.get_agent_response(board_state={}, command="play e4")
    assert resp.model_latencies_ms == (400, 250, 900)
    assert len(resp.model_latencies_ms) == resp.model_calls


def test_a_raised_round_trip_is_timed_too():
    # A call that died still spent wall clock — often the most of any of them —
    # and the count already includes it, so the timing must too.
    brain, _ = make_brain(
        ProviderError("llama-server went away"), clock=stub_clock(0.0, 3.5)
    )
    resp = brain.get_agent_response(board_state={}, command="play e4")
    assert resp.stop_reason == "provider_error"
    assert resp.model_latencies_ms == (3500,)


def test_narrate_times_its_one_call():
    brain, _ = make_brain(text_turn("e4!"), clock=stub_clock(0.0, 0.12))
    narration = brain.narrate(board_state={}, changes=[])
    assert narration.latency_ms == 120
    assert narration.model_latencies_ms == (120,), "read as the loop's is"


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
        text_turn("played e4", usage=Usage(prompt_tokens=6, completion_tokens=2)),
        text_turn("e4.", usage=Usage(prompt_tokens=3, completion_tokens=1)),
    )
    resp = brain.get_agent_response(board_state={}, command="play e4")
    # The raised call, the good call, the note, the reply.
    assert resp.model_calls == 4
    assert resp.prompt_tokens == 13  # only the three that returned usage
    assert resp.completion_tokens == 4


# --- thinking mode: OFF for speed, ON once analysis lands ------------------


def test_the_first_turn_does_not_think():
    brain, provider = make_brain(text_turn("ok"))
    brain.get_agent_response(board_state={"fen": "startpos"}, command="play e4")
    assert provider.calls[0]["enable_thinking"] is False


@pytest.mark.parametrize(
    "tool", ["evaluate_position", "get_best_moves", "analyze_last_move"]
)
def test_thinking_turns_on_once_an_analysis_result_lands(tool):
    # BRIEF: thinking OFF for fast move parsing, ON for analysis. The split
    # sharpens the rule: planner turns are tool-picking — a parse — so they
    # never think, even with an analysis result in context; the narrator is the
    # phase that reasons about that result, and it alone flips ON. One thinking
    # turn per analysis question, not two.
    brain, provider = make_brain(
        tool_calls_turn((tool, {})),
        text_turn("evaluated the position"),
        text_turn("deep thoughts"),
    )
    brain.get_agent_response(board_state={}, command="how am I doing?")
    assert [c["enable_thinking"] for c in provider.calls] == [False, False, True]


def test_a_plain_move_never_thinks():
    brain, provider = make_brain(
        tool_calls_turn(("make_move", {"move": "e4"})),
        text_turn("played e4"),
        text_turn("nice move"),
    )
    brain.get_agent_response(board_state={}, command="play e4")
    assert [c["enable_thinking"] for c in provider.calls] == [False, False, False]


def test_thinking_can_be_forced_on_for_the_whole_run():
    brain, provider = make_brain(text_turn("analysis"), enable_thinking=True)
    brain.get_agent_response(board_state={}, command="was that a blunder?")
    assert provider.calls[0]["enable_thinking"] is True


# --- request the brain sends ----------------------------------------------


def test_planner_prompt_includes_system_board_state_and_command():
    brain, provider = make_brain(text_turn("ok"))
    brain.get_agent_response(board_state={"fen": "8/8/8/8"}, command="play Nf3")
    messages = provider.calls[0]["messages"]
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == PLANNER
    blob = " ".join(m["content"] for m in messages)
    assert "8/8/8/8" in blob  # board state reached the model
    assert "Nf3" in blob  # command reached the model


# --- the two phases: the planner decides, the narrator speaks ----------------
#
# `docs/planner-narrator.md`. The loop is unchanged (audit 15 keeps #124's win);
# what moved out of it is the voice. So every turn that finishes its work is two
# prompts: the compact contract the tools were chosen under, then the persona
# prompt that phrases the reply — with no tools on offer, so the closing pass is
# tool-free by construction rather than by the model declining.


def test_loop_turns_run_on_the_planner_prompt_and_the_closer_on_the_persona():
    brain, provider = make_brain(
        tool_calls_turn(("make_move", {"move": "e4"})),
        text_turn("played e4"),
        text_turn("e4 it is."),
    )
    brain.get_agent_response(board_state={}, command="play e4")
    assert system_prompts(provider) == [PLANNER, PLANNER, PERSONA]


def test_the_narrator_is_offered_no_tools():
    # Audit item 9's mechanism: the phase that talks to the player cannot act,
    # because it is never handed anything to act with.
    brain, provider = make_brain(
        tool_calls_turn(("make_move", {"move": "e4"})),
        text_turn("played e4"),
        text_turn("e4 it is."),
    )
    brain.get_agent_response(board_state={}, command="play e4")
    assert provider.calls[-1]["tools"] is None


def test_the_narrators_text_is_the_reply_and_the_planners_note_is_not():
    # The handoff: the planner's line is context for the narrator and nothing
    # else. It must never be what the player reads.
    brain, provider = make_brain(
        tool_calls_turn(("make_move", {"move": "e4"})),
        text_turn("made the move e4, it was legal"),
        text_turn("e4 it is."),
    )
    resp = brain.get_agent_response(board_state={}, command="play e4")

    assert resp.text == "e4 it is."
    assert "made the move e4" not in resp.text
    narrator = " ".join(m["content"] for m in provider.calls[-1]["messages"])
    assert "made the move e4, it was legal" in narrator  # the note carried over


def test_the_narrator_sees_the_utterance_and_what_the_turn_did():
    brain, provider = make_brain(
        tool_calls_turn(("make_move", {"move": "e4"})),
        text_turn("played e4"),
        text_turn("Classic."),
        dispatcher=FakeDispatcher({"make_move": {"legal": True, "san": "e4"}}),
    )
    brain.get_agent_response(board_state={}, command="king's pawn, two squares")

    narrator = " ".join(m["content"] for m in provider.calls[-1]["messages"])
    assert "king's pawn, two squares" in narrator  # what was asked
    assert json.dumps({"legal": True, "san": "e4"}) in narrator  # what happened


def test_the_narrator_gets_the_transcript():
    brain, provider = make_brain(text_turn("nothing to do"), text_turn("ok"))
    brain.get_agent_response(board_state={}, command="hi", transcript=TRANSCRIPT)
    assert provider.calls[-1]["messages"][1:3] == TRANSCRIPT


@pytest.mark.parametrize(
    ("tool", "thinks"),
    [
        ("evaluate_position", True),
        ("get_best_moves", True),
        ("analyze_last_move", True),
        ("make_move", False),
        ("new_game", False),
    ],
)
def test_the_narrator_thinks_only_after_an_analysis_result(tool, thinks):
    brain, provider = make_brain(
        tool_calls_turn((tool, {})),
        text_turn("did the thing"),
        text_turn("words for the player"),
    )
    brain.get_agent_response(board_state={}, command="do it")
    assert provider.calls[-1]["enable_thinking"] is thinks


def test_a_budget_stop_never_reaches_the_narrator():
    # Nothing verified came back, so there is nothing to speak from: the turn
    # ends silent and the pipeline's canned stuck reply covers it.
    brain, provider = make_brain(
        tool_calls_turn(("make_move", {"move": "e4"})),
        tool_calls_turn(("make_move", {"move": "e5"})),  # new work every turn
        max_iterations=2,
    )
    resp = brain.get_agent_response(board_state={}, command="play e4")
    assert resp.stop_reason == "max_iterations"
    assert resp.text == ""
    assert system_prompts(provider) == [PLANNER, PLANNER]  # never the persona
    assert resp.model_calls == 2


# --- per-phase sampling ------------------------------------------------------


def test_the_planners_temperature_rides_on_its_own_calls_only():
    # The split's other half (the absorbed sampling experiment): the planner may
    # sample cooler than the narrator, whose job is words and wants the
    # BRIEF's default. `None` is "the provider's default", not "no temperature".
    brain, provider = make_brain(
        tool_calls_turn(("make_move", {"move": "e4"})),
        text_turn("played e4"),
        text_turn("e4 it is."),
        planner_temperature=0.2,
    )
    brain.get_agent_response(board_state={}, command="play e4")
    assert [call["temperature"] for call in provider.calls] == [0.2, 0.2, None]


def test_no_planner_temperature_leaves_every_call_on_the_default():
    brain, provider = make_brain(
        tool_calls_turn(("make_move", {"move": "e4"})),
        text_turn("played e4"),
        text_turn("e4 it is."),
    )
    brain.get_agent_response(board_state={}, command="play e4")
    assert [call["temperature"] for call in provider.calls] == [None, None, None]


def test_narrate_stays_on_the_default_temperature():
    # The fast path's request must not change at all in this slice.
    brain, provider = make_brain(text_turn("ok"), planner_temperature=0.2)
    brain.narrate(board_state={}, changes=[])
    assert provider.calls[0]["temperature"] is None


# --- token caps: every model call is bounded ---------------------------------
#
# Without a `max_tokens`, llama-server runs n_predict -1 and a degenerate
# thought loop generates until the provider's 300 s read timeout fires
# (observed live 2026-07-27, 20k+ tokens on ordinary planner calls). Every
# phase therefore carries its own ceiling, and a call the ceiling cut off
# (`finish_reason == "length"`) is a failed turn whose fragment never travels.


def test_every_planner_call_carries_the_planner_token_cap():
    brain, provider = make_brain(
        tool_calls_turn(("make_move", {"move": "e4"})),
        text_turn("played e4"),
        text_turn("e4 it is."),
    )
    brain.get_agent_response(board_state={}, command="play e4")
    planner_calls = provider.calls[:-1]
    assert planner_calls
    assert all(call["max_tokens"] == brain.planner_max_tokens for call in planner_calls)


def test_the_narrator_carries_its_own_larger_cap():
    # The narrator is the one phase that thinks, and thinking tokens count
    # toward max_tokens on this server — its ceiling must clear the largest
    # measured legitimate narration (~2.6k tokens), which the planner's need not.
    brain, provider = make_brain(text_turn("note"), text_turn("reply"))
    brain.get_agent_response(board_state={}, command="hi")
    assert provider.calls[-1]["max_tokens"] == brain.narrator_max_tokens
    assert brain.narrator_max_tokens > brain.planner_max_tokens


def test_narrate_carries_the_narrator_cap():
    # The fast path's commentary turn is the same phase as the loop's closer.
    brain, provider = make_brain(text_turn("nice"))
    brain.narrate(board_state={}, changes=[])
    assert provider.calls[0]["max_tokens"] == brain.narrator_max_tokens


def test_the_caps_are_generous_enough_for_measured_real_turns():
    # The floor the numbers may never sink under: legitimate thinking-on
    # narrations reached 2,633 completion tokens (docs/agent-evals.md;
    # a live 2,408 in the 2026-07-27 trace), and clipping a real turn is the
    # failure this fix must not trade for. The planner never thinks; its real
    # output is tool calls or a one-line note.
    brain, _ = make_brain(text_turn("ok"))
    assert brain.narrator_max_tokens >= 3000
    assert brain.planner_max_tokens >= 1024


def test_a_truncated_planner_turn_is_a_failed_turn_not_a_handoff():
    # `finish_reason == "length"`: the cap cut the model off, so whatever
    # content survived is a fragment (or nothing at all — a thought loop can
    # spend the whole budget in reasoning). It must never travel onward as the
    # note; the loop ends the phase itself, exactly as a repeat-stop does.
    brain, provider = make_brain(
        text_turn("okay so first I should probably", finish_reason="length"),
        text_turn("reply"),
    )
    resp = brain.get_agent_response(board_state={}, command="hi")
    assert resp.stop_reason == "no_progress"
    assert resp.text == "reply"
    brief = provider.calls[-1]["messages"][-1]["content"]
    assert "okay so first" not in brief
    assert _NO_PROGRESS_NOTE in brief


def test_a_truncated_planner_turn_with_no_content_at_all():
    # The observed live shape: the whole budget went to reasoning_content
    # (which the provider drops) and `content` came back empty.
    brain, _ = make_brain(
        text_turn(None, finish_reason="length"),
        text_turn("reply"),
    )
    resp = brain.get_agent_response(board_state={}, command="hi")
    assert resp.stop_reason == "no_progress"
    assert resp.text == "reply"


def test_a_truncated_planner_turn_keeps_what_the_turn_verified():
    # A move that landed before the truncation is real board history; the
    # narrator still closes the turn from it, like any no-progress stop.
    brain, _ = make_brain(
        tool_calls_turn(("make_move", {"move": "e4"})),
        text_turn("and now I will", finish_reason="length"),
        text_turn("e4, done."),
    )
    resp = brain.get_agent_response(board_state={}, command="play e4")
    assert resp.stop_reason == "no_progress"
    assert [call.name for call in resp.tool_calls] == ["make_move"]
    assert resp.text == "e4, done."


def test_a_truncated_narrator_says_nothing_but_still_costs():
    # Half a sentence shown to the player is worse than none — the pipeline
    # already composes its deterministic lines around an empty reply. The
    # tokens the cut-off call generated are still the turn's to pay for.
    brain, _ = make_brain(
        text_turn("note"),
        text_turn(
            "and with that, the game is basically",
            finish_reason="length",
            usage=Usage(prompt_tokens=9, completion_tokens=4096),
        ),
    )
    resp = brain.get_agent_response(board_state={}, command="hi")
    assert resp.text == ""
    assert resp.stop_reason == "completed"
    assert resp.completion_tokens == 4096


def test_a_truncated_narrate_says_nothing_but_still_counts():
    brain, _ = make_brain(
        text_turn(
            "half a",
            finish_reason="length",
            usage=Usage(prompt_tokens=5, completion_tokens=7),
        )
    )
    narration = brain.narrate(board_state={}, changes=[])
    assert narration.text == ""
    assert narration.completion_tokens == 7


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


# --- the fast path's brief says whose move it is reacting to ------------------
#
# The brief used to open "You just acted on the player's behalf", which a 12B
# reads as "I moved": live, Glitch narrated the player's capture as his own. The
# opener now states the attribution outright — and states *only* that. The first
# version also named each side's color, and that was an identity the narrator
# used: reacting mid-turn from a board where it is the engine's move, with the
# engine's legal moves in the state block, "you are playing black" turned the
# reaction beat into a move-selection beat — every reaction in the 2026-07-28
# game announced a reply ("i'll go with d6") the narrator had no way to know,
# since Stockfish was still computing it. The attribution the fix needed was
# purely negative: whose move the results are NOT.

FAST_PATH_STATE = {"fen": "8/8/8/8", "turn": "black", "player_color": "white"}


def test_fast_path_brief_does_not_say_the_narrator_acted():
    assert "on the player's behalf" not in _fast_path_brief(FAST_PATH_STATE, [])


def test_fast_path_brief_attributes_the_move_to_the_player():
    brief = _fast_path_brief(FAST_PATH_STATE, [])
    assert "the player's move" in brief
    assert "not yours" in brief


def test_fast_path_brief_gives_the_narrator_no_side_to_play():
    # Naming the narrator's color handed it a move to choose. The brief carries
    # no color for either side, whichever the player has.
    for color in ("white", "black"):
        brief = _fast_path_brief({**FAST_PATH_STATE, "player_color": color}, [])
        assert "is playing" not in brief
        assert "you are playing" not in brief


def test_fast_path_brief_keeps_its_structure():
    brief = _fast_path_brief(
        FAST_PATH_STATE, [{"name": "make_move", "result": {"san": "exd5"}}]
    )
    assert "exd5" in brief, "the results still reach the narrator"
    assert "8/8/8/8" in brief, "so does the new board"
    assert "in-character" in brief
    assert "Do not call any tools." in brief


# --- recovery: a provider failure mid-turn (audit item 20) ------------------
#
# A dead llama-server is a fact about the turn, not an exception the pipeline
# should have to catch after the board already changed. The loop converts it to
# `stop_reason="provider_error"` and hands back everything that verifiably ran,
# so the pipeline can still close the turn (collect the engine's reply,
# broadcast) and tell the player the truth. `narrate` is the one deliberate
# exception: its only caller wraps it, and it runs after no tools at all.


def test_provider_failure_mid_loop_returns_the_turn_so_far():
    registry, session = real_registry()
    brain, _ = make_brain(
        tool_calls_turn(("make_move", {"move": "e4"})),
        ProviderError("llama-server went away"),
        dispatcher=registry,
        tool_definitions=registry.definitions(),
    )

    resp = brain.get_agent_response(board_state={}, command="push the king pawn")

    assert resp.stop_reason == "provider_error"
    assert resp.text == ""
    assert [r["name"] for r in resp.tool_results] == ["make_move"]
    assert session.move_history()[0] == "e4", "the move that landed stands"


def test_provider_failure_on_the_first_call_reports_an_empty_turn():
    brain, _ = make_brain(ProviderError("connection refused"))

    resp = brain.get_agent_response(board_state={}, command="hello?")

    assert resp.stop_reason == "provider_error"
    assert resp.tool_results == () and resp.text == ""
    # The round trip is still counted — the docstring's convention for a call
    # that raised before returning a result.
    assert resp.model_calls == 1


def test_provider_failure_in_the_narrator_still_reports_the_tools():
    # The planner finished; the persona call died. The tool results are the
    # record of a turn that really happened, so they come back regardless.
    brain, _ = make_brain(
        tool_calls_turn(("make_move", {"move": "e4"})),
        text_turn("played e4"),
        ProviderError("timed out"),
    )

    resp = brain.get_agent_response(board_state={}, command="e pawn forward")

    assert resp.stop_reason == "provider_error"
    assert resp.text == ""
    assert [r["name"] for r in resp.tool_results] == ["make_move"]


# --- and *which* failure it was --------------------------------------------
#
# `provider_error` alone says the turn died, not why, and the two whys want
# opposite handling: a crashed socket is worth asking again, a request the
# server rejects is worth asking about. The loop used to catch `ProviderError`
# bare and drop the exception on the floor, so a 400 and a dead server reached
# the eval harness identically and it retried both.


def test_a_provider_death_names_its_failure_kind():
    brain, _ = make_brain(
        ProviderRequestError("llama-server returned 400", ProviderFailure.REJECTED)
    )

    resp = brain.get_agent_response(board_state={}, command="hello?")

    assert resp.stop_reason == "provider_error"
    assert resp.provider_failure == "rejected"


def test_a_narrator_death_names_its_failure_kind_too():
    # The second bare `except ProviderError` — same defect, same fix, and the
    # one a long transcript actually hits (the narrator carries the whole
    # conversation, so it is where a context overrun surfaces first).
    brain, _ = make_brain(
        tool_calls_turn(("make_move", {"move": "e4"})),
        text_turn("played e4"),
        ProviderRequestError("llama-server returned 400", ProviderFailure.REJECTED),
    )

    resp = brain.get_agent_response(board_state={}, command="e pawn forward")

    assert resp.stop_reason == "provider_error"
    assert resp.provider_failure == "rejected"


def test_a_turn_that_did_not_die_names_no_failure():
    brain, _ = make_brain(text_turn("still here"))

    resp = brain.get_agent_response(board_state={}, command="hello?")

    assert resp.stop_reason == "completed"
    assert resp.provider_failure == ""


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


def test_create_llama_brain_defaults_to_the_planner_prompt_for_the_loop():
    brain = create_llama_brain(
        base_url="http://localhost:8200/v1",
        model="gemma",
        dispatcher=FakeDispatcher(),
        tool_definitions=[],
    )
    assert brain.planner_prompt == PLANNER_PROMPT
    assert brain.planner_temperature is None  # the provider's default


def test_create_llama_brain_carries_a_planner_temperature():
    brain = create_llama_brain(
        base_url="http://localhost:8200/v1",
        model="gemma",
        dispatcher=FakeDispatcher(),
        tool_definitions=[],
        planner_temperature=0.3,
    )
    assert brain.planner_temperature == 0.3


# --- live prompt switching --------------------------------------------------


def test_callable_system_prompt_is_resolved_per_request():
    # Live switching: the brain may carry a provider instead of a frozen
    # string, so a settings change (verbosity) between commands takes
    # effect on the very next command — the prompt is resolved fresh each
    # request. Verbosity lands on the narrator, the phase that has words.
    live = {"verbosity": "normal"}
    provider = ScriptedProvider(text_turn("a"), text_turn("b"))
    brain = LlamaBrain(
        provider=provider,
        dispatcher=FakeDispatcher(),
        tool_definitions=TOOLS,
        system_prompt=lambda: system_prompt_for(verbosity=live["verbosity"]),
        planner_prompt=PLANNER,
    )
    brain.get_agent_response(board_state={}, command="hi")
    assert system_prompts(provider) == [PLANNER, system_prompt_for()]
    live["verbosity"] = "low"
    brain.get_agent_response(board_state={}, command="hi again")
    assert provider.calls[-1]["messages"][0]["content"] == system_prompt_for(
        verbosity="low"
    )


def test_callable_planner_prompt_is_resolved_per_request():
    # Same seam, planner side: the shipped contract is static today, but the
    # wire must keep resolving fresh — the next live-tuned prompt input will
    # arrive on it.
    live = {"suffix": ""}
    provider = ScriptedProvider(text_turn("a"), text_turn("b"))
    brain = LlamaBrain(
        provider=provider,
        dispatcher=FakeDispatcher(),
        tool_definitions=TOOLS,
        system_prompt=PERSONA,
        planner_prompt=lambda: PLANNER_PROMPT + live["suffix"],
    )
    brain.get_agent_response(board_state={}, command="hi")
    assert provider.calls[0]["messages"][0]["content"] == PLANNER_PROMPT
    live["suffix"] = "\nExtra line.\n"
    brain.get_agent_response(board_state={}, command="what should I play?")
    assert provider.calls[2]["messages"][0]["content"] == (
        PLANNER_PROMPT + "\nExtra line.\n"
    )


def test_callable_tool_definitions_are_resolved_per_request():
    # Live capability switching (audit 11): the offer changes with live state
    # (a draw becoming claimable), so it may be a provider too — resolved fresh
    # per command, the same seam as the two prompts.
    live = {"tools": TOOLS[:1]}
    provider = ScriptedProvider(text_turn("a"), text_turn("b"))
    brain = LlamaBrain(
        provider=provider,
        dispatcher=FakeDispatcher(),
        tool_definitions=lambda: live["tools"],
        system_prompt=PERSONA,
        planner_prompt=PLANNER,
    )
    brain.get_agent_response(board_state={}, command="hi")
    assert provider.calls[0]["tools"] == TOOLS[:1]
    live["tools"] = TOOLS
    brain.get_agent_response(board_state={}, command="hi again")
    assert provider.calls[2]["tools"] == TOOLS


def test_a_withheld_tool_is_never_dispatched():
    # The capability restriction's enforcement: a tool outside the resolved
    # offer is a schema-level unknown, even when the registry behind the
    # dispatcher could run it — what the model may call is exactly what it was
    # offered.
    dispatcher = FakeDispatcher()
    offered = [t for t in TOOLS if t["function"]["name"] != "get_best_moves"]
    provider = ScriptedProvider(
        tool_calls_turn(("get_best_moves", {})),
        text_turn("note"),
        text_turn("no hints from me."),
    )
    brain = LlamaBrain(
        provider=provider,
        dispatcher=dispatcher,
        tool_definitions=lambda: offered,
        system_prompt=PERSONA,
        planner_prompt=PLANNER,
    )
    resp = brain.get_agent_response(board_state={}, command="what should I play?")
    assert dispatcher.calls == []
    assert "unknown tool" in resp.tool_results[0]["result"]["error"]


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


def test_create_llama_brain_accepts_a_live_planner_prompt_provider():
    live = {"suffix": ""}
    brain = create_llama_brain(
        base_url="http://localhost:8200/v1",
        model="gemma",
        dispatcher=FakeDispatcher(),
        tool_definitions=[],
        planner_prompt_provider=lambda: PLANNER_PROMPT + live["suffix"],
        provider=ScriptedProvider(text_turn("ok")),
    )
    assert brain.planner_prompt() == PLANNER_PROMPT
    live["suffix"] = "\nExtra line.\n"
    assert brain.planner_prompt() == PLANNER_PROMPT + "\nExtra line.\n"


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


# --- the brain reports its own two phases ------------------------------------
#
# Live progress (audit item 19) needs to say more than "thinking": a turn's two
# model phases are different waits, and the narrator one is also the
# coordinator's observation beat. The brain reports in *its own* vocabulary —
# planning, narrating — and knows nothing about turns or websockets.


def test_the_loop_reports_planning_before_every_planner_call():
    seen: list[str] = []
    brain, _ = make_brain(
        tool_calls_turn(("make_move", {"move": "e4"})),
        text_turn("done"),
        text_turn("Pawn pushed."),
        on_phase=seen.append,
    )
    brain.get_agent_response({"fen": "x"}, "play e4")
    assert seen == ["planning", "planning", "narrating"]


def test_a_plain_answer_reports_one_plan_and_one_narration():
    seen: list[str] = []
    brain, _ = make_brain(
        text_turn("nothing to do"), text_turn("All quiet."), on_phase=seen.append
    )
    brain.get_agent_response({"fen": "x"}, "how's it look?")
    assert seen == ["planning", "narrating"]


def test_a_budget_stop_never_reports_narrating():
    """No narrator runs on a budget stop, so nothing may claim one did — the
    observation beat would open on a turn that never speaks."""
    seen: list[str] = []
    brain, _ = make_brain(
        tool_calls_turn(("make_move", {"move": "e4"})),
        tool_calls_turn(("make_move", {"move": "e5"})),  # new work every turn
        on_phase=seen.append,
        max_iterations=2,
    )
    response = brain.get_agent_response({"fen": "x"}, "play e4")
    assert response.stop_reason == "max_iterations"
    assert set(seen) == {"planning"}


def test_a_provider_death_in_the_loop_never_reports_narrating():
    seen: list[str] = []
    brain, _ = make_brain(ProviderError("llama-server is down"), on_phase=seen.append)
    response = brain.get_agent_response({"fen": "x"}, "play e4")
    assert response.stop_reason == "provider_error"
    assert seen == ["planning"]


def test_the_fast_paths_narration_reports_narrating():
    """`narrate` is the narrator phase on its own, so it reports the same
    phase — which is what makes the observation beat real on the fast path
    whichever brain is behind it."""
    seen: list[str] = []
    brain, _ = make_brain(text_turn("Nice."), on_phase=seen.append)
    brain.narrate({"fen": "x"}, [{"name": "make_move", "result": {"ok": True}}])
    assert seen == ["narrating"]


def test_a_failing_phase_report_never_costs_the_turn():
    def explode(_name):
        raise RuntimeError("socket went away")

    brain, _ = make_brain(text_turn("done"), text_turn("Quiet."), on_phase=explode)
    assert brain.get_agent_response({"fen": "x"}, "hello").text == "Quiet."


def test_a_brain_with_no_observer_is_the_default():
    brain, _ = make_brain(text_turn("done"), text_turn("Quiet."))
    assert brain.get_agent_response({"fen": "x"}, "hello").text == "Quiet."


# --- reading a free-text answer to a pending question (walkthrough #6) --------
#
# One tool-free round trip that returns one of three words. It sits in front of
# a destructive op, so every way it can go wrong has to land on `unrelated` —
# the answer that changes nothing.


@pytest.mark.parametrize(
    ("content", "verdict"),
    [
        ("confirm", CONFIRM),
        ("cancel", CANCEL),
        ("unrelated", UNRELATED),
        # A 12B wraps its one word in whatever it likes.
        ("  Confirm.  ", CONFIRM),
        ("CANCEL", CANCEL),
    ],
)
def test_read_answer_returns_the_verdict(content, verdict):
    brain, _ = make_brain(text_turn(content))
    assert brain.read_answer("Resign?", "just do it").verdict == verdict


@pytest.mark.parametrize(
    "content",
    [
        None,
        "",
        "yes",  # not on the menu — the menu is the contract
        "I think they want to resign, probably",
    ],
)
def test_a_word_off_the_menu_is_unrelated(content):
    """`unrelated` changes nothing, and that is the only safe place for a
    reading nobody can trust to land."""
    brain, _ = make_brain(text_turn(content))
    assert brain.read_answer("Resign?", "mmhm").verdict == UNRELATED


def test_a_dead_provider_cannot_confirm_anything():
    brain, _ = make_brain(ProviderError("llama-server is gone"))
    answer = brain.read_answer("Resign?", "just do it")
    assert answer.verdict == UNRELATED
    assert answer.model_calls == 0


def test_read_answer_offers_no_tools_and_does_not_think():
    """It cannot act on what it reads, and a call in front of a destructive op
    is not a place for a thought loop to live."""
    brain, provider = make_brain(text_turn("confirm"))
    brain.read_answer("Resign?", "just do it")

    call = provider.calls[0]
    assert call["tools"] is None
    assert call["enable_thinking"] is False
    assert call["max_tokens"] == _ANSWER_MAX_TOKENS


def test_read_answer_carries_the_question_and_the_reply():
    """A reply is read against a question: "no, the other one" answers a choice
    and cancels a resignation."""
    brain, provider = make_brain(text_turn("cancel"))
    brain.read_answer("That's the game if you mean it. Resign?", "no, the other one")

    sent = provider.calls[0]["messages"][-1]["content"]
    assert "That's the game if you mean it. Resign?" in sent
    assert "no, the other one" in sent
    assert PERSONA not in provider.calls[0]["messages"][0]["content"], (
        "no persona: this phase reads, it does not speak"
    )


def test_read_answer_is_billed_as_one_call():
    brain, _ = make_brain(
        text_turn("confirm", usage=Usage(prompt_tokens=30, completion_tokens=2))
    )
    answer = brain.read_answer("Resign?", "go on then")
    assert (answer.model_calls, answer.prompt_tokens, answer.completion_tokens) == (
        1,
        30,
        2,
    )
