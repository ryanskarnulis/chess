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

from chessapp.brain import AgentResponse
from chessapp.game import GameSession
from chessapp.llama_brain import LlamaBrain, create_llama_brain
from chessapp.personality import PLANNER_PROMPT, planner_prompt_for, system_prompt_for
from chessapp.provider import ProviderError, ToolCallArgumentsError, Usage
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
    brain, provider = make_brain(
        tool_calls_turn(("make_move", {"move": "e4"})),  # repeats forever
        max_iterations=3,
    )
    resp = brain.get_agent_response(board_state={}, command="play e4")
    assert resp.stop_reason == "max_iterations"
    assert len(provider.calls) == 3  # the loop's turns only — no narrator
    assert resp.text == ""
    assert len(resp.tool_results) == 3  # everything it did is still reported


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
        tool_calls_turn(("make_move", {"move": "e4"})),  # repeats forever
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
    assert brain.planner_prompt == planner_prompt_for()
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
    # string, so a settings change (verbosity/hints) between commands takes
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
    # Same live-settings seam, planner side: hints mode changes what the loop is
    # told about `get_best_moves`, so the planner prompt is resolved fresh too.
    live = {"hints": False}
    provider = ScriptedProvider(text_turn("a"), text_turn("b"))
    brain = LlamaBrain(
        provider=provider,
        dispatcher=FakeDispatcher(),
        tool_definitions=TOOLS,
        system_prompt=PERSONA,
        planner_prompt=lambda: planner_prompt_for(hints_mode=live["hints"]),
    )
    brain.get_agent_response(board_state={}, command="hi")
    assert provider.calls[0]["messages"][0]["content"] == PLANNER_PROMPT
    live["hints"] = True
    brain.get_agent_response(board_state={}, command="what should I play?")
    assert provider.calls[2]["messages"][0]["content"] == planner_prompt_for(
        hints_mode=True
    )


def test_callable_tool_definitions_are_resolved_per_request():
    # Live capability switching (audit 11): hints gating changes what tools the
    # loop is *offered*, so the offer may be a provider too — resolved fresh per
    # command, the same seam as the two prompts.
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
    # The capability side of hints gating: a tool outside the resolved offer is
    # a schema-level unknown, even when the registry behind the dispatcher
    # could run it — what the model may call is exactly what it was offered.
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
    from chessapp.tools import Settings

    settings = Settings()  # hints default to off
    brain = create_llama_brain(
        base_url="http://localhost:8200/v1",
        model="gemma",
        dispatcher=FakeDispatcher(),
        tool_definitions=[],
        planner_prompt_provider=lambda: planner_prompt_for(settings.hints_mode),
        provider=ScriptedProvider(text_turn("ok")),
    )
    assert brain.planner_prompt() == planner_prompt_for()
    settings.hints_mode = True
    assert brain.planner_prompt() == planner_prompt_for(hints_mode=True)


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
