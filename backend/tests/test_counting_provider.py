"""`CountingProvider` — the eval harness's model-round-trip recorder.

The agent evals assert on how many times the *model* was called (a plain move
on the fast path must be zero; a tool-using utterance is the tool turn plus the
loop's closing turn) and on the thinking flag per turn. Nothing in production
counts either — `ChatProvider` is the only seam every round trip passes
through, so the harness wraps it.

These tests pin the wrapper itself, over a `ScriptedProvider` inner, so the
thing the live evals trust is verified without a GPU.
"""

import pytest

from chessapp.provider import ToolCallArgumentsError, Usage
from fakes import CountingProvider, ScriptedProvider, text_turn, tool_calls_turn

_USER = [{"role": "user", "content": "hi"}]


def test_it_returns_the_inner_result_untouched():
    turn = text_turn("nice opening")
    provider = CountingProvider(ScriptedProvider(turn))

    assert provider.chat(_USER) is turn


def test_it_records_one_call_per_round_trip():
    provider = CountingProvider(ScriptedProvider(text_turn("a")))

    assert provider.calls == []
    provider.chat(_USER)
    provider.chat(_USER)

    assert len(provider.calls) == 2


def test_it_records_the_thinking_flag_per_call():
    provider = CountingProvider(ScriptedProvider(text_turn("a")))

    provider.chat(_USER, enable_thinking=False)
    provider.chat(_USER, enable_thinking=True)

    assert [call.thinking for call in provider.calls] == [False, True]


def test_it_times_each_round_trip():
    provider = CountingProvider(ScriptedProvider(text_turn("a")))

    provider.chat(_USER)

    assert provider.calls[0].seconds >= 0.0


def test_it_passes_tools_max_tokens_and_temperature_through():
    inner = ScriptedProvider(tool_calls_turn(("make_move", {"move": "e4"})))
    provider = CountingProvider(inner)
    tools = [{"type": "function", "function": {"name": "make_move"}}]

    provider.chat(
        _USER, tools=tools, enable_thinking=True, max_tokens=64, temperature=0.2
    )

    assert inner.calls[0]["tools"] == tools
    assert inner.calls[0]["enable_thinking"] is True
    # The evals measure the planner's own sampling, so the decorator must not
    # quietly swallow the knob on its way to the live wire.
    assert inner.calls[0]["temperature"] == 0.2


def test_it_records_the_token_usage_of_each_round_trip():
    """Sprint 5's mechanism question. Latency alone cannot tell "the narrator
    emitted more reasoning tokens" from "generation was slower" — in
    milliseconds those look identical. Usage per *call* separates them, and this
    seam is the only place it exists: the trace sums the turn's tokens, and a
    sum has the same blind spot `model_ms` had."""
    inner = ScriptedProvider(
        text_turn("planning", usage=Usage(prompt_tokens=2100, completion_tokens=8)),
        text_turn("narrating", usage=Usage(prompt_tokens=2900, completion_tokens=940)),
    )
    provider = CountingProvider(inner)

    provider.chat(_USER)
    provider.chat(_USER)

    assert [call.prompt_tokens for call in provider.calls] == [2100, 2900]
    assert [call.completion_tokens for call in provider.calls] == [8, 940]


def test_a_result_that_reports_no_usage_is_unknown_not_zero():
    # llama-server always sends usage, but a provider that does not must not be
    # recorded as a free call — the same rule the latency reading follows for an
    # unmeasured phase.
    provider = CountingProvider(ScriptedProvider(text_turn("a", usage=None)))

    provider.chat(_USER)

    assert provider.calls[0].prompt_tokens is None
    assert provider.calls[0].completion_tokens is None


def test_a_raising_round_trip_still_counts():
    # A wire-level failure is a round trip the model was paid for: the loop
    # burns a correction for it, so the count must not silently drop it. Its
    # tokens are unknown rather than zero: the result never came back, and the
    # model generated *something* before the wire rejected it.
    provider = CountingProvider(
        ScriptedProvider(ToolCallArgumentsError("make_move", "not a JSON object"))
    )

    with pytest.raises(ToolCallArgumentsError):
        provider.chat(_USER)

    assert len(provider.calls) == 1
    assert provider.calls[0].prompt_tokens is None
    assert provider.calls[0].completion_tokens is None


def test_reset_clears_the_record():
    # Each eval scenario measures one utterance; the fixture is reused.
    provider = CountingProvider(ScriptedProvider(text_turn("a")))
    provider.chat(_USER)

    provider.reset()

    assert provider.calls == []
