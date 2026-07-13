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

from chessapp.provider import ToolCallArgumentsError
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


def test_it_passes_tools_and_max_tokens_through():
    inner = ScriptedProvider(tool_calls_turn(("make_move", {"move": "e4"})))
    provider = CountingProvider(inner)
    tools = [{"type": "function", "function": {"name": "make_move"}}]

    provider.chat(_USER, tools=tools, enable_thinking=True, max_tokens=64)

    assert inner.calls[0]["tools"] == tools
    assert inner.calls[0]["enable_thinking"] is True


def test_a_raising_round_trip_still_counts():
    # A wire-level failure is a round trip the model was paid for: the loop
    # burns a correction for it, so the count must not silently drop it.
    provider = CountingProvider(
        ScriptedProvider(ToolCallArgumentsError("make_move", "not a JSON object"))
    )

    with pytest.raises(ToolCallArgumentsError):
        provider.chat(_USER)

    assert len(provider.calls) == 1


def test_reset_clears_the_record():
    # Each eval scenario measures one utterance; the fixture is reused.
    provider = CountingProvider(ScriptedProvider(text_turn("a")))
    provider.chat(_USER)

    provider.reset()

    assert provider.calls == []
