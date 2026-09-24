"""`scripts/watch_context.py` — the live context watcher (#359), off the GPU.

The watcher's promise is that the captured text reaches the terminal exactly
as captured, under the turn it belongs to, followed by that turn's decisions
from the trace. These feed it lines the way the two files would and read what
it printed.
"""

from __future__ import annotations

import io
import json
from typing import Any

from watch_context import SOURCE_CONTEXT, SOURCE_TRACE, Tail, Watcher, main

# Deliberately awkward text: indentation, blank lines, trailing spaces, JSON
# with its own spacing. Any reformatting would show.
_PROMPT = "<|turn>system\n  Pick the tools.  \n\n<turn|>\n<|turn>model\n"
_RESPONSE = '{"choices":[{"message":{"content":null,"reasoning_content":"hm  "}}]}'
_REQUEST = '{"model":"gemma-4-12b","messages":[{"role":"user","content":"e4"}]}'


def call(
    correlation_id: str | None = "abc123",
    turn_id: int | None = 7,
    phase: str = "planner",
    seq: int = 1,
    **overrides: Any,
) -> str:
    record = {
        "kind": "model_call",
        "schema": 1,
        "correlation_id": correlation_id,
        "turn_id": turn_id,
        "phase": phase,
        "seq": seq,
        "ms": 812,
        "status_code": 200,
        "request": _REQUEST,
        "response": _RESPONSE,
        "error": "",
        "template": {"prompt": _PROMPT},
        **overrides,
    }
    return json.dumps(record)


def turn(correlation_id: str = "abc123", turn_id: int = 7) -> str:
    return json.dumps(
        {
            "kind": "turn",
            "schema": 2,
            "correlation_id": correlation_id,
            "turn_id": turn_id,
            "utterance": "push my e pawn",
            "route": "brain",
            "stop_reason": "completed",
            "tools": [
                {"name": "make_move", "args": {"move": "e4"}, "result": {"ok": True}}
            ],
            "engine_reply": {"san": "e5", "uci": "e7e5"},
            "guarded": True,
            "guarded_claims": ["check"],
            "suppressed": "Check!",
            "rewrite": "spoken",
            "commentary": "e4. Your move.",
        }
    )


def watch(*lines: tuple[str, str], **kwargs: Any) -> str:
    out = io.StringIO()
    watcher = Watcher(out, **kwargs)
    for line, source in lines:
        watcher.feed(line, source)
    return out.getvalue()


def test_the_prompt_and_response_are_printed_exactly_as_captured():
    printed = watch((call(), SOURCE_CONTEXT))
    assert "turn 7 · corr abc123" in printed
    assert "── planner · seq 1 · 812 ms · HTTP 200 ──" in printed
    assert _PROMPT in printed
    assert _RESPONSE + "\n" in printed
    assert _REQUEST not in printed, "the rendered prompt, not the JSON, by default"


def test_calls_group_under_one_header_and_the_decisions_join_them():
    printed = watch(
        (call(seq=1), SOURCE_CONTEXT),
        (call(seq=2), SOURCE_CONTEXT),
        (call(phase="closer", seq=3), SOURCE_CONTEXT),
        (turn(), SOURCE_TRACE),
    )
    assert printed.count("turn 7 · corr abc123") == 1
    order = [
        printed.index("seq 1"),
        printed.index("seq 2"),
        printed.index("closer · seq 3"),
        printed.index("── decisions ──"),
    ]
    assert order == sorted(order)
    decisions = printed[printed.index("── decisions ──") :]
    assert "utterance: push my e pawn" in decisions
    assert 'tool make_move {"move": "e4"} → {"ok": true}' in decisions
    assert "engine reply: e5" in decisions
    assert "guard: cut [check] · rewrite spoken" in decisions
    assert "commentary: e4. Your move." in decisions


def test_a_new_turn_gets_its_own_header():
    printed = watch(
        (call(), SOURCE_CONTEXT),
        (call(correlation_id="def456", turn_id=8), SOURCE_CONTEXT),
    )
    assert "turn 7 · corr abc123" in printed
    assert "turn 8 · corr def456" in printed


def test_a_call_outside_any_turn_is_labelled_so():
    printed = watch((call(correlation_id=None, turn_id=None), SOURCE_CONTEXT))
    assert "outside a turn" in printed


def test_phase_follows_one_phase_only():
    printed = watch(
        (call(phase="planner", seq=1), SOURCE_CONTEXT),
        (call(phase="closer", seq=2), SOURCE_CONTEXT),
        phase="closer",
    )
    assert "closer · seq 2" in printed
    assert "planner" not in printed


def test_json_prints_the_request_body_instead():
    printed = watch((call(), SOURCE_CONTEXT), show_json=True)
    assert "▸ request\n" + _REQUEST + "\n" in printed
    assert _PROMPT not in printed


def test_a_missing_template_falls_back_to_the_request_and_says_why():
    printed = watch((call(template={"error": "HTTP 404: nope"}), SOURCE_CONTEXT))
    assert "prompt unavailable (HTTP 404: nope)" in printed
    assert _REQUEST in printed


def test_a_failed_call_shows_its_error_and_no_response():
    printed = watch(
        (
            call(status_code=None, response=None, template=None, error="boom"),
            SOURCE_CONTEXT,
        )
    )
    assert "no response" in printed
    assert "▸ error: boom" in printed
    assert "▸ response" not in printed


def test_other_records_and_bad_lines_do_not_stop_the_watch():
    printed = watch(
        (json.dumps({"kind": "serving"}), SOURCE_TRACE),
        ("{not json", SOURCE_CONTEXT),
        (call(), SOURCE_CONTEXT),
    )
    assert "unreadable context line skipped" in printed
    assert "serving" not in printed
    assert _PROMPT in printed


def test_tail_reads_complete_lines_and_survives_truncation(tmp_path):
    path = tmp_path / "context.jsonl"
    tail = Tail(path, from_start=False)
    assert tail.read() == [], "a file that does not exist yet is waited for"
    path.write_text("one\ntw")
    assert tail.read() == ["one"]
    with path.open("a") as handle:
        handle.write("o\n")
    assert tail.read() == ["two"]
    path.write_text("three\n")
    assert tail.read() == ["three"]


def test_tail_starts_at_the_end_unless_asked_to_replay(tmp_path):
    path = tmp_path / "context.jsonl"
    path.write_text("old\n")
    assert Tail(path, from_start=False).read() == []
    assert Tail(path, from_start=True).read() == ["old"]


def test_main_replays_both_files(tmp_path):
    context = tmp_path / "context.jsonl"
    trace = tmp_path / "turns.jsonl"
    context.write_text(call() + "\n")
    trace.write_text(turn() + "\n")
    out = io.StringIO()
    assert (
        main([str(context), "--trace", str(trace), "--from-start", "--once"], out) == 0
    )
    printed = out.getvalue()
    assert _PROMPT in printed and "commentary: e4. Your move." in printed


def test_main_puts_each_turns_decisions_after_its_own_calls(tmp_path):
    context = tmp_path / "context.jsonl"
    trace = tmp_path / "turns.jsonl"
    context.write_text(
        call(ended_at="2026-09-24T10:00:01+00:00")
        + "\n"
        + call(correlation_id="def456", turn_id=8, ended_at="2026-09-24T10:00:05+00:00")
        + "\n"
    )
    first = json.loads(turn())
    second = json.loads(turn(correlation_id="def456", turn_id=8))
    trace.write_text(
        json.dumps({"ts": "2026-09-24T10:00:02+00:00", **first})
        + "\n"
        + json.dumps({"ts": "2026-09-24T10:00:06+00:00", **second})
        + "\n"
    )
    out = io.StringIO()
    main([str(context), "--trace", str(trace), "--from-start", "--once"], out)
    printed = out.getvalue()
    assert printed.count("turn 7 · corr abc123") == 1
    assert printed.count("turn 8 · corr def456") == 1
    assert (
        printed.index("turn 7")
        < printed.index("── decisions ──")
        < printed.index("turn 8")
    )
