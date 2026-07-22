"""Turn tracing: the record of what the agent actually did, per command.

The pipeline picks one of four routes for an utterance (deterministic
confirmation, deterministic fast path, deterministic resignation, or the brain's
tool loop) and until now recorded *which* nowhere — so a turn that went wrong
left nothing to review. Every turn writes one JSONL record: the utterance, the
route, the whole tool trajectory (name, args, result), the loop's stop reason,
and whether the honesty guard had to suppress the commentary.

Tracing is diagnostics, never a dependency: a tracer that fails must not cost
the player their turn.
"""

import json

import pytest
from fastapi.testclient import TestClient

from chessapp.brain import AgentResponse, Narration, ToolCall
from chessapp.game import GameSession
from chessapp.tools import Settings, ToolContext
from chessapp.trace import JsonlTracer, turn_record
from fakes import FakeEngine, ScriptedBrain, scripted_app


def read_records(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


@pytest.fixture
def trace_path(tmp_path):
    return tmp_path / "turns.jsonl"


def make_client(trace_path, *responses: AgentResponse, engine=None):
    ctx = ToolContext(session=GameSession(), engine=engine)
    app, brain = scripted_app(ctx, *responses, tracer=JsonlTracer(trace_path))
    return TestClient(app), ctx


# --- the tracer itself ------------------------------------------------------


def test_jsonl_tracer_appends_one_line_per_record(trace_path):
    tracer = JsonlTracer(trace_path)
    tracer.record({"utterance": "e4"})
    tracer.record({"utterance": "undo"})
    records = read_records(trace_path)
    assert [r["utterance"] for r in records] == ["e4", "undo"]
    assert all(r["ts"] for r in records)  # stamped by the tracer


def _record_fields(**overrides):
    fields = {
        "utterance": "e4",
        "route": "brain",
        "commentary": "done",
        "stop_reason": "completed",
        "changed": True,
        "fen_before": "before",
        "fen_after": "after",
        "tool_calls": [],
        "tool_results": [],
    }
    fields.update(overrides)
    return turn_record(**fields)


def test_turn_record_carries_model_cost_when_given():
    record = _record_fields(model_calls=3, prompt_tokens=2841, completion_tokens=96)
    assert record["model_calls"] == 3
    assert record["prompt_tokens"] == 2841
    assert record["completion_tokens"] == 96


def test_turn_record_model_cost_defaults_to_zero():
    """A route that made no model call (a canned confirmation) still records a
    cost — zero — so a reader never has to distinguish 'free' from 'unrecorded'."""
    record = _record_fields()
    assert record["model_calls"] == 0
    assert record["prompt_tokens"] == 0
    assert record["completion_tokens"] == 0


# --- what a turn records ----------------------------------------------------


def test_fast_path_turn_is_traced_as_such(trace_path):
    client, _ = make_client(trace_path)
    client.post("/api/command", json={"text": "e4"})
    (record,) = read_records(trace_path)
    assert record["route"] == "fast_path"
    assert record["utterance"] == "e4"
    assert [t["name"] for t in record["tools"]] == ["make_move"]
    assert record["tools"][0]["args"] == {"move": "e4"}
    assert record["tools"][0]["result"]["legal"] is True
    assert record["changed"] is True


def test_brain_turn_traces_the_whole_trajectory(trace_path):
    """The question tracing exists to answer: what did the model actually call?"""
    response = AgentResponse(
        text="done",
        tool_calls=(ToolCall(name="undo", args={"plies": 2}),),
        tool_results=(),  # the app's registry fills the real result in
        stop_reason="completed",
    )
    client, ctx = make_client(trace_path, response, engine=FakeEngine())
    client.post("/api/command", json={"text": "take that back"})
    (record,) = read_records(trace_path)
    assert record["route"] == "brain"
    assert record["utterance"] == "take that back"
    assert record["stop_reason"] == "completed"
    assert record["commentary"] == "done"
    assert [t["name"] for t in record["tools"]] == ["undo"]
    assert record["tools"][0]["args"] == {"plies": 2}


def test_brain_turn_records_its_model_cost(trace_path):
    """The number every context-shrinking cut is measured against: what the
    turn cost at the provider boundary reaches the trace off the AgentResponse."""
    response = AgentResponse(
        text="done",
        model_calls=3,
        prompt_tokens=2841,
        completion_tokens=96,
    )
    client, _ = make_client(trace_path, response)
    client.post("/api/command", json={"text": "what should I do?"})
    (record,) = read_records(trace_path)
    assert record["model_calls"] == 3
    assert record["prompt_tokens"] == 2841
    assert record["completion_tokens"] == 96


def test_fast_path_records_the_narration_call(trace_path):
    """A fast-path move above verbosity=low narrates — one model call — and that
    single call's cost is what the record must show."""
    ctx = ToolContext(session=GameSession())
    brain = ScriptedBrain(
        dispatcher=None,
        narrations=(Narration(text="e4!", prompt_tokens=40, completion_tokens=6),),
    )
    app, _ = scripted_app(ctx, brain=brain, tracer=JsonlTracer(trace_path))
    TestClient(app).post("/api/command", json={"text": "e4"})
    (record,) = read_records(trace_path)
    assert record["route"] == "fast_path"
    assert record["model_calls"] == 1
    assert record["prompt_tokens"] == 40
    assert record["completion_tokens"] == 6


def test_a_canned_low_verbosity_move_records_zero_cost(trace_path):
    """At verbosity=low a plain move is zero-LLM; the record says so with a real
    zero, not a gap — 'free' is distinguishable from 'unrecorded'."""
    ctx = ToolContext(session=GameSession(), settings=Settings(verbosity="low"))
    app, _ = scripted_app(ctx, tracer=JsonlTracer(trace_path))
    TestClient(app).post("/api/command", json={"text": "e4"})
    (record,) = read_records(trace_path)
    assert record["route"] == "fast_path"
    assert record["model_calls"] == 0
    assert record["prompt_tokens"] == 0
    assert record["completion_tokens"] == 0


def test_resign_turn_is_traced_as_its_own_route(trace_path):
    ctx = ToolContext(session=GameSession())
    for san in ("e4", "e5", "Nf3", "Nc6"):
        ctx.session.submit_move(san)
    app, _ = scripted_app(ctx, tracer=JsonlTracer(trace_path))

    TestClient(app).post("/api/command", json={"text": "i give up. i resign"})

    (record,) = read_records(trace_path)
    assert record["route"] == "resign"
    assert [t["name"] for t in record["tools"]] == ["resign"]
    assert record["changed"] is False, "gated: it asked, it did not end the game"


def test_a_suppressed_claim_is_a_countable_event(trace_path):
    """The guard swaps the lie for the truth, so the record keeps the *event*
    (something tried to fake an ending) even though the lie itself is gone."""
    client, _ = make_client(trace_path, AgentResponse(text="Word. Game over."))

    client.post("/api/command", json={"text": "i'm bored of this"})

    (record,) = read_records(trace_path)
    assert record["guarded"] is True
    assert "Game over" not in record["commentary"]


def test_an_ordinary_turn_is_not_guarded(trace_path):
    client, _ = make_client(trace_path)
    client.post("/api/command", json={"text": "e4"})
    (record,) = read_records(trace_path)
    assert record["guarded"] is False


def test_trace_records_a_budget_stop(trace_path):
    client, _ = make_client(
        trace_path, AgentResponse(text="", stop_reason="max_iterations")
    )
    client.post("/api/command", json={"text": "what should I do?"})
    (record,) = read_records(trace_path)
    assert record["stop_reason"] == "max_iterations"
    assert record["tools"] == []


def test_trace_carries_the_position_the_turn_acted_from(trace_path):
    client, ctx = make_client(trace_path)
    client.post("/api/command", json={"text": "e4"})
    (record,) = read_records(trace_path)
    assert record["fen_before"].startswith("rnbqkbnr/pppppppp")
    assert record["fen_after"] == ctx.session.fen()


def test_a_failing_tracer_never_costs_the_player_their_turn(tmp_path):
    class BrokenTracer:
        def record(self, turn):
            raise OSError("disk full")

    ctx = ToolContext(session=GameSession())
    app, _ = scripted_app(ctx, tracer=BrokenTracer())
    response = TestClient(app).post("/api/command", json={"text": "e4"})
    assert response.status_code == 200
    assert ctx.session.move_history() == ["e4"]


def test_no_tracer_configured_is_fine(trace_path):
    ctx = ToolContext(session=GameSession())
    app, _ = scripted_app(ctx)
    assert TestClient(app).post("/api/command", json={"text": "e4"}).status_code == 200
