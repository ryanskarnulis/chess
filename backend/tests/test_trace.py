"""Turn tracing: the record of what the agent actually did, per command.

The pipeline picks one of three routes for an utterance (deterministic
confirmation, deterministic fast path, or the brain's tool loop) and until now
recorded *which* nowhere — so a turn that went wrong left nothing to review.
Every turn writes one JSONL record: the utterance, the route, the whole tool
trajectory (name, args, result) and the loop's stop reason.

Tracing is diagnostics, never a dependency: a tracer that fails must not cost
the player their turn.
"""

import json

import pytest
from fastapi.testclient import TestClient

from chessapp.brain import AgentResponse, ToolCall
from chessapp.game import GameSession
from chessapp.tools import ToolContext
from chessapp.trace import JsonlTracer
from fakes import FakeEngine, scripted_app


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
