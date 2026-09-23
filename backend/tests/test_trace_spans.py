"""Where a turn's time went, and what served it (#290, PR 2).

A trace used to carry per-call model latencies and nothing else about time,
and nothing at all about configuration — so a slow turn could not say whether
it waited on the lock, a tool, Stockfish or the guard, and two eval baselines
could not be tied to the prompts and server that produced them. `spans_ms`
answers the first and `serving` the second.
"""

import json
import threading
import time
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from chessapp.api import create_app
from chessapp.brain import AgentResponse, ToolCall
from chessapp.coordinator import TurnCoordinator
from chessapp.game import GameSession
from chessapp.llama_brain import create_llama_brain
from chessapp.tools import ToolContext, build_registry
from chessapp.trace import JsonlTracer, turn_record
from fakes import FakeEngine, ScriptedBrain, ScriptedProvider
from test_llama_brain import FakeDispatcher

IDENTITY = {"planner_prompt": "aaa", "narrator_prompt": "bbb", "model": "m"}


def read_records(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


@pytest.fixture
def trace_path(tmp_path):
    return tmp_path / "turns.jsonl"


def build(trace_path, *responses, engine=None, serving_identity=None, **brain_kw):
    ctx = ToolContext(session=GameSession(), engine=engine)
    coordinator = TurnCoordinator(ctx)
    registry = build_registry(ctx, coordinator, atomic_exchange=False)
    brain = ScriptedBrain(*responses, dispatcher=registry, **brain_kw)
    app = create_app(
        ctx,
        brain=brain,
        registry=registry,
        coordinator=coordinator,
        tracer=JsonlTracer(trace_path),
        serving_identity=serving_identity,
    )
    return TestClient(app), ctx


# --- the record -------------------------------------------------------------


def test_a_record_built_outside_a_request_has_neither():
    record = turn_record(
        utterance="e4",
        route="brain",
        commentary="",
        stop_reason="completed",
        changed=False,
        turn_id=1,
        correlation_id="x",
        mutations=0,
        fen_before="f",
        fen_after="f",
        tool_calls=[],
        tool_results=[],
    )
    assert record["spans_ms"] is None
    assert record["serving"] is None


# --- spans ------------------------------------------------------------------


def test_every_turn_records_its_queue_and_total(trace_path):
    client, _ = build(trace_path, AgentResponse(text="Hi."))
    client.post("/api/command", json={"text": "hello"})
    (record,) = read_records(trace_path)
    spans = record["spans_ms"]
    assert {"queue", "total"} <= spans.keys()
    assert all(isinstance(v, int) and v >= 0 for v in spans.values())
    assert spans["total"] >= max(v for k, v in spans.items() if k != "total")


def test_the_queue_is_the_wait_behind_another_request(trace_path):
    """A request that arrives while another holds the lock waits; that wait is
    `queue`, and it is inside `total`."""
    client, ctx = build(trace_path, AgentResponse(text="Hi."))
    held = threading.Event()

    def hold() -> None:
        with ctx.mutation_lock:
            held.set()
            time.sleep(0.4)

    holder = threading.Thread(target=hold)
    holder.start()
    held.wait(timeout=5)
    client.post("/api/command", json={"text": "hello"})
    holder.join()

    (record,) = read_records(trace_path)
    assert record["spans_ms"]["queue"] >= 200
    assert record["spans_ms"]["total"] >= record["spans_ms"]["queue"]


def test_a_tool_call_records_tool_time(trace_path):
    client, _ = build(
        trace_path,
        AgentResponse(
            text="Here it is.",
            tool_calls=(ToolCall(name="get_board_state", args={}),),
        ),
    )
    client.post("/api/command", json={"text": "show me the board"})
    (record,) = read_records(trace_path)
    assert "tool" in record["spans_ms"]


def test_a_turn_with_no_tool_has_no_tool_span(trace_path):
    """Absent, not zero: a phase that did not run is not one that was fast."""
    client, _ = build(trace_path, AgentResponse(text="Hi."))
    client.post("/api/command", json={"text": "hello"})
    (record,) = read_records(trace_path)
    assert "tool" not in record["spans_ms"]
    assert "engine" not in record["spans_ms"]


def test_a_fast_path_move_records_the_engine_reply(trace_path):
    client, _ = build(trace_path, engine=FakeEngine())
    client.post("/api/command", json={"text": "e4"})
    (record,) = read_records(trace_path)
    assert record["engine_reply"] is not None
    assert {"tool", "engine", "guard"} <= record["spans_ms"].keys()


def test_a_dragged_move_records_the_engine_reply_too(trace_path):
    client, _ = build(trace_path, engine=FakeEngine())
    client.post("/api/game/move", json={"move": "e2e4"})
    (record,) = read_records(trace_path)
    assert record["route"] == "board"
    assert {"queue", "tool", "engine", "guard", "total"} <= record["spans_ms"].keys()


def test_the_guard_span_leaves_the_rewrite_to_model_time(trace_path, monkeypatch):
    """The rewrite is a model call already in `model_ms`; charging it to the
    guard as well would count the same seconds twice."""
    client, _ = build(
        trace_path,
        AgentResponse(text="Word. Game over."),
        rewrites=("Word. Your move.",),
    )
    rewrite = ScriptedBrain.rewrite

    def slow_rewrite(self, *args, **kwargs):
        # A 200 ms round trip that reports its own latency, as the real
        # brain's `rewrite` does.
        time.sleep(0.2)
        return replace(rewrite(self, *args, **kwargs), latency_ms=200)

    monkeypatch.setattr(ScriptedBrain, "rewrite", slow_rewrite)
    client.post("/api/command", json={"text": "i'm bored of this"})

    (record,) = read_records(trace_path)
    assert record["rewrite"] == "spoken"
    assert 200 in record["model_latencies_ms"]
    assert record["spans_ms"]["guard"] < 150


def test_the_registry_reports_each_handler_it_ran_with_its_time():
    ctx = ToolContext(session=GameSession())
    registry = build_registry(ctx, TurnCoordinator(ctx), atomic_exchange=False)
    timed = []
    registry.on_tool_done = lambda name, ms: timed.append((name, ms))

    registry.dispatch("get_board_state", {})
    registry.dispatch("no_such_tool", {})  # never reached a handler

    assert [name for name, _ in timed] == ["get_board_state"]
    assert all(isinstance(ms, int) and ms >= 0 for _, ms in timed)


def test_a_timing_observer_that_raises_costs_nothing():
    ctx = ToolContext(session=GameSession())
    registry = build_registry(ctx, TurnCoordinator(ctx), atomic_exchange=False)

    def boom(name, ms):
        raise RuntimeError("observer died")

    registry.on_tool_done = boom
    assert "fen" in json.dumps(registry.dispatch("get_board_state", {}))


# --- serving identity -------------------------------------------------------


def test_the_record_carries_what_served_the_turn(trace_path):
    client, _ = build(
        trace_path, AgentResponse(text="Hi."), serving_identity=lambda: IDENTITY
    )
    client.post("/api/command", json={"text": "hello"})
    (record,) = read_records(trace_path)
    assert record["serving"] == IDENTITY


def test_an_identity_that_raises_loses_the_field_not_the_record(trace_path):
    def broken():
        raise RuntimeError("tool offer unavailable")

    client, _ = build(trace_path, AgentResponse(text="Hi."), serving_identity=broken)
    client.post("/api/command", json={"text": "hello"})
    (record,) = read_records(trace_path)
    assert record["serving"] is None
    assert record["commentary"] == "Hi."


def make_identity_brain(planner="plan v1", persona="persona v1"):
    return create_llama_brain(
        base_url="http://llama:8080/v1",
        model="gemma-4",
        dispatcher=FakeDispatcher(),
        tool_definitions=lambda: [{"type": "function", "function": {"name": "x"}}],
        system_prompt_provider=lambda: persona,
        planner_prompt_provider=lambda: planner,
        provider=ScriptedProvider(),
    )


def test_the_brain_names_its_model_and_server_and_hashes_the_rest():
    identity = make_identity_brain().serving_identity()
    assert identity["model"] == "gemma-4"
    assert identity["server"] == "http://llama:8080/v1"
    for key in ("planner_prompt", "narrator_prompt", "tool_schemas"):
        assert len(identity[key]) == 12
    assert identity == make_identity_brain().serving_identity(), "stable"


def test_a_prompt_change_changes_only_its_own_hash():
    before = make_identity_brain().serving_identity()
    after = make_identity_brain(planner="plan v2").serving_identity()
    assert after["planner_prompt"] != before["planner_prompt"]
    assert after["narrator_prompt"] == before["narrator_prompt"]
    assert after["tool_schemas"] == before["tool_schemas"]
