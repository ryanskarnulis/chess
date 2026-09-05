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

from chessapp.brain import CONFIRM, AgentResponse, Answer, Narration, ToolCall
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
        "turn_id": 1,
        "correlation_id": "c0ffee",
        "mutations": 0,
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


def test_turn_record_carries_the_engine_reply_when_there_was_one():
    record = _record_fields(engine_reply={"san": "e5", "uci": "e7e5"})
    assert record["engine_reply"] == {"san": "e5", "uci": "e7e5"}


def test_turn_record_engine_reply_defaults_to_none():
    assert _record_fields()["engine_reply"] is None


def test_turn_record_model_cost_defaults_to_zero():
    """A route that made no model call (a canned confirmation) still records a
    cost — zero — so a reader never has to distinguish 'free' from 'unrecorded'."""
    record = _record_fields()
    assert record["model_calls"] == 0
    assert record["prompt_tokens"] == 0
    assert record["completion_tokens"] == 0


def test_turn_record_carries_the_ids_that_locate_the_turn():
    record = _record_fields(turn_id=7, correlation_id="ab12cd34")
    assert record["turn_id"] == 7
    assert record["correlation_id"] == "ab12cd34"


def test_turn_record_carries_the_mutation_count():
    record = _record_fields(mutations=2)
    assert record["mutations"] == 2


def test_turn_record_times_each_model_call_and_sums_them():
    """Per-call latency is what tells a slow narrator from a slow planner; the
    total is derived here, so the two can never disagree in a record."""
    record = _record_fields(model_calls=2, model_latencies_ms=[120, 900])
    assert record["model_latencies_ms"] == [120, 900]
    assert record["model_ms"] == 1020


def test_turn_record_latency_defaults_to_no_calls():
    record = _record_fields()
    assert record["model_latencies_ms"] == []
    assert record["model_ms"] == 0


def test_turn_record_names_the_provider_failure_when_the_turn_died():
    """`provider_error` says the turn died; this says what killed it. Reading a
    trace, that is the difference between "llama-server is crash-looping" and
    "this prompt no longer fits the context" — and the two want opposite fixes."""
    record = _record_fields(stop_reason="provider_error", provider_failure="rejected")
    assert record["provider_failure"] == "rejected"


def test_turn_record_provider_failure_defaults_to_none_named():
    """Every other turn records the empty string, not a missing key: a reader
    never has to tell "did not die" from "not recorded"."""
    assert _record_fields()["provider_failure"] == ""


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


def test_a_move_turn_records_the_engine_reply(trace_path):
    """The reply is no longer inside `make_move`'s result, so without this the
    trace of a move turn would show the player's move and no answer to it — and
    a duplicated or missing engine move is exactly what a trace is read for."""
    client, _ = make_client(trace_path, engine=FakeEngine("e7e5"))
    client.post("/api/command", json={"text": "e4"})
    (record,) = read_records(trace_path)
    assert record["tools"][0]["result"].get("engine_move") is None
    assert record["engine_reply"] == {"san": "e5", "uci": "e7e5"}


def test_a_turn_that_owed_no_reply_records_none(trace_path):
    client, _ = make_client(trace_path, AgentResponse(text="hi"), engine=FakeEngine())
    client.post("/api/command", json={"text": "hello"})
    (record,) = read_records(trace_path)
    assert record["engine_reply"] is None


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


def test_a_confirmed_draw_claim_is_traced_as_one_mutation(trace_path):
    """A claim ends the game, so the record has to show exactly one board change
    on the confirmation turn — no engine reply is owed to a finished game."""
    ctx = ToolContext(session=GameSession())
    for _ in range(2):
        for san in ("Nf3", "Nf6", "Ng1", "Ng8"):
            ctx.session.submit_move(san)
    app, _ = scripted_app(
        ctx,
        AgentResponse(text="sure?", tool_calls=(ToolCall(name="claim_draw", args={}),)),
        tracer=JsonlTracer(trace_path),
    )
    client = TestClient(app)

    client.post("/api/command", json={"text": "can we call it a draw?"})
    client.post("/api/command", json={"text": "yes"})

    asked, confirmed = read_records(trace_path)
    assert asked["route"] == "brain"
    assert asked["mutations"] == 0, "the gate only asked"
    assert confirmed["route"] == "confirmation"
    assert [t["name"] for t in confirmed["tools"]] == ["claim_draw"]
    assert confirmed["mutations"] == 1
    assert confirmed["changed"] is True


def test_a_suppressed_claim_is_a_countable_event(trace_path):
    """The guard swaps the lie for the truth, so `commentary` is what the player
    saw — and the record keeps the event beside it."""
    client, _ = make_client(trace_path, AgentResponse(text="Word. Game over."))

    client.post("/api/command", json={"text": "i'm bored of this"})

    (record,) = read_records(trace_path)
    assert record["guarded"] is True
    assert "Game over" not in record["commentary"]


def test_a_suppressed_claim_records_what_it_was_and_why(trace_path):
    """A guard that eats its own evidence is a guard nobody can debug. Twice now
    a live misfire has been diagnosed by guessing, because the suppressed text
    survived nowhere: not in `commentary` (replaced), not in the transcript
    (`api._remembered_facts` keeps it out on purpose) and not in the log (the
    classes rode in `extra`, which the default formatter drops). So the record
    keeps both halves — what was said, and which classes the facts didn't
    back."""
    client, _ = make_client(trace_path, AgentResponse(text="Word. Game over."))

    client.post("/api/command", json={"text": "i'm bored of this"})

    (record,) = read_records(trace_path)
    assert record["guarded_claims"] == ["ending"]
    assert record["suppressed"] == "Word. Game over."


def test_a_leaked_hint_records_what_it_was_too(trace_path):
    """The advice guard is a suppression like any other, and it named nothing in
    the record at all — a turn cut for leaking a move looked identical to one cut
    for inventing a capture."""
    client, _ = make_client(
        trace_path, AgentResponse(text="Easy: play Nf3 and thank me later.")
    )

    client.post("/api/command", json={"text": "what should I play?"})

    (record,) = read_records(trace_path)
    assert record["guarded"] is True
    assert record["guarded_claims"] == ["move_advice"]
    assert record["suppressed"] == "Easy: play Nf3 and thank me later."


def test_an_ordinary_turn_is_not_guarded(trace_path):
    client, _ = make_client(trace_path)
    client.post("/api/command", json={"text": "e4"})
    (record,) = read_records(trace_path)
    assert record["guarded"] is False
    assert record["guarded_claims"] == []
    assert record["suppressed"] == ""


def test_trace_records_a_budget_stop(trace_path):
    client, _ = make_client(
        trace_path, AgentResponse(text="", stop_reason="max_iterations")
    )
    client.post("/api/command", json={"text": "what should I do?"})
    (record,) = read_records(trace_path)
    assert record["stop_reason"] == "max_iterations"
    assert record["tools"] == []


def test_trace_carries_the_provider_failure_through_the_pipeline(trace_path):
    """End to end: the kind the brain named reaches the record on disk. This is
    the seam the eval harness reads a sample's classification off, so a break
    here is a harness that retries a deterministic failure five times."""
    client, _ = make_client(
        trace_path,
        AgentResponse(
            text="", stop_reason="provider_error", provider_failure="rejected"
        ),
    )
    client.post("/api/command", json={"text": "what should I do?"})
    (record,) = read_records(trace_path)
    assert record["stop_reason"] == "provider_error"
    assert record["provider_failure"] == "rejected"


def test_a_healthy_turn_names_no_provider_failure(trace_path):
    client, _ = make_client(trace_path)
    client.post("/api/command", json={"text": "e4"})
    (record,) = read_records(trace_path)
    assert record["provider_failure"] == ""


def test_trace_carries_the_position_the_turn_acted_from(trace_path):
    client, ctx = make_client(trace_path)
    client.post("/api/command", json={"text": "e4"})
    (record,) = read_records(trace_path)
    assert record["fen_before"].startswith("rnbqkbnr/pppppppp")
    assert record["fen_after"] == ctx.session.fen()


# --- locating a turn: the ids, the mutations, the latencies -----------------


def test_a_move_turn_counts_both_of_its_mutations(trace_path):
    """A move turn changes the board twice — the player's move and the engine's
    answer — so two is what a healthy record says. Three is the bug this counts
    for: a move that landed twice under one turn id."""
    client, _ = make_client(trace_path, engine=FakeEngine("e7e5"))
    client.post("/api/command", json={"text": "e4"})
    (record,) = read_records(trace_path)
    assert record["mutations"] == 2


def test_an_engineless_move_turn_counts_one(trace_path):
    client, _ = make_client(trace_path)
    client.post("/api/command", json={"text": "e4"})
    (record,) = read_records(trace_path)
    assert record["mutations"] == 1


def test_a_turn_that_touched_nothing_counts_no_mutations(trace_path):
    client, _ = make_client(trace_path, AgentResponse(text="Ruy Lopez, obviously."))
    client.post("/api/command", json={"text": "favorite opening?"})
    (record,) = read_records(trace_path)
    assert record["mutations"] == 0
    assert record["changed"] is False


def test_each_turn_records_the_coordinator_turn_it_ran_under(trace_path):
    """The id the mutation count is read against: two mutations under one turn
    id is a turn, four under one id is a duplicated move."""
    client, _ = make_client(trace_path, engine=FakeEngine("e7e5"))
    client.post("/api/command", json={"text": "e4"})
    client.post("/api/command", json={"text": "Nf3"})
    assert [r["turn_id"] for r in read_records(trace_path)] == [1, 2]


def test_every_turn_gets_its_own_correlation_id(trace_path):
    """One id per user interaction, tying the record to the log lines that turn
    emitted — so two turns are never confused for one."""
    client, _ = make_client(trace_path, engine=FakeEngine("e7e5"))
    client.post("/api/command", json={"text": "e4"})
    client.post("/api/command", json={"text": "Nf3"})
    ids = [r["correlation_id"] for r in read_records(trace_path)]
    assert all(ids), "every turn is locatable"
    assert len(set(ids)) == 2


def test_a_dragged_move_is_traced_with_the_same_accounting(trace_path):
    """The board route is a turn like any other: same ids, same mutation count."""
    client, _ = make_client(trace_path, engine=FakeEngine("e7e5"))
    client.post("/api/game/move", json={"move": "e2e4"})
    (record,) = read_records(trace_path)
    assert record["route"] == "board"
    assert record["turn_id"] == 1
    assert record["mutations"] == 2
    assert record["correlation_id"]


def test_brain_turn_records_a_latency_per_model_call(trace_path):
    client, _ = make_client(
        trace_path,
        AgentResponse(text="done", model_calls=2, model_latencies_ms=(300, 800)),
    )
    client.post("/api/command", json={"text": "what should I do?"})
    (record,) = read_records(trace_path)
    assert record["model_latencies_ms"] == [300, 800]
    assert record["model_ms"] == 1100


def test_fast_path_records_the_narration_latency(trace_path):
    ctx = ToolContext(session=GameSession())
    brain = ScriptedBrain(
        dispatcher=None, narrations=(Narration(text="e4!", latency_ms=140),)
    )
    app, _ = scripted_app(ctx, brain=brain, tracer=JsonlTracer(trace_path))
    TestClient(app).post("/api/command", json={"text": "e4"})
    (record,) = read_records(trace_path)
    assert record["model_latencies_ms"] == [140]
    assert record["model_ms"] == 140


def test_a_zero_llm_turn_records_no_latencies(trace_path):
    ctx = ToolContext(session=GameSession(), settings=Settings(verbosity="low"))
    app, _ = scripted_app(ctx, tracer=JsonlTracer(trace_path))
    TestClient(app).post("/api/command", json={"text": "e4"})
    (record,) = read_records(trace_path)
    assert record["model_latencies_ms"] == []
    assert record["model_ms"] == 0


# --- the control surface: a destructive op leaves a record either way -------


def _in_progress(trace_path, **kwargs):
    """A client on a game with moves on it — the state where the gate asks."""
    client, ctx = make_client(trace_path, **kwargs)
    for san in ("e4", "e5", "Nf3"):
        ctx.session.submit_move(san)
    return client, ctx


def test_an_armed_reset_button_is_traced_as_asking(trace_path):
    """The gate refused and asked. Nothing changed — and that is the record's
    point: it is the half of the interaction the spoken road already traced."""
    client, _ = _in_progress(trace_path)
    assert client.post("/api/game/new", json={}).status_code == 409
    (record,) = read_records(trace_path)
    assert record["route"] == "control"
    assert record["utterance"] == "new_game"
    assert record["mutations"] == 0
    assert record["changed"] is False
    assert record["tools"][0]["result"]["ok"] is False


def test_a_button_confirmed_reset_is_traced(trace_path):
    """The gap `docs/turn-coordinator.md` left for this slice: `/api/game/confirm`
    ran a real destructive op and wrote nothing down."""
    client, _ = _in_progress(trace_path)
    client.post("/api/game/new", json={})  # arms it (409 + the question)
    client.post("/api/game/confirm", json={"confirm": True})
    asked, confirmed = read_records(trace_path)
    assert asked["mutations"] == 0
    assert confirmed["route"] == "control"
    assert confirmed["utterance"] == "new_game"
    assert confirmed["changed"] is True
    assert confirmed["mutations"] == 1
    assert confirmed["model_calls"] == 0, "no model stands between a yes and a reset"


def test_a_declined_reset_is_traced_as_having_run_nothing(trace_path):
    client, _ = _in_progress(trace_path)
    client.post("/api/game/new", json={})
    client.post("/api/game/confirm", json={"confirm": False})
    _, declined = read_records(trace_path)
    assert declined["mutations"] == 0
    assert declined["changed"] is False
    assert declined["tools"] == [], "nothing was dispatched to report"


def test_a_control_record_carries_the_turn_it_answered_in(trace_path):
    client, _ = _in_progress(trace_path, engine=FakeEngine("e7e5"))
    client.post("/api/command", json={"text": "Nc6"})  # turn 1, then turn 2 opens
    client.post("/api/game/resign", json={})
    _, asked = read_records(trace_path)
    assert asked["turn_id"] == 2
    assert asked["correlation_id"]


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


def test_a_free_text_confirmation_bills_both_of_its_round_trips(trace_path):
    """The confirmation route can now spend two model calls — reading what the
    player's reply meant, then narrating what the op did — and a turn that
    recorded only the second would under-report every free-text confirmation
    (walkthrough #6)."""
    ctx = ToolContext(session=GameSession())
    for san in ("e4", "e5", "Nf3", "Nc6"):
        assert ctx.session.submit_move(san).legal
    brain = ScriptedBrain(
        AgentResponse(text="you sure?", tool_calls=(ToolCall(name="resign", args={}),)),
        answers=(Answer(verdict=CONFIRM, model_calls=1, prompt_tokens=40),),
        narrations=(Narration(text="That's game.", model_calls=1, prompt_tokens=310),),
    )
    app, _ = scripted_app(ctx, brain=brain, tracer=JsonlTracer(trace_path))
    client = TestClient(app)

    client.post("/api/command", json={"text": "i resign"})
    client.post("/api/command", json={"text": "get on with it"})

    assert ctx.session.is_game_over()
    confirmation = read_records(trace_path)[-1]
    assert confirmation["route"] == "confirmation"
    assert confirmation["model_calls"] == 2
    assert confirmation["prompt_tokens"] == 350
