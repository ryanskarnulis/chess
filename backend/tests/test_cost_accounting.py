"""Every model round trip a turn made reaches its trace cost (#290).

The trace's cost fields are what every context cut and latency campaign is
measured against, and they used to under-report in two ways. A turn that
passed through more than one phase — a reply the reader called `unrelated`
going on down the brain road — kept only the last phase's cost, because each
route *assigned* its cost instead of adding to it. And a round trip that
raised — a dead reader, an observe beat the provider killed or the budget cut,
a lost rewrite — left no call and no latency behind, though the turn had waited
on it.

The rule these pin: `model_calls` is the calls attempted, each one has a
latency, and a call that reported no usage is counted in `unmetered_calls`
rather than passing as a measured zero.
"""

import json

import pytest
from fastapi.testclient import TestClient

from chessapp.brain import UNRELATED, AgentResponse, Answer, Narration
from chessapp.game import GameSession
from chessapp.provider import ProviderError, ProviderRequestError, Usage
from chessapp.tools import ToolContext
from chessapp.trace import JsonlTracer
from fakes import FakeEngine, ScriptedBrain, scripted_app, text_turn
from test_llama_brain import make_brain


def read_records(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


@pytest.fixture
def trace_path(tmp_path):
    return tmp_path / "turns.jsonl"


def client_for(trace_path, brain, *, engine=None, opening=()):
    ctx = ToolContext(session=GameSession(), engine=engine)
    for san in opening:
        assert ctx.session.submit_move(san).legal
    app, _ = scripted_app(ctx, brain=brain, tracer=JsonlTracer(trace_path))
    return TestClient(app), ctx


def arm_resign(client):
    """Put a resign question in front of the player: the gate asks, nothing
    runs, and no model call is made (the resign route is deterministic)."""
    client.post("/api/command", json={"text": "i resign"})


# --- the reader's round trip is added to whatever road the words take ------


def test_an_unrelated_reading_adds_to_the_brain_route_it_falls_through_to(
    trace_path,
):
    """The issue's reproduction: the reader's call and the brain's two are
    three calls, not the brain's two."""
    brain = ScriptedBrain(
        AgentResponse(text="Here's the board.", model_calls=2, prompt_tokens=300),
        answers=(Answer(verdict=UNRELATED, model_calls=1, prompt_tokens=80),),
    )
    client, _ = client_for(trace_path, brain, opening=("e4", "e5"))
    arm_resign(client)

    client.post("/api/command", json={"text": "actually show me the position instead"})

    record = read_records(trace_path)[-1]
    assert record["route"] == "brain"
    assert record["model_calls"] == 3
    assert record["prompt_tokens"] == 380


def test_an_unrelated_reading_adds_to_the_fast_path_it_falls_through_to(
    trace_path,
):
    brain = ScriptedBrain(
        answers=(Answer(verdict=UNRELATED, model_calls=1, prompt_tokens=80),),
        narrations=(Narration(text="Nf3, sure.", prompt_tokens=40),),
    )
    client, _ = client_for(trace_path, brain, engine=FakeEngine(), opening=("e4", "e5"))
    arm_resign(client)

    client.post("/api/command", json={"text": "Nf3"})

    record = read_records(trace_path)[-1]
    assert record["route"] == "fast_path"
    assert record["model_calls"] == 2
    assert record["prompt_tokens"] == 120


def test_a_reader_that_died_is_still_a_call_on_the_turn(trace_path):
    """The real brain reports a dead reader as one unmetered call
    (`LlamaBrain.read_answer`); the pipeline must not drop it for having
    produced no verdict."""
    brain = ScriptedBrain(
        AgentResponse(text="Sure.", model_calls=1, prompt_tokens=200),
        answers=(
            Answer(verdict=UNRELATED, model_calls=1, unmetered_calls=1, latency_ms=900),
        ),
    )
    client, _ = client_for(trace_path, brain, opening=("e4", "e5"))
    arm_resign(client)

    client.post("/api/command", json={"text": "just get on with the game"})

    record = read_records(trace_path)[-1]
    assert record["model_calls"] == 2
    assert record["unmetered_calls"] == 1
    assert record["prompt_tokens"] == 200
    assert 900 in record["model_latencies_ms"]


def test_the_real_reader_counts_a_dead_provider_as_one_timed_call():
    brain, _ = make_brain(ProviderRequestError("connection refused"))
    answer = brain.read_answer("Resign?", "just do it")
    assert answer.verdict == UNRELATED
    assert (answer.model_calls, answer.unmetered_calls) == (1, 1)
    assert len(answer.model_latencies_ms) == 1


# --- optional words that raised are still round trips ----------------------


def test_a_dead_observe_beat_is_one_call_on_the_fast_path(trace_path):
    brain = ScriptedBrain(narrations=(ProviderError("llama-server is gone"),))
    client, _ = client_for(trace_path, brain, engine=FakeEngine())

    client.post("/api/command", json={"text": "e4"})

    (record,) = read_records(trace_path)
    assert record["route"] == "fast_path"
    assert record["model_calls"] == 1
    assert record["unmetered_calls"] == 1
    assert record["prompt_tokens"] == 0
    assert len(record["model_latencies_ms"]) == 1


def test_a_dead_observe_beat_is_one_call_on_a_drag(trace_path):
    brain = ScriptedBrain(narrations=(ProviderError("llama-server is gone"),))
    client, _ = client_for(trace_path, brain, engine=FakeEngine())

    client.post("/api/game/move", json={"move": "e2e4"})

    (record,) = read_records(trace_path)
    assert record["route"] == "board"
    assert record["model_calls"] == 1
    assert record["unmetered_calls"] == 1
    assert len(record["model_latencies_ms"]) == 1


def test_a_dead_narration_of_a_confirmed_op_is_one_call(trace_path):
    """A bare "yes" is read deterministically, so the only round trip is the
    narration of what the resignation did — and it died."""
    brain = ScriptedBrain(narrations=(ProviderError("llama-server is gone"),))
    client, ctx = client_for(trace_path, brain, opening=("e4", "e5"))
    arm_resign(client)

    client.post("/api/command", json={"text": "yes"})

    assert ctx.session.is_game_over()
    record = read_records(trace_path)[-1]
    assert record["route"] == "confirmation"
    assert record["model_calls"] == 1
    assert record["unmetered_calls"] == 1


def test_a_lost_rewrite_is_a_call_beside_the_draft_it_was_asked_to_fix(
    trace_path,
):
    brain = ScriptedBrain(
        AgentResponse(
            text="Word. Game over.",
            model_calls=2,
            prompt_tokens=500,
            model_latencies_ms=(300, 700),
        ),
        rewrites=(ProviderError("llama-server is gone"),),
    )
    client, _ = client_for(trace_path, brain)

    client.post("/api/command", json={"text": "i'm bored of this"})

    (record,) = read_records(trace_path)
    assert record["rewrite"] == "lost"
    assert record["model_calls"] == 3
    assert record["unmetered_calls"] == 1
    assert record["prompt_tokens"] == 500
    assert len(record["model_latencies_ms"]) == 3


# --- a call with no usage is unknown, not free ------------------------------


def test_a_narration_with_no_usage_is_unmetered():
    brain, _ = make_brain(text_turn("Nice."))
    narration = brain.narrate({}, [])
    assert narration.model_calls == 1
    assert narration.unmetered_calls == 1


def test_a_narration_with_usage_is_metered():
    brain, _ = make_brain(
        text_turn("Nice.", usage=Usage(prompt_tokens=40, completion_tokens=3))
    )
    assert brain.narrate({}, []).unmetered_calls == 0


def test_the_loop_counts_its_raised_and_usage_less_calls_as_unmetered():
    """A planner turn that reported usage and a narrator that died: two calls,
    one of them unmetered, the tokens the one that was measured."""
    brain, _ = make_brain(
        text_turn("nothing to do", usage=Usage(prompt_tokens=900, completion_tokens=4)),
        ProviderError("llama-server is gone"),
    )
    response = brain.get_agent_response({"fen": "startpos"}, "hi")
    assert response.stop_reason == "provider_error"
    assert response.model_calls == 2
    assert response.unmetered_calls == 1
    assert response.prompt_tokens == 900
    assert len(response.model_latencies_ms) == 2
