"""One voice interaction, followed across its records (#317).

The browser mints an interaction id when the player stops speaking and sends
it with the transcription, the command and the speech request; the server
writes one `speech` record per speech round trip, the turn record, and — from
the browser's own report — one `voice` record of client milestones. These pin
the server's half: the id reaches every record, no audio or words are stored,
the report is validated before a byte of it is written, and none of it can
cost the player anything.
"""

import json
from typing import Any

import pytest
from fastapi.testclient import TestClient

from chessapp.api import create_app
from chessapp.brain import AgentResponse
from chessapp.game import GameSession
from chessapp.tools import ToolContext
from chessapp.trace import JsonlTracer
from chessapp.voice import create_speech_client
from fakes import ScriptedBrain, scripted_app
from test_voice import FakeSpeechServer

ID = "a1b2c3d4e5f6"


def records(path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


@pytest.fixture
def trace_path(tmp_path):
    return tmp_path / "turns.jsonl"


def speech_app(trace_path, server: FakeSpeechServer | None = None) -> TestClient:
    speech = create_speech_client(
        base_url="http://speaches:8000/v1",
        client=(server or FakeSpeechServer()).client(),
    )
    ctx = ToolContext(session=GameSession())
    return TestClient(create_app(ctx, speech=speech, tracer=JsonlTracer(trace_path)))


def report(**overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "interaction_id": ID,
        "correlation_id": "c0ffee000001",
        "origin": "voice",
        "clock": "client_monotonic_ms",
        "start": "speech_end",
        "marks": {
            "stt_done": 2400,
            "command_sent": 2410,
            "first_board_update": 3100,
            "engine_reply": 3900,
            "command_done": 5200,
            "tts_requested": 5210,
            "tts_ready": 6000,
            "playback_started": 6050,
            "playback_ended": 9800,
        },
        "outcome": "ended",
        "censored": False,
    }
    body.update(overrides)
    return body


# --- speech records -----------------------------------------------------------


def test_a_transcription_is_recorded_by_interaction_without_the_audio(trace_path):
    client = speech_app(trace_path, FakeSpeechServer(text="pawn to e4"))

    response = client.post(
        "/api/voice/transcribe",
        files={"audio": ("clip.wav", b"RIFF....", "audio/wav")},
        headers={"X-Interaction-Id": ID},
    )

    assert response.json() == {"text": "pawn to e4"}
    (record,) = records(trace_path)
    assert record["schema"] == 2
    assert record["kind"] == "speech"
    assert record["op"] == "stt"
    assert record["interaction_id"] == ID
    assert record["status"] == "ok"
    assert record["bytes"] == 8
    assert record["chars"] == len("pawn to e4")
    assert isinstance(record["ms"], int)
    assert "pawn" not in json.dumps(record)


def test_a_synthesis_is_recorded_by_interaction_without_the_words(trace_path):
    client = speech_app(trace_path, FakeSpeechServer(audio=b"mp3-bytes"))

    client.post(
        "/api/voice/speak",
        json={"text": "Bold move."},
        headers={"X-Interaction-Id": ID},
    )

    (record,) = records(trace_path)
    assert (record["kind"], record["op"], record["status"]) == ("speech", "tts", "ok")
    assert record["interaction_id"] == ID
    assert (record["chars"], record["bytes"]) == (len("Bold move."), len(b"mp3-bytes"))
    assert "Bold" not in json.dumps(record)


def test_a_failed_speech_call_is_recorded_as_failed(trace_path):
    client = speech_app(trace_path, FakeSpeechServer(status=500))

    response = client.post(
        "/api/voice/speak",
        json={"text": "Bold move."},
        headers={"X-Interaction-Id": ID},
    )

    assert response.status_code == 502
    (record,) = records(trace_path)
    assert (record["op"], record["status"]) == ("tts", "failed")


def test_a_malformed_interaction_header_is_dropped_not_refused(trace_path):
    """Refusing would cost the player their speech over a diagnostic."""
    client = speech_app(trace_path)

    response = client.post(
        "/api/voice/speak",
        json={"text": "hi"},
        headers={"X-Interaction-Id": "<script>alert(1)</script>"},
    )

    assert response.status_code == 200
    assert records(trace_path)[0]["interaction_id"] == ""


def test_speech_works_untraced_and_with_a_tracer_that_raises():
    class Broken:
        def record(self, _turn):
            raise OSError("disk full")

    speech = create_speech_client(
        base_url="http://speaches:8000/v1", client=FakeSpeechServer().client()
    )
    for tracer in (None, Broken()):
        ctx = ToolContext(session=GameSession())
        client = TestClient(create_app(ctx, speech=speech, tracer=tracer))
        assert client.post("/api/voice/speak", json={"text": "hi"}).status_code == 200


# --- the command -----------------------------------------------------------------


def test_the_command_carries_the_interaction_into_its_turn_and_names_the_turn_back(
    trace_path,
):
    ctx = ToolContext(session=GameSession())
    app, _ = scripted_app(
        ctx,
        brain=ScriptedBrain(AgentResponse(text="Even.")),
        tracer=JsonlTracer(trace_path),
    )
    client = TestClient(app)

    body = client.post(
        "/api/command", json={"text": "how am I doing?", "interaction_id": ID}
    ).json()

    (turn,) = records(trace_path)
    assert turn["interaction_id"] == ID
    assert body["correlation_id"] == turn["correlation_id"]


def test_a_command_without_an_interaction_records_none(trace_path):
    ctx = ToolContext(session=GameSession())
    app, _ = scripted_app(
        ctx,
        brain=ScriptedBrain(AgentResponse(text="Even.")),
        tracer=JsonlTracer(trace_path),
    )
    TestClient(app).post("/api/command", json={"text": "how am I doing?"})
    assert records(trace_path)[0]["interaction_id"] == ""


def test_a_command_with_a_malformed_interaction_is_refused_before_it_runs(trace_path):
    ctx = ToolContext(session=GameSession())
    app, _ = scripted_app(
        ctx,
        brain=ScriptedBrain(AgentResponse(text="Even.")),
        tracer=JsonlTracer(trace_path),
    )
    response = TestClient(app).post(
        "/api/command", json={"text": "e4", "interaction_id": "x" * 65}
    )
    assert response.status_code == 422
    assert not trace_path.exists()


# --- the browser's report ---------------------------------------------------------


def test_the_browsers_milestones_are_written_as_reported(trace_path):
    client = speech_app(trace_path)
    before = client.get("/api/state").json()

    response = client.post("/api/telemetry/voice", json=report())

    assert response.status_code == 204
    (record,) = records(trace_path)
    assert record["kind"] == "voice"
    assert record["schema"] == 2
    assert record["interaction_id"] == ID
    assert record["clock"] == "client_monotonic_ms"
    assert record["marks"]["playback_ended"] == 9800
    assert record["outcome"] == "ended"
    # The server stamps its own receipt and reconciles nothing with the
    # client's clock.
    assert "ts" in record
    assert client.get("/api/state").json() == before


def test_a_censored_report_says_so(trace_path):
    client = speech_app(trace_path)
    client.post(
        "/api/telemetry/voice",
        json=report(outcome="timeout", censored=True, marks={"stt_done": 1200}),
    )
    assert records(trace_path)[0]["censored"] is True


@pytest.mark.parametrize(
    "overrides",
    [
        {"marks": {"not_a_mark": 10}},
        {"marks": {"stt_done": -1}},
        {"marks": {"stt_done": 3_600_001}},
        {"outcome": "exploded"},
        {"clock": "server"},
        {"interaction_id": "../../etc"},
        {"origin": "robot"},
        {"extra": "field"},
    ],
)
def test_a_report_outside_the_contract_is_refused_and_writes_nothing(
    trace_path, overrides
):
    client = speech_app(trace_path)
    response = client.post("/api/telemetry/voice", json=report(**overrides))
    assert response.status_code == 422
    assert not trace_path.exists()


def test_a_report_is_accepted_with_nothing_tracing():
    ctx = ToolContext(session=GameSession())
    client = TestClient(create_app(ctx))
    assert client.post("/api/telemetry/voice", json=report()).status_code == 204


def test_a_report_survives_a_tracer_that_raises():
    class Broken:
        def record(self, _turn):
            raise OSError("disk full")

    ctx = ToolContext(session=GameSession())
    client = TestClient(create_app(ctx, tracer=Broken()))
    assert client.post("/api/telemetry/voice", json=report()).status_code == 204


def test_speech_requests_without_a_speech_service_still_503(trace_path):
    ctx = ToolContext(session=GameSession())
    client = TestClient(create_app(ctx, tracer=JsonlTracer(trace_path)))
    response = client.post(
        "/api/voice/speak", json={"text": "hi"}, headers={"X-Interaction-Id": ID}
    )
    assert response.status_code == 503
    assert not trace_path.exists()
