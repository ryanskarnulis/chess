"""Voice layer: STT proxying through an OpenAI-compatible speech server.

The browser never talks to Speaches directly — it posts audio to the app,
which forwards to the speech server and hands back plain text destined for
the same command pipeline as typed input. The speech client is injected, so
these tests run without a live Speaches (same pattern as the brain tests).
"""

import pytest
from fastapi.testclient import TestClient

from chessapp.api import create_app
from chessapp.game import GameSession
from chessapp.tools import ToolContext
from chessapp.voice import DEFAULT_STT_MODEL, SpeechClient, create_speech_client


class FakeTranscriptions:
    def __init__(self, text="pawn to e4", error=None):
        self.text = text
        self.error = error
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error

        class Result:
            text = self.text

        return Result()


class FakeSpeechBackend:
    """Quacks like `OpenAI(...)` for the audio surface the voice layer uses."""

    def __init__(self, text="pawn to e4", error=None):
        self.transcriptions = FakeTranscriptions(text=text, error=error)

        class Audio:
            pass

        self.audio = Audio()
        self.audio.transcriptions = self.transcriptions


# --- SpeechClient unit -------------------------------------------------------


def test_transcribe_forwards_audio_and_returns_text():
    backend = FakeSpeechBackend(text="knight f3")
    speech = SpeechClient(client=backend, stt_model="whisper-test")
    text = speech.transcribe(b"opus-bytes", filename="clip.webm")
    assert text == "knight f3"
    (call,) = backend.transcriptions.calls
    assert call["model"] == "whisper-test"
    assert call["file"] == ("clip.webm", b"opus-bytes")


def test_create_speech_client_uses_injected_client_and_default_model():
    backend = FakeSpeechBackend()
    speech = create_speech_client(base_url="http://speaches:8000/v1", client=backend)
    assert speech.client is backend
    assert speech.stt_model == DEFAULT_STT_MODEL


# --- API endpoint ------------------------------------------------------------


@pytest.fixture
def ctx():
    return ToolContext(session=GameSession())


def _client(ctx, speech=None):
    return TestClient(create_app(ctx, speech=speech))


def test_transcribe_endpoint_returns_text(ctx):
    backend = FakeSpeechBackend(text="castle kingside")
    client = _client(ctx, speech=SpeechClient(client=backend))
    response = client.post(
        "/api/voice/transcribe",
        files={"audio": ("clip.webm", b"opus-bytes", "audio/webm")},
    )
    assert response.status_code == 200
    assert response.json() == {"text": "castle kingside"}
    (call,) = backend.transcriptions.calls
    assert call["file"] == ("clip.webm", b"opus-bytes")


def test_transcribe_without_speech_service_is_503(ctx):
    client = _client(ctx, speech=None)
    response = client.post(
        "/api/voice/transcribe",
        files={"audio": ("clip.webm", b"opus-bytes", "audio/webm")},
    )
    assert response.status_code == 503


def test_transcribe_upstream_failure_is_502(ctx):
    backend = FakeSpeechBackend(error=RuntimeError("speaches is down"))
    client = _client(ctx, speech=SpeechClient(client=backend))
    response = client.post(
        "/api/voice/transcribe",
        files={"audio": ("clip.webm", b"opus-bytes", "audio/webm")},
    )
    assert response.status_code == 502


def test_transcribe_never_touches_game_state(ctx):
    client = _client(ctx, speech=SpeechClient(client=FakeSpeechBackend()))
    before = ctx.session.fen()
    client.post(
        "/api/voice/transcribe",
        files={"audio": ("clip.webm", b"opus-bytes", "audio/webm")},
    )
    assert ctx.session.fen() == before
