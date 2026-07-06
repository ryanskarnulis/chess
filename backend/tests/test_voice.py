"""Voice layer: STT proxying through an OpenAI-compatible speech server.

The browser never talks to Speaches directly — it posts audio to the app,
which forwards to the speech server and hands back plain text destined for
the same command pipeline as typed input. The speech client is injected, so
these tests run without a live Speaches (same pattern as the brain tests).
"""

import pytest
from fastapi.testclient import TestClient

from chessapp.api import create_app
from chessapp.brain import AgentResponse, ToolCall
from chessapp.game import GameSession
from chessapp.tools import ToolContext
from chessapp.voice import (
    DEFAULT_STT_MODEL,
    DEFAULT_TTS_MODEL,
    DEFAULT_TTS_VOICE,
    SpeechClient,
    create_speech_client,
)
from fakes import ScriptedBrain


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


class FakeSpeech:
    def __init__(self, audio=b"mp3-bytes", error=None):
        self.audio = audio
        self.error = error
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error

        class Result:
            content = self.audio

        return Result()


class FakeSpeechBackend:
    """Quacks like `OpenAI(...)` for the audio surface the voice layer uses."""

    def __init__(self, text="pawn to e4", audio=b"mp3-bytes", error=None):
        self.transcriptions = FakeTranscriptions(text=text, error=error)
        self.speech = FakeSpeech(audio=audio, error=error)

        class Audio:
            pass

        self.audio = Audio()
        self.audio.transcriptions = self.transcriptions
        self.audio.speech = self.speech


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
    assert speech.tts_model == DEFAULT_TTS_MODEL
    assert speech.tts_voice == DEFAULT_TTS_VOICE


def test_speak_forwards_text_and_returns_audio_bytes():
    backend = FakeSpeechBackend(audio=b"kokoro-mp3")
    speech = SpeechClient(client=backend, tts_model="tts-test", tts_voice="af_test")
    audio = speech.speak("Check!")
    assert audio == b"kokoro-mp3"
    (call,) = backend.speech.calls
    assert call["model"] == "tts-test"
    assert call["voice"] == "af_test"
    assert call["input"] == "Check!"
    assert call["response_format"] == "mp3"


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


def test_speak_endpoint_returns_audio(ctx):
    backend = FakeSpeechBackend(audio=b"kokoro-mp3")
    client = _client(ctx, speech=SpeechClient(client=backend))
    response = client.post("/api/voice/speak", json={"text": "Check!"})
    assert response.status_code == 200
    assert response.content == b"kokoro-mp3"
    assert response.headers["content-type"] == "audio/mpeg"


def test_speak_without_speech_service_is_503(ctx):
    response = _client(ctx, speech=None).post(
        "/api/voice/speak", json={"text": "Check!"}
    )
    assert response.status_code == 503


def test_speak_upstream_failure_is_502(ctx):
    backend = FakeSpeechBackend(error=RuntimeError("speaches is down"))
    client = _client(ctx, speech=SpeechClient(client=backend))
    response = client.post("/api/voice/speak", json={"text": "Check!"})
    assert response.status_code == 502


def test_speak_rejects_blank_text(ctx):
    client = _client(ctx, speech=SpeechClient(client=FakeSpeechBackend()))
    response = client.post("/api/voice/speak", json={"text": "   "})
    assert response.status_code == 422


# --- command response carries the speak flag ---------------------------------


def _command_client(ctx, brain):
    return TestClient(create_app(ctx, brain=brain))


def test_command_response_says_speak_when_voice_output_on(ctx):
    ctx.settings.voice_output = True
    brain = ScriptedBrain(
        AgentResponse(
            text="", tool_calls=(ToolCall(name="make_move", args={"move": "e4"}),)
        )
    )
    body = (
        _command_client(ctx, brain)
        .post("/api/command", json={"text": "play e4"})
        .json()
    )
    assert body["speak"] is True


def test_command_response_says_no_speak_when_voice_output_off(ctx):
    brain = ScriptedBrain(AgentResponse(text="hello", tool_calls=()))
    body = _command_client(ctx, brain).post("/api/command", json={"text": "hi"}).json()
    assert body["speak"] is False


def test_transcribe_never_touches_game_state(ctx):
    client = _client(ctx, speech=SpeechClient(client=FakeSpeechBackend()))
    before = ctx.session.fen()
    client.post(
        "/api/voice/transcribe",
        files={"audio": ("clip.webm", b"opus-bytes", "audio/webm")},
    )
    assert ctx.session.fen() == before
