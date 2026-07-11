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
    STT_PROMPT,
    SpeechClient,
    create_speech_client,
    normalize_transcript,
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


# --- split TTS backend (custom-voice epic) ------------------------------------
#
# The custom voice (Glitch) is served by a dedicated Kokoro-FastAPI container
# while STT stays on Speaches, so the SpeechClient can carry a second
# OpenAI-compatible client for TTS only. No `tts_client` means the one
# backend serves both, exactly as before.


def test_speak_uses_the_dedicated_tts_client_when_given():
    stt_backend = FakeSpeechBackend()
    tts_backend = FakeSpeechBackend(audio=b"kokoro-blend-mp3")
    speech = SpeechClient(client=stt_backend, tts_client=tts_backend)
    assert speech.speak("Check!") == b"kokoro-blend-mp3"
    assert stt_backend.speech.calls == []
    (call,) = tts_backend.speech.calls
    assert call["input"] == "Check!"


def test_transcribe_ignores_the_tts_client():
    stt_backend = FakeSpeechBackend(text="knight f3")
    tts_backend = FakeSpeechBackend()
    speech = SpeechClient(client=stt_backend, tts_client=tts_backend)
    assert speech.transcribe(b"opus-bytes") == "knight f3"
    assert tts_backend.transcriptions.calls == []


def test_create_speech_client_wires_an_injected_tts_client():
    stt_backend = FakeSpeechBackend()
    tts_backend = FakeSpeechBackend(audio=b"blend")
    speech = create_speech_client(
        base_url="http://speaches:8000/v1",
        tts_base_url="http://kokoro:8880/v1",
        client=stt_backend,
        tts_client=tts_backend,
    )
    assert speech.speak("hi") == b"blend"
    assert stt_backend.speech.calls == []


def test_create_speech_client_without_tts_base_url_uses_one_backend():
    backend = FakeSpeechBackend(audio=b"one-backend")
    speech = create_speech_client(base_url="http://speaches:8000/v1", client=backend)
    assert speech.tts_client is None
    assert speech.speak("hi") == b"one-backend"


def test_speech_from_env_builds_a_separate_tts_client(monkeypatch):
    from chessapp.app import _speech_from_env

    monkeypatch.setenv("CHESSAPP_SPEACHES_URL", "http://speaches:8000/v1")
    monkeypatch.setenv("CHESSAPP_TTS_URL", "http://kokoro:8880/v1")
    speech = _speech_from_env()
    assert speech is not None
    assert speech.tts_client is not None
    assert "kokoro:8880" in str(speech.tts_client.base_url)
    assert "speaches:8000" in str(speech.client.base_url)


def test_speech_from_env_without_tts_url_stays_single_backend(monkeypatch):
    from chessapp.app import _speech_from_env

    monkeypatch.setenv("CHESSAPP_SPEACHES_URL", "http://speaches:8000/v1")
    monkeypatch.delenv("CHESSAPP_TTS_URL", raising=False)
    speech = _speech_from_env()
    assert speech is not None
    assert speech.tts_client is None


# --- STT hardening (agent-reliability epic) -----------------------------------
#
# Voice games die when a spoken move transcribes badly. Two deterministic
# defenses: whisper is biased toward chess vocabulary via its `prompt`
# parameter, and known transcription slips are repaired before the text ever
# reaches the command pipeline.


def test_transcribe_biases_whisper_with_the_chess_vocabulary_prompt():
    backend = FakeSpeechBackend()
    speech = SpeechClient(client=backend)
    speech.transcribe(b"opus-bytes")
    (call,) = backend.transcriptions.calls
    assert call["prompt"] == STT_PROMPT


def test_stt_prompt_covers_the_chess_vocabulary():
    # The prompt biases recognition toward these over their homophones, and
    # shows squares glued (e4, not "e 4") — whisper mimics its formatting.
    for term in ("knight", "kingside", "queenside", "e4", "pawn", "en passant"):
        assert term in STT_PROMPT


@pytest.mark.parametrize(
    ("raw", "repaired"),
    [
        ("e 4", "e4"),
        ("E 4", "e4"),
        ("pawn to e 4", "pawn to e4"),
        ("e four", "e4"),
        ("B six", "b6"),
        ("knight to f 3", "knight to f3"),
        ("night to f3", "knight to f3"),
        ("Night takes e5", "knight takes e5"),
        ("castle king side", "castle kingside"),
        ("castle queen side", "castle queenside"),
        ("d ex e5", "dxe5"),  # how STT hears "dxe5"
        ("d x e5", "dxe5"),
        ("D ex E 5", "dxe5"),
    ],
)
def test_normalizer_repairs_known_transcription_slips(raw, repaired):
    assert normalize_transcript(raw) == repaired


@pytest.mark.parametrize(
    "text",
    [
        "knight to f3",
        "castle kingside",
        "what are my legal moves?",
        "resign",
        "a knight for a bishop",
        "exd5",
        "",
    ],
)
def test_normalizer_leaves_clean_text_alone(text):
    assert normalize_transcript(text) == text


def test_stt_prompt_shows_the_file_capture_form():
    # "d takes e5" is how players pronounce dxe5; showing it biases whisper
    # away from mangled forms like "d ex e5".
    assert "d takes e5" in STT_PROMPT


def test_transcribe_returns_the_normalized_transcript():
    backend = FakeSpeechBackend(text="night to f 3")
    speech = SpeechClient(client=backend)
    assert speech.transcribe(b"opus-bytes") == "knight to f3"


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


def test_transcribe_endpoint_returns_repaired_text(ctx):
    # The text the browser feeds back into /api/command is the normalized
    # transcript, so voice slips are fixed before the pipeline ever sees them.
    backend = FakeSpeechBackend(text="night to e 4")
    client = _client(ctx, speech=SpeechClient(client=backend))
    response = client.post(
        "/api/voice/transcribe",
        files={"audio": ("clip.webm", b"opus-bytes", "audio/webm")},
    )
    assert response.json() == {"text": "knight to e4"}


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
