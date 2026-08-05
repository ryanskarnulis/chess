"""Voice layer: STT proxying through an OpenAI-compatible speech server.

The browser never talks to Speaches directly — it posts audio to the app,
which forwards to the speech server and hands back plain text destined for
the same command pipeline as typed input. The speech layer speaks the OpenAI
audio wire format over plain httpx (per ../agent-standard/voice.md); tests
inject an `httpx.MockTransport`, so no live speech server is ever required
(same pattern as the provider tests).
"""

import asyncio
import json
import threading

import httpx
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
    SpeechRequestError,
    SpeechResponseError,
    create_speech_client,
    normalize_transcript,
    speakable,
)
from fakes import ScriptedBrain, scripted_app


class FakeSpeechServer:
    """An OpenAI-audio-shaped server behind `httpx.MockTransport`.

    Records every request so tests can assert on the wire: multipart fields
    for /audio/transcriptions, the JSON body for /audio/speech.
    """

    def __init__(self, text="pawn to e4", audio=b"mp3-bytes", status=200, body=None):
        self.text = text
        self.audio = audio
        self.status = status
        self.body = body  # overrides the transcription JSON body when set
        self.requests: list[httpx.Request] = []

    def _handler(self, request: httpx.Request) -> httpx.Response:
        request.read()
        self.requests.append(request)
        if self.status != 200:
            return httpx.Response(self.status, text="upstream sad")
        if request.url.path.endswith("/audio/transcriptions"):
            body = self.body if self.body is not None else {"text": self.text}
            return httpx.Response(200, json=body)
        if request.url.path.endswith("/audio/speech"):
            return httpx.Response(
                200, content=self.audio, headers={"content-type": "audio/mpeg"}
            )
        return httpx.Response(404)

    def client(self, base_url="http://speaches:8000/v1") -> httpx.Client:
        return httpx.Client(
            base_url=base_url, transport=httpx.MockTransport(self._handler)
        )


def _multipart_body(request: httpx.Request) -> bytes:
    return request.read()


# --- SpeechClient unit -------------------------------------------------------


def test_transcribe_forwards_audio_and_returns_text():
    server = FakeSpeechServer(text="knight f3")
    speech = SpeechClient(client=server.client(), stt_model="whisper-test")
    text = speech.transcribe(b"opus-bytes", filename="clip.webm")
    assert text == "knight f3"
    (request,) = server.requests
    assert request.url.path == "/v1/audio/transcriptions"
    body = _multipart_body(request)
    assert b'filename="clip.webm"' in body
    assert b"opus-bytes" in body
    assert b"whisper-test" in body


def test_create_speech_client_uses_injected_client_and_default_model():
    server = FakeSpeechServer()
    client = server.client()
    speech = create_speech_client(base_url="http://speaches:8000/v1", client=client)
    assert speech.client is client
    assert speech.stt_model == DEFAULT_STT_MODEL
    assert speech.tts_model == DEFAULT_TTS_MODEL
    assert speech.tts_voice == DEFAULT_TTS_VOICE


def test_speak_forwards_text_and_returns_audio_bytes():
    server = FakeSpeechServer(audio=b"kokoro-mp3")
    speech = SpeechClient(
        client=server.client(), tts_model="tts-test", tts_voice="af_test"
    )
    audio = speech.speak("Check!")
    assert audio == b"kokoro-mp3"
    (request,) = server.requests
    assert request.url.path == "/v1/audio/speech"
    payload = json.loads(request.read())
    assert payload == {
        "model": "tts-test",
        "voice": "af_test",
        "input": "Check!",
        "response_format": "mp3",
    }


# --- typed errors (standard SpeechClient contract) -----------------------------


def test_transcribe_upstream_http_error_raises_request_error():
    server = FakeSpeechServer(status=500)
    speech = SpeechClient(client=server.client())
    with pytest.raises(SpeechRequestError):
        speech.transcribe(b"opus-bytes")


def test_transcribe_unreachable_server_raises_request_error():
    def refuse(request):
        raise httpx.ConnectError("connection refused")

    client = httpx.Client(
        base_url="http://speaches:8000/v1", transport=httpx.MockTransport(refuse)
    )
    with pytest.raises(SpeechRequestError):
        SpeechClient(client=client).transcribe(b"opus-bytes")


def test_transcribe_malformed_body_raises_response_error():
    server = FakeSpeechServer(body={"transcript": "wrong key"})
    speech = SpeechClient(client=server.client())
    with pytest.raises(SpeechResponseError):
        speech.transcribe(b"opus-bytes")


def test_speak_upstream_http_error_raises_request_error():
    server = FakeSpeechServer(status=503)
    speech = SpeechClient(client=server.client())
    with pytest.raises(SpeechRequestError):
        speech.speak("Check!")


# --- split TTS backend (custom-voice epic) ------------------------------------
#
# The custom voice (Glitch) is served by a dedicated Kokoro-FastAPI container
# while STT stays on Speaches, so the SpeechClient can carry a second
# OpenAI-compatible client for TTS only. No `tts_client` means the one
# backend serves both, exactly as before.


def test_speak_uses_the_dedicated_tts_client_when_given():
    stt_server = FakeSpeechServer()
    tts_server = FakeSpeechServer(audio=b"kokoro-blend-mp3")
    speech = SpeechClient(client=stt_server.client(), tts_client=tts_server.client())
    assert speech.speak("Check!") == b"kokoro-blend-mp3"
    assert stt_server.requests == []
    (request,) = tts_server.requests
    assert json.loads(request.read())["input"] == "Check!"


def test_transcribe_ignores_the_tts_client():
    stt_server = FakeSpeechServer(text="knight f3")
    tts_server = FakeSpeechServer()
    speech = SpeechClient(client=stt_server.client(), tts_client=tts_server.client())
    assert speech.transcribe(b"opus-bytes") == "knight f3"
    assert tts_server.requests == []


def test_create_speech_client_wires_an_injected_tts_client():
    stt_server = FakeSpeechServer()
    tts_server = FakeSpeechServer(audio=b"blend")
    speech = create_speech_client(
        base_url="http://speaches:8000/v1",
        tts_base_url="http://kokoro:8880/v1",
        client=stt_server.client(),
        tts_client=tts_server.client("http://kokoro:8880/v1"),
    )
    assert speech.speak("hi") == b"blend"
    assert stt_server.requests == []


def test_create_speech_client_without_tts_base_url_uses_one_backend():
    server = FakeSpeechServer(audio=b"one-backend")
    speech = create_speech_client(
        base_url="http://speaches:8000/v1", client=server.client()
    )
    assert speech.tts_client is None
    assert speech.speak("hi") == b"one-backend"


def test_speech_from_env_builds_a_separate_tts_client(monkeypatch):
    from chessapp.app import _speech_from_env

    monkeypatch.setenv("SPEECH_BASE_URL", "http://speaches:8000/v1")
    monkeypatch.setenv("TTS_BASE_URL", "http://kokoro:8880/v1")
    speech = _speech_from_env()
    assert speech is not None
    assert speech.tts_client is not None
    assert "kokoro:8880" in str(speech.tts_client.base_url)
    assert "speaches:8000" in str(speech.client.base_url)


def test_speech_from_env_without_tts_url_stays_single_backend(monkeypatch):
    from chessapp.app import _speech_from_env

    monkeypatch.setenv("SPEECH_BASE_URL", "http://speaches:8000/v1")
    monkeypatch.delenv("TTS_BASE_URL", raising=False)
    speech = _speech_from_env()
    assert speech is not None
    assert speech.tts_client is None


def test_speech_from_env_without_base_url_is_none(monkeypatch):
    from chessapp.app import _speech_from_env

    monkeypatch.delenv("SPEECH_BASE_URL", raising=False)
    assert _speech_from_env() is None


# --- STT hardening (agent-reliability epic) -----------------------------------
#
# Voice games die when a spoken move transcribes badly. Two deterministic
# defenses: whisper is biased toward chess vocabulary via its `prompt`
# parameter, and known transcription slips are repaired before the text ever
# reaches the command pipeline.


def test_transcribe_biases_whisper_with_the_chess_vocabulary_prompt():
    server = FakeSpeechServer()
    speech = SpeechClient(client=server.client())
    speech.transcribe(b"opus-bytes")
    (request,) = server.requests
    assert STT_PROMPT.encode() in _multipart_body(request)


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
    server = FakeSpeechServer(text="night to f 3")
    speech = SpeechClient(client=server.client())
    assert speech.transcribe(b"opus-bytes") == "knight to f3"


# --- speakable commentary (voice-out hardening, #197) --------------------------
#
# Kokoro's G2P treats SAN tokens as unknown words and sounds them out
# ("exd4" → "EXTVUR", "Nxe5" → "In Zephyv", "O-O" → "Oh"), so the TTS
# direction gets the mirror of `normalize_transcript`: a deterministic
# `speakable()` applied in `SpeechClient.speak()`. The bubble keeps SAN —
# only the audio copy is re-rendered. The renderings were tuned by
# round-tripping the live TTS through the STT: squares must be spaced
# ("e 5" survives, glued "e5" garbles), and the a-file must be "A," —
# bare "a" reduces to the English article, a hyphen reads as "minus".


@pytest.mark.parametrize(
    ("san", "spoken"),
    [
        # captures — the reported bug
        ("exd4.", "e takes d 4."),
        ("Nxe5.", "knight takes e 5."),
        ("Bxf7", "bishop takes f 7"),
        # plain piece moves and pawn pushes (the reply announcement's shape)
        ("Nf3", "knight to f 3"),
        ("e5.", "e 5."),
        ("Kg2!", "king to g 2!"),
        # check and mate suffixes
        ("Qh5+", "queen to h 5, check"),
        ("Qxf7#", "queen takes f 7, checkmate"),
        # castling
        ("O-O.", "castles kingside."),
        ("O-O-O", "castles queenside"),
        ("O-O+", "castles kingside, check"),
        # disambiguators: the letter (or rank) becomes a label with a pause
        ("Rad1", "rook A, to d 1"),
        ("Nbd2", "knight b, to d 2"),
        ("R1a3", "rook 1, to A, 3"),
        # promotion
        ("e8=Q", "e 8, promoting to a queen"),
        ("exd8=N+", "e takes d 8, promoting to a knight, check"),
        # the a-file article problem
        ("a4", "A, 4"),
        ("axb5", "A, takes b 5"),
        ("Rxa1", "rook takes A, 1"),
    ],
)
def test_speakable_renders_san_as_spoken_chess(san, spoken):
    assert speakable(san) == spoken


@pytest.mark.parametrize(
    ("result", "spoken"),
    [
        ("Game over: 1-0 (checkmate).", "Game over: white wins (checkmate)."),
        ("Game over: 0-1 (resignation).", "Game over: black wins (resignation)."),
        ("Game over: 1/2-1/2 (stalemate).", "Game over: a draw (stalemate)."),
    ],
)
def test_speakable_renders_results_as_words(result, spoken):
    assert speakable(result) == spoken


def test_speakable_transforms_san_inside_prose():
    # The reply announcement the pipeline appends to Glitch's reaction…
    assert speakable("Bold. Very bold.\n\nexd4.") == "Bold. Very bold.\n\ne takes d 4."
    # …and squares Glitch names mid-sentence.
    assert speakable("your knight on f6 is toast") == "your knight on f 6 is toast"


@pytest.mark.parametrize(
    "text",
    [
        "Take the exit and check the door",  # chess words in plain English
        "Be careful out there",  # sentence-initial piece letter
        "an extra exam for excellence",  # x-words near the pawn-capture shape
        "I lost the thread on that one — say it again?",
        "Check!",
        "",
    ],
)
def test_speakable_leaves_prose_alone(text):
    assert speakable(text) == text


def test_speakable_is_idempotent():
    mixed = "Rad1! Then O-O-O, then axb8=Q#. Game over: 1-0 (checkmate)."
    once = speakable(mixed)
    assert speakable(once) == once


def test_speak_sends_the_speakable_rendering():
    # The wire body carries the spoken form; the caller's string — what the
    # bubble already showed — is never what the TTS reads.
    server = FakeSpeechServer()
    speech = SpeechClient(client=server.client())
    speech.speak("Nice.\n\nNxe5.")
    (request,) = server.requests
    assert json.loads(request.read())["input"] == "Nice.\n\nknight takes e 5."


# --- API endpoint ------------------------------------------------------------


@pytest.fixture
def ctx():
    return ToolContext(session=GameSession())


def _client(ctx, speech=None):
    return TestClient(create_app(ctx, speech=speech))


def test_transcribe_endpoint_returns_text(ctx):
    server = FakeSpeechServer(text="castle kingside")
    client = _client(ctx, speech=SpeechClient(client=server.client()))
    response = client.post(
        "/api/voice/transcribe",
        files={"audio": ("clip.webm", b"opus-bytes", "audio/webm")},
    )
    assert response.status_code == 200
    assert response.json() == {"text": "castle kingside"}
    (request,) = server.requests
    body = _multipart_body(request)
    assert b'filename="clip.webm"' in body
    assert b"opus-bytes" in body


def test_transcribe_endpoint_returns_repaired_text(ctx):
    # The text the browser feeds back into /api/command is the normalized
    # transcript, so voice slips are fixed before the pipeline ever sees them.
    server = FakeSpeechServer(text="night to e 4")
    client = _client(ctx, speech=SpeechClient(client=server.client()))
    response = client.post(
        "/api/voice/transcribe",
        files={"audio": ("clip.webm", b"opus-bytes", "audio/webm")},
    )
    assert response.json() == {"text": "knight to e4"}


async def test_transcribe_endpoint_keeps_event_loop_free_while_speech_blocks(ctx):
    """A slow synchronous STT request must not stall unrelated async work."""
    entered = threading.Event()
    heartbeat = threading.Event()
    release = threading.Event()
    heartbeat_seen: list[bool] = []

    class Audio:
        filename = "clip.webm"

        async def read(self):
            return b"opus-bytes"

    class BlockingSpeech:
        def transcribe(self, data, filename):
            assert data == b"opus-bytes"
            assert filename == "clip.webm"
            entered.set()
            assert release.wait(2), "test never released transcription"
            return "castle kingside"

    def release_after_heartbeat() -> None:
        assert entered.wait(2), "transcription never started"
        heartbeat_seen.append(heartbeat.wait(1))
        release.set()

    speech = BlockingSpeech()
    app = create_app(ctx, speech=speech)
    endpoint = next(
        route.endpoint
        for route in app.routes
        if getattr(route, "path", None) == "/api/voice/transcribe"
    )
    witness = threading.Thread(target=release_after_heartbeat)
    witness.start()

    request = asyncio.create_task(endpoint(Audio()))
    await asyncio.sleep(0)
    heartbeat.set()

    assert await request == {"text": "castle kingside"}
    witness.join(2)
    assert heartbeat_seen == [True], "transcription blocked the event loop"


def test_transcribe_without_speech_service_is_503(ctx):
    client = _client(ctx, speech=None)
    response = client.post(
        "/api/voice/transcribe",
        files={"audio": ("clip.webm", b"opus-bytes", "audio/webm")},
    )
    assert response.status_code == 503


def test_transcribe_upstream_failure_is_502(ctx):
    server = FakeSpeechServer(status=500)
    client = _client(ctx, speech=SpeechClient(client=server.client()))
    response = client.post(
        "/api/voice/transcribe",
        files={"audio": ("clip.webm", b"opus-bytes", "audio/webm")},
    )
    assert response.status_code == 502


def test_speak_endpoint_returns_audio(ctx):
    server = FakeSpeechServer(audio=b"kokoro-mp3")
    client = _client(ctx, speech=SpeechClient(client=server.client()))
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
    server = FakeSpeechServer(status=503)
    client = _client(ctx, speech=SpeechClient(client=server.client()))
    response = client.post("/api/voice/speak", json={"text": "Check!"})
    assert response.status_code == 502


def test_speak_rejects_blank_text(ctx):
    client = _client(ctx, speech=SpeechClient(client=FakeSpeechServer().client()))
    response = client.post("/api/voice/speak", json={"text": "   "})
    assert response.status_code == 422


# --- command response carries the speak flag ---------------------------------


def _command_client(ctx, brain):
    app, _ = scripted_app(ctx, brain=brain)
    return TestClient(app)


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
    client = _client(ctx, speech=SpeechClient(client=FakeSpeechServer().client()))
    before = ctx.session.fen()
    client.post(
        "/api/voice/transcribe",
        files={"audio": ("clip.webm", b"opus-bytes", "audio/webm")},
    )
    assert ctx.session.fen() == before
