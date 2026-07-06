"""Voice layer: STT + TTS via an OpenAI-compatible speech server (Speaches).

Speaches speaks the OpenAI audio API, so — like the brain — the client is
just the OpenAI SDK pointed at its base URL. This module is the only place
that knows which speech backend/models are in use; the rest of the app sees
`SpeechClient`. The client is injected in tests, so no live Speaches is ever
required (voice is tested at this boundary, never against real audio models).
"""

from dataclasses import dataclass
from typing import Any

# Speaches serves faster-whisper models by HF repo id; `small` is a good
# home-box default (CPU-friendly, solid English accuracy for short commands).
DEFAULT_STT_MODEL = "Systran/faster-whisper-small"
# Kokoro is Speaches' recommended TTS: ~82M params, natural voices, CPU-fine.
DEFAULT_TTS_MODEL = "speaches-ai/Kokoro-82M-v1.0-ONNX"
DEFAULT_TTS_VOICE = "af_heart"


@dataclass
class SpeechClient:
    """STT + TTS bound to one OpenAI-compatible backend and model choices."""

    client: Any
    stt_model: str = DEFAULT_STT_MODEL
    tts_model: str = DEFAULT_TTS_MODEL
    tts_voice: str = DEFAULT_TTS_VOICE

    def transcribe(self, audio: bytes, filename: str = "audio.webm") -> str:
        """Audio bytes (any container whisper accepts; the browser sends
        webm/opus) → plain text for the command pipeline. The filename's
        extension is how the backend sniffs the container format."""
        result = self.client.audio.transcriptions.create(
            model=self.stt_model,
            file=(filename, audio),
        )
        return result.text

    def speak(self, text: str) -> bytes:
        """Text → spoken audio bytes. mp3, because every browser <audio>
        plays it and it's small enough for LAN round-trips."""
        result = self.client.audio.speech.create(
            model=self.tts_model,
            voice=self.tts_voice,
            input=text,
            response_format="mp3",
        )
        return result.content


def create_speech_client(
    *,
    base_url: str,
    stt_model: str = DEFAULT_STT_MODEL,
    tts_model: str = DEFAULT_TTS_MODEL,
    tts_voice: str = DEFAULT_TTS_VOICE,
    api_key: str = "speaches-needs-no-key",
    client: Any | None = None,
) -> SpeechClient:
    """Build a SpeechClient against a real Speaches (e.g. speaches:8000/v1).

    `client` is injected in tests / alternate backends; otherwise the factory
    builds a real OpenAI client against `base_url`.
    """
    if client is None:
        from openai import OpenAI

        client = OpenAI(base_url=base_url, api_key=api_key)
    return SpeechClient(
        client=client, stt_model=stt_model, tts_model=tts_model, tts_voice=tts_voice
    )
