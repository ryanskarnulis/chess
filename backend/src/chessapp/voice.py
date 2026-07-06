"""Voice layer: STT via an OpenAI-compatible speech server (Speaches).

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


@dataclass
class SpeechClient:
    """STT bound to one OpenAI-compatible backend and model choice."""

    client: Any
    stt_model: str = DEFAULT_STT_MODEL

    def transcribe(self, audio: bytes, filename: str = "audio.webm") -> str:
        """Audio bytes (any container whisper accepts; the browser sends
        webm/opus) → plain text for the command pipeline. The filename's
        extension is how the backend sniffs the container format."""
        result = self.client.audio.transcriptions.create(
            model=self.stt_model,
            file=(filename, audio),
        )
        return result.text


def create_speech_client(
    *,
    base_url: str,
    stt_model: str = DEFAULT_STT_MODEL,
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
    return SpeechClient(client=client, stt_model=stt_model)
