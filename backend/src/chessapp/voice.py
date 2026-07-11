"""Voice layer: STT + TTS via an OpenAI-compatible speech server (Speaches).

Speaches speaks the OpenAI audio API, so — like the brain — the client is
just the OpenAI SDK pointed at its base URL. This module is the only place
that knows which speech backend/models are in use; the rest of the app sees
`SpeechClient`. The client is injected in tests, so no live Speaches is ever
required (voice is tested at this boundary, never against real audio models).
"""

import re
from dataclasses import dataclass
from typing import Any

# Speaches serves faster-whisper models by HF repo id; `small` is a good
# home-box default (CPU-friendly, solid English accuracy for short commands).
DEFAULT_STT_MODEL = "Systran/faster-whisper-small"
# Kokoro is Speaches' recommended TTS: ~82M params, natural voices, CPU-fine.
DEFAULT_TTS_MODEL = "speaches-ai/Kokoro-82M-v1.0-ONNX"
DEFAULT_TTS_VOICE = "af_heart"

# Whisper conditions on this prompt as if it preceded the audio, biasing
# recognition toward its vocabulary ("knight" over "night") and mimicking its
# formatting — squares appear glued (e4) so they transcribe glued.
STT_PROMPT = (
    "Chess commands: pawn to e4, knight to f3, bishop takes c6, d takes e5, "
    "rook to d1, queen to h5, king to g2, castle kingside, castle queenside, "
    "en passant, pawn promotes to a queen on e8, check, checkmate, capture, "
    "resign, new game, undo my move, give me a hint, what are my legal moves."
)

# Transcript repairs, deliberately conservative: only slips with exactly one
# chess reading. Fuzzier mishears ("sea five") are the agent's job — its
# prompt teaches transcript repair against the legal-move list.
_SPOKEN_RANKS = "one|two|three|four|five|six|seven|eight"
_SPLIT_SQUARE = re.compile(rf"\b([a-h]) ([1-8]|{_SPOKEN_RANKS})\b", re.IGNORECASE)
_RANK_DIGITS = {w: str(i) for i, w in enumerate(_SPOKEN_RANKS.split("|"), start=1)}
_NIGHT = re.compile(r"\bnight\b", re.IGNORECASE)
_CASTLE_SIDE = re.compile(r"\b(king|queen) side\b", re.IGNORECASE)
_FILE_X_SQUARE = re.compile(r"\b([a-h]) (?:x|ex) ([a-h][1-8])\b", re.IGNORECASE)


def normalize_transcript(text: str) -> str:
    """Repair unambiguous STT slips before the text enters the command
    pipeline: split squares ("e 4", "b six" → "e4", "b6"), split file
    captures ("d ex e5" → "dxe5"), the knight/night homophone, and
    "king side"/"queen side". Deterministic and idempotent; anything it
    can't fix passes through unchanged."""
    text = _SPLIT_SQUARE.sub(
        lambda m: m.group(1).lower() + _RANK_DIGITS.get(m.group(2).lower(), m.group(2)),
        text,
    )
    text = _FILE_X_SQUARE.sub(
        lambda m: m.group(1).lower() + "x" + m.group(2).lower(), text
    )
    text = _NIGHT.sub("knight", text)
    return _CASTLE_SIDE.sub(lambda m: m.group(1).lower() + "side", text)


@dataclass
class SpeechClient:
    """STT + TTS bound to one OpenAI-compatible backend and model choices."""

    client: Any
    stt_model: str = DEFAULT_STT_MODEL
    tts_model: str = DEFAULT_TTS_MODEL
    tts_voice: str = DEFAULT_TTS_VOICE
    stt_prompt: str = STT_PROMPT

    def transcribe(self, audio: bytes, filename: str = "audio.webm") -> str:
        """Audio bytes (any container whisper accepts; the browser sends
        webm/opus) → plain text for the command pipeline, vocabulary-biased
        via the STT prompt and repaired by the normalizer. The filename's
        extension is how the backend sniffs the container format."""
        result = self.client.audio.transcriptions.create(
            model=self.stt_model,
            file=(filename, audio),
            prompt=self.stt_prompt,
        )
        return normalize_transcript(result.text)

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
