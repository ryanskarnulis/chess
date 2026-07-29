"""Voice layer: STT + TTS via OpenAI-compatible speech servers over plain httpx.

Speaches (STT) and Kokoro-FastAPI (TTS) both speak the OpenAI audio API, so —
like the brain's provider — the client is two POSTs, no SDK:
`/audio/transcriptions` (multipart) and `/audio/speech` (JSON → mp3). The
response body is validated by a Pydantic wire model at the boundary; failures
raise typed errors rather than being best-effort parsed. This module is the
only place that knows which speech backend/models are in use; the rest of the
app sees `SpeechClient`. The httpx clients are injected in tests (via
`httpx.MockTransport`), so no live speech server is ever required — voice is
tested at this boundary, never against real audio models.

This module is chess's implementation of the fleet voice contract
(`../agent-standard/voice.md`) and its `SpeechClient` shape is the reference.
"""

import re
from dataclasses import dataclass

import httpx
from pydantic import BaseModel, ValidationError

# Speaches serves faster-whisper models by HF repo id; `small` is a good
# home-box default (CPU-friendly, solid English accuracy for short commands).
DEFAULT_STT_MODEL = "Systran/faster-whisper-small"
# Kokoro is Speaches' recommended TTS: ~82M params, natural voices, CPU-fine.
DEFAULT_TTS_MODEL = "speaches-ai/Kokoro-82M-v1.0-ONNX"
DEFAULT_TTS_VOICE = "af_heart"

# STT of a short clip is seconds even on CPU; TTS similar. A cold model load
# on the speech server is the only slow path, so the read timeout is generous
# while connect stays short (a dead server fails fast) — provider.py's shape.
_TIMEOUT = httpx.Timeout(60.0, connect=10.0)

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


# The TTS mirror of the repairs above (#197): Kokoro's G2P treats SAN tokens
# as unknown words and sounds them out ("exd4" → "EXTVUR", "Nxe5" →
# "In Zephyv", "O-O" → "Oh"), so commentary is re-rendered as spoken chess
# before synthesis. The renderings were tuned by round-tripping the live TTS
# through the STT: squares must be spaced ("e 5" survives, glued "e5"
# garbles), and the a-file must be "A," — bare "a" reduces to the English
# article, a hyphen reads as "minus", "ay" reads as "eye". Chess-domain
# vocabulary, like STT_PROMPT — not part of the fleet SpeechClient shape.
_PIECE_NAMES = {"K": "king", "Q": "queen", "R": "rook", "B": "bishop", "N": "knight"}
_PROMOTION_NAMES = {"Q": "a queen", "R": "a rook", "B": "a bishop", "N": "a knight"}
_SUFFIX_WORDS = {"+": ", check", "#": ", checkmate"}
_RESULT_WORDS = {"1-0": "white wins", "0-1": "black wins", "1/2-1/2": "a draw"}
_RESULT = re.compile(r"\b(1-0|0-1|1/2-1/2)\b")
# One token-bounded pass over the SAN grammar. Case-sensitive on purpose
# (files lowercase, pieces uppercase) so ordinary English never matches; the
# pawn-push alternative doubles as the bare-square rule for squares named in
# prose ("your knight on f6"). The trailing lookahead stands in for \b, which
# cannot follow the non-word "+"/"#".
_SAN = re.compile(
    r"""\b(?:
        (?P<castle>O-O(?:-O)?)
      | (?P<piece>[KQRBN])(?P<disambig>[a-h1-8]?)(?P<capture>x?)(?P<square>[a-h][1-8])
      | (?P<file>[a-h])(?:x(?P<taken>[a-h][1-8])|(?P<rank>[1-8]))
    )
    (?P<promotion>=[QRBN])?
    (?P<suffix>[+#])?
    (?!\w)""",
    re.VERBOSE,
)


def _spoken_file(file: str) -> str:
    return "A," if file == "a" else file


def _spoken_square(square: str) -> str:
    return f"{_spoken_file(square[0])} {square[1]}"


def _spoken_san(m: re.Match[str]) -> str:
    if m.group("castle"):
        side = "queenside" if m.group("castle") == "O-O-O" else "kingside"
        words = f"castles {side}"
    elif m.group("piece"):
        parts = [_PIECE_NAMES[m.group("piece")]]
        if disambig := m.group("disambig"):
            parts.append("A," if disambig == "a" else disambig + ",")
        parts.append("takes" if m.group("capture") else "to")
        parts.append(_spoken_square(m.group("square")))
        words = " ".join(parts)
    elif m.group("taken"):
        words = (
            f"{_spoken_file(m.group('file'))} takes {_spoken_square(m.group('taken'))}"
        )
    else:
        words = _spoken_square(m.group("file") + m.group("rank"))
    if m.group("promotion"):
        words += f", promoting to {_PROMOTION_NAMES[m.group('promotion')[1]]}"
    if m.group("suffix"):
        words += _SUFFIX_WORDS[m.group("suffix")]
    return words


def speakable(text: str) -> str:
    """Re-render chess notation as the spoken chess a TTS can say: SAN tokens
    ("exd4" → "e takes d 4", "O-O" → "castles kingside"), bare squares, and
    game results ("1-0" → "white wins"). Deterministic and idempotent; prose
    passes through untouched. Applied to the audio copy only — the bubble
    keeps real SAN."""
    text = _RESULT.sub(lambda m: _RESULT_WORDS[m.group(1)], text)
    return _SAN.sub(_spoken_san, text)


class SpeechError(Exception):
    """Base for everything a speech round-trip can raise."""


class SpeechRequestError(SpeechError):
    """No usable HTTP response: connect/timeout failure or a non-200 status."""


class SpeechResponseError(SpeechError):
    """The server answered 200 but the body failed validation."""


class _TranscriptionResponse(BaseModel):
    """The one field of the OpenAI transcription response the app uses."""

    text: str


@dataclass
class SpeechClient:
    """STT + TTS bound to OpenAI-compatible backends and model choices.

    One backend serves both by default; `tts_client` splits TTS onto its own
    server (the custom Glitch voice lives in a Kokoro-FastAPI container while
    STT stays on Speaches)."""

    client: httpx.Client
    stt_model: str = DEFAULT_STT_MODEL
    tts_model: str = DEFAULT_TTS_MODEL
    tts_voice: str = DEFAULT_TTS_VOICE
    stt_prompt: str = STT_PROMPT
    tts_client: httpx.Client | None = None

    def transcribe(self, audio: bytes, filename: str = "audio.webm") -> str:
        """Audio bytes (any container whisper accepts; the browser sends
        webm/opus) → plain text for the command pipeline, vocabulary-biased
        via the STT prompt and repaired by the normalizer. The filename's
        extension is how the backend sniffs the container format."""
        try:
            response = self.client.post(
                "audio/transcriptions",
                files={"file": (filename, audio)},
                data={"model": self.stt_model, "prompt": self.stt_prompt},
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise SpeechRequestError(f"transcription request failed: {exc}") from exc
        try:
            parsed = _TranscriptionResponse.model_validate_json(response.content)
        except ValidationError as exc:
            raise SpeechResponseError(
                f"malformed transcription response: {exc}"
            ) from exc
        return normalize_transcript(parsed.text)

    def speak(self, text: str) -> bytes:
        """Text → spoken audio bytes. mp3, because every browser <audio>
        plays it and it's small enough for LAN round-trips. The text is
        re-rendered by `speakable` on the way out — the caller's string is
        display text, and SAN read verbatim comes out garbled (#197)."""
        client = self.tts_client if self.tts_client is not None else self.client
        try:
            response = client.post(
                "audio/speech",
                json={
                    "model": self.tts_model,
                    "voice": self.tts_voice,
                    "input": speakable(text),
                    "response_format": "mp3",
                },
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise SpeechRequestError(f"speech request failed: {exc}") from exc
        return response.content


def create_speech_client(
    *,
    base_url: str,
    tts_base_url: str | None = None,
    stt_model: str = DEFAULT_STT_MODEL,
    tts_model: str = DEFAULT_TTS_MODEL,
    tts_voice: str = DEFAULT_TTS_VOICE,
    client: httpx.Client | None = None,
    tts_client: httpx.Client | None = None,
) -> SpeechClient:
    """Build a SpeechClient against a real Speaches (e.g. host:8400/v1).

    `tts_base_url` points TTS at its own OpenAI-compatible server (e.g. the
    Kokoro-FastAPI container serving the custom voice); without it the one
    backend serves both. `client`/`tts_client` inject fakes in tests /
    alternate backends; otherwise the factory builds real httpx clients.
    """
    if client is None:
        client = httpx.Client(base_url=base_url, timeout=_TIMEOUT)
    if tts_client is None and tts_base_url is not None:
        tts_client = httpx.Client(base_url=tts_base_url, timeout=_TIMEOUT)
    return SpeechClient(
        client=client,
        stt_model=stt_model,
        tts_model=tts_model,
        tts_voice=tts_voice,
        tts_client=tts_client,
    )
