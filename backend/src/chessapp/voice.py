"""Chess's voice vocabulary over the fleet speech client.

The client itself — the OpenAI-audio wire calls, typed errors, factory — is
the vendored `speech_client` (canonical in `../speech/client/python/`, per the
fleet voice contract `../agent-standard/voice.md`). This module holds what is
chess's own: the whisper prompt, the transcript normalizer and the
SAN-to-spoken-chess renderer, bundled as `VOCABULARY` for the client.
"""

import re

from chessapp.speech_client import Vocabulary

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
# vocabulary, like STT_PROMPT — not part of the fleet speech client.
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


VOCABULARY = Vocabulary(
    stt_prompt=STT_PROMPT, normalize=normalize_transcript, speakable=speakable
)
