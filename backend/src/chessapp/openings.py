"""The opening's name and ECO code, read off the move line by code (#373).

"What opening is this?" used to have nothing behind it but the 12B's memory
of opening names, which is unreliable: it names the Italian after 1. e4 e5
2. Nf3 Nc6 3. Bb5. The names come from Lichess's `chess-openings` data set
(CC0, vendored as `data/openings.tsv`), matched by position rather than by
move order, so a transposition reaches the same name.

The name is the deepest book position the standing line has reached, looked
for no deeper than the longest line in the table (`Book.max_ply`): past that
nothing can match, which is how the name stops updating once the game has
left the book. A game from a set-up position has no opening.
"""

import functools
import unicodedata
from dataclasses import dataclass
from importlib import resources
from typing import Any

import chess

from chessapp.game import GameSession


@dataclass(frozen=True)
class Book:
    """The table: each named position's key (`_key`) to its `(eco, name)`,
    and the longest line in it, in plies."""

    positions: dict[str, tuple[str, str]]
    max_ply: int


def _key(fen: str) -> str:
    """A position without its move counters: placement, side to move,
    castling rights and en passant — what makes two move orders the same
    opening."""
    return " ".join(fen.split()[:4])


@functools.cache
def book() -> Book:
    """The vendored table, read once. Each line carries its length and its
    position's key already (`scripts/build_openings.py`): parsing the PGN
    here cost 0.86 s on the first turn. A position named twice keeps its
    first name (the data set's own order, by ECO code)."""
    text = resources.files("chessapp").joinpath("data/openings.tsv").read_text()
    positions: dict[str, tuple[str, str]] = {}
    max_ply = 0
    for line in text.splitlines():
        if not line or line.startswith("#") or line.startswith("eco\t"):
            continue
        eco, name, _pgn, plies, key = line.split("\t")
        positions.setdefault(key, (eco, name))
        max_ply = max(max_ply, int(plies))
    return Book(positions, max_ply)


def _matches(session: GameSession) -> list[tuple[str, str]]:
    """Every book position the standing line passed through, in order."""
    fens = session.position_fens()
    if _key(fens[0]) != _key(chess.STARTING_FEN):
        return []
    table = book()
    return [
        table.positions[key]
        for key in map(_key, fens[1 : table.max_ply + 1])
        if key in table.positions
    ]


def opening_of(session: GameSession) -> dict[str, Any] | None:
    """`{"eco", "name"}` of the deepest book position the line has reached,
    or None: before the first move, from a set-up position, or when no
    position of the line is in the book."""
    matches = _matches(session)
    if not matches:
        return None
    eco, name = matches[-1]
    return {"eco": eco, "name": name}


def names_on_line(session: GameSession) -> frozenset[str]:
    """Every opening name the line passed through: "the Ruy Lopez" is true
    of a game now in the Ruy Lopez: Morphy Defense, and so is "it started as
    a King's Pawn Game"."""
    return frozenset(name for _, name in _matches(session))


@functools.cache
def families() -> frozenset[str]:
    """Every opening family the book names: the part of a name before its
    variation ("Ruy Lopez" of "Ruy Lopez: Morphy Defense")."""
    return frozenset(name.split(":")[0] for _, name in book().positions.values())


# The trailing words a family is spoken without: "the Sicilian", "the Ruy",
# never "the Sicilian Defense" in talk.
_GENERIC = frozenset(
    {"defense", "defence", "opening", "game", "gambit", "attack", "system"}
)

# Heads that are ordinary words, or chess words, before they are openings.
# "The English" or "the modern" is table talk, and "your king's knight" or
# "the four knights" is pieces, so a head like these is only ever read in its
# full family name ("English Opening").
_COMMON_HEADS = frozenset(
    {
        "amazon",
        "bird",
        "center",
        "elephant",
        "english",
        "formation",
        "four knights",
        "french",
        "global",
        "indian",
        "kangaroo",
        "king's",
        "king's knight",
        "king's pawn",
        "lemming",
        "lion",
        "modern",
        "polish",
        "queen's",
        "queen's pawn",
        "rat",
        "sodium",
        "three knights",
        "vulture",
    }
)


def normalized(name: str) -> str:
    """An opening name as the speech reading compares it: lower case, no
    accents, hyphens as spaces, one spelling of "defense" and of the
    apostrophe."""
    plain = unicodedata.normalize("NFKD", name.replace("’", "'").replace("-", " "))
    plain = "".join(c for c in plain if not unicodedata.combining(c))
    return " ".join(plain.lower().replace("defence", "defense").split())


@functools.cache
def spoken_names() -> tuple[frozenset[str], frozenset[str]]:
    """How the families are said, normalized, in two sets: each family in
    full ("ruy lopez", "sicilian defense"), and each distinctive family
    without its trailing generic words ("sicilian", "caro-kann"), which the
    reading takes only after an article ("the Sicilian"). A variation name
    ("Najdorf") is in neither: the vocabulary is the families and nothing
    narrower (`docs/speech-accuracy.md`)."""
    full_names: set[str] = set()
    heads: set[str] = set()
    for family in families():
        full = normalized(family.split(",")[0])
        if full in _COMMON_HEADS:
            continue  # "Formation", a family of one ordinary word
        full_names.add(full)
        words = full.split()
        while words and words[-1] in _GENERIC:
            words.pop()
        head = " ".join(words)
        if head and head != full and head not in _COMMON_HEADS:
            heads.add(head)
    return frozenset(full_names), frozenset(heads - full_names)
