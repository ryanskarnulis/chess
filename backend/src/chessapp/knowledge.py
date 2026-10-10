"""The second brain (#374): what `lookup` answers from, found by keyword and,
since #450, by meaning too.

"What's the idea behind the Sicilian?" used to have nothing behind it but the
12B's memory, which knows some of it and makes up the rest. Now the planner
calls `lookup` with what the player is after, and the answer is a few short
notes from a local, curated corpus; the narrator phrases them.

**One tool, sources behind it.** The planner is offered `lookup(query)` and
nothing else: no source or topic to pick, because with one source that is a
choice the 12B can only get wrong. Sources are a detail of this module. The
first is `chess_knowledge`, the notes under `data/knowledge/`: openings,
strategy, tactics, endgames, the rules and the game's history, written for
this app in its own words. Each file opens with the references its facts were
checked against, so a wrong claim can be traced and fixed. Player memory
(#377) would be the next source, not the next tool.

**Search is BM25 on the CPU.** Every note is one document: its title and
aliases count three times, its text once. Nothing leaves the machine
(BRIEF: it works offline). On top of the ranking, a name said in full wins:
a query that contains a note's title or one of its aliases, word for word,
earns that note a bonus weighted by how rare the name's words are, so
"Sicilian Najdorf" is the Najdorf note and not the Sicilian one, and an alias
made of a common word ("move") earns next to nothing.

**Hybrid search (#450).** `Index.hybrid` adds EmbeddingGemma vectors from
the shared embeddings service (`embeddings.py`) to the keyword ranking and
fuses the two by reciprocal rank, so "that famous computer match in 97"
finds the Deep Blue note with no word in common. A relevance bar on absolute signals
decides whether anything is found at all; the fused rank only orders what
cleared it. Without the service, `search` (keywords alone) is the answer.

**Nothing found is an answer.** A query that shares no meaningful word with
any note, or only weakly, comes back empty, and Glitch can say he doesn't
know rather than guess.

Pure apart from reading the vendored notes once, and the vector cache file
`NoteVectors` reads and writes when it is given one.
"""

import functools
import hashlib
import json
import math
import os
import re
import unicodedata
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

from chessapp import embeddings

# The corpus files, in the order their notes are numbered.
KNOWLEDGE_DIR = "data/knowledge"

# How many notes one lookup returns at most, and how close to the best a
# runner-up must score to come along. Three short notes is about 300 words,
# which the narrator reads in well under a second of prompt processing.
MAX_PASSAGES = 3
RELATIVE_CUTOFF = 0.5

# Below this score the best match is a coincidence of common words ("what
# time is it?"), and the lookup reports nothing found. A single rare word
# matched in a note's name scores well above it. Calibrated on the retrieval
# table in `tests/test_knowledge.py`.
MIN_SCORE = 6.0

# A title or alias found word for word in the query earns this much per unit
# of the name's summed IDF: the rarer and longer the name said, the more.
NAME_BONUS = 1.5

# BM25's usual constants.
_K1 = 1.2
_B = 0.75

# Title and aliases count this many times over the text.
_NAME_WEIGHT = 3

# --- hybrid search (#450) ---
#
# Cosine alone is no bar: every query is somewhat close to every note, and
# chatter can sit closer to its nearest note than a reworded question does to
# its answer ("pawn to d4" is 0.82 from the Queen's Pawn Game note, "I always
# run short on the clock" 0.67 from time management). What separates them is
# how far a note stands out from the query's own crowd: its *lift*, the
# cosine above the query's LIFT_BASELINE-th best note's. A note clears the bar
# by lifting far on meaning alone (a paraphrase with no word in common), or by
# lifting some and scoring on keywords too (a name said, with meaning that
# agrees). Keywords alone no longer clear it: "play e4" names the King's Pawn
# Game note in BM25 and nothing in meaning.
#
# Calibrated on the tables in `tests/test_knowledge.py` (2026-10-10,
# EmbeddingGemma 2 Q8_0): the highest lift any chatter reaches is 0.056
# ("what should I play now"); LIFT_ALONE sits above it, and LIFT_SUPPORT and
# BM25_SUPPORT are where recall peaked in a sweep (docs/second-brain.md).
LIFT_BASELINE = 10
LIFT_ALONE = 0.065
LIFT_SUPPORT = 0.015
BM25_SUPPORT = 8.0

# Reciprocal rank fusion's usual constant.
RRF_K = 60

# A note in the opening on the board gains this much fused score: about one
# extra first place, enough to order "the advance variation" by the game, but
# only among notes that already cleared the bar.
OPENING_BOOST = 1 / (RRF_K + 1)

# Stored and compared vector size. EmbeddingGemma is Matryoshka-trained, so
# the leading dimensions, re-normalized, are a smaller vector of the same
# meaning.
DIMS = 768


# Words that carry no topic. Question words, fillers and pronouns: a spoken
# query is full of them, and a note matched on "what" is no match.
_STOPWORDS = frozenset(
    """
    a about after again all also am an and any are as at be because been
    before being between both but by can could did do does doing done down
    during each either else ever every for from further get gets getting go
    going had has have having he her here hers him his how i if in into is it
    its itself just know let like me more most much my myself no nor not now
    of off on once only or other our ours out over own please really same say
    she should so some such tell than that the their theirs them then there
    these they this those through thing things to too under until up us very
    want was we were what when where which while who whom whose why will with
    would you your yours yourself explain mean means meant okay ok hey um uh
    """.split()
)

# One spelling for words written two ways. Applied after lower-casing.
_SPELLINGS = {"defence": "defense", "centre": "center"}

_TOKEN = re.compile(r"[a-z0-9]+")


def _plain(text: str) -> str:
    """Lower case, no accents, one apostrophe, possessives dropped."""
    text = unicodedata.normalize("NFKD", text.replace("’", "'"))
    text = "".join(c for c in text if not unicodedata.combining(c)).lower()
    return re.sub(r"'s\b", "", text)


def _stem(word: str) -> str:
    """A light suffix strip, the same on both sides of the search: "pawns"
    and "pawn", "castling" and "castle", "pinned" and "pin" meet."""
    word = _SPELLINGS.get(word, word)
    if len(word) > 5 and word.endswith("ing"):
        word = word[:-3]
    elif len(word) > 4 and word.endswith("ed"):
        word = word[:-2]
    elif len(word) > 4 and word.endswith("ies"):
        word = word[:-3] + "y"
    elif len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        word = word[:-1]
    if len(word) > 3 and word.endswith("e"):
        word = word[:-1]
    # "pinned" → "pinn" → "pin": a doubled consonant left by the strip.
    if len(word) > 3 and word[-1] == word[-2] and word[-1] not in "aeiouls":
        word = word[:-1]
    return word


def tokens(text: str) -> list[str]:
    """The words of `text` that can match: normalized, stemmed, stopwords and
    lone digits dropped ("1. e4" keeps "e4"; "fifty" and "50" both stay)."""
    words = _TOKEN.findall(_plain(text).replace("-", " "))
    return [
        _stem(word)
        for word in words
        if word not in _STOPWORDS and not (word.isdigit() and len(word) == 1)
    ]


# Words a name is often said without: "the Caro-Kann Advance" names "Caro-Kann
# Defense: Advance Variation". Left out of both sides when a name is matched
# word for word, and nowhere else. Stemmed, as everything compared is.
_GENERIC = frozenset(map(_stem, ("defense", "variation")))


def _bare(words: Sequence[str]) -> tuple[str, ...]:
    """`words` without the generic ones a name can be said without."""
    return tuple(word for word in words if word not in _GENERIC)


@dataclass(frozen=True)
class Note:
    """One note: `id` is its file and title ("openings/sicilian-defense"),
    `aliases` the other names it goes by, `text` what the narrator reads."""

    id: str
    title: str
    aliases: tuple[str, ...]
    text: str


@dataclass(frozen=True)
class Hit:
    note: Note
    score: float


def _slug(title: str) -> str:
    return "-".join(_TOKEN.findall(_plain(title)))


def parse(text: str, topic: str) -> list[Note]:
    """The notes in one corpus file.

    A note starts at a `## ` heading, its title. An optional first line
    `Also: a, b, c` lists other names for it; the paragraphs after that are
    its text, joined into one. Everything before the first heading (the
    file's title and its sources) is not a note."""
    notes: list[Note] = []
    for block in re.split(r"^## ", text, flags=re.MULTILINE)[1:]:
        title, _, body = block.partition("\n")
        title = title.strip()
        aliases: tuple[str, ...] = ()
        lines = body.strip().splitlines()
        if lines and lines[0].startswith("Also:"):
            aliases = tuple(
                alias.strip()
                for alias in lines[0].removeprefix("Also:").split(";")
                if alias.strip()
            )
            lines = lines[1:]
        prose = " ".join(" ".join(lines).split())
        notes.append(Note(f"{topic}/{_slug(title)}", title, aliases, prose))
    return notes


class Index:
    """BM25 over a set of notes, with the said-in-full bonus on top."""

    def __init__(self, notes: Sequence[Note]) -> None:
        self.notes = tuple(notes)
        self._docs: list[Counter[str]] = []
        self._names: list[tuple[tuple[str, ...], ...]] = []
        frequency: Counter[str] = Counter()
        for note in self.notes:
            names = (note.title, *note.aliases)
            doc = Counter(tokens(note.text))
            for name in names:
                for word in tokens(name):
                    doc[word] += _NAME_WEIGHT
            self._docs.append(doc)
            self._names.append(tuple(t for n in names if (t := _bare(tokens(n)))))
            frequency.update(doc.keys())
        count = len(self.notes)
        self._idf = {
            word: math.log(1 + (count - n + 0.5) / (n + 0.5))
            for word, n in frequency.items()
        }
        self._lengths = [sum(doc.values()) for doc in self._docs]
        self._average = sum(self._lengths) / count if count else 0.0

    def _bm25(self, index: int, query: Iterable[str]) -> float:
        doc, length = self._docs[index], self._lengths[index]
        score = 0.0
        for word in query:
            tf = doc.get(word, 0)
            if not tf:
                continue
            norm = tf + _K1 * (1 - _B + _B * length / self._average)
            score += self._idf[word] * tf * (_K1 + 1) / norm
        return score

    def _said(self, index: int, query: Sequence[str]) -> float:
        """The summed IDF of the rarest-worded of this note's names found in
        the query as a run, or 0."""
        best = 0.0
        for name in self._names[index]:
            size = len(name)
            if any(
                tuple(query[i : i + size]) == name for i in range(len(query) - size + 1)
            ):
                best = max(best, sum(self._idf[word] for word in name))
        return best

    def _keyword_scores(self, query: str) -> dict[int, float]:
        """BM25 plus the said-in-full bonus, for every note that shares a
        meaningful word with `query`."""
        words = tokens(query)
        distinct = list(dict.fromkeys(words))
        bare = _bare(words)
        scores = {}
        for i in range(len(self.notes)):
            relevance = self._bm25(i, distinct)
            if relevance:
                scores[i] = relevance + NAME_BONUS * self._said(i, bare)
        return scores

    def search(self, query: str, limit: int = MAX_PASSAGES) -> list[Hit]:
        """The best notes for `query`, best first: at most `limit`, none below
        `MIN_SCORE`, and none under `RELATIVE_CUTOFF` of the best."""
        scored = sorted(
            (Hit(self.notes[i], s) for i, s in self._keyword_scores(query).items()),
            key=lambda hit: hit.score,
            reverse=True,
        )
        if not scored or scored[0].score < MIN_SCORE:
            return []
        floor = scored[0].score * RELATIVE_CUTOFF
        return [hit for hit in scored[:limit] if hit.score >= floor]

    def hybrid(
        self,
        query: str,
        query_vector: Sequence[float],
        vectors: "NoteVectors",
        *,
        opening: str | None = None,
        limit: int = MAX_PASSAGES,
    ) -> list[Hit]:
        """The best notes for `query` by keywords and meaning together: at
        most `limit`, best first, and only notes that clear the bar (the
        comment on LIFT_ALONE). A hit's score is its fused rank score.

        `opening` is the book name of the opening on the board ("French
        Defense: Advance Variation"): notes in its family rank higher, so
        "the advance variation" is this game's, but the boost never lets a
        note past the bar."""
        cosines = vectors.cosines(self.notes, query_vector)
        keyword = self._keyword_scores(query)
        baseline = sorted(cosines, reverse=True)[min(LIFT_BASELINE, len(cosines) - 1)]

        def clears(i: int) -> bool:
            lift = cosines[i] - baseline
            return lift >= LIFT_ALONE or (
                lift >= LIFT_SUPPORT and keyword.get(i, 0.0) >= BM25_SUPPORT
            )

        fused: dict[int, float] = {}
        by_meaning = sorted(range(len(cosines)), key=lambda i: -cosines[i])
        by_keyword = sorted(keyword, key=lambda i: -keyword[i])
        for ranking in (by_meaning, by_keyword):
            for rank, i in enumerate(ranking):
                fused[i] = fused.get(i, 0.0) + 1 / (RRF_K + rank + 1)
        family = opening.split(":")[0] if opening else None
        hits = [
            Hit(
                self.notes[i],
                score
                + (OPENING_BOOST if self.notes[i].title.split(":")[0] == family else 0),
            )
            for i, score in fused.items()
            if clears(i)
        ]
        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits[:limit]


def note_document(note: Note) -> str:
    """What a note is embedded as: its title, then its other names, then its
    text. The names matter: "castle kingside" is said, not written, in the
    castling note's prose."""
    names = f"Also: {'; '.join(note.aliases)}. " if note.aliases else ""
    return embeddings.document_text(note.title, names + note.text)


@functools.cache
def note_key(note: Note) -> str:
    """A note's cache key: a hash of exactly what is embedded, so an edited
    note is re-embedded and an unchanged one never is."""
    return hashlib.sha256(note_document(note).encode()).hexdigest()[:16]


def fit(vector: Sequence[float]) -> list[float]:
    """`vector` cut to DIMS and scaled back to unit length (Matryoshka)."""
    head = [float(x) for x in vector[:DIMS]]
    norm = math.sqrt(sum(x * x for x in head)) or 1.0
    return [x / norm for x in head]


@dataclass
class NoteVectors:
    """Note vectors from one model, keyed by `note_key`, kept in one JSON
    file: embedding every note takes ~20 s, a cached set loads in
    milliseconds. The test fixture is the same file plus the table queries."""

    model: str
    vectors: dict[str, list[float]]

    @classmethod
    def load(cls, path: Path) -> "NoteVectors | None":
        """The cached set at `path`, or None if there is none or it is not
        one (a cache is never worth an error)."""
        try:
            data = json.loads(path.read_text())
            return cls(str(data["model"]), dict(data["notes"]))
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def save(self, path: Path) -> None:
        """Write atomically: a crash mid-write leaves the old cache."""
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        body = {"model": self.model, "dims": DIMS, "notes": self.vectors}
        tmp.write_text(json.dumps(body, separators=(",", ":")))
        os.replace(tmp, path)

    def missing(self, notes: Iterable[Note]) -> list[Note]:
        return [note for note in notes if note_key(note) not in self.vectors]

    def cosines(
        self, notes: Sequence[Note], query_vector: Sequence[float]
    ) -> list[float]:
        """Each note's cosine to the query, in `notes` order. Stored vectors
        are already unit length, so this is a dot product."""
        query = fit(query_vector)
        return [
            sum(a * b for a, b in zip(query, self.vectors[note_key(note)], strict=True))
            for note in notes
        ]


def ensure_vectors(
    index: Index,
    embedder: embeddings.Embedder,
    cached: NoteVectors | None = None,
) -> NoteVectors:
    """Vectors for every note in `index`, embedding only those `cached` lacks
    and dropping those of notes that no longer exist. A different model
    starts over: its vectors don't compare with the old. Raises
    `EmbeddingsUnavailable` like the embedder."""
    todo = list(index.notes) if cached is None else cached.missing(index.notes)
    if cached is not None and not todo:
        keep = {note_key(note) for note in index.notes}
        return NoteVectors(
            cached.model, {k: v for k, v in cached.vectors.items() if k in keep}
        )
    done = embedder.embed_documents([note_document(note) for note in todo])
    if cached is None or cached.model != done.model:
        if len(todo) < len(index.notes):
            return ensure_vectors(index, embedder)
        cached = NoteVectors(done.model, {})
    fresh = {note_key(n): fit(v) for n, v in zip(todo, done.vectors, strict=True)}
    return ensure_vectors(
        index, embedder, NoteVectors(cached.model, {**cached.vectors, **fresh})
    )


def _files() -> list[tuple[str, str]]:
    root = resources.files("chessapp").joinpath(KNOWLEDGE_DIR)
    return sorted(
        (entry.name.removesuffix(".md"), entry.read_text(encoding="utf-8"))
        for entry in root.iterdir()
        if entry.name.endswith(".md")
    )


@functools.cache
def chess_knowledge() -> Index:
    """The chess notes, read and indexed once (a few milliseconds)."""
    return Index([note for topic, text in _files() for note in parse(text, topic)])


def lookup(query: str, limit: int = MAX_PASSAGES) -> list[Note]:
    """The notes that answer `query`, from every source: one, so far."""
    return [hit.note for hit in chess_knowledge().search(query, limit)]
