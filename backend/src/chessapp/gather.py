"""The gather step (#448, #451): notes found from the player's own words.

Before either phase runs, the turn searches the chess notes with exactly what
the player said and hands whatever clears the relevance bar to the planner
and the narrator as context. No tool is called, no query is written, and
nothing changes: code supplies material, and the model still decides what
the player meant and what to say. The planner used to call `lookup` for this
on every chat turn (#422); it is the MCP surface's tool now.

**Search.** Hybrid (`knowledge.Index.hybrid`) when the embeddings service is
up and the note vectors are ready; keywords alone (`Index.search`) otherwise,
so a stopped service costs recall, never the turn. A cold start embeds the
notes in the background (~20 s) and caches them in the save directory.

**"This opening".** The player's words for the opening on the board ("what's
the plan in this opening?") name no opening, so no opening note clears the
bar. When the search's own hits include a note about openings in general,
the note for the opening on the board comes along (decided 2026-10-10,
#451). Chatter never reaches those notes, so it never brings the opening's.
"""

import logging
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from chessapp import embeddings, knowledge

logger = logging.getLogger(__name__)

Source = Literal["hybrid", "bm25_fallback"]

# The notes about openings in general. A hit on one says the ask is about
# openings, and the opening on the board is the one it means.
OPENING_GENERAL = frozenset(
    {
        "strategy/opening-principles",
        "strategy/choosing-an-opening",
        "terms/opening-theory",
    }
)

# After a failed warm-up, how long gather waits before trying the service
# again. Gather never waits on a warm-up: it searches by keyword meanwhile.
RETRY_WARM_S = 60.0

# The file the note vectors are cached in, under the save directory.
CACHE_NAME = "knowledge_vectors.json"


@dataclass(frozen=True)
class Gathered:
    """What one turn gathered. Empty `passages` is an answer: nothing in the
    notes is about what the player said."""

    passages: tuple[knowledge.Hit, ...] = ()
    source: Source = "bm25_fallback"
    ms: float = 0.0
    # The embedding model a hybrid search ran on, None for keywords alone:
    # vectors from two models rank differently, so a record names its own.
    model: str | None = None

    def __bool__(self) -> bool:
        return bool(self.passages)

    def view(self) -> list[dict[str, str]]:
        """The passages as the phases read them, best first."""
        return [
            {"topic": hit.note.title, "text": hit.note.text} for hit in self.passages
        ]

    def as_trace(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "model": self.model,
            "ms": round(self.ms, 1),
            "passages": [
                {
                    "id": hit.note.id,
                    "topic": hit.note.title,
                    "score": round(hit.score, 4),
                }
                for hit in self.passages
            ],
        }


def stand_in(
    index: knowledge.Index, hits: Sequence[knowledge.Hit], opening: str | None
) -> list[knowledge.Hit]:
    """`hits`, with the note for `opening` added when a hit is about openings
    in general. The book name's own note if there is one, else its family's;
    nothing if the notes have neither. It takes the place of the weakest
    general note when the list is full, and scores just under the best hit."""
    if not opening or not any(hit.note.id in OPENING_GENERAL for hit in hits):
        return list(hits)
    family = opening.split(":")[0]
    titled = {note.title: note for note in index.notes}
    note = titled.get(opening) or titled.get(family)
    if note is None or any(hit.note.id == note.id for hit in hits):
        return list(hits)
    kept = list(hits)
    if len(kept) >= knowledge.MAX_PASSAGES:
        weakest = max(
            (i for i, hit in enumerate(kept) if hit.note.id in OPENING_GENERAL),
        )
        kept.pop(weakest)
    best = kept[0].score if kept else 1.0
    kept.insert(1 if kept else 0, knowledge.Hit(note, best))
    return kept


@dataclass
class Searcher:
    """The app's one gather: the notes, and the embeddings that search them
    by meaning when there are any. Thread-safe: the warm-up runs beside
    turns."""

    index: knowledge.Index
    embedder: embeddings.Embedder | None = None
    cache_path: Path | None = None
    _vectors: knowledge.NoteVectors | None = field(default=None, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    _warming: bool = field(default=False, init=False)
    _last_try: float = field(default=float("-inf"), init=False)

    def __post_init__(self) -> None:
        if self.embedder is None:
            return
        cached = (
            knowledge.NoteVectors.load(self.cache_path) if self.cache_path else None
        )
        if cached is not None and not cached.missing(self.index.notes):
            self._vectors = cached

    @property
    def service(self) -> str | None:
        """The embeddings service's URL, or None for keywords alone (the
        serving manifest names it)."""
        return str(self.embedder.client.base_url) if self.embedder else None

    @property
    def model(self) -> str | None:
        """The embedding model the note vectors came from, once they are
        ready (the serving manifest names it)."""
        vectors = self._vectors
        return vectors.model if vectors is not None else None

    def warm(self) -> bool:
        """Embed whatever notes the cache lacks and save it. True when the
        vectors are ready. Safe to call from any thread, any number of times."""
        if self.embedder is None:
            return False
        with self._lock:
            if self._vectors is not None:
                return True
            if self._warming:
                return False
            self._warming = True
            self._last_try = time.monotonic()
        try:
            cached = (
                knowledge.NoteVectors.load(self.cache_path) if self.cache_path else None
            )
            vectors = knowledge.ensure_vectors(self.index, self.embedder, cached)
            if self.cache_path is not None:
                try:
                    vectors.save(self.cache_path)
                except OSError:
                    logger.warning("knowledge_vectors_not_saved", exc_info=True)
            with self._lock:
                self._vectors = vectors
            logger.info("knowledge_vectors_ready model=%s", vectors.model)
            return True
        except embeddings.EmbeddingsUnavailable as exc:
            logger.warning("knowledge_vectors_unavailable: %s", exc)
            return False
        finally:
            with self._lock:
                self._warming = False

    def warm_in_background(self) -> None:
        """Start a warm-up unless one is running or failed too recently."""
        if self.embedder is None or self._vectors is not None:
            return
        with self._lock:
            due = time.monotonic() - self._last_try >= RETRY_WARM_S
            if self._warming or not due:
                return
        threading.Thread(target=self.warm, name="knowledge-warm", daemon=True).start()

    def gather(self, text: str, opening: str | None = None) -> Gathered:
        """The notes about what the player said, for this turn's phases."""
        started = time.monotonic()
        hits, source = self._search(text, opening)
        hits = stand_in(self.index, hits, opening)
        return Gathered(
            tuple(hits),
            source,
            (time.monotonic() - started) * 1000,
            self.model if source == "hybrid" else None,
        )

    def _search(
        self, text: str, opening: str | None
    ) -> tuple[list[knowledge.Hit], Source]:
        if not text.strip():
            return [], "bm25_fallback"
        vectors = self._vectors
        if self.embedder is not None and vectors is not None:
            try:
                query = self.embedder.embed_query(text)
            except embeddings.EmbeddingsUnavailable as exc:
                logger.warning("gather_fell_back: %s", exc)
            else:
                if query.model == vectors.model:
                    hits = self.index.hybrid(
                        text, query.vectors[0], vectors, opening=opening
                    )
                    return hits, "hybrid"
                # The service changed models under the cache: re-embed.
                logger.warning("gather_model_changed %s", query.model)
                with self._lock:
                    self._vectors = None
                    self._last_try = float("-inf")
        self.warm_in_background()
        return self.index.search(text), "bm25_fallback"


def searcher_from(
    base_url: str | None, cache_dir: Path | None, *, warm: bool = True
) -> Searcher:
    """The app's searcher: hybrid against the service at `base_url`, keywords
    alone without one. `warm` starts embedding the notes in the background
    now, so the first turns don't wait for it (they search by keyword)."""
    embedder = (
        embeddings.create_embedder(base_url, timeout=embeddings.GATHER_TIMEOUT)
        if base_url
        else None
    )
    searcher = Searcher(
        knowledge.chess_knowledge(),
        embedder,
        cache_dir / CACHE_NAME if cache_dir is not None else None,
    )
    if warm:
        searcher.warm_in_background()
    return searcher
