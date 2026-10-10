"""Text embeddings from the shared workspace service (#449), over plain httpx.

The service is a CPU llama-server running EmbeddingGemma 2 behind the OpenAI
`/v1/embeddings` wire (`../embeddings`, port 8600). It is optional the way
the brain and voice are: any failure is `EmbeddingsUnavailable`, and the
caller falls back to keyword search. Nothing here decides what a vector
means; `knowledge.py` ranks with them.

EmbeddingGemma is trained with task prompts and the server adds none, so the
client prefixes every input: a query as a search query, a note as a titled
document. A query and a document only compare when both carry theirs.

The httpx client is injected in tests (`httpx.MockTransport`), so no live
service is ever required.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field

import httpx

# One query is ~10 ms on the service, so a caller waits at most this long for
# one before it goes without. Connecting to a stopped container fails fast.
QUERY_TIMEOUT = httpx.Timeout(1.0, connect=0.2)

# The gather step's budget (#451): it runs before every turn's first model
# call, so a slow service costs the turn its notes (keywords stand in), never
# more than this.
GATHER_TIMEOUT = httpx.Timeout(0.3, connect=0.2)

# Embedding the corpus is a batch job (~20 s for every note), off the turn.
BATCH_TIMEOUT = httpx.Timeout(120.0, connect=2.0)

# Notes per request when embedding the corpus.
BATCH_SIZE = 16


def query_text(query: str) -> str:
    """`query` as EmbeddingGemma's retrieval query."""
    return f"task: search result | query: {query}"


def document_text(title: str, text: str) -> str:
    """A titled passage as EmbeddingGemma's retrieval document."""
    return f"title: {title} | text: {text}"


class EmbeddingsUnavailable(Exception):
    """No usable vectors: the service is down, slow, erroring or garbled."""


@dataclass(frozen=True)
class Embedded:
    """Vectors in input order, and the model that made them (a cache key:
    vectors from two models never compare)."""

    model: str
    vectors: list[list[float]]


@dataclass
class Embedder:
    client: httpx.Client
    query_timeout: httpx.Timeout = field(default_factory=lambda: QUERY_TIMEOUT)

    def embed(
        self, inputs: Sequence[str], timeout: httpx.Timeout | None = None
    ) -> Embedded:
        """Vectors for `inputs`, already prefixed, in one request, within
        `timeout` (the query timeout by default)."""
        try:
            response = self.client.post(
                "embeddings",
                json={"input": list(inputs)},
                timeout=timeout or self.query_timeout,
            )
            response.raise_for_status()
            body = response.json()
            data = sorted(body["data"], key=lambda item: item["index"])
            vectors = [[float(x) for x in item["embedding"]] for item in data]
            model = str(body["model"])
        except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
            raise EmbeddingsUnavailable(f"embeddings request failed: {exc}") from exc
        if len(vectors) != len(inputs) or not all(vectors):
            raise EmbeddingsUnavailable(
                f"asked for {len(inputs)} vectors, got {len(vectors)}"
            )
        return Embedded(model, vectors)

    def embed_query(self, query: str) -> Embedded:
        """One player's words, as a search query."""
        return self.embed([query_text(query)])

    def embed_documents(self, documents: Sequence[str]) -> Embedded:
        """Many documents, already in `document_text` form, in batches."""
        model, vectors = "", []
        for start in range(0, len(documents), BATCH_SIZE):
            batch = self.embed(documents[start : start + BATCH_SIZE], BATCH_TIMEOUT)
            if model and batch.model != model:
                raise EmbeddingsUnavailable("the model changed mid-batch")
            model = batch.model
            vectors += batch.vectors
        return Embedded(model, vectors)


def create_embedder(
    base_url: str,
    client: httpx.Client | None = None,
    *,
    timeout: httpx.Timeout = QUERY_TIMEOUT,
) -> Embedder:
    """An Embedder against the service's OpenAI root (e.g. host:8600/v1),
    waiting at most `timeout` for a query."""
    if client is None:
        client = httpx.Client(base_url=base_url, timeout=timeout)
    return Embedder(client, timeout)
