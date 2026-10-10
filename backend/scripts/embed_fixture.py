"""Regenerate the pinned vector fixture for the hybrid-search tests (#450).

CI never runs the embedding model: `tests/test_knowledge.py` reads note and
query vectors from `tests/fixtures/knowledge_vectors.json`, which this script
writes from the live embeddings service (`../embeddings`, #449). Re-run it
after editing a note or adding a query to `tests/knowledge_tables.py`; the
test that finds a vector missing says so.

    cd backend
    python scripts/embed_fixture.py                       # service on 127.0.0.1:8600
    python scripts/embed_fixture.py --url http://host:8600/v1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))
from chessapp import embeddings, knowledge  # noqa: E402
from knowledge_tables import (  # noqa: E402
    ANSWERS,
    CHATTER,
    SAID,
    SAID_IN_OPENING,
    TOPICAL,
)

FIXTURE = (
    Path(__file__).resolve().parent.parent
    / "tests"
    / "fixtures"
    / "knowledge_vectors.json"
)

# Enough to keep every ranking the tests pin, small enough to keep the file
# about a megabyte.
DECIMALS = 4


def queries() -> list[str]:
    """Every query the hybrid tests embed, once each, sorted."""
    return sorted(
        {q for q, _ in ANSWERS}
        | {q for q, _ in SAID}
        | {q for q, _, _ in SAID_IN_OPENING}
        | set(CHATTER)
        | set(TOPICAL)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--url", default="http://127.0.0.1:8600/v1")
    args = parser.parse_args()

    embedder = embeddings.create_embedder(args.url)
    index = knowledge.chess_knowledge()
    notes = knowledge.ensure_vectors(index, embedder)
    asked = queries()
    embedded = embedder.embed_documents([embeddings.query_text(q) for q in asked])
    if embedded.model != notes.model:
        sys.exit(f"the model changed mid-run: {notes.model} → {embedded.model}")
    root = args.url.rstrip("/").removesuffix("/v1")
    build = httpx.get(f"{root}/props", timeout=5).json().get("build_info", "")

    def rounded(vector: list[float]) -> list[float]:
        return [round(x, DECIMALS) for x in vector]

    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE.write_text(
        json.dumps(
            {
                "model": notes.model,
                "build": build,
                "dims": knowledge.DIMS,
                "notes": {k: rounded(v) for k, v in sorted(notes.vectors.items())},
                "queries": {
                    q: rounded(knowledge.fit(v))
                    for q, v in zip(asked, embedded.vectors, strict=True)
                },
            },
            separators=(",", ":"),
        )
        + "\n"
    )
    size = FIXTURE.stat().st_size / 1e6
    print(
        f"{len(notes.vectors)} notes, {len(asked)} queries, {knowledge.DIMS} dims "
        f"→ {FIXTURE.name} ({size:.1f} MB, {build})"
    )


if __name__ == "__main__":
    main()
