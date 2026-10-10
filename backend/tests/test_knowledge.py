"""The second brain's retrieval and the `lookup` tool (#374).

Retrieval is deterministic, so it is tested here the way the rest of the core
is: a table of questions players ask, each with the note that answers it,
and a table of asks no note should answer. A wording that stops finding its
note is a ranking regression, caught without a model in the loop.

The corpus is tested for shape too: every file names its sources, every note
is short enough to read out, and every opening note agrees with the Lichess
book the app names openings from — its title is a book name, and the moves it
opens with reach a position the book files under the same family.
"""

import json
import re
from pathlib import Path

import chess
import pytest

from chessapp import embeddings, handoff, knowledge, openings
from chessapp.game import GameSession
from chessapp.tools import RETRY_DIFFERENT_ARGS, ToolContext, build_registry
from knowledge_tables import (
    ANSWERS,
    CHATTER,
    NOTHING,
    SAID,
    SAID_IN_OPENING,
    TOPICAL,
)

INDEX = knowledge.chess_knowledge()


@pytest.mark.parametrize(("query", "expected"), ANSWERS)
def test_the_note_that_answers_comes_first(query, expected):
    hits = INDEX.search(query)
    assert hits, f"nothing found for {query!r}"
    assert hits[0].note.id == expected, [hit.note.id for hit in hits]


@pytest.mark.parametrize("query", NOTHING)
def test_asks_no_note_answers_find_nothing(query):
    assert INDEX.search(query) == []


def test_every_expected_note_exists():
    ids = {note.id for note in INDEX.notes}
    assert {expected for _, expected in ANSWERS} <= ids


def test_a_lookup_returns_at_most_three_notes_best_first():
    hits = INDEX.search("Sicilian Defense")
    assert 1 <= len(hits) <= knowledge.MAX_PASSAGES
    scores = [hit.score for hit in hits]
    assert scores == sorted(scores, reverse=True)
    assert all(score >= scores[0] * knowledge.RELATIVE_CUTOFF for score in scores)


def test_a_longer_name_said_beats_its_family():
    # "Sicilian Najdorf" names both notes; the longer, rarer name wins.
    first, second = INDEX.search("Sicilian Najdorf")[:2]
    assert first.note.id == "openings/sicilian-defense-najdorf-variation"
    assert second.note.id == "openings/sicilian-defense"


def test_spelling_and_case_do_not_matter():
    for query in ("GRÜNFELD", "grunfeld", "Grünfeld Defence"):
        assert INDEX.search(query)[0].note.id == "openings/grunfeld-defense"


def test_tokens_drop_fillers_and_meet_on_stems():
    assert knowledge.tokens("um what's the idea, uh?") == ["idea"]
    assert knowledge.tokens("castling") == knowledge.tokens("castle")
    assert knowledge.tokens("pinned") == knowledge.tokens("pins")
    assert knowledge.tokens("1. e4 e5") == ["e4", "e5"]


# --- hybrid search (#450) -------------------------------------------------------
#
# Vectors come from the pinned fixture, never a live model: regenerate it with
# `python scripts/embed_fixture.py` after editing a note or a table.

FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures" / "knowledge_vectors.json").read_text()
)
VECTORS = knowledge.NoteVectors(FIXTURE["model"], FIXTURE["notes"])
QUERY_VECTORS = FIXTURE["queries"]

# Hybrid recall@3 over ANSWERS + SAID when the bar was calibrated
# (2026-10-10): 160 of 183. A change that loses a match fails here; one that
# gains raises the floor.
HYBRID_RECALL_FLOOR = 160


def _hybrid(query: str, opening: str | None = None) -> list[str]:
    hits = INDEX.hybrid(query, QUERY_VECTORS[query], VECTORS, opening=opening)
    return [hit.note.id for hit in hits]


def _keywords(query: str) -> list[str]:
    return [hit.note.id for hit in INDEX.search(query)]


def _meaning(query: str) -> list[str]:
    """Embeddings alone, with only the lift-alone bar: the third column."""
    cosines = VECTORS.cosines(INDEX.notes, QUERY_VECTORS[query])
    ranked = sorted(range(len(cosines)), key=lambda i: -cosines[i])
    baseline = cosines[ranked[knowledge.LIFT_BASELINE]]
    return [
        INDEX.notes[i].id
        for i in ranked[: knowledge.MAX_PASSAGES]
        if cosines[i] - baseline >= knowledge.LIFT_ALONE
    ]


def test_the_fixture_covers_every_note_and_query():
    missing_notes = [note.id for note in VECTORS.missing(INDEX.notes)]
    asked = (
        {q for q, _ in ANSWERS}
        | {q for q, _ in SAID}
        | {q for q, _, _ in SAID_IN_OPENING}
        | set(CHATTER)
        | set(TOPICAL)
    )
    missing_queries = sorted(asked - set(QUERY_VECTORS))
    assert not missing_notes and not missing_queries, (
        "the vector fixture is stale: run `python scripts/embed_fixture.py` "
        f"(notes {missing_notes[:5]}, queries {missing_queries[:5]})"
    )
    assert FIXTURE["dims"] == knowledge.DIMS


@pytest.mark.parametrize("query", CHATTER)
def test_chatter_finds_nothing(query):
    assert _hybrid(query) == []


def test_hybrid_finds_at_least_what_keywords_find():
    rows = ANSWERS + SAID
    hybrid = sum(expected in _hybrid(query) for query, expected in rows)
    keywords = sum(expected in _keywords(query) for query, expected in rows)
    assert hybrid >= keywords
    assert hybrid >= HYBRID_RECALL_FLOOR, f"recall@3 fell to {hybrid}/{len(rows)}"


@pytest.mark.parametrize(("query", "opening", "expected"), SAID_IN_OPENING)
def test_the_opening_on_the_board_ranks_its_own_note_first(query, opening, expected):
    assert _hybrid(query, opening)[:1] == [expected]


def test_the_opening_boost_never_lets_a_note_past_the_bar():
    # Chatter in a named opening still finds nothing.
    for query in ("play e4", "nice move", "undo that"):
        assert _hybrid(query, "Sicilian Defense: Najdorf Variation") == []


def test_hybrid_returns_at_most_three_notes_best_first():
    hits = INDEX.hybrid(
        "what's the idea behind the Sicilian",
        QUERY_VECTORS["what's the idea behind the Sicilian"],
        VECTORS,
    )
    assert 1 <= len(hits) <= knowledge.MAX_PASSAGES
    scores = [hit.score for hit in hits]
    assert scores == sorted(scores, reverse=True)


def test_retrieval_report():
    """Not a gate: the table `docs/second-brain.md` records. Run with -s."""
    methods = {"bm25": _keywords, "embeddings": _meaning, "hybrid": _hybrid}
    print(
        f"\n{'':12}{'ANSWERS r@1/r@3':>18}{'SAID r@1/r@3':>16}"
        f"{'CHATTER found':>15}{'TOPICAL found':>15}"
    )
    for name, find in methods.items():
        cells = []
        for rows in (ANSWERS, SAID):
            found = [find(query) for query, _ in rows]
            first = sum(f[:1] == [e] for f, (_, e) in zip(found, rows, strict=True))
            top = sum(e in f for f, (_, e) in zip(found, rows, strict=True))
            cells.append(f"{first}/{top} of {len(rows)}")
        chatter = sum(bool(find(q)) for q in CHATTER)
        topical = sum(bool(find(q)) for q in TOPICAL)
        print(
            f"{name:12}{cells[0]:>18}{cells[1]:>16}"
            f"{f'{chatter}/{len(CHATTER)}':>15}{f'{topical}/{len(TOPICAL)}':>15}"
        )


# --- the vector cache ------------------------------------------------------------


class _CountingEmbedder:
    """Stands in for the service: a vector per document from its text, and a
    record of what it was asked to embed."""

    def __init__(self, model: str = "m1") -> None:
        self.model = model
        self.asked: list[str] = []

    def embed_documents(self, documents):
        self.asked += documents
        vectors = [
            [float(len(d)), float(sum(map(ord, d)) % 97), 1.0] for d in documents
        ]
        return embeddings.Embedded(self.model, vectors)


def _small_index(fork_text: str = "One piece attacks two at once.") -> knowledge.Index:
    return knowledge.Index(
        [
            knowledge.Note("tactics/fork", "Fork", ("double attack",), fork_text),
            knowledge.Note("tactics/pin", "Pin", (), "A piece cannot move away."),
            knowledge.Note("rules/check", "Check", (), "The king is attacked."),
        ]
    )


def test_the_cache_embeds_every_note_once():
    index, embedder = _small_index(), _CountingEmbedder()
    vectors = knowledge.ensure_vectors(index, embedder)
    assert len(embedder.asked) == 3 and not vectors.missing(index.notes)
    again = knowledge.ensure_vectors(index, embedder, vectors)
    assert len(embedder.asked) == 3
    assert again.vectors == vectors.vectors


def test_an_edited_note_is_reembedded_alone_and_its_old_vector_dropped():
    embedder = _CountingEmbedder()
    vectors = knowledge.ensure_vectors(_small_index(), embedder)
    edited = _small_index("One piece attacks two pieces at the same time.")
    updated = knowledge.ensure_vectors(edited, embedder, vectors)
    assert len(embedder.asked) == 4
    assert "One piece attacks two pieces" in embedder.asked[-1]
    assert set(updated.vectors) == {knowledge.note_key(n) for n in edited.notes}


def test_a_new_model_reembeds_everything():
    index = _small_index()
    old = knowledge.ensure_vectors(index, _CountingEmbedder("m1"))
    newer = _CountingEmbedder("m2")
    edited = _small_index("Two targets, one attacker.")
    vectors = knowledge.ensure_vectors(edited, newer, old)
    assert vectors.model == "m2"
    assert len(newer.asked) == 1 + 3  # the edited note, then everything again
    assert not vectors.missing(edited.notes)


def test_a_note_document_carries_its_names():
    note = _small_index().notes[0]
    assert knowledge.note_document(note) == (
        "title: Fork | text: Also: double attack. One piece attacks two at once."
    )


def test_the_cache_file_round_trips(tmp_path):
    index = _small_index()
    vectors = knowledge.ensure_vectors(index, _CountingEmbedder())
    path = tmp_path / "cache" / "vectors.json"
    vectors.save(path)
    assert knowledge.NoteVectors.load(path) == vectors
    assert not list(path.parent.glob("*.tmp"))


def test_a_missing_or_broken_cache_loads_as_none(tmp_path):
    assert knowledge.NoteVectors.load(tmp_path / "absent.json") is None
    (tmp_path / "junk.json").write_text("{not json")
    assert knowledge.NoteVectors.load(tmp_path / "junk.json") is None
    (tmp_path / "shape.json").write_text("[1, 2]")
    assert knowledge.NoteVectors.load(tmp_path / "shape.json") is None


def test_vectors_are_cut_and_normalized_to_dims():
    vector = knowledge.fit([3.0, 4.0] + [0.0] * (knowledge.DIMS + 10))
    assert len(vector) == knowledge.DIMS
    assert vector[:2] == pytest.approx([0.6, 0.8])


# --- the corpus ------------------------------------------------------------------


def _files() -> list[tuple[str, str]]:
    return knowledge._files()


def test_every_file_names_its_sources():
    for topic, text in _files():
        preamble = text.split("\n## ", 1)[0]
        assert "Sources" in preamble, f"{topic}.md says nothing of its sources"


def test_every_note_has_a_unique_id_and_readable_length():
    ids = [note.id for note in INDEX.notes]
    assert len(ids) == len(set(ids)), "two notes share a title in one file"
    for note in INDEX.notes:
        words = len(note.text.split())
        assert 20 <= words <= 130, f"{note.id} is {words} words"


def test_the_corpus_is_broad():
    topics = {note.id.split("/")[0] for note in INDEX.notes}
    assert topics >= {"openings", "strategy", "tactics", "endgames", "rules", "history"}
    assert len(INDEX.notes) >= 300


def test_opening_titles_are_book_names():
    """An opening note's title is the name the app reads off the board
    (`openings.book`), so a planner that copies the opening from its state
    block names the note word for word."""
    names = {name for _, name in openings.book().positions.values()}
    for note in INDEX.notes:
        if note.id.startswith("openings/"):
            assert note.title in names or note.title in openings.families(), note.title


_MOVE = re.compile(r"^(?:\d+\.(?:\.\.)?)?([KQRBNOa-h][^\s]*?)[?!]*$")


def _leading_moves(text: str) -> list[str]:
    """The SAN moves a note opens with: "1. e4 e5 2. Nf3 Nc6 3. Bb5." → the
    five moves. Move numbers are skipped; the run ends at the first word that
    is not a move."""
    moves = []
    for word in text.split():
        word = word.rstrip(".,;:")
        if re.fullmatch(r"\d+\.?(?:\.\.)?", word):
            continue
        match = _MOVE.match(word)
        if not match or not re.search(r"\d|O-O", match.group(1)):
            break
        moves.append(match.group(1))
    return moves


def test_opening_lines_are_legal_and_reach_the_named_opening():
    """Every opening note that opens with its moves gives legal moves, and
    the book files the position they reach under the note's own family."""
    checked = 0
    for note in INDEX.notes:
        if not note.id.startswith("openings/") or not note.text.startswith("1."):
            continue
        session = GameSession()
        for san in _leading_moves(note.text):
            assert session.submit_move(san).legal, f"{note.id}: {san}"
        reached = openings.names_on_line(session)
        family = note.title.split(":")[0]
        assert any(name.split(":")[0] == family for name in reached), (
            f"{note.id}: the line reaches {sorted(reached)}"
        )
        checked += 1
    assert checked >= 50


def test_every_move_line_in_the_corpus_parses():
    """Any run of numbered moves anywhere in a note starts from the opening
    position and is legal: "1. e4 e5 2. Bc4 ..." in a trap note, too."""
    for note in INDEX.notes:
        for match in re.finditer(r"(?<![\w.])1\. [^()]*", note.text):
            moves = _leading_moves(match.group(0))
            board = chess.Board()
            for san in moves:
                board.push_san(san)


# --- the tool --------------------------------------------------------------------


@pytest.fixture
def registry():
    return build_registry(ToolContext(session=GameSession()))


def test_lookup_returns_the_passages(registry):
    result = registry.dispatch("lookup", {"query": "how does en passant work"})
    assert result["ok"] is True
    first = result["passages"][0]
    assert first["topic"] == "En passant"
    assert "fifth rank" in first["text"]
    assert set(first) == {"topic", "text"}


def test_lookup_with_no_match_says_nothing_was_found(registry):
    result = registry.dispatch("lookup", {"query": "pizza recipe"})
    assert result == {
        "ok": True,
        "passages": [],
        "summary": "Nothing in the chess notes matches that.",
    }


@pytest.mark.parametrize("query", ["what is it?", "  ", "um, uh"])
def test_a_lookup_that_names_nothing_is_refused(registry, query):
    result = registry.dispatch("lookup", {"query": query})
    assert result["ok"] is False
    assert result["retry"] == RETRY_DIFFERENT_ARGS


def test_an_empty_or_long_query_fails_the_schema(registry):
    assert registry.dispatch("lookup", {"query": ""})["ok"] is False
    assert registry.dispatch("lookup", {"query": "x" * 201})["ok"] is False
    assert registry.dispatch("lookup", {})["ok"] is False


def test_lookup_moves_nothing(registry):
    ctx = registry.context
    before = (ctx.session.fen(), ctx.board_version)
    registry.dispatch("lookup", {"query": "Sicilian"})
    assert (ctx.session.fen(), ctx.board_version) == before


def test_the_handoff_reads_a_lookup_as_consulted():
    result = {"ok": True, "passages": [{"topic": "Fork", "text": "..."}]}
    built = handoff.build([{"name": "lookup", "result": result}])
    assert built.kind == "completed"
    assert [entry.tool for entry in built.consulted] == ["lookup"]
    assert not built.performed
