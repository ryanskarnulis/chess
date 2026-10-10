"""The gather step (#451): notes found from the player's own words, before
either phase runs, and what each phase is shown.

The searcher runs on the pinned vector fixture (`test_knowledge.py`), with a
stand-in embedder that answers each table query with its recorded vector, so
no model runs here. The pipeline tests use the keyword searcher (no
embedder), which is deterministic and needs no fixture.
"""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from chessapp import api, embeddings, gather, knowledge
from chessapp.coordinator import TurnCoordinator
from chessapp.game import GameSession
from chessapp.handoff import NARRATOR_NOTES_LABEL, PLANNER_NOTES_LABEL
from chessapp.llama_brain import LlamaBrain
from chessapp.tools import (
    LOOKUP,
    ToolContext,
    brain_tool_definitions,
    build_registry,
)
from fakes import (
    CollectedTurns,
    FakeEngine,
    ScriptedProvider,
    scripted_app,
    text_turn,
)

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "knowledge_vectors.json"
FIXTURE = json.loads(FIXTURE_PATH.read_text())
INDEX = knowledge.chess_knowledge()


class FixtureEmbedder:
    """The service, as the fixture recorded it: a table query's vector, and
    every note's. `fail` makes it a stopped service."""

    def __init__(self, model: str = FIXTURE["model"], fail: bool = False) -> None:
        self.model = model
        self.fail = fail
        self.queries: list[str] = []
        self.documents = 0
        by_key = FIXTURE["notes"]
        self._docs = {
            knowledge.note_document(n): by_key[knowledge.note_key(n)]
            for n in INDEX.notes
        }
        self.client = type("Client", (), {"base_url": "http://embeddings.test/v1"})

    def embed_query(self, query: str) -> embeddings.Embedded:
        self.queries.append(query)
        if self.fail:
            raise embeddings.EmbeddingsUnavailable("connection refused")
        return embeddings.Embedded(self.model, [FIXTURE["queries"][query]])

    def embed_documents(self, documents) -> embeddings.Embedded:
        if self.fail:
            raise embeddings.EmbeddingsUnavailable("connection refused")
        self.documents += len(documents)
        return embeddings.Embedded(self.model, [self._docs[d] for d in documents])


@pytest.fixture
def cache(tmp_path) -> Path:
    """A copy of the fixture as the searcher's cache: a searcher may rewrite
    its cache, and the fixture is never its to rewrite."""
    path = tmp_path / gather.CACHE_NAME
    path.write_text(FIXTURE_PATH.read_text())
    return path


@pytest.fixture
def ready(cache):
    """A searcher whose cache is the fixture: ready from the first turn."""

    def make(embedder=None) -> gather.Searcher:
        return gather.Searcher(INDEX, embedder or FixtureEmbedder(), cache)

    return make


def _topics(found: gather.Gathered) -> list[str]:
    return [passage["topic"] for passage in found.view()]


# --- the searcher -----------------------------------------------------------------


def test_a_question_gathers_its_note_by_meaning(ready):
    found = ready().gather("the computer that beat Kasparov in the nineties")
    assert _topics(found)[:1] == ["Deep Blue"]
    assert found.source == "hybrid"
    assert found.model == FIXTURE["model"]


@pytest.mark.parametrize("text", ["play e4", "undo that", "nice move", "good game"])
def test_chatter_gathers_nothing(text, ready):
    found = ready().gather(text)
    assert not found and found.source == "hybrid"


def test_words_with_nothing_in_them_gather_nothing_and_ask_nobody(ready):
    embedder = FixtureEmbedder()
    assert not ready(embedder).gather("   ")
    assert embedder.queries == []


def test_a_note_about_openings_brings_the_one_on_the_board(ready):
    """ "Opening principles" is about openings in general: in a Ruy Lopez game
    the Ruy Lopez note comes along (the stand-in, decided 2026-10-10)."""
    without = _topics(ready().gather("opening principles"))
    within = _topics(ready().gather("opening principles", "Ruy Lopez: Morphy Defense"))
    assert "Ruy Lopez: Morphy Defense" not in without
    assert "Ruy Lopez: Morphy Defense" in within
    assert within[0] == without[0], "the general note keeps its place"


def test_the_opening_alone_never_brings_its_note(ready):
    for text in ("play e4", "nice move", "how does the horse move again"):
        found = ready().gather(text, "Ruy Lopez: Morphy Defense")
        assert "Ruy Lopez: Morphy Defense" not in _topics(found)


def _hit(note_id: str, score: float = 1.0) -> knowledge.Hit:
    return knowledge.Hit(next(n for n in INDEX.notes if n.id == note_id), score)


def test_the_stand_in_takes_a_general_notes_place_when_the_list_is_full():
    hits = [
        _hit("strategy/opening-principles", 3),
        _hit("strategy/choosing-an-opening", 2),
        _hit("terms/opening-theory", 1),
    ]
    kept = gather.stand_in(INDEX, hits, "Italian Game")
    assert [h.note.id for h in kept] == [
        "strategy/opening-principles",
        "openings/italian-game",
        "strategy/choosing-an-opening",
    ]


def test_the_stand_in_falls_back_to_the_family_note():
    kept = gather.stand_in(
        INDEX, [_hit("strategy/opening-principles")], "Italian Game: Some Line"
    )
    assert [h.note.id for h in kept][1:] == ["openings/italian-game"]


def test_the_stand_in_adds_nothing_without_a_note_for_the_opening():
    hits = [_hit("strategy/opening-principles")]
    assert gather.stand_in(INDEX, hits, "No Such Opening: At All") == hits
    assert gather.stand_in(INDEX, hits, None) == hits


def test_a_stopped_service_falls_back_to_keywords(ready):
    found = ready(FixtureEmbedder(fail=True)).gather("how does en passant work")
    assert found.source == "bm25_fallback" and found.model is None
    assert _topics(found)[0] == "En passant"


def test_a_cold_start_searches_by_keyword_until_the_notes_are_embedded(tmp_path):
    embedder = FixtureEmbedder()
    searcher = gather.Searcher(INDEX, embedder, tmp_path / "vectors.json")
    searcher.warm_in_background = lambda: None  # warmed by hand below
    assert searcher.gather("how does en passant work").source == "bm25_fallback"
    assert searcher.warm()
    assert embedder.documents == len(INDEX.notes)
    assert (tmp_path / "vectors.json").exists()
    found = searcher.gather("the computer that beat Kasparov in the nineties")
    assert found.source == "hybrid"
    # A restart reads the cache instead of embedding again.
    again = gather.Searcher(INDEX, FixtureEmbedder(), tmp_path / "vectors.json")
    assert again.model == FIXTURE["model"]


def test_a_new_model_on_the_service_drops_the_old_vectors(ready, monkeypatch):
    searcher = ready(FixtureEmbedder(model="another-model"))
    warmed = []
    monkeypatch.setattr(searcher, "warm_in_background", lambda: warmed.append(1))
    assert searcher.gather("play e4").source == "bm25_fallback"
    assert searcher.model is None
    assert warmed, "the notes are re-embedded on the new model"


def test_without_a_service_the_searcher_is_keywords_alone():
    searcher = gather.searcher_from(None, None)
    assert searcher.service is None
    assert searcher.gather("how does en passant work").source == "bm25_fallback"


def test_the_trace_names_what_was_found_and_how(ready):
    found = ready().gather("the computer that beat Kasparov in the nineties")
    traced = found.as_trace()
    assert traced["source"] == "hybrid"
    assert traced["model"] == FIXTURE["model"]
    assert traced["passages"][0]["id"] == "history/deep-blue"
    assert set(traced["passages"][0]) == {"id", "topic", "score"}


# --- the turn ---------------------------------------------------------------------


def _keyword_app(*responses, narrations=(), engine=None, tracer=None):
    ctx = ToolContext(session=GameSession(), engine=engine or FakeEngine("e7e5"))
    app, brain = scripted_app(
        ctx,
        *responses,
        searcher=gather.Searcher(INDEX),
        tracer=tracer,
    )
    brain._narrations = list(narrations)
    return TestClient(app), brain, ctx


def test_the_planner_is_handed_the_notes_its_words_found():
    from chessapp.brain import AgentResponse

    client, brain, _ = _keyword_app(AgentResponse(text="Fifty moves, no capture."))
    client.post("/api/command", json={"text": "what is the fifty move rule"})
    assert [p["topic"] for p in brain.gathered[0]][:1] == ["Fifty-move rule"]


def test_a_fast_path_move_still_gathers_for_its_narration():
    client, brain, _ = _keyword_app(narrations=("Classic.",))
    client.post("/api/command", json={"text": "e4"})
    assert brain.calls == []
    assert len(brain.narrate_gathered) == 1  # the beat was handed what was found


def test_a_drag_has_no_words_and_gathers_nothing():
    client, brain, _ = _keyword_app(narrations=("ok",))
    client.post("/api/game/move", json={"move": "e4"})
    assert brain.narrate_gathered == [[]]


def test_every_turn_record_says_what_was_gathered():
    from chessapp.brain import AgentResponse

    turns = CollectedTurns()
    client, _, _ = _keyword_app(AgentResponse(text="It's a rule."), tracer=turns)
    client.post("/api/command", json={"text": "how does en passant work"})
    (record,) = [r for r in turns.records if r.get("utterance")]
    assert record["gather"]["source"] == "bm25_fallback"
    assert record["gather"]["passages"][0]["topic"] == "En passant"
    gathered = record["evidence"]["gathered"]
    assert gathered[0]["topic"] == "En passant" and "fifth rank" in gathered[0]["text"]


def test_an_app_with_no_searcher_gathers_nothing_and_says_so():
    from chessapp.brain import AgentResponse

    turns = CollectedTurns()
    ctx = ToolContext(session=GameSession(), engine=FakeEngine())
    app, brain = scripted_app(ctx, AgentResponse(text="hi"), tracer=turns)
    TestClient(app).post("/api/command", json={"text": "how does en passant work"})
    assert brain.gathered == [[]]
    (record,) = [r for r in turns.records if r.get("utterance")]
    assert record["gather"] is None


# --- what each phase reads ---------------------------------------------------------


def _llama_client(*turns):
    ctx = ToolContext(session=GameSession(), engine=FakeEngine("e7e5"))
    coordinator = TurnCoordinator(ctx)
    registry = build_registry(ctx, coordinator, atomic_exchange=False)
    provider = ScriptedProvider(*turns)
    brain = LlamaBrain(
        provider=provider,
        dispatcher=registry,
        tool_definitions=lambda: brain_tool_definitions(registry, ctx),
        system_prompt="You are a chess opponent.",
        planner_prompt="Pick the tools.",
        narrator_facts=lambda: api.narrator_facts(ctx, coordinator),
        settle_reply=coordinator.settle_owed_reply,
    )
    app = api.create_app(
        ctx,
        brain=brain,
        registry=registry,
        coordinator=coordinator,
        searcher=gather.Searcher(INDEX),
    )
    return TestClient(app), provider


def test_the_notes_sit_between_the_board_and_the_command():
    client, provider = _llama_client(text_turn(""), text_turn("It's a rule."))
    client.post("/api/command", json={"text": "how does en passant work"})
    planner = provider.calls[0]
    system, user = planner["messages"]
    assert system["content"] == "Pick the tools."
    board = user["content"].index("Board state:")
    notes = user["content"].index(f"{PLANNER_NOTES_LABEL}:\n- En passant: ")
    command = user["content"].index("Command: how does en passant work")
    assert board < notes < command
    assert LOOKUP not in {t["function"]["name"] for t in planner["tools"]}


def test_no_notes_no_section():
    client, provider = _llama_client(text_turn(""), text_turn("Sure."))
    client.post("/api/command", json={"text": "hello there"})
    assert PLANNER_NOTES_LABEL not in provider.calls[0]["messages"][1]["content"]


def test_the_narrator_answers_from_the_notes_on_a_reply_turn():
    client, provider = _llama_client(text_turn(""), text_turn("It's a rule."))
    body = client.post("/api/command", json={"text": "how does en passant work"})
    brief = provider.calls[1]["messages"][-1]["content"]
    assert f"{NARRATOR_NOTES_LABEL}:\n- En passant: " in brief
    # Notes are not results: the turn did nothing and called no tool.
    assert "No tool was called this turn." in brief
    assert "Done this turn: nothing." in brief
    assert body.json()["tool_results"] == []


def test_the_planner_offer_lacks_lookup_and_the_registry_keeps_it():
    ctx = ToolContext(session=GameSession())
    registry = build_registry(ctx, TurnCoordinator(ctx), atomic_exchange=False)
    offered = {t["function"]["name"] for t in brain_tool_definitions(registry, ctx)}
    assert LOOKUP not in offered
    assert LOOKUP in {t["function"]["name"] for t in registry.definitions()}
