"""The brain setting (#434): the player picks Glitch's model from a code-owned
list, and the switch lands at the next turn boundary — never inside a turn.

Each model gets its own scripted provider through `build_app`'s `provider_for`
seam, so which provider recorded a call is which brain served it.
"""

import json
import threading
from pathlib import Path

from fastapi.testclient import TestClient

from chessapp.app import DEFAULT_MODEL, build_app
from chessapp.profiles import BRAIN_CHOICES, load_profile
from chessapp.tools import SETTINGS_FILENAME
from fakes import (
    CollectedTurns,
    FakeEngine,
    ScriptedBrain,
    ScriptedProvider,
    text_turn,
)

SMALL = "gemma-4-12b"
BIG = "gemma-4-26b-a4b"
FIXTURES = Path(__file__).parent / "fixtures"
# The 12B's crutch text in `undo`'s description; the 26B's profile drops it
# (#432), so the offered tools say which planner profile a turn ran under.
UNDO_CRUTCH = "call this again for each further named move"


def _undo_text(call: dict) -> str:
    text = next(
        t["function"]["description"]
        for t in call["tools"]
        if t["function"]["name"] == "undo"
    )
    return " ".join(text.split())


def _app(providers: dict, **kwargs):
    return build_app(
        model=BIG,
        engine=FakeEngine(),
        provider_for=lambda model: providers[model],
        **kwargs,
    )


def _providers() -> dict[str, ScriptedProvider]:
    return {model: ScriptedProvider(text_turn("Hello.")) for model in BRAIN_CHOICES}


# --- the choices -------------------------------------------------------------


def test_every_choice_has_a_profile_and_pinned_request_bytes():
    assert DEFAULT_MODEL in BRAIN_CHOICES
    for model in BRAIN_CHOICES:
        assert load_profile(model).name == model
        fixture = FIXTURES / f"{model.replace('-', '_')}_requests.json"
        assert fixture.is_file(), f"{model} has no byte pin"


def test_an_unknown_brain_is_refused_and_nothing_changes():
    client = TestClient(_app(_providers()))
    response = client.post("/api/settings/brain", json={"model": "gpt-9"})
    assert response.status_code == 422
    assert "gemma-4-12b" in response.json()["detail"]
    assert client.get("/api/settings").json()["brain"] == BIG


def test_a_fixed_brain_cannot_switch():
    client = TestClient(build_app(brain=ScriptedBrain()))
    response = client.post("/api/settings/brain", json={"model": SMALL})
    assert response.status_code == 409
    assert client.get("/api/settings").json()["brain_choices"] == []


# --- switching at the boundary -----------------------------------------------


def test_a_switch_lands_at_the_next_turn_with_its_profiles_crutches():
    providers = _providers()
    client = TestClient(_app(providers))
    client.post("/api/command", json={"text": "how am I doing?"})
    assert providers[BIG].calls and not providers[SMALL].calls
    assert UNDO_CRUTCH not in _undo_text(providers[BIG].calls[0])

    body = client.post("/api/settings/brain", json={"model": SMALL}).json()
    # Chosen but not yet serving: the UI can say a slow first turn is coming.
    assert body["brain"] == SMALL
    assert body["brain_serving"] == BIG
    assert body["brain_cold"] is True

    providers[BIG].calls.clear()
    client.post("/api/command", json={"text": "how am I doing?"})
    assert not providers[BIG].calls
    assert UNDO_CRUTCH in _undo_text(providers[SMALL].calls[0])
    settings = client.get("/api/settings").json()
    assert settings["brain_serving"] == SMALL
    assert settings["brain_cold"] is False, "its first turn is done"


def test_the_first_turn_on_a_new_brain_is_reported_cold_until_it_ends():
    providers = _providers()
    client = TestClient(_app(providers))
    client.post("/api/settings/brain", json={"model": SMALL})
    # A board drag is a boundary too, and its narration is the new brain's.
    client.post("/api/game/move", json={"move": "e4"})
    assert providers[SMALL].calls and not providers[BIG].calls
    assert client.get("/api/settings").json()["brain_cold"] is False


class _BlockingPlanner(ScriptedProvider):
    """A provider whose first planner call waits to be released, so a switch
    can be asked for while the turn is provably still running."""

    def __init__(self, *turns) -> None:
        super().__init__(*turns)
        self.entered = threading.Event()
        self.release = threading.Event()

    def chat(self, messages, *, tools=None, **kwargs):
        if tools is not None and not self.entered.is_set():
            self.entered.set()
            assert self.release.wait(timeout=10)
        return super().chat(messages, tools=tools, **kwargs)


def test_a_switch_asked_for_mid_turn_waits_for_that_turn_to_end():
    providers = _providers()
    providers[BIG] = _BlockingPlanner(text_turn("Thinking done."))
    client = TestClient(_app(providers))
    done = threading.Event()

    def turn() -> None:
        client.post("/api/command", json={"text": "what's the plan?"})
        done.set()

    worker = threading.Thread(target=turn)
    worker.start()
    assert providers[BIG].entered.wait(timeout=10)
    # The setting answers at once: it takes no lock the turn holds.
    response = client.post("/api/settings/brain", json={"model": SMALL})
    assert response.status_code == 200
    providers[BIG].release.set()
    worker.join(timeout=10)
    assert done.is_set()

    # The running turn planned *and* narrated on the brain it started with.
    assert len(providers[BIG].calls) >= 2
    assert providers[BIG].calls[-1]["tools"] is None, "the narrator, on the 26B"
    assert not providers[SMALL].calls

    client.post("/api/command", json={"text": "and now?"})
    assert providers[SMALL].calls


def test_a_pending_question_is_answered_through_the_new_brain():
    """Game truth lives in code: a question the old brain's turn armed is read
    and run by the new one."""
    providers = _providers()
    client = TestClient(_app(providers))
    # A game worth asking about: resigning on move one needs no question.
    for move in ("e4", "Nf3"):
        assert client.post("/api/game/move", json={"move": move}).status_code == 200
    body = client.post("/api/command", json={"text": "i give up"}).json()
    assert body["tool_results"][0]["result"]["ok"] is False, "asked, not run"
    providers[SMALL].rescript(text_turn("confirm"), text_turn("Good game."))
    assert client.get("/api/state").json()["game_over"] is False

    client.post("/api/settings/brain", json={"model": SMALL})
    body = client.post("/api/command", json={"text": "go on then, do it"}).json()

    assert body["state"]["game_over"] is True
    assert "Their reply" in str(providers[SMALL].calls[0]["messages"])


# --- persistence and modes ---------------------------------------------------


def test_the_choice_persists_and_serves_from_the_start(tmp_path):
    providers = _providers()
    client = TestClient(_app(providers, save_dir=tmp_path))
    client.post("/api/settings/brain", json={"model": SMALL})
    assert json.loads((tmp_path / SETTINGS_FILENAME).read_text())["brain"] == SMALL

    providers = _providers()
    client = TestClient(_app(providers, save_dir=tmp_path))
    settings = client.get("/api/settings").json()
    assert settings["brain"] == settings["brain_serving"] == SMALL
    client.post("/api/command", json={"text": "hello"})
    assert providers[SMALL].calls and not providers[BIG].calls
    assert UNDO_CRUTCH in _undo_text(providers[SMALL].calls[0])


def test_a_brain_no_longer_on_the_list_falls_back_to_the_default(tmp_path):
    (tmp_path / SETTINGS_FILENAME).write_text(json.dumps({"brain": "retired-9b"}))
    providers = _providers()
    client = TestClient(_app(providers, save_dir=tmp_path))
    settings = client.get("/api/settings").json()
    assert settings["brain"] == settings["brain_serving"] == BIG


def test_direct_mode_stores_the_choice_and_still_plays():
    client = TestClient(build_app(agent_enabled=False, engine=FakeEngine()))
    response = client.post("/api/settings/brain", json={"model": SMALL})
    assert response.status_code == 200
    assert client.get("/api/settings").json()["brain"] == SMALL
    assert client.post("/api/game/move", json={"move": "e4"}).status_code == 200


# --- per-turn attribution ----------------------------------------------------


def test_every_turn_record_names_the_brain_that_served_it():
    tracer = CollectedTurns()
    client = TestClient(_app(_providers(), tracer=tracer))
    client.post("/api/command", json={"text": "hello"})
    client.post("/api/settings/brain", json={"model": SMALL})
    client.post("/api/command", json={"text": "hello again"})

    turns = [r for r in tracer.records if r.get("kind") == "turn"]
    manifests = [r for r in tracer.records if r.get("kind") == "serving"]
    first, second = turns[0]["serving"], turns[1]["serving"]
    assert (first["model"], first["profile"]) == (BIG, BIG)
    assert (second["model"], second["profile"]) == (SMALL, SMALL)
    assert first["manifest_id"] != second["manifest_id"]
    assert first["session"] == second["session"], "one process, one session"
    # A manifest for each brain as it started serving.
    assert [m["client"]["model"] for m in manifests] == [BIG, SMALL]
