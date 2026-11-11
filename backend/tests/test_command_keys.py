"""The panel command route's Idempotency-Key (#553).

The page sends one key per conductor handoff and the same key on every retry
of it (after its 30 s timeout, a Retry tap, or a reload). The backend answers a
repeated key from the first run and never runs the words twice, including when
the retry arrives while the first run is still going.
"""

import asyncio
import json
import threading
import time

from fastapi.testclient import TestClient

from chessapp.api import create_app
from chessapp.brain import AgentResponse
from chessapp.command_keys import COMMAND_KEYS_FILENAME, CommandKeys
from chessapp.game import GameSession
from chessapp.tools import ToolContext
from fakes import FakeEngine, ScriptedBrain, scripted_app
from test_command import BlockingBrain, QueueingLock, provider_died

KEY = "intent-0123456789abcdef"


def keyed(key: str = KEY) -> dict[str, str]:
    return {"Idempotency-Key": key}


def handoff(text: str, **extra) -> dict:
    return {"text": text, "origin": "handoff", **extra}


def game_client(*responses: AgentResponse, save_dir=None, **client_kwargs):
    """A panel with an engine that answers e4 with e5, so a fast-path "e4" is a
    real move the board would show twice if it ran twice."""
    ctx = ToolContext(session=GameSession(), engine=FakeEngine("e7e5"))
    if save_dir is not None:
        ctx.save_dir = save_dir
    brain = ScriptedBrain(*responses)
    app, _ = scripted_app(ctx, brain=brain)
    return TestClient(app, **client_kwargs), brain, ctx


# --- the route -----------------------------------------------------------------


def test_a_retry_after_success_replays_the_first_answer_and_runs_nothing():
    client, _, ctx = game_client()
    first = client.post("/api/command", json=handoff("e4", version=0), headers=keyed())
    assert first.status_code == 200
    assert ctx.session.move_history() == ["e4", "e5"]

    # The page timed out and retries with the version it sent the first time.
    # Without the key this is a stale 409 at best, and a second move at worst.
    again = client.post("/api/command", json=handoff("e4", version=0), headers=keyed())

    assert again.status_code == 200
    body = again.json()
    assert ctx.session.move_history() == ["e4", "e5"], "the move ran once"
    assert body["replayed"] is True
    assert first.json()["replayed"] is False
    for field in ("commentary", "tool_results", "correlation_id"):
        assert body[field] == first.json()[field], field
    assert body["retryable"] is False
    assert body["state"]["history"] == ["e4", "e5"]


def test_a_retry_after_a_reload_is_answered_from_the_first_run():
    """A reload re-reads the board, so the retry carries the new version and the
    same key: still answered, still not run."""
    client, _, ctx = game_client()
    client.post("/api/command", json=handoff("e4"), headers=keyed())
    version = client.get("/api/state").json()["version"]

    again = client.post(
        "/api/command", json=handoff("e4", version=version), headers=keyed()
    )

    assert again.json()["replayed"] is True
    assert ctx.session.move_history() == ["e4", "e5"]


class DoorbellEvent(asyncio.Event):
    """The key's `settled` event, ringing a thread-side bell when someone waits
    on it: the moment a retry has provably parked behind the first run."""

    def __init__(self) -> None:
        super().__init__()
        self.waited = threading.Event()

    async def wait(self) -> bool:
        self.waited.set()
        return await super().wait()


def test_a_retry_during_the_first_run_waits_for_it_and_never_runs_again():
    ctx = ToolContext(session=GameSession())
    ctx.mutation_lock = QueueingLock()
    brain = BlockingBrain(AgentResponse(text="Looks level."))
    app, _ = scripted_app(ctx, brain=brain)
    results: dict[str, object] = {}

    # One client, one event loop: the two requests overlap in the loop the
    # app runs on, as they do under uvicorn.
    with TestClient(app) as client:

        def post(name: str) -> None:
            results[name] = client.post(
                "/api/command", json=handoff("how does it look?"), headers=keyed()
            )

        first = threading.Thread(target=post, args=("first",))
        first.start()
        assert brain.reached.wait(10), "the first run never reached the brain"
        entry = app.state.command_keys.get(KEY)
        assert entry is not None and entry.status == "running"
        entry.settled = DoorbellEvent()

        retry = threading.Thread(target=post, args=("retry",))
        retry.start()
        assert entry.settled.waited.wait(10), "the retry never parked on the key"
        brain.release.set()
        for thread in (first, retry):
            thread.join(10)
            assert not thread.is_alive(), "a command never came back"

    first_body, retry_body = results["first"].json(), results["retry"].json()
    assert results["retry"].status_code == 200
    assert len(brain.calls) == 1, "one run: the retry was answered, not run"
    assert not ctx.mutation_lock.contended.is_set(), "the retry never queued to run"
    assert retry_body["commentary"] == first_body["commentary"] == "Looks level."
    assert retry_body["correlation_id"] == first_body["correlation_id"]
    assert retry_body["replayed"] is True
    assert ctx.transcript.to_dict() == [
        {"role": "user", "content": "how does it look?"},
        {"role": "assistant", "content": "Looks level."},
    ], "recorded once"


def test_a_new_intent_under_a_new_key_runs():
    client, brain, _ = game_client(AgentResponse(text="one"), AgentResponse(text="two"))
    a = client.post(
        "/api/command", json=handoff("how is it?"), headers=keyed("a")
    ).json()
    b = client.post(
        "/api/command", json=handoff("how is it?"), headers=keyed("b")
    ).json()
    assert (a["commentary"], b["commentary"]) == ("one", "two")
    assert b["replayed"] is False
    assert len(brain.calls) == 2


def test_a_plain_panel_command_without_a_key_always_runs():
    client, brain, _ = game_client(AgentResponse(text="one"), AgentResponse(text="two"))
    a = client.post("/api/command", json={"text": "how is it?"}).json()
    b = client.post("/api/command", json={"text": "how is it?"}).json()
    assert (a["commentary"], b["commentary"]) == ("one", "two")
    assert len(brain.calls) == 2


def test_a_key_cannot_replay_another_commands_result():
    client, brain, ctx = game_client(AgentResponse(text="Looks level."))
    client.post("/api/command", json=handoff("how is it?"), headers=keyed())

    other_words = client.post("/api/command", json=handoff("e4"), headers=keyed())
    other_origin = client.post(
        "/api/command", json={"text": "how is it?"}, headers=keyed()
    )

    assert other_words.status_code == other_origin.status_code == 422
    assert ctx.session.move_history() == []
    assert len(brain.calls) == 1


def test_a_stale_first_attempt_ran_nothing_so_the_retry_runs():
    client, _, ctx = game_client()
    stale = client.post("/api/command", json=handoff("e4", version=7), headers=keyed())
    assert stale.status_code == 409
    assert ctx.session.move_history() == []

    fresh = client.post("/api/command", json=handoff("e4", version=0), headers=keyed())

    assert fresh.status_code == 200
    assert fresh.json()["replayed"] is False
    assert ctx.session.move_history() == ["e4", "e5"]


def test_a_retryable_first_attempt_is_run_again_on_retry():
    """The provider died before anything ran (#546): the page keeps the intent
    and retries under the same key, and that retry must reach the model."""
    client, brain, _ = game_client(provider_died(), AgentResponse(text="Back now."))
    first = client.post("/api/command", json=handoff("how is it?"), headers=keyed())
    assert first.json()["retryable"] is True

    again = client.post("/api/command", json=handoff("how is it?"), headers=keyed())

    assert again.json()["commentary"] == "Back now."
    assert again.json()["replayed"] is False
    assert len(brain.calls) == 2


class ExplodingBrain(ScriptedBrain):
    def get_agent_response(self, board_state, command, *, earlier=None, gathered=()):
        self.calls.append((board_state, command))
        raise RuntimeError("a bug, mid-turn")


def test_a_run_that_died_mid_turn_is_not_run_again():
    ctx = ToolContext(session=GameSession())
    app, brain = scripted_app(ctx, brain=ExplodingBrain())
    client = TestClient(app, raise_server_exceptions=False)
    assert (
        client.post(
            "/api/command", json=handoff("how is it?"), headers=keyed()
        ).status_code
        == 500
    )

    again = client.post("/api/command", json=handoff("how is it?"), headers=keyed())

    assert again.status_code == 409
    body = again.json()
    assert body["incomplete"] is True
    assert body["state"]["version"] == ctx.board_version
    assert "stale" not in body, "not a stale board: the client must not resend"
    assert len(brain.calls) == 1


def test_a_finished_key_survives_a_restart(tmp_path):
    client, _, ctx = game_client(save_dir=tmp_path)
    first = client.post("/api/command", json=handoff("e4"), headers=keyed()).json()

    # A new app over the same save dir, and a brain that would fail loudly if
    # anything reached it.
    restarted, _, _ = game_client(save_dir=tmp_path)
    again = restarted.post("/api/command", json=handoff("e4"), headers=keyed())

    assert again.status_code == 200
    assert again.json()["replayed"] is True
    assert again.json()["commentary"] == first["commentary"]


def test_a_run_cut_off_by_a_restart_reads_back_incomplete(tmp_path):
    keys = CommandKeys(tmp_path / COMMAND_KEYS_FILENAME)
    keys.start(KEY, "e4", "handoff")  # and then the process died

    restarted, brain, ctx = game_client(save_dir=tmp_path)
    again = restarted.post("/api/command", json=handoff("e4"), headers=keyed())

    assert again.status_code == 409
    assert again.json()["incomplete"] is True
    assert ctx.session.move_history() == []
    assert brain.calls == []


def test_no_brain_is_still_503_with_a_key():
    client = TestClient(create_app(ToolContext(session=GameSession())))
    response = client.post("/api/command", json=handoff("e4"), headers=keyed())
    assert response.status_code == 503


def test_an_oversized_key_is_refused():
    client, brain, _ = game_client()
    response = client.post("/api/command", json=handoff("e4"), headers=keyed("k" * 129))
    assert response.status_code == 422
    assert brain.calls == []


# --- the store -----------------------------------------------------------------


def reply(commentary: str = "") -> dict:
    return {"commentary": commentary, "tool_results": [], "correlation_id": "c0ffee"}


class Clock:
    def __init__(self) -> None:
        self.now = 1_000_000.0

    def __call__(self) -> float:
        return self.now


def test_keys_expire_after_their_ttl():
    clock = Clock()
    keys = CommandKeys(ttl_seconds=60, clock=clock)
    keys.finish("a", keys.start("a", "e4", "handoff"), reply())
    clock.now += 59
    assert keys.get("a") is not None
    clock.now += 2
    assert keys.get("a") is None


def test_the_store_keeps_at_most_its_cap_oldest_dropped_first():
    keys = CommandKeys(max_keys=2)
    for key in ("a", "b", "c"):
        keys.finish(key, keys.start(key, key, "handoff"), reply(key))
    assert keys.get("a") is None
    assert keys.get("b") is not None and keys.get("c") is not None


def test_a_running_key_is_never_dropped():
    """Dropping it would let its waiting retry start a second run."""
    clock = Clock()
    keys = CommandKeys(max_keys=1, ttl_seconds=60, clock=clock)
    running = keys.start("a", "e4", "handoff")
    keys.finish("b", keys.start("b", "e5", "handoff"), reply())
    clock.now += 3600
    assert keys.get("a") is running
    assert keys.get("b") is None


def test_forget_frees_the_key_and_settles_waiters():
    keys = CommandKeys()
    entry = keys.start("a", "e4", "handoff")
    keys.forget("a", entry)
    assert keys.get("a") is None
    assert entry.settled.is_set()


def test_an_unreadable_or_invalid_file_is_an_empty_store(tmp_path):
    path = tmp_path / COMMAND_KEYS_FILENAME
    for content in ("{not json", json.dumps({"format": 99, "keys": []})):
        path.write_text(content)
        assert CommandKeys(path).get("a") is None
    path.write_text(
        json.dumps(
            {
                "format": 1,
                "keys": [
                    {
                        "key": "a",
                        "text": "e4",
                        "origin": "handoff",
                        "created_at": time.time(),
                        "status": "done",
                        "reply": None,
                    }
                ],
            }
        )
    )
    assert CommandKeys(path).get("a") is None, "a done key with no reply is junk"
    data = json.loads(path.read_text())
    data["keys"][0]["reply"] = {"commentary": "hi"}
    path.write_text(json.dumps(data))
    assert CommandKeys(path).get("a") is None, "nor one missing what a replay sends"


def test_expired_keys_are_dropped_on_load(tmp_path):
    path = tmp_path / COMMAND_KEYS_FILENAME
    clock = Clock()
    keys = CommandKeys(path, ttl_seconds=60, clock=clock)
    keys.finish("a", keys.start("a", "e4", "handoff"), reply())
    assert CommandKeys(path, ttl_seconds=60, clock=clock).get("a") is not None
    clock.now += 120
    assert CommandKeys(path, ttl_seconds=60, clock=clock).get("a") is None


def test_a_failed_write_costs_a_warning_not_the_command(tmp_path, caplog):
    blocker = tmp_path / "file"
    blocker.write_text("")
    keys = CommandKeys(blocker / COMMAND_KEYS_FILENAME)
    entry = keys.start("a", "e4", "handoff")
    assert keys.get("a") is entry
    assert "could not persist command keys" in caplog.text
