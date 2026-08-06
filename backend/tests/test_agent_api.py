"""Delegate REST API: conversation persistence + the messages round trip.

The workspace delegate contract (`../agent-standard/delegate-api.md`) so a
conductor agent can drive chess over HTTP. Mirrors PCC's `test_agent_api.py`
against chess's fixtures — the `ScriptedBrain` double (never a live LLM) and
the shared command pipeline extracted from `/api/command`. The conversation
store is in-memory (chess's documented divergence from PCC's SQLite), so a
fresh `create_app` gives each test a fresh store; the per-IP rate limiter is
module-global, so it is reset around every test.
"""

import asyncio
import logging

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from chessapp.agent_api import (
    LOOP_ACTOR,
    ConversationStore,
    build_agent_router,
    reset_rate_limit,
)
from chessapp.api import UNVERIFIED_CLAIM_REPLY, CommandOutcome
from chessapp.brain import AgentResponse, ToolCall
from chessapp.conversation import RECENT_TURNS
from chessapp.game import GameSession
from chessapp.provider import ProviderRequestError
from chessapp.tools import ToolContext
from fakes import ScriptedBrain, receive_state, scripted_app


@pytest.fixture(autouse=True)
def _reset_rate_limit():
    reset_rate_limit()
    yield
    reset_rate_limit()


def make_client(*responses, narrations=(), brain=None, verbosity="normal"):
    ctx = ToolContext(session=GameSession())
    ctx.settings.verbosity = verbosity
    if brain is None:
        brain = ScriptedBrain(*responses, narrations=narrations)
    app, brain = scripted_app(ctx, brain=brain)
    return TestClient(app), brain, ctx


def move(san, text="on it"):
    return AgentResponse(
        text=text, tool_calls=(ToolCall(name="make_move", args={"move": san}),)
    )


def new_conversation(client, **body):
    return client.post("/api/agent/conversations", json=body).json()["id"]


def send(client, conversation_id, content, **kwargs):
    return client.post(
        f"/api/agent/conversations/{conversation_id}/messages",
        json={"content": content},
        **kwargs,
    )


# --- CRUD ---------------------------------------------------------------------


def test_create_conversation_is_201_with_null_title():
    client, _, _ = make_client()
    created = client.post("/api/agent/conversations", json={})
    assert created.status_code == 201
    body = created.json()
    assert body["title"] is None
    assert isinstance(body["id"], int)
    assert "created_at" in body and "updated_at" in body


def test_create_conversation_keeps_a_given_title():
    client, _, _ = make_client()
    body = client.post(
        "/api/agent/conversations", json={"title": "Weekly triage"}
    ).json()
    assert body["title"] == "Weekly triage"


def test_list_conversations_most_recent_first():
    client, _, _ = make_client()
    first = new_conversation(client)
    second = new_conversation(client)
    listed = client.get("/api/agent/conversations").json()
    assert [c["id"] for c in listed] == [second, first]


def test_detail_of_a_fresh_conversation_has_no_messages():
    client, _, _ = make_client()
    conversation_id = new_conversation(client, title="Endgame drill")
    detail = client.get(f"/api/agent/conversations/{conversation_id}")
    assert detail.status_code == 200
    body = detail.json()
    assert body["title"] == "Endgame drill"
    assert body["messages"] == []


def test_missing_and_soft_deleted_threads_404_across_endpoints():
    """A never-existed id and a soft-deleted thread must 404 identically on
    GET, POST-messages, and DELETE — the rule conductor's recreate-once retry
    depends on (a pruned thread is indistinguishable from one that never was)."""
    client, _, _ = make_client()

    missing = 999_999
    assert client.get(f"/api/agent/conversations/{missing}").status_code == 404
    assert send(client, missing, "hi").status_code == 404
    assert client.delete(f"/api/agent/conversations/{missing}").status_code == 404

    conversation_id = new_conversation(client)
    assert (
        client.delete(f"/api/agent/conversations/{conversation_id}").status_code == 204
    )
    assert client.get(f"/api/agent/conversations/{conversation_id}").status_code == 404
    assert send(client, conversation_id, "hi").status_code == 404
    assert (
        client.delete(f"/api/agent/conversations/{conversation_id}").status_code == 404
    )


# --- message flow -------------------------------------------------------------


def test_post_message_runs_pipeline_and_persists_the_exchange():
    client, _, ctx = make_client(move("e4", text="Pawn to e4. Your move."))
    conversation_id = new_conversation(client)

    response = send(client, conversation_id, "open with the king's pawn")
    assert response.status_code == 200
    exchange = response.json()

    user = exchange["user_message"]
    assert user["role"] == "user"
    assert user["content"] == "open with the king's pawn"
    assert user["tool_calls"] is None
    assert user["stop_reason"] is None

    assistant = exchange["assistant_message"]
    assert assistant["role"] == "assistant"
    assert assistant["content"] == "Pawn to e4. Your move."
    assert assistant["stop_reason"] == "completed"
    calls = assistant["tool_calls"]
    assert [c["tool"] for c in calls] == ["make_move"]
    assert calls[0]["arguments"] == {"move": "e4"}
    assert calls[0]["result"] is not None and calls[0]["error"] is None

    # The move landed on the single shared game session.
    assert ctx.session.move_history() == ["e4"]

    # History survives a "reload" as an ordered thread.
    detail = client.get(f"/api/agent/conversations/{conversation_id}").json()
    assert detail["title"] == "open with the king's pawn"
    assert [m["role"] for m in detail["messages"]] == ["user", "assistant"]


def test_delegate_brain_is_told_which_saved_games_exist(tmp_path):
    """Both seams build the brain's view from the same `_agent_state_dict`, and
    they must not drift: the delegate wire is handed the saves on disk too."""
    session = GameSession()
    session.save(tmp_path / "scholars.json")
    ctx = ToolContext(session=GameSession(), save_dir=tmp_path)
    app, brain = scripted_app(ctx, AgentResponse(text="hi"))
    client = TestClient(app)

    send(client, new_conversation(client), "load up the game I saved as scholars")

    assert brain.calls[0][0]["saved_games"] == ["scholars"]


def test_tool_calls_is_null_when_no_tools_ran():
    client, _, _ = make_client(AgentResponse(text="Looks balanced to me."))
    conversation_id = new_conversation(client)
    assistant = send(client, conversation_id, "how does the board look?").json()[
        "assistant_message"
    ]
    assert assistant["content"] == "Looks balanced to me."
    assert assistant["tool_calls"] is None
    assert assistant["stop_reason"] == "completed"


def test_ok_false_result_maps_to_error_not_result():
    client, _, _ = make_client(
        AgentResponse(
            text="I can't do that.",
            tool_calls=(ToolCall(name="launch_rocket", args={}),),
        )
    )
    conversation_id = new_conversation(client)
    assistant = send(client, conversation_id, "do a barrel roll").json()[
        "assistant_message"
    ]
    call = assistant["tool_calls"][0]
    assert call["tool"] == "launch_rocket"
    assert call["result"] is None
    assert "unknown tool" in call["error"]
    # The loop read the error, gave up on the tool, and answered in words.
    assert assistant["content"] == "I can't do that."
    assert assistant["stop_reason"] == "completed"


def test_rejected_move_is_a_result_and_a_budget_stop_is_reported():
    """A `legal: false` move rejection is a legitimate domain outcome — it
    rides on `result`, never `error`. And when the brain's loop gives up on its
    own budget, that stop reason reaches the delegate wire verbatim."""
    client, _, ctx = make_client(
        AgentResponse(
            text="",
            tool_calls=tuple(
                ToolCall(name="make_move", args={"move": "Nf6"}) for _ in range(3)
            ),
            stop_reason="max_iterations",
        )
    )
    conversation_id = new_conversation(client)
    assistant = send(client, conversation_id, "knight to f6").json()[
        "assistant_message"
    ]

    calls = assistant["tool_calls"]
    assert len(calls) == 3
    assert all(c["result"] is not None and c["error"] is None for c in calls)
    assert assistant["content"]  # a budget stop still says something
    assert assistant["stop_reason"] == "max_iterations"
    assert ctx.session.move_history() == []


def test_title_is_derived_from_the_first_user_message():
    client, _, _ = make_client(AgentResponse(text="ok"))
    conversation_id = new_conversation(client)
    assert (
        client.get(f"/api/agent/conversations/{conversation_id}").json()["title"]
        is None
    )
    send(client, conversation_id, "what's my best plan here")
    assert (
        client.get(f"/api/agent/conversations/{conversation_id}").json()["title"]
        == "what's my best plan here"
    )


def test_history_replays_text_turns_only_no_tool_payloads():
    """The second message's brain call sees only prior *text* turns — the
    commentary the caller saw, never the persisted tool trajectory."""
    brain = ScriptedBrain(
        AgentResponse(  # turn 1: the loop played e4 and commented
            text="Pawn to e4.",
            tool_calls=(ToolCall(name="make_move", args={"move": "e4"}),),
        ),
        AgentResponse(text="I opened with e4."),  # turn 2
    )
    client, _, _ = make_client(brain=brain)
    conversation_id = new_conversation(client)

    send(client, conversation_id, "start with the king's pawn")
    send(client, conversation_id, "what did you play?")

    replayed = brain.transcripts[-1]
    assert replayed == [
        {"role": "user", "content": "start with the king's pawn"},
        {"role": "assistant", "content": "Pawn to e4."},
    ]
    # Text turns only: no tool payloads smuggled into model context.
    assert all(set(turn) == {"role", "content"} for turn in replayed)


def test_the_caller_sees_the_correction_but_the_loop_never_replays_it():
    """The delegate store is one field doing two jobs — the wire record and the
    loop's memory — and a guarded turn needs them to differ. The caller is told
    the claim was pulled; the brain is given the facts, because a canned
    first-person correction replayed as its own words is a register it imitates
    (`api._remembered_facts`)."""
    brain = ScriptedBrain(
        AgentResponse(  # invents a capture on an opening move
            text="Took your knight. Easy.",
            tool_calls=(ToolCall(name="make_move", args={"move": "e4"}),),
        ),
        AgentResponse(text="Nothing much."),
    )
    client, _, _ = make_client(brain=brain)
    conversation_id = new_conversation(client)

    body = send(client, conversation_id, "start with the king's pawn").json()
    send(client, conversation_id, "what did you play?")

    assert body["assistant_message"]["content"] == UNVERIFIED_CLAIM_REPLY
    assert brain.transcripts[-1] == [
        {"role": "user", "content": "start with the king's pawn"},
        {"role": "assistant", "content": "e4."},
    ]


def test_the_delegate_wire_condenses_older_turns_like_the_panel_does():
    """One memory policy, not two: the delegate's replay is `condense`d off its
    own store, so a long conversation reaches the brain as recent turns behind a
    digest — same as `/api/command` (`docs/turn-memory.md`)."""
    turns = RECENT_TURNS + 2
    brain = ScriptedBrain(*[AgentResponse(text=f"reply {i}") for i in range(turns + 1)])
    client, _, _ = make_client(brain=brain)
    conversation_id = new_conversation(client)

    send(client, conversation_id, "tell me about the Sicilian")
    for i in range(RECENT_TURNS):
        send(client, conversation_id, f"and what about line {i}")
    send(client, conversation_id, "so what should I study?")

    replayed = brain.transcripts[-1]
    assert len(replayed) == 2 + 2 * RECENT_TURNS
    assert '"tell me about the Sicilian"' in replayed[0]["content"]
    assert replayed[-2:] == [
        {"role": "user", "content": f"and what about line {RECENT_TURNS - 1}"},
        {"role": "assistant", "content": f"reply {RECENT_TURNS}"},
    ]


def test_fast_path_move_skips_the_brain_loop():
    # A ScriptedBrain with no scripted responses raises if the loop is consulted.
    client, brain, ctx = make_client(narrations=("Classic opener.",))
    conversation_id = new_conversation(client)
    assistant = send(client, conversation_id, "e4").json()["assistant_message"]

    assert brain.calls == []  # the loop never consulted
    assert assistant["content"] == "Classic opener."
    assert assistant["tool_calls"][0]["tool"] == "make_move"
    assert ctx.session.move_history() == ["e4"]


def test_delegate_move_broadcasts_to_the_web_board():
    client, _, _ = make_client(narrations=("ok",))
    conversation_id = new_conversation(client)
    with client.websocket_connect("/ws") as ws:
        receive_state(ws)  # connect snapshot
        send(client, conversation_id, "e4")
        message = receive_state(ws)
    assert message["state"]["history"] == ["e4"]


def test_provider_failure_is_502_and_keeps_the_user_message():
    class BoomBrain:
        def get_agent_response(self, board_state, command, transcript=()):
            raise ProviderRequestError("connect timeout")

        def react(self, board_state, changes, transcript=()):  # pragma: no cover
            raise AssertionError("react should not run")

    client, _, _ = make_client(brain=BoomBrain())
    conversation_id = new_conversation(client)

    response = send(client, conversation_id, "how am I doing?")
    assert response.status_code == 502

    detail = client.get(f"/api/agent/conversations/{conversation_id}").json()
    assert [m["role"] for m in detail["messages"]] == ["user"]
    assert detail["messages"][0]["content"] == "how am I doing?"


@pytest.mark.parametrize(
    "content", ["", "   ", "x" * 8001], ids=["empty", "whitespace", "too-long"]
)
def test_message_content_validation_is_422(content):
    client, _, _ = make_client()
    conversation_id = new_conversation(client)
    assert send(client, conversation_id, content).status_code == 422


def test_missing_content_field_is_422():
    client, _, _ = make_client()
    conversation_id = new_conversation(client)
    assert (
        client.post(
            f"/api/agent/conversations/{conversation_id}/messages", json={}
        ).status_code
        == 422
    )


def test_post_message_is_rate_limited(monkeypatch):
    monkeypatch.setenv("CHESSAPP_AGENT_MESSAGES_PER_MIN", "1")
    client, _, _ = make_client(narrations=("ok", "ok"))
    conversation_id = new_conversation(client)

    assert send(client, conversation_id, "e4").status_code == 200
    second = send(client, conversation_id, "e5")
    assert second.status_code == 429
    assert "Retry-After" in second.headers


def test_recognized_actor_binds_the_conductor(caplog):
    client, _, _ = make_client(narrations=("ok",))
    conversation_id = new_conversation(client)
    with caplog.at_level(logging.INFO, logger="chessapp.agent_api"):
        send(
            client,
            conversation_id,
            "e4",
            headers={"X-Agent-Actor": "agent:conductor"},
        )
    assert any("actor=agent:conductor" in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize(
    "headers",
    [{}, {"X-Agent-Actor": "agent:bogus"}],
    ids=["absent", "unrecognized"],
)
def test_actor_falls_back_to_the_default_when_absent_or_unrecognized(caplog, headers):
    client, _, _ = make_client(narrations=("ok",))
    conversation_id = new_conversation(client)
    with caplog.at_level(logging.INFO, logger="chessapp.agent_api"):
        send(client, conversation_id, "e4", headers=headers)
    assert any(f"actor={LOOP_ACTOR}" in r.getMessage() for r in caplog.records)


# --- concurrency: one exchange at a time per thread ---------------------------
#
# These drive the router over `httpx.ASGITransport` rather than `TestClient`,
# because they need two requests genuinely in flight in *one* event loop:
# `TestClient` is synchronous and spins up a fresh loop per request, so it
# cannot express an overlap at all. The pipeline is stubbed at the
# `run_command` seam (the router's only collaborator), so what is under test is
# the endpoint's own sequencing — not the brain, the board, or the engine.


class HeldPipeline:
    """A `run_command` double that parks the one run it is told to hold.

    Records `(text, transcript)` per call, so a test can assert both *when* a
    run started and what history it was replayed.
    """

    def __init__(self, hold: str) -> None:
        self._hold = hold
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.calls: list[tuple[str, list[dict[str, str]]]] = []

    async def __call__(self, text, transcript, version=None) -> CommandOutcome:
        self.calls.append((text, list(transcript)))
        if text == self._hold:
            self.entered.set()
            await self.release.wait()
        return CommandOutcome(
            commentary=f"answer {text}",
            tool_results=[],
            tool_args=[],
            state={},
            changed=False,
            stop_reason="completed",
            # An ordinary turn Glitch spoke himself, so it is remembered by his
            # own words. Left at the default `""` this would be a *substituted*
            # turn — one the app spoke in his place — and the store would
            # rightly replay the inert ack instead of the answer.
            memory=f"answer {text}",
        )

    @property
    def started(self) -> list[str]:
        return [text for text, _ in self.calls]


def delegate_app(run_command):
    """The delegate router alone, over a fresh store. Returns `(app, store)`."""
    store = ConversationStore()
    app = FastAPI()
    app.include_router(build_agent_router(store=store, run_command=run_command))
    return app, store


def async_client(app):
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://testserver"
    )


async def start_conversation(client):
    return (await client.post("/api/agent/conversations", json={})).json()["id"]


def post_message(client, conversation_id, content):
    return client.post(
        f"/api/agent/conversations/{conversation_id}/messages",
        json={"content": content},
    )


async def let_the_queued_request_run() -> None:
    """Give a just-posted request every chance to reach the endpoint and block.

    The overlap has to be real for the in-flight assertions below to mean
    anything. `ASGITransport` runs the app in this same loop, but FastAPI
    resolves the sync rate-limit dependency in a threadpool, so yielding the
    loop once is not enough to guarantee the hop. Erring short only makes a
    test prove less — never fail — so the wait is generous rather than tuned.
    """
    await asyncio.sleep(0.1)


async def test_concurrent_posts_to_one_conversation_are_serialized_exchanges():
    """Two posts to one thread are two exchanges, not four interleaved turns.

    The failure this pins (#221): with the history read, the user append, the
    run and the assistant append unserialized, both user turns were committed
    before either assistant turn — so the stored thread stopped alternating,
    and the second run was handed a transcript ending in the first, still
    unanswered question. Every later `history_for_loop` then replayed that
    malformed order to the model.
    """
    pipeline = HeldPipeline("first")
    app, store = delegate_app(pipeline)

    async with async_client(app) as client:
        conversation_id = await start_conversation(client)

        first = asyncio.create_task(post_message(client, conversation_id, "first"))
        await asyncio.wait_for(pipeline.entered.wait(), 2)

        second = asyncio.create_task(post_message(client, conversation_id, "second"))
        await let_the_queued_request_run()

        # Mid-hold: the queued message has neither committed its user turn nor
        # started its run.
        stored = store.get(conversation_id)
        assert [m.role for m in stored.messages] == ["user"]
        assert pipeline.started == ["first"]

        pipeline.release.set()
        first_response, second_response = await asyncio.gather(first, second)

    assert first_response.status_code == 200
    assert second_response.status_code == 200

    # The thread alternates, and each answer sits behind its own question.
    assert [(m.role, m.content) for m in stored.messages] == [
        ("user", "first"),
        ("assistant", "answer first"),
        ("user", "second"),
        ("assistant", "answer second"),
    ]

    # And the second run reasoned from the *finished* first exchange.
    assert pipeline.started == ["first", "second"]
    assert pipeline.calls[1][1] == [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "answer first"},
    ]


async def test_a_held_exchange_does_not_block_a_different_conversation():
    """Serialization is per thread, never a global funnel: one conversation
    waiting on a slow model must not stall an unrelated one."""
    pipeline = HeldPipeline("hold me")
    app, store = delegate_app(pipeline)

    async with async_client(app) as client:
        held_thread = await start_conversation(client)
        other_thread = await start_conversation(client)

        held = asyncio.create_task(post_message(client, held_thread, "hold me"))
        await asyncio.wait_for(pipeline.entered.wait(), 2)

        # Runs to completion while the other thread's run is parked, or hangs.
        other = await asyncio.wait_for(post_message(client, other_thread, "quick"), 2)
        assert other.status_code == 200
        assert other.json()["assistant_message"]["content"] == "answer quick"

        pipeline.release.set()
        assert (await held).status_code == 200

    assert [m.role for m in store.get(other_thread).messages] == ["user", "assistant"]
    assert [m.role for m in store.get(held_thread).messages] == ["user", "assistant"]


async def test_a_thread_deleted_while_a_message_waits_404s():
    """A queued message whose thread is deleted while it waits its turn must
    404 like any other unknown thread — never a `KeyError`, and never an append
    to a soft-deleted ghost. So existence is re-checked *after* the wait, not
    only before it."""
    pipeline = HeldPipeline("first")
    app, _ = delegate_app(pipeline)

    async with async_client(app) as client:
        conversation_id = await start_conversation(client)

        first = asyncio.create_task(post_message(client, conversation_id, "first"))
        await asyncio.wait_for(pipeline.entered.wait(), 2)

        second = asyncio.create_task(post_message(client, conversation_id, "second"))
        await let_the_queued_request_run()

        deleted = await client.delete(f"/api/agent/conversations/{conversation_id}")
        assert deleted.status_code == 204

        pipeline.release.set()
        # The exchange already in flight finishes on the thread it holds; the
        # queued one finds it gone and never runs.
        assert (await first).status_code == 200
        assert (await second).status_code == 404

    assert pipeline.started == ["first"]
