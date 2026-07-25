"""The board-version precondition on mutations (Sprint 2, slice 2 — audit 7).

One shared game session, several clients: the web board, the delegate API a
conductor drives, and an MCP session. Until now none of them carried any notion
of *which* board they were acting on, so two clients that had both read the
position could each play a move into it and the second one would land on a board
it had never seen. The audit's item 7, and its E2E acceptance test — "concurrent
clients cannot advance the same turn twice".

The fix is two halves, and both are code's rather than the model's:

- a **monotonic board version** on the shared context, bumped by every board
  mutation and published in the state document, and
- an **optional `version` precondition** on every mutating request, checked
  under the context's mutation lock so the check and the mutation cannot be
  separated by another client's turn.

What is deliberately *not* here: any version in the brain's view of the world or
in a tool schema. Versions are transport bookkeeping between the app and its
clients; the model is never asked to know one (and the schemas the brain is
offered are frozen by the eval floor besides).
"""

import threading
import time

import pytest
from fastapi.testclient import TestClient

from chessapp.api import _agent_state_dict, create_app
from chessapp.brain import AgentResponse, ToolCall
from chessapp.coordinator import TurnCoordinator, TurnPhase
from chessapp.game import GameSession
from chessapp.mcp_server import build_mcp_server
from chessapp.tools import ToolContext, build_registry
from fakes import FakeEngine, scripted_app

# --- helpers ----------------------------------------------------------------


@pytest.fixture
def ctx():
    return ToolContext(session=GameSession())


@pytest.fixture
def client(ctx):
    """Direct mode (no brain): the endpoints answer their own document plus the
    one new key, so the precondition is pinned where nothing else moves."""
    return TestClient(create_app(ctx))


def engine_client(engine=None):
    ctx = ToolContext(session=GameSession(), engine=engine or FakeEngine())
    return TestClient(create_app(ctx)), ctx


def version_of(client: TestClient) -> int:
    return client.get("/api/state").json()["version"]


def developed(ctx: ToolContext) -> None:
    """A game in progress — enough player investment for the destructive gate."""
    for san in ("e4", "e5", "Nf3", "Nc6"):
        ctx.session.submit_move(san)


# --- the version itself -----------------------------------------------------


def test_state_carries_a_board_version(client):
    body = client.get("/api/state").json()
    assert isinstance(body["version"], int)


def test_reads_never_move_the_version(client):
    before = version_of(client)
    client.get("/api/state")
    client.get("/api/game/pgn")
    assert version_of(client) == before


def test_every_move_that_lands_bumps_the_version(client):
    """One bump per mutation, not per request: engine-free a move is one."""
    before = version_of(client)
    client.post("/api/game/move", json={"move": "e4"})
    assert version_of(client) == before + 1


def test_the_engines_reply_bumps_it_too(ctx):
    """The reply is a board mutation like any other, so a full exchange moves the
    version by two — a client that read N before its move never sees N+1."""
    client, _ = engine_client()
    before = version_of(client)
    client.post("/api/game/move", json={"move": "e4"})
    assert version_of(client) == before + 2


def test_an_illegal_move_leaves_the_version_alone(client):
    before = version_of(client)
    body = client.post("/api/game/move", json={"move": "e5"}).json()
    assert body["legal"] is False
    assert version_of(client) == before


def test_undo_bumps_and_a_refused_undo_does_not(client):
    client.post("/api/game/move", json={"move": "e4"})
    before = version_of(client)
    client.post("/api/game/undo", json={})
    after_undo = version_of(client)
    assert after_undo == before + 1
    assert client.post("/api/game/undo", json={}).status_code == 409
    assert version_of(client) == after_undo, "nothing was taken back"


def test_a_new_game_bumps(client):
    before = version_of(client)
    assert client.post("/api/game/new", json={"color": "white"}).status_code == 200
    assert version_of(client) == before + 1


def test_a_resignation_bumps(client):
    before = version_of(client)
    assert client.post("/api/game/resign", json={}).status_code == 200
    assert version_of(client) == before + 1


def test_an_op_the_gate_only_armed_does_not_bump(client, ctx):
    """A refused destructive op did not happen: the gate armed it and asked. The
    version is what the board is, so an unanswered question cannot move it."""
    developed(ctx)
    before = version_of(client)
    response = client.post("/api/game/new", json={"color": "white"})
    assert response.status_code == 409 and response.json()["confirm"] is True
    assert version_of(client) == before
    # …and answering it does.
    assert client.post("/api/game/confirm", json={"confirm": True}).status_code == 200
    assert version_of(client) == before + 1


def test_resuming_a_save_bumps_it(tmp_path):
    """`resume_game` swaps the session object on the context rather than mutating
    a board, which is exactly why the version cannot live on the session: a
    resumed game is a new position under the same clients."""
    ctx = ToolContext(session=GameSession(), save_dir=tmp_path)
    registry = build_registry(ctx)
    ctx.session.submit_move("e4")
    registry.dispatch("save_game", {"name": "s"})
    registry.dispatch("undo", {})
    before = ctx.board_version
    assert registry.dispatch("resume_game", {"name": "s"})["ok"] is True
    assert ctx.board_version == before + 1


# --- the precondition -------------------------------------------------------


def test_a_matching_version_is_accepted(client):
    version = version_of(client)
    response = client.post("/api/game/move", json={"move": "e4", "version": version})
    assert response.status_code == 200
    assert response.json()["state"]["version"] == version + 1


def test_an_omitted_version_is_todays_behavior(client):
    """Backward compatible on purpose: the precondition is a client's opt-in, so
    a client that never learned about versions keeps working exactly as it did."""
    response = client.post("/api/game/move", json={"move": "e4"})
    assert response.status_code == 200
    assert response.json()["san"] == "e4"


def test_a_stale_move_is_rejected_and_the_board_is_untouched(client):
    stale = version_of(client)
    client.post("/api/game/move", json={"move": "e4"})
    current = version_of(client)

    response = client.post("/api/game/move", json={"move": "d4", "version": stale})

    assert response.status_code == 409
    body = response.json()
    assert body["stale"] is True
    assert body["version"] == current, "the client is told what to resync to"
    assert body["state"]["history"] == ["e4"], "and gets the position with it"
    assert "e4" in body["detail"] or "changed" in body["detail"]
    assert client.get("/api/state").json()["history"] == ["e4"]


@pytest.mark.parametrize(
    ("path", "payload"),
    [
        ("/api/game/move", {"move": "d4"}),
        ("/api/game/undo", {}),
        ("/api/game/new", {"color": "white"}),
        ("/api/game/resign", {}),
        ("/api/game/confirm", {"confirm": True}),
    ],
)
def test_every_mutating_endpoint_rejects_a_stale_version(client, path, payload):
    stale = version_of(client)
    client.post("/api/game/move", json={"move": "e4"})
    history = client.get("/api/state").json()["history"]

    response = client.post(path, json={**payload, "version": stale})

    assert response.status_code == 409
    assert response.json()["stale"] is True
    assert client.get("/api/state").json()["history"] == history


def test_a_stale_check_precedes_abandoning_the_turn(ctx):
    """ "The board must be untouched" includes the turn machine: the check runs
    before anything is thrown away, so a stale undo does not cost the open turn
    the engine still owes a reply to."""
    coordinator = TurnCoordinator(ctx)
    registry = build_registry(ctx, coordinator, atomic_exchange=False)
    client = TestClient(create_app(ctx, registry=registry, coordinator=coordinator))
    stale = version_of(client)
    coordinator.apply_player_move("e4")

    response = client.post("/api/game/undo", json={"version": stale})

    assert response.status_code == 409
    assert coordinator.phase is TurnPhase.PLAYER_MOVE_APPLIED
    assert coordinator.turn_id == 1


def test_a_stale_command_never_reaches_the_brain():
    """The command pipeline is a mutation path too — one utterance can move the
    board — so it carries the same precondition, and a stale one is refused
    before the model is asked anything."""
    ctx = ToolContext(session=GameSession())
    app, brain = scripted_app(
        ctx,
        AgentResponse(
            text="ok", tool_calls=(ToolCall(name="make_move", args={"move": "d4"}),)
        ),
    )
    client = TestClient(app)
    stale = version_of(client)
    ctx.session.submit_move("e4")

    response = client.post("/api/command", json={"text": "play d4", "version": stale})

    assert response.status_code == 409
    assert response.json()["stale"] is True
    assert brain.calls == []
    assert ctx.session.move_history() == ["e4"]


def test_the_delegate_wire_carries_the_precondition():
    """A conductor driving chess over the delegate API shares the one session
    with the web board, so it gets the same opt-in precondition at its own entry
    point."""
    ctx = ToolContext(session=GameSession())
    app, brain = scripted_app(ctx, AgentResponse(text="sure"))
    client = TestClient(app)
    conversation = client.post("/api/agent/conversations", json={}).json()["id"]
    stale = version_of(client)
    ctx.session.submit_move("e4")

    response = client.post(
        f"/api/agent/conversations/{conversation}/messages",
        json={"content": "how am I doing?", "version": stale},
    )

    assert response.status_code == 409
    assert response.json()["stale"] is True
    assert brain.calls == []


# --- the concurrency acceptance test ----------------------------------------


class BarrierEngine(FakeEngine):
    """A `FakeEngine` that parks inside its reply, and refuses to overlap.

    Two jobs, both about concurrency. `entered` fires as the engine starts
    thinking, so a test can hold a second client until the first is provably
    mid-mutation; and any *overlap* between two calculations is recorded, which
    is what serialization means when there is no version to check (the MCP
    surface).
    """

    def __init__(self, reply_uci: str = "e7e5", delay: float = 0.2) -> None:
        super().__init__(reply_uci=reply_uci)
        self.entered = threading.Event()
        self.delay = delay
        self._inside = 0
        self._lock = threading.Lock()
        self.max_overlap = 0

    def choose_move(self, session):
        with self._lock:
            self._inside += 1
            self.max_overlap = max(self.max_overlap, self._inside)
        self.entered.set()
        time.sleep(self.delay)
        with self._lock:
            self._inside -= 1
        # Whatever is legal *now*: these tests play several turns, and a fixed
        # scripted reply would come back illegal on the second one and quietly
        # turn "the turn advanced" into "half of it did".
        return session.legal_moves()[0]

    def play_move(self, session):
        return session.submit_move(self.choose_move(session))


def test_two_clients_cannot_advance_the_same_turn_twice():
    """The audit's acceptance test (item 22, "concurrent clients cannot advance
    the same turn twice").

    Both clients read version N and both submit a move. The first is inside the
    engine's calculation — holding the mutation lock — when the second arrives,
    which is the interleaving the whole slice exists for: without the lock the
    second would pass its version check against a board the first has already
    changed. Exactly one move is accepted, the other is told it is stale, and the
    board advanced exactly one turn.
    """
    engine = BarrierEngine()
    client, ctx = engine_client(engine)
    version = version_of(client)
    results: dict[str, int] = {}

    def submit(name: str, move: str) -> None:
        results[name] = client.post(
            "/api/game/move", json={"move": move, "version": version}
        ).status_code

    first = threading.Thread(target=submit, args=("first", "e4"))
    first.start()
    assert engine.entered.wait(5), "the first client never reached the engine"
    second = threading.Thread(target=submit, args=("second", "d4"))
    second.start()
    for thread in (first, second):
        thread.join(10)

    assert sorted(results.values()) == [200, 409]
    assert results["first"] == 200
    history = ctx.session.move_history()
    assert len(history) == 2 and history[0] == "e4", "exactly one turn advanced"
    assert version_of(client) == version + 2


def test_the_mcp_surface_serializes_its_mutations():
    """MCP gets no `version` parameter — its tools are advertised from the *same*
    schema objects the brain is offered, and changing those collapses the eval
    floor on gemma-4-12b (TODO.md's standing warning). What it gets instead is
    the other half: every dispatch goes through the context's mutation lock, so
    two concurrent MCP calls cannot interleave a turn, and its `make_move` is the
    atomic exchange, so neither can leave one half-played.
    """
    engine = BarrierEngine(delay=0.1)
    ctx = ToolContext(session=GameSession(), engine=engine)
    server = build_mcp_server(ctx)
    tools = {tool.name: tool for tool in server._tool_manager.list_tools()}
    make_move = tools["make_move"]

    threads = [
        threading.Thread(target=make_move.fn, kwargs={"move": move})
        for move in ("e4", "d4")
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10)

    assert engine.max_overlap == 1, "two MCP turns overlapped on one session"
    assert len(ctx.session.move_history()) == 4, "two whole exchanges, none torn"


# --- what the model is never told -------------------------------------------


def test_the_brain_is_never_shown_a_version(ctx):
    """Transport bookkeeping, not board truth: a version in the prompt is one
    more number the model could try to reason with, and it has nothing to do
    with a move."""
    assert "version" not in _agent_state_dict(ctx)


def test_no_tool_schema_gained_a_version_argument(ctx):
    """The schemas the brain is offered are frozen by the eval floor (the shelved
    schema-cut experiment). The precondition lives at the transport layer for
    exactly that reason."""
    for definition in build_registry(ctx).definitions():
        properties = definition["function"]["parameters"].get("properties", {})
        assert "version" not in properties
