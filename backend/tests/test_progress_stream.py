"""Live phase progress on the wire (audit item 19).

A turn stopped being a black box when it gained intentional phases — validate
the player's move, let Glitch react, let Stockfish answer, close — and a
spinner is the wrong shape for that. These tests are the route-level assertion
that the phases *reach the client*, and that they reach it **while the turn is
still running**, which is the only property that makes the feature worth
anything. That property is pinned in two halves, because no single test in
this harness can carry both: one shows the events are *published* while the
turn is provably still inside the model, the other shows the event loop is
free to deliver them (which is what the off-loop hop in `api._offloop` buys).
Together they are the guarantee; separately neither is.

The unit contracts live one layer down (`test_progress.py` for the reporter,
`test_coordinator.py` for the phase observer, `test_tools.py` for the dispatch
observer, `test_llama_brain.py` for the brain's two phases). What this file
pins is the wiring: which events a real route produces, in what order, under
one correlation id.
"""

import threading

import pytest
from fastapi.testclient import TestClient

from chessapp.api import create_app
from chessapp.app import build_app
from chessapp.brain import AgentResponse, ToolCall
from chessapp.coordinator import TurnPhase
from chessapp.game import GameSession
from chessapp.progress import (
    BRAIN_NARRATING,
    BRAIN_PLANNING,
    KIND_BEGIN,
    KIND_BRAIN,
    KIND_END,
    KIND_PHASE,
    KIND_TOOL,
)
from chessapp.tools import ToolContext
from fakes import (
    FakeEngine,
    ScriptedBrain,
    ScriptedProvider,
    scripted_app,
    text_turn,
    tool_calls_turn,
)


def scripted_client(*responses: AgentResponse, narrations=(), engine=None):
    """The pipeline over a `ScriptedBrain` — enough for every route whose
    phases come from the coordinator and the registry."""
    ctx = ToolContext(session=GameSession(), engine=engine or FakeEngine())
    app, brain = scripted_app(
        ctx, brain=ScriptedBrain(*responses, narrations=narrations)
    )
    return TestClient(app), ctx


def live_client(*turns):
    """The *assembled* app over a real `LlamaBrain` with a scripted provider —
    the only wiring in which the brain's own two phases are reported, because
    only assembly can reach the brain (`build_app`)."""
    return TestClient(
        build_app(brain=None, provider=ScriptedProvider(*turns), engine=FakeEngine())
    )


def drain(ws, client_call) -> list[dict]:
    """Run `client_call` and collect every progress event of the one
    interaction it produced, `begin` through `end`."""
    client_call()
    events: list[dict] = []
    while True:
        message = ws.receive_json()
        if message["type"] != "progress":
            continue
        events.append(message["progress"])
        if message["progress"]["kind"] == KIND_END:
            return events


def steps(events: list[dict]) -> list[tuple[str, str]]:
    return [(event["kind"], event["name"]) for event in events]


# --- what a move turn tells the player it is doing ---------------------------


def test_a_fast_path_move_reports_every_beat_of_the_turn():
    """The audit's three example states, in the order a move actually passes
    through them: the move is validated, Glitch reacts to the verified move,
    Stockfish answers. Each comes from the machine that owns it — the registry
    for the tool, the coordinator's phase setter for the rest."""
    client, _ = scripted_client(narrations=("Bold.",))
    with client.websocket_connect("/ws") as ws:
        events = drain(ws, lambda: client.post("/api/command", json={"text": "e4"}))
    assert steps(events) == [
        (KIND_BEGIN, ""),
        (KIND_TOOL, "make_move"),
        (KIND_PHASE, TurnPhase.PLAYER_MOVE_APPLIED),
        (KIND_PHASE, TurnPhase.AGENT_OBSERVING),
        (KIND_PHASE, TurnPhase.ENGINE_CALCULATING),
        (KIND_PHASE, TurnPhase.ENGINE_MOVE_APPLIED),
        (KIND_PHASE, TurnPhase.COMPLETED),
        (KIND_PHASE, TurnPhase.AWAITING_PLAYER),
        (KIND_END, ""),
    ]


def test_a_dragged_move_reports_the_same_beats():
    """One road in: a drag runs the fast path's beats through the same helper,
    so it says the same things — the drag is not a quieter kind of turn."""
    client, _ = scripted_client(narrations=("Bold.",))
    with client.websocket_connect("/ws") as ws:
        events = drain(ws, lambda: client.post("/api/game/move", json={"move": "e2e4"}))
    assert steps(events)[:4] == [
        (KIND_BEGIN, ""),
        (KIND_TOOL, "make_move"),
        (KIND_PHASE, TurnPhase.PLAYER_MOVE_APPLIED),
        (KIND_PHASE, TurnPhase.AGENT_OBSERVING),
    ]
    assert steps(events)[-1] == (KIND_END, "")


def test_a_read_only_command_reports_its_tool_and_nothing_the_board_did():
    """Nothing moved, so no phase moved. The turn still brackets itself, which
    is what clears the line when the answer arrives."""
    client, _ = scripted_client(
        AgentResponse(
            text="Twenty of them.",
            tool_calls=(ToolCall(name="get_legal_moves", args={}),),
        )
    )
    with client.websocket_connect("/ws") as ws:
        events = drain(
            ws, lambda: client.post("/api/command", json={"text": "my options?"})
        )
    assert steps(events) == [
        (KIND_BEGIN, ""),
        (KIND_TOOL, "get_legal_moves"),
        (KIND_END, ""),
    ]


def test_every_event_of_a_turn_carries_one_correlation_id():
    """The key the trace record already uses: a line the player saw and the
    record of the turn behind it are one search apart."""
    client, _ = scripted_client(narrations=("Bold.",))
    with client.websocket_connect("/ws") as ws:
        events = drain(ws, lambda: client.post("/api/command", json={"text": "e4"}))
    assert len({event["correlation_id"] for event in events}) == 1
    assert {event["turn_id"] for event in events} == {1}


def test_an_illegal_move_still_closes_its_progress_line():
    """Nothing moved and no phase moved, but the interaction opened — so it
    closes. A turn that went wrong is the one whose line must not be left
    spinning on the player's screen."""
    client, ctx = scripted_client(
        AgentResponse(
            text="Can't do that.",
            tool_calls=(ToolCall(name="make_move", args={"move": "e5"}),),
        )
    )
    with client.websocket_connect("/ws") as ws:
        events = drain(
            ws, lambda: client.post("/api/command", json={"text": "pawn to e5"})
        )
    assert steps(events) == [
        (KIND_BEGIN, ""),
        (KIND_TOOL, "make_move"),
        (KIND_END, ""),
    ]
    assert ctx.session.move_history() == []


# --- the brain's own phases, and the observe beat becoming real --------------


def test_the_brain_route_reports_planning_then_narrating():
    client = live_client(
        tool_calls_turn(("make_move", {"move": "e4"})),
        text_turn("note: played e4"),
        text_turn("Pawn to e4. Your move."),
    )
    with client.websocket_connect("/ws") as ws:
        events = drain(
            ws, lambda: client.post("/api/command", json={"text": "play e4"})
        )
    brain_steps = [event["name"] for event in events if event["kind"] == KIND_BRAIN]
    assert brain_steps == [BRAIN_PLANNING, BRAIN_PLANNING, BRAIN_NARRATING]


def test_the_narrator_turn_is_the_observation_beat():
    """`agent_observing` was a phase the app never entered — the gap
    `docs/turn-coordinator.md` left for this slice. On the brain route the
    reaction happens *inside* `get_agent_response`, which holds no coordinator,
    so the brain's report of its narrator phase is what opens the beat."""
    client = live_client(
        tool_calls_turn(("make_move", {"move": "e4"})),
        text_turn("note: played e4"),
        text_turn("Pawn to e4."),
    )
    with client.websocket_connect("/ws") as ws:
        events = drain(
            ws, lambda: client.post("/api/command", json={"text": "play e4"})
        )
    ordered = steps(events)
    assert (KIND_PHASE, TurnPhase.AGENT_OBSERVING) in ordered
    # And in the right place: after the move landed, before Stockfish answered.
    assert (
        ordered.index((KIND_PHASE, TurnPhase.PLAYER_MOVE_APPLIED))
        < ordered.index((KIND_PHASE, TurnPhase.AGENT_OBSERVING))
        < ordered.index((KIND_PHASE, TurnPhase.ENGINE_CALCULATING))
    )


def test_a_turn_that_moves_nothing_never_claims_an_observation():
    """The narrator runs on every turn; the observe beat exists only when a
    move is waiting on one."""
    client = live_client(text_turn("note: nothing to do"), text_turn("All quiet."))
    with client.websocket_connect("/ws") as ws:
        events = drain(
            ws, lambda: client.post("/api/command", json={"text": "how's it look?"})
        )
    assert (KIND_PHASE, TurnPhase.AGENT_OBSERVING) not in steps(events)
    assert (KIND_BRAIN, BRAIN_NARRATING) in steps(events)


# --- direct mode says nothing, because there is nothing to say ---------------


def test_direct_mode_reports_no_progress():
    """No brain, no multi-phase turn: `/api/game/move` answers the one
    deterministic exchange it always answered, and the state broadcast is the
    whole story. Nothing opens an interaction, so nothing is reported — the
    reporter's own rule (an event outside an interaction does not exist), not a
    special case anybody wrote for this route."""
    ctx = ToolContext(session=GameSession(), engine=FakeEngine())
    client = TestClient(create_app(ctx))
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()  # connect snapshot
        client.post("/api/game/move", json={"move": "e2e4"})
        message = ws.receive_json()
    assert message["type"] == "state"
    assert message["state"]["history"] == ["e4", "e5"]


# --- live, or it is nothing --------------------------------------------------
#
# The two tests the feature actually stands on. Both drive the app from a
# second thread, because a progress event that only arrives once the HTTP
# response does is not progress — it is a transcript. They split the guarantee
# deliberately: `TestClient` gives a websocket session its *own* event loop
# (`portal_factory` is called again per session), so a socket in this harness
# can be served by a loop the request is not on — which is exactly the
# confusion that would let a "live" assertion pass against a blocked server.
# So one test pins the publish side and the other pins the loop side, through
# the plainest probe there is.


class BlockingBrain:
    """A brain that stops inside the turn until the test lets it go, so the
    test can look at what the client has been told *so far*."""

    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()
        self.dispatcher = None

    def get_agent_response(self, board_state, command, transcript=()):
        self.entered.set()
        assert self.release.wait(5), "test never released the brain"
        return AgentResponse(text="Done.")

    def narrate(self, board_state, changes, transcript=()):  # pragma: no cover
        raise AssertionError("this route never narrates")


@pytest.fixture
def blocking():
    """A client **entered as a context manager**, which is load-bearing: only
    then does `TestClient` hold one portal — one event loop — across every
    request. Left unentered it spins a fresh loop per request, and a test for
    "the loop stayed free" would pass against a loop nothing shared."""
    brain = BlockingBrain()
    ctx = ToolContext(session=GameSession(), engine=FakeEngine())
    app, _ = scripted_app(ctx, brain=brain)
    with TestClient(app) as client:
        yield client, brain
        brain.release.set()


def test_progress_is_published_before_the_turn_finishes(blocking):
    """Half one: the turn is *provably* still inside the model — the brain is
    parked on an event this thread has not set — and the client already has the
    turn's first event. Whatever else is true, the stream is not a batch
    flushed at the end."""
    client, brain = blocking
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()  # connect snapshot
        caller = threading.Thread(
            target=lambda: client.post("/api/command", json={"text": "think hard"})
        )
        caller.start()
        try:
            assert brain.entered.wait(5), "the command never reached the brain"
            # The turn is provably still inside the brain, and the begin event
            # is already here. That is the whole feature.
            message = ws.receive_json()
            assert message["type"] == "progress"
            assert message["progress"]["kind"] == KIND_BEGIN
        finally:
            brain.release.set()
            caller.join(5)


def test_the_event_loop_stays_free_while_a_turn_runs(blocking):
    """Half two, and the mechanism the whole feature rests on: nothing can be
    delivered live from a loop parked inside a model call, so the pipeline's
    blocking steps run off it (`api._offloop`). Read through the plainest
    possible probe — another request, answered mid-turn. Delete the hop and
    this is the test that goes red."""
    client, brain = blocking
    caller = threading.Thread(
        target=lambda: client.post("/api/command", json={"text": "think hard"})
    )
    caller.start()
    try:
        assert brain.entered.wait(5), "the command never reached the brain"
        answered = threading.Event()
        probe = threading.Thread(
            target=lambda: (client.get("/api/state"), answered.set())
        )
        probe.start()
        assert answered.wait(5), "the event loop was blocked by the turn in flight"
        probe.join(5)
    finally:
        brain.release.set()
        caller.join(5)
