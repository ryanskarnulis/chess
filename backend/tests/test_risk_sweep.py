"""The audit's E2E risk list, swept for what nothing else pins (audit item 22).

Item 22 named eight architectural risks and asked for an end-to-end test each.
Seven of the eight arrived as acceptance tests inside the slices that built the
architecture, and this file does not duplicate them — it records where they live
and closes the residue the sweep found:

- *a board drag passes through Glitch* — `test_board_controls.py`, whole file.
- *Glitch observes the player move before the engine reply* —
  `test_command.py`'s "observe beat" block, `test_coordinator.py`'s phases;
  **plus the delegate wire, below.**
- *a command cannot produce duplicate moves* — `test_closing_pass.py`,
  `test_trace.py`'s mutation counts, `test_coordinator.py`'s rejections.
- *hints-off never exposes advice* — `test_app.py` (the withheld offer),
  `test_command.py` (the advice guard); **plus the delegate wire and the line
  between the agent's advice and the player's own ask, below.**
- *pending destructive ops cannot be overwritten* — `test_tools.py`,
  `test_command.py`, `test_board_controls.py`; **plus staleness across
  surfaces, below.**
- *provider failure leaves a resumable position* — `test_command.py`'s
  "Recovery" block; **plus the delegate wire, below.**
- *stale board versions are rejected* — `test_board_version.py`.
- *concurrent clients cannot advance the same turn twice* —
  `test_board_version.py`, in direct mode; **plus agent mode, below.**

What the sweep found, and what is new here:

1. **An armed destructive question outlived the board it was about.** Every
   *command* disarms a pending op on its way past (an unrelated utterance is a
   new intent), but the other surfaces that move the board — a drag, an undo, a
   resume, another client — did not, so a question asked about one position
   could be answered against a different one: arm a resignation by voice, drag
   a move, say "yes", and a game two plies further on ended. Fixed where the
   answer is read rather than at each surface that would have had to remember:
   `PendingOp` carries the board it is about (`ToolContext.live_pending`), so
   the op is *derived* to be stale rather than explicitly cleared — the same
   move as the board-version precondition one layer up.

2. **Agent mode had no concurrency test of its own.** The item-8 acceptance
   test drives `/api/game/move` in direct mode; in agent mode that endpoint is
   a different code path (the coordinator's beats, a narration, an offloop
   hop), and a turn is longer, so the window is wider.

3. **The delegate wire was never asserted to share the guards.** It runs the
   same pipeline as the panel, which is the design — but "one road in" is a
   claim about behavior, and nothing pinned it on the observe split, the advice
   guard or provider recovery.
"""

import threading
import time

from fastapi.testclient import TestClient

from chessapp.api import (
    MOVE_ADVICE_REPLY,
    PROVIDER_LOST_TURN_STANDS,
    create_app,
)
from chessapp.brain import AgentResponse, ToolCall
from chessapp.coordinator import TurnCoordinator
from chessapp.engine import CandidateMove
from chessapp.game import GameSession
from chessapp.tools import ToolContext, build_registry
from fakes import FakeEngine, ScriptedBrain, scripted_app

# --- helpers ----------------------------------------------------------------


def agent_client(*responses: AgentResponse, narrations: tuple = (), engine=None):
    """Agent mode over a fresh game: a scripted brain in the path, never a
    model. `narrations` covers the observe beat on the move routes."""
    ctx = ToolContext(session=GameSession(), engine=engine)
    brain = ScriptedBrain(*responses, narrations=narrations)
    app, _ = scripted_app(ctx, brain=brain)
    return TestClient(app), ctx


def developed(ctx: ToolContext) -> ToolContext:
    """A game with player investment in it — enough for the gate to ask."""
    for san in ("e4", "e5", "Nf3", "Nc6"):
        ctx.session.submit_move(san)
    return ctx


def destructive(name: str) -> AgentResponse:
    """A brain turn that asks for a destructive op (the gate refuses and arms
    it; the words are the question the pipeline relays)."""
    return AgentResponse(
        text="You sure about that?", tool_calls=(ToolCall(name=name, args={}),)
    )


def version_of(client: TestClient) -> int:
    return client.get("/api/state").json()["version"]


def conversation_on(client: TestClient) -> int:
    return client.post("/api/agent/conversations", json={}).json()["id"]


def say(client: TestClient, conversation: int, content: str, **extra) -> dict:
    """One delegate-wire turn — the conductor's entry point into the pipeline."""
    return client.post(
        f"/api/agent/conversations/{conversation}/messages",
        json={"content": content, **extra},
    ).json()


# --- 1. A question is about a board: an answer to it cannot outlive that board


def test_a_drag_between_the_question_and_the_answer_drops_the_question():
    """The gate armed a resignation about *this* position. Then the player
    dragged a move, so the position they were asked about is two plies gone —
    and a "yes" said after that is not an answer to it. The question is dropped
    and the utterance is a new intent, exactly as any other unrelated command
    disarms an op."""
    client, ctx = agent_client(
        AgentResponse(text="Yes to what?"), narrations=("Fine.",), engine=FakeEngine()
    )
    developed(ctx)
    client.post("/api/command", json={"text": "i resign"})
    assert ctx.pending is not None, "the gate asked"

    client.post("/api/game/move", json={"move": "f1c4"})
    body = client.post("/api/command", json={"text": "yes"}).json()

    assert not ctx.session.is_game_over(), "a stale yes ended a game it never saw"
    assert ctx.pending is None, "and the question is not left armed either"
    # With nothing live to confirm, the "yes" is just an utterance: it reaches
    # the brain, which can ask what they meant. Nothing about the board changed.
    assert body["commentary"] == "Yes to what?"


def test_an_undo_between_the_question_and_the_answer_drops_it_too():
    """Same rule, a different surface: the board control that takes a move
    back is not an answer to a question about the board before it."""
    client, ctx = agent_client(AgentResponse(text="Yes to what?"), engine=FakeEngine())
    developed(ctx)
    client.post("/api/command", json={"text": "i resign"})

    assert client.post("/api/game/undo", json={}).status_code == 200
    client.post("/api/command", json={"text": "yes"})

    assert not ctx.session.is_game_over()
    assert ctx.session.move_history() == ["e4", "e5"], "the takeback stands"


def test_the_confirm_button_refuses_a_question_about_a_board_that_moved():
    """The button half of the same gate: a click that arrives after the board
    moved has nothing to confirm (409), rather than confirming something
    else."""
    client, ctx = agent_client(narrations=("Fine.", "Fine."), engine=FakeEngine())
    developed(ctx)
    client.post("/api/command", json={"text": "i resign"})

    client.post("/api/game/move", json={"move": "f1c4"})
    response = client.post("/api/game/confirm", json={"confirm": True})

    assert response.status_code == 409
    assert not ctx.session.is_game_over()


def test_an_op_armed_in_a_turn_that_also_moved_is_still_answerable():
    """The other side of the rule, and the reason the stamp is refreshed when
    the turn closes: "play Bc4 and start over" arms the reset *mid*-turn, and
    the engine's reply lands after it. The board the player is looking at when
    they hear the question is the one at the end of that turn, so their yes is
    an answer to it — a stamp taken at arm time would have gone stale before
    they could speak."""
    ctx = ToolContext(session=GameSession(), engine=FakeEngine())
    developed(ctx)
    brain = ScriptedBrain(
        AgentResponse(
            text="Bc4 played. Sure you want a fresh board?",
            tool_calls=(
                ToolCall(name="make_move", args={"move": "Bc4"}),
                ToolCall(name="new_game", args={}),
            ),
        ),
        narrations=("Fine.",),
    )
    app, _ = scripted_app(ctx, brain=brain)
    client = TestClient(app)

    client.post("/api/command", json={"text": "play Bc4 and then start over"})
    assert ctx.pending is not None and ctx.pending.name == "new_game"
    client.post("/api/command", json={"text": "yes"})

    assert ctx.session.move_history() == [], "the confirmed reset ran"


def test_the_newest_question_is_the_one_a_yes_answers():
    """Two asks, two turns: the second replaces the first rather than queueing
    behind it, so a yes answers the question the player just heard. (Inside one
    command the destructive budget refuses the second outright — `test_closing_
    pass.py` — so this is the only way two can be asked.)"""
    client, ctx = agent_client(
        destructive("new_game"), destructive("resign"), narrations=("Fine.",)
    )
    developed(ctx)

    client.post("/api/command", json={"text": "new game"})
    client.post("/api/command", json={"text": "actually, i resign"})
    assert ctx.pending is not None and ctx.pending.name == "resign"
    client.post("/api/command", json={"text": "yes"})

    assert ctx.session.is_game_over(), "the resignation the player just asked for"
    assert ctx.session.move_history() == ["e4", "e5", "Nf3", "Nc6"], "not a reset"


# --- 2. Concurrency, on the agent-mode routes


class BarrierEngine(FakeEngine):
    """A `FakeEngine` that parks inside its reply and records overlap — the
    interleaving a second client has to arrive in to race a turn (see
    `test_board_version.py`, which pins the direct-mode half)."""

    def __init__(self, delay: float = 0.2) -> None:
        super().__init__()
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
        return session.legal_moves()[0]

    def play_move(self, session):
        return session.submit_move(self.choose_move(session))


def dragging_client(engine):
    """Agent mode with a coordinator in hand, for the concurrency assertions."""
    ctx = ToolContext(session=GameSession(), engine=engine)
    coordinator = TurnCoordinator(ctx)
    registry = build_registry(ctx, coordinator, atomic_exchange=False)
    brain = ScriptedBrain(narrations=("a", "b", "c", "d"), dispatcher=registry)
    app = create_app(ctx, brain=brain, registry=registry, coordinator=coordinator)
    return TestClient(app), coordinator, ctx


def drag_both(client, first_move: str, second_move: str, engine, **body) -> dict:
    """Two clients drag into the same board, the second arriving while the
    first is provably inside the engine's calculation."""
    results: dict[str, int] = {}

    def submit(name: str, move: str) -> None:
        results[name] = client.post(
            "/api/game/move", json={"move": move, **body}
        ).status_code

    first = threading.Thread(target=submit, args=("first", first_move))
    first.start()
    assert engine.entered.wait(5), "the first client never reached the engine"
    second = threading.Thread(target=submit, args=("second", second_move))
    second.start()
    for thread in (first, second):
        thread.join(10)
    return results


def test_two_agent_mode_drags_on_one_version_advance_one_turn():
    """Item 8's acceptance test on the agent route. Agent mode is a longer turn
    than direct mode — the coordinator's beats plus a narration — so the window
    for a second client is wider; the precondition still admits exactly one."""
    engine = BarrierEngine()
    client, _, ctx = dragging_client(engine)
    version = version_of(client)

    results = drag_both(client, "e2e4", "d2d4", engine, version=version)

    assert sorted(results.values()) == [200, 409]
    assert results["first"] == 200
    history = ctx.session.move_history()
    assert len(history) == 2 and history[0] == "e4", "exactly one turn advanced"


def test_two_unversioned_agent_mode_drags_are_whole_turns_never_a_torn_one():
    """No `version` is no exclusion — a client that wants it sends one. What
    the mutation lock still guarantees is that the two turns cannot interleave:
    two full exchanges, in order, and never a player move that landed twice
    inside one."""
    engine = BarrierEngine()
    client, coordinator, ctx = dragging_client(engine)

    results = drag_both(client, "e2e4", "d2d4", engine)

    assert sorted(results.values()) == [200, 200]
    assert engine.max_overlap == 1, "two turns overlapped on one session"
    assert len(ctx.session.move_history()) == 4, "two whole exchanges, none torn"
    assert coordinator.turn_id == 3, "one turn each, neither of them twice"


def test_two_clicks_on_the_same_question_run_one_op():
    """A double-clicked confirmation: `confirm_pending` consumes the armed op
    under the mutation lock, so the second click finds nothing to confirm
    instead of resetting the board the first one just made."""
    client, ctx = agent_client(destructive("new_game"), narrations=("Fine.",))
    developed(ctx)
    client.post("/api/command", json={"text": "new game"})
    statuses: list[int] = []

    def confirm() -> None:
        statuses.append(
            client.post("/api/game/confirm", json={"confirm": True}).status_code
        )

    threads = [threading.Thread(target=confirm) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10)

    assert sorted(statuses) == [200, 409]
    assert ctx.session.move_history() == [], "reset once"


# --- 3. The delegate wire is the same road, guards included


def test_the_delegate_wire_observes_the_player_move_before_the_reply():
    """A conductor-driven move is a turn like any other: the reaction is to the
    player's move alone, and the app announces the reply itself."""
    ctx = ToolContext(session=GameSession(), engine=FakeEngine("e7e5"))
    brain = ScriptedBrain(narrations=("Bold.",))
    app, _ = scripted_app(ctx, brain=brain)
    client = TestClient(app)

    exchange = say(client, conversation_on(client), "e4")

    state, changes = brain.narrate_calls[0]
    assert state["history"] == ["e4"], "the engine has not replied yet"
    assert "engine_move" not in changes[0]["result"]
    assert exchange["assistant_message"]["content"] == "Bold.\n\ne5."
    assert ctx.session.move_history() == ["e4", "e5"]


def test_the_delegate_wire_scrubs_move_advice_with_hints_off():
    """The advice guard is the pipeline's, so it holds at every entry point:
    hints off means no move is handed over, whoever asked."""
    ctx = ToolContext(session=GameSession())
    app, _ = scripted_app(ctx, AgentResponse(text="Easy — play Nf3 and thank me."))
    client = TestClient(app)

    exchange = say(client, conversation_on(client), "what should I play?")

    assert ctx.settings.hints_mode is False
    assert exchange["assistant_message"]["content"] == MOVE_ADVICE_REPLY


def test_a_provider_failure_on_the_delegate_wire_keeps_the_move_and_says_so():
    """Recovery, at the conductor's entry point: a move that landed before the
    provider died stands, the reply is collected, the turn closes, and the
    stored assistant turn carries `provider_error` — so the caller can tell a
    turn that half-ran from one that never started."""
    ctx = ToolContext(session=GameSession(), engine=FakeEngine("e7e5"))
    app, _ = scripted_app(
        ctx,
        AgentResponse(
            text="",
            tool_calls=(ToolCall(name="make_move", args={"move": "e4"}),),
            stop_reason="provider_error",
        ),
    )
    client = TestClient(app)

    exchange = say(client, conversation_on(client), "push the king pawn")

    assistant = exchange["assistant_message"]
    assert assistant["stop_reason"] == "provider_error"
    assert assistant["content"] == f"{PROVIDER_LOST_TURN_STANDS}\n\ne5."
    assert ctx.session.move_history() == ["e4", "e5"], "kept, and the reply collected"


def test_the_hint_button_is_the_players_own_ask_not_the_agents_advice():
    """The boundary the hints setting draws, pinned so it stays deliberate:
    `hints_mode` governs what *Glitch* volunteers — the tool it is offered and
    the moves its commentary may name. `/api/game/hint` is the player pressing
    Hint, on the trusted path with no model in it, and it answers (the recorded
    decision when hints gating shipped: MCP, the delegate wire and this
    endpoint keep the full registry)."""
    best = CandidateMove(uci="g1f3", san="Nf3", score_cp=30, mate_in=None)
    ctx = ToolContext(session=GameSession(), engine=FakeEngine(best_moves=(best,)))
    client = TestClient(create_app(ctx))

    assert ctx.settings.hints_mode is False
    assert client.get("/api/game/hint").json()["san"] == "Nf3"
