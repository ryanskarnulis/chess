"""Board controls in agent mode: a drag enters the same turn sequence.

Sprint 1, slice 4 of the agent-control replan (audit items 1 and 4). Until now
`/api/game/move` was a silent bypass: a dragged move played the whole exchange
itself and Glitch never saw it, so the "personality in the path" story covered
only typed and spoken commands. Here the endpoint becomes *mode-aware* — with a
brain configured it runs the same beats the command pipeline's fast path runs
(dispatch `make_move`, observe, collect the reply, close), and with no brain it
is byte-for-byte the direct-mode endpoint it always was.

The other half is the confirmation gate: the UI's new-game and resign buttons
used to act immediately while the spoken path armed `ctx.pending` and asked.
They now dispatch through the same registry, so the *same* gate decides, and an
op armed by a button can be confirmed by a typed "yes" (and vice versa).

The brain is a `ScriptedBrain` throughout — never a live model. What is pinned
here is the route, the beats' order, the wire shapes, and that direct mode is
untouched.
"""

import json

from fastapi.testclient import TestClient

from chessapp.api import UNTRUE_CLAIM_REPLY, create_app
from chessapp.brain import AgentResponse, ToolCall
from chessapp.coordinator import TurnCoordinator, TurnPhase
from chessapp.game import GameSession
from chessapp.provider import ProviderError
from chessapp.tools import ToolContext, build_registry
from chessapp.trace import JsonlTracer
from fakes import FakeEngine, ScriptedBrain, scripted_app

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def agent_client(
    *responses: AgentResponse,
    narrations: tuple = (),
    verbosity: str = "normal",
    engine=None,
    **create_kwargs,
):
    """An app with a brain in the path — agent mode — over a fresh game."""
    ctx = ToolContext(session=GameSession(), engine=engine)
    ctx.settings.verbosity = verbosity
    brain = ScriptedBrain(*responses, narrations=narrations)
    app, _ = scripted_app(ctx, brain=brain, **create_kwargs)
    return TestClient(app), brain, ctx


def coordinated_client(*, narrations: tuple = (), engine=None):
    """Agent mode with the coordinator in hand, for the tests that assert on the
    turn machine itself. Mirrors app assembly: one coordinator, one registry
    built `atomic_exchange=False`, shared by the brain and the endpoints."""
    ctx = ToolContext(session=GameSession(), engine=engine)
    coordinator = TurnCoordinator(ctx)
    registry = build_registry(ctx, coordinator, atomic_exchange=False)
    brain = ScriptedBrain(narrations=narrations, dispatcher=registry)
    app = create_app(ctx, brain=brain, registry=registry, coordinator=coordinator)
    return TestClient(app), coordinator, ctx


def developed(ctx: ToolContext) -> ToolContext:
    for san in ("e4", "e5", "Nf3", "Nc6"):
        ctx.session.submit_move(san)
    return ctx


# --- A drag enters the same sequence ----------------------------------------


def test_a_drag_reaches_glitch_and_the_engine_answers():
    """The audit's headline: in agent mode a dragged move produces a Glitch turn.
    The reaction is to the *player's* move alone — the reply does not exist when
    the narrator is asked — and the app announces the reply itself."""
    client, brain, _ = agent_client(narrations=("Bold opener.",), engine=FakeEngine())

    body = client.post("/api/game/move", json={"move": "e2e4"}).json()

    assert brain.calls == [], "a drag is a move, not an utterance for the planner"
    state, changes = brain.narrate_calls[0]
    assert state["history"] == ["e4"], "the engine has not replied yet"
    assert changes[0]["name"] == "make_move"
    assert changes[0]["result"]["san"] == "e4"
    assert body["commentary"] == "Bold opener.\n\ne5."
    assert body["state"]["history"] == ["e4", "e5"]


def test_a_drag_keeps_the_move_response_shape():
    client, _, _ = agent_client(narrations=("Sure.",), engine=FakeEngine())
    body = client.post("/api/game/move", json={"move": "e2e4"}).json()
    assert body["legal"] is True
    assert body["san"] == "e4"
    assert body["uci"] == "e2e4"
    assert body["reason"] is None
    assert body["engine_move"] == {"legal": True, "san": "e5", "uci": "e7e5"}
    assert body["state"]["turn"] == "white"


def test_a_drag_carries_the_speak_flag():
    """The UI needs to know whether to voice the reaction; the server owns the
    setting, the client owns the playback — the contract `/api/command` has."""
    client, _, ctx = agent_client(narrations=("Loud.", "Louder."))
    assert client.post("/api/game/move", json={"move": "e2e4"}).json()["speak"] is False
    ctx.settings.voice_output = True
    assert client.post("/api/game/move", json={"move": "e7e5"}).json()["speak"] is True


def test_a_drag_closes_its_turn_on_the_shared_machine():
    client, coordinator, _ = coordinated_client(
        narrations=("Fine.",), engine=FakeEngine()
    )

    client.post("/api/game/move", json={"move": "e2e4"})

    assert coordinator.turn_id == 2
    assert coordinator.phase == TurnPhase.AWAITING_PLAYER


def test_a_drag_out_of_turn_is_409_and_settles_the_open_turn():
    """A turn-state rejection is a domain failure on the trusted path, exactly as
    in direct mode — the agent, dispatching the same tool, reads it as result data
    instead. The refused drag plays nothing, but the beats still close the turn
    that was left open, so the machine heals rather than wedging (the fast path's
    behavior, because it is the same helper)."""
    client, coordinator, ctx = coordinated_client(
        narrations=("Fine.",), engine=FakeEngine()
    )
    coordinator.apply_player_move("e4")  # a turn is open, mid-sequence

    response = client.post("/api/game/move", json={"move": "d2d4"})

    assert response.status_code == 409
    assert "d4" not in ctx.session.move_history(), "the refused drag played nothing"
    assert ctx.session.move_history() == ["e4", "e5"], "the owed reply was settled"
    assert coordinator.phase == TurnPhase.AWAITING_PLAYER


def test_low_verbosity_keeps_a_dragged_move_zero_llm():
    """The latency floor, on the board route too: verbosity=low skips the
    reaction and one canned line covers the move and the reply."""
    client, brain, _ = agent_client(verbosity="low", engine=FakeEngine())

    body = client.post("/api/game/move", json={"move": "e2e4"}).json()

    assert brain.narrate_calls == [] and brain.calls == []
    assert body["commentary"] == "e4. e5."


def test_a_failed_reaction_still_gets_the_reply_and_the_turn():
    """The reaction is optional by construction; the engine's reply is not. A
    provider failure costs the words and nothing else."""
    client, _, ctx = agent_client(
        narrations=(ProviderError("llama-server went away"),), engine=FakeEngine()
    )

    body = client.post("/api/game/move", json={"move": "e2e4"}).json()

    assert body["commentary"] == "e4. e5."
    assert ctx.session.move_history() == ["e4", "e5"]
    # And the turn closed: the next drag lands straight away.
    second = client.post("/api/game/move", json={"move": "g1f3"}).json()
    assert second["state"]["history"][:3] == ["e4", "e5", "Nf3"]


def test_an_illegal_drag_narrates_nothing_and_changes_nothing():
    client, brain, _ = agent_client(engine=FakeEngine())

    body = client.post("/api/game/move", json={"move": "e2e5"}).json()

    assert body["legal"] is False
    assert "e2e5" in body["reason"]
    assert body["commentary"] == "", "nothing happened, so there is nothing to react to"
    assert brain.narrate_calls == []
    assert body["state"]["fen"] == START_FEN


def test_a_game_ending_drag_has_no_reply_to_announce():
    client, _, ctx = agent_client(narrations=("Called it.",), engine=FakeEngine())
    for san in ("e4", "f6", "d3", "g5"):
        ctx.session.submit_move(san)

    body = client.post("/api/game/move", json={"move": "d1h5"}).json()

    assert ctx.session.is_game_over()
    assert body["commentary"] == "Called it."
    assert body["engine_move"] is None


def test_a_dishonest_drag_reaction_is_guarded():
    """The honesty guard runs on every route, this one included: a reaction that
    invents an ending is replaced with the truth."""
    client, _, ctx = agent_client(
        narrations=("That's the game. Game over.",), engine=FakeEngine()
    )

    body = client.post("/api/game/move", json={"move": "e2e4"}).json()

    assert not ctx.session.is_game_over()
    assert body["commentary"] == UNTRUE_CLAIM_REPLY


def test_a_drag_records_the_turn_on_the_transcript():
    """So Glitch's later turns remember its own drag reactions: the move's SAN
    stands in for the utterance a drag doesn't have."""
    client, brain, ctx = agent_client(
        AgentResponse(text="you opened with e4"),
        narrations=("Sharp.",),
        engine=FakeEngine(),
    )

    client.post("/api/game/move", json={"move": "e2e4"})

    assert ctx.transcript.window() == [
        {"role": "user", "content": "e4"},
        {"role": "assistant", "content": "Sharp.\n\ne5."},
    ]
    # And the next command sees it.
    client.post("/api/command", json={"text": "how did I open?"})
    assert brain.transcripts[0][0] == {"role": "user", "content": "e4"}


def test_a_drag_broadcasts_the_new_state():
    client, _, _ = agent_client(narrations=("ok",), engine=FakeEngine())
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()  # connect snapshot
        client.post("/api/game/move", json={"move": "e2e4"})
        message = ws.receive_json()
    assert message["state"]["history"] == ["e4", "e5"]


def test_a_drag_is_traced_as_the_board_route(tmp_path):
    trace_path = tmp_path / "turns.jsonl"
    client, _, _ = agent_client(
        narrations=("Bold.",), engine=FakeEngine(), tracer=JsonlTracer(trace_path)
    )

    client.post("/api/game/move", json={"move": "e2e4"})

    record = json.loads(trace_path.read_text().splitlines()[0])
    assert record["route"] == "board"
    assert record["utterance"] == "e2e4", "the structured move, never prose"
    assert record["engine_reply"] == {"san": "e5", "uci": "e7e5"}
    assert [t["name"] for t in record["tools"]] == ["make_move"]
    assert record["fen_before"] == START_FEN
    assert record["stop_reason"] == "completed"


# --- Direct mode: byte-identical -------------------------------------------


def test_direct_mode_drag_answers_exactly_as_before():
    """No brain, no agent beats, and no new keys on the wire: the LLM-off board
    route is the one thing this slice may not touch."""
    ctx = ToolContext(session=GameSession(), engine=FakeEngine())
    client = TestClient(create_app(ctx))

    body = client.post("/api/game/move", json={"move": "e2e4"}).json()

    assert set(body) == {"legal", "san", "uci", "reason", "engine_move", "state"}
    assert body["engine_move"] == {"legal": True, "san": "e5", "uci": "e7e5"}
    assert body["state"]["history"] == ["e4", "e5"]


# --- The confirmation gate, shared by the buttons and the words --------------


def test_new_game_mid_game_asks_instead_of_resetting():
    ctx = developed(ToolContext(session=GameSession()))
    client = TestClient(create_app(ctx))
    fen_before = ctx.session.fen()

    response = client.post("/api/game/new", json={"color": "white"})

    assert response.status_code == 409
    body = response.json()
    assert body["confirm"] is True
    assert body["op"] == "new_game"
    assert body["detail"], "the gate's question, in the app's own words"
    assert ctx.session.fen() == fen_before
    assert ctx.pending is not None and ctx.pending.name == "new_game"


def test_new_game_on_an_untouched_board_just_runs():
    """The gate guards a game in progress, not the idea of one."""
    ctx = ToolContext(session=GameSession())
    client = TestClient(create_app(ctx))
    response = client.post("/api/game/new", json={"color": "black"})
    assert response.status_code == 200
    assert response.json()["state"]["player_color"] == "black"
    assert ctx.pending is None


def test_new_game_after_the_game_ended_just_runs():
    ctx = developed(ToolContext(session=GameSession()))
    ctx.session.resign("white")
    client = TestClient(create_app(ctx))
    assert client.post("/api/game/new").status_code == 200


def test_the_armed_new_game_carries_the_requested_color():
    ctx = developed(ToolContext(session=GameSession()))
    client = TestClient(create_app(ctx))
    client.post("/api/game/new", json={"color": "black"})
    assert ctx.pending.args == {"player_color": "black"}


def test_a_random_color_is_resolved_before_it_is_armed(monkeypatch):
    """Resolved by the endpoint, exactly as before, so the op the player confirms
    is the game they were asked about — not another roll."""
    monkeypatch.setattr("chessapp.api.random.choice", lambda options: "black")
    ctx = developed(ToolContext(session=GameSession()))
    client = TestClient(create_app(ctx))
    client.post("/api/game/new")
    assert ctx.pending.args == {"player_color": "black"}


def test_confirming_runs_the_armed_new_game():
    ctx = developed(ToolContext(session=GameSession()))
    client = TestClient(create_app(ctx))
    client.post("/api/game/new", json={"color": "white"})

    body = client.post("/api/game/confirm", json={"confirm": True}).json()

    assert body["op"] == "new_game"
    assert body["confirmed"] is True
    assert body["state"]["fen"] == START_FEN
    assert ctx.session.move_history() == []
    assert ctx.pending is None, "the op never survives its answering turn"


def test_cancelling_disarms_the_pending_op():
    ctx = developed(ToolContext(session=GameSession()))
    client = TestClient(create_app(ctx))
    client.post("/api/game/new", json={"color": "white"})
    fen_before = ctx.session.fen()

    body = client.post("/api/game/confirm", json={"confirm": False}).json()

    assert body["op"] == "new_game"
    assert body["confirmed"] is False
    assert body["state"]["fen"] == fen_before
    assert ctx.session.fen() == fen_before
    assert ctx.pending is None


def test_confirming_nothing_is_409():
    ctx = ToolContext(session=GameSession())
    client = TestClient(create_app(ctx))
    response = client.post("/api/game/confirm", json={"confirm": True})
    assert response.status_code == 409


def test_resign_mid_game_asks_first():
    ctx = developed(ToolContext(session=GameSession()))
    client = TestClient(create_app(ctx))

    response = client.post("/api/game/resign", json={})

    assert response.status_code == 409
    assert response.json()["op"] == "resign"
    assert not ctx.session.is_game_over(), "gated: it asks before it ends the game"


def test_confirming_a_resignation_ends_the_game():
    ctx = developed(ToolContext(session=GameSession()))
    client = TestClient(create_app(ctx))
    client.post("/api/game/resign", json={})

    body = client.post("/api/game/confirm", json={"confirm": True}).json()

    assert ctx.session.is_game_over()
    assert body["state"]["game_over"] is True
    assert body["state"]["outcome"]["termination"] == "resignation"


def test_a_resignation_resigns_the_player_not_the_side_to_move():
    """The same rule the tool derives (trace review, finding 8): the side to move
    is only coincidentally the player."""
    ctx = ToolContext(session=GameSession(player_color="black"))
    for san in ("e4", "e5", "Nf3"):
        ctx.session.submit_move(san)
    client = TestClient(create_app(ctx))

    client.post("/api/game/resign", json={})
    client.post("/api/game/confirm", json={"confirm": True})

    assert ctx.session.outcome().winner == "white", "black (the player) resigned"


def test_resigning_a_finished_game_is_still_409():
    ctx = ToolContext(session=GameSession())
    client = TestClient(create_app(ctx))
    client.post("/api/game/resign", json={})  # fresh board: the gate stands aside
    assert client.post("/api/game/resign", json={}).status_code == 409


def test_undo_is_not_gated():
    """A takeback throws nothing away — it keeps its direct endpoint."""
    ctx = developed(ToolContext(session=GameSession()))
    client = TestClient(create_app(ctx))
    body = client.post("/api/game/undo", json={}).json()
    assert body["undone"] == ["Nc6"]
    assert ctx.pending is None


def test_a_button_armed_op_is_confirmed_by_a_typed_yes():
    """One gate, one pending op: which surface armed it and which answered it are
    independent."""
    client, _, ctx = agent_client(narrations=("Fresh meat.",))
    developed(ctx)
    client.post("/api/game/new", json={"color": "white"})
    assert ctx.pending is not None

    client.post("/api/command", json={"text": "yes"})

    assert ctx.session.move_history() == []
    assert ctx.pending is None


def test_a_spoken_ask_is_confirmed_by_the_button():
    client, _, ctx = agent_client(
        AgentResponse(
            text="you sure?", tool_calls=(ToolCall(name="new_game", args={}),)
        )
    )
    developed(ctx)
    client.post("/api/command", json={"text": "new game"})
    assert ctx.pending is not None and ctx.pending.name == "new_game"

    body = client.post("/api/game/confirm", json={"confirm": True}).json()

    assert body["confirmed"] is True
    assert ctx.session.move_history() == []


def test_the_confirmed_op_broadcasts_the_new_state():
    ctx = developed(ToolContext(session=GameSession()))
    client = TestClient(create_app(ctx))
    client.post("/api/game/new", json={"color": "white"})
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()  # connect snapshot
        client.post("/api/game/confirm", json={"confirm": True})
        message = ws.receive_json()
    assert message["state"]["fen"] == START_FEN


def test_a_cancelled_op_broadcasts_nothing():
    ctx = developed(ToolContext(session=GameSession()))
    client = TestClient(create_app(ctx))
    client.post("/api/game/new", json={"color": "white"})
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()
        client.post("/api/game/confirm", json={"confirm": False})
        # The next frame is the undo's, not a phantom one from the cancel.
        client.post("/api/game/undo", json={})
        assert ws.receive_json()["state"]["history"] == ["e4", "e5", "Nf3"]


# --- Direct mode is visible -------------------------------------------------


def test_settings_report_that_an_agent_is_available():
    client, _, _ = agent_client()
    assert client.get("/api/settings").json()["agent_available"] is True


def test_settings_report_direct_mode_without_a_brain():
    client = TestClient(create_app(ToolContext(session=GameSession())))
    assert client.get("/api/settings").json()["agent_available"] is False
