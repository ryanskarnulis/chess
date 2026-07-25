"""Turn coordinator: the deterministic owner of the turn sequence.

`player_move → (observe) → engine_reply → (close)` is code's sequence, not the
model's. These tests pin the state machine directly — every phase transition,
every rejection — and then the same rules seen through the tool boundary, which
is the only way the agent can reach it.
"""

import pytest

from chessapp.coordinator import TurnCoordinator, TurnPhase, TurnStateError
from chessapp.game import GameSession
from chessapp.tools import ToolContext, build_registry
from fakes import FakeEngine

# White to move, Qxf7# available (scholar's mate pattern).
WHITE_MATE_IN_1 = "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 0 1"


class MustNotPlay:
    """Engine double that fails the test if asked to move at all."""

    def play_move(self, session):
        raise AssertionError("engine must not move here")


@pytest.fixture
def session():
    return GameSession()


@pytest.fixture
def ctx(session):
    return ToolContext(session=session, engine=FakeEngine())


@pytest.fixture
def coordinator(ctx):
    return TurnCoordinator(ctx)


# --- the happy path, step by step -------------------------------------------


def test_starts_awaiting_player_on_turn_one(coordinator):
    assert coordinator.phase == TurnPhase.AWAITING_PLAYER
    assert coordinator.turn_id == 1


def test_phases_advance_through_the_full_sequence(coordinator, session):
    player = coordinator.apply_player_move("e4")
    assert player.legal is True
    assert coordinator.phase == TurnPhase.PLAYER_MOVE_APPLIED

    coordinator.begin_observation()
    assert coordinator.phase == TurnPhase.AGENT_OBSERVING

    reply = coordinator.engine_reply()
    assert reply is not None and reply.san == "e5"
    assert coordinator.phase == TurnPhase.ENGINE_MOVE_APPLIED

    coordinator.complete_turn()
    # Completing rolls straight into the next turn: there is no idle state
    # between two turns, only the boundary the turn id counts.
    assert coordinator.phase == TurnPhase.AWAITING_PLAYER
    assert coordinator.turn_id == 2
    assert session.move_history() == ["e4", "e5"]


def test_observation_is_skippable(coordinator):
    coordinator.apply_player_move("e4")
    reply = coordinator.engine_reply()
    assert reply is not None
    assert coordinator.phase == TurnPhase.ENGINE_MOVE_APPLIED


def test_engine_calculating_is_the_phase_while_the_engine_thinks(ctx):
    """The phase exists to be *observed* mid-calculation (the UI's "Stockfish is
    calculating"), so it has to be set before the engine is asked, not after."""
    seen = []
    coordinator = TurnCoordinator(ctx)

    class SlowEngine:
        def play_move(self, session):
            seen.append(coordinator.phase)
            return session.submit_move("e7e5")

    ctx.engine = SlowEngine()
    coordinator.apply_player_move("e4")
    coordinator.engine_reply()
    assert seen == [TurnPhase.ENGINE_CALCULATING]


def test_turn_id_increments_across_exchanges(session):
    class ScriptedEngine:
        """Plays out 1.e4 e5 2.Nf3 Nc6 3.Bc4 Nf6 from the black side."""

        def __init__(self):
            self.replies = ["e7e5", "b8c6", "g8f6"]

        def play_move(self, session):
            return session.submit_move(self.replies.pop(0))

    coordinator = TurnCoordinator(ToolContext(session=session, engine=ScriptedEngine()))
    for expected, move in enumerate(["e4", "Nf3", "Bc4"], start=1):
        assert coordinator.turn_id == expected
        player, reply = coordinator.play_exchange(move)
        assert player.legal and reply is not None and reply.legal
    assert coordinator.turn_id == 4


# --- rejections: an action that doesn't belong in the phase ------------------


def test_engine_reply_before_a_player_move_is_rejected(coordinator):
    with pytest.raises(TurnStateError):
        coordinator.engine_reply()
    assert coordinator.phase == TurnPhase.AWAITING_PLAYER


def test_second_player_move_mid_turn_is_rejected(coordinator, session):
    coordinator.apply_player_move("e4")
    with pytest.raises(TurnStateError):
        coordinator.apply_player_move("d4")
    assert session.move_history() == ["e4"]
    assert coordinator.phase == TurnPhase.PLAYER_MOVE_APPLIED


def test_complete_turn_before_a_player_move_is_rejected(coordinator):
    with pytest.raises(TurnStateError):
        coordinator.complete_turn()
    assert coordinator.turn_id == 1


def test_observation_only_follows_a_player_move(coordinator):
    with pytest.raises(TurnStateError):
        coordinator.begin_observation()


def test_second_engine_reply_in_one_turn_is_rejected(coordinator):
    coordinator.apply_player_move("e4")
    coordinator.engine_reply()
    with pytest.raises(TurnStateError):
        coordinator.engine_reply()


def test_turn_state_error_is_a_value_error():
    """Deliberate: `ToolRegistry.dispatch` already turns ValueError into
    `{"ok": False, "error": ...}`, so a turn-state rejection reaches the agent
    through the same one validation path as a schema or domain failure."""
    assert issubclass(TurnStateError, ValueError)


# --- a turn that closes early ------------------------------------------------


def test_complete_turn_closes_a_turn_with_no_engine_reply(session):
    coordinator = TurnCoordinator(ToolContext(session=session))
    coordinator.apply_player_move("e4")
    coordinator.complete_turn()
    assert coordinator.phase == TurnPhase.AWAITING_PLAYER
    assert coordinator.turn_id == 2


def test_complete_turn_closes_from_observation(session):
    coordinator = TurnCoordinator(ToolContext(session=session))
    coordinator.apply_player_move("e4")
    coordinator.begin_observation()
    coordinator.complete_turn()
    assert coordinator.turn_id == 2


def test_complete_turn_cannot_skip_a_pending_engine_reply(coordinator, session):
    """The whole point of the coordinator: nothing can close a turn out from
    under the engine's move."""
    coordinator.apply_player_move("e4")
    with pytest.raises(TurnStateError):
        coordinator.complete_turn()
    assert coordinator.turn_id == 1
    assert session.move_history() == ["e4"]


def test_illegal_player_move_leaves_the_turn_untouched(coordinator, session):
    before = session.fen()
    result = coordinator.apply_player_move("e5")
    assert result.legal is False
    assert coordinator.phase == TurnPhase.AWAITING_PLAYER
    assert coordinator.turn_id == 1
    assert session.fen() == before


def test_game_ending_player_move_completes_the_turn_with_no_reply():
    session = GameSession(WHITE_MATE_IN_1)
    ctx = ToolContext(session=session, engine=MustNotPlay())
    coordinator = TurnCoordinator(ctx)
    result = coordinator.apply_player_move("Qxf7#")
    assert result.legal is True
    assert session.is_game_over() is True
    # Nothing is left to wait for, so the turn is already closed.
    assert coordinator.phase == TurnPhase.AWAITING_PLAYER
    assert coordinator.turn_id == 2


def test_engine_reply_is_none_without_an_engine(session):
    coordinator = TurnCoordinator(ToolContext(session=session))
    coordinator.apply_player_move("e4")
    assert coordinator.engine_reply() is None
    assert coordinator.phase == TurnPhase.ENGINE_MOVE_APPLIED


# --- play_exchange: the whole sequence, atomically ---------------------------


def test_play_exchange_runs_the_whole_sequence(coordinator, session):
    player, reply = coordinator.play_exchange("e4")
    assert player.legal is True
    assert reply is not None and reply.san == "e5"
    assert session.move_history() == ["e4", "e5"]
    assert coordinator.phase == TurnPhase.AWAITING_PLAYER
    assert coordinator.turn_id == 2


def test_play_exchange_illegal_move_consumes_no_turn(coordinator, session):
    player, reply = coordinator.play_exchange("e5")
    assert player.legal is False
    assert reply is None
    assert coordinator.phase == TurnPhase.AWAITING_PLAYER
    assert coordinator.turn_id == 1
    assert session.move_history() == []


def test_play_exchange_without_an_engine_completes_the_turn(session):
    coordinator = TurnCoordinator(ToolContext(session=session))
    player, reply = coordinator.play_exchange("e4")
    assert player.legal is True
    assert reply is None
    assert coordinator.turn_id == 2


def test_play_exchange_on_a_game_ending_move_asks_for_no_reply():
    session = GameSession(WHITE_MATE_IN_1)
    coordinator = TurnCoordinator(ToolContext(session=session, engine=MustNotPlay()))
    player, reply = coordinator.play_exchange("Qxf7#")
    assert player.legal is True
    assert reply is None
    assert coordinator.turn_id == 2


def test_play_exchange_mid_turn_is_rejected(coordinator):
    coordinator.apply_player_move("e4")
    with pytest.raises(TurnStateError):
        coordinator.play_exchange("d4")


# --- the engine's opening move (player takes black) -------------------------


def test_engine_opening_move_does_not_consume_a_turn(session):
    session.new_game("black")
    ctx = ToolContext(session=session, engine=FakeEngine("e2e4"))
    coordinator = TurnCoordinator(ctx)
    reply = coordinator.engine_opening_move()
    assert reply is not None and reply.san == "e4"
    # The player's first turn hasn't happened yet, so the turn id stands.
    assert coordinator.phase == TurnPhase.AWAITING_PLAYER
    assert coordinator.turn_id == 1


def test_engine_opening_move_without_an_engine_is_none(session):
    session.new_game("black")
    coordinator = TurnCoordinator(ToolContext(session=session))
    assert coordinator.engine_opening_move() is None
    assert coordinator.turn_id == 1


def test_engine_owes_no_opening_move_when_the_player_has_white(session):
    """Whose the opening move is, is session state — not something each caller
    re-derives before reaching for the engine."""
    coordinator = TurnCoordinator(ToolContext(session=session, engine=MustNotPlay()))
    assert coordinator.engine_opening_move() is None


def test_engine_opening_move_mid_turn_is_rejected(coordinator):
    coordinator.apply_player_move("e4")
    with pytest.raises(TurnStateError):
        coordinator.engine_opening_move()


# --- the live context: resume_game swaps the session -------------------------


def test_reads_the_session_off_the_context_each_call(ctx):
    """`resume_game` replaces `ctx.session`; the coordinator must follow the
    context, never a session captured at construction."""
    coordinator = TurnCoordinator(ctx)
    ctx.session = GameSession()
    coordinator.play_exchange("d4")
    assert ctx.session.move_history() == ["d4", "e5"]


# --- through the tool boundary ----------------------------------------------


def test_make_move_tool_drives_the_shared_coordinator(session):
    ctx = ToolContext(session=session, engine=FakeEngine())
    coordinator = TurnCoordinator(ctx)
    registry = build_registry(ctx, coordinator)
    result = registry.dispatch("make_move", {"move": "e4"})
    assert result["engine_move"] == {"san": "e5", "uci": "e7e5"}
    assert coordinator.turn_id == 2


def test_turn_state_rejection_reaches_the_agent_as_result_data(session):
    """The agent never sees an exception: drive the machine mid-turn and the
    refused `make_move` comes back as an ordinary error result."""
    ctx = ToolContext(session=session, engine=FakeEngine())
    coordinator = TurnCoordinator(ctx)
    registry = build_registry(ctx, coordinator)
    coordinator.apply_player_move("e4")
    result = registry.dispatch("make_move", {"move": "d4"})
    assert result["ok"] is False
    assert result["error"]
    assert session.move_history() == ["e4"]


def test_build_registry_makes_its_own_coordinator_when_given_none(session):
    registry = build_registry(ToolContext(session=session, engine=FakeEngine()))
    first = registry.dispatch("make_move", {"move": "e4"})
    assert first["engine_move"] == {"san": "e5", "uci": "e7e5"}
    # Turn after turn, with nobody handing it a coordinator.
    second = registry.dispatch("make_move", {"move": "d4"})
    assert second["legal"] is True


def test_engine_reply_is_not_a_callable_tool(session):
    """The sequence is code's, not the model's: there is no tool that makes the
    engine move, so the model can never block, skip, or reorder the reply."""
    registry = build_registry(ToolContext(session=session, engine=FakeEngine()))
    names = {d["function"]["name"] for d in registry.definitions()}
    assert not names & {"engine_reply", "request_engine_reply", "apply_engine_move"}
