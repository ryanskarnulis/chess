"""Turn coordinator: the deterministic owner of the turn sequence.

`player_move → (observe) → engine_reply → (close)` is code's sequence, not the
model's. These tests pin the state machine directly — every phase transition,
every rejection — and then the same rules seen through the tool boundary, which
is the only way the agent can reach it.

Since the move flow split, the reply is *computed* in the background from the
moment the player's move lands and *applied* when the turn collects it, so the
observe beat and Stockfish overlap. That is a latency requirement, so it is
pinned here too: what the background may never do is touch the session.
"""

import threading

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

    def choose_move(self, session):
        raise AssertionError("engine must not be asked to think here")


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

    reply = coordinator.collect_engine_reply()
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
    reply = coordinator.collect_engine_reply()
    assert reply is not None
    assert coordinator.phase == TurnPhase.ENGINE_MOVE_APPLIED


def test_engine_calculating_is_the_phase_while_the_turn_waits_on_the_reply(session):
    """The phase exists to be *observed* while the turn is blocked on the reply
    (the UI's "Stockfish is calculating"), so it is set before the engine is
    asked, not after.

    The background computation starts earlier — the moment the player's move
    lands — and finishes without the phase moving, because during it the turn
    belongs to the observation beat. What this pins is the ask `collect` makes
    itself: no engine at move time, so nothing was started in the background."""
    seen = []
    ctx = ToolContext(session=session)
    coordinator = TurnCoordinator(ctx)

    class SlowEngine:
        def choose_move(self, session):
            seen.append(coordinator.phase)
            return "e7e5"

    coordinator.apply_player_move("e4")
    ctx.engine = SlowEngine()  # attached mid-turn: nothing is pending
    coordinator.collect_engine_reply()
    assert seen == [TurnPhase.ENGINE_CALCULATING]


def test_turn_id_increments_across_exchanges(session):
    class ScriptedEngine:
        """Plays out 1.e4 e5 2.Nf3 Nc6 3.Bc4 Nf6 from the black side."""

        def __init__(self):
            self.replies = ["e7e5", "b8c6", "g8f6"]

        def choose_move(self, session):
            return self.replies.pop(0)

    coordinator = TurnCoordinator(ToolContext(session=session, engine=ScriptedEngine()))
    for expected, move in enumerate(["e4", "Nf3", "Bc4"], start=1):
        assert coordinator.turn_id == expected
        player, reply = coordinator.play_exchange(move)
        assert player.legal and reply is not None and reply.legal
    assert coordinator.turn_id == 4


# --- rejections: an action that doesn't belong in the phase ------------------


def test_engine_reply_before_a_player_move_is_rejected(coordinator):
    with pytest.raises(TurnStateError):
        coordinator.collect_engine_reply()
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
    coordinator.collect_engine_reply()
    with pytest.raises(TurnStateError):
        coordinator.collect_engine_reply()


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
    assert coordinator.collect_engine_reply() is None
    assert coordinator.phase == TurnPhase.ENGINE_MOVE_APPLIED


# --- the reply computed in the background ------------------------------------
#
# Latency is the acceptance criterion for the observe beat: a plain move must not
# feel slower for having gained a reaction. So the engine starts thinking the
# moment the player's move lands and the turn collects the answer after the
# narration — the two overlap. The rule that makes it safe is that the
# background touches nothing: it works from a copy of the position and only the
# collecting thread ever submits a move.


class BlockingEngine:
    """Engine double that parks inside `choose_move` until released."""

    def __init__(self, reply_uci: str = "e7e5") -> None:
        self.reply_uci = reply_uci
        self.entered = threading.Event()
        self.release = threading.Event()
        self.positions: list[str] = []

    def choose_move(self, session):
        self.positions.append(session.fen())
        self.entered.set()
        assert self.release.wait(timeout=5), "test never released the engine"
        return self.reply_uci


def test_the_engine_starts_thinking_as_soon_as_the_player_move_lands(session):
    """The overlap itself: `apply_player_move` returns with Stockfish already
    working, which is the whole reason the observation beat is free."""
    engine = BlockingEngine()
    coordinator = TurnCoordinator(ToolContext(session=session, engine=engine))

    coordinator.apply_player_move("e4")

    assert engine.entered.wait(timeout=5), "the reply was not started in the background"
    assert coordinator.phase == TurnPhase.PLAYER_MOVE_APPLIED
    assert session.move_history() == ["e4"], "the background must not touch the session"
    engine.release.set()
    reply = coordinator.collect_engine_reply()
    assert reply is not None and reply.san == "e5"
    assert session.move_history() == ["e4", "e5"]


def test_the_background_computation_reads_a_copy_of_the_position(session):
    """It is handed the position, not the session: nothing it does can be a
    mutation, which is what makes it safe to run while the narrator talks."""
    engine = BlockingEngine()
    coordinator = TurnCoordinator(ToolContext(session=session, engine=engine))
    coordinator.apply_player_move("e4")
    assert engine.entered.wait(timeout=5)
    engine.release.set()
    coordinator.collect_engine_reply()
    assert engine.positions == [
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
    ]


def test_collect_applies_the_reply_the_background_computed(coordinator, session):
    player = coordinator.apply_player_move("e4")
    assert player.legal is True
    reply = coordinator.collect_engine_reply()
    assert reply is not None and reply.uci == "e7e5"
    assert coordinator.phase == TurnPhase.ENGINE_MOVE_APPLIED
    assert session.move_history() == ["e4", "e5"]


def test_collect_owes_nothing_once_the_engine_is_gone(session):
    """Whether a reply is owed is read off the live context at collect time, not
    remembered from when the computation started — so an engine that went away
    mid-turn drops the reply instead of applying a stale one."""
    engine = BlockingEngine()
    ctx = ToolContext(session=session, engine=engine)
    coordinator = TurnCoordinator(ctx)
    coordinator.apply_player_move("e4")
    assert engine.entered.wait(timeout=5)
    ctx.engine = None
    engine.release.set()

    assert coordinator.collect_engine_reply() is None
    assert coordinator.phase == TurnPhase.ENGINE_MOVE_APPLIED
    assert session.move_history() == ["e4"]


def test_collect_recomputes_when_the_board_moved_under_the_computation(session):
    """The board changed while the engine was thinking, so its answer is about a
    position that no longer exists. It is discarded and the reply is computed
    from the board as it stands — never applied blind, which would be a move
    from the wrong position."""
    engine = FakeEngine("e7e5")
    ctx = ToolContext(session=session, engine=engine)
    coordinator = TurnCoordinator(ctx)
    coordinator.apply_player_move("e4")  # computed against 1.e4

    # Someone took the move back mid-turn: e7e5 is not even black's to play now.
    session.undo(1)
    engine.reply_uci = "e2e4"

    reply = coordinator.collect_engine_reply()
    assert reply is not None and reply.legal is True
    assert reply.san == "e4", "the reply must be legal for the board as it stands"
    assert session.move_history() == ["e4"]


def test_a_second_begin_while_one_is_pending_is_rejected(session):
    """Two computations for one turn would mean two candidate replies, and
    nothing sensible to do with the loser."""
    engine = BlockingEngine()
    coordinator = TurnCoordinator(ToolContext(session=session, engine=engine))
    coordinator.apply_player_move("e4")
    assert engine.entered.wait(timeout=5)
    with pytest.raises(TurnStateError):
        coordinator.begin_engine_reply()
    engine.release.set()


# --- abandoning a turn: any non-move mutation ends it ------------------------


def test_abandon_turn_resets_the_machine_and_bumps_the_turn_id(coordinator, session):
    coordinator.apply_player_move("e4")
    assert coordinator.phase == TurnPhase.PLAYER_MOVE_APPLIED

    coordinator.abandon_turn()

    assert coordinator.phase == TurnPhase.AWAITING_PLAYER
    assert coordinator.turn_id == 2, "the abandoned turn is still a boundary"
    # The player's move stands on the board — abandoning a turn is not an undo.
    assert session.move_history() == ["e4"]
    # And the machine is open for business: a move is accepted straight away.
    assert coordinator.apply_player_move("e5").legal is True


def test_abandon_turn_discards_the_pending_computation(session):
    """The reply the engine was working out was for a position the abandoning
    mutation just threw away, so it must never reach the board."""
    engine = BlockingEngine()
    ctx = ToolContext(session=session, engine=engine)
    coordinator = TurnCoordinator(ctx)
    coordinator.apply_player_move("e4")
    assert engine.entered.wait(timeout=5)

    coordinator.abandon_turn()
    engine.release.set()  # it finishes; its answer goes nowhere

    session.undo(1)  # what the abandoning mutation would have done
    assert session.move_history() == []
    assert coordinator.phase == TurnPhase.AWAITING_PLAYER


def test_abandon_turn_between_turns_changes_nothing(coordinator):
    """It is called before every non-move mutation, most of which happen while
    no turn is open — so on an idle machine it is a no-op, and the turn id keeps
    counting real turns."""
    coordinator.abandon_turn()
    assert coordinator.phase == TurnPhase.AWAITING_PLAYER
    assert coordinator.turn_id == 1


# --- the command window: one destructive op per user interaction -------------
#
# The phases already spend a turn's move budget: one player move, one engine
# reply, and a second of either is refused. The destructive ops have no such
# home, because they *end* the turn they run in — so the budget for them is
# scoped to the command instead, one user interaction wide. Only the pipeline
# opens a window, because only the pipeline can chain several dispatches inside
# one interaction (the brain loop). Everything else — the board buttons, the
# confirm endpoint, MCP — is one dispatch per interaction by construction and
# stays unconstrained.


def test_the_destructive_budget_allows_one_op_per_command(coordinator):
    coordinator.begin_command()
    coordinator.require_destructive_budget()  # nothing spent yet

    coordinator.record_destructive_op()

    with pytest.raises(TurnStateError) as excinfo:
        coordinator.require_destructive_budget()
    # The message is read by the model, so it has to say what to do instead.
    assert "one per turn" in str(excinfo.value)


def test_the_budget_is_not_enforced_outside_a_command_window(coordinator):
    """A surface that dispatches once per interaction cannot spend a budget
    twice, so it is never asked to hold one — MCP included, where a client may
    legitimately start game after game across a session."""
    coordinator.record_destructive_op()
    coordinator.require_destructive_budget()

    coordinator.begin_command()
    coordinator.record_destructive_op()
    coordinator.end_command()

    coordinator.require_destructive_budget()
    coordinator.record_destructive_op()
    coordinator.require_destructive_budget()


def test_a_new_command_window_starts_with_a_fresh_budget(coordinator):
    coordinator.begin_command()
    coordinator.record_destructive_op()

    coordinator.begin_command()

    coordinator.require_destructive_budget()


def test_the_spent_budget_survives_abandoning_the_turn(coordinator):
    """The load-bearing one. `new_game` and `resign` abandon the open turn as
    part of doing their work, so a budget the phase machine owned would reset
    itself on the way out and enforce nothing. The window is command state, not
    turn state."""
    coordinator.begin_command()
    coordinator.record_destructive_op()

    coordinator.abandon_turn()

    with pytest.raises(TurnStateError):
        coordinator.require_destructive_budget()


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
