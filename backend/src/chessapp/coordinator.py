"""Turn coordinator: deterministic code owns the shape of a turn.

A turn is `player_move → (observe) → engine_reply → (close)`, and every step of
that sequence is this module's decision, never the model's. The engine's reply
used to live inside the `make_move` tool and inside `/api/game/move` — two
copies of the same rule, both reachable only by *making a move*, so there was no
point in a turn at which anything could stand between the player's move and
Stockfish's answer. The coordinator is that point. It is deliberately not a
tool: a model that could call `request_engine_reply` could also fail to, and a
stalled 12B would stall a game the app must be able to play with the LLM off.

The phases are explicit so an action that doesn't belong in the current one can
be *rejected* rather than quietly done twice. `TurnStateError` subclasses
`ValueError`, which is the one interesting design choice here: `ToolRegistry.
dispatch` already converts `ValueError` into `{"ok": False, "error": ...}`, so a
turn-state rejection reaches the agent as ordinary result data on the same road
as a schema failure or an illegal move — one validation layer, three kinds of
"no" — while trusted callers (the API) catch it and answer 409.

`begin_observation` marks the beat where Glitch reacts to the verified player
move, and it is skippable by construction — collecting the reply is legal with
or without it, so a missing, slow, or switched-off model can never hold up the
engine.

**The reply is computed in the background and applied later.** The moment a
legal player move lands, `apply_player_move` starts the engine thinking on a
*copy* of the position (`begin_engine_reply`); `collect_engine_reply` joins that
work and submits the answer through the session. Latency is the observe beat's
acceptance criterion — a plain move must not feel slower for having gained a
reaction — and this is what pays for it: the narration and Stockfish overlap
instead of queueing. The background thread never touches the session, which is
the whole reason it is safe to leave running while the narrator talks; the
collecting thread checks that the position it computed from is still the
position on the board and recomputes if not.

**The destructive ops are budgeted per *command*, not per turn.** One player
move and one engine move per turn are already the phases' business — a second of
either is refused — but `new_game` and `resign` abandon the turn they run in, so
a flag scoped to the turn would reset itself and enforce nothing. The window a
`begin_command`/`end_command` pair opens is one *user interaction* wide and
independent of the turn ids, and inside it a second destructive op is refused.
Only the command pipeline opens one, because only the pipeline can chain several
dispatches inside a single interaction (the brain loop); the board buttons, the
confirm endpoint and MCP dispatch once per interaction by construction, so they
stay unconstrained — an MCP client may start game after game across a session.

**The phase is observable, and only through one door.** Every transition goes
through `_enter`, which is what `on_phase` is told — so the live progress the UI
shows (`progress.py`, audit item 19) reads the machine instead of being narrated
alongside it. A hand-placed "now we are calculating" would be a second copy of
this sequence, free to drift from the real one; the same reason `mutations` is a
`board_version` delta rather than a tally. Observing is decoration, so an
observer that raises is swallowed: nothing about watching a turn may cost one.

State lives on the coordinator, board truth stays in `GameSession`: the phases
say what may happen next, never what is on the board. `ctx.session` and
`ctx.engine` are read live on every call because `resume_game` swaps the session
object on the context.

**Every reply is chosen by one move source** (`_move_source`, #471). Whether a
reply is owed and which move it is are both asked of it, on every route: the
background collect, its synchronous fallback, the settle and so `play_exchange`.
Stockfish is the only source today; the Glitch tier puts its mover behind the
same seam (`docs/glitch-difficulty.md`). A source only *chooses*: the move still
enters the game through `session.submit_move`, here.
"""

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Protocol

from chessapp.game import GameSession, MoveResult

if TYPE_CHECKING:  # tools.py imports this module; don't import it back.
    from chessapp.tools import ToolContext

logger = logging.getLogger(__name__)


class MoveSource(Protocol):
    """What chooses the reply's move: `EnginePlayer` today.

    It returns a UCI string and nothing else. It never submits the move — the
    coordinator does, through the session's legality gate — so a source can
    be wrong without the game being wrong.
    """

    def choose_move(self, session: GameSession) -> str: ...


class TurnPhase(StrEnum):
    """Where a turn is. A `StrEnum` because these values are headed for a trace
    record and a UI progress line, where they need to be plain strings."""

    AWAITING_PLAYER = "awaiting_player"
    PLAYER_MOVE_APPLIED = "player_move_applied"
    AGENT_OBSERVING = "agent_observing"
    ENGINE_CALCULATING = "engine_calculating"
    ENGINE_MOVE_APPLIED = "engine_move_applied"
    COMPLETED = "completed"


class TurnStateError(ValueError):
    """An action that doesn't belong in the current phase.

    A `ValueError` on purpose — see the module docstring: it is what lets one
    rejection serve the agent (as `dispatch` error data) and the trusted API
    paths (as a 409) without either learning a new failure shape.
    """


@dataclass(frozen=True)
class ReplySettlement:
    """How an owed reply was settled: the move the engine played, or what
    killed it.

    `reply` is None both when the engine had nothing to play (no engine, the
    player's move ended the game) and when it died; `failure` tells the two
    apart — empty unless Stockfish raised, in which case the turn is left
    where the coordinator put it, with the reply still owed (#284).
    """

    reply: MoveResult | None = None
    failure: str = ""


def failure_name(exc: BaseException) -> str:
    """One failure, as the short string a record can carry: class and message.

    The trace's `provider_failure` names a *kind* from a vocabulary the brain
    owns; nothing owns a vocabulary for a dying Stockfish or for whatever else
    escapes a turn, so the exception names itself. Class alone when it carries
    no message — `EngineTerminatedError` is already the whole story.
    """
    detail = str(exc).strip()
    return f"{type(exc).__name__}: {detail}" if detail else type(exc).__name__


class _PendingReply:
    """One engine reply being computed off the main thread.

    It holds the position it was asked about, because by the time anyone wants
    the answer the board may have moved on (an undo mid-turn) — and an answer to
    a position that no longer exists is not a move, it is a bug waiting to be
    applied. The thread only ever reads its own `GameSession` copy, so nothing
    here mutates game state.
    """

    def __init__(self, session: GameSession) -> None:
        self.fen = session.fen()
        # A replayed copy, so the source sees the game's moves and not just the
        # position: Stockfish reads only the FEN, a mover reads the moves too.
        self._probe = GameSession.from_dict(session.to_dict())
        self.uci: str | None = None
        self._thread: threading.Thread | None = None

    def start(self, source: MoveSource) -> None:
        probe = self._probe

        def _compute() -> None:
            try:
                self.uci = source.choose_move(probe)
            except Exception:
                # A failed background computation is simply no answer: the
                # collector falls back to asking the engine itself, so the
                # failure surfaces there, where today's error semantics already
                # live (audit item 20 owns defining better ones).
                self.uci = None

        self._thread = threading.Thread(
            target=_compute, name="engine-reply", daemon=True
        )
        self._thread.start()

    def result_for(self, fen: str) -> str | None:
        """The computed move, or None when there isn't a usable one — the work
        failed, or it was about a different position than the board now holds."""
        if self._thread is not None:
            self._thread.join()
        return self.uci if fen == self.fen else None


class TurnCoordinator:
    """The turn sequence, owned by code.

    Holds the shared `ToolContext` (not a session or an engine): the context is
    the source of truth for both, and `resume_game` replaces the session on it
    mid-game.
    """

    def __init__(self, ctx: "ToolContext") -> None:
        self._ctx = ctx
        self._turn_id = 1
        self._phase = TurnPhase.AWAITING_PLAYER
        # Told every transition, by `_enter` and nothing else. Assigned after
        # construction (`create_app` does it) so the machine can be built before
        # whatever watches it; None is the ordinary unwatched state.
        self.on_phase: Callable[[TurnPhase], None] | None = None
        self._pending: _PendingReply | None = None
        # The command window (see the module docstring). Deliberately not phase
        # state: the ops it budgets abandon the turn as part of running.
        self._command_open = False
        self._destructive_spent = False
        # The save names this command has written itself (`record_save`).
        self._saved_this_command: set[str] = set()
        # The reply `settle_owed_reply` last settled, until someone takes it
        # (`take_settlement`) or a new command begins (#365).
        self._settlement: ReplySettlement | None = None
        # Told how long each settle waited on the engine, in whole ms — the
        # request's `engine` span (#290). Assigned by `create_app`, like
        # `on_phase`.
        self.on_engine_time: Callable[[int], None] | None = None

    @property
    def phase(self) -> TurnPhase:
        return self._phase

    def _move_source(self) -> MoveSource | None:
        """The one thing that chooses the reply, or None when no reply is ever
        owed. Read live, like `ctx.engine`, because the context changes under
        the coordinator. Stockfish for now; the `glitch` tier picks its mover
        here (#473), so every route changes source at once.
        """
        return self._ctx.engine

    @property
    def turn_id(self) -> int:
        """Counts turn boundaries, from 1. Bumped when a turn completes, so a
        duplicated move is visible as two mutations under one id."""
        return self._turn_id

    def _enter(self, phase: TurnPhase) -> None:
        """Move to `phase` and tell the observer. The only writer of `_phase`.

        One door so that what is watched and what is enforced are the same
        thing. The observer runs *after* the move, so anything it triggers
        (opening the observe beat, sending a progress event) reads a machine
        that is already where it says it is.
        """
        self._phase = phase
        if self.on_phase is None:
            return
        try:
            self.on_phase(phase)
        except Exception:
            logger.warning("phase_observer_failed", exc_info=True)

    def _require(self, action: str, *allowed: TurnPhase) -> None:
        if self._phase not in allowed:
            expected = ", ".join(allowed)
            raise TurnStateError(
                f"cannot {action} while {self._phase}: expected {expected}"
            )

    def apply_player_move(self, move: str) -> MoveResult:
        """Submit the player's move through the session's legality gate.

        An illegal move is a *result*, not a state change: the turn stays open
        and awaiting a player move, exactly as it would if nothing had been
        said. A legal move opens the rest of the sequence — unless it ended the
        game, in which case there is nothing left to wait for and the turn
        closes here.

        A legal move that leaves the game running also sets the engine thinking
        immediately, in the background. That is not the caller's business (no
        tool handler or endpoint has to ask for it) and it is not optional: it is
        the same rule as "the engine owes a reply", started as early as it can
        possibly be started so the observation beat costs nothing in wall clock.
        """
        self._require("apply a player move", TurnPhase.AWAITING_PLAYER)
        session = self._ctx.session
        if (
            self._move_source() is not None
            and not session.is_game_over()
            and session.turn != session.player_color
        ):
            # Against the engine the player owns one colour. A board left with
            # the engine to move under an awaiting phase is a bug somewhere
            # else, but it must never become the player moving the engine's
            # pieces (#329) — the legality gate alone cannot say whose piece a
            # legal move belongs to.
            raise TurnStateError(
                f"cannot apply a player move: it is {session.turn}'s move and "
                f"the player has {session.player_color}"
            )
        result = session.submit_move(move)
        if not result.legal:
            return result
        self._enter(TurnPhase.PLAYER_MOVE_APPLIED)
        if self._ctx.session.is_game_over():
            self.complete_turn()
            return result
        self.begin_engine_reply()
        return result

    def begin_engine_reply(self) -> None:
        """Start the engine computing its reply, off the main thread.

        A no-op when no reply is owed (no engine, or the game is over) — the same
        derivation `collect_engine_reply` makes, so neither the caller nor the
        model ever decides it. The phase deliberately does *not* move: the turn
        now belongs to the observation beat, and `engine_calculating` marks the
        point where the turn is actually *waiting* on Stockfish (see `collect`).
        """
        self._require("begin the engine's reply", TurnPhase.PLAYER_MOVE_APPLIED)
        if self._pending is not None:
            raise TurnStateError("the engine is already computing a reply")
        source = self._move_source()
        session = self._ctx.session
        if source is None or session.is_game_over():
            return
        pending = _PendingReply(session)
        pending.start(source)
        self._pending = pending

    def begin_observation(self) -> None:
        """Open the beat where the agent reacts to the verified player move.

        It is a phase rather than a callback so that the reaction is *optional*
        by construction: `collect_engine_reply` is legal straight from
        `player_move_applied` too, so skipping the model (verbosity low, no
        brain, a provider failure) skips only the words. The engine is already
        thinking either way.
        """
        self._require("begin observation", TurnPhase.PLAYER_MOVE_APPLIED)
        self._enter(TurnPhase.AGENT_OBSERVING)

    def mark_observation(self) -> bool:
        """Open the beat *if* a verified player move is waiting on one.

        The conditional form of `begin_observation`, and the one the app's two
        narration sites actually reach for. Both of them run on turns where no
        move landed — a question, a settings change, a refused op — and a
        reaction to nothing is not an observation; both also run on turns where
        the beat is already open. Neither is an error, so neither may raise.
        `begin_observation` stays strict: the rule is still that you cannot
        observe a move that hasn't been made.

        Returns whether it opened one, which is how a caller can tell "I am the
        observation" from "I am just talking".
        """
        if self._phase is not TurnPhase.PLAYER_MOVE_APPLIED:
            return False
        self.begin_observation()
        return True

    def collect_engine_reply(self) -> MoveResult | None:
        """Take the engine's answer and put it on the board.

        Returns None — advancing the turn all the same — when there is no engine
        or the game is already over: "the engine owes a reply" is derivable from
        the session, so nobody upstream has to work it out, and it is re-derived
        *here* rather than remembered from when the computation started.

        The pending background answer is used only if the board is still the
        board it was computed from. Otherwise (something moved under it, or the
        computation failed) it is discarded and the reply is computed here and
        now, which is also what happens when nothing was started at all. The
        `engine_calculating` phase is entered before either ask, which is the
        only ordering that lets an observer see it while it is true.
        """
        self._require(
            "collect the engine's reply",
            TurnPhase.PLAYER_MOVE_APPLIED,
            TurnPhase.AGENT_OBSERVING,
        )
        pending, self._pending = self._pending, None
        source = self._move_source()
        session = self._ctx.session
        if source is None or session.is_game_over():
            self._enter(TurnPhase.ENGINE_MOVE_APPLIED)
            return None
        self._enter(TurnPhase.ENGINE_CALCULATING)
        try:
            uci = pending.result_for(session.fen()) if pending is not None else None
            if uci is None:
                uci = source.choose_move(session)
        except Exception:
            # An engine that dies mid-calculation must not take the turn with
            # it. The player's move stands and the reply is still owed, so the
            # phase goes back to where that is true rather than staying parked
            # in `engine_calculating` — where `_require` refuses every ordinary
            # move for good and only an undo, a reset or a resume can dig the
            # machine out. Restored, the pipeline's own healing branches
            # (`api._play_move`'s close beat and the command convergence, both
            # of which collect from `player_move_applied`) make the next command
            # finish the turn: its move is refused as mid-turn, the owed reply
            # is played, and the game goes on — which is what
            # `docs/turn-coordinator.md` claimed all along and this makes true
            # (audit 2026-09-05, engine-failure recovery). The raise itself gets
            # no further than those branches: each turns it into a structured
            # outcome for a caller whose move is already on the board (#284).
            self._enter(TurnPhase.PLAYER_MOVE_APPLIED)
            raise
        # Every engine move still enters the game through the session's legality
        # gate — background computation changes when it is decided, never who
        # decides whether it is legal.
        reply = session.submit_move(uci)
        self._note_difficulty(session)
        self._enter(TurnPhase.ENGINE_MOVE_APPLIED)
        return reply

    def _note_difficulty(self, session: GameSession) -> None:
        """Tell the game the strength its engine move was played at (#460).
        Here because every engine move in the app comes through this class
        (`settle_engine_turn`), so no road onto the board can miss it; the
        session keeps only the first, the strength the game was played at."""
        session.note_difficulty(self._ctx.settings.snapshot())

    def abandon_turn(self) -> None:
        """Throw the open turn away and start a fresh one.

        Every non-move mutation — undo, new game, resignation, resuming a save —
        runs this first. Those all change (or replace) the position the open turn
        was about, so the turn's remaining steps are meaningless: any reply being
        computed is discarded, and the machine comes back out awaiting the
        player. It is not an undo — whatever is on the board stays there; the
        caller is the one about to change that.

        Between turns it is a no-op, so the turn id keeps counting real turns
        rather than every button press.
        """
        if self._phase is TurnPhase.AWAITING_PLAYER and self._pending is None:
            return
        # The thread cannot be cancelled, so it is simply dropped: it finishes
        # against its own copy of a position nobody cares about any more and its
        # answer goes nowhere. Waiting for it would make an undo pay for a
        # calculation it just made irrelevant.
        self._pending = None
        self._enter(TurnPhase.COMPLETED)
        self._turn_id += 1
        self._enter(TurnPhase.AWAITING_PLAYER)

    def begin_command(self) -> None:
        """Open a command window: one user interaction, one destructive op.

        The command pipeline calls this, and it is the only caller by design —
        it is the one surface that can chain several dispatches inside a single
        interaction (the brain loop), which is the only way a budget can be
        spent twice. The board buttons, `/api/game/confirm`, `/api/game/move`
        and the MCP server dispatch once per interaction by construction and
        deliberately stay windowless, so nothing about their behavior changes.

        Idempotent about the past: opening a window resets the budget, so a
        previous command's spent op never charges this one.
        """
        self._command_open = True
        self._destructive_spent = False
        self._saved_this_command = set()
        self._settlement = None

    def end_command(self) -> None:
        """Close the command window; outside one the budget is not enforced.

        The pipeline runs this in a `finally`, because a command that raised
        half-way must not leak an open window into the next button press.
        """
        self._command_open = False

    @property
    def command_open(self) -> bool:
        """Whether a command window is open — one user interaction the pipeline
        is running, inside which several dispatches can chain. The confirmation
        gate reads it to tell "this command already asked its question" from
        "a fresh interaction is asking a new one" (`tools._gate`)."""
        return self._command_open

    def require_destructive_budget(self) -> None:
        """Refuse a destructive op when this command has already had one.

        A `TurnStateError`, so the model reads it as ordinary result data (it is
        a `ValueError`, and `dispatch` converts those) and a trusted caller
        answers 409 — the budget adds no new failure shape anywhere. The message
        is written for the model, which is why it says "turn": that is the
        vocabulary the tool contract uses, and what it needs to hear is *stop
        retrying and report*.

        Outside a window this is a no-op: a caller that dispatches once per
        interaction cannot spend a budget twice, so it is never asked to hold
        one.
        """
        if self._command_open and self._destructive_spent:
            raise TurnStateError(
                "a destructive operation already ran this turn — at most one "
                "per turn; report what happened to the player instead of "
                "retrying"
            )

    def record_destructive_op(self) -> None:
        """Spend the window's destructive budget. A no-op with no window open.

        Called after the op has actually mutated the session, never before:
        check then record, so a call refused by the confirmation gate or one
        that raised on the way (resigning a finished game) leaves the budget
        where it was. Nothing was thrown away, so nothing was spent.
        """
        if self._command_open:
            self._destructive_spent = True

    def record_save(self, name: str) -> None:
        """Remember that this command wrote save `name`. A no-op with no window.

        Replacing a named save asks first (#291), but not when the save being
        replaced was written moments ago by the same command: "save this as
        checkpoint, undo, save it again, then play d4" is one ask, and the
        second save throws away nothing the player had before they asked
        (audit 2026-09-05, finding 8). Outside a window there is only ever one
        call per interaction, so there is nothing to remember.
        """
        if self._command_open:
            self._saved_this_command.add(name)

    def saved_this_command(self, name: str) -> bool:
        """Whether this command already wrote save `name` (`record_save`)."""
        return self._command_open and name in self._saved_this_command

    def settle_owed_reply(self) -> ReplySettlement | None:
        """Collect the reply a landed player move is owed, and close the turn.

        The close beat, in one place for every route (#365): the narrator
        speaks *after* this, so the engine's move is on the board — and in the
        narrator's brief — before a word is said about it. The fast path and a
        drag call it before they narrate; the brain calls it through its
        `settle_reply` seam as the planner hands off; the command pipeline's
        convergence calls it for whatever is still owed after that.

        None when nothing is owed. A Stockfish that dies here is not the
        caller's failure (#284): the player's move is committed, so it comes
        back as a settlement naming what died, with the turn left owing the
        reply for the next interaction to settle. The result is kept until
        `take_settlement`, so the brain's settle and the pipeline's read of it
        are one collect, not two.
        """
        if self._phase not in (
            TurnPhase.PLAYER_MOVE_APPLIED,
            TurnPhase.AGENT_OBSERVING,
        ):
            return None
        started = time.monotonic()
        try:
            reply = self.collect_engine_reply()
        except Exception as exc:
            logger.warning("engine_reply_failed", exc_info=True)
            settlement = ReplySettlement(failure=failure_name(exc))
        else:
            self.complete_turn()
            settlement = ReplySettlement(reply=reply)
        finally:
            elapsed = max(0, round((time.monotonic() - started) * 1000))
            self._tell(self.on_engine_time, elapsed)
        self._settlement = settlement
        return settlement

    @staticmethod
    def _tell(observer: Callable[..., None] | None, *args: object) -> None:
        """Call an observer; one that raises is logged, never the turn's."""
        if observer is None:
            return
        try:
            observer(*args)
        except Exception:
            logger.warning("coordinator_observer_failed", exc_info=True)

    @property
    def settlement(self) -> ReplySettlement | None:
        """The reply the last settle produced and nobody has taken yet — what
        the narrator is told the engine played (`api.narrator_facts`)."""
        return self._settlement

    def take_settlement(self) -> ReplySettlement | None:
        """Hand over the kept settlement and forget it, so a later narration
        (a confirmed op, a resignation) is never told about this reply."""
        settlement, self._settlement = self._settlement, None
        return settlement

    def settle_engine_turn(self) -> MoveResult | None:
        """Move for the engine on a board that was left with the engine to play.

        A *restored* position can arrive with nobody scheduled to move it: a new
        game the player takes as black, a save taken between the player's move
        and the reply, an explicit odd-ply takeback that pops the reply and
        leaves the engine on move. In every one of those the board is settled
        and legal and simply waiting on a side the player does not own, so the
        turn machine would sit awaiting a player move that cannot come.

        Returns None when nothing is owed — no engine, the game is over, or the
        player is the side to move. That condition is read off the session at
        call time, never remembered: `resume_game` swaps the session, and whose
        move it is afterwards is that session's fact, not the caller's judgment.
        This is why the restoring tools ask rather than each testing a color and
        reaching for the engine themselves — every engine move in the app comes
        from this one place, and none of it is a tool the model can call.

        It does not consume a turn: the settle is not an answer to a player
        move, so nothing was open and the player's next turn is still to come.
        The turn id stands and the phase comes back to awaiting the player —
        unless the engine died, when it raises with the reply left owed.
        """
        self._require("settle the engine's turn", TurnPhase.AWAITING_PLAYER)
        source = self._move_source()
        session = self._ctx.session
        if source is None or session.is_game_over():
            return None
        if session.turn == session.player_color:
            return None
        self._enter(TurnPhase.ENGINE_CALCULATING)
        try:
            # The same source and the same gate as `collect_engine_reply`.
            reply = session.submit_move(source.choose_move(session))
        except Exception:
            # The engine died with the board waiting on it (#329). Back to
            # awaiting the player would be a lie the whole app acts on: nothing
            # collects a reply from there, and the player's next move would be
            # played for the engine's side. The reply is owed, so the phase is
            # the one that says so — the one `collect_engine_reply` leaves on
            # the same failure — and the healing branches that collect from it
            # (the command convergence, the fast path's close beat, a drag's)
            # settle it on the next interaction, exactly once.
            self._enter(TurnPhase.PLAYER_MOVE_APPLIED)
            raise
        self._note_difficulty(session)
        self._enter(TurnPhase.AWAITING_PLAYER)
        return reply

    def complete_turn(self) -> None:
        """Close the turn and roll straight into the next one.

        `completed` is a boundary, not a resting state — there is no idle phase
        between two turns, so the machine passes through it and comes back out
        awaiting the player with the turn id bumped.

        Closing early is allowed only when there is genuinely no reply to wait
        for (no engine, or the game is over). Otherwise this would be a way to
        *skip* the engine's move, and that is precisely what the coordinator
        exists to make impossible.
        """
        if self._phase in (TurnPhase.PLAYER_MOVE_APPLIED, TurnPhase.AGENT_OBSERVING):
            if self._move_source() is not None and not self._ctx.session.is_game_over():
                raise TurnStateError(
                    f"cannot complete the turn while {self._phase}: "
                    "the engine still owes a reply"
                )
        else:
            self._require("complete the turn", TurnPhase.ENGINE_MOVE_APPLIED)
        self._enter(TurnPhase.COMPLETED)
        self._turn_id += 1
        self._enter(TurnPhase.AWAITING_PLAYER)

    def play_exchange(self, move: str) -> tuple[MoveResult, MoveResult | None]:
        """The whole sequence in one call: player move, engine reply, close.

        The *atomic* turn — no observation beat, nothing between the two moves.
        `/api/game/move` runs it (direct mode: the board UI's own path, with no
        agent in it) and so does the `make_move` tool when the registry was built
        for a caller that has no pipeline to collect the reply for it (MCP). The
        beats are the pipeline's version of this same sequence, spelled out.
        """
        player = self.apply_player_move(move)
        # An illegal move changed nothing; a game-ending one already closed the
        # turn in `apply_player_move`. Either way there is no reply to fetch.
        if not player.legal or self._phase is TurnPhase.AWAITING_PLAYER:
            return player, None
        reply = self.collect_engine_reply()
        self.complete_turn()
        return player, reply
