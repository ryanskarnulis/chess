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

State lives on the coordinator, board truth stays in `GameSession`: the phases
say what may happen next, never what is on the board. `ctx.session` and
`ctx.engine` are read live on every call because `resume_game` swaps the session
object on the context.
"""

import threading
from enum import StrEnum
from typing import TYPE_CHECKING

from chessapp.game import GameSession, MoveResult

if TYPE_CHECKING:  # tools.py imports this module; don't import it back.
    from chessapp.tools import ToolContext


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


class _PendingReply:
    """One engine reply being computed off the main thread.

    It holds the position it was asked about, because by the time anyone wants
    the answer the board may have moved on (an undo mid-turn) — and an answer to
    a position that no longer exists is not a move, it is a bug waiting to be
    applied. The thread only ever reads its own `GameSession` copy, so nothing
    here mutates game state.
    """

    def __init__(self, fen: str) -> None:
        self.fen = fen
        self.uci: str | None = None
        self._thread: threading.Thread | None = None

    def start(self, engine: "object") -> None:
        probe = GameSession(self.fen)

        def _compute() -> None:
            try:
                self.uci = engine.choose_move(probe)  # type: ignore[attr-defined]
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
        self._pending: _PendingReply | None = None

    @property
    def phase(self) -> TurnPhase:
        return self._phase

    @property
    def turn_id(self) -> int:
        """Counts turn boundaries, from 1. Bumped when a turn completes, so a
        duplicated move is visible as two mutations under one id."""
        return self._turn_id

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
        result = self._ctx.session.submit_move(move)
        if not result.legal:
            return result
        self._phase = TurnPhase.PLAYER_MOVE_APPLIED
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
        engine = self._ctx.engine
        session = self._ctx.session
        if engine is None or session.is_game_over():
            return
        pending = _PendingReply(session.fen())
        pending.start(engine)
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
        self._phase = TurnPhase.AGENT_OBSERVING

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
        engine = self._ctx.engine
        session = self._ctx.session
        if engine is None or session.is_game_over():
            self._phase = TurnPhase.ENGINE_MOVE_APPLIED
            return None
        self._phase = TurnPhase.ENGINE_CALCULATING
        uci = pending.result_for(session.fen()) if pending is not None else None
        if uci is None:
            uci = engine.choose_move(session)
        # Every engine move still enters the game through the session's legality
        # gate — background computation changes when it is decided, never who
        # decides whether it is legal.
        reply = session.submit_move(uci)
        self._phase = TurnPhase.ENGINE_MOVE_APPLIED
        return reply

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
        self._phase = TurnPhase.COMPLETED
        self._turn_id += 1
        self._phase = TurnPhase.AWAITING_PLAYER

    def engine_opening_move(self) -> MoveResult | None:
        """The engine's opening move on a game the player takes as black.

        Returns None when it owes none — no engine, or the player has white.
        Whether it does is session state, not a caller's judgment, which is why
        both `new_game` paths now ask this instead of each testing the color and
        reaching for the engine themselves.

        It does not consume a turn: the game's first *player* turn is still to
        come, so the turn id stands and the phase returns to awaiting the
        player.
        """
        self._require("play the engine's opening move", TurnPhase.AWAITING_PLAYER)
        engine = self._ctx.engine
        if engine is None or self._ctx.session.player_color != "black":
            return None
        self._phase = TurnPhase.ENGINE_CALCULATING
        try:
            return engine.play_move(self._ctx.session)
        finally:
            self._phase = TurnPhase.AWAITING_PLAYER

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
            if self._ctx.engine is not None and not self._ctx.session.is_game_over():
                raise TurnStateError(
                    f"cannot complete the turn while {self._phase}: "
                    "the engine still owes a reply"
                )
        else:
            self._require("complete the turn", TurnPhase.ENGINE_MOVE_APPLIED)
        self._phase = TurnPhase.COMPLETED
        self._turn_id += 1
        self._phase = TurnPhase.AWAITING_PLAYER

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
