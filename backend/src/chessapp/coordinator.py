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

The observation slot exists here but nothing fills it yet: `begin_observation`
marks the beat where Glitch reacts to the verified player move (Sprint 1's later
slice), and it is skippable by construction — `engine_reply` is legal with or
without it, so a missing, slow, or switched-off model can never hold up the
engine.

State lives on the coordinator, board truth stays in `GameSession`: the phases
say what may happen next, never what is on the board. `ctx.session` and
`ctx.engine` are read live on every call because `resume_game` swaps the session
object on the context.
"""

from enum import StrEnum
from typing import TYPE_CHECKING

from chessapp.game import MoveResult

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
        """
        self._require("apply a player move", TurnPhase.AWAITING_PLAYER)
        result = self._ctx.session.submit_move(move)
        if not result.legal:
            return result
        self._phase = TurnPhase.PLAYER_MOVE_APPLIED
        if self._ctx.session.is_game_over():
            self.complete_turn()
        return result

    def begin_observation(self) -> None:
        """Open the beat where the agent reacts to the verified player move.

        Nothing fills it in this slice. It is a phase rather than a callback so
        that the reaction is *optional* by construction: `engine_reply` is legal
        straight from `player_move_applied` too, so skipping the model (verbosity
        low, no brain, a timeout) skips only the words.
        """
        self._require("begin observation", TurnPhase.PLAYER_MOVE_APPLIED)
        self._phase = TurnPhase.AGENT_OBSERVING

    def engine_reply(self) -> MoveResult | None:
        """Have the engine answer the player's move.

        Returns None — advancing the turn all the same — when there is no engine
        or the game is already over: "the engine owes a reply" is derivable from
        the session, so nobody upstream has to work it out. The
        `engine_calculating` phase is entered *before* Stockfish is asked, which
        is the only ordering that lets an observer see it while it is true.
        """
        self._require(
            "have the engine reply",
            TurnPhase.PLAYER_MOVE_APPLIED,
            TurnPhase.AGENT_OBSERVING,
        )
        engine = self._ctx.engine
        if engine is None or self._ctx.session.is_game_over():
            self._phase = TurnPhase.ENGINE_MOVE_APPLIED
            return None
        self._phase = TurnPhase.ENGINE_CALCULATING
        reply = engine.play_move(self._ctx.session)
        self._phase = TurnPhase.ENGINE_MOVE_APPLIED
        return reply

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

        This is what the `make_move` tool and `/api/game/move` both run, so a
        move played by the agent and a move dragged on the board go through the
        same machine and produce the same pair of results. The observation beat
        is skipped here — filling it is the next slice's job, and until then a
        plain move costs exactly what it costs today.
        """
        player = self.apply_player_move(move)
        # An illegal move changed nothing; a game-ending one already closed the
        # turn in `apply_player_move`. Either way there is no reply to fetch.
        if not player.legal or self._phase is TurnPhase.AWAITING_PLAYER:
            return player, None
        reply = self.engine_reply()
        self.complete_turn()
        return player, reply
