"""Tool layer: the validated boundary the agent talks through.

The LLM never touches `GameSession` or `EnginePlayer` directly — it names a
tool and passes args; `ToolRegistry.dispatch` validates the args against the
tool's JSON schema and runs the handler. Bad tool names, malformed args, and
domain errors (ValueError) all come back as `{"ok": False, "error": ...}` so
the agent loop can feed the failure back to the model instead of crashing.

Tools are registered with the `@registry.tool()` decorator: the name comes
from the function's `__name__`, the description from its docstring, and the
argument JSON Schema is derived from the typed signature via FastMCP's
`func_metadata` (the workspace agent standard — see
`../agent-standard/STANDARD.md` §1). No hand-written schemas, except the one
documented `parameters=` escape hatch for `set_difficulty`, whose
exactly-one-of `oneOf` cannot come from a plain signature.

**A refusal carries what recovery needs** (audit item 14). Every "no" this
layer produces — a schema failure, a domain rejection, a turn-state error, an
illegal move — comes back with `retry` (`different_args` or `never`: whether
calling again can possibly help, decided by code rather than inferred from the
wording) and `board_version` (which board it was about). Where a better call
exists, the refusal names it: legal `alternatives` for a rejected move, the
saves that exist for a bad save name (a bad *difficulty* needs no such key —
the tiers are a schema `enum`, so the validator's own message lists them).
That is the confirmation gate's pattern — its refusal has always carried the
exact line to relay — spread across the surface, and it is the difference
between an agent that corrects itself in one round trip and one that spends its
whole iteration budget asking what is legal.
"""

import inspect
import json
import logging
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Annotated, Any, Literal

import jsonschema
from mcp.server.fastmcp.utilities.func_metadata import func_metadata
from pydantic import Field

from chessapp.analysis import analyze_last_move as _analyze_last_move
from chessapp.analysis import review_game as _review_game
from chessapp.conversation import Transcript
from chessapp.coordinator import TurnCoordinator
from chessapp.engine import (
    DEFAULT_TIER,
    DIFFICULTY_TIERS,
    ELO_MAX,
    ELO_MIN,
    SKILL_MAX,
    SKILL_MIN,
    EnginePlayer,
)
from chessapp.game import GameSession, MoveResult

logger = logging.getLogger(__name__)

GET_BEST_MOVES_MAX = 10
UNDO_PLIES_MAX = 100
# Save names become filenames: one path segment, no traversal.
SAVE_NAME_PATTERN = r"^[A-Za-z0-9_-]{1,64}$"

# The two answers to "is calling again worth a round trip?" — the `retry` key on
# every refusal. Two rather than a scale because the agent only ever does one of
# two things with the answer: fix the arguments and call again, or stop and tell
# the player. `never` is the default for an unclassified failure: a loop that
# does not know why it failed must not spend its budget finding out.
RETRY_DIFFERENT_ARGS = "different_args"
RETRY_NEVER = "never"


class ToolError(ValueError):
    """A domain refusal that says how to recover from it.

    A `ValueError` like every other domain rejection here, so nothing about the
    dispatch path changes — but one the handler can attach recovery data to:
    `retry` says whether another call could work, and `details` become extra
    keys on the result (`saved_games`, and whatever a later refusal needs to
    hand back). A handler that raises a
    plain `ValueError` still works and gets `RETRY_NEVER`, which is the honest
    answer for a failure nobody classified.
    """

    def __init__(
        self, message: str, *, retry: str = RETRY_NEVER, **details: Any
    ) -> None:
        super().__init__(message)
        self.retry = retry
        self.details = details


@dataclass
class Settings:
    """Agent-adjustable app settings. Difficulty records exactly one of
    tier / skill_level / elo (the last one set); it is applied to the live
    engine when present and applied at assembly when an engine attaches.
    The personality (Glitch) is fixed, not a setting — see
    `chessapp.personality`."""

    verbosity: str = "normal"
    hints_mode: bool = False
    voice_output: bool = False
    tier: str | None = DEFAULT_TIER
    skill_level: int | None = None
    elo: int | None = None


@dataclass(frozen=True)
class Tool:
    """One agent-callable capability: OpenAI-style schema + bound handler."""

    name: str
    description: str
    parameters: dict[str, Any]  # JSON schema for the args object
    handler: Callable[..., dict[str, Any]]


def _derive_schema(fn: Callable[..., Any]) -> dict[str, Any]:
    """JSON Schema for `fn`'s arguments, from its typed signature.

    FastMCP's `func_metadata` builds a Pydantic arg model from the signature;
    its JSON Schema is the same one the MCP server would advertise. We add
    `additionalProperties: false` — chess keeps closed argument schemas
    (`dispatch` rejects extra args and Gemma is prompted against them), which
    `func_metadata` does not emit on its own.
    """
    schema = func_metadata(fn).arg_model.model_json_schema(by_alias=True)
    schema["additionalProperties"] = False
    return schema


@dataclass(frozen=True)
class PendingOp:
    """A destructive call that was refused, held for the player to confirm.

    `board_version` is the board the question is about (see
    `ToolContext.live_pending`): a "yes" is an answer to a position, and the
    position is part of the question rather than something the answer has to
    be trusted to still match.
    """

    name: str
    args: dict[str, Any]
    board_version: int


@dataclass
class ToolContext:
    """What tools operate on. `engine` is optional: analysis tools report
    an error result without one; everything else works engine-free.
    `save_dir` is where save_game/resume_game keep their files; without it
    those tools report an error. `resume_game` replaces `session` (and
    `transcript`), so the context — not any captured reference — is the
    source of truth. `transcript` is the conversation memory the agent loop
    reads and records; it rides inside save files so a resumed game keeps
    its conversational thread.

    `pending` is the armed destructive op (see `DESTRUCTIVE_TOOLS`): set by the
    gate when it refuses a call, consumed by the pipeline on the next user turn,
    and read through `live_pending` rather than directly. `_confirming` is the
    gate's only key, and it is deliberately not reachable from a tool argument —
    `confirm_pending` is the sole thing that turns it on, so nothing the model
    can emit opens the gate.

    `board_version` and `mutation_lock` are the concurrency pair (audit item 7):
    the version says *which* board a client is acting on, the lock is what makes
    checking it and acting on it one indivisible step. Both live here rather
    than on the session because the session is the thing that gets replaced —
    `resume_game` swaps it, and resuming is itself a mutation clients must be
    told about."""

    session: GameSession
    engine: EnginePlayer | None = None
    save_dir: Path | None = None
    settings: Settings = field(default_factory=Settings)
    transcript: Transcript = field(default_factory=Transcript)
    pending: PendingOp | None = None
    _confirming: bool = False
    # Carried across session swaps so the version never goes backwards; see
    # `replace_session`. Not a public counter — `board_version` is.
    _version_base: int = 0
    # Plain and non-reentrant on purpose. Plain because the async transport
    # acquires it off the event loop and releases it back on it — an owner-bound
    # `RLock` would refuse that release. Non-reentrant because there is exactly
    # one holder per request by construction: a transport takes it once around a
    # whole mutation and nothing under it reaches for it again. Nesting two
    # guarded regions in one thread would deadlock; don't.
    mutation_lock: Any = field(default_factory=threading.Lock)

    @property
    def board_version(self) -> int:
        """Monotonic id of the current board, bumped by every mutation.

        Derived rather than counted by hand: `GameSession.revision` is the one
        chokepoint every board mutation already passes through, so a player
        move, the engine's reply, an undo, a new game and a resignation all
        move this without a single `bump()` call scattered through the
        handlers. What the session cannot see is its own replacement, and
        `_version_base` is exactly that gap closed.
        """
        return self._version_base + self.session.revision

    def live_pending(self) -> PendingOp | None:
        """The armed destructive op, if it is still about the board on screen.

        A confirmation is an answer to a *question about a position* — "this
        game, the one in front of you, thrown away?" — so the position belongs
        to the question. If the board moved between the two (the player dragged
        a move, took one back, resumed a save, another client played), the game
        the player was asked about is gone and their "yes" cannot be an answer
        to it: the op is dropped here, where the answer is read, rather than by
        asking every surface that can move a board to remember to clear it. The
        same move as the board-version precondition one layer up, for the same
        reason — derive it from the chokepoint, don't maintain it by hand.

        Live, this was reachable in three keystrokes: "I resign" (the gate asks),
        a dragged move, then "yes" — and a game two plies further on ended.
        """
        pending = self.pending
        if pending is None:
            return None
        if pending.board_version != self.board_version:
            self.pending = None
            return None
        return pending

    def restamp_pending(self) -> None:
        """Point the armed op at the board the player is being asked about.

        The gate arms mid-turn, and the turn can still mutate after it: "play
        e4 and start over" arms the reset before the engine's reply lands. What
        the player sees when they hear the question is the board at the *end* of
        that turn, so that is the board their yes answers — a stamp left at arm
        time would be stale before they could speak. Called once, where the
        interaction that armed it finishes.
        """
        if self.pending is not None:
            self.pending = replace(self.pending, board_version=self.board_version)

    def replace_session(self, session: GameSession, transcript: Transcript) -> None:
        """Swap in a resumed game — a mutation like any other, version included.

        The base absorbs whatever the two sessions counted for themselves and
        leaves the swap worth exactly one bump — the incoming session's own
        revisions are not the shared board's history (a resumed game arrives
        with one per replayed move, which no client ever saw happen). Clients
        holding the old version are stale, which is the truth: they are holding
        a number for a game that is no longer on the board.
        """
        self._version_base = self.board_version + 1 - session.revision
        self.session = session
        self.transcript = transcript


# The two tools that throw a real game away.
DESTRUCTIVE_TOOLS = ("new_game", "resign")

# Reads whose answers are strict subsets of the board state the brain is handed
# in its prompt every single turn (`_agent_state_dict`). They stay registered —
# callers with no such injection (the MCP server, the delegate wire) need them,
# and `dispatch` runs them for anyone who asks — but app assembly does not
# *offer* them to the brain: it cannot learn anything from a call it already has
# the answer to, and the round trip costs a quarter of its iteration budget.
BOARD_STATE_TOOLS = (
    "get_board_state",
    "get_legal_moves",
    "get_move_history",
    "get_captured_pieces",
)


def confirm_pending(
    registry: "ToolRegistry", ctx: ToolContext
) -> tuple[str, dict[str, Any]] | None:
    """Run the armed destructive op, with the gate open. Returns
    `(name, result)`, or None when nothing is armed.

    The only way past `_gate`. The pipeline calls this — and only after the
    player themselves answered yes on a *new* turn — which is what makes the
    confirmation deterministic: by this point there is nothing left to decide,
    so no model call stands between the yes and the reset.

    Reads through `live_pending`, so an op about a board that has since moved is
    dropped rather than run: this is the last gate before a game is thrown away,
    and it owes its callers that check rather than trusting each of them to have
    made it.
    """
    pending = ctx.live_pending()
    if pending is None:
        return None
    ctx.pending = None
    ctx._confirming = True
    try:
        return pending.name, registry.dispatch(pending.name, dict(pending.args))
    finally:
        ctx._confirming = False


@dataclass
class ToolRegistry:
    """The dispatch boundary. `context` is optional and read-only from here —
    it exists so a refusal can name the board it was about (`board_version`).
    A registry built without one (unit tests, the schema fixture) simply omits
    the key rather than inventing a number.

    `on_tool` is told the name of each call that is about to *run* — the live
    progress seam (`progress.py`). It hangs off dispatch rather than off the
    callers because this is the one road every tool call takes, whoever made
    it, so a label the player sees is always a tool that really ran."""

    _tools: dict[str, Tool] = field(default_factory=dict)
    context: "ToolContext | None" = None
    on_tool: "Callable[[str], None] | None" = None

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"tool already registered: {tool.name}")
        jsonschema.Draft202012Validator.check_schema(tool.parameters)
        self._tools[tool.name] = tool

    def tool(
        self, *, parameters: dict[str, Any] | None = None
    ) -> Callable[[Callable[..., dict[str, Any]]], Callable[..., dict[str, Any]]]:
        """Decorator that registers a handler as a tool.

        Name comes from `__name__`, description from the docstring, and the
        argument schema is derived from the typed signature — unless
        `parameters=` is given (the documented escape hatch for schemas a
        plain signature can't express, e.g. `set_difficulty`'s `oneOf`), in
        which case that JSON Schema is used verbatim (still `check_schema`-d
        at registration).
        """

        def decorator(
            fn: Callable[..., dict[str, Any]],
        ) -> Callable[..., dict[str, Any]]:
            description = inspect.getdoc(fn)
            if not description:
                raise ValueError(f"tool {fn.__name__} must have a docstring")
            schema = parameters if parameters is not None else _derive_schema(fn)
            self.register(
                Tool(
                    name=fn.__name__,
                    description=description,
                    parameters=schema,
                    handler=fn,
                )
            )
            return fn

        return decorator

    def definitions(self, exclude: Sequence[str] = ()) -> list[dict[str, Any]]:
        """Tools as OpenAI-style function definitions, minus `exclude`.

        Excluding narrows only what a caller is *offered*; `dispatch` still runs
        every registered tool. That split is the point: the brain is handed the
        board state each turn and so is not offered `BOARD_STATE_TOOLS`, while
        the MCP server and the delegate wire — which inject nothing — still get
        the full list.
        """
        excluded = set(exclude)
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in self._tools.values()
            if tool.name not in excluded
        ]

    def refusal(self, error: str, retry: str, **details: Any) -> dict[str, Any]:
        """The one shape every "no" from this layer takes.

        Public because the tools that report a refusal *without* raising build
        it too — a rejected move is result data, not an exception, and it is
        owed the same `retry`/`board_version` as everything else.
        """
        result: dict[str, Any] = {"ok": False, "error": error, "retry": retry}
        if self.context is not None:
            result["board_version"] = self.context.board_version
        result.update(details)
        return result

    def dispatch(self, name: str, args: Any) -> dict[str, Any]:
        """Run tool `name` with `args`; never raises on agent-caused faults.

        The three ways a call can fail are told apart by whether a *different*
        call could succeed. A name that does not exist cannot be fixed by
        retrying (the offer is the offer); malformed args can, and that is the
        whole point of saying so. A handler's `ValueError` is a domain "no" —
        `never` unless it was raised as a `ToolError` that knows better.
        """
        tool = self._tools.get(name)
        if tool is None:
            return self.refusal(f"unknown tool: {name}", RETRY_NEVER)
        try:
            jsonschema.validate(args, tool.parameters)
        except jsonschema.ValidationError as exc:
            return self.refusal(
                f"invalid args for {name}: {exc.message}", RETRY_DIFFERENT_ARGS
            )
        # Reported here rather than at the top: a name that does not exist and
        # args the schema rejects never reach a handler, and a progress line for
        # work nobody did is worse than none.
        self._report(name)
        try:
            return tool.handler(**args)
        except ToolError as exc:
            return self.refusal(str(exc), exc.retry, **exc.details)
        except ValueError as exc:
            return self.refusal(str(exc), RETRY_NEVER)

    def _report(self, name: str) -> None:
        """Tell the observer, and never let that cost the call. Same rule the
        tracer has: a lost label is not a lost tool call."""
        if self.on_tool is None:
            return
        try:
            self.on_tool(name)
        except Exception:
            logger.warning("tool_observer_failed", exc_info=True)


def _outcome_dict(session: GameSession) -> dict[str, Any] | None:
    outcome = session.outcome()
    if outcome is None:
        return None
    return {
        "termination": outcome.termination,
        "winner": outcome.winner,
        "result": outcome.result,
    }


def _engine_move_dict(reply: MoveResult | None) -> dict[str, Any] | None:
    """How the move tools report an engine move: `{"san", "uci"}`, or None when
    the coordinator says the engine owed none. One shape for `make_move`'s reply
    and `new_game`'s opening move, which is what their results promise."""
    if reply is None:
        return None
    return {"san": reply.san, "uci": reply.uci}


def _require_engine(ctx: ToolContext) -> EnginePlayer:
    if ctx.engine is None:
        raise ValueError("engine unavailable: analysis tools need Stockfish")
    return ctx.engine


def _takeback_plies(ctx: ToolContext) -> int:
    """How many plies a player's takeback pops — the same rule the board UI's
    undo button uses (`api.undo`), and deliberately not the model's to work out.

    Vs the engine it is the player's turn after an exchange, so a takeback pops
    their move *and* the engine's reply; popping one would leave their move on
    the board with the engine to move and nothing to move it, which is what an
    agent-driven undo used to do. One ply when the game ended on the player's
    own move (no reply came) or there is no engine at all. Never the engine's
    lone opening move: that is not the player's to take back, and asking for two
    plies when only one has been played comes back as an ordinary error result.
    """
    vs_engine = ctx.engine is not None
    if vs_engine and ctx.session.turn == ctx.session.player_color:
        return 2
    return 1


def _player_has_moved(ctx: ToolContext) -> bool:
    """Whether any move on the board is the *player's* — the destructive-op
    gate's notion of a game worth guarding, derived the same way the takeback
    rule and the UI derive whose plies are whose. Vs the engine as black the
    engine owns the first ply, so a lone opening move is not player investment;
    engine-free, every move was played by the player's own hand."""
    plies = len(ctx.session.move_history())
    if ctx.engine is not None and ctx.session.player_color == "black":
        return plies >= 2
    return plies >= 1


def _save_path(ctx: ToolContext, name: str) -> Path:
    if ctx.save_dir is None:
        raise ValueError("saving unavailable: no save directory configured")
    return ctx.save_dir / f"{name}.json"


def saved_game_names(ctx: ToolContext) -> list[str]:
    """The saves on disk right now. Lives here because this layer owns the
    `{name}.json` convention (`_save_path`), and it is read fresh per turn —
    `api._agent_state_dict` hands it to the brain so that whether a saved game
    exists is deterministic state, never something the model has to infer from
    what it said earlier."""
    if ctx.save_dir is None or not ctx.save_dir.is_dir():
        return []
    return sorted(path.stem for path in ctx.save_dir.glob("*.json"))


# `make_move`'s description, in two variants — see `build_registry`'s
# `atomic_exchange`. Only the account of the engine's reply differs, because that
# is the only caller-visible difference: in one mode the reply has already been
# played when the result comes back, in the other it lands a moment later.
# Either way it is automatic and none of the model's business, and neither
# variant offers it anything to do about it. Neither says how *many* times to
# call it, either: the phase machine refuses a second player move mid-turn, and a
# rule code owns does not also live in the prompt. Written out as constants (rather
# than one template) so the atomic text stays byte-for-byte what the schema
# golden recorded; the wrapping is `inspect.getdoc`'s.
_MAKE_MOVE_HOW = """Submit the player's move: a string copied from the board state's
`legal_moves` list — SAN ('Nf3') or UCI ('g1f3'). Map loose phrasing to
the matching entry ("push the queen's bishop pawn one square" → 'c3')
and fix voice slips ("e 4" → 'e4'); never invent a string that isn't in
`legal_moves`."""

_MAKE_MOVE_ATOMIC_TAIL = (
    "The engine plays its reply inside the same call. If you proposed a\n"
    'move and the player accepts ("yes"), call it now; announcing a move\n'
    "in words is not making it."
)

_MAKE_MOVE_SPLIT_TAIL = (
    "The engine answers as soon as your move lands, without being asked.\n"
    'If you proposed a move and the player accepts ("yes"), call it now;\n'
    "announcing a move in words is not making it."
)


def _make_move_doc(atomic_exchange: bool) -> str:
    tail = _MAKE_MOVE_ATOMIC_TAIL if atomic_exchange else _MAKE_MOVE_SPLIT_TAIL
    return f"{_MAKE_MOVE_HOW}\n\n{tail}"


def build_registry(
    ctx: ToolContext,
    coordinator: TurnCoordinator | None = None,
    atomic_exchange: bool = True,
) -> ToolRegistry:
    """All read and write tools bound to `ctx`. Settings tools join in the
    final slice of the tool-layer epic.

    `coordinator` is the turn state machine the move tools drive. Pass the app's
    one shared instance (app assembly does) so a move played through a tool and a
    move dragged on the board advance the same turn; omit it and the registry
    builds its own over the same `ctx` — enough for the MCP server, which is the
    only session in the process that owns nothing else.

    `atomic_exchange` says who *sequences* a move turn — the boundary is the same
    either way. True (the default): `make_move` runs the whole exchange, player
    move and engine reply, and closes the turn. False: it applies the player's
    move and stops, leaving the coordinator mid-turn for a caller that owns the
    beats that follow — the observation reaction, then collecting the reply. App
    assembly passes False because the command pipeline is that caller; the MCP
    server keeps True, because an MCP call has nothing behind it to collect the
    reply and its game must never stall half-way through a turn.
    """
    registry = ToolRegistry(context=ctx)
    if coordinator is None:
        coordinator = TurnCoordinator(ctx)

    @registry.tool()
    def get_board_state() -> dict[str, Any]:
        """Current position: FEN, side to move, whether the game is over,
        and the outcome if it is."""
        return {
            "ok": True,
            "fen": ctx.session.fen(),
            "turn": ctx.session.turn,
            "game_over": ctx.session.is_game_over(),
            "outcome": _outcome_dict(ctx.session),
        }

    @registry.tool()
    def get_legal_moves() -> dict[str, Any]:
        "All legal moves in the current position, in SAN."
        return {"ok": True, "moves": ctx.session.legal_moves()}

    @registry.tool()
    def get_move_history() -> dict[str, Any]:
        "Moves played so far, in SAN, in order."
        return {"ok": True, "moves": ctx.session.move_history()}

    @registry.tool()
    def get_captured_pieces() -> dict[str, Any]:
        "Piece symbols each color has captured so far, in capture order."
        captured = ctx.session.captured_pieces()
        return {"ok": True, "white": captured["white"], "black": captured["black"]}

    @registry.tool()
    def evaluate_position() -> dict[str, Any]:
        """Stockfish evaluation of the current position from White's point of
        view: centipawns, or mate-in-N. This is how you answer "who's winning?"
        — from the result, never from a guess."""
        evaluation = _require_engine(ctx).evaluate_position(ctx.session)
        return {
            "ok": True,
            "score_cp": evaluation.score_cp,
            "mate_in": evaluation.mate_in,
        }

    @registry.tool()
    def get_best_moves(
        n: Annotated[
            int,
            Field(
                ge=1,
                le=GET_BEST_MOVES_MAX,
                description="How many candidate moves to return.",
            ),
        ] = 3,
    ) -> dict[str, Any]:
        """Top n candidate moves from Stockfish (MultiPV), best first, with
        SAN, UCI, and White-POV scores."""
        candidates = _require_engine(ctx).get_best_moves(ctx.session, n=n)
        return {
            "ok": True,
            "moves": [
                {
                    "uci": c.uci,
                    "san": c.san,
                    "score_cp": c.score_cp,
                    "mate_in": c.mate_in,
                }
                for c in candidates
            ],
        }

    @registry.tool()
    def analyze_last_move(
        color: Literal["white", "black"] | None = None,
    ) -> dict[str, Any]:
        """Analyze a move: how it compares to Stockfish's best from the same
        position — centipawn loss, verdict (good/inaccuracy/mistake/blunder),
        and what was best. This is how you answer "how good was that move?" or
        "what was my mistake?" — from the result, never from a guess. Defaults
        to the player's own last move (what 'my mistake' means); pass a color to
        analyze that side's last move instead."""
        # Which move "my mistake" refers to is not a question for the model.
        # On the player's turn the last *ply* is always the engine's reply, so
        # the old no-args default analyzed the opponent's move every time
        # (trace review, finding 1). The session knows whose side is whose.
        color = color or ctx.session.player_color  # type: ignore[assignment]
        analysis = _analyze_last_move(_require_engine(ctx), ctx.session, color=color)
        return {
            "ok": True,
            "played": analysis.played_san,
            "played_uci": analysis.played_uci,
            "best": analysis.best_san,
            "best_uci": analysis.best_uci,
            "cp_loss": analysis.cp_loss,
            "classification": analysis.classification,
            "color": analysis.color,
        }

    @registry.tool()
    def review_game() -> dict[str, Any]:
        """Review the whole game so far: every move classified
        (good/inaccuracy/mistake/blunder) with centipawn loss and the best
        alternative, plus per-color accuracy scores."""
        review = _review_game(_require_engine(ctx), ctx.session)
        return {
            "ok": True,
            "moves": [
                {
                    "san": m.san,
                    "uci": m.uci,
                    "color": m.color,
                    "cp_loss": m.cp_loss,
                    "classification": m.classification,
                    "best": m.best_san,
                    "accuracy": m.accuracy,
                }
                for m in review.moves
            ],
            "accuracy": review.accuracy,
            "counts": review.counts,
        }

    def make_move(
        move: Annotated[str, Field(description="SAN or UCI move.")],
    ) -> dict[str, Any]:
        # The turn sequence — player move, then the engine's reply — is the
        # coordinator's, not this tool's and not the model's. All this tool
        # chooses is how much of the sequence to run before answering: the whole
        # exchange for a caller with nothing behind it (`atomic_exchange`), or the
        # player's move alone, leaving the beats to the pipeline.
        if atomic_exchange:
            result, reply = coordinator.play_exchange(move)
        else:
            result, reply = coordinator.apply_player_move(move), None
        if not result.legal:
            # Still `ok: True` — legality is the engine's answer, not a fault,
            # and the whole app reads a rejected move as data. What it now
            # carries is the way out: the legal moves that answer what was
            # asked for, and the fact that a corrected call is worth making.
            # Nothing to correct on a finished game, so that one says so.
            retriable = bool(result.alternatives)
            return {
                "ok": True,
                "legal": False,
                "reason": result.reason,
                "alternatives": list(result.alternatives),
                "retry": RETRY_DIFFERENT_ARGS if retriable else RETRY_NEVER,
                "board_version": ctx.board_version,
            }
        payload = {
            "ok": True,
            "legal": True,
            "san": result.san,
            "uci": result.uci,
            # What the move did, for whoever narrates it: the piece it took (a
            # symbol, or null) and whether it left the opponent in check. Board
            # truth, derived by the session at move time.
            "capture": result.capture,
            "check": result.check,
            "game_over": ctx.session.is_game_over(),
            "fen": ctx.session.fen(),
            "turn": ctx.session.turn,
        }
        if atomic_exchange:
            payload["engine_move"] = _engine_move_dict(reply)
        return payload

    make_move.__doc__ = _make_move_doc(atomic_exchange)
    registry.tool()(make_move)

    @registry.tool()
    def undo(
        plies: Annotated[
            int | None,
            Field(
                ge=1,
                le=UNDO_PLIES_MAX,
                description=(
                    "How many half-moves to take back. Omit for a normal "
                    "takeback — the app works out the right number itself."
                ),
            ),
        ] = None,
    ) -> dict[str, Any]:
        """Take back the player's last move. For any normal takeback ("undo",
        "take that back") omit plies: the app pops the full exchange — the
        player's move and the engine's reply — leaving the player to move
        again. Pass plies only when the player asked for an explicit count of
        half-moves."""
        # A takeback replaces the position the open turn is about, so the turn
        # goes with it (and any reply being computed for it). Same rule for every
        # non-move mutation below.
        coordinator.abandon_turn()
        result = ctx.session.undo(_takeback_plies(ctx) if plies is None else plies)
        if not result.ok:
            # Nothing to take back, or a game ended by resignation: asking again
            # with a different count does not conjure plies that were never
            # played, so this one is for the player to hear, not to retry.
            return registry.refusal(result.reason or "cannot undo", RETRY_NEVER)
        return {
            "ok": True,
            "undone": list(result.undone),
            "fen": ctx.session.fen(),
            "turn": ctx.session.turn,
        }

    def _gate(name: str, args: dict[str, Any]) -> dict[str, Any] | None:
        """Refuse an unconfirmed destructive call; arm it for the player's yes.

        The prompt asks the agent to confirm before `new_game`/`resign`, but the
        model honors that only about half the time (docs/agent-evals.md), so the
        rule is enforced here, where the model has no say. A refusal is an
        ordinary rejection *result* — the agent reads it and asks the player,
        exactly as it reads an illegal move and corrects — so the gate needs no
        special path through the loop.

        Re-arming is not confirming: a second unconfirmed call inside the same
        turn is refused again. Only `confirm_pending`, on a later user turn,
        opens the gate — and only while the board it armed against is still the
        board on screen (`ToolContext.live_pending`).

        The gate stands aside when there is no game to lose — it guards the
        *player's* investment, not the idea of a game. That is true once it is
        over, and equally true before the player has moved: an untouched
        starting position is nothing to confirm, and so is a board holding only
        the engine's opening move ("switch to white" right after the engine
        opened must not cost a question about a game the player never joined).
        Which plies are the player's is derived, not judged: vs the engine as
        black the engine owns the first ply, so one move on the board is still
        no investment; engine-free, every move was played by the player's own
        hand.
        """
        if ctx._confirming:
            return None
        if ctx.session.is_game_over() or not _player_has_moved(ctx):
            return None
        ctx.pending = PendingOp(
            name=name, args=dict(args), board_version=ctx.board_version
        )
        # `never`, and it is the sharpest example of why the key earns its
        # place: the refusal is not a failure to fix but a question to relay,
        # and calling again is the exact wrong move — the docstrings say "do not
        # call again" and the model obeyed that about half the time.
        return registry.refusal(
            f"confirmation required: {name} would end the current game. "
            "Ask the player to confirm; it runs when they say yes.",
            RETRY_NEVER,
        )

    @registry.tool()
    def new_game(
        player_color: Literal["white", "black"] | None = None,
    ) -> dict[str, Any]:
        """Reset to the starting position and begin a new game. Pass
        `player_color` to put the player on that side ("let's play as black");
        omitted, they keep the side they have. Call this as soon as the player
        asks — do not ask them to confirm first. If a game is in progress the
        result comes back refusing and asking you to confirm; relay that to the
        player in your own words and stop, do not call again. When it returns
        ok, the game really did reset."""
        # The budget is checked before the gate, so a refused-for-budget call
        # neither arms an op nor overwrites one that is already armed: it did
        # not happen, and the player is owed no question about it.
        coordinator.require_destructive_budget()
        # The color rides along in the armed op, so the player's "yes" on the
        # next turn confirms the game they actually asked for — a gate that
        # dropped it would confirm "new game, I'll take black" into a game as
        # white.
        refusal = _gate("new_game", {"player_color": player_color})
        if refusal is not None:
            return refusal
        coordinator.abandon_turn()  # the old turn is about a board that is gone
        ctx.session.new_game(player_color)
        # The reset *is* the destructive act, so the budget is spent here rather
        # than after the opening move: by this point the old game is gone whatever
        # else happens.
        coordinator.record_destructive_op()
        # When the player has black the engine owns white and makes the opening
        # move — otherwise the fresh board would sit waiting for a move only the
        # engine can make. The coordinator owns that call (and the condition):
        # every engine move in the app comes from one place.
        return {
            "ok": True,
            "engine_move": _engine_move_dict(coordinator.engine_opening_move()),
            "fen": ctx.session.fen(),
            "turn": ctx.session.turn,
        }

    @registry.tool()
    def resign(color: Literal["white", "black"] | None = None) -> dict[str, Any]:
        """Resign the game. Defaults to the player's own side; pass a color to
        resign for that side. Call this as soon as the player concedes — do not
        ask them to confirm first. If a game is in progress the result comes
        back refusing and asking you to confirm; relay that to the player and
        stop, do not call again. When it returns ok, the game really did end."""
        # Whose resignation this is, is not a question for the model: an
        # unqualified "I resign" is the player's, and the session knows which
        # side that is. The old default — the side to move — was only
        # coincidentally the player (trace review, finding 8).
        color = color or ctx.session.player_color  # type: ignore[assignment]
        # Before the gate, as in `new_game`: a call the budget refuses arms
        # nothing. The gap this closes is exactly a resignation reaching a board
        # a `new_game` in the same command has just made — no player move on it,
        # so the gate would wave it straight through.
        coordinator.require_destructive_budget()
        refusal = _gate("resign", {"color": color})
        if refusal is not None:
            return refusal
        coordinator.abandon_turn()  # no reply is owed on a game that just ended
        outcome = ctx.session.resign(color)
        coordinator.record_destructive_op()
        return {
            "ok": True,
            "outcome": {
                "termination": outcome.termination,
                "winner": outcome.winner,
                "result": outcome.result,
            },
        }

    @registry.tool()
    def export_pgn() -> dict[str, Any]:
        "Export the game so far as PGN."
        return {"ok": True, "pgn": ctx.session.export_pgn()}

    @registry.tool()
    def save_game(
        name: Annotated[str, Field(pattern=SAVE_NAME_PATTERN)] = "autosave",
    ) -> dict[str, Any]:
        "Save the current game under a name (default 'autosave')."
        # The transcript rides in the same file under a key GameSession
        # ignores, so game truth and conversation stay in one save and old
        # saves (no transcript key) remain loadable.
        path = _save_path(ctx, name)
        data = ctx.session.to_dict()
        data["transcript"] = ctx.transcript.to_dict()
        path.write_text(json.dumps(data, indent=2))
        return {"ok": True, "name": name}

    @registry.tool()
    def resume_game(
        name: Annotated[str, Field(pattern=SAVE_NAME_PATTERN)] = "autosave",
    ) -> dict[str, Any]:
        "Resume a previously saved game by name (default 'autosave')."
        path = _save_path(ctx, name)
        if not path.exists():
            # The saves that do exist ride along: the request was for a real
            # capability with the wrong argument, and the right argument is a
            # fact the app holds. Without it the agent's only recovery was to
            # guess another name or invent an apology about saving being broken
            # (the self-poisoning shape `saved_games` in the prompt also fixes).
            raise ToolError(
                f"no saved game named {name!r}",
                retry=RETRY_DIFFERENT_ARGS,
                saved_games=saved_game_names(ctx),
            )
        data = json.loads(path.read_text())
        # Validate both parts before touching the context, so a corrupt file
        # can't leave a restored board with someone else's conversation.
        session = GameSession.from_dict(data)
        transcript = Transcript.from_dict(data.get("transcript", []))
        # A different game replaces the position entirely, so the open turn (and
        # anything being computed for the *old* session) is abandoned first.
        coordinator.abandon_turn()
        ctx.replace_session(session, transcript)
        return {
            "ok": True,
            "name": name,
            "fen": ctx.session.fen(),
            "turn": ctx.session.turn,
        }

    def set_difficulty(
        tier: str | None = None,
        skill_level: int | None = None,
        elo: int | None = None,
    ) -> dict[str, Any]:
        if tier is not None:
            if ctx.engine is not None:
                ctx.engine.set_tier(tier)
            ctx.settings.tier = tier
            ctx.settings.skill_level = None
            ctx.settings.elo = None
        elif skill_level is not None:
            if ctx.engine is not None:
                ctx.engine.set_skill_level(skill_level)
            ctx.settings.skill_level = skill_level
            ctx.settings.tier = None
            ctx.settings.elo = None
        else:
            if ctx.engine is not None:
                ctx.engine.set_elo(elo)
            ctx.settings.elo = elo
            ctx.settings.tier = None
            ctx.settings.skill_level = None
        return {"ok": True, "tier": tier, "skill_level": skill_level, "elo": elo}

    # Exactly-one-of tier/skill_level/elo is a JSON-Schema `oneOf`, which a
    # plain signature can't express — the documented `parameters=` escape
    # hatch. The dynamic description (strength bounds from engine constants)
    # is likewise not a constant docstring, so it's set here.
    set_difficulty.__doc__ = (
        "Set engine strength: pass exactly one of tier (named level: "
        "beginner ~500, casual ~1000, intermediate ~1500, advanced "
        f"~2000, maximum = full strength), skill_level ({SKILL_MIN}-"
        f"{SKILL_MAX}), or elo ({ELO_MIN}-{ELO_MAX}). Prefer tier "
        "unless the user asks for a specific number."
    )
    registry.tool(
        parameters={
            "type": "object",
            "properties": {
                "tier": {"type": "string", "enum": list(DIFFICULTY_TIERS)},
                "skill_level": {
                    "type": "integer",
                    "minimum": SKILL_MIN,
                    "maximum": SKILL_MAX,
                },
                "elo": {
                    "type": "integer",
                    "minimum": ELO_MIN,
                    "maximum": ELO_MAX,
                },
            },
            "oneOf": [
                {"required": ["tier"]},
                {"required": ["skill_level"]},
                {"required": ["elo"]},
            ],
            "additionalProperties": False,
        }
    )(set_difficulty)

    @registry.tool()
    def set_verbosity(verbosity: Literal["low", "normal", "high"]) -> dict[str, Any]:
        "How chatty the agent's commentary is."
        ctx.settings.verbosity = verbosity
        return {"ok": True, "verbosity": verbosity}

    @registry.tool()
    def set_hints_mode(enabled: bool) -> dict[str, Any]:
        "Turn move hints on or off."
        ctx.settings.hints_mode = enabled
        return {"ok": True, "hints_mode": enabled}

    @registry.tool()
    def set_voice_output(enabled: bool) -> dict[str, Any]:
        "Turn spoken (TTS) output on or off."
        ctx.settings.voice_output = enabled
        return {"ok": True, "voice_output": enabled}

    return registry
