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
from contextlib import suppress
from dataclasses import dataclass, field, replace
from dataclasses import fields as dataclass_fields
from pathlib import Path
from typing import Annotated, Any, Literal

import chess
import jsonschema
from mcp.server.fastmcp.utilities.func_metadata import func_metadata
from pydantic import Field

from chessapp.analysis import analyze_last_move as _analyze_last_move
from chessapp.analysis import review_game as _review_game

# The `retry` vocabulary is the dispatcher protocol's, not this layer's — it is
# defined beside `ToolDispatcher` and re-exported here, so `tools.RETRY_NEVER`
# still names the one definition both sides of the seam read.
from chessapp.brain import RETRY_DIFFERENT_ARGS, RETRY_NEVER
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


SETTINGS_FILENAME = "settings.json"
GAME_SAVE_DIRNAME = "games"


def _migrate_legacy_game_saves(save_dir: Path) -> None:
    """Move valid saves from the old shared root into the game namespace.

    Releases before the game/settings split wrote both documents as
    ``{save_dir}/{name}.json``. Validate with the deterministic loader before
    moving so arbitrary JSON metadata is never promoted to a game. Migration
    is best-effort like settings persistence: an unwritable home volume must
    not prevent the app from starting.
    """
    if not save_dir.is_dir():
        return
    game_dir = save_dir / GAME_SAVE_DIRNAME
    for legacy_path in save_dir.glob("*.json"):
        try:
            data = json.loads(legacy_path.read_text())
        except OSError:
            logger.warning("could not inspect legacy save %s", legacy_path)
            continue
        except ValueError:
            continue
        if not isinstance(data, dict):
            continue
        try:
            GameSession.from_dict(data)
        except (AttributeError, KeyError, TypeError, ValueError):
            continue
        destination = game_dir / legacy_path.name
        if destination.exists():
            logger.warning(
                "legacy save %s not migrated: %s already exists",
                legacy_path,
                destination,
            )
            continue
        try:
            game_dir.mkdir(parents=True, exist_ok=True)
            legacy_path.replace(destination)
        except OSError:
            logger.warning("could not migrate legacy save %s", legacy_path)


@dataclass
class Settings:
    """Agent-adjustable app settings. Difficulty records exactly one of
    tier / skill_level / elo (the last one set); it is applied to the live
    engine when present and applied at assembly when an engine attaches.
    The personality (Glitch) is fixed, not a setting — see
    `chessapp.personality`.

    With a store attached (`ToolContext.__post_init__`, off `save_dir`),
    every field assignment writes the whole object through to disk —
    `__setattr__` is the one chokepoint all mutation sites already pass
    through, so no `set_*` tool or endpoint can forget to persist."""

    verbosity: str = "normal"
    voice_output: bool = False
    tier: str | None = DEFAULT_TIER
    skill_level: int | None = None
    elo: int | None = None

    def snapshot(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in dataclass_fields(self)}

    def attach_store(self, write: Callable[[dict[str, Any]], None]) -> None:
        # Not a dataclass field: the store is plumbing, not a setting — it
        # must stay out of snapshots, comparisons and the schema.
        object.__setattr__(self, "_write", write)

    def __setattr__(self, name: str, value: Any) -> None:
        object.__setattr__(self, name, value)
        write = self.__dict__.get("_write")
        if write is not None:
            write(self.snapshot())


def _valid_setting(name: str, value: Any) -> bool:
    """Whether a value from the settings file is one the app could have
    written. The file is ours, but a hand-edit or a partial write must never
    turn into an unconfigured engine or a crash at assembly."""
    if name == "verbosity":
        return value in ("low", "normal", "high")
    if name == "voice_output":
        # A settings file written before 2026-09-01 may still carry a
        # `hints_mode` key; the restore loop reads only current dataclass
        # fields, so the stale key is skipped without needing a rule here.
        return isinstance(value, bool)
    if name == "tier":
        return value is None or value in DIFFICULTY_TIERS
    if name == "skill_level":
        return value is None or (
            isinstance(value, int) and SKILL_MIN <= value <= SKILL_MAX
        )
    if name == "elo":
        return value is None or (isinstance(value, int) and ELO_MIN <= value <= ELO_MAX)
    return False


def _restore_settings(settings: Settings, path: Path) -> None:
    """Best-effort restore: a missing, corrupt or invalid file means the
    defaults stand — persistence must never stop assembly."""
    try:
        data = json.loads(path.read_text())
    except FileNotFoundError:
        return
    except (OSError, ValueError):
        logger.warning("ignoring unreadable settings file %s", path)
        return
    if not isinstance(data, dict):
        logger.warning("ignoring malformed settings file %s", path)
        return
    for f in dataclass_fields(Settings):
        if f.name in data and _valid_setting(f.name, data[f.name]):
            setattr(settings, f.name, data[f.name])


def _write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    """Write `data` as JSON so that a failed write costs nothing already on disk.

    The bytes land in a sibling temp file and one `replace` swaps it in, so a
    reader sees either the whole old document or the whole new one. Writing
    straight to the destination has no such rollback point: it truncates first,
    and anything that dies after that (a full disk, an I/O error, the host going
    away) has destroyed the file it was replacing — which for a named game save
    is the player's last good game (#219). The temp file goes away on the way
    out so a failure leaves no litter, and the `.json.tmp` suffix keeps one that
    outlives a hard kill outside `saved_game_names`'s `*.json` glob, so a
    half-written save is never advertised as a save.

    The `OSError` is re-raised for the caller to classify: the two callers answer
    to different contracts — settings persistence is best-effort, a save the
    player asked for owes them a refusal.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_text(json.dumps(data, indent=2))
        tmp.replace(path)
    except OSError:
        # Suppressed so the cleanup's own failure can never stand in for the
        # real one — the caller is owed the error that lost the write.
        with suppress(OSError):
            tmp.unlink(missing_ok=True)
        raise


def _write_settings_file(path: Path, data: dict[str, Any]) -> None:
    """Atomic best-effort write: a failed save must never fail the mutation
    that triggered it (the same rule the tracer and progress observers live
    by), and a crash mid-write must never leave a half-file to restore from."""
    try:
        _write_json_atomic(path, data)
    except OSError:
        logger.warning("could not persist settings to %s", path)


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


def _admits_integer(schema: Any) -> bool:
    """Whether a property's schema accepts an `integer` for that property.

    Looks through the combinators as well as a plain `type`, because a nullable
    integer is the shape the tool signatures actually produce: `int | None`
    comes out of pydantic as `anyOf: [{type: integer, …}, {type: null}]`, and
    `undo`'s `plies` is exactly that.
    """
    if not isinstance(schema, dict):
        return False
    declared = schema.get("type")
    if declared == "integer" or (isinstance(declared, list) and "integer" in declared):
        return True
    return any(
        _admits_integer(branch)
        for key in ("anyOf", "oneOf", "allOf")
        for branch in schema.get(key) or ()
    )


def _narrow_integral_floats(args: Any, schema: dict[str, Any]) -> Any:
    """`args` with each top-level integral float narrowed to the `int` its
    schema already called it.

    JSON has one number type, and JSON Schema honors that: `1.0` *is* an
    integer to the validator, so `undo(plies=1.0)` passes the schema and then
    reaches a Python slice, which does not agree (audit 2026-09-05, finding 5).
    The validator's reading is the right one — the value the model sent is one
    half-move — so the number is narrowed to what the handler's signature
    already promised rather than refused for how it was spelled.

    Only where the property's schema actually names `integer`: a `number`
    parameter that arrives as `2.0` is a float on purpose and stays one, and a
    property nobody declared is left exactly as it came.
    """
    if not isinstance(args, dict):
        return args
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        return args
    return {
        name: int(value)
        if (
            isinstance(value, float)
            and value.is_integer()
            and _admits_integer(properties.get(name))
        )
        else value
        for name, value in args.items()
    }


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

    def __post_init__(self) -> None:
        # Settings ride the save dir: restored before anything reads them
        # (assembly applies the difficulty to the engine off this object),
        # then written through on every mutation. No save dir — tests, or a
        # deployment without the volume — means plain in-memory settings,
        # exactly as before.
        if self.save_dir is not None:
            _migrate_legacy_game_saves(self.save_dir)
            path = self.save_dir / SETTINGS_FILENAME
            _restore_settings(self.settings, path)
            self.settings.attach_store(lambda data: _write_settings_file(path, data))

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


# The tools that throw a real game away: the reset, and the two ways a player
# can end one deliberately.
DESTRUCTIVE_TOOLS = ("new_game", "resign", "claim_draw")

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

# `describe_position` is a read too, and is deliberately *not* in that tuple.
# The four above are withheld because their answers already sit in the planner's
# prompt; this one's consumer is the **narrator**, which is handed no board at
# all (`api._narrator_state_dict`, #193). Its result is the only route by which
# a description of the position reaches the phase that speaks — withhold it and
# a "what's the position?" turn has nothing to describe from, which is how the
# ask came back as an eval ("You're cooked") in the 2026-09-04 walkthrough.


def brain_tool_exclusions(ctx: ToolContext) -> list[str]:
    """Which registered tools the brain is *not* offered, right now.

    The offer policy, in one place, resolved per command off live state — app
    assembly and the eval harness both ask this rather than each keeping its own
    copy, because a copy that drifts means the measured agent is not the shipped
    one (and a bigger or different tool list is itself a variable: the 2026-07-13
    trace review saw capture phrasings behave differently under two lists).

    Two reasons a tool is withheld, and neither is a prompt rule:

    - `BOARD_STATE_TOOLS`, always: their answers are strict subsets of the state
      block the brain is handed every turn, so a call only burns a round trip.
    - `claim_draw` while no draw is claimable: whether the rules allow a claim is
      board truth, so the tool simply is not there until it can be used — never
      the model's judgment. This also keeps the planner's schema unchanged on
      every turn where no claim exists, which is what the eval baseline is
      measured against.

    `get_best_moves` used to be the third (audit item 11: withheld while hints
    were off). Hints mode was retired 2026-09-01 — a hint is on-request now, so
    the tool that answers the ask is in the offer on every turn, and what keeps
    advice honest is the pipeline's guard: commentary may name a legal move only
    if an analysis tool reported it this turn.

    Callers with no state injection (the MCP server, the delegate wire,
    `/api/game/hint`) still get the full registry; a withheld tool stays
    registered and dispatchable, and refuses on its own terms when it cannot run.
    """
    exclude = list(BOARD_STATE_TOOLS)
    if not ctx.session.claimable_draws():
        exclude.append("claim_draw")
    return exclude


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
    it, so a label the player sees is always a tool that really ran.

    `on_mutation` is told once a call has *moved the board*, which is
    `board_version` before the handler differing from after it. It hangs off
    dispatch for the same reason: this is the one road every model-initiated
    mutation takes, so a board document published from here is one the game
    really reached. A handler that moved the board and then refused still
    moved it, so the report is owed either way."""

    _tools: dict[str, Tool] = field(default_factory=dict)
    context: "ToolContext | None" = None
    on_tool: "Callable[[str], None] | None" = None
    on_mutation: "Callable[[], None] | None" = None

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
        `never` unless it was raised as a `ToolError` that knows better. A
        handler's `TypeError` is the fourth, and the odd one: arguments the
        schema said yes to that Python cannot take.
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
        args = _narrow_integral_floats(args, tool.parameters)
        # Reported here rather than at the top: a name that does not exist and
        # args the schema rejects never reach a handler, and a progress line for
        # work nobody did is worse than none.
        self._report(name)
        version_before = None if self.context is None else self.context.board_version
        try:
            return tool.handler(**args)
        except ToolError as exc:
            return self.refusal(str(exc), exc.retry, **exc.details)
        except ValueError as exc:
            return self.refusal(str(exc), RETRY_NEVER)
        except TypeError as exc:
            # The schema said yes and Python said no: arguments whose JSON
            # shape the validator admits but whose Python type the handler
            # cannot work with. `undo(plies=1.0)` was the live one — narrowed
            # above now — and it did not merely fail: the `TypeError` escaped
            # `dispatch` entirely, past the rest of the batch and past the
            # trace, and reached the player as an HTTP 500 after the board had
            # already moved (audit 2026-09-05, finding 5). Nothing may leave
            # this boundary as an exception, so the next one of these lands
            # here as ordinary error data, and `different_args` is the honest
            # advice for a call the handler could not take. Logged with the
            # traceback, because a real bug wearing this shape is still a bug
            # and the logs are where it has to stay visible.
            logger.warning("tool_call_type_error name=%s", name, exc_info=True)
            return self.refusal(f"invalid args for {name}: {exc}", RETRY_DIFFERENT_ARGS)
        finally:
            # In a `finally` because the board does not care how the call
            # ended: a handler that mutated and then raised has left a new
            # position behind, and a client not told about it is wrong rather
            # than merely late.
            if (
                self.context is not None
                and self.context.board_version != version_before
            ):
                self._report_mutation()

    def _report(self, name: str) -> None:
        """Tell the observer, and never let that cost the call. Same rule the
        tracer has: a lost label is not a lost tool call."""
        if self.on_tool is None:
            return
        try:
            self.on_tool(name)
        except Exception:
            logger.warning("tool_observer_failed", exc_info=True)

    def _report_mutation(self) -> None:
        """Same rule as `_report`: a lost board frame is not a lost move."""
        if self.on_mutation is None:
            return
        try:
            self.on_mutation()
        except Exception:
            logger.warning("mutation_observer_failed", exc_info=True)


def _outcome_dict(session: GameSession) -> dict[str, Any] | None:
    outcome = session.outcome()
    if outcome is None:
        return None
    return {
        "termination": outcome.termination,
        "winner": outcome.winner,
        "result": outcome.result,
    }


# Who made a move, as the payloads say it. Constants because these strings are a
# contract with whoever reads a result — the narrator most of all.
_MOVER_PLAYER = "player"
_MOVER_ENGINE = "engine"

# How a summary addresses each mover: how the move is introduced, whose the
# captured piece was, and who was put in check. The player's summary is written
# *to the narrator*, who is the player's opponent — so a piece the player took
# was the narrator's own ("your pawn"). The engine's summary is neutral instead
# of second-person, because that move is the narrator's and its readers vary (an
# MCP caller's atomic exchange, `new_game`'s opening move).
_SUMMARY_VOICE = {
    _MOVER_PLAYER: ("The player played", "your", "you"),
    _MOVER_ENGINE: ("The engine played", "the player's", "the player"),
}


def _piece_word(symbol: str) -> str:
    """The word a summary calls a captured piece by ("n" -> "knight").

    python-chess owns the vocabulary, so this layer keeps no second copy of it.
    The symbol always comes from `GameSession._captured_symbol`, i.e. board
    truth — the fallback exists only so that composing prose can never turn a
    move that already landed into an error result.
    """
    try:
        return chess.piece_name(chess.PIECE_SYMBOLS.index(symbol))
    except ValueError:  # pragma: no cover - unreachable off board truth
        return "piece"


def _move_summary(result: MoveResult, mover: str) -> str:
    """One English sentence saying who moved, what they played, whose piece it
    took and whether it checks.

    Composed by code from `MoveResult`, because attribution is deterministic
    state and a 12B asked to derive it from move-history parity gets it wrong:
    live, the narrator claimed the player's capture as its own ("I took your
    knight" for a piece the player had just taken off it). The symbol/boolean
    fields stay on the payload for every other reader; this is the same facts in
    the one form the narrator cannot misread.

    En passant needs no special case here: `MoveResult.capture` is already the
    pawn the move took, whether or not it stood on the destination square.
    """
    opening, owner, target = _SUMMARY_VOICE[mover]
    clauses = []
    if result.capture:
        clauses.append(f"capturing {owner} {_piece_word(result.capture)}")
    if result.check:
        clauses.append(f"putting {target} in check")
    if not clauses:
        return f"{opening} {result.san}."
    return f"{opening} {result.san}, {' and '.join(clauses)}."


# How a *position* summary addresses each side: the subject, and the two verbs
# that follow it. Second person for the same reason `_SUMMARY_VOICE` uses it —
# the paragraph is written to the narrator, who is the player's opponent — so
# "you" is Glitch and "the player" is the human, and a narrator that reads
# "you are up two pawns" has been told about its own material, not the
# player's.
_PLAYER_VOICE = ("The player", "has", "is")
_NARRATOR_VOICE = ("You", "have", "are")


def _piece_clause(placement: dict[str, list[str]]) -> str:
    """One side's pieces as "king e1; rooks a1, h1" — the clause a description
    sentence hangs on. Plural by count, so a lone rook is not "rooks a1"."""
    return "; ".join(
        f"{name if len(squares) == 1 else name + 's'} {', '.join(squares)}"
        for name, squares in placement.items()
    )


def _castling_clause(voice: tuple[str, str, str], status: dict[str, Any]) -> str:
    """One side's castling, as a sentence. Having castled outranks the rights,
    which castling itself spends: reporting the rights alone would describe a
    king that castled ten moves ago and one that merely walked out of its
    rights in exactly the same words."""
    subject, has, _ = voice
    if status["castled"] is not None:
        return f"{subject} {has} castled {status['castled']}."
    rights = status["rights"]
    if not rights:
        return f"{subject} can no longer castle."
    if len(rights) == 2:
        return f"{subject} can still castle either side."
    return f"{subject} can still castle {rights[0]} only."


def _position_summary(
    session: GameSession,
    placement: dict[str, dict[str, list[str]]],
    castling: dict[str, dict[str, Any]],
) -> str:
    """The whole position as one deterministic English paragraph.

    The same argument as `_move_summary`, one step further. That one exists
    because a 12B misreads structured data about a single move; this one exists
    because the narrator — the phase that actually speaks — is handed no board
    at all (`api._narrator_state_dict`, #193), so until this tool there was no
    result whose content was a *description*. The structured keys stay on the
    payload for every other reader; this is the same facts in the one form the
    narrator cannot misread.

    Two things it never says. It never names a side to move: the observe beat
    runs while the engine's reply is still being computed, and every leak of
    "it is your move" produced a narrator announcing a move of its own. And it
    never says who played the last move — SAN alone, because naming the mover
    names the side to move one ply later.
    """
    sentences: list[str] = []
    outcome = session.outcome()
    if outcome is not None:
        ending = f"{outcome.winner} wins" if outcome.winner is not None else "drawn"
        # The termination name is an identifier everywhere else in the app;
        # here it is read aloud, so `fifty_moves` becomes "fifty moves".
        termination = outcome.termination.replace("_", " ")
        sentences.append(f"The game is over: {termination}, {ending}.")
    elif session.plies == 0 and session.fen() == chess.STARTING_FEN:
        sentences.append("Nothing has been played yet — the starting position.")
    else:
        # A session can be rooted on a FEN nobody played into (a resumed save,
        # the guard's per-turn boards), so "nothing played" alone does not mean
        # "the starting position" and this branch is what an unplayed custom
        # position gets.
        sentences.append(
            f"Move {session.fullmove_number}, {session.plies} plies played."
        )

    # Material, phrased as "up N pawns" on purpose: that is the shape the
    # honesty guard's material class reads, so an echo of this sentence is a
    # claim the board can be made to back rather than one the guard has to eat.
    balance = session.material_balance()
    if balance == 0:
        sentences.append("Material is level.")
    else:
        subject, _, is_verb = _PLAYER_VOICE if balance > 0 else _NARRATOR_VOICE
        pawns = abs(balance)
        unit = "pawn" if pawns == 1 else "pawns"
        sentences.append(f"{subject} {is_verb} up {pawns} {unit} of material.")

    opponent = "black" if session.player_color == "white" else "white"
    sides = ((session.player_color, _PLAYER_VOICE), (opponent, _NARRATOR_VOICE))
    for color, voice in sides:
        subject, has, _ = voice
        pieces = _piece_clause(placement[color])
        sentences.append(f"{subject} ({color.capitalize()}) {has}: {pieces}.")
    for color, voice in sides:
        sentences.append(_castling_clause(voice, castling[color]))

    if session.is_check():
        # A colour, never "you" or "the player": a king in check is the side to
        # move's, so saying whose it is in the second person is one more way of
        # handing the narrator a turn to take.
        sentences.append(f"The {session.turn} king is in check.")
    history = session.move_history()
    if history:
        sentences.append(f"Last move: {history[-1]}.")
    return " ".join(sentences)


def _engine_move_dict(reply: MoveResult | None) -> dict[str, Any] | None:
    """How the move tools report an engine move: `{"san", "uci", "summary"}`, or
    None when the coordinator says the engine owed none. One shape for
    `make_move`'s reply and for every move that settles a restored board —
    `new_game`'s opening, an odd-ply `undo`, a `resume_game` that came back
    mid-exchange — which is what their results promise, and what lets the
    pipeline announce any of them with one branch."""
    if reply is None:
        return None
    return {
        "san": reply.san,
        "uci": reply.uci,
        "summary": _move_summary(reply, _MOVER_ENGINE),
    }


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
    return ctx.save_dir / GAME_SAVE_DIRNAME / f"{name}.json"


def saved_game_names(ctx: ToolContext) -> list[str]:
    """The saves on disk right now. Lives here because this layer owns the
    `games/{name}.json` convention (`_save_path`), and it is read fresh per turn —
    `api._agent_state_dict` hands it to the brain so that whether a saved game
    exists is deterministic state, never something the model has to infer from
    what it said earlier."""
    if ctx.save_dir is None or not ctx.save_dir.is_dir():
        return []
    game_dir = ctx.save_dir / GAME_SAVE_DIRNAME
    return sorted(path.stem for path in game_dir.glob("*.json"))


# The two facts about *this* app that every game it exports shares. Not
# settings: nobody is entering a tournament from the sofa, and a `Site` a
# player can read is worth more than one they have to fill in.
_PGN_EVENT = "Casual game"
_PGN_SITE = "Chess vs Glitch (home network)"
_PGN_UNKNOWN_DATE = "????.??.??"


def _engine_strength(settings: Settings) -> str | None:
    """How hard the engine is playing, spelled for a human reading the header.

    Exactly one of the three is ever set (`set_difficulty` clears the others),
    but all three are checked rather than assumed: a hand-edited settings file
    can leave none of them, and a `Black` tag is not the place to raise.
    """
    if settings.tier is not None:
        return settings.tier
    if settings.elo is not None:
        return f"{settings.elo} Elo"
    if settings.skill_level is not None:
        return f"skill {settings.skill_level}"
    return None


def pgn_headers(ctx: ToolContext, session: GameSession | None = None) -> dict[str, str]:
    """The Seven Tag Roster, with a real answer wherever the app has one.

    One composer for both exports — the `export_pgn` tool and
    `GET /api/game/pgn` — because the two are the same PGN by two routes, and
    a player who copies from the chat and from the post-game screen must not
    get two different games. Live, neither filled anything in: "export the pgn"
    came back `[Event "?"] [Site "?"] [Date "????.??.??"] … [White "?"]
    [Black "?"]`, every tag a question mark for facts the app was holding all
    along (2026-09-04 walkthrough).

    `Result` is deliberately absent: it is board truth and `export_pgn` writes
    it from `outcome()`, so it is the one tag a caller may not compose.

    `session` is the game being exported, defaulting to the live one. The
    endpoint exports a *copy* taken at the mutation boundary (`_session_snapshot`),
    and passing that copy is what keeps the date and the sides on the same game
    as the moves when a `new_game` lands mid-export.
    """
    session = ctx.session if session is None else session
    # What the opponent was, as precisely as this deployment can say it. No
    # engine attached is not a weak Stockfish, it is no Stockfish — that game
    # was played against something else, and the header may not imply otherwise.
    if ctx.engine is None:
        glitch = "Glitch"
    elif (strength := _engine_strength(ctx.settings)) is None:
        glitch = "Glitch (Stockfish)"
    else:
        glitch = f"Glitch (Stockfish, {strength})"
    started = session.started
    player_is_white = session.player_color == "white"
    return {
        "Event": _PGN_EVENT,
        "Site": _PGN_SITE,
        # PGN dates are dot-separated; a save from before games recorded one
        # keeps the standard's own "unknown" spelling rather than claiming today.
        "Date": started.replace("-", ".") if started is not None else _PGN_UNKNOWN_DATE,
        # No tournament, so no round to number — the standard's "not applicable".
        "Round": "-",
        "White": "Player" if player_is_white else glitch,
        "Black": glitch if player_is_white else "Player",
    }


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
    def describe_position() -> dict[str, Any]:
        """What is on the board right now, in words: where each side's pieces
        stand, who is ahead in material and by how much, castling, check, and
        the last move. This is how you answer a description ask — "what's the
        position?", "what's on the board?", "where are my pieces?" — call it
        rather than describing the board yourself. It is a description, not a
        verdict: who is *winning* is `evaluate_position`."""
        session = ctx.session
        placement = session.piece_placement()
        castling = session.castling_status()
        history = session.move_history()
        return {
            "ok": True,
            # The facts as prose, for the narrator; the keys below are the same
            # facts for everyone else.
            "summary": _position_summary(session, placement, castling),
            "pieces": placement,
            # Positive means the *player* is ahead — the convention the honesty
            # guard counts in (`VerifiedFacts.material`), so a claim read off
            # this result and a claim checked against the board agree on sign.
            "material": {"player_advantage": session.material_balance()},
            "castling": castling,
            "in_check": session.is_check(),
            # SAN and no mover, under a key that is not `san`: `_verified_facts`
            # reads a top-level `san` off every result as a move the turn
            # played, and naming who played the last ply would tell the
            # narrator whose move is next (#193) — which is the one thing this
            # result must not do.
            "last_move": history[-1] if history else None,
            "move_number": session.fullmove_number,
            "plies": session.plies,
            "game_over": session.is_game_over(),
            "outcome": _outcome_dict(session),
        }

    @registry.tool()
    def evaluate_position() -> dict[str, Any]:
        """Stockfish's verdict on who is better, from White's point of view:
        centipawns, or mate-in-N. This is how you answer a judgment question —
        "who's winning?", "how am I doing?", "is this good for me?" — from the
        result, never from a guess. It says nothing about what is on the board:
        "what's the position?" is a description ask, which is
        `describe_position`."""
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
        and what was best. `played_captures`/`best_captures` name the piece
        each of those moves takes, or are null for a quiet move — never name a
        captured piece the result did not. This is how you answer "how good was
        that move?" or "what was my mistake?" — from the result, never from a
        guess. Defaults to the player's own last move (what 'my mistake'
        means); pass a color to analyze that side's last move instead."""
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
            # What each move takes, so a narration never has to guess the
            # victim of a capture it is describing. None means a quiet move.
            "played_captures": analysis.played_captures,
            "best_captures": analysis.best_captures,
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
            # Whose move this was. Constant on this payload — `make_move` submits
            # the *player's* move and nothing else, with the engine's answer
            # reported separately under `engine_move` — but stated as data rather
            # than left implicit, because the narrator that reads this result has
            # no board and would otherwise have to infer attribution from move
            # parity, which is exactly what it got wrong.
            "mover": _MOVER_PLAYER,
            # The same facts as one sentence, for whoever narrates from them.
            "summary": _move_summary(result, _MOVER_PLAYER),
            "game_over": ctx.session.is_game_over(),
        }
        if atomic_exchange:
            payload["engine_move"] = _engine_move_dict(reply)
            # The settled position, for the caller with no pipeline behind the
            # call. The split payload deliberately reports the move and not the
            # board it left: mid-exchange, that fen/turn describe a position
            # the engine's reply is about to supersede — and `turn` names a
            # side to play for to every narrator that reads the result, which
            # is the engine's color for exactly as long as the reply is still
            # being computed (#193, the "My turn. ...Be6." announcements).
            payload["fen"] = ctx.session.fen()
            payload["turn"] = ctx.session.turn
        return payload

    make_move.__doc__ = _make_move_doc(atomic_exchange)
    registry.tool()(make_move)

    # These two strings are the lever for the two-undo miss: "undo the bishop
    # move and undo the knight move, then play d4 instead" reached
    # `undo(plies=2)` — one exchange for two named moves — and the replacement
    # then landed on a board that still held the knight move
    # (docs/agent-evals.md, `undo_twice_and_replace`).
    #
    # **They carry no numbers at all, and that is measured.** A first cut
    # explained the arithmetic — a ply is a half-move, a player's move and the
    # engine's reply are two of them — and it made the miss worse: 4/20 against
    # the old text's 9/20, sixteen of twenty samples passing `plies=2`. It also
    # spread to the sibling, where `undo_and_replace` held 19/20 but eighteen of
    # those passes stopped omitting the argument and passed a count instead. Any
    # number written here is copied into the argument, so the facts are all
    # about *calls*: a move the player named is a whole takeback and never a
    # count, several named moves are that many calls, and the count itself is
    # the app's to work out. Same lesson as `set_difficulty`'s note below —
    # facts, not triggers — one level down: not even a fact that contains a
    # number, in a description whose argument is a number.
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
        "take that back", "undo the bishop move") omit plies — the app pops the
        whole exchange itself, leaving the player to move again. When the player
        names several moves to take back, call this again for each further named
        move, plies omitted every time. Pass plies only when the player asked for
        an explicit count of half-moves."""
        # Attempt first, abandon second — and only if the takeback happened.
        # A takeback replaces the position the open turn is about, so the turn
        # goes with it (and any reply being computed for it); a *refused* one
        # replaces nothing, and the turn it would have thrown away is still
        # owed an engine reply. Abandoning up front cost exactly that: in one
        # batch, `make_move(e4), undo(plies=100)` left e4 on the board with
        # Black to move, the coordinator back awaiting the player and nobody
        # left to answer (audit 2026-09-05, finding 1). The same rule the other
        # non-move mutations below already follow, because each of them
        # abandons only on the path where its mutation really runs.
        result = ctx.session.undo(_takeback_plies(ctx) if plies is None else plies)
        if not result.ok:
            # Nothing to take back, or a game ended by resignation: asking again
            # with a different count does not conjure plies that were never
            # played, so this one is for the player to hear, not to retry.
            return registry.refusal(result.reason or "cannot undo", RETRY_NEVER)
        coordinator.abandon_turn()
        # An odd count pops the engine's reply and hands the board back with the
        # engine to move — a settled, legal position with nobody scheduled to
        # play it, because a takeback is not a turn and opens none. The
        # coordinator settles it, exactly as it does the opening move of a game
        # taken as black; the default paired takeback leaves the player to move
        # and this reports `None` (audit 2026-09-05, finding 2).
        reply = coordinator.settle_engine_turn()
        return {
            "ok": True,
            "undone": list(result.undone),
            "engine_move": _engine_move_dict(reply),
            "fen": ctx.session.fen(),
            "turn": ctx.session.turn,
        }

    def _gate(name: str, args: dict[str, Any]) -> dict[str, Any] | None:
        """Refuse an unconfirmed destructive call; arm it for the player's yes.

        The prompt asks the agent to confirm before a `DESTRUCTIVE_TOOLS` call,
        but the model honors that only about half the time (docs/agent-evals.md),
        so the rule is enforced here, where the model has no say. A refusal is an
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
        # engine can make. A fresh board as black is one more position left with
        # the engine to move, so it is settled by the same method an odd
        # takeback and a resumed mid-exchange save use. The coordinator owns
        # that call (and the condition): every engine move in the app comes from
        # one place.
        return {
            "ok": True,
            "engine_move": _engine_move_dict(coordinator.settle_engine_turn()),
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
    def claim_draw() -> dict[str, Any]:
        """Claim a draw by threefold repetition or the fifty-move rule. Call this
        as soon as the player asks to claim a draw — do not ask them to confirm
        first. If a game is in progress the result comes back refusing and asking
        you to confirm; relay that to the player and stop, do not call again.
        When it returns ok, the game really did end in a draw."""
        coordinator.require_destructive_budget()
        # Whether a claim exists is board truth, and it is checked *before* the
        # gate for the same reason the budget is: a call that cannot run must not
        # arm a question, because the player's "yes" to it would then fail on a
        # game they were told they could end. `never`, and not for want of
        # classifying it — there are no arguments to vary, and what changes the
        # answer is a move, not another call.
        if not ctx.session.claimable_draws():
            raise ToolError(
                "cannot claim a draw: no draw is available to claim in this position",
                retry=RETRY_NEVER,
            )
        refusal = _gate("claim_draw", {})
        if refusal is not None:
            return refusal
        coordinator.abandon_turn()  # no reply is owed on a game that just ended
        outcome = ctx.session.claim_draw()
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
        """Export the game so far as PGN. The app shows the player the notation
        itself, with a button to copy it, so the reply should say it is ready —
        never recite the moves or the headers."""
        # The description above is a prompt change: live, the narrator read the
        # whole `[Event "?"] …` dump into the bubble and, with voice on, out
        # loud (2026-09-04 walkthrough). The notation is app-owned text now —
        # rendered under the reply with a copy button — so reciting it is the
        # old behaviour rather than a wording preference.
        return {"ok": True, "pgn": ctx.session.export_pgn(pgn_headers(ctx))}

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
        try:
            _write_json_atomic(path, data)
        except OSError as exc:
            # `never`, and not for want of classifying it: a volume that cannot
            # take these bytes will not take them under another name, so the
            # loop's remaining iterations buy nothing and the honest move is to
            # tell the player the game is not on disk. The message is the app's
            # own rather than the errno string — that varies by host and names a
            # path the player has no use for — with the real one logged for
            # whoever has to go fix the disk.
            logger.warning("could not write game save %s: %s", path, exc)
            raise ToolError(
                f"could not write the save file for {name!r}; "
                "no save was created or replaced",
                retry=RETRY_NEVER,
            ) from exc
        # Which board went into the file. Without it two saves under one name
        # answer identically for two different games, and the loop's
        # results-keyed progress rule reads the second as a stall: live,
        # "save as checkpoint, undo, save it again, then play d4" ended the
        # planning phase on the second save and the replacement move was never
        # asked for (audit 2026-09-05, finding 8). The stamp every refusal
        # already carries, on the success too — and deliberately a bare int:
        # `fen`/`turn` would name a side to move to whoever narrates from this
        # result, which is the one thing a result may not do (#193).
        return {"ok": True, "name": name, "board_version": ctx.board_version}

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
        # A save can be taken mid-exchange — "play e4 and save this" writes the
        # board between the player's move and the reply — so the game that comes
        # back can be the engine's to move. Nothing was open to collect it: the
        # obligation died with the session that owed it. The coordinator settles
        # the restored board instead, which is what makes a resumed exchange
        # finish rather than park (audit 2026-09-05, finding 2). `None` when the
        # save was the player's move to make, which is most of them.
        reply = coordinator.settle_engine_turn()
        return {
            "ok": True,
            "name": name,
            "engine_move": _engine_move_dict(reply),
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
    #
    # The text carries the fact the model cannot derive from a schema: strength
    # is this setting and nothing else, because personality is tone only and
    # never shapes move choice. Live, "go easy on me without changing the
    # difficulty" set beginner anyway — with no other lever named, "go easy"
    # had to be read as a difficulty change, and the constraint the player
    # stated had nothing to bind to.
    #
    # Facts, and no triggers. A first cut also listed the asks that mean this
    # call ("go easy, ease up, play harder, or crank it up is this call"), and
    # under the thread the miss happened in that clause measured 9/20 against
    # 20/20 without it (docs/agent-evals.md): an enumerated trigger outranked
    # the caveat that followed it, which is the phrase-list failure #249 warns
    # about, written into a description instead of a regex.
    set_difficulty.__doc__ = (
        "Set how hard the engine plays: pass exactly one of tier (named "
        "level: beginner ~500, casual ~1000, intermediate ~1500, advanced "
        f"~2000, maximum = full strength), skill_level ({SKILL_MIN}-"
        f"{SKILL_MAX}), or elo ({ELO_MIN}-{ELO_MAX}). Prefer tier unless "
        "the player names a number. This is the only thing that changes how "
        "hard the engine plays — there is no softer or sharper style to "
        "switch to. The setting persists and the player owns it: if they ask "
        "for an easier game but rule out changing the difficulty, there is "
        "nothing left to change — say so and ask, do not call this."
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
        """Set how much the app says back: low is one short line, normal is
        the default, high adds a remark on the position. This is the only way
        to change it — the setting persists across turns and the player owns
        it, so any ask to talk more or less, be briefer, chattier, or quieter
        is this call and not a change of style in one reply."""
        ctx.settings.verbosity = verbosity
        return {"ok": True, "verbosity": verbosity}

    @registry.tool()
    def set_voice_output(enabled: bool) -> dict[str, Any]:
        "Turn spoken (TTS) output on or off."
        ctx.settings.voice_output = enabled
        return {"ok": True, "voice_output": enabled}

    return registry
