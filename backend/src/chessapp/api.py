"""HTTP API: game lifecycle + state fetch for the board UI.

The API is trusted code, so it drives `GameSession` directly through the
shared `ToolContext`; the tool registry remains the LLM-only boundary.
Conventions:

- Illegal moves are data (`legal: false`), not HTTP errors — legality is
  the engine's answer, not a transport failure.
- Domain failures on mutations (nothing to undo, resigning a finished
  game, an action that doesn't belong in the current turn phase) are 409s.
- Board mutations run through the shared `TurnCoordinator`, which owns the
  turn sequence and the engine's reply: when the context has an engine and
  the player's move leaves the game running, the engine replies in the same
  request — that's the LLM-off vs-Stockfish mode the app must always
  support, now going through the same machine the agent's moves do.
  Any non-move mutation (undo, new game, resignation) abandons an open turn
  first, here as in the tools: the position that turn was about is the thing
  being replaced.
- **`/api/game/move` is mode-aware** (audit items 1/4). With a brain
  configured — agent mode — a dragged move runs the same beats the command
  pipeline's fast path runs (`_play_move`: dispatch `make_move`, let Glitch
  react to the verified player move, collect the reply, close the turn), so a
  drag-played game produces Glitch turns too and the drag is *not* a silent
  bypass. With no brain — direct mode — it runs the atomic exchange it always
  ran, answering byte-for-byte what it answered before: LLM-off play is a
  binding invariant, and `/api/settings`' `agent_available` is what makes the
  mode visible in the UI rather than a per-input surprise. `CHESSAPP_AGENT=off`
  is how a deployment selects it (`app._agent_enabled_from_env`) — the mode was
  written and correct long before anything could reach it.
- **Every mutation is version-checkable and serialized** (audit item 7). The
  state document carries `version` (`ToolContext.board_version`), and every
  mutating request may carry the one it last saw: superseded means 409 with
  `stale: true` and the current state, omitted means today's behavior. The check
  and the mutation happen under `ctx.mutation_lock` and cannot be split, so two
  clients on the one shared session cannot advance the same turn twice — see
  `_mutation` and `docs/turn-coordinator.md`.
- **The destructive-op gate is one system.** `/api/game/new`,
  `/api/game/resign` and `/api/game/claim-draw` dispatch through the registry,
  so the same deterministic gate that refuses an unconfirmed
  `new_game`/`resign`/`claim_draw` for the agent arms `ctx.pending` for a
  button press too: mid-game those endpoints answer 409
  with the gate's question (`confirm: true`), and `/api/game/confirm` answers
  it. It is the *same* armed op the spoken road uses, so a question asked by a
  button can be answered by a typed "yes" and vice versa — they are one origin
  (`tools.PANEL_ORIGIN`), the player at their own screen. A delegate
  conversation is not: its question is answered in that conversation and
  nowhere else (`tools.PendingOp.origin`, #281). Undo is not destructive and
  keeps its direct endpoint.

Always read `ctx.session` per request: `resume_game` swaps the session
object on the context.
"""

import asyncio
import logging
import mimetypes
import random
import re
import time
from collections.abc import (
    AsyncIterator,
    Awaitable,
    Callable,
    Iterator,
    Sequence,
)
from contextlib import asynccontextmanager, contextmanager, suppress
from copy import deepcopy
from dataclasses import dataclass
from dataclasses import field as dc_field
from pathlib import Path
from typing import Annotated, Any, Literal

import anyio.to_thread
import chess
import chess.engine
from fastapi import (
    FastAPI,
    Header,
    HTTPException,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, model_validator

from chessapp import clarification
from chessapp.agent_api import (
    CONVERSATIONS_FILENAME,
    MAX_AGENT_MESSAGE_LENGTH,
    ConversationStore,
    build_agent_router,
)
from chessapp.analysis import review_game
from chessapp.brain import (
    CALL_FAILED,
    CALL_LATE,
    CALL_OK,
    CANCEL,
    CONFIRM,
    PHASE_ANSWER,
    PHASE_REACTION,
    PHASE_REWRITE,
    PHASE_UNKNOWN,
    Brain,
    ModelCall,
    Narration,
)
from chessapp.coordinator import TurnCoordinator, TurnPhase, TurnStateError
from chessapp.deadline import LateReaction
from chessapp.deadline import within_budget as _within_budget
from chessapp.engine import validate_elo, validate_skill_level, validate_tier
from chessapp.facts import (
    TurnEvidence,
    analysis_moves,
    assemble,
    relative_outcome,
    reported_moves,
    settings_of,
)
from chessapp.fastparse import parse_confirmation, parse_move, parse_resign
from chessapp.game import GameSession, MoveResult
from chessapp.honesty import (
    Unverified,
    VerifiedFacts,
    corrections,
    unlicensed_advice,
    unverified,
)
from chessapp.progress import ProgressEvent, ProgressReporter
from chessapp.provider import ProviderError
from chessapp.tools import (
    CONFIRM_QUESTIONS,
    PANEL_ORIGIN,
    UNDO_PLIES_MAX,
    ToolContext,
    ToolRegistry,
    build_registry,
    confirm_pending,
    live_checkpoint,
    pgn_headers,
    saved_game_names,
    write_live_checkpoint,
)
from chessapp.trace import KIND_SPEECH as TRACE_SPEECH
from chessapp.trace import KIND_VOICE as TRACE_VOICE
from chessapp.trace import (
    ROUTE_BOARD,
    ROUTE_BRAIN,
    ROUTE_CONFIRMATION,
    ROUTE_CONTROL,
    ROUTE_FAST_PATH,
    ROUTE_RESIGN,
    TRACE_SCHEMA,
    Tracer,
    new_correlation_id,
    turn_record,
)
from chessapp.voice import SpeechClient

logger = logging.getLogger(__name__)


class StaleVersionError(Exception):
    """A mutating request about a board that has already moved on.

    Raised by the mutation guard *before* anything changes, and turned into the
    409 below by an app-wide handler — which is why it is an exception rather
    than a returned response: the check happens inside a context manager
    wrapping the whole mutation, and the one thing a guard must be able to do
    from there is stop the body from running at all.
    """

    def __init__(self, expected: int, current: int, state: dict[str, Any]) -> None:
        super().__init__(f"stale board version {expected}; current is {current}")
        self.expected = expected
        self.current = current
        self.state = state


class WrongGameError(StaleVersionError):
    """A mutating request about a game that is no longer on the board.

    A `StaleVersionError` — a new game or a resume bumps the version too, so
    this is the same refusal with a truer message — for a client that sent the
    `game_id` it was playing rather than (or as well as) a version.
    """

    def __init__(self, expected_game: str, state: dict[str, Any]) -> None:
        super().__init__(-1, state["version"], state)
        self.expected_game = expected_game


class VersionedRequest(BaseModel):
    """The optional board-version precondition every mutating request carries.

    `version` is the `state.version` the client last saw. Supplied and still
    current, the request proceeds; supplied and superseded, it is refused 409
    without touching the board (`StaleVersionError`). Omitted — the default —
    is exactly today's behavior, because the precondition is a client's opt-in:
    the board UI adopts it when it is ready to, and a client that never heard
    of versions keeps playing.

    Optional rather than required for one more reason: the version is a
    *transport* fact. Nothing about a legal move depends on it, so a request
    that doesn't care about concurrency should not have to prove anything.
    """

    version: int | None = None
    # The same opt-in, one level up (#291): the `state.game_id` the client is
    # playing. A version is bumped by every move; this changes only when the
    # game does — a new game, a resume — so it is the precondition for "this
    # game, whatever has happened on it since". A mismatch is the same 409.
    game_id: str | None = None


class MoveRequest(VersionedRequest):
    move: str


# What an interaction id may look like: short and inert, because it is echoed
# into a trace file a person reads. Anything else is refused (422) on a body
# and ignored on a header, where refusing would cost the player their speech.
_ID_PATTERN = r"^[A-Za-z0-9_-]{1,64}$"
_ID_MAX = 64


def _header_id(value: str | None) -> str:
    """An interaction id off a header, or "" when absent or not id-shaped."""
    return value if value is not None and re.fullmatch(_ID_PATTERN, value) else ""


class CommandRequest(VersionedRequest):
    # The delegate's cap, on the panel's route too (#288): one command is what
    # the planner and the narrator both carry into their prompts, so an
    # unbounded one is an unbounded prompt. Refused 422 before any turn opens.
    text: str = Field(max_length=MAX_AGENT_MESSAGE_LENGTH)
    # The browser's id for the interaction this command belongs to (#317): the
    # same id rides the transcription before it and the speech request after,
    # so the three records join. Optional — a typed client that sends none
    # loses nothing but the join.
    interaction_id: str | None = Field(
        default=None, max_length=_ID_MAX, pattern=_ID_PATTERN
    )


class SpeakRequest(BaseModel):
    text: str = Field(min_length=1, pattern=r"\S")


# The browser's milestones for one interaction (#317). Offsets in its own
# monotonic clock from the interaction's start — the moment the VAD heard the
# player stop (`speech_end`) or the command was submitted (`submit`) — and
# never compared with a server time: the two clocks share no origin.
VoiceMark = Literal[
    "stt_done",
    "command_sent",
    "first_board_update",
    "engine_reply",
    "command_done",
    "tts_requested",
    "tts_ready",
    "playback_started",
    "playback_ended",
]
# A reading past an hour is no interaction this report is about.
_MARK_CEILING_MS = 3_600_000.0


class VoiceTelemetry(BaseModel):
    """One interaction's client-side milestones, as the browser reports them.

    Bounded on every axis — known mark names only, numeric offsets inside an
    hour, a closed outcome vocabulary, no extra fields — because it is written
    verbatim to a file on the server and nothing about it is trusted."""

    model_config = ConfigDict(extra="forbid")

    interaction_id: str = Field(max_length=_ID_MAX, pattern=_ID_PATTERN)
    correlation_id: str | None = Field(
        default=None, max_length=_ID_MAX, pattern=_ID_PATTERN
    )
    origin: Literal["voice", "typed"]
    clock: Literal["client_monotonic_ms"]
    start: Literal["speech_end", "submit"]
    marks: dict[VoiceMark, Annotated[float, Field(ge=0, le=_MARK_CEILING_MS)]]
    outcome: Literal[
        "ended",
        "error",
        "interrupted",
        "blocked",
        "no_audio",
        "timeout",
        "silent",
        "no_command",
    ]
    # True when the interaction was given up on before it settled (the
    # client's ceiling): its last mark is a lower bound, not a finish.
    censored: bool = False


class VoiceOutputRequest(BaseModel):
    enabled: bool


class NewGameRequest(VersionedRequest):
    """`color` is the side the player takes; `random` (the default) rolls."""

    color: str = Field(default="random", pattern="^(white|black|random)$")


class UndoRequest(VersionedRequest):
    """None means "the player's takeback": vs the engine that's the full
    exchange (their move plus the engine's reply), engine-free one ply —
    the endpoint decides from the live context."""

    plies: int | None = Field(default=None, ge=1, le=UNDO_PLIES_MAX)


class ResignRequest(VersionedRequest):
    color: str | None = Field(default=None, pattern="^(white|black)$")


class ClaimDrawRequest(VersionedRequest):
    """No fields of its own: which rule the claim lands under is board truth
    (`GameSession.claim_draw`), never the caller's choice — so the body carries
    only the version precondition every mutating request may."""


class OfferDrawRequest(VersionedRequest):
    """No fields of its own: the answer is the rule's (`chessapp.draw_offer`),
    so the body carries only the version precondition."""


class ConfirmRequest(VersionedRequest):
    """The player's answer to an armed destructive op: yes runs it, no drops
    it. The same two answers `parse_confirmation` reads off a spoken turn, and
    the same one `ctx.pending` — a question asked by a button can be answered
    in words, and the other way round."""

    confirm: bool


class DifficultyRequest(BaseModel):
    """Exactly one of tier / skill_level / elo — the same contract as the
    `set_difficulty` tool. Range is validated by the engine when applied."""

    tier: str | None = None
    skill_level: int | None = None
    elo: int | None = None

    @model_validator(mode="after")
    def _exactly_one(self) -> "DifficultyRequest":
        given = [v for v in (self.tier, self.skill_level, self.elo) if v is not None]
        if len(given) != 1:
            raise ValueError("pass exactly one of tier, skill_level, or elo")
        return self


def _outcome_dict(session: GameSession) -> dict[str, Any] | None:
    outcome = session.outcome()
    if outcome is None:
        return None
    return {
        "termination": outcome.termination,
        "winner": outcome.winner,
        "result": outcome.result,
    }


def _state_dict(ctx: ToolContext) -> dict[str, Any]:
    """Take one coherent snapshot of the full state the board UI renders.

    Reads share the mutation boundary rather than merely reading the version
    beside a series of live session fields. Otherwise a move can land after the
    old FEN is read and produce a document whose version/FEN describe one board
    while its turn/history describe the next.

    Callers already inside `_mutation` must use `_state_dict_unlocked`: the
    context deliberately owns a plain, non-reentrant lock.
    """
    with ctx.mutation_lock:
        return _state_dict_unlocked(ctx)


def _state_dict_unlocked(ctx: ToolContext) -> dict[str, Any]:
    """Serialize UI state while the caller holds `ctx.mutation_lock`.

    Takes the context rather than the session because `version` is the
    context's (`ToolContext.board_version`) — a resumed game replaces the
    session, and that replacement is one of the things a client has to be able
    to notice. Every mutation response embeds this document, so the version a
    client needs for its *next* request always comes back with the answer to
    this one.
    """
    session = ctx.session
    return {
        "version": ctx.board_version,
        # Which game this board is (#291): a client that wants to act only on
        # the game it was playing sends it back as `game_id`.
        "game_id": session.game_id,
        "fen": session.fen(),
        "turn": session.turn,
        "player_color": session.player_color,
        "game_over": session.is_game_over(),
        "outcome": _outcome_dict(session),
        "history": session.move_history(),
        "fens": session.position_fens(),
        "captured": session.captured_pieces(),
        "legal_moves": session.legal_moves(),
        "dests": session.legal_destinations(),
        # Which draw rules the side to move may claim right now (empty when
        # none). A claim is the one ending the rules leave to the *player*, so a
        # client cannot offer it without being told it exists — and it is named
        # rather than flagged so the offer can say which rule, exactly as
        # `outcome` names the termination once one is taken.
        "claimable_draws": list(session.claimable_draws()),
    }


def _session_snapshot(ctx: ToolContext) -> tuple[int, GameSession]:
    """Detach the game from the live board, under the mutation boundary.

    The reads that take real time — the whole-game review's Stockfish sweep, the
    hint's search — must not run against `ctx.session` itself. A read that lands
    between a mutation's two steps walks a half-applied board: a `to_dict()`
    inside `undo`'s pop found an empty move stack and raised out of
    `board.root()`, which reached the player as a 500 (#230). Nor may they hold
    the lock while they run: a turn already holds it while the brain thinks, so
    a multi-second sweep holding it too would stall every concurrent drag — a
    worse failure than the tear.

    So the boundary buys a **copy**, and nothing else happens inside it. Held
    across the cheap, pure serialization only (`to_dict`, plus the version that
    names it); the replay that turns those strings back into a board is
    deterministic and touches nothing shared, so it happens after the release.
    A plain `board.copy()` would not do — it walks the move stack, which is the
    very thing that tears.

    The version is the one the returned position *is*, captured in the same
    indivisible step. That is what makes the hint's `version` (#218) exact
    rather than merely fail-safe: the search runs on this copy, so the number
    and the analyzed board cannot come apart.

    Callers already inside `_mutation` must not use this: the context
    deliberately owns a plain, non-reentrant lock.
    """
    with ctx.mutation_lock:
        version = ctx.board_version
        data = ctx.session.to_dict()
    return version, GameSession.from_dict(data)


def _agent_state_dict(ctx: ToolContext) -> dict[str, Any]:
    """The view the brain reasons from: board truth (fen, turn, check, SAN
    history, captures, legal moves, outcome) plus which color the player is and
    which games are saved. Deliberately not `_state_dict` — the UI document's
    per-ply `fens` and `dests` are prompt noise that grows every move and never
    helps the agent.

    `player_color` is read from the session, which owns it: whose *turn* it is
    is board truth and changes every ply, but which side the human plays is
    session state and doesn't.

    `saved_games` is here for the same reason `legal_moves` is: it is a fact the
    app holds and the model would otherwise have to infer. Without it, the only
    thing in context claiming to know about saves was the agent's own past
    prose — and one stale "saving isn't set up" turn was enough to make it deny
    a save sitting on disk (the self-poisoning bug, trace review 2026-07-13).
    Read fresh every turn, so a game saved this session is visible the next.

    `settings` is here for that same reason: difficulty, voice output and
    verbosity appeared nowhere in the agent's per-turn view, so "how hard am I
    playing?" could only be answered from stale conversation text — the
    self-poisoning shape again. Kept small (it ships in the prompt every turn
    on a 12B): only the one difficulty field `Settings` actually has set, so
    the block can never imply two difficulties are in force at once.

    `captures` is what each legal capturing move takes, and it is here for the
    reason `legal_moves` is: a fact the app holds and the model would otherwise
    have to infer. SAN says that `exd5` captures and never what, and a 12B
    cannot read the victim off the FEN — so "take the pawn" on a board with
    nothing to take was answered "which pawn?" (2026-09-04 walkthrough), the
    planner having no way to see that no capture existed. With the list in
    view, a capture asked for by its victim resolves the way a move asked for
    by its square does: one match plays, several ask, none is refused.

    Verbosity was the last one out, and its absence cost a real behavior
    (walkthrough #3): the narrator's prompt carries a *layer* for the current
    verbosity, so the model was told how to talk but never what the setting
    was. With no setting in view, "talk more" is a note about style that the
    turn can satisfy by talking more — which it did, twice, while the setting
    stayed `low` on disk and the next turn was terse again. A player-owned
    setting the model cannot see is one it cannot be asked to change.
    """
    session = ctx.session
    return {
        "fen": session.fen(),
        "turn": session.turn,
        "player_color": session.player_color,
        "in_check": session.is_check(),
        "game_over": session.is_game_over(),
        "outcome": _outcome_dict(session),
        "history": session.move_history(),
        "captured": session.captured_pieces(),
        "legal_moves": session.legal_moves(),
        "captures": session.legal_captures(),
        "saved_games": saved_game_names(ctx),
        "settings": _agent_settings_dict(ctx),
    }


# Why a question the planner may still see in the transcript no longer stands
# (`closed_question`), by `clarification.staleness`'s reason.
_CLOSED_BECAUSE = {
    clarification.BOARD_CHANGED: "the board changed after it was asked",
    clarification.GAME_CHANGED: "a different game is on the board now",
}


def planner_state(
    state: dict[str, Any],
    question: clarification.Clarification | None,
    expired: clarification.Closed | None,
) -> dict[str, Any]:
    """The opening board state the planner reads (#319): the agent view plus,
    when there is one, the question its conversation has open.

    `open_question` is the player's own ask and the moves they were asked to
    choose between — the record, not the narrator's wording of it, so it
    survives a rephrased question, asides, the digest dropping Glitch's turns
    and the input budget trimming the conversation (the state block is never
    trimmed). What the player now means by "the one to f3" is still the
    planner's to read; the answer is still an ordinary `make_move` against
    `legal_moves`. It holds no board fact of its own — no FEN, no menu — so it
    can never be a second, ageing copy of the position, and it is only here
    while `ToolContext.live_clarification` says the question stands.

    `closed_question`, once, is the other half: a question the transcript
    still shows, about a board that is gone. Without it "the first one" after
    another client moved reads as an answer to a question nothing stands
    behind any more.

    Built from what the turn already read rather than by reading again,
    because the read drops a stale record: one read per turn, one report.
    Deliberately not in `_agent_state_dict`, which the narrator's views are
    derived from and the turn's change detection compares.
    """
    view = dict(state)
    if question is not None:
        view["open_question"] = {
            "player_asked": question.request,
            "choose_between": list(question.candidates),
        }
    elif expired is not None:
        view["closed_question"] = {
            "player_asked": expired.record.request,
            "why": _CLOSED_BECAUSE.get(expired.reason, "it no longer stands"),
        }
    return view


def _agent_settings_dict(ctx: ToolContext) -> dict[str, Any]:
    """The live settings the brain is shown: difficulty (exactly the one field
    of tier / skill_level / elo that is set), voice output, and verbosity."""
    settings = ctx.settings
    difficulty: dict[str, Any] = {}
    for field in ("tier", "skill_level", "elo"):
        value = getattr(settings, field)
        if value is not None:
            difficulty = {field: value}
            break
    return {
        "difficulty": difficulty,
        "voice_output": settings.voice_output,
        "verbosity": settings.verbosity,
    }


def _narrator_state_dict(ctx: ToolContext) -> dict[str, Any]:
    """The view the narrator speaks from: the agent view minus every spelling
    of "it is your move" — no `turn`, no `legal_moves`, and no `fen`, whose
    string itself names the side to move.

    The narrator reacts mid-turn, from the board the player's move just left —
    a board where it is the engine's move and the legal moves are the engine's
    options. Handed that as data, it treats the reaction beat as a
    move-selection beat: #188 cut "you are playing black" from the brief and
    the next game announced a reply all the same ("My turn. ...Be6.", #193),
    because `turn` beside `player_color` says the same thing in JSON and
    `legal_moves` is the menu to pick from. What commentary actually uses
    stays: the game so far, which color the player is (capture talk needs its
    direction), the captures, the outcome once there is one, and the saves and
    settings it may be asked about. The planner keeps the full view — mapping
    an utterance onto a move is what `legal_moves` exists for.

    Derived by deletion rather than built up, so the two views cannot drift
    apart on the facts they share; the deletion list is the invariant.
    """
    state = _agent_state_dict(ctx)
    # `captures` goes with `legal_moves`: it is the same menu, narrowed to the
    # moves that take something, and a menu is exactly what the narrator must
    # not be handed.
    for key in ("fen", "turn", "legal_moves", "captures"):
        del state[key]
    return state


# What a mid-command refresh carries: the menu, and the facts that say whose it
# is and whether there is one. Named against `_agent_state_dict`'s keys rather
# than re-derived from the session, so the block that supersedes the opening one
# cannot describe the position in a second vocabulary — a rename there is a
# `KeyError` here on the first mutating command, and a test pins the
# containment. `player_color` is in it because `new_game` can change it
# mid-command ("new game as white and open e4"); `game_over` because it is what
# empties the menu.
_REFRESH_KEYS = (
    "turn",
    "player_color",
    "in_check",
    "game_over",
    "legal_moves",
    "captures",
)


def planner_board_refresh(
    ctx: ToolContext, coordinator: TurnCoordinator
) -> dict[str, Any] | None:
    """The board as the planner's *next* decision inside a command must see it.

    The opening state block is the only `legal_moves` the brain's loop ever
    holds, and it is stale the moment one of the turn's own tools mutates —
    while the planner's contract says every move it submits must be an entry of
    that list (#282). No tool result can close the gap either: a mutation
    reports `fen`/`turn`/`engine_move`, never the menu, and it must not report
    the menu, because the same results are what the narrator speaks from (the
    reason `save_game` answers with a bare `board_version`). This is the view
    that supersedes the opening block, and it reaches the planner alone.

    It carries the menu and the little that qualifies it, and deliberately not
    the rest of the opening block. Everything else that can move inside a
    command is already in the planner's context by the time it decides again:
    each settings tool answers with its own new value, `save_game` with the
    name it wrote, and every mutating tool describes what it did — `undo`
    reports the moves it took back, `make_move` the move it made. The
    legal-move menu is the one fact nothing reports.

    `history` is the pointed omission, and it was measured: the first cut sent
    the whole view, and `undo_twice_and_replace` went 19/20 → 4/20 (interleaved
    blocks of five against unchanged main on one server, 2026-09-17). Every
    miss was the same — "undo the bishop move and undo the knight move, then
    play d4" took one exchange back and played d4 on a board still holding the
    knight move. A history the bishop move has just left reads to a 12B as the
    takebacks being done, so a block meant to tell the planner what it *may
    play* was answering a question about what it had already *finished*. The
    tool's own result says what it undid; this says what is legal now.

    `None` when a reply is owed. Mid-exchange the side to move is the engine's
    and `legal_moves` is the engine's menu, and handing a move-choosing phase
    that menu is #193's shape one layer up — which is exactly why `make_move`'s
    split payload reports the move and not the board it left. Nothing is lost:
    a second player move under one turn is refused by the phase machine, and
    the move that landed is fully described by its own result. The invariant is
    that the planner sees a board the player is to move on, or no board at all.

    Takes no lock: the command already holds `ctx.mutation_lock` while the
    brain thinks, and that lock is not reentrant.
    """
    if coordinator.phase in (
        TurnPhase.PLAYER_MOVE_APPLIED,
        TurnPhase.AGENT_OBSERVING,
    ):
        return None
    # `board_version` rides along because it is what the loop decides on: a
    # refresh is sent when the *board* is a different one, and this is the
    # app's counter for that. The rest of the view is here because it is the
    # planner's own block re-read — the same dict, so the two cannot describe
    # the position in two different vocabularies — and a fact in it that moved
    # without the board moving (a setting, a save) is one the tool that moved
    # it already reported. It is also the vocabulary every refusal speaks, so a
    # rejected call's version and this block's line up.
    state = _agent_state_dict(ctx)
    return {"board_version": ctx.board_version} | {
        key: state[key] for key in _REFRESH_KEYS
    }


def narrator_facts(ctx: ToolContext, coordinator: TurnCoordinator) -> dict[str, Any]:
    """The game as the brain route's narrator may state it (#289).

    Read once, as the planner hands off — after every tool of the turn has
    run and before the engine's reply is collected — so it is the board the
    narrator's words will be spoken over. Until this existed the brain route's
    narrator had no board at all, only whatever the tool results carried.

    A subset of `_narrator_state_dict`, and deliberately a small one. No side
    to move, for #193's reason. No `history`, for the refresh block's reason
    one phase earlier: a history the turn's own undo has just shortened reads
    to a 12B as the ask being finished, and every move this turn made is in
    its own result. No saves or settings: the tools that change them report
    their new values, and the prompt's verbosity layer is already there. The
    outcome is the player-relative one the guard certifies (`relative_outcome`,
    #287), so the narrator is told who won in the same words it is checked in.

    `reply_owed` is the coordinator's: the player's move landed and the
    engine's answer has not. The brain lifts it out of the facts into the
    handoff, where it becomes the line telling the narrator the reply is the
    app's to announce.
    """
    return {
        "player_color": ctx.session.player_color,
        "in_check": ctx.session.is_check(),
        "game_over": ctx.session.is_game_over(),
        "outcome": relative_outcome(ctx.session),
        "captured": ctx.session.captured_pieces(),
        "reply_owed": coordinator.phase
        in (TurnPhase.PLAYER_MOVE_APPLIED, TurnPhase.AGENT_OBSERVING),
    }


def _move_dict(result: MoveResult) -> dict[str, Any]:
    return {"legal": result.legal, "san": result.san, "uci": result.uci}


def _move_reply_dict(reply: MoveResult | None) -> dict[str, Any] | None:
    """The engine's reply for the trace record: `{"san", "uci"}`, or None when
    none was owed. It no longer rides inside a tool result, so this is the only
    place a traced move turn can learn what answered it."""
    if reply is None:
        return None
    return {"san": reply.san, "uci": reply.uci}


def _settled_engine_move(
    tool_results: Sequence[dict[str, Any]],
) -> MoveResult | None:
    """The engine move a restore already played, rebuilt from the tool result.

    `new_game`, `undo` and `resume_game` can each leave the engine to move —
    a game taken as black, an odd-ply takeback, a save written mid-exchange —
    and the coordinator settles that board inside the call rather than opening a
    turn nobody would close. So by the time the command converges there is
    nothing left to collect, and the only record of the move is the
    `engine_move` the result carries, in the same shape `make_move`'s atomic
    result uses.

    The *last* one, because a command can restore twice (resume, then undo) and
    the only move worth announcing is the one that answers the board the player
    is left looking at. None when this command settled nothing.
    """
    for record in reversed(tool_results):
        result = record["result"]
        if result.get("ok") is not True:
            continue
        if played := result.get("engine_move"):
            return MoveResult(legal=True, san=played["san"], uci=played["uci"])
    return None


def _destructive_confirmation(
    name: str, result: dict[str, Any], session: GameSession
) -> str:
    """Deterministic stand-in for the reaction after a confirmed destructive op
    at verbosity=low — the twin of `_move_confirmation`, keeping a confirmed
    destructive op a zero-LLM turn like a plain move is.

    `new_game`, `resume_game` and `save_game` leave a game to play and are
    reported by what they did; every other op here ended one, and an ending is
    reported by its outcome (a resignation's result, or the half point a claimed
    draw produces) rather than by name.

    It says nothing about the engine's opening move on a game taken as black.
    That used to be spelled here, and is now one case of a rule with three: a
    restored board the coordinator settled reports its move under `engine_move`
    whichever tool restored it, and the pipeline announces any of them with the
    one line every other engine move gets (`_reply_announcement`, composed
    around whatever spoke for the turn). Two composers for one fact would have
    said it twice on this route.
    """
    if name == "new_game":
        return "New game."
    # The two gated ops that end no game (#291): each is reported by what it
    # did, never by an outcome — a resumed game is very much still on.
    if name == "resume_game":
        return f"Loaded {result.get('name', 'the saved game')}."
    if name == "save_game":
        return f"Saved over {result.get('name', 'the old save')}."
    outcome = result.get("outcome") or _outcome_dict(session)
    if outcome:
        return f"Game over: {outcome['result']} ({outcome['termination']})."
    return "Game over."


def _reply_announcement(reply: MoveResult | None, session: GameSession) -> str:
    """The close beat, in the app's own words: the engine's reply, plus the
    outcome if that reply ended the game. Empty when no reply was owed.

    Deterministic on purpose. The turn's one narration already happened — during
    the observation beat, while this very move was being computed — and asking
    Glitch to react to the reply as well would cost a second round trip on every
    move, which is precisely the latency the observation beat is required not to
    add. So the reaction is the model's and the announcement is the app's.
    """
    parts: list[str] = []
    if reply is not None and reply.san:
        parts.append(f"{reply.san}.")
    if session.is_game_over():
        outcome = _outcome_dict(session)
        if outcome:
            parts.append(f"Game over: {outcome['result']} ({outcome['termination']}).")
    return " ".join(parts)


def _move_confirmation(
    result: dict[str, Any], reply: MoveResult | None, session: GameSession
) -> str:
    """Deterministic stand-in for a whole fast-path move turn: the player's move,
    the engine's reply, and the outcome if the game ended — facts from the two
    results, zero LLM calls. What verbosity=low says, and what a failed
    observation degrades to."""
    return " ".join(
        part
        for part in (f"{result['san']}.", _reply_announcement(reply, session))
        if part
    )


def _move_commentary(
    reaction: str,
    result: dict[str, Any],
    reply: MoveResult | None,
    owed_reply: bool,
    session: GameSession,
) -> str:
    """The words for one move turn: Glitch's reaction to the verified player
    move, then the app's own line announcing what answered it.

    Shared by every route that plays a move — the fast path, a board drag — so
    the two cannot drift apart in what a move turn *says*. With no reaction to
    show (verbosity=low, a provider failure) one canned confirmation covers the
    move and the reply together. `owed_reply` is why the announcement is not
    derived from `reply` alone: a game-ending player move is owed nothing, and
    its outcome already belongs to the reaction's turn rather than to a reply
    that never came.
    """
    if not reaction:
        return _move_confirmation(result, reply, session)
    if owed_reply and (line := _reply_announcement(reply, session)):
        return f"{reaction}\n\n{line}"
    return reaction


def _late_close_words(
    tool_results: Sequence[dict[str, Any]],
    reply: MoveResult | None,
    owed_reply: bool,
    changed: bool,
    session: GameSession,
) -> str:
    """What a brain-route turn says when its closer was too late (#316).

    The brain route's version of `_move_confirmation`, which is what the fast
    path says when *its* reaction is late: the player's moves the turn played
    and the engine's answer, facts from the results. Not `STUCK_REPLY` on a
    turn that moved something — "say it again" would invite replaying a move
    that already landed — so a turn that changed the board some other way says
    what the lost-brain line says when a turn stands. Only a turn that changed
    nothing is left to ask again.
    """
    played = [
        f"{record['result']['san']}."
        for record in tool_results
        if record["name"] == "make_move"
        and record["result"].get("legal") is True
        and record["result"].get("san")
    ]
    if played or owed_reply:
        if line := _reply_announcement(reply, session):
            played.append(line)
    if played:
        return " ".join(played)
    return PROVIDER_LOST_TURN_STANDS if changed else STUCK_REPLY


# What the player hears when the brain's loop ran out of budget instead of
# answering (`max_iterations` / `correction_limit`): those stops carry no
# commentary, and an empty bubble would read as a crash. Public so tests pin
# the substitution, not a wording.
STUCK_REPLY = "I lost the thread on that one — say it again?"
_DECLINED_REPLY = "Alright, keeping it. Your move."

# What the player hears when the provider died mid-turn (audit item 20). Two
# lines because the two cases carry opposite retry advice, and that advice is a
# fact the code knows: nothing changed means saying it again is safe; something
# changed means repeating the utterance could replay a move, so the line says
# the work stands instead of inviting a retry. Public so tests pin the
# substitution, not a wording.
PROVIDER_LOST_RETRY = (
    "My brain cut out before anything happened — the board is untouched. Say it again."
)
PROVIDER_LOST_TURN_STANDS = (
    "My brain cut out mid-turn, but everything it already did stands."
)

# What the player hears when *Stockfish* died after their move was already on
# the board (#284). The same deal as the lines above, one layer down: the move
# is committed and broadcast, so the failure is not the request's — it is a
# turn with half its moves in it, and the player is owed the fact rather than a
# 500. It says the reply is still owed because it is: the coordinator puts the
# phase back to `player_move_applied` and the next command settles it. Public
# so tests pin the substitution, not a wording.
ENGINE_LOST_REPLY_OWED = (
    "My engine dropped out before it answered — your move stands, "
    "and I still owe you a reply."
)


def _engine_lost_words(commentary: str) -> str:
    """The engine-lost line composed around whatever the turn already said.

    After it, not before: this line stands exactly where the reply
    announcement would have (`_reply_announcement`), because it is the same
    sentence's negative — the app reporting what answered the player's move,
    and here what did not. The lost-brain line leads instead, for the opposite
    reason: it explains why there is nothing else to read.
    """
    return (
        f"{commentary}\n\n{ENGINE_LOST_REPLY_OWED}"
        if commentary
        else ENGINE_LOST_REPLY_OWED
    )


# How long the app waits for Glitch's *optional* words before going on without
# them (#283). The reaction is optional by construction — the coordinator starts
# Stockfish the moment the player's move lands and collecting the reply is legal
# with or without a narration — but until this it was only optional in the sense
# that it could be *skipped*, never that it could be *late*: the reply was
# applied after the words came back, so a stalled narrator held an answer already
# sitting in memory and, with it, the mutation lock every other road onto the
# board waits on.
#
# Measured rather than derived from the token cap, which bounds generation and
# not queueing or a dead server. Across 58 observe beats in the deployed trace
# (routes `fast_path` and `board`, one thinking-off narrator call each) the
# reaction took 0.7–2.1 s, median ~1.5 s, with a single 7.5 s outlier. Ten
# seconds is above every reaction ever observed with room for the shared GPU
# having a bad minute, and still far below the point where a player decides the
# board is frozen. A cold llama-swap load (~100 s to first byte, first move
# after a reboot) is over it and loses that one reaction to the app's own line;
# hanging up does not unload the upstream, so the next turn is warm.
_REACTION_BUDGET_S = 10.0


def _failure_name(exc: BaseException) -> str:
    """One failure, as the short string a record can carry: class and message.

    The trace's `provider_failure` names a *kind* from a vocabulary the brain
    owns; nothing owns a vocabulary for a dying Stockfish or for whatever else
    escapes a turn, so the exception names itself. Class alone when it carries
    no message — `EngineTerminatedError` is already the whole story.
    """
    detail = str(exc).strip()
    return f"{type(exc).__name__}: {detail}" if detail else type(exc).__name__


# There is no canned line for a claim the honesty guard cuts. Until 2026-09-10
# there were three ("Scratch that — the game's still live..."), and every one
# of them put the app's words in Glitch's mouth on a turn the player had heard
# nothing wrong on yet — the guard runs before anything is spoken, so the line
# apologised for a sentence nobody heard, and a false positive cost the whole
# reply. Now a cut claim goes back to the narrator with the true facts
# (`_honest_words`), and only a second draft that still asserts something the
# board does not back falls through to what the app already says when the
# model has nothing usable: the move confirmation on a move turn, `STUCK_REPLY`
# on any other.

# The confirmation question for a resignation the pipeline itself dispatched.
# Deterministic, like the gate it came from: the model is not consulted about a
# resignation at any point, including how to ask about one.
_RESIGN_CONFIRM = "That's the game if you mean it. Say yes and I'll resign for you."

# The same questions for a destructive op the *board UI* armed live with the
# gate (`tools.CONFIRM_QUESTIONS`), where the MCP server reads them too — so
# every surface that can answer an armed op asks the one question a yes will
# answer. Deterministic for the same reason (`_RESIGN_CONFIRM` is this rule on
# the spoken road), and phrased for a dialog rather than for a spoken yes.


def _confirm_question(op: str) -> str:
    """What the player is being asked, for a reader that needs to know.

    The model gets the app's own wording rather than whatever paraphrase the
    turn happened to speak, because it is reading a reply *against a question*
    and the question is the pipeline's fact, not the narration's.
    """
    return CONFIRM_QUESTIONS.get(op, f"Confirm {op}?")


def _confirm_required(op: str) -> JSONResponse:
    """409 + the gate's question, for a destructive UI action the gate armed.

    A 409 because nothing happened and the request as sent cannot be completed —
    the same status the endpoints already answer for an impossible undo. What
    makes this one different is `confirm: true`: it marks the body as a *question*
    rather than a failure, so a client can tell "answer me" from "no", and
    `detail` stays what every other 409 here puts there — the line to show the
    player. `op` names the armed op, so the client knows what it is confirming.
    """
    return JSONResponse(
        status_code=409,
        content={
            "detail": _confirm_question(op),
            "confirm": True,
            "op": op,
        },
    )


@dataclass(frozen=True)
class _MoveBeats:
    """One player move played through the coordinator's beats.

    What `_play_move` did, for whoever asked it: the `make_move` result as
    `changes` (the `{"name", "result"}` shape a narration and the trace both
    read), the reaction if one was produced, and the engine's reply with whether
    one was ever owed — `owed_reply` is False both when the game ended on the
    player's own move and when the coordinator had nothing open, and the
    commentary needs to tell those apart from "the engine passed".

    `observed_fen` is the board the narration was written from — the position
    after the player's move, before the reply. The honesty guard needs it:
    checking a reaction against the position that came *after* the one it
    reacted to is how ordinary trades came to be guarded as lies. `None` when
    no narration ran, because then there is nothing that saw a board.

    `engine_failure` names what killed the reply, on the one shape where
    `owed_reply` is True and `engine_reply` is None because Stockfish died
    rather than because it had nothing to say. Empty on every other turn — the
    record's way of saying "did not die" rather than "not recorded", the same
    as the trace's `provider_failure`.

    `reaction_late` marks the beat the budget cut (`_REACTION_BUDGET_S`): the
    words were still being written when the turn went on without them. False
    on a beat that spoke, on one a provider failure killed, and on one that
    never ran — the player hears the same deterministic line on three of those
    four, so this is the only place the difference survives.

    `cost` is the observe beat's round trip — the narration's own, or one call
    and the time waited when the provider died or the budget cut it (#290).
    Zero on a beat that never ran.
    """

    changes: list[dict[str, Any]]
    narration: Narration | None
    engine_reply: MoveResult | None
    owed_reply: bool
    observed_fen: str | None = None
    engine_failure: str = ""
    reaction_late: bool = False
    cost: "_ModelCost" = dc_field(default_factory=lambda: _ModelCost())

    @property
    def result(self) -> dict[str, Any]:
        """The `make_move` result itself."""
        return self.changes[0]["result"]

    @property
    def legal(self) -> bool:
        return self.result.get("legal") is True


def _turn_evidence(
    ctx: ToolContext,
    tool_results: Sequence[dict[str, Any]],
    engine_reply: MoveResult | None,
    fen_before: str,
    fens_observed: Sequence[str] = (),
    pending_reply_fen: str | None = None,
) -> TurnEvidence:
    """The record `facts.assemble` builds a turn's facts from, read off the
    live context: the session as the turn left it and the claimable settings,
    beside what the route knows about its own boards (see `facts.assemble`
    for what each board is for)."""
    return TurnEvidence(
        session=ctx.session.to_dict(),
        settings=settings_of(
            ctx.settings.voice_output, ctx.settings.verbosity, ctx.settings.tier
        ),
        tool_results=list(tool_results),
        engine_reply_san=engine_reply.san if engine_reply is not None else None,
        fen_before=fen_before,
        fens_observed=tuple(fens_observed),
        pending_reply_fen=pending_reply_fen,
    )


def _verified_facts(
    ctx: ToolContext,
    tool_results: Sequence[dict[str, Any]],
    engine_reply: MoveResult | None,
    fen_before: str,
    fens_observed: Sequence[str] = (),
    pending_reply_fen: str | None = None,
) -> VerifiedFacts:
    """What this turn may honestly say (audit item 13, the pipeline's half),
    by way of the same evidence a trace record carries — so a turn re-judged
    offline is judged on the facts the live turn had (#367)."""
    return assemble(
        _turn_evidence(
            ctx,
            tool_results,
            engine_reply,
            fen_before,
            fens_observed,
            pending_reply_fen,
        )
    )


def _settle_question(
    ctx: ToolContext,
    origin: str,
    question: clarification.Clarification | None,
    version_before: int,
    tool_results: Sequence[dict[str, Any]],
    request: str = "",
    asked: Sequence[str] = (),
) -> dict[str, Any]:
    """Close or open `origin`'s question as an interaction ends (#319), and
    say what happened for the trace.

    `question` is what `ToolContext.live_clarification` read on the way in —
    live then, on the board `version_before` names. If this interaction moved
    the board it is settled (`clarification.settle`: answered by a candidate,
    or superseded); if it only asked again, the new question replaces it. A
    turn that did neither leaves it open, which is the continuation policy.
    `asked` is the handoff's candidates on a `clarify` turn, and the new
    question is stamped with the board at the end of the turn — the one the
    player hears it over, the reason `restamp_pending` exists.

    Under the mutation lock like everything that touches the context.
    """
    closed: clarification.Closed | None = None
    created: clarification.Clarification | None = None
    if question is not None and ctx.clarifications.get(origin) is question:
        if ctx.board_version != version_before:
            closed = clarification.settle(question, tool_results)
            del ctx.clarifications[origin]
        elif asked:
            closed = clarification.Closed(
                question, clarification.SUPERSEDED, clarification.ASKED_AGAIN
            )
    if asked:
        created = clarification.ask(
            origin=origin,
            game_id=ctx.session.game_id,
            board_version=ctx.board_version,
            request=request,
            candidates=asked,
        )
        ctx.clarifications[origin] = created
    return {
        "created": created.trace() if created is not None else None,
        "closed": closed.trace() if closed is not None else None,
    }


def _question_trace(
    question: clarification.Clarification | None,
    expired: clarification.Closed | None,
) -> dict[str, Any]:
    """The turn record's `clarification` field as an interaction opens: which
    question stood (`open`), and which one this read found gone (`expired`).
    `_settle_question` fills in the rest as it ends."""
    return {
        "open": question.id if question is not None else None,
        "expired": expired.trace() if expired is not None else None,
        "created": None,
        "closed": None,
    }


def _remembered_facts(
    tool_results: Sequence[dict[str, Any]],
    engine_reply: MoveResult | None,
    session: GameSession,
) -> str:
    """What a turn the *app* spoke for is remembered as: the deterministic facts
    in the app's own register, or nothing at all.

    The transcript's other half of the honesty rule. The guard's canned
    corrections (retired 2026-09-10 for the rewrite) were written in the first
    person, and recording one as the assistant's turn handed the narrator its
    own apology as something it said — `condense` gives the last few turns
    back verbatim, so Glitch read it and imitated the register, on turns where
    nothing was guarded at all. Live, that is exactly what happened: one
    guarded trade was enough to have him volunteering "I almost said something
    that didn't happen. That's my bad." The rule outlives the lines: the stuck
    line and the move confirmation are the app's too.

    So a substituted turn remembers what the turn *did*, never what the app said
    about it. A move turn has a line for that already — the same one verbosity=low
    and a dead provider fall back to — and every route puts its `make_move`
    result in `tool_results`. A turn that moved nothing has no facts to remember
    and says so with an empty string; `conversation.condense` turns that into the
    inert ack rather than shipping an empty assistant message at a chat template.
    """
    for record in tool_results:
        result = record["result"]
        if record["name"] == "make_move" and result.get("legal") is True:
            return _move_confirmation(result, engine_reply, session)
    return ""


@dataclass(frozen=True)
class _AdviceLicence:
    """What the advice guard checks a reply against, when it applies at all.

    `unlicensed` is the legal list minus every move a tool reported this
    turn; `legal` is the whole list, which the clarification rule counts
    over; `evidence` is what the analysis tools said, which is what the
    rewrite brief names. `None` in place of one of these means the guard
    does not apply: the board changed (a reaction is description), or no
    analysis ran (an opinion is Glitch's to give).
    """

    unlicensed: frozenset[str]
    legal: frozenset[str]
    evidence: frozenset[str]


@dataclass(frozen=True)
class _Guarded:
    """What the honesty guard let the model say, and what it took to get there.

    `text` is the model's half of the turn as the player will hear it: the
    first draft when it was clean, the rewrite when the first was not and the
    second is, or `""` when neither was — the caller composes the app's own
    deterministic lines around whichever it is. `claims` names the classes the
    first draft asserted without backing (empty on a clean turn), `suppressed`
    keeps that draft, and `rewrite` / `rewrite_claims` / `rewrite_suppressed`
    say what became of the second try, in the trace's vocabulary
    (`trace.turn_record`). `cost` is the rewrite's round trip — the one that
    died too — to be added to the turn's.
    """

    text: str
    claims: tuple[str, ...] = ()
    suppressed: str = ""
    rewrite: str = ""
    rewrite_claims: tuple[str, ...] = ()
    rewrite_suppressed: str = ""
    cost: "_ModelCost" = dc_field(default_factory=lambda: _ModelCost())

    @property
    def fired(self) -> bool:
        return bool(self.claims)

    @property
    def fell_back(self) -> bool:
        """The player hears the deterministic facts and nothing of the model's:
        the first draft was cut and no rewrite replaced it."""
        return self.fired and self.rewrite != "spoken"


def _unbacked(
    commentary: str, facts: VerifiedFacts, advice: _AdviceLicence | None
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """The claim classes this commentary asserts without backing, and the
    correction lines a rewrite brief needs for them — the honesty classes
    first, then the advice guard's, which is a licence check rather than a
    fact check and so gets its own line."""
    found: tuple[Unverified, ...] = unverified(commentary, facts)
    claims = list(dict.fromkeys(item.claim for item in found))
    lines = list(corrections(found, facts))
    if advice is not None and unlicensed_advice(
        commentary, advice.unlicensed, advice.legal
    ):
        claims.append("move_advice")
        lines.append(
            "The engine's moves this turn were: "
            f"{', '.join(sorted(advice.evidence))}. Name only those moves, or no "
            "move at all — do not add a move of your own beside the engine's."
        )
    return tuple(claims), tuple(lines)


async def _honest_words(
    brain: Brain,
    offloop: Callable[..., Awaitable[Any]],
    commentary: str,
    facts: VerifiedFacts,
    advice: _AdviceLicence | None,
    transcript: Sequence[dict[str, str]],
    logged: dict[str, Any],
) -> _Guarded:
    """The commentary the turn's facts support, by way of a second draft.

    Every route converges here. The model may neither *do* an unasked
    destructive op (the gate) nor *say* it did (the ending class) nor announce
    any other fact the turn cannot back (the rest of them) nor, once it has
    asked the engine, hand over a move the engine did not name (the advice
    licence). What it may do is try again: a first draft with an unbacked
    claim goes back to the narrator with the true facts in plain words
    (`honesty.corrections`), and the rewrite is checked the same way. Code
    decides what is true; the model decides how to say it; a false positive
    costs one round trip and not the reply.

    A rewrite that still asserts something unbacked is cut, and the caller
    composes the turn from the app's deterministic lines alone — the move
    confirmation, the stuck line — which are true by construction and which
    the app already says when the model has nothing usable. There is no
    third try: a narrator that invents the same fact twice against an explicit
    correction is not going to be talked out of it, and each try is a round
    trip the player is waiting on.

    Both drafts and both verdicts go into the log *message* rather than
    `extra`, and into the trace: with twelve classes a false positive is the
    likelier failure, and the two live misfires before the trace kept the text
    were both diagnosed by guessing at phrasings. The default formatter drops
    `extra`, so a field nobody sees is a field that does not exist.
    """
    claims, lines = _unbacked(commentary, facts, advice)
    if not claims:
        return _Guarded(commentary)
    said = commentary.replace("\n", " ")
    logger.warning(
        "commentary_claimed_unverified_fact claims=%s suppressed=%r",
        ",".join(claims),
        said,
        extra={**logged, "claims": list(claims)},
    )
    started = time.monotonic()
    try:
        narration = await offloop(brain.rewrite, commentary, lines, transcript)
    except ProviderError as exc:
        # The turn is settled; the words are the only thing a dead provider can
        # cost here, and the fallback is what a turn with no words already says.
        # The round trip is still the turn's, and is counted as one (#290).
        logger.warning("rewrite_failed", exc_info=True, extra=logged)
        return _Guarded(
            "",
            claims,
            commentary,
            rewrite="lost",
            cost=_ModelCost.failed(started, PHASE_REWRITE, exc),
        )
    cost = _ModelCost.of(narration, PHASE_REWRITE)
    second = narration.text
    again, _ = _unbacked(second, facts, advice) if second else ((), ())
    if second and not again:
        return _Guarded(second, claims, commentary, rewrite="spoken", cost=cost)
    logger.warning(
        "rewrite_still_unverified claims=%s suppressed=%r",
        ",".join(again),
        second.replace("\n", " "),
        extra={**logged, "claims": list(again)},
    )
    return _Guarded(
        "",
        claims,
        commentary,
        rewrite="cut",
        rewrite_claims=again,
        rewrite_suppressed=second,
        cost=cost,
    )


def _ms_since(started: float) -> int:
    """Whole milliseconds since `started` (`time.monotonic()`), never negative —
    the resolution and type the trace reads every duration in."""
    return max(0, round((time.monotonic() - started) * 1000))


@dataclass
class _Spans:
    """Where one request's wall clock went outside the model (#290).

    Opened by `_mutation` the moment a request asks for the lock, so `queue`
    — the wait behind another turn — is measured by the one piece of code that
    waits, and `total` runs from the player's side of that wait. The other
    phases are added as the turn reaches them (`tool` by the registry's timing
    observer, `engine` around the reply's collect, `guard` around the honesty
    check) and summed, because a turn can pass through one more than once.

    Model time is not a span here: the trace already has it per call
    (`model_latencies_ms`), and a second copy summed another way is a second
    number that can disagree.
    """

    started: float
    ms: dict[str, int] = dc_field(default_factory=dict)

    def add(self, phase: str, elapsed_ms: int) -> None:
        self.ms[phase] = self.ms.get(phase, 0) + max(0, elapsed_ms)

    def as_trace(self) -> dict[str, int]:
        return {**self.ms, "total": _ms_since(self.started)}


@dataclass(frozen=True)
class _ModelCost:
    """What one turn spent at the provider boundary, whichever route spent it.

    A turn's model calls come in phases — reading an answer to a pending
    question, a narrated confirmation, a fast-path reaction, a resignation's
    words, the brain's whole loop, a rewrite — and a turn can pass through more
    than one: a reply the reader calls `unrelated` goes on down whichever road
    the words take. So a turn's cost is always *summed* with `plus`, never
    assigned, and every phase owes the trace the same numbers. Reading them off
    the `AgentResponse`/`Narration`/`Answer` in one place is what keeps a route
    from quietly recording three of the four.

    A call that raised is still a call (#290): the turn waited on it, so it is
    counted with its latency and no tokens (`failed`), and its tokens are
    unknown — the token totals are then a lower bound, not a measured zero.

    Held as the calls themselves, one `ModelCall` each, tagged with the phase
    that made it and how it ended (#317); every number the trace sums is summed
    off them in `turn_record`, so there is no second tally to disagree with.
    """

    calls: tuple[ModelCall, ...] = ()

    @property
    def latencies_ms(self) -> tuple[int, ...]:
        return tuple(call.ms for call in self.calls)

    @classmethod
    def of(cls, source: Any | None, phase: str = PHASE_UNKNOWN) -> "_ModelCost":
        """The cost an `AgentResponse`, `Narration` or `Answer` reports; `None`
        — a route that never called the model — costs nothing.

        A source that tagged its own calls (the brain's loop) is taken as it
        is. The single-call seams cannot know which beat they served, so the
        caller names it as `phase`. A source that reports only totals (a test
        double) is read call by call off its latencies, keeping both of its
        totals: the last `unmetered_calls` of them are the unmetered ones, and
        the token sums ride on the first metered call."""
        if source is None:
            return cls()
        tagged = tuple(getattr(source, "calls", ()))
        if tagged:
            return cls(calls=tagged)
        readings = tuple(source.model_latencies_ms)
        count = source.model_calls
        if len(readings) < count:
            readings += (0,) * (count - len(readings))
        status = getattr(source, "status", CALL_OK)
        failure = getattr(source, "failure", "")
        metered = count - source.unmetered_calls
        calls = tuple(
            ModelCall(
                phase,
                status,
                ms,
                (source.prompt_tokens if i == 0 else 0) if i < metered else None,
                (source.completion_tokens if i == 0 else 0) if i < metered else None,
                failure=failure,
                server=getattr(source, "server", None) if count == 1 else None,
            )
            for i, ms in enumerate(readings[:count])
        )
        return cls(calls=calls)

    @classmethod
    def failed(
        cls,
        started: float,
        phase: str,
        exc: BaseException | None = None,
        budget_s: float | None = None,
    ) -> "_ModelCost":
        """One round trip that raised out of the brain — a dead provider or a
        reaction the app stopped waiting for — timed from `started`
        (`time.monotonic()`) by the caller, the only one still around to.
        `exc` says which of the two: a `LateReaction` is `late`, censored at
        `budget_s`; anything else is `failed`, with its provider failure kind."""
        late = isinstance(exc, LateReaction)
        failure = getattr(exc, "failure", "") if exc is not None and not late else ""
        return cls(
            calls=(
                ModelCall(
                    phase,
                    CALL_LATE if late else CALL_FAILED,
                    _ms_since(started),
                    failure=str(failure),
                    budget_ms=None if budget_s is None else round(budget_s * 1000),
                ),
            )
        )

    def plus(self, other: "_ModelCost") -> "_ModelCost":
        """Two phases of one turn, added. The confirmation route can now spend
        two round trips — reading the player's answer, then narrating what the
        op did — and a turn that reported only the second would under-report
        every free-text confirmation."""
        return _ModelCost(calls=self.calls + other.calls)

    def as_trace(self) -> dict[str, Any]:
        """These calls under the name `turn_record` takes them by."""
        return {"calls": self.calls}


@dataclass(frozen=True)
class CommandOutcome:
    """One command-pipeline run, shared by `/api/command` and the delegate
    messages endpoint. `tool_results` is the `{"name", "result"}` list of
    everything the agent ran, which `/api/command` returns verbatim;
    `tool_args` holds each call's arguments in the same order — the delegate
    endpoint needs them to build its wire `tool_calls`, but `/api/command`
    never exposes them. `stop_reason` is the brain loop's, in the fleet's
    vocabulary: `completed` when the agent finished with an answer,
    `no_progress` when the loop ended a planner that had started repeating
    itself (an answer too — the narrator still ran),
    `max_iterations` or `correction_limit` when it ran out of budget first,
    `provider_error` when the provider died mid-turn (the results of whatever
    ran are still here, and the turn was still closed). The fast path is
    always `completed` — it never reaches the model.

    `engine_failure` names what killed the engine's reply, empty on every turn
    it did not (`_failure_name`). It is a field of its own rather than a stop
    reason because the *loop* stopped normally: the stop reason is the delegate
    wire's word for how the run ended, and a run that ended with an answer did,
    whatever Stockfish was doing. What the engine's death changes is the turn —
    the player's move stands, the reply is still owed, and the commentary says
    so in the app's own line.

    `memory` is the assistant text the turn is *remembered* by: what Glitch
    himself said, and never the app's words. They part company two ways. When
    the app spoke *in his place* — a guard cut whose rewrite failed too, a
    budget stop, a dead provider — the turn remembers the deterministic facts
    instead, because an app line fed back as his own words is a register he
    imitates (`_remembered_facts`). A guard rewrite that passed is his own
    second draft and is remembered as such. And when the app spoke *after*
    him — the reply announcement composed onto every move turn, or the line
    standing in for it when the engine died — only his reaction is remembered,
    because the appended "\\n\\ne5." fed back as his own words is a format he
    completes at the beat where the reply does not exist yet (#193). The player
    gets the whole composed line; the model gets its own words or the facts."""

    commentary: str
    tool_results: list[dict[str, Any]]
    tool_args: list[dict[str, Any]]
    state: dict[str, Any]
    changed: bool
    stop_reason: str
    memory: str = ""
    engine_failure: str = ""
    # This interaction's trace id (`trace.new_correlation_id`), handed back so
    # the browser can name the turn its own milestones belong to (#317).
    correlation_id: str = ""


class StateBroadcaster:
    """Fans the state document out to every connected board UI — and, on the
    same channel, the live progress of the turn producing it.

    Two kinds of message, one socket, told apart by `type`. That envelope was
    always there for this: the board document is authoritative and the progress
    events are ephemera about how it came to change, and a client that wants
    one wants the other. Send failures mean the client went away; the socket is
    dropped, never allowed to fail the mutation that triggered the broadcast.

    **`publish` is callable from any thread, and has to be.** The pipeline's
    blocking steps run in worker threads on purpose — a progress event is worth
    nothing if it arrives after the turn it describes, and a blocked event loop
    delivers nothing at all — so a report crosses back onto the loop through
    `call_soon_threadsafe` and lands on a queue. One pump drains it, which is
    what keeps `begin` in front of `end`: several `ensure_future`d sends could
    interleave at their first await, a single consumer cannot.

    **Board documents go down that same queue**, which is why `broadcast` is
    `publish` with a different envelope rather than a send of its own. A turn
    now publishes a document mid-flight — the player's move, before the engine
    has answered — so two state frames describe the same turn and their order
    is the board: delivered the other way round, the player's piece would
    snap back. Two paths onto one socket could not promise that; one queue
    can.

    Unstarted is a working state, not a broken one: `publish` drops the event.
    A process with no UI attached (a unit test, an MCP session) still runs its
    turns; it just has nobody to tell.
    """

    def __init__(self) -> None:
        self._clients: set[WebSocket] = set()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._queue: asyncio.Queue[dict[str, Any]] | None = None
        self._pump: asyncio.Task[None] | None = None

    def start(self) -> None:
        """Bind to the running loop and start the pump. Idempotent, and called
        from both ends — app startup, and a client connecting — because the
        second is the one that runs when a test drives the app without its
        lifespan."""
        if self._pump is not None:
            return
        self._loop = asyncio.get_running_loop()
        self._queue = asyncio.Queue()
        self._pump = asyncio.create_task(self._drain())

    async def stop(self) -> None:
        pump, self._pump = self._pump, None
        if pump is None:
            return
        pump.cancel()
        with suppress(asyncio.CancelledError):
            await pump

    async def connect(self, websocket: WebSocket) -> None:
        self.start()
        await websocket.accept()
        self._clients.add(websocket)

    def disconnect(self, websocket: WebSocket) -> None:
        self._clients.discard(websocket)

    def broadcast(self, state: dict[str, Any]) -> None:
        self.publish({"type": "state", "state": state})

    def publish(self, message: dict[str, Any]) -> None:
        """Queue a message from any thread. Never raises: the caller is in the
        middle of a turn, and losing a progress line is not losing a turn."""
        loop, queue = self._loop, self._queue
        if loop is None or queue is None:
            return
        try:
            loop.call_soon_threadsafe(queue.put_nowait, message)
        except RuntimeError:
            # The loop is closing under us (shutdown, a torn-down test client).
            pass

    async def _drain(self) -> None:
        assert self._queue is not None
        while True:
            await self._send(await self._queue.get())

    async def _send(self, message: dict[str, Any]) -> None:
        for client in list(self._clients):
            try:
                await client.send_json(message)
            except Exception:
                self.disconnect(client)


def create_app(
    ctx: ToolContext,
    brain: Brain | None = None,
    speech: SpeechClient | None = None,
    static_dir: Path | None = None,
    registry: ToolRegistry | None = None,
    tracer: Tracer | None = None,
    coordinator: TurnCoordinator | None = None,
    progress: ProgressReporter | None = None,
    reaction_budget: float = _REACTION_BUDGET_S,
    serving_identity: Callable[[], dict[str, str]] | None = None,
) -> FastAPI:
    """Pass the same `registry` the brain dispatches through (app assembly
    does), so what the agent is offered is exactly what the app runs; omit it
    and the app builds its own over the same `ctx`.

    `coordinator` is the turn state machine the board endpoints drive. App
    assembly passes the *same* one it gave `build_registry`, so a move dragged on
    the board and a move typed at the agent advance one turn machine rather than
    two that can disagree; omit both and the app builds a matched pair itself.

    `tracer` records one JSONL row per command (route taken, tool trajectory,
    stop reason) for review; omit it and nothing is traced.

    `progress` is the live-progress reporter (audit item 19). Pass the one app
    assembly built — it is the only place that can reach the *brain*, whose two
    phases nothing else can see — and this binds it to the websocket and points
    the coordinator and the registry at it. Omit it and the app builds one, so a
    turn's phases and tool calls are reported whoever assembled the app; only
    the brain's own two phases go unheard.

    `reaction_budget` is how many seconds Glitch's optional words get before the
    turn goes on without them (`_REACTION_BUDGET_S`, and see `_narrate`). A
    parameter so a test can hand it a fraction of a second instead of sleeping
    through the real one.

    `serving_identity` answers what serves a turn — prompt and tool-schema
    hashes, model and server (`LlamaBrain.serving_identity`) — and every trace
    record carries it (#290). App assembly wires it to the brain it built;
    omit it and records say `None`."""
    app = FastAPI(title="chessapp", lifespan=lambda _app: _lifespan())
    broadcaster = StateBroadcaster()
    if coordinator is None:
        coordinator = TurnCoordinator(ctx)
    if registry is None:
        registry = build_registry(ctx, coordinator)
    if progress is None:
        progress = ProgressReporter()

    # The two chokepoints are pointed at the reporter *here* rather than at
    # construction, so nothing that reports has to be built after the thing it
    # reports to — and so the wiring is one place to read.
    def _publish_progress(event: ProgressEvent) -> None:
        broadcaster.publish({"type": "progress", "progress": event.as_dict()})

    # Public reads serve this immutable-by-convention document rather than
    # waiting behind the turn-wide mutation lock. A command holds that lock
    # while the brain thinks, so acquiring it in GET /state made the liveness
    # probe (and therefore the progress stream) wait for the whole turn.
    #
    # Writers replace the reference under the mutation lock; they never mutate
    # a document already published. Readers capture one reference and deepcopy
    # it, so a response is one board even if a newer document is installed at
    # the same instant. Assembly takes the initial coherent snapshot once.
    published_state = _state_dict(ctx)
    last_broadcast_version = published_state["version"]

    def _published_state() -> dict[str, Any]:
        """Return the latest coherent state without waiting for a long turn.

        App mutations publish at their chokepoints below. The non-blocking
        refresh closes the remaining seam: a caller may mutate the shared
        context through another correctly locked registry (or directly in a
        test) that is not wired to this app's observer. Once that mutation has
        released the lock, the next read notices the version mismatch and
        catches up. While it is still in flight, the last complete document is
        preferable to a torn one or a read that stalls live progress.
        """
        nonlocal published_state
        state = published_state
        if ctx.board_version != state["version"] and ctx.mutation_lock.acquire(False):
            try:
                if ctx.board_version != published_state["version"]:
                    state = _state_dict_unlocked(ctx)
                    published_state = state
                else:
                    state = published_state
            finally:
                ctx.mutation_lock.release()
        return deepcopy(state)

    # What the live checkpoint last recorded, so an unchanged game is not
    # rewritten on every request that happened to take the lock.
    last_checkpoint: tuple[int, int, int] | None = None

    def _checkpoint() -> None:
        """Write the live game to disk if it changed (#291, `live.json`).

        Called with the mutation lock held — from the mutation guard's exit and
        from the broadcast — so the board, its version and the transcript are
        one coherent snapshot. What counts as a change is the board version and
        the transcript's identity and length: every move, takeback, reset and
        resume bumps the first, and every exchange the panel records grows the
        last. Best-effort (`tools.write_live_checkpoint`): a full disk costs a
        warning, never a move.
        """
        nonlocal last_checkpoint
        if ctx.save_dir is None:
            return
        signature = (
            ctx.board_version,
            id(ctx.transcript),
            len(ctx.transcript.to_dict()),
        )
        if signature == last_checkpoint:
            return
        write_live_checkpoint(ctx, live_checkpoint(ctx))
        last_checkpoint = signature

    def _publish_state() -> None:
        """Send the board document, once per board.

        The one emitter, because there are now two reasons to send: a mutation
        as it happens (below), and an endpoint closing its turn. `board_version`
        is what tells them apart — one frame per board, so a registry-dispatched
        op followed by its endpoint's own send is one document, not the same one
        twice.

        Called from worker threads (dispatch runs off the loop) as well as from
        endpoints. Both hold the app's mutation lock, which is what makes
        reading the version and snapshotting the board one coherent step; the
        queue behind `broadcast` is what makes the crossing back safe.
        """
        nonlocal last_broadcast_version, published_state
        if ctx.board_version == last_broadcast_version:
            return
        state = _state_dict_unlocked(ctx)
        published_state = state
        last_broadcast_version = state["version"]
        _checkpoint()
        broadcaster.broadcast(state)

    # The boards the open command's mutating tool calls have left behind, in
    # order — or `None` between commands, which is every other road onto the
    # board (a drag, a button, the confirm endpoint) recording nothing.
    # `_command_window` owns it at both ends; the honesty guard reads it (see
    # `_verified_facts`).
    command_boards: list[str] | None = None
    # The request holding the mutation lock, timed (#290): opened by
    # `_mutation` before it waits, closed when it lets go, so there is at most
    # one and whatever runs under the lock — a tool, a collect, the guard —
    # adds to the right request's spans. `None` outside the lock.
    current_spans: _Spans | None = None

    def _add_span(phase: str, elapsed_ms: int) -> None:
        """Charge time to the request under the lock; nothing outside one."""
        if current_spans is not None:
            current_spans.add(phase, elapsed_ms)

    @contextmanager
    def _span(phase: str) -> Iterator[None]:
        started = time.monotonic()
        try:
            yield
        finally:
            _add_span(phase, _ms_since(started))

    def _record_mutation() -> None:
        """Remember the board the call left, then publish it.

        The mutation chokepoint's one callback. The trail is here rather than
        at the call sites for the reason the broadcast is: `dispatch` is the
        one road every model-initiated mutation takes, so a position recorded
        here is a position the game really reached — and the honesty guard's
        whole problem is that a turn holds more boards than its two ends
        (audit finding 7). Appended before the send, so a broadcast that fails
        cannot cost the guard its evidence.
        """
        if command_boards is not None:
            command_boards.append(ctx.session.fen())
        _publish_state()

    progress.bind(_publish_progress)
    coordinator.on_phase = progress.phase
    registry.on_tool = progress.tool
    # The mutation chokepoint, pointed at the same emitter: the player's move
    # reaches the board when it is validated rather than when the turn ends —
    # while Glitch reacts and Stockfish thinks (`docs/turn-coordinator.md`).
    registry.on_mutation = _record_mutation
    # Every tool handler's time, charged to the request running it (#290).
    registry.on_tool_done = lambda _name, elapsed_ms: _add_span("tool", elapsed_ms)
    # On disk beside the live game when there is a save dir (#291), so a
    # restart keeps the delegate threads — and their idempotency keys — as it
    # keeps the board.
    store = ConversationStore(
        ctx.save_dir / CONVERSATIONS_FILENAME if ctx.save_dir is not None else None
    )

    @asynccontextmanager
    async def _lifespan() -> AsyncIterator[None]:
        broadcaster.start()
        try:
            yield
        finally:
            await broadcaster.stop()

    async def _offloop[T](fn: Callable[..., T], *args: Any) -> T:
        """Run one blocking step of a turn in a worker thread.

        Not an optimization — a requirement of saying anything *live*. A turn
        spends seconds inside the model and Stockfish, and while the event loop
        sits inside one of those calls it cannot deliver a websocket frame, so
        every progress event would arrive in a burst after the turn it was
        describing had finished. Off the loop, the pump is free to send as the
        turn runs.

        Cancellation is deliberately not abandoned: a step that mutates the
        board must finish rather than be left running behind a disconnected
        client. The mutation lock is held around all of this either way, so the
        extra thread hop changes no ordering.
        """
        return await anyio.to_thread.run_sync(fn, *args)

    def _narrate(
        board_state: dict[str, Any],
        changes: list[dict[str, Any]],
        transcript: Sequence[dict[str, str]],
        correlation_id: str,
    ) -> Narration:
        """Glitch's words for one beat, or `LateReaction` if they are late.

        Every narration in the app goes through here, so the budget is a rule
        rather than a special case at the one site that exposed it: the observe
        beat holds a computed engine reply while it waits, and the confirmed-op
        and resign beats hold the mutation lock every other road onto the board
        needs. All three already lose their words to a dead provider and say
        the app's own line instead, which is exactly what a late one does.

        The words the late call eventually writes are dropped, never spoken a
        beat behind the board they were about: by then the reply has landed and
        the turn has closed, so they would describe a position that no longer
        exists — and voice-first, stale audio arrives over whatever is true now.
        """
        assert brain is not None  # every narration site is agent-mode only
        try:
            return _within_budget(
                lambda: brain.narrate(board_state, changes, transcript),
                reaction_budget,
            )
        except LateReaction:
            logger.warning(
                "narration_late budget=%.1fs",
                reaction_budget,
                extra={"correlation_id": correlation_id},
            )
            raise

    @app.exception_handler(StaleVersionError)
    async def _stale_version(_request: Request, exc: StaleVersionError) -> JSONResponse:
        """409 for a request about a superseded board (audit item 7).

        A 409 for the same reason every other one here is: nothing happened, and
        the request as sent cannot be completed. The body follows the gate's
        convention — `detail` is the line to show, and a flag (`stale`, next to
        the gate's `confirm`) says which kind of "no" this is — and it carries
        the current `version` *and* the current state, because a client that
        just found out it is behind needs both to catch up and retry without a
        second round trip.
        """
        # Captured by the mutation guard while it still held the boundary. The
        # handler neither waits behind a later turn nor races that turn into a
        # recovery document different from the version the rejection names.
        state = deepcopy(exc.state)
        current = state["version"]
        if isinstance(exc, WrongGameError):
            logger.info(
                "wrong_game expected=%s current=%s",
                exc.expected_game,
                state["game_id"],
            )
            detail = (
                f"that game is no longer on the board — you sent game "
                f"{exc.expected_game}, the board is game {state['game_id']}"
            )
        else:
            logger.info("stale_version expected=%s current=%s", exc.expected, current)
            detail = (
                "the board changed since you last saw it — "
                f"you sent version {exc.expected}, it is now {current}"
            )
        return JSONResponse(
            status_code=409,
            content={
                "detail": detail,
                "stale": True,
                "version": current,
                "game_id": state["game_id"],
                "state": state,
            },
        )

    @asynccontextmanager
    async def _mutation(
        expected: int | None, game_id: str | None = None
    ) -> AsyncIterator[None]:
        """Hold the mutation lock across one request's check *and* its mutation.

        The two halves are inseparable or the precondition is theatre: between a
        version read and the move it authorizes, another client's whole turn can
        land — FastAPI runs sync endpoints in a threadpool and async ones on the
        loop, so requests genuinely interleave. Everything a mutating endpoint
        does runs inside this, the check first, so a stale request is refused
        with the board untouched — including untouched by `abandon_turn`, which
        would otherwise throw away an open turn on behalf of a request that was
        never going to be allowed.

        The lock is acquired in a worker thread rather than by blocking the
        event loop. That is not a nicety: a waiter that blocks the loop would
        stop the *holder* from ever finishing its own awaits, and the two would
        deadlock. Acquiring off-loop means a waiting request costs a parked
        thread and nothing else.
        """
        nonlocal current_spans
        requested = time.monotonic()
        await anyio.to_thread.run_sync(ctx.mutation_lock.acquire)
        # Set only once the lock is held, so the one request that may write
        # `current_spans` is the one holding it.
        current_spans = _Spans(started=requested)
        current_spans.add("queue", _ms_since(requested))
        try:
            if game_id is not None and game_id != ctx.session.game_id:
                raise WrongGameError(game_id, _state_dict_unlocked(ctx))
            if expected is not None and expected != ctx.board_version:
                current = ctx.board_version
                raise StaleVersionError(expected, current, _state_dict_unlocked(ctx))
            yield
        finally:
            current_spans = None
            # Every road onto the board ends here, still holding the lock: the
            # one place a checkpoint is both complete (the transcript a command
            # records lands inside the guard too) and coherent.
            _checkpoint()
            ctx.mutation_lock.release()

    @app.get("/api/state")
    def get_state() -> dict[str, Any]:
        return _published_state()

    @app.websocket("/ws")
    async def state_channel(websocket: WebSocket) -> None:
        await broadcaster.connect(websocket)
        await websocket.send_json({"type": "state", "state": _published_state()})
        try:
            # The channel is one-way; we only read to notice the disconnect.
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            broadcaster.disconnect(websocket)

    def _play_move(
        move: str, transcript: Sequence[dict[str, str]], correlation_id: str
    ) -> _MoveBeats:
        """One move through the coordinator's beats: apply, observe, close.

        The move-turn orchestration, in one place, because two callers own those
        beats — the command pipeline's fast path and a board drag in agent mode —
        and "one road in" is worth nothing if the two roads sequence a turn
        differently. `move` is always a structured move string (SAN or UCI), never
        natural language: the fast path has already parsed the utterance against
        this board, and a drag never had words in the first place.

        The order is the whole point. `make_move` applies the player's move and
        stops, the engine starts thinking the moment it lands, and the reaction
        runs *while* it does — so the observation costs no wall clock. Then the
        reply is collected and the turn closed. The reaction is optional by
        construction: verbosity=low skips it, a `ProviderError` costs the words
        and nothing else, and a narrator that is merely slow costs the same
        (`_narrate`'s budget) — because the move it was about is already on the
        board and the engine's answer is not the model's to hold up.

        The reply is not optional, but it can be *lost*: an engine that dies on
        the collect leaves the turn open with the move standing, and that comes
        back as `engine_failure` rather than as an exception — the player's
        move is committed either way, and a committed move may not reach the
        caller as a failed request.

        `correlation_id` is the caller's id for the interaction, carried only so
        the beat's own warning lands under it: a lost reaction is a thing you
        find in the log and then want the turn record for.
        """
        assert brain is not None  # both callers are agent-mode only
        result = registry.dispatch("make_move", {"move": move})
        changes = [{"name": "make_move", "result": result}]
        narration: Narration | None = None
        observed_fen: str | None = None
        reaction_late = False
        cost = _ModelCost()
        if result.get("legal") is True and ctx.settings.verbosity != "low":
            # This is the observe beat, so the machine is told so — the phase
            # the coordinator has always had a slot for, finally entered
            # (`docs/turn-coordinator.md`). Conditional because the move may
            # have ended the game, which closes the turn where it stands; the
            # collect below accepts either phase, so nothing else changes.
            coordinator.mark_observation()
            fen_at_observation = ctx.session.fen()
            started = time.monotonic()
            try:
                narration = _narrate(
                    _narrator_state_dict(ctx), changes, transcript, correlation_id
                )
                cost = _ModelCost.of(narration, PHASE_REACTION)
                # Kept only once something was actually said from it: this is
                # "the board the reaction was written from", and a beat the
                # provider killed wrote no reaction. Read off the session,
                # because the narrator's own view deliberately carries no FEN.
                observed_fen = fen_at_observation
            except LateReaction as exc:
                # The budget expired with the reply already computed and
                # waiting. Falling through is the whole fix: the collect below
                # puts Stockfish's answer on the board, the turn closes on the
                # app's own announcement, and the lock goes back to whoever is
                # queued behind this one. `_narrate` has logged it; the beat
                # records it so the trace can tell a late reaction from a lost
                # one or a skipped one.
                reaction_late = True
                cost = _ModelCost.failed(
                    started, PHASE_REACTION, exc, budget_s=reaction_budget
                )
            except ProviderError as exc:
                cost = _ModelCost.failed(
                    started, PHASE_REACTION, exc, budget_s=reaction_budget
                )
                logger.warning(
                    "observe_narration_failed",
                    exc_info=True,
                    extra={"correlation_id": correlation_id},
                )
        # The close beat. A turn still mid-sequence is one whose player move
        # landed without its reply — including one *this* call did not open, left
        # owing by a route that raised, which is settled here rather than left to
        # wedge the machine.
        owed_reply = coordinator.phase in (
            TurnPhase.PLAYER_MOVE_APPLIED,
            TurnPhase.AGENT_OBSERVING,
        )
        engine_reply: MoveResult | None = None
        engine_failure = ""
        if owed_reply:
            try:
                with _span("engine"):
                    engine_reply = coordinator.collect_engine_reply()
            except Exception as exc:
                # Stockfish died with the player's move already committed and
                # broadcast, so the failure is not this request's to fail on
                # (#284): it is a turn holding one move instead of two, and it
                # goes back as that. The turn is deliberately *not* completed —
                # the coordinator has put the phase back to
                # `player_move_applied`, where the reply is still owed and the
                # next command settles it, and completing it here would be the
                # one thing the coordinator exists to refuse: skipping the
                # engine's move. `owed_reply` stays True for the same reason.
                engine_failure = _failure_name(exc)
                logger.warning(
                    "engine_reply_failed",
                    exc_info=True,
                    extra={"correlation_id": correlation_id},
                )
            else:
                coordinator.complete_turn()
        elif (played := result.get("engine_move")) is not None:
            # An atomic registry played the reply inside the tool (not how the
            # app is assembled — see `build_registry`'s `atomic_exchange`), so
            # there is nothing left to collect. Take its word for the reply
            # rather than report a silence the board would contradict.
            owed_reply = True
            engine_reply = MoveResult(legal=True, san=played["san"], uci=played["uci"])
        return _MoveBeats(
            changes=changes,
            narration=narration,
            engine_reply=engine_reply,
            owed_reply=owed_reply,
            observed_fen=observed_fen,
            engine_failure=engine_failure,
            reaction_late=reaction_late,
            cost=cost,
        )

    async def _agent_move(move: str) -> dict[str, Any]:
        """A dragged move in agent mode: the same beats, the same one turn.

        The audit's item 4. The board sends the structured move it always sent and
        gets back the response it always got — plus the reaction Glitch had to it,
        and whether to speak it. The turn is recorded on the panel transcript
        under the move's SAN, so Glitch's later turns remember games the player
        dragged as well as games they talked their way through.
        """
        before = _agent_state_dict(ctx)
        # A turn like any other, so it is located like one — see `_command_turn`.
        turn_id = coordinator.turn_id
        correlation_id = new_correlation_id()
        version_before = ctx.board_version
        # A drag is one interaction like a command, so its phases and tool
        # calls are bracketed the same way — but *without* a command window:
        # a drag dispatches once by construction and is deliberately
        # unbudgeted (see `TurnCoordinator.begin_command`).
        # The panel's open question (#319): a drag is the player's own board,
        # so a dragged candidate answers it like a spoken one.
        question, expired = ctx.live_clarification(PANEL_ORIGIN)
        asked_about = _question_trace(question, expired)
        with progress.interaction(correlation_id, turn_id):
            transcript = ctx.transcript.memory()
            beats = await _offloop(_play_move, move, transcript, correlation_id)
            result = beats.result
            narration = beats.narration
            if result.get("ok") is False:
                # A turn-state rejection: 409 on the trusted path, exactly as direct
                # mode answers it (the agent reads the same refusal as result data).
                # The drag played nothing, but the beats may have settled a turn that
                # was left open — that reply is on the board now, so every client
                # hears about it before the refusal goes back.
                if ctx.session.fen() != before["fen"]:
                    _publish_state()
                raise HTTPException(status_code=409, detail=result["error"])
            commentary = ""
            verdict = _Guarded("")
            if beats.legal:
                # The honesty guard, on this road too: a reaction that announces
                # something the drag did not actually do goes back to the
                # narrator with the truth, and a rewrite that still does is
                # cut. On the reaction alone — the announcement composed
                # around it below is the app's own deterministic line, so there
                # is nothing in it to guard and everything to lose by taking it
                # back with the reaction. No advice licence: the board moved,
                # so a move named here is a reaction to it.
                guard_started = time.monotonic()
                verdict = await _honest_words(
                    brain,
                    _offloop,
                    narration.text if narration is not None else "",
                    _verified_facts(
                        ctx,
                        beats.changes,
                        beats.engine_reply,
                        before["fen"],
                        # A drag opens no command window, so there is no trail
                        # to read: one dispatch, and the beats already know
                        # which board they narrated from.
                        [beats.observed_fen] if beats.observed_fen is not None else [],
                        # The reaction was spoken before the reply existed.
                        beats.observed_fen if beats.owed_reply else None,
                    ),
                    None,
                    transcript,
                    {"move": move, "correlation_id": correlation_id},
                )
                # The guard's own time; its rewrite is model time (#290).
                _add_span(
                    "guard", _ms_since(guard_started) - sum(verdict.cost.latencies_ms)
                )
                commentary = _move_commentary(
                    verdict.text,
                    result,
                    beats.engine_reply,
                    beats.owed_reply,
                    ctx.session,
                )
                if beats.engine_failure:
                    # The drag landed and the reply did not. The app says so
                    # where the reply announcement would have been — one more
                    # deterministic line composed around Glitch's reaction,
                    # never a thing he is asked to say (#284).
                    commentary = _engine_lost_words(commentary)
                # What the turn is remembered by, the same rule as the command
                # pipeline's: the reaction when Glitch spoke one (never the
                # composed commentary — the appended reply line is the app's,
                # and remembered as his it becomes a format he completes a beat
                # early, #193), and the deterministic facts when he didn't (a
                # reaction cut by the guard, a silent low-verbosity turn).
                remembered = "" if verdict.fell_back else verdict.text
                ctx.transcript.record(
                    result["san"],
                    remembered
                    or _remembered_facts(
                        beats.changes, beats.engine_reply, ctx.session
                    ),
                )
                asked_about.update(
                    _settle_question(
                        ctx, PANEL_ORIGIN, question, version_before, beats.changes
                    )
                )
                _publish_state()
            _trace_turn(
                utterance=move,
                route=ROUTE_BOARD,
                origin=PANEL_ORIGIN,  # a drag on the player's own board
                commentary=commentary,
                stop_reason="completed",
                changed=beats.legal,
                turn_id=turn_id,
                correlation_id=correlation_id,
                mutations=ctx.board_version - version_before,
                fen_before=before["fen"],
                fen_after=ctx.session.fen(),
                tool_calls=[{"move": move}],
                tool_results=beats.changes,
                engine_reply=_move_reply_dict(beats.engine_reply),
                guarded=verdict.fired,
                guarded_claims=verdict.claims,
                suppressed=verdict.suppressed,
                rewrite=verdict.rewrite,
                rewrite_claims=verdict.rewrite_claims,
                rewrite_suppressed=verdict.rewrite_suppressed,
                engine_failure=beats.engine_failure,
                reaction_late=beats.reaction_late,
                clarification=asked_about,
                **beats.cost.plus(verdict.cost).as_trace(),
            )
            return {
                "legal": beats.legal,
                "san": result.get("san"),
                "uci": result.get("uci"),
                "reason": result.get("reason"),
                "engine_move": (
                    _move_dict(beats.engine_reply)
                    if beats.engine_reply is not None
                    else None
                ),
                "state": _state_dict_unlocked(ctx),
                "commentary": commentary,
                # Whether the client should voice it — the user's voice_output
                # setting, the same contract `/api/command` has.
                "speak": ctx.settings.voice_output,
            }

    def _settle_owed_reply() -> bool:
        """Play the reply a previous turn was left owing, if there is one.

        Direct mode's half of the healing the agent path gets for free: there,
        the refused move is a `make_move` *result* and `_play_move`'s close beat
        runs anyway, so the owed reply is collected on the way past. Here the
        refusal is an exception out of the atomic exchange, so the settling is
        spelled out — same rule, same one reply, so the two surfaces recover a
        dead-engine turn identically.

        An engine that is *still* dead settles nothing and says so: the turn
        stays open with the reply owed, which is exactly where it already was.
        Returns whether the board moved.
        """
        if coordinator.phase not in (
            TurnPhase.PLAYER_MOVE_APPLIED,
            TurnPhase.AGENT_OBSERVING,
        ):
            return False
        try:
            coordinator.collect_engine_reply()
        except Exception:
            logger.warning("engine_reply_failed", exc_info=True)
            return False
        coordinator.complete_turn()
        return True

    @app.post("/api/game/move")
    async def submit_move(request: MoveRequest) -> dict[str, Any]:
        """A move from the board: through the agent's beats when there is an
        agent, straight down the deterministic exchange when there isn't.

        The mode split is the whole of audit items 1/4. Direct mode is not a
        fallback here, it is the LLM-off invariant: no brain means no reaction to
        wait for, so the coordinator runs the exchange atomically and the response
        carries not one new key. Agent mode adds the beats — and only the beats;
        legality, the engine's reply, and the response's existing fields are the
        same machine's answers either way.

        The one key direct mode does answer with is `commentary`, and only on
        the turn Stockfish dies on (#284): the move is committed, so the
        request did not fail, and a board that moved once with no word about
        the answer that never came is a board the player cannot follow. It is
        the app's own line, the same one the agent path composes onto Glitch's
        reaction — no model is consulted in either mode.
        """
        async with _mutation(request.version, request.game_id):
            if brain is not None:
                return await _agent_move(request.move)
            # The coordinator runs the exchange: player move, then the engine's
            # reply if one is owed. Trusted path, so a turn-state rejection is a
            # 409 rather than the error *result* the agent gets for the same thing.
            fen_before = ctx.session.fen()
            try:
                result, reply = await _offloop(coordinator.play_exchange, request.move)
            except TurnStateError as exc:
                # Mid-turn: a previous exchange's engine died on the collect and
                # the reply is still owed. Settle that one before refusing this
                # move, so the game goes on at the cost of one drag rather than
                # needing an undo or a reset to dig it out.
                if await _offloop(_settle_owed_reply):
                    _publish_state()
                raise HTTPException(status_code=409, detail=str(exc)) from exc
            except Exception:
                # The engine died with the player's move already applied. The
                # atomic call's own `MoveResult` went with the raise, so what is
                # reported is read off the board it left behind — the one thing
                # that certainly survived — and only when the board says a move
                # landed at all. Nothing committed means nothing to protect, and
                # the failure is the request's after all.
                if ctx.session.fen() == fen_before:
                    raise
                logger.warning("engine_reply_failed", exc_info=True)
                history = ctx.session.move_history()
                _publish_state()
                return {
                    "legal": True,
                    "san": history[-1] if history else None,
                    "uci": None,
                    "reason": None,
                    "engine_move": None,
                    "state": _state_dict_unlocked(ctx),
                    "commentary": ENGINE_LOST_REPLY_OWED,
                }
            engine_move = _move_dict(reply) if reply is not None else None
            if result.legal:
                _publish_state()
            return {
                "legal": result.legal,
                "san": result.san,
                "uci": result.uci,
                "reason": result.reason,
                "engine_move": engine_move,
                "state": _state_dict_unlocked(ctx),
            }

    def _trace_control(
        op: str,
        args: dict[str, Any],
        result: dict[str, Any] | None,
        *,
        turn_id: int,
        version_before: int,
        fen_before: str,
    ) -> None:
        """Record one board-control interaction — the buttons' half of a turn.

        This surface has no utterance and no commentary (the dialog said what was
        about to happen, and no model stands between a yes and a reset), so the op
        name stands in for the words exactly as the structured move does on the
        board route. It is recorded anyway because it *can change the board*, and
        the spoken answer to the same question has always left a record: a reset
        confirmed by voice was diagnosable and the identical reset confirmed by a
        button was not. `result` is None when nothing was dispatched — a decline —
        which the record shows as no tools and no mutations rather than a
        fabricated result.
        """
        _trace_turn(
            utterance=op,
            route=ROUTE_CONTROL,
            # The buttons are the player's own screen, which is one origin with
            # the panel's free text (`tools.PANEL_ORIGIN`).
            origin=PANEL_ORIGIN,
            commentary="",
            stop_reason="completed",
            changed=ctx.board_version != version_before,
            turn_id=turn_id,
            correlation_id=new_correlation_id(),
            mutations=ctx.board_version - version_before,
            fen_before=fen_before,
            fen_after=ctx.session.fen(),
            tool_calls=[args] if result is not None else [],
            tool_results=[{"name": op, "result": result}] if result is not None else [],
        )

    def _run_destructive(
        name: str, args: dict[str, Any]
    ) -> dict[str, Any] | JSONResponse:
        """Dispatch a destructive UI action through the registry — gate included.

        The button and the spoken command now ask the *same* question, because
        they run the same code: `_gate` decides whether a game is in progress and
        arms `ctx.pending` if it is, and this returns the 409 that relays its
        question. On a fresh or finished board the gate stands aside and the op
        simply runs, which is why these endpoints keep their old behavior in
        exactly the cases where confirming would have been a question about
        nothing.

        Returns the tool result when the op ran, or the confirm-required response
        when the gate armed it instead. An op that *ran* clears anything else that
        was armed: a pending op is about a game that no longer exists.

        The dispatch is the whole interaction, so it is traced right here — one
        record whichever of the three ways it ends (ran, armed and asked, refused
        outright), rather than one per branch that remembered to.
        """
        turn_id = coordinator.turn_id
        version_before = ctx.board_version
        fen_before = ctx.session.fen()
        # Whose question this would be, declared before the gate can arm one
        # (#281): a button press is the player at their own screen, the same
        # origin the panel's free text answers from.
        ctx.origin = PANEL_ORIGIN
        result = registry.dispatch(name, args)
        _trace_control(
            name,
            args,
            result,
            turn_id=turn_id,
            version_before=version_before,
            fen_before=fen_before,
        )
        armed = ctx.pending
        if result.get("ok") is False and armed is not None and armed.name == name:
            return _confirm_required(name)
        if result.get("ok") is not True:
            raise HTTPException(
                status_code=409, detail=result.get("error", f"cannot {name}")
            )
        ctx.pending = None
        return result

    @app.post("/api/game/new")
    async def new_game(request: NewGameRequest | None = None) -> Any:
        """Start a new game — through the gate, so a game in progress is never
        thrown away without an answer (409 + the question; `/api/game/confirm`
        answers it).

        `random` is resolved here, before the op is armed, so the game the player
        confirms is the game they were asked about rather than a fresh roll. The
        turn the old board had open, and the engine's opening move when the player
        takes black, are the `new_game` tool's business now — which is the
        coordinator's, in one place, exactly as before.
        """
        color = request.color if request is not None else "random"
        if color == "random":
            color = random.choice(["white", "black"])
        async with _mutation(
            request.version if request is not None else None,
            request.game_id if request is not None else None,
        ):
            outcome = _run_destructive("new_game", {"player_color": color})
            if isinstance(outcome, JSONResponse):
                return outcome
            _publish_state()
            return {"state": _state_dict_unlocked(ctx)}

    @app.post("/api/game/confirm")
    async def confirm_destructive(request: ConfirmRequest) -> dict[str, Any]:
        """Answer the armed destructive op: yes runs it, no drops it.

        The button half of the confirmation gate, and deliberately the *same*
        `ctx.pending` the spoken road uses — an op armed by a button and confirmed
        by a typed "yes" works, and so does the reverse. Like a spoken answer it
        settles the op for good: whichever way it is answered, nothing stays
        armed, so a stale click can never revive a reset the player declined.

        No commentary: by this point there is nothing left to decide, so no model
        call stands between the yes and the reset (the same reason
        `confirm_pending` exists).

        A click that arrives after the board moved has nothing to confirm (409,
        like a click with nothing armed at all): the question was about a
        position, and that position is gone. A click while a *delegate* thread's
        question is standing is the same 409 (#281) — that question was put to
        someone else, and the button neither answers it nor disarms it, because
        a click with nothing of its own to confirm is not a new command.
        """
        async with _mutation(request.version, request.game_id):
            armed = ctx.live_pending(PANEL_ORIGIN)
            if armed is None:
                raise HTTPException(status_code=409, detail="nothing to confirm")
            turn_id = coordinator.turn_id
            version_before = ctx.board_version
            fen_before = ctx.session.fen()
            if not request.confirm:
                ctx.pending = None
                _trace_control(
                    armed.name,
                    dict(armed.args),
                    None,  # declined: nothing was dispatched
                    turn_id=turn_id,
                    version_before=version_before,
                    fen_before=fen_before,
                )
                return {
                    "op": armed.name,
                    "confirmed": False,
                    "state": _state_dict_unlocked(ctx),
                }
            confirmed = confirm_pending(registry, ctx, PANEL_ORIGIN)
            assert confirmed is not None  # armed above, and only this consumes it
            name, result = confirmed
            _trace_control(
                name,
                dict(armed.args),
                result,
                turn_id=turn_id,
                version_before=version_before,
                fen_before=fen_before,
            )
            if result.get("ok") is not True:
                raise HTTPException(
                    status_code=409, detail=result.get("error", f"cannot {name}")
                )
            _publish_state()
            return {
                "op": name,
                "confirmed": True,
                "state": _state_dict_unlocked(ctx),
            }

    @app.post("/api/game/undo")
    async def undo(request: UndoRequest) -> dict[str, Any]:
        async with _mutation(request.version, request.game_id):
            plies = request.plies
            if plies is None:
                # The player's takeback: vs the engine it is always the player's
                # turn after an exchange, so pop their move and the engine's
                # reply; when the game ended on the player's own move (no reply)
                # or there is no engine, one ply. Never the engine's lone
                # opening — that is not the player's to take back (409 below).
                session = ctx.session
                vs_engine = ctx.engine is not None
                plies = 2 if vs_engine and session.turn == session.player_color else 1
            # Attempt first, abandon second, exactly as the `undo` tool does: a
            # takeback that cannot happen replaces no position, so the open turn
            # (and the engine reply it is owed) is not this request's to throw
            # away. All of it inside the guard, so a stale takeback costs the
            # open turn nothing either: the position it was about is not the one
            # this request meant.
            result = ctx.session.undo(plies)
            if not result.ok:
                raise HTTPException(status_code=409, detail=result.reason)
            coordinator.abandon_turn()
            # A client may send its own `plies`, and an odd count leaves the
            # engine to move on a board no turn is open over. The coordinator
            # settles it before anyone is shown the position — the same rule the
            # tool follows, because it is the same board either way.
            try:
                coordinator.settle_engine_turn()
            except chess.engine.EngineError:
                # The takeback stands and the reply is left owed (#329): the
                # next move request or command collects it, as after any
                # engine death with the engine to move.
                logger.warning("engine_settle_failed", exc_info=True)
            _publish_state()
            return {
                "undone": list(result.undone),
                "state": _state_dict_unlocked(ctx),
            }

    @app.post("/api/game/resign")
    async def resign(request: ResignRequest) -> Any:
        """Resign — through the same gate as `new_game` and as a spoken "I
        resign" (409 + the question mid-game, `/api/game/confirm` answers it).

        Whose resignation an unqualified one is, is not a caller's judgment any
        more than it is the model's: the `resign` tool defaults it to the
        player's own side, and the side to move is only coincidentally that
        (trace review, finding 8).
        """
        async with _mutation(request.version, request.game_id):
            outcome = _run_destructive(
                "resign", {"color": request.color or ctx.session.player_color}
            )
            if isinstance(outcome, JSONResponse):
                return outcome
            _publish_state()
            return {
                "outcome": outcome["outcome"],
                "state": _state_dict_unlocked(ctx),
            }

    @app.post("/api/game/claim-draw")
    async def claim_draw(request: ClaimDrawRequest) -> Any:
        """Claim a threefold-repetition or fifty-move draw — through the same
        gate as `resign` (409 + the question mid-game, `/api/game/confirm`
        answers it).

        The button half of a claim the state document already advertises
        (`claimable_draws`): until this endpoint existed the tool was reachable
        only through the brain, so in direct mode a claim was unreachable (#220
        follow-up). Whether a claim exists is the tool's own check, before the
        gate — nothing to claim is a plain 409 with nothing armed, so a yes can
        never be an answer to a question about a draw the rules do not allow.
        """
        async with _mutation(request.version, request.game_id):
            outcome = _run_destructive("claim_draw", {})
            if isinstance(outcome, JSONResponse):
                return outcome
            _publish_state()
            return {
                "outcome": outcome["outcome"],
                "state": _state_dict_unlocked(ctx),
            }

    @app.post("/api/game/offer-draw")
    async def offer_draw(request: OfferDrawRequest) -> dict[str, Any]:
        """Offer the engine a draw — the button half of the `offer_draw` tool
        (`docs/draw-offer.md`). Not gated: a decline changes nothing and an
        acceptance ends a position the rule has judged drawn, so there is no
        question to ask. The answer comes back as the tool reports it —
        `accepted`, `reason`, `evaluation`, `material`, plus `outcome` when the
        game ended — and the UI composes its own line from those fields; no
        model stands on this path, as on no other button's. A finished game or
        a missing engine is a 409 with the tool's message.

        Traced as a control interaction like the other buttons, and windowless,
        so the tool's budget check is a no-op here as it is for every button.
        """
        async with _mutation(request.version, request.game_id):
            turn_id = coordinator.turn_id
            version_before = ctx.board_version
            fen_before = ctx.session.fen()
            result = registry.dispatch("offer_draw", {})
            _trace_control(
                "offer_draw",
                {},
                result,
                turn_id=turn_id,
                version_before=version_before,
                fen_before=fen_before,
            )
            if result.get("ok") is not True:
                raise HTTPException(
                    status_code=409, detail=result.get("error", "cannot offer a draw")
                )
            if result["accepted"]:
                # The game is over: whatever question was armed was about it.
                ctx.pending = None
                _publish_state()
            return {
                key: result[key]
                for key in ("accepted", "reason", "evaluation", "material", "outcome")
                if key in result
            } | {"state": _state_dict_unlocked(ctx)}

    @app.post("/api/game/difficulty")
    def set_difficulty(request: DifficultyRequest) -> dict[str, Any]:
        """Set engine strength directly (trusted UI path, not the LLM tool).

        Range is validated here regardless of whether an engine is attached,
        so the setting is always sane; it is applied to the live engine when
        present and re-applied when one attaches later. This does not touch
        board state, so nothing is broadcast.
        """
        try:
            if request.tier is not None:
                validate_tier(request.tier)
                if ctx.engine is not None:
                    ctx.engine.set_tier(request.tier)
                ctx.settings.tier = request.tier
                ctx.settings.skill_level = None
                ctx.settings.elo = None
            elif request.skill_level is not None:
                validate_skill_level(request.skill_level)
                if ctx.engine is not None:
                    ctx.engine.set_skill_level(request.skill_level)
                ctx.settings.skill_level = request.skill_level
                ctx.settings.tier = None
                ctx.settings.elo = None
            else:
                validate_elo(request.elo)
                if ctx.engine is not None:
                    ctx.engine.set_elo(request.elo)
                ctx.settings.elo = request.elo
                ctx.settings.tier = None
                ctx.settings.skill_level = None
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except chess.engine.EngineError as exc:
            # The engine is called before the setting is recorded, so a dead
            # one leaves both where they were (#329).
            raise HTTPException(
                status_code=503, detail="engine unavailable: difficulty not changed"
            ) from exc
        return {
            "tier": ctx.settings.tier,
            "skill_level": ctx.settings.skill_level,
            "elo": ctx.settings.elo,
        }

    def _trace_turn(**fields: Any) -> None:
        """Record the turn, and never let that cost the player one.

        A tracer is a diagnostic sink — a file that may be full, unwritable, or
        on a disk that just went away. None of that is the game's problem, so a
        failure here is logged and dropped rather than turned into a 500 on a
        turn whose moves have already been played. The swallow covers the
        record's own assembly too, which is the other half of that rule now
        that the pipeline traces from a `finally`: the turn handed over may be
        a half-finished one, and a field it never reached must not become a
        second exception on top of the one that ended it.
        """
        if tracer is None:
            return
        try:
            # Board truth at record time, the same rule as `fen_after`: every
            # route's record says how the game stood when the turn was over,
            # so a finished game's commentary can be re-judged from the trace.
            fields.setdefault("outcome", relative_outcome(ctx.session))
            fields.setdefault("game_id", ctx.session.game_id)
            # Written from under the lock on every route, so the spans are
            # this request's; `total` is read now, as the record is made.
            fields.setdefault(
                "spans_ms",
                current_spans.as_trace() if current_spans is not None else None,
            )
            fields.setdefault("serving", _serving())
            tracer.record(turn_record(**fields))
        except Exception:
            logger.warning("trace_failed", exc_info=True)

    def _serving() -> dict[str, str] | None:
        """What served this turn, or None — and never let asking cost the
        record: the identity resolves the live prompts and tool offer, and a
        seam that raises there must lose one field, not the whole trace."""
        if serving_identity is None:
            return None
        try:
            return serving_identity()
        except Exception:
            logger.warning("serving_identity_failed", exc_info=True)
            return None

    @contextmanager
    def _command_window(correlation_id: str, turn_id: int) -> Iterator[None]:
        """One user interaction, opened and closed at both ends at once.

        The coordinator's destructive budget, the progress stream's brackets
        and the trail of boards the command passes through are the same
        interaction seen from three sides — what may happen inside it, what the
        player is told is happening inside it, and where the board went while
        it did — so they are opened together and, more to the point, closed
        together in the same `finally`. A command that raises half-way must
        neither leak an open window into the next one, nor leave a progress
        line spinning, nor hand the next command's honesty guard a board this
        one visited.

        A plain list is enough for the trail: commands are serialized under
        `ctx.mutation_lock`, so there is only ever one window open, and the
        dispatches that fill it are awaited before anything reads it.
        """
        nonlocal command_boards
        assert progress is not None  # bound above; narrows for the type checker
        with progress.interaction(correlation_id, turn_id):
            coordinator.begin_command()
            command_boards = []
            try:
                yield
            finally:
                coordinator.end_command()
                command_boards = None

    async def _run_command(
        text: str,
        transcript: Sequence[dict[str, str]],
        version: int | None = None,
        *,
        origin: str,
        game_id: str | None = None,
    ) -> CommandOutcome:
        """The command pipeline, under the mutation guard — the delegate route's
        entry (the panel opens the guard itself; see `command`).

        A command is a mutation path like any other — one utterance can move the
        board — so it carries the same optional `version` precondition, and a
        stale one is refused before the model is asked anything. The guard wraps
        the *whole* run rather than the individual dispatches inside it: a turn
        is one thing that happens to one board, and half of it landing on a
        board someone else changed is exactly the race this closes.

        `origin` says whose turn this is (`tools.delegate_origin(id)` from the
        delegate router) and is keyword-only and required: a confirmation may be
        armed or answered here, and both halves belong to one conversation.
        """
        async with _mutation(version, game_id):
            return await _command_turn(text, transcript, origin=origin)

    async def _command_turn(
        text: str,
        transcript: Sequence[dict[str, str]],
        *,
        origin: str,
        interaction_id: str = "",
    ) -> CommandOutcome:
        """The single pipeline: user string → brain's tool loop → new state.
        Shared by `/api/command` and the delegate messages endpoint against the
        one game session.

        One call into the brain does the whole turn. Inside it, the agent loop
        runs the utterance to a conclusion — calling tools, reading each result
        (through the validated registry, so a brain mistake comes back as error
        *data*, never an HTTP failure or corrupted state), correcting itself,
        and finally answering in words. That final tool-less turn is the
        commentary, and it is the game loop's "react from the new board": it is
        offered no tools, so it can only comment, never act on the utterance a
        second time. The pipeline no longer decides anything about tools — it
        hands the brain the board, takes back what happened, and broadcasts.

        **The beats around a move.** `make_move` applies the player's move and
        stops, so whatever narration the chosen route produced is a reaction to a
        verified *player* move — the coordinator's observation beat, filled at
        last (audit item 5). The pipeline then closes the turn: collect the reply
        the engine has been computing since the move landed, complete the turn,
        and append a deterministic line announcing it. Two properties are worth
        being explicit about, because both are acceptance criteria. The
        narration overlaps Stockfish rather than queueing behind it, so a plain
        move costs the one model call it always cost. And the announcement is the
        app's own words, not a second narration: Glitch reacts to what the player
        did, and the app reports what answered.

        **One command, one destructive op.** The turn's move budget is the
        phases' (a second player move mid-turn is refused); the destructive ops
        need a wider scope than a turn, because they end the one they run in. So
        this brackets the whole run in a coordinator *command window* — the only
        surface that does, and the only one that can chain dispatches inside a
        single interaction.

        The brain sees the `transcript` the caller supplies (prior turns'
        commands plus the commentary that was shown, a bounded window, final
        answers only), so the agent can follow references to earlier turns —
        but this is transcript-agnostic: it never reads or records either the
        web panel's `ctx.transcript` or a delegate conversation, leaving that
        to the caller. The board broadcast fires here on any board change, so
        a conductor-played move shows up live on the web board too.

        The fast path (the seam BRIEF reserves): an utterance that is exactly
        one unambiguous legal move skips the brain entirely and goes straight
        to make_move — through the same registry, so the road stays one road,
        minus the model. `Brain.narrate` is that route's observation beat. At
        verbosity=low it is skipped and one canned confirmation covers the move
        and the reply, making a plain move a zero-LLM turn; a provider failure
        degrades to the same line, because the reaction is optional and the
        engine's reply is not. Anything ambiguous or non-move reaches the brain
        unchanged.

        **Whose turn this is.** `origin` is the caller's surface — `PANEL_ORIGIN`
        for the web panel, `tools.delegate_origin(id)` for one delegate
        conversation — and it is declared on the context on the way in, under the
        mutation lock, so the gate stamps whatever it arms with the conversation
        the player is being asked in. The same string is what the confirmation
        branch below answers with, so a "yes" settles the question *this* thread
        was asked and no other's (#281).
        """
        assert brain is not None  # both callers guard; documents the invariant
        # Under `_mutation` in both callers, so this is the one interaction
        # running: whoever the gate arms for below, it is this one's question.
        ctx.origin = origin
        # This origin's open question (#319), read the way `live_pending` is:
        # one that no longer stands on this board is dropped here and reported
        # once, as `expired`.
        question, expired = ctx.live_clarification(origin)
        before = _agent_state_dict(ctx)
        # What locates this turn afterwards (audit item 18): the coordinator turn
        # it opened under, an id for this one interaction, and the board version
        # it started from — the mutation count is that version's delta, so the
        # number is *derived* from the chokepoint every mutation already passes
        # through rather than tallied by whichever branch remembered to.
        turn_id = coordinator.turn_id
        correlation_id = new_correlation_id()
        version_before = ctx.board_version
        # One user interaction, one destructive op: the brain loop is the only
        # thing in the app that can chain several dispatches inside a single
        # command, so it is the only surface that needs the coordinator's
        # destructive budget (the buttons and the confirm endpoint dispatch once
        # by construction). The window also brackets the turn's progress stream
        # — same interaction, both ends closed in the same `finally`.
        with _command_window(correlation_id, turn_id):
            # `tool_results` is the {"name", "result"} list the UI sees; `tool_args`
            # mirrors it with each call's arguments, for the delegate wire — kept
            # parallel so the UI-facing shape stays untouched.
            tool_results: list[dict[str, Any]] = []
            tool_args: list[dict[str, Any]] = []
            commentary = ""
            # What the turn is *remembered* by, when that is not what the player
            # was told. `None` means the two are the same, which is every turn
            # Glitch spoke on himself; a route that substitutes the app's own
            # words sets it, and `_remembered_facts` fills an empty one in below.
            memory: str | None = None
            stop_reason = "completed"
            # The road this turn took. Its default is the confirmation branch's,
            # and each other branch names its own on the way in; it is settled
            # here rather than beside that branch because the record below is
            # owned by a `finally` now, and a turn that dies before it has
            # chosen a road still has to say which one it was on.
            route = ROUTE_CONFIRMATION
            # Named only when the brain's loop died on the provider; every other
            # route leaves it empty, which is the record's way of saying "did not
            # die" rather than "not recorded".
            provider_failure = ""
            # The same, for the engine dying on the reply this turn's move had
            # already earned (#284). Not a stop reason: the loop stopped however
            # it stopped, and what changed is that the turn is one move short
            # and still owes the other.
            engine_failure = ""
            # Whether this turn's narration was still being written when the
            # turn went on without it (`_narrate`'s budget). Not a failure
            # either: the words were the only thing owed and the app said its
            # own line instead, so the record is the only place it shows.
            reaction_late = False
            # The brain route's closer was cut (#316); see `_late_close_words`.
            closer_late = False
            # The fast path's move beats, held for the commentary below: with no
            # narration to speak for the turn (verbosity=low, or a provider failure)
            # the move and the engine's reply become one canned confirmation. None on
            # every other route — those close their own turn further down.
            move_beats: _MoveBeats | None = None
            # The board a narration was spoken over while the engine's reply
            # was still owed, on whichever route spoke one (#289) — or None.
            narrated_before_reply: str | None = None
            # The turn's cost at the provider boundary, summed across whatever model
            # calls the chosen route made. The deterministic branches (a canned
            # confirmation, a declined op) leave this at zero — a real, readable
            # zero, which is what tells a later cut it changed nothing here.
            cost = _ModelCost()
            # The turn's trace record, filled in as the turn learns things and
            # written exactly once, by the `finally` below. Owned there rather
            # than by the happy path because the turn a reviewer most wants a
            # record of is the one that died half-way through (#284) — an
            # engine, a narrator, anything — and such a turn used to leave none
            # at all, so whatever it had already put on the board had no
            # explanation anywhere. Now it leaves the utterance, whatever route
            # and results it had reached, and the exception that ended it.
            traced: dict[str, Any] = {
                "utterance": text,
                # Which surface said it — the panel, or one delegate thread.
                # Cheap, and it is the field this whole class of bug is read
                # off: a turn that answered a question asked somewhere else
                # (#281) is invisible in a record that names no origin.
                "origin": origin,
                # What was said, and what it was about: filled in by the happy
                # path alone, because a turn that died said nothing to anybody.
                "commentary": "",
                "changed": False,
                "turn_id": turn_id,
                "correlation_id": correlation_id,
                "interaction_id": interaction_id,
                "fen_before": before["fen"],
                "clarification": _question_trace(question, expired),
            }
            # The candidates of a question this turn asked (`clarify` handoff).
            asked: tuple[str, ...] = ()
            try:
                # An armed destructive op (the tool gate refused new_game/resign
                # last turn and asked). This turn is its answer — and the answer is
                # ours, not the model's: a bare yes runs it with the gate open, a
                # bare no drops it, and anything else is a new intent that disarms
                # it on the way past. The op never survives the turn, so a stale
                # "yes" can never revive it.
                # Read through `live_pending`: a question is about a position, so an
                # op armed against a board that has since moved — the player dragged
                # a move, undid one, another client played — is not something this
                # "yes" can be an answer to, and is dropped instead of run. It is a
                # question asked in a *conversation* too, so it is read with this
                # turn's origin: an op armed for another thread (or for the panel,
                # or by the buttons) is not this utterance's to answer, the reader
                # is never asked to judge it, and the words go down the ordinary
                # road as the fresh intent they are (#281). The disarm below is
                # unconditional either way — every command from any origin clears
                # what was pending on its way in, so the intervening interaction
                # drops the question and the origin that *was* asked hears nothing
                # run either.
                armed = ctx.live_pending(origin)
                ctx.pending = None
                answer = parse_confirmation(text) if armed is not None else None
                if armed is not None and answer is None and brain is not None:
                    # Deterministic first, model second (walkthrough #6). The
                    # literal reader is a short list of bare affirmations, and a
                    # player who says "just do it" after a resign question has
                    # answered it as plainly as "yes" — they just did not use the
                    # word. Reading that is understanding, which is the model's
                    # job; *acting* on it is not, so a confirm goes down the same
                    # `confirm_pending` path a bare yes takes and the model never
                    # touches a destructive tool. `unrelated` — the answer for a
                    # new intent, a provider death, or anything the reader could
                    # not place — leaves the op disarmed and the turn falls
                    # through, exactly as it did before.
                    read = await _offloop(
                        brain.read_answer, _confirm_question(armed.name), text
                    )
                    # Summed, never assigned: an `unrelated` reading goes on
                    # down another road, which adds its own calls to this one
                    # (#290). A reader that died still made a round trip.
                    cost = cost.plus(_ModelCost.of(read, PHASE_ANSWER))
                    if read.verdict == CONFIRM:
                        answer = True
                    elif read.verdict == CANCEL:
                        answer = False
                if armed is not None and answer is not None:
                    if answer:
                        ctx.pending = armed  # confirm_pending consumes it
                        confirmed = await _offloop(
                            confirm_pending, registry, ctx, origin
                        )
                        assert confirmed is not None
                        name, result = confirmed
                        tool_results.append({"name": name, "result": result})
                        tool_args.append(dict(armed.args))
                        if ctx.settings.verbosity == "low":
                            commentary = _destructive_confirmation(
                                name, result, ctx.session
                            )
                        else:
                            # The op already ran; the narration is a garnish on a
                            # board that changed, so a provider failure costs the
                            # words and degrades to the canned line — never a 500
                            # after the mutation, before the broadcast.
                            started = time.monotonic()
                            try:
                                narration = await _offloop(
                                    _narrate,
                                    _narrator_state_dict(ctx),
                                    tool_results,
                                    transcript,
                                    correlation_id,
                                )
                            except ProviderError as exc:
                                # Late words and lost words cost the same thing
                                # here — the canned line — and differ only in
                                # what the record says happened. A late one is
                                # already logged by `_narrate`.
                                reaction_late = isinstance(exc, LateReaction)
                                cost = cost.plus(
                                    _ModelCost.failed(
                                        started,
                                        PHASE_REACTION,
                                        exc,
                                        budget_s=reaction_budget,
                                    )
                                )
                                if not reaction_late:
                                    logger.warning(
                                        "close_narration_failed", exc_info=True
                                    )
                                commentary = _destructive_confirmation(
                                    name, result, ctx.session
                                )
                            else:
                                commentary = narration.text
                                cost = cost.plus(
                                    _ModelCost.of(narration, PHASE_REACTION)
                                )
                    else:
                        # Declined: nothing ran, so there is nothing to narrate from.
                        commentary = _DECLINED_REPLY
                elif (fast_san := parse_move(text, ctx.session.fen())) is not None:
                    # The same beats a board drag runs, on the same helper: the parse is
                    # what differs between the two routes, never the sequencing.
                    route = ROUTE_FAST_PATH
                    move_beats = await _offloop(
                        _play_move, fast_san, transcript, correlation_id
                    )
                    tool_results.extend(move_beats.changes)
                    tool_args.append({"move": fast_san})
                    # The beat's round trip, spoken, lost or late alike (#290).
                    cost = cost.plus(move_beats.cost)
                    if not move_beats.legal:
                        # `parse_move` already matched the move against this board, so a
                        # refusal here is a turn-state rejection (a previous turn left
                        # the machine mid-sequence), not an illegal move: nothing moved,
                        # so there is nothing to react to. The beats still settled
                        # whatever that turn left owing.
                        commentary = STUCK_REPLY
                        memory = ""
                    elif move_beats.narration is not None:
                        commentary = move_beats.narration.text
                elif parse_resign(text):
                    # An explicit resignation is deterministic text, so the model
                    # gets no vote on whether it happened: live, it took one and
                    # answered "Word. Game over." with no tool call on a live
                    # board. The call still goes through the registry, so the gate
                    # arms it and the player's yes — not the agent's word — is
                    # what ends the game.
                    route = ROUTE_RESIGN
                    args = {"color": ctx.session.player_color}
                    result = registry.dispatch("resign", args)
                    tool_results.append({"name": "resign", "result": result})
                    tool_args.append(args)
                    if not result.get("ok"):
                        commentary = (
                            _RESIGN_CONFIRM  # the gate armed it; the answer is theirs
                        )
                    elif ctx.settings.verbosity == "low":
                        commentary = _destructive_confirmation(
                            "resign", result, ctx.session
                        )
                    else:
                        # Same degradation as the confirmed-op narration above: the
                        # resignation is already on the record, so the words are
                        # the only thing a dead provider may cost.
                        started = time.monotonic()
                        try:
                            narration = await _offloop(
                                _narrate,
                                _narrator_state_dict(ctx),
                                tool_results,
                                transcript,
                                correlation_id,
                            )
                        except ProviderError as exc:
                            # Same deal as the confirmed op above: the
                            # resignation is on the record either way, and only
                            # the record tells late from lost.
                            reaction_late = isinstance(exc, LateReaction)
                            cost = cost.plus(
                                _ModelCost.failed(
                                    started,
                                    PHASE_REACTION,
                                    exc,
                                    budget_s=reaction_budget,
                                )
                            )
                            if not reaction_late:
                                logger.warning("close_narration_failed", exc_info=True)
                            commentary = _destructive_confirmation(
                                "resign", result, ctx.session
                            )
                        else:
                            commentary = narration.text
                            cost = cost.plus(_ModelCost.of(narration, PHASE_REACTION))
                else:
                    route = ROUTE_BRAIN
                    response = await _offloop(
                        brain.get_agent_response,
                        planner_state(before, question, expired),
                        text,
                        transcript,
                    )
                    tool_results = list(response.tool_results)
                    tool_args = [call.args for call in response.tool_calls]
                    stop_reason = response.stop_reason
                    provider_failure = response.provider_failure
                    cost = cost.plus(_ModelCost.of(response))
                    # Which boards the planner was re-shown as its own tools
                    # moved them (#282). Stamped on `traced` directly rather
                    # than carried through `_ModelCost`: it is not a cost, and
                    # this is the only route that has a loop to report one.
                    traced["state_refreshes"] = response.state_refreshes
                    traced["planning"] = response.planning
                    traced["offer_refreshes"] = response.offer_refreshes
                    # Which per-turn budget ended the planning phase (#288),
                    # the same way: only this route has a loop to report one.
                    traced["budget"] = response.budget
                    traced["input_trimmed"] = response.input_trimmed
                    # A budget stop with nothing done carries no commentary: no
                    # narrator ran (#288; one after real work is narrated). A provider
                    # stop is left empty here — what it should say depends on
                    # whether anything changed, which the close beat below settles.
                    traced["handoff"] = (
                        response.handoff.trace()
                        if response.handoff is not None
                        else None
                    )
                    if (
                        response.handoff is not None
                        and response.handoff.kind == "clarify"
                    ):
                        asked = response.handoff.candidates
                    # The board the narrator just spoke over, when it spoke
                    # before the engine's reply existed — read here, before
                    # the close beat below collects that reply (#289).
                    if coordinator.phase in (
                        TurnPhase.PLAYER_MOVE_APPLIED,
                        TurnPhase.AGENT_OBSERVING,
                    ):
                        narrated_before_reply = ctx.session.fen()
                    commentary = response.text
                    # A closer the brain stopped waiting for (#316): the plan's
                    # record stands and the words are gone, the same shape as a
                    # late observe beat. The app's own line is composed after
                    # the guard, like every deterministic line, so it is never
                    # guarded or rewritten; nothing of the words is remembered.
                    closer_late = reaction_late = response.narration_late
                    if closer_late:
                        memory = ""
                    elif not commentary and stop_reason != "provider_error":
                        commentary = STUCK_REPLY
                        memory = ""
                    elif stop_reason == "provider_error":
                        # The lost-brain line prefixed below is the app's, not
                        # Glitch's. Whatever he managed to say before the provider
                        # died is his and is remembered; an empty one falls through
                        # to the facts.
                        memory = commentary
                # The close beat, at the one point every route converges. A
                # coordinator left mid-sequence means the player's move landed
                # without its reply — whichever route played it — and whatever
                # narration that route produced was this turn's reaction to it.
                # So: collect the answer the engine has been computing all along,
                # close the turn, and announce the reply in the app's own words.
                # `complete_turn` is deliberately the pipeline's and not the
                # tool's: nothing may close a turn the engine still owes a move to.
                engine_reply: MoveResult | None = None
                owed_reply = False
                if move_beats is not None:
                    # The fast path ran the beats already, close included; what they
                    # settled is what this turn has to say for itself — including a
                    # reply its engine died on.
                    engine_reply, owed_reply, engine_failure = (
                        move_beats.engine_reply,
                        move_beats.owed_reply,
                        move_beats.engine_failure,
                    )
                    reaction_late = move_beats.reaction_late
                elif coordinator.phase in (
                    TurnPhase.PLAYER_MOVE_APPLIED,
                    TurnPhase.AGENT_OBSERVING,
                ):
                    owed_reply = True
                    try:
                        with _span("engine"):
                            engine_reply = await _offloop(
                                coordinator.collect_engine_reply
                            )
                    except Exception as exc:
                        # The player's move is committed and broadcast, so an
                        # engine that dies here is not this command's failure to
                        # report — it is a turn with one move in it (#284). The
                        # same recovery the fast path's close beat has: name what
                        # died, leave the turn open where the coordinator put it
                        # (the reply is still owed and the next command settles
                        # it), and tell the player in the app's own line below.
                        engine_failure = _failure_name(exc)
                        logger.warning(
                            "engine_reply_failed",
                            exc_info=True,
                            extra={"correlation_id": correlation_id},
                        )
                    else:
                        coordinator.complete_turn()
                elif (settled := _settled_engine_move(tool_results)) is not None:
                    # No turn was open, and the engine moved anyway: a restore left
                    # it on move and the coordinator settled that board inside the
                    # tool (a resumed mid-exchange save, an odd-ply takeback, a new
                    # game as black). Nothing to collect, but the same thing to say
                    # — the player asked to load a game and the board moved twice,
                    # so the app announces the move it made the way it announces
                    # every other one. Voice-first, an unannounced reply is a board
                    # the player cannot see changing under them.
                    owed_reply = True
                    engine_reply = settled
                # The honesty guard, at the one point every route converges: an
                # operational claim the turn cannot back is not shown to the player.
                # The board, the engine's reply and the tool results are the record of
                # what happened; the model's prose is not, and live it has claimed
                # resignations and checkmates that never occurred (trace review,
                # finding 6). This is the same rule as the gate, applied one step
                # later — the model may neither *do* a destructive op unasked nor
                # *say* it did, nor announce any other fact it invented. What it
                # may do is say it again with the facts right (`_honest_words`).
                #
                # It runs on the *model's* half of the turn and nothing else. The
                # app's own lines — the reply announcement, the canned confirmation,
                # the lost-brain line — are composed around whatever survives, below.
                # They are deterministic truth by construction, so there is nothing
                # in them to guard; running them through it only risks taking back
                # the engine's move along with the lie, which is the one fact a
                # guarded turn cannot afford to drop (the board moved under the
                # player and a rewrite may say nothing about how).

                # Every board this turn actually held, and not just its two ends.
                # The trail is what the command's own mutating calls left behind, in
                # order; the fast path names its observation board on top, because
                # that route knows *which* position its words were written from.
                # (The two overlap — a fast-path `make_move` is a dispatch like any
                # other — and `_verified_facts` dedupes them.)
                observed = list(command_boards or ())
                if move_beats is not None and move_beats.observed_fen is not None:
                    observed.append(move_beats.observed_fen)
                    if move_beats.owed_reply:
                        # The observe beat is, by construction, a narration
                        # spoken before the reply exists.
                        narrated_before_reply = move_beats.observed_fen
                # The advice guard rides along, at the same point (audit item 11's
                # second half): a currently-playable move in the commentary is a
                # hint whatever prose carries it. It applies on a turn that left
                # the *board* alone — reacting to a move just played is
                # description, not advice — and only once the turn has evidence:
                # an analysis tool reported moves, and the reply names a playable
                # move outside everything the tools reported. With no analysis in
                # the turn there is nothing to contradict, and a move Glitch names
                # is his opinion (decided 2026-09-10; `analysis_moves`).
                #
                # The board, specifically, and not the agent view: a turn that
                # changed a *setting* changed nothing about what the player should
                # play, so a setter must not buy an exemption. It used to, for
                # every setter whose value the view carried — `set_verbosity` was
                # only ever the exception because verbosity was missing from that
                # view, which is the very gap walkthrough #3 came out of.
                #
                # A question naming two or more legal moves is exempt, and it is
                # the model doing its job: "Do you mean Nf3 or Nh3?" is what an
                # ambiguous request deserves, and the guard used to eat it whole
                # (audit finding 6). `unlicensed_advice` owns that reading — the
                # code still decides what is licensed, the model still owns the
                # words.
                advice = None
                if ctx.board_version == version_before and (
                    evidence := analysis_moves(tool_results)
                ):
                    legal = frozenset(ctx.session.legal_moves())
                    advice = _AdviceLicence(
                        unlicensed=legal - reported_moves(tool_results),
                        legal=legal,
                        evidence=frozenset(evidence),
                    )
                guard_started = time.monotonic()
                verdict = await _honest_words(
                    brain,
                    _offloop,
                    commentary,
                    _verified_facts(
                        ctx,
                        tool_results,
                        engine_reply,
                        before["fen"],
                        observed,
                        narrated_before_reply,
                    ),
                    advice,
                    transcript,
                    {"text": text, "correlation_id": correlation_id},
                )
                # The guard's own time; its rewrite is model time (#290).
                _add_span(
                    "guard", _ms_since(guard_started) - sum(verdict.cost.latencies_ms)
                )
                commentary = verdict.text
                cost = cost.plus(verdict.cost)
                if verdict.fell_back:
                    memory = ""
                elif memory is None:
                    # What Glitch himself said, taken *before* the app's lines are
                    # composed around it below. The reply announcement is the app's
                    # voice: remembered as his, its trailing "\n\ne5." is a format
                    # he completes at the beat where the reply does not exist yet —
                    # live, the first announced move followed the first remembered
                    # announcement by exactly one turn (#193). The player hears the
                    # composed whole; the model is given back only its own words.
                    memory = commentary
                if move_beats is not None and move_beats.legal:
                    commentary = _move_commentary(
                        commentary,
                        move_beats.result,
                        engine_reply,
                        owed_reply,
                        ctx.session,
                    )
                elif closer_late and not commentary:
                    commentary = _late_close_words(
                        tool_results,
                        engine_reply,
                        owed_reply,
                        _agent_state_dict(ctx) != before,
                        ctx.session,
                    )
                elif owed_reply and (
                    reply_line := _reply_announcement(engine_reply, ctx.session)
                ):
                    commentary = (
                        f"{commentary}\n\n{reply_line}" if commentary else reply_line
                    )
                if verdict.fell_back and not commentary:
                    # Both drafts cut and no deterministic line to stand in: the
                    # same thing the player hears when the loop ran out of budget,
                    # because it is the same situation — the model produced no
                    # usable answer — and an empty bubble reads as a crash.
                    commentary = STUCK_REPLY
                if stop_reason == "provider_error":
                    # Recovery semantics (audit item 20): the turn is already
                    # settled — whatever ran stands, the reply was collected above —
                    # so the only thing left to own is what the player is told. The
                    # line the code picks is the one the board supports.
                    lost = (
                        PROVIDER_LOST_TURN_STANDS
                        if _agent_state_dict(ctx) != before
                        else PROVIDER_LOST_RETRY
                    )
                    commentary = f"{lost}\n\n{commentary}" if commentary else lost
                if engine_failure:
                    # The other half of the same deal, one layer down: the move
                    # stands, the reply does not exist, and the player is told
                    # so in the app's own words rather than by a 500 arriving
                    # after their move was already broadcast. Composed after
                    # the reply announcement's slot, because it is the sentence
                    # that slot could not hold (`_engine_lost_words`), and
                    # deliberately not in `memory`: the app's lines are shown
                    # to the player and never fed back as Glitch's.
                    commentary = _engine_lost_words(commentary)
                agent_state = _agent_state_dict(ctx)
                # If this turn armed a destructive op, the question goes to the
                # player about the board they can see *now* — this turn's mutations
                # included, since the gate arms mid-turn and the engine's reply can
                # land after it ("play e4 and start over"). Their yes next turn is
                # an answer to that board and no other.
                ctx.restamp_pending()
                # The same moment for the open question: settled by what this
                # turn did to the board, or replaced by the one it asked.
                traced["clarification"].update(
                    _settle_question(
                        ctx, origin, question, version_before, tool_results, text, asked
                    )
                )
                # The UI still gets its own full document; a mutation shows up in the
                # agent view too (any board change moves the fen), so that comparison
                # decides the broadcast. What the turn already published as it ran
                # is not sent again — the emitter dedupes by board version.
                state = _state_dict_unlocked(ctx)
                changed = agent_state != before
                if changed:
                    _publish_state()
                traced.update(
                    commentary=commentary,
                    changed=changed,
                    engine_reply=_move_reply_dict(engine_reply),
                    guarded=verdict.fired,
                    guarded_claims=verdict.claims,
                    suppressed=verdict.suppressed,
                    rewrite=verdict.rewrite,
                    rewrite_claims=verdict.rewrite_claims,
                    rewrite_suppressed=verdict.rewrite_suppressed,
                    provider_failure=provider_failure,
                    engine_failure=engine_failure,
                )
                return CommandOutcome(
                    commentary=commentary,
                    tool_results=tool_results,
                    tool_args=tool_args,
                    state=state,
                    changed=changed,
                    stop_reason=stop_reason,
                    # Every branch above has settled `memory` by here (the guard
                    # block fills the last None in), so what is left is only the
                    # empty case: a turn Glitch said nothing on remembers the
                    # deterministic facts, or nothing at all.
                    memory=memory
                    or _remembered_facts(tool_results, engine_reply, ctx.session),
                    engine_failure=engine_failure,
                    correlation_id=correlation_id,
                )
            except BaseException as exc:
                # Nothing is handled here — the caller still gets its exception,
                # and an endpoint still answers however it answers. What this
                # buys is the *record*: the turn names what killed it, and the
                # `finally` below writes the one it would otherwise never leave.
                traced["error"] = _failure_name(exc)
                # Board truth, not the agent view the happy path compares: a
                # read that can itself fail would mask the exception being
                # re-raised, and what a dead turn is asked here is whether it
                # left anything behind.
                traced["changed"] = ctx.board_version != version_before
                raise
            finally:
                # What happened, as opposed to what was said: known wherever
                # the turn got to, so it is read here rather than remembered at
                # some point the turn may never have reached. The road is
                # whichever branch claimed it (the default one if none did) and
                # the stop reason is the loop's if the loop ran; the rest is
                # read off the board, so a turn that died between a mutation
                # and its bookkeeping still counts what it moved.
                traced["route"] = route
                traced["stop_reason"] = stop_reason
                traced["reaction_late"] = reaction_late
                # The calls the turn had paid for by the time it ended — all of
                # them on a finished turn, and on one that died, the ones before
                # it did (#290): a dead turn's round trips were still made.
                traced.update(cost.as_trace())
                traced["mutations"] = ctx.board_version - version_before
                traced["fen_after"] = ctx.session.fen()
                # `turn_record` zips the call args and the results strictly, and
                # a turn that died between a dispatch and the args beside it has
                # one of a pair. Recording the complete pairs is worth more than
                # a `TypeError` swallowed into no record at all.
                pairs = min(len(tool_args), len(tool_results))
                traced["tool_calls"] = tool_args[:pairs]
                traced["tool_results"] = tool_results[:pairs]
                _trace_turn(**traced)

    @app.post("/api/command")
    async def command(request: CommandRequest) -> dict[str, Any]:
        """User string → brain → tool call(s) → new state, for the web panel.

        Supplies the panel's own conversation memory and records the settled
        turn back onto it. Memory, not the raw window: recent turns verbatim
        behind a digest of what the player asked for earlier
        (`docs/turn-memory.md`).

        **Read, run, record is one serialized section**, which is why this
        opens `_mutation` itself rather than going through `_run_command` (the
        delegate route's entry, which guards the run alone because `agent_api`
        already serializes the three under its own per-conversation lock). Split
        them and the panel's conversational causality reorders: two overlapping
        commands — voice and text, or two tabs — would both snapshot memory
        before either ran, and the second would reason from a conversation
        missing the exchange that finished before it started, while the board it
        was handed was perfectly up to date (#285). Under the guard, a queued
        follow-up's read cannot begin until the turn ahead of it has recorded.

        The lock order is "mutation lock only": there is no second, panel-side
        exchange lock to take, so a follow-up parks on `ctx.mutation_lock` and
        nothing else. The version precondition still refuses a stale request
        before anything runs — `_mutation` checks it on the inside of the
        acquire — so a 409 turn reads no memory and records nothing.
        """
        if brain is None:
            raise HTTPException(status_code=503, detail="agent unavailable: no brain")
        async with _mutation(request.version, request.game_id):
            transcript = ctx.transcript.memory()
            outcome = await _command_turn(
                request.text,
                transcript,
                origin=PANEL_ORIGIN,
                interaction_id=request.interaction_id or "",
            )
            # Record on the context, not a captured reference: resume_game may
            # have just swapped in the saved game's transcript, and this turn
            # belongs to that thread.
            ctx.transcript.record(request.text, outcome.memory)
        return {
            "commentary": outcome.commentary,
            "tool_results": outcome.tool_results,
            "state": outcome.state,
            # Whether the client should voice the commentary (the user's
            # voice_output setting, agent-togglable via set_voice_output).
            # The server owns the decision; the client owns the playback.
            "speak": ctx.settings.voice_output,
            # The difficulty tier after the turn, for the same reason as
            # `speak`: the UI selector reflects only what the server confirms,
            # and an agent-side set_difficulty otherwise stays invisible until
            # a reload. Null when strength was set outside the tiers.
            "tier": ctx.settings.tier,
            # The turn's trace id, so the browser's milestones for this
            # interaction can name the turn record they belong to (#317).
            "correlation_id": outcome.correlation_id,
        }

    app.include_router(
        build_agent_router(
            store=store,
            run_command=_run_command if brain is not None else None,
        )
    )

    @app.get("/api/settings")
    def get_settings() -> dict[str, Any]:
        """The agent-adjustable settings, for the UI to render its controls
        from (the same truth the tools mutate).

        `agent_available` is not a setting but the operating mode: whether a brain
        is configured at all. It is here because direct mode has to be *visible* —
        the audit's item 1 — rather than something the player discovers when the
        command box 503s and their drags come back without a word from Glitch.
        """
        s = ctx.settings
        return {
            "verbosity": s.verbosity,
            "voice_output": s.voice_output,
            "tier": s.tier,
            "skill_level": s.skill_level,
            "elo": s.elo,
            "agent_available": brain is not None,
        }

    @app.post("/api/settings/voice")
    def set_voice_output(request: VoiceOutputRequest) -> dict[str, Any]:
        """Voice output on/off from the UI (trusted path, mirroring the
        `set_voice_output` tool — the mute button shouldn't need the LLM).
        Not a board mutation, so nothing is broadcast."""
        ctx.settings.voice_output = request.enabled
        return {"voice_output": ctx.settings.voice_output}

    def _trace_event(kind: str, fields: dict[str, Any]) -> None:
        """Append a non-turn record (#317), best-effort like every trace write:
        a diagnostic that fails is logged and dropped, never a failed request."""
        if tracer is None:
            return
        try:
            tracer.record({"schema": TRACE_SCHEMA, "kind": kind, **fields})
        except Exception:
            logger.warning("trace_failed", exc_info=True)

    def _speech_record(
        op: str, interaction_id: str | None, started: float, status: str, **sizes: int
    ) -> None:
        """One speech round trip, for the latency report: which interaction,
        how long, how it ended, and sizes — never the audio or the words."""
        _trace_event(
            TRACE_SPEECH,
            {
                "op": op,
                "interaction_id": _header_id(interaction_id),
                "ms": _ms_since(started),
                "status": status,
                **sizes,
            },
        )

    @app.post("/api/voice/transcribe")
    async def transcribe(
        audio: UploadFile,
        interaction_id: Annotated[str | None, Header(alias="X-Interaction-Id")] = None,
    ) -> dict[str, Any]:
        """STT proxy: browser audio in, plain text out. The text goes back to
        the client, which feeds it into the same /api/command pipeline as
        typed input — voice never gets its own path to the game. Board state
        is untouched here, so nothing is broadcast. Like the brain (503
        without one), voice is optional: no speech service means 503, and an
        unreachable/failing one is the upstream's fault (502)."""
        if speech is None:
            raise HTTPException(
                status_code=503, detail="voice unavailable: no speech service"
            )
        data = await audio.read()
        started = time.monotonic()
        try:
            text = await _offloop(
                speech.transcribe, data, audio.filename or "audio.webm"
            )
        except Exception as exc:
            _speech_record("stt", interaction_id, started, "failed", bytes=len(data))
            raise HTTPException(
                status_code=502, detail=f"speech service error: {exc}"
            ) from exc
        _speech_record(
            "stt", interaction_id, started, "ok", bytes=len(data), chars=len(text)
        )
        return {"text": text}

    @app.post("/api/voice/speak")
    def speak_text(
        request: SpeakRequest,
        interaction_id: Annotated[str | None, Header(alias="X-Interaction-Id")] = None,
    ) -> Response:
        """TTS proxy: text in, mp3 out — the audio for whatever commentary
        the client decided to voice. Same optionality contract as
        /api/voice/transcribe: 503 without a speech service, 502 when the
        upstream fails; board state is never touched."""
        if speech is None:
            raise HTTPException(
                status_code=503, detail="voice unavailable: no speech service"
            )
        started = time.monotonic()
        chars = len(request.text)
        try:
            audio = speech.speak(request.text)
        except Exception as exc:
            _speech_record("tts", interaction_id, started, "failed", chars=chars)
            raise HTTPException(
                status_code=502, detail=f"speech service error: {exc}"
            ) from exc
        _speech_record(
            "tts", interaction_id, started, "ok", chars=chars, bytes=len(audio)
        )
        return Response(content=audio, media_type="audio/mpeg")

    @app.post("/api/telemetry/voice", status_code=204)
    def voice_telemetry(report: VoiceTelemetry) -> Response:
        """The browser's milestones for one interaction (#317): speech end,
        transcript, board, reply, first audio, playback end — in its own clock.

        Appended to the trace beside the turn and speech records it joins by
        id, and nothing else: it touches no session, takes no lock, and answers
        204 whether or not anything is tracing, so a client never has to know.
        The server stamps its own receipt time (`ts`) and never reconciles it
        with the client's offsets — the two clocks have no common origin."""
        _trace_event(TRACE_VOICE, report.model_dump())
        return Response(status_code=204)

    @app.get("/api/game/hint")
    def get_hint() -> dict[str, Any]:
        """Best move for the side to move, for the UI's hint arrow (trusted
        path, same engine the `suggest_moves` analysis uses). Needs an engine
        (503 without one); a finished or move-less position is a domain
        failure (409). Read-only — nothing is broadcast.

        The payload names the board it analyzed (`version`, the same counter the
        state document publishes), because a search takes real time and nothing
        holds the board still while it runs — another client, or this one
        dragging a piece. Without it a client cannot tell an answer about its
        board from an answer about the board before it, and a late response
        paints the old position's arrow onto the new one (#218).

        The version and the searched position come out of one `_session_snapshot`
        — the same indivisible step — and the engine runs on that snapshot, so
        the number and the analyzed board are the same board by construction.
        Reading the version off the live session merely failed *safe* (an old
        number labelling a newer board, which a client discards); it also left
        the search itself reading a session another thread was mid-`push` on
        (#230). The search runs with the lock released: the copy is what the
        boundary is held for.
        """
        if ctx.engine is None:
            raise HTTPException(status_code=503, detail="hint unavailable: no engine")
        version, snapshot = _session_snapshot(ctx)
        if snapshot.is_game_over():
            raise HTTPException(
                status_code=409, detail="cannot suggest moves: game is over"
            )
        try:
            candidates = ctx.engine.get_best_moves(snapshot, n=1)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except chess.engine.EngineError as exc:
            raise HTTPException(
                status_code=503, detail="hint unavailable: the engine stopped"
            ) from exc
        if not candidates:
            raise HTTPException(status_code=409, detail="no candidate moves")
        best = candidates[0]
        # uci[2:4] is the destination even for 5-char promotion UCIs.
        return {
            "uci": best.uci,
            "san": best.san,
            "from": best.uci[:2],
            "to": best.uci[2:4],
            "version": version,
        }

    @app.get("/api/game/review")
    def get_review() -> dict[str, Any]:
        """Whole-game review for the UI (trusted path, same numbers as the
        `review_game` tool). Analysis needs Stockfish (503 without one);
        reviewing an empty game is a domain failure (409). Read-only —
        nothing is broadcast.

        Reviewed off a `_session_snapshot` rather than the live game: the sweep
        is one engine analysis per position — seconds — and it opens by
        serializing the whole move stack, which is what tore when a mutation
        landed underneath it (#230). The copy is taken at the mutation boundary
        and the sweep runs with the lock released, so a review never makes a
        drag wait for it."""
        if ctx.engine is None:
            raise HTTPException(status_code=503, detail="review unavailable: no engine")
        _, snapshot = _session_snapshot(ctx)
        try:
            review = review_game(ctx.engine, snapshot)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except chess.engine.EngineError as exc:
            raise HTTPException(
                status_code=503, detail="review unavailable: the engine stopped"
            ) from exc
        return {
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

    @app.get("/api/game/pgn")
    def export_pgn() -> dict[str, Any]:
        """The game so far as PGN, off a `_session_snapshot`. `export_pgn`
        replays the whole move stack, and replaying one that another thread is
        popping is what made "Copy PGN" raise out of `board.root()` and answer
        the player a 500 (#230).

        Headers come from the same composer the `export_pgn` tool uses, so the
        PGN the post-game screen copies and the one the chat hands over are the
        same document — and composed for the snapshot, not the live game, so
        the tags describe the moves that ship with them."""
        _, snapshot = _session_snapshot(ctx)
        return {"pgn": snapshot.export_pgn(pgn_headers(ctx, snapshot))}

    if static_dir is not None:
        # Serve the built frontend from the same origin as the API, so the
        # UI's relative /api + /ws URLs work with no proxy or CORS. Mounted
        # last: explicit routes above always win over the catch-all.
        # The hands-free VAD ships onnxruntime WASM under /vad/; browsers
        # compile it with instantiateStreaming, which requires the response
        # to be application/wasm — not every Python's mimetypes table knows
        # the extension, so register it explicitly.
        mimetypes.add_type("application/wasm", ".wasm")
        app.mount("/", StaticFiles(directory=static_dir, html=True), name="frontend")

    return app
