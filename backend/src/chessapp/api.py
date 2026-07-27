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
  mode visible in the UI rather than a per-input surprise.
- **Every mutation is version-checkable and serialized** (audit item 7). The
  state document carries `version` (`ToolContext.board_version`), and every
  mutating request may carry the one it last saw: superseded means 409 with
  `stale: true` and the current state, omitted means today's behavior. The check
  and the mutation happen under `ctx.mutation_lock` and cannot be split, so two
  clients on the one shared session cannot advance the same turn twice — see
  `_mutation` and `docs/turn-coordinator.md`.
- **The destructive-op gate is one system.** `/api/game/new` and
  `/api/game/resign` dispatch through the registry, so the same deterministic
  gate that refuses an unconfirmed `new_game`/`resign` for the agent arms
  `ctx.pending` for a button press too: mid-game those endpoints answer 409
  with the gate's question (`confirm: true`), and `/api/game/confirm` answers
  it. It is the *same* armed op the spoken road uses, so a question asked by a
  button can be answered by a typed "yes" and vice versa. Undo is not
  destructive and keeps its direct endpoint.

Always read `ctx.session` per request: `resume_game` swaps the session
object on the context.
"""

import asyncio
import logging
import math
import mimetypes
import random
from collections.abc import AsyncIterator, Callable, Iterator, Sequence
from contextlib import asynccontextmanager, contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anyio.to_thread
from fastapi import (
    FastAPI,
    HTTPException,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, model_validator

from chessapp.agent_api import ConversationStore, build_agent_router
from chessapp.analysis import review_game
from chessapp.brain import Brain, Narration
from chessapp.coordinator import TurnCoordinator, TurnPhase, TurnStateError
from chessapp.engine import validate_elo, validate_skill_level, validate_tier
from chessapp.fastparse import parse_confirmation, parse_move, parse_resign
from chessapp.game import GameSession, MoveResult
from chessapp.honesty import VerifiedFacts, names_a_legal_move, unverified_claims
from chessapp.progress import ProgressEvent, ProgressReporter
from chessapp.provider import ProviderError
from chessapp.tools import (
    DESTRUCTIVE_TOOLS,
    UNDO_PLIES_MAX,
    ToolContext,
    ToolRegistry,
    build_registry,
    confirm_pending,
    saved_game_names,
)
from chessapp.trace import (
    ROUTE_BOARD,
    ROUTE_BRAIN,
    ROUTE_CONFIRMATION,
    ROUTE_CONTROL,
    ROUTE_FAST_PATH,
    ROUTE_RESIGN,
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

    def __init__(self, expected: int, current: int) -> None:
        super().__init__(f"stale board version {expected}; current is {current}")
        self.expected = expected
        self.current = current


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


class MoveRequest(VersionedRequest):
    move: str


class CommandRequest(VersionedRequest):
    text: str


class SpeakRequest(BaseModel):
    text: str = Field(min_length=1, pattern=r"\S")


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
    """The full state document the board UI renders from.

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
    }


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

    `settings` is here for that same reason. Verbosity and hints re-resolve into
    the system prompt per command, but difficulty and voice-output appeared
    nowhere in the agent's per-turn view, so "how hard am I playing?" could only
    be answered from stale conversation text — the self-poisoning shape again.
    Kept small (it ships in the prompt every turn on a 12B): only the one
    difficulty field `Settings` actually has set, so the block can never imply
    two difficulties are in force at once.
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
        "saved_games": saved_game_names(ctx),
        "settings": _agent_settings_dict(ctx),
    }


def _agent_settings_dict(ctx: ToolContext) -> dict[str, Any]:
    """The live settings the brain is shown: difficulty (exactly the one field
    of tier / skill_level / elo that is set) and voice output."""
    settings = ctx.settings
    difficulty: dict[str, Any] = {}
    for field in ("tier", "skill_level", "elo"):
        value = getattr(settings, field)
        if value is not None:
            difficulty = {field: value}
            break
    return {"difficulty": difficulty, "voice_output": settings.voice_output}


def _move_dict(result: MoveResult) -> dict[str, Any]:
    return {"legal": result.legal, "san": result.san, "uci": result.uci}


def _move_reply_dict(reply: MoveResult | None) -> dict[str, Any] | None:
    """The engine's reply for the trace record: `{"san", "uci"}`, or None when
    none was owed. It no longer rides inside a tool result, so this is the only
    place a traced move turn can learn what answered it."""
    if reply is None:
        return None
    return {"san": reply.san, "uci": reply.uci}


def _destructive_confirmation(
    name: str, result: dict[str, Any], session: GameSession
) -> str:
    """Deterministic stand-in for the reaction after a confirmed new_game/resign
    at verbosity=low — the twin of `_move_confirmation`, keeping a confirmed
    destructive op a zero-LLM turn like a plain move is."""
    if name == "resign":
        outcome = result.get("outcome") or _outcome_dict(session)
        if outcome:
            return f"Game over: {outcome['result']} ({outcome['termination']})."
        return "Game over."
    parts = ["New game."]
    engine_move = result.get("engine_move")
    if engine_move:
        parts.append(f"{engine_move['san']}.")
    return " ".join(parts)


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


# What the player hears when the brain's loop ran out of budget instead of
# answering (`max_iterations` / `correction_limit`): those stops carry no
# commentary, and an empty bubble would read as a crash.
_STUCK_REPLY = "I lost the thread on that one — say it again?"
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

# What the player hears instead of a lie. The guard below catches commentary that
# announces the game ended (or restarted) when the board says otherwise; rather
# than emit it, the pipeline says what is actually true. Public so tests can pin
# the substitution rather than a wording.
UNTRUE_CLAIM_REPLY = (
    "Scratch that — the game's still live, and I didn't actually do that. "
    "Say it again if you meant it."
)

# What the player hears instead of any *other* invented fact (audit item 13):
# a capture that never happened, a move that was never on the board, a setting
# nothing set, an engine number no engine produced. Separate from the line
# above because the two corrections carry different information and the ending
# one carries the fact that matters most — the game is still live. Public so
# tests pin the substitution, not a wording.
UNVERIFIED_CLAIM_REPLY = (
    "Scratch that — I said something the board doesn't back up. "
    "Ask me again and I'll stick to what actually happened."
)

# Board symbol → the word commentary uses for it, for the capture claim class.
_PIECE_NAMES = {
    "p": "pawn",
    "n": "knight",
    "b": "bishop",
    "r": "rook",
    "q": "queen",
    "k": "king",
}

# What the player hears instead of a hint they turned off. The advice guard
# catches commentary that hands over a currently-playable move when hints are
# off and no analysis tool the player asked for reported it (audit item 11's
# response check — the capability cut stops the tool call, this stops the
# model's own head). Public so tests pin the substitution, not a wording.
MOVE_ADVICE_REPLY = (
    "Hints are off, so you're not getting a move out of me. "
    "Ask me to turn hints on if you want help."
)

# The confirmation question for a resignation the pipeline itself dispatched.
# Deterministic, like the gate it came from: the model is not consulted about a
# resignation at any point, including how to ask about one.
_RESIGN_CONFIRM = "That's the game if you mean it. Say yes and I'll resign for you."

# The same questions for a destructive op the *board UI* armed. Deterministic for
# the same reason (`_RESIGN_CONFIRM` is this rule on the spoken road), and
# phrased for a dialog rather than for a spoken yes.
_CONFIRM_QUESTIONS = {
    "new_game": "That ends the game in progress. Start a new one?",
    "resign": "That's the game if you mean it. Resign?",
}


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
            "detail": _CONFIRM_QUESTIONS.get(op, f"Confirm {op}?"),
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
    """

    changes: list[dict[str, Any]]
    narration: Narration | None
    engine_reply: MoveResult | None
    owed_reply: bool

    @property
    def result(self) -> dict[str, Any]:
        """The `make_move` result itself."""
        return self.changes[0]["result"]

    @property
    def legal(self) -> bool:
        return self.result.get("legal") is True


def _reported_moves(tool_results: Sequence[dict[str, Any]]) -> set[str]:
    """The moves the turn's analysis tools actually named, in SAN.

    The advice guard's licence, and it is scoped to these moves rather than
    switched off wholesale. Analysis the player asked for reports moves, and
    commentary repeating one is a fact — "what was my mistake?" works with
    hints off, and its answer names the move played and the move that beat it.
    What a *successful analysis call* is not is a licence for every other legal
    move: live, the planner answered "what should I play here?" with
    `evaluate_position` + `analyze_last_move` and the old boolean test read
    that as permission, letting the narrator hand over a list of moves no tool
    had mentioned (docs/agent-evals.md, 2026-07-25).
    """
    reported: set[str] = set()
    for r in tool_results:
        result = r["result"]
        if result.get("ok") is not True:
            continue
        if r["name"] == "get_best_moves":
            reported.update(m["san"] for m in result.get("moves", ()) if m.get("san"))
        elif r["name"] == "analyze_last_move":
            reported.update(
                san for san in (result.get("played"), result.get("best")) if san
            )
    return reported


def _analysis_numbers(tool_results: Sequence[dict[str, Any]]) -> set[str]:
    """Every number the turn's analysis tools reported, in the spellings a
    commentary might quote them in: raw centipawns, pawns to two places, and
    both one-place roundings, signed and unsigned. The evaluation claim class
    checks against this, so a score with no analysis behind it has nothing to
    derive from.

    Both roundings because the sign and the magnitude are the fact and the
    rounding is wording — 147 centipawns said as "1.4" is the same report as
    "1.5", and replacing good commentary over the tenths place would cost more
    than that lie is worth.
    """
    numbers: set[str] = set()

    def record(score_cp: int | None, mate_in: int | None) -> None:
        if score_cp is not None:
            numbers.add(str(score_cp))
            pawns = score_cp / 100
            tenths = (math.floor(pawns * 10) / 10, math.ceil(pawns * 10) / 10)
            for text in (f"{pawns:.2f}", *(f"{tenth:.1f}" for tenth in tenths)):
                numbers.add(text)
                numbers.add(f"+{text}" if not text.startswith("-") else text)
        if mate_in is not None:
            numbers.update({str(mate_in), str(abs(mate_in))})

    for r in tool_results:
        result = r["result"]
        if result.get("ok") is not True:
            continue
        if r["name"] == "evaluate_position":
            record(result.get("score_cp"), result.get("mate_in"))
        elif r["name"] == "get_best_moves":
            for candidate in result.get("moves", ()):
                record(candidate.get("score_cp"), candidate.get("mate_in"))
        elif r["name"] == "analyze_last_move":
            record(result.get("cp_loss"), None)
        elif r["name"] == "review_game":
            for move in result.get("moves", ()):
                record(move.get("cp_loss"), None)
            numbers.update(str(value) for value in result.get("accuracy", {}).values())
            numbers.update(str(value) for value in result.get("counts", {}).values())
    return numbers


def _verified_facts(
    ctx: ToolContext,
    tool_results: Sequence[dict[str, Any]],
    engine_reply: MoveResult | None,
    fen_before: str,
) -> VerifiedFacts:
    """What this turn may honestly say, assembled from the record of it.

    Audit item 13, the pipeline's half. The ending guard's evidence — board plus
    tool results — generalized to every operational fact a turn produces, and
    assembled here for the same reason the ending check is: this is the one
    place that holds the tool results, the engine's reply and the board at once.

    `moves` deliberately spans the whole game rather than this turn's position.
    A reaction legitimately names the move the player *didn't* play ("Bb5 was
    better"), which stopped being legal the moment the turn played something
    else, and reciting the move list or a PGN is a read the tools support —
    both turn up in the 46 recorded live turns, and both are board truth.
    """
    outcome = ctx.session.outcome()
    captured = ctx.session.captured_pieces()
    opponent = "black" if ctx.session.player_color == "white" else "white"
    moves = (
        set(ctx.session.legal_moves())
        | set(GameSession(fen=fen_before).legal_moves())
        | set(ctx.session.move_history())
    )
    moves |= _reported_moves(tool_results)
    if engine_reply is not None and engine_reply.san:
        moves.add(engine_reply.san)
    checked = ctx.session.is_check()
    for r in tool_results:
        result = r["result"]
        if result.get("ok") is not True:
            continue
        if san := result.get("san"):
            moves.add(san)
        moves.update(result.get("undone", ()))
        if played := result.get("engine_move"):
            moves.add(played["san"])
        checked = checked or result.get("check") is True
    settings = {
        "hints": "on" if ctx.settings.hints_mode else "off",
        "voice": "on" if ctx.settings.voice_output else "off",
        "verbosity": ctx.settings.verbosity,
    }
    if ctx.settings.tier is not None:
        # Only a named tier is a claimable difficulty. A session dialed in by
        # elo or skill level has no tier to be honest about, and mapping one
        # back would be the code inventing the fact instead of the model.
        settings["difficulty"] = ctx.settings.tier
    return VerifiedFacts(
        ended=ctx.session.is_game_over() or _destructive_succeeded(tool_results),
        drawn=outcome is not None and outcome.winner is None,
        check=checked,
        captured_by_player=frozenset(
            _PIECE_NAMES[symbol] for symbol in captured[ctx.session.player_color]
        ),
        captured_by_opponent=frozenset(
            _PIECE_NAMES[symbol] for symbol in captured[opponent]
        ),
        moves=frozenset(moves),
        saved=any(
            r["name"] in ("save_game", "resume_game") and r["result"].get("ok") is True
            for r in tool_results
        ),
        settings=settings,
        numbers=frozenset(_analysis_numbers(tool_results)),
        # Board truth, and the one fact here no tool has to have run for: who
        # is ahead is a piece count, so it is always available and always the
        # post-reply position, exactly like `check`.
        material=ctx.session.material_balance(),
    )


def _destructive_succeeded(tool_results: Sequence[dict[str, Any]]) -> bool:
    """Did a destructive op actually run this turn? A gate refusal is `ok:
    False`, so an armed-but-unconfirmed resign correctly counts as nothing
    happening."""
    return any(
        r["name"] in DESTRUCTIVE_TOOLS and r["result"].get("ok") is True
        for r in tool_results
    )


def _guarded_commentary(
    commentary: str, facts: VerifiedFacts, logged: dict[str, Any]
) -> tuple[str, bool]:
    """The commentary the turn's facts support, and whether a claim was cut.

    Every route converges here. The model may neither *do* an unasked
    destructive op (the gate) nor *say* it did (the ending class) nor announce
    any other fact the turn cannot back (the rest of them). Which class fired
    is logged, because that is the thing worth knowing when a guarded turn
    turns up in a trace.
    """
    claims = unverified_claims(commentary, facts)
    if not claims:
        return commentary, False
    if "ending" in claims:
        # The worst one keeps its own event name and its own correction: the
        # player has just been told the game ended, and the fact they need
        # back is that it didn't.
        logger.warning("commentary_claimed_untrue_outcome", extra=logged)
        return UNTRUE_CLAIM_REPLY, True
    logger.warning(
        "commentary_claimed_unverified_fact", extra={**logged, "claims": list(claims)}
    )
    return UNVERIFIED_CLAIM_REPLY, True


@dataclass(frozen=True)
class _ModelCost:
    """What one turn spent at the provider boundary, whichever route spent it.

    The five routes each pay for their own model calls — a narrated
    confirmation, a fast-path reaction, a resignation's words, the brain's whole
    loop, or nothing at all — and every one of them owes the trace the same four
    numbers. Reading them off the `AgentResponse`/`Narration` in one place is
    what keeps a route from quietly recording three of the four.
    """

    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latencies_ms: tuple[int, ...] = ()

    @classmethod
    def of(cls, source: Any | None) -> "_ModelCost":
        """The cost an `AgentResponse` or a `Narration` reports; `None` — a route
        that never called the model — costs nothing."""
        if source is None:
            return cls()
        return cls(
            calls=source.model_calls,
            prompt_tokens=source.prompt_tokens,
            completion_tokens=source.completion_tokens,
            latencies_ms=tuple(source.model_latencies_ms),
        )

    def as_trace(self) -> dict[str, Any]:
        """These numbers under the names `turn_record` takes them by."""
        return {
            "model_calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "model_latencies_ms": self.latencies_ms,
        }


@dataclass(frozen=True)
class CommandOutcome:
    """One command-pipeline run, shared by `/api/command` and the delegate
    messages endpoint. `tool_results` is the `{"name", "result"}` list of
    everything the agent ran, which `/api/command` returns verbatim;
    `tool_args` holds each call's arguments in the same order — the delegate
    endpoint needs them to build its wire `tool_calls`, but `/api/command`
    never exposes them. `stop_reason` is the brain loop's, in the fleet's
    vocabulary: `completed` when the agent finished with an answer,
    `max_iterations` or `correction_limit` when it ran out of budget first,
    `provider_error` when the provider died mid-turn (the results of whatever
    ran are still here, and the turn was still closed). The fast path is
    always `completed` — it never reaches the model."""

    commentary: str
    tool_results: list[dict[str, Any]]
    tool_args: list[dict[str, Any]]
    state: dict[str, Any]
    changed: bool
    stop_reason: str


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

    async def broadcast(self, state: dict[str, Any]) -> None:
        await self._send({"type": "state", "state": state})

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
    the brain's own two phases go unheard."""
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

    progress.bind(_publish_progress)
    coordinator.on_phase = progress.phase
    registry.on_tool = progress.tool
    store = ConversationStore()

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

    async def _broadcast_state() -> None:
        await broadcaster.broadcast(_state_dict(ctx))

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
        logger.info("stale_version expected=%s current=%s", exc.expected, exc.current)
        return JSONResponse(
            status_code=409,
            content={
                "detail": (
                    "the board changed since you last saw it — "
                    f"you sent version {exc.expected}, it is now {exc.current}"
                ),
                "stale": True,
                "version": exc.current,
                "state": _state_dict(ctx),
            },
        )

    @asynccontextmanager
    async def _mutation(expected: int | None) -> AsyncIterator[None]:
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
        await anyio.to_thread.run_sync(ctx.mutation_lock.acquire)
        try:
            if expected is not None and expected != ctx.board_version:
                raise StaleVersionError(expected, ctx.board_version)
            yield
        finally:
            ctx.mutation_lock.release()

    @app.get("/api/state")
    def get_state() -> dict[str, Any]:
        return _state_dict(ctx)

    @app.websocket("/ws")
    async def state_channel(websocket: WebSocket) -> None:
        await broadcaster.connect(websocket)
        await websocket.send_json({"type": "state", "state": _state_dict(ctx)})
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
        construction: verbosity=low skips it, and a `ProviderError` costs the
        words and nothing else, because the move it was about is already on the
        board and the engine's answer is not the model's to hold up.

        `correlation_id` is the caller's id for the interaction, carried only so
        the beat's own warning lands under it: a lost reaction is a thing you
        find in the log and then want the turn record for.
        """
        assert brain is not None  # both callers are agent-mode only
        result = registry.dispatch("make_move", {"move": move})
        changes = [{"name": "make_move", "result": result}]
        narration: Narration | None = None
        if result.get("legal") is True and ctx.settings.verbosity != "low":
            # This is the observe beat, so the machine is told so — the phase
            # the coordinator has always had a slot for, finally entered
            # (`docs/turn-coordinator.md`). Conditional because the move may
            # have ended the game, which closes the turn where it stands; the
            # collect below accepts either phase, so nothing else changes.
            coordinator.mark_observation()
            try:
                narration = brain.narrate(_agent_state_dict(ctx), changes, transcript)
            except ProviderError:
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
        if owed_reply:
            engine_reply = coordinator.collect_engine_reply()
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
        with progress.interaction(correlation_id, turn_id):
            beats = await _offloop(
                _play_move, move, ctx.transcript.memory(), correlation_id
            )
            result = beats.result
            narration = beats.narration
            if result.get("ok") is False:
                # A turn-state rejection: 409 on the trusted path, exactly as direct
                # mode answers it (the agent reads the same refusal as result data).
                # The drag played nothing, but the beats may have settled a turn that
                # was left open — that reply is on the board now, so every client
                # hears about it before the refusal goes back.
                if ctx.session.fen() != before["fen"]:
                    await _broadcast_state()
                raise HTTPException(status_code=409, detail=result["error"])
            commentary = ""
            guarded = False
            if beats.legal:
                commentary = _move_commentary(
                    narration.text if narration is not None else "",
                    result,
                    beats.engine_reply,
                    beats.owed_reply,
                    ctx.session,
                )
                # The honesty guard, on this road too: a reaction that announces
                # something the drag did not actually do is replaced with the truth.
                commentary, guarded = _guarded_commentary(
                    commentary,
                    _verified_facts(
                        ctx, beats.changes, beats.engine_reply, before["fen"]
                    ),
                    {"move": move},
                )
                ctx.transcript.record(result["san"], commentary)
                await _broadcast_state()
            _trace_turn(
                utterance=move,
                route=ROUTE_BOARD,
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
                guarded=guarded,
                **_ModelCost.of(narration).as_trace(),
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
                "state": _state_dict(ctx),
                "commentary": commentary,
                # Whether the client should voice it — the user's voice_output
                # setting, the same contract `/api/command` has.
                "speak": ctx.settings.voice_output,
            }

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
        """
        async with _mutation(request.version):
            if brain is not None:
                return await _agent_move(request.move)
            # The coordinator runs the exchange: player move, then the engine's
            # reply if one is owed. Trusted path, so a turn-state rejection is a
            # 409 rather than the error *result* the agent gets for the same thing.
            try:
                result, reply = await _offloop(coordinator.play_exchange, request.move)
            except TurnStateError as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from exc
            engine_move = _move_dict(reply) if reply is not None else None
            if result.legal:
                await _broadcast_state()
            return {
                "legal": result.legal,
                "san": result.san,
                "uci": result.uci,
                "reason": result.reason,
                "engine_move": engine_move,
                "state": _state_dict(ctx),
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
        async with _mutation(request.version if request is not None else None):
            outcome = _run_destructive("new_game", {"player_color": color})
            if isinstance(outcome, JSONResponse):
                return outcome
            await _broadcast_state()
            return {"state": _state_dict(ctx)}

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
        position, and that position is gone.
        """
        async with _mutation(request.version):
            armed = ctx.live_pending()
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
                    "state": _state_dict(ctx),
                }
            confirmed = confirm_pending(registry, ctx)
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
            await _broadcast_state()
            return {"op": name, "confirmed": True, "state": _state_dict(ctx)}

    @app.post("/api/game/undo")
    async def undo(request: UndoRequest) -> dict[str, Any]:
        async with _mutation(request.version):
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
            # Inside the guard, so a stale takeback costs the open turn nothing:
            # the position it was about is not the one this request meant.
            coordinator.abandon_turn()
            result = ctx.session.undo(plies)
            if not result.ok:
                raise HTTPException(status_code=409, detail=result.reason)
            await _broadcast_state()
            return {"undone": list(result.undone), "state": _state_dict(ctx)}

    @app.post("/api/game/resign")
    async def resign(request: ResignRequest) -> Any:
        """Resign — through the same gate as `new_game` and as a spoken "I
        resign" (409 + the question mid-game, `/api/game/confirm` answers it).

        Whose resignation an unqualified one is, is not a caller's judgment any
        more than it is the model's: the `resign` tool defaults it to the
        player's own side, and the side to move is only coincidentally that
        (trace review, finding 8).
        """
        async with _mutation(request.version):
            outcome = _run_destructive(
                "resign", {"color": request.color or ctx.session.player_color}
            )
            if isinstance(outcome, JSONResponse):
                return outcome
            await _broadcast_state()
            return {"outcome": outcome["outcome"], "state": _state_dict(ctx)}

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
        return {
            "tier": ctx.settings.tier,
            "skill_level": ctx.settings.skill_level,
            "elo": ctx.settings.elo,
        }

    def _trace_turn(**fields: Any) -> None:
        """Record the finished turn, and never let that cost the player one.

        A tracer is a diagnostic sink — a file that may be full, unwritable, or
        on a disk that just went away. None of that is the game's problem, so a
        failure here is logged and dropped rather than turned into a 500 on a
        turn whose moves have already been played.
        """
        if tracer is None:
            return
        try:
            tracer.record(turn_record(**fields))
        except Exception:
            logger.warning("trace_failed", exc_info=True)

    @contextmanager
    def _command_window(correlation_id: str, turn_id: int) -> Iterator[None]:
        """One user interaction, opened and closed at both ends at once.

        The coordinator's destructive budget and the progress stream's brackets
        are the same interaction seen from two sides — what may happen inside
        it, and what the player is told is happening inside it — so they are
        opened together and, more to the point, closed together in the same
        `finally`. A command that raises half-way must neither leak an open
        window into the next one nor leave a progress line spinning.
        """
        assert progress is not None  # bound above; narrows for the type checker
        with progress.interaction(correlation_id, turn_id):
            coordinator.begin_command()
            try:
                yield
            finally:
                coordinator.end_command()

    async def _run_command(
        text: str,
        transcript: Sequence[dict[str, str]],
        version: int | None = None,
    ) -> CommandOutcome:
        """The command pipeline, under the mutation guard.

        A command is a mutation path like any other — one utterance can move the
        board — so it carries the same optional `version` precondition, and a
        stale one is refused before the model is asked anything. The guard wraps
        the *whole* run rather than the individual dispatches inside it: a turn
        is one thing that happens to one board, and half of it landing on a
        board someone else changed is exactly the race this closes.
        """
        async with _mutation(version):
            return await _command_turn(text, transcript)

    async def _command_turn(
        text: str, transcript: Sequence[dict[str, str]]
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
        """
        assert brain is not None  # both callers guard; documents the invariant
        before = _agent_state_dict(ctx)
        # What locates this turn afterwards (audit item 18): the coordinator turn
        # it opened under, an id for this one interaction, and the board version
        # it started from — the mutation count is that version's delta, so the
        # number is *derived* from the chokepoint every mutation already passes
        # through rather than tallied by whichever branch remembered to.
        turn_id = coordinator.turn_id
        correlation_id = new_correlation_id()
        version_before = ctx.board_version
        # The hints setting that governs this turn is the one the player asked
        # under, snapshotted here for the advice guard below. App assembly
        # resolves the tool *offer* off the same instant, and the two must not
        # disagree: live, the planner answered "what should I play here?" by
        # calling `set_hints_mode(True)` and then naming moves — the model
        # granting itself the permission the player had switched off. The
        # change stands; it takes effect from the next turn.
        hints_at_ask = ctx.settings.hints_mode
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
            stop_reason = "completed"
            # Named only when the brain's loop died on the provider; every other
            # route leaves it empty, which is the record's way of saying "did not
            # die" rather than "not recorded".
            provider_failure = ""
            # The fast path's move beats, held for the commentary below: with no
            # narration to speak for the turn (verbosity=low, or a provider failure)
            # the move and the engine's reply become one canned confirmation. None on
            # every other route — those close their own turn further down.
            move_beats: _MoveBeats | None = None
            # The turn's cost at the provider boundary, summed across whatever model
            # calls the chosen route made. The deterministic branches (a canned
            # confirmation, a declined op) leave this at zero — a real, readable
            # zero, which is what tells a later cut it changed nothing here.
            cost = _ModelCost()
            # An armed destructive op (the tool gate refused new_game/resign last
            # turn and asked). This turn is its answer — and the answer is ours, not
            # the model's: a bare yes runs it with the gate open, a bare no drops it,
            # and anything else is a new intent that disarms it on the way past. The
            # op never survives the turn, so a stale "yes" can never revive it.
            # Read through `live_pending`: a question is about a position, so an
            # op armed against a board that has since moved — the player dragged
            # a move, undid one, another client played — is not something this
            # "yes" can be an answer to, and is dropped instead of run.
            armed = ctx.live_pending()
            ctx.pending = None
            answer = parse_confirmation(text) if armed is not None else None
            route = ROUTE_CONFIRMATION
            if armed is not None and answer is not None:
                if answer:
                    ctx.pending = armed  # confirm_pending consumes it
                    confirmed = await _offloop(confirm_pending, registry, ctx)
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
                        try:
                            narration = await _offloop(
                                brain.narrate,
                                _agent_state_dict(ctx),
                                tool_results,
                                transcript,
                            )
                        except ProviderError:
                            logger.warning("close_narration_failed", exc_info=True)
                            commentary = _destructive_confirmation(
                                name, result, ctx.session
                            )
                        else:
                            commentary = narration.text
                            cost = _ModelCost.of(narration)
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
                if not move_beats.legal:
                    # `parse_move` already matched the move against this board, so a
                    # refusal here is a turn-state rejection (a previous turn left
                    # the machine mid-sequence), not an illegal move: nothing moved,
                    # so there is nothing to react to. The beats still settled
                    # whatever that turn left owing.
                    commentary = _STUCK_REPLY
                elif move_beats.narration is not None:
                    commentary = move_beats.narration.text
                    cost = _ModelCost.of(move_beats.narration)
            elif parse_resign(text):
                # An explicit resignation is deterministic text, so the model gets no
                # vote on whether it happened: live, it took one and answered "Word.
                # Game over." with no tool call on a live board. The call still goes
                # through the registry, so the gate arms it and the player's yes —
                # not the agent's word — is what ends the game.
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
                    try:
                        narration = await _offloop(
                            brain.narrate,
                            _agent_state_dict(ctx),
                            tool_results,
                            transcript,
                        )
                    except ProviderError:
                        logger.warning("close_narration_failed", exc_info=True)
                        commentary = _destructive_confirmation(
                            "resign", result, ctx.session
                        )
                    else:
                        commentary = narration.text
                        cost = _ModelCost.of(narration)
            else:
                route = ROUTE_BRAIN
                response = await _offloop(
                    brain.get_agent_response, before, text, transcript
                )
                tool_results = list(response.tool_results)
                tool_args = [call.args for call in response.tool_calls]
                stop_reason = response.stop_reason
                provider_failure = response.provider_failure
                cost = _ModelCost.of(response)
                # A budget stop (max_iterations / correction_limit) carries no
                # commentary: the loop never reached a text turn. A provider
                # stop is left empty here — what it should say depends on
                # whether anything changed, which the close beat below settles.
                commentary = response.text
                if not commentary and stop_reason != "provider_error":
                    commentary = _STUCK_REPLY
            # The close beat, at the one point every route converges. A coordinator
            # left mid-sequence means the player's move landed without its reply —
            # whichever route played it — and whatever narration that route produced
            # was this turn's reaction to it. So: collect the answer the engine has
            # been computing all along, close the turn, and announce the reply in the
            # app's own words. `complete_turn` is deliberately the pipeline's and not
            # the tool's: nothing may close a turn the engine still owes a move to.
            engine_reply: MoveResult | None = None
            owed_reply = False
            if move_beats is not None:
                # The fast path ran the beats already, close included; what they
                # settled is what this turn has to say for itself.
                engine_reply, owed_reply = (
                    move_beats.engine_reply,
                    move_beats.owed_reply,
                )
            elif coordinator.phase in (
                TurnPhase.PLAYER_MOVE_APPLIED,
                TurnPhase.AGENT_OBSERVING,
            ):
                owed_reply = True
                engine_reply = await _offloop(coordinator.collect_engine_reply)
                coordinator.complete_turn()
            if move_beats is not None and move_beats.legal:
                commentary = _move_commentary(
                    commentary, move_beats.result, engine_reply, owed_reply, ctx.session
                )
            elif owed_reply and (
                reply_line := _reply_announcement(engine_reply, ctx.session)
            ):
                commentary = (
                    f"{commentary}\n\n{reply_line}" if commentary else reply_line
                )
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
            # The honesty guard, at the one point every route converges: an
            # operational claim the turn cannot back is not shown to the player.
            # The board, the engine's reply and the tool results are the record of
            # what happened; the model's prose is not, and live it has claimed
            # resignations and checkmates that never occurred (trace review,
            # finding 6). This is the same rule as the gate, applied one step
            # later — the model may neither *do* a destructive op unasked nor
            # *say* it did, nor announce any other fact it invented.
            commentary, guarded = _guarded_commentary(
                commentary,
                _verified_facts(ctx, tool_results, engine_reply, before["fen"]),
                {"text": text},
            )
            agent_state = _agent_state_dict(ctx)
            # The advice guard, same convergence point (audit item 11's second
            # half): with hints off, a currently-playable move in the
            # commentary is a hint whatever prose carries it. It only fires on
            # a turn that changed nothing — reacting to a move just played is
            # description, not advice — and never over the moves an analysis
            # the player asked for actually reported, which are verified facts.
            # Those moves, and only those, come off the list the guard checks.
            unlicensed = set(ctx.session.legal_moves()) - _reported_moves(tool_results)
            if (
                not guarded
                and not hints_at_ask
                and agent_state == before
                and names_a_legal_move(commentary, unlicensed)
            ):
                logger.warning(
                    "commentary_leaked_move_advice",
                    extra={"text": text, "correlation_id": correlation_id},
                )
                commentary = MOVE_ADVICE_REPLY
                guarded = True
            # If this turn armed a destructive op, the question goes to the
            # player about the board they can see *now* — this turn's mutations
            # included, since the gate arms mid-turn and the engine's reply can
            # land after it ("play e4 and start over"). Their yes next turn is
            # an answer to that board and no other.
            ctx.restamp_pending()
            # The UI still gets its own full document; a mutation shows up in the
            # agent view too (any board change moves the fen), so that comparison
            # decides the broadcast.
            state = _state_dict(ctx)
            changed = agent_state != before
            if changed:
                await broadcaster.broadcast(state)
            _trace_turn(
                utterance=text,
                route=route,
                commentary=commentary,
                stop_reason=stop_reason,
                changed=changed,
                turn_id=turn_id,
                correlation_id=correlation_id,
                mutations=ctx.board_version - version_before,
                fen_before=before["fen"],
                fen_after=agent_state["fen"],
                tool_calls=tool_args,
                tool_results=tool_results,
                engine_reply=_move_reply_dict(engine_reply),
                guarded=guarded,
                provider_failure=provider_failure,
                **cost.as_trace(),
            )
            return CommandOutcome(
                commentary=commentary,
                tool_results=tool_results,
                tool_args=tool_args,
                state=state,
                changed=changed,
                stop_reason=stop_reason,
            )

    @app.post("/api/command")
    async def command(request: CommandRequest) -> dict[str, Any]:
        """User string → brain → tool call(s) → new state, for the web panel.
        A thin wrapper over `_run_command` that supplies the panel's own
        conversation memory and records the settled turn back onto it. Memory,
        not the raw window: recent turns verbatim behind a digest of what the
        player asked for earlier (`docs/turn-memory.md`)."""
        if brain is None:
            raise HTTPException(status_code=503, detail="agent unavailable: no brain")
        transcript = ctx.transcript.memory()
        outcome = await _run_command(request.text, transcript, request.version)
        # Record on the context, not a captured reference: resume_game may
        # have just swapped in the saved game's transcript, and this turn
        # belongs to that thread.
        ctx.transcript.record(request.text, outcome.commentary)
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
            "hints_mode": s.hints_mode,
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

    @app.post("/api/voice/transcribe")
    async def transcribe(audio: UploadFile) -> dict[str, Any]:
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
        try:
            text = speech.transcribe(data, filename=audio.filename or "audio.webm")
        except Exception as exc:
            raise HTTPException(
                status_code=502, detail=f"speech service error: {exc}"
            ) from exc
        return {"text": text}

    @app.post("/api/voice/speak")
    def speak_text(request: SpeakRequest) -> Response:
        """TTS proxy: text in, mp3 out — the audio for whatever commentary
        the client decided to voice. Same optionality contract as
        /api/voice/transcribe: 503 without a speech service, 502 when the
        upstream fails; board state is never touched."""
        if speech is None:
            raise HTTPException(
                status_code=503, detail="voice unavailable: no speech service"
            )
        try:
            audio = speech.speak(request.text)
        except Exception as exc:
            raise HTTPException(
                status_code=502, detail=f"speech service error: {exc}"
            ) from exc
        return Response(content=audio, media_type="audio/mpeg")

    @app.get("/api/game/hint")
    def get_hint() -> dict[str, Any]:
        """Best move for the side to move, for the UI's hint arrow (trusted
        path, same engine the `suggest_moves` analysis uses). Needs an engine
        (503 without one); a finished or move-less position is a domain
        failure (409). Read-only — nothing is broadcast."""
        if ctx.engine is None:
            raise HTTPException(status_code=503, detail="hint unavailable: no engine")
        if ctx.session.is_game_over():
            raise HTTPException(
                status_code=409, detail="cannot suggest moves: game is over"
            )
        try:
            candidates = ctx.engine.get_best_moves(ctx.session, n=1)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        if not candidates:
            raise HTTPException(status_code=409, detail="no candidate moves")
        best = candidates[0]
        # uci[2:4] is the destination even for 5-char promotion UCIs.
        return {
            "uci": best.uci,
            "san": best.san,
            "from": best.uci[:2],
            "to": best.uci[2:4],
        }

    @app.get("/api/game/review")
    def get_review() -> dict[str, Any]:
        """Whole-game review for the UI (trusted path, same numbers as the
        `review_game` tool). Analysis needs Stockfish (503 without one);
        reviewing an empty game is a domain failure (409). Read-only —
        nothing is broadcast."""
        if ctx.engine is None:
            raise HTTPException(status_code=503, detail="review unavailable: no engine")
        try:
            review = review_game(ctx.engine, ctx.session)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
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
        return {"pgn": ctx.session.export_pgn()}

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
