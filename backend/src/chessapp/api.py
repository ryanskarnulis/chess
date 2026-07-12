"""HTTP API: game lifecycle + state fetch for the board UI.

The API is trusted code, so it drives `GameSession` directly through the
shared `ToolContext`; the tool registry remains the LLM-only boundary.
Conventions:

- Illegal moves are data (`legal: false`), not HTTP errors — legality is
  the engine's answer, not a transport failure.
- Domain failures on mutations (nothing to undo, resigning a finished
  game) are 409s.
- When the context has an engine and the player's move leaves the game
  running, the engine replies in the same request — that's the LLM-off
  vs-Stockfish mode the app must always support.

Always read `ctx.session` per request: `resume_game` swaps the session
object on the context.
"""

import mimetypes
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, model_validator

from chessapp.agent_api import ConversationStore, build_agent_router
from chessapp.analysis import review_game
from chessapp.brain import Brain
from chessapp.engine import validate_elo, validate_skill_level, validate_tier
from chessapp.fastparse import parse_move
from chessapp.game import GameSession, MoveResult
from chessapp.tools import UNDO_PLIES_MAX, ToolContext, build_registry
from chessapp.voice import SpeechClient


class MoveRequest(BaseModel):
    move: str


class CommandRequest(BaseModel):
    text: str


class SpeakRequest(BaseModel):
    text: str = Field(min_length=1, pattern=r"\S")


class VoiceOutputRequest(BaseModel):
    enabled: bool


class UndoRequest(BaseModel):
    plies: int = Field(default=1, ge=1, le=UNDO_PLIES_MAX)


class ResignRequest(BaseModel):
    color: str | None = Field(default=None, pattern="^(white|black)$")


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


def _state_dict(session: GameSession) -> dict[str, Any]:
    """The full state document the board UI renders from."""
    return {
        "fen": session.fen(),
        "turn": session.turn,
        "game_over": session.is_game_over(),
        "outcome": _outcome_dict(session),
        "history": session.move_history(),
        "fens": session.position_fens(),
        "captured": session.captured_pieces(),
        "legal_moves": session.legal_moves(),
        "dests": session.legal_destinations(),
    }


def _agent_state_dict(session: GameSession, player_color: str) -> dict[str, Any]:
    """The view the brain reasons from: board truth (fen, turn, check, SAN
    history, captures, legal moves, outcome) plus which color the player is.
    Deliberately not `_state_dict` — the UI document's per-ply `fens` and
    `dests` are prompt noise that grows every move and never helps the agent.
    """
    return {
        "fen": session.fen(),
        "turn": session.turn,
        "player_color": player_color,
        "in_check": session.is_check(),
        "game_over": session.is_game_over(),
        "outcome": _outcome_dict(session),
        "history": session.move_history(),
        "captured": session.captured_pieces(),
        "legal_moves": session.legal_moves(),
    }


def _move_dict(result: MoveResult) -> dict[str, Any]:
    return {"legal": result.legal, "san": result.san, "uci": result.uci}


def _move_confirmation(result: dict[str, Any], session: GameSession) -> str:
    """Deterministic stand-in for the reaction on the fast path at
    verbosity=low: the move, the engine's reply, and the outcome if the game
    ended — facts from the tool result, zero LLM calls."""
    parts = [f"{result['san']}."]
    engine_move = result.get("engine_move")
    if engine_move:
        parts.append(f"{engine_move['san']}.")
    if result.get("game_over"):
        outcome = _outcome_dict(session)
        if outcome:
            parts.append(f"Game over: {outcome['result']} ({outcome['termination']}).")
    return " ".join(parts)


# How many self-correction rounds a turn's failed tool calls get before the
# pipeline gives up and reacts to the failures as they stand. Generalizes the
# brain-internal schema-retry idea to domain errors, which only surface here —
# after dispatch — so it lives in the pipeline and benefits any Brain.
MAX_TOOL_RETRY_ROUNDS = 2


def _failed_calls(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The tool results worth a self-correction round: dispatch/domain errors
    (`ok: false`) and rejected moves (`legal: false`)."""
    return [
        r
        for r in results
        if r["result"].get("ok") is False or r["result"].get("legal") is False
    ]


def _retry_command(failures: list[dict[str, Any]], session: GameSession) -> str:
    """The correction fed back to the brain: what failed and why, plus — when
    a move was rejected — the legal moves to pick from. Plumbing, not
    conversation: it is never recorded in the transcript."""
    lines = []
    move_rejected = False
    for failure in failures:
        result = failure["result"]
        if result.get("legal") is False:
            move_rejected = True
            reason = result.get("reason") or "illegal move"
            lines.append(f"{failure['name']}: rejected — {reason}")
        else:
            lines.append(f"{failure['name']}: {result.get('error', 'failed')}")
    command = (
        "Your tool call(s) failed: " + "; ".join(lines) + ". "
        "Fix the problem and call the tool again, or answer the player "
        "in words if you cannot."
    )
    if move_rejected:
        command += " Legal moves: " + ", ".join(session.legal_moves())
    return command


@dataclass(frozen=True)
class CommandOutcome:
    """One command-pipeline run, shared by `/api/command` and the delegate
    messages endpoint. `tool_results` is the `{"name", "result"}` list the
    reaction saw and `/api/command` returns verbatim; `tool_args` holds each
    dispatched call's arguments in the same order — the delegate endpoint needs
    them to build its wire `tool_calls`, but `/api/command` never exposes them.
    `stop_reason` is `completed`, or `correction_limit` when the domain-retry
    loop exhausted its rounds with failures still unresolved (chess never emits
    `max_iterations` — the two-phase brain has no iteration loop)."""

    commentary: str
    tool_results: list[dict[str, Any]]
    tool_args: list[dict[str, Any]]
    state: dict[str, Any]
    changed: bool
    stop_reason: str


class StateBroadcaster:
    """Fans the state document out to every connected board UI.

    Send failures mean the client went away; the socket is dropped, never
    allowed to fail the mutation that triggered the broadcast.
    """

    def __init__(self) -> None:
        self._clients: set[WebSocket] = set()

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self._clients.add(websocket)

    def disconnect(self, websocket: WebSocket) -> None:
        self._clients.discard(websocket)

    async def broadcast(self, state: dict[str, Any]) -> None:
        message = {"type": "state", "state": state}
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
) -> FastAPI:
    app = FastAPI(title="chessapp")
    broadcaster = StateBroadcaster()
    registry = build_registry(ctx)
    store = ConversationStore()

    async def _broadcast_state() -> None:
        await broadcaster.broadcast(_state_dict(ctx.session))

    @app.get("/api/state")
    def get_state() -> dict[str, Any]:
        return _state_dict(ctx.session)

    @app.websocket("/ws")
    async def state_channel(websocket: WebSocket) -> None:
        await broadcaster.connect(websocket)
        await websocket.send_json({"type": "state", "state": _state_dict(ctx.session)})
        try:
            # The channel is one-way; we only read to notice the disconnect.
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            broadcaster.disconnect(websocket)

    @app.post("/api/game/move")
    async def submit_move(request: MoveRequest) -> dict[str, Any]:
        result = ctx.session.submit_move(request.move)
        engine_move: dict[str, Any] | None = None
        if result.legal and ctx.engine is not None and not ctx.session.is_game_over():
            engine_move = _move_dict(ctx.engine.play_move(ctx.session))
        if result.legal:
            await _broadcast_state()
        return {
            "legal": result.legal,
            "san": result.san,
            "uci": result.uci,
            "reason": result.reason,
            "engine_move": engine_move,
            "state": _state_dict(ctx.session),
        }

    @app.post("/api/game/new")
    async def new_game() -> dict[str, Any]:
        ctx.session.new_game()
        await _broadcast_state()
        return {"state": _state_dict(ctx.session)}

    @app.post("/api/game/undo")
    async def undo(request: UndoRequest) -> dict[str, Any]:
        result = ctx.session.undo(request.plies)
        if not result.ok:
            raise HTTPException(status_code=409, detail=result.reason)
        await _broadcast_state()
        return {"undone": list(result.undone), "state": _state_dict(ctx.session)}

    @app.post("/api/game/resign")
    async def resign(request: ResignRequest) -> dict[str, Any]:
        try:
            outcome = ctx.session.resign(request.color)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        await _broadcast_state()
        return {
            "outcome": {
                "termination": outcome.termination,
                "winner": outcome.winner,
                "result": outcome.result,
            },
            "state": _state_dict(ctx.session),
        }

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

    async def _run_command(
        text: str, transcript: Sequence[dict[str, str]]
    ) -> CommandOutcome:
        """The single pipeline: user string → brain → tool call(s) → engine
        executes → agent reacts from the *new* state. Shared by `/api/command`
        and the delegate messages endpoint against the one game session.

        Phase one (`get_agent_response`) turns the utterance into tool calls;
        the validated registry runs them, so brain mistakes (unknown tools,
        bad args, domain errors) come back as error results in `tool_results`,
        never as HTTP failures or corrupted state. Failed calls get a bounded
        self-correction loop: the error (plus legal moves, for rejected
        moves) goes back to the brain against the fresh state, up to
        MAX_TOOL_RETRY_ROUNDS times — so one illegal-move guess doesn't end
        the turn. Phase two (`react`) reads the resulting state and what
        changed — not the raw utterance — and produces the commentary. When
        nothing was done (a question or a clarifying reply) there is nothing
        to react to, so the direct answer stands; a retry round that answers
        in words instead of tools is treated the same way.

        Both phases see the `transcript` the caller supplies (prior turns'
        commands plus the commentary that was shown, a bounded window, final
        answers only), so the agent can follow references to earlier turns —
        but this is transcript-agnostic: it never reads or records either the
        web panel's `ctx.transcript` or a delegate conversation, leaving that
        to the caller. The board broadcast fires here on any board change, so
        a conductor-played move shows up live on the web board too.

        The fast path (the seam BRIEF reserves): an utterance that is exactly
        one unambiguous legal move skips phase one and goes straight to
        make_move — through the same registry, so the road stays one road,
        minus a model round-trip. The reaction still runs as usual; at
        verbosity=low a canned confirmation stands in for it too, making a
        plain move a zero-LLM turn. Anything ambiguous or non-move reaches
        the brain unchanged. No retry loop here: the parse matched a
        currently legal move, so the dispatch cannot be rejected.
        """
        assert brain is not None  # both callers guard; documents the invariant
        # The player is whoever's turn it is when the command arrives (the
        # engine replies inside make_move, so it is never the engine's turn
        # here). Captured once so react's view still names the right color
        # when the player's own move flips the turn or ends the game.
        player_color = ctx.session.turn
        before = _agent_state_dict(ctx.session, player_color)
        direct_reply: str | None = None
        pre_action_text = ""
        # `tool_results` is the {"name", "result"} list react and the UI see;
        # `tool_args` mirrors it with each call's arguments, for the delegate
        # wire — kept parallel so the UI-facing shape stays untouched.
        tool_results: list[dict[str, Any]] = []
        tool_args: list[dict[str, Any]] = []
        failures: list[dict[str, Any]] = []
        fast_san = parse_move(text, ctx.session.fen())
        if fast_san is not None:
            args = {"move": fast_san}
            tool_results.append(
                {"name": "make_move", "result": registry.dispatch("make_move", args)}
            )
            tool_args.append(args)
        else:
            response = brain.get_agent_response(before, text, transcript)
            pre_action_text = response.text
            for call in response.tool_calls:
                tool_results.append(
                    {
                        "name": call.name,
                        "result": registry.dispatch(call.name, call.args),
                    }
                )
                tool_args.append(call.args)
            failures = _failed_calls(tool_results)
            for _ in range(MAX_TOOL_RETRY_ROUNDS):
                if not failures:
                    break
                retry = brain.get_agent_response(
                    _agent_state_dict(ctx.session, player_color),
                    _retry_command(failures, ctx.session),
                    transcript,
                )
                if not retry.tool_calls:
                    # The brain answered in words instead of retrying — a
                    # clarifying question or a concession. That is the final,
                    # user-facing reply; reacting to the failure would bury it.
                    direct_reply = retry.text
                    break
                round_results = []
                for call in retry.tool_calls:
                    result = registry.dispatch(call.name, call.args)
                    round_results.append({"name": call.name, "result": result})
                    tool_results.append({"name": call.name, "result": result})
                    tool_args.append(call.args)
                failures = _failed_calls(round_results)
        agent_state = _agent_state_dict(ctx.session, player_color)
        if direct_reply is not None:
            commentary = direct_reply
        elif fast_san is not None and ctx.settings.verbosity == "low":
            commentary = _move_confirmation(tool_results[0]["result"], ctx.session)
        elif tool_results:
            commentary = brain.react(agent_state, tool_results, transcript)
        else:
            commentary = pre_action_text
        # Failures still unresolved after the retry budget → correction_limit;
        # a wordy concession (direct_reply) is a normal completion.
        stop_reason = (
            "correction_limit" if direct_reply is None and failures else "completed"
        )
        # The UI still gets its own full document; a mutation shows up in the
        # agent view too (any board change moves the fen), so that comparison
        # decides the broadcast.
        state = _state_dict(ctx.session)
        changed = agent_state != before
        if changed:
            await broadcaster.broadcast(state)
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
        transcript window and records the settled turn back onto it."""
        if brain is None:
            raise HTTPException(status_code=503, detail="agent unavailable: no brain")
        transcript = ctx.transcript.window()
        outcome = await _run_command(request.text, transcript)
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
        from (the same truth the tools mutate)."""
        s = ctx.settings
        return {
            "verbosity": s.verbosity,
            "hints_mode": s.hints_mode,
            "voice_output": s.voice_output,
            "tier": s.tier,
            "skill_level": s.skill_level,
            "elo": s.elo,
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
