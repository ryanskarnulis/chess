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

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, model_validator

from chessapp.analysis import review_game
from chessapp.brain import Brain
from chessapp.engine import validate_elo, validate_skill_level, validate_tier
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

    @app.post("/api/command")
    async def command(request: CommandRequest) -> dict[str, Any]:
        """The single pipeline: user string → brain → tool call(s) → engine
        executes → agent reacts from the *new* state.

        Phase one (`get_agent_response`) turns the utterance into tool calls;
        the validated registry runs them, so brain mistakes (unknown tools,
        bad args, domain errors) come back as error results in `tool_results`,
        never as HTTP failures or corrupted state. Phase two (`react`) reads
        the resulting state and what changed — not the raw utterance — and
        produces the commentary. When nothing was done (a question or a
        clarifying reply) there is nothing to react to, so the direct answer
        stands.

        Both phases see the transcript — prior turns' commands plus the
        commentary the user actually saw (a bounded window, final answers
        only) — and the turn is recorded once its commentary is settled, so
        the agent can follow references to earlier conversation.
        """
        if brain is None:
            raise HTTPException(status_code=503, detail="agent unavailable: no brain")
        # The player is whoever's turn it is when the command arrives (the
        # engine replies inside make_move, so it is never the engine's turn
        # here). Captured once so react's view still names the right color
        # when the player's own move flips the turn or ends the game.
        player_color = ctx.session.turn
        before = _agent_state_dict(ctx.session, player_color)
        transcript = ctx.transcript.window()
        response = brain.get_agent_response(before, request.text, transcript)
        tool_results = [
            {"name": call.name, "result": registry.dispatch(call.name, call.args)}
            for call in response.tool_calls
        ]
        agent_state = _agent_state_dict(ctx.session, player_color)
        if tool_results:
            commentary = brain.react(agent_state, tool_results, transcript)
        else:
            commentary = response.text
        # Record on the context, not a captured reference: resume_game may
        # have just swapped in the saved game's transcript, and this turn
        # belongs to that thread.
        ctx.transcript.record(request.text, commentary)
        # The UI still gets its own full document; a mutation shows up in the
        # agent view too (any board change moves the fen), so that comparison
        # decides the broadcast.
        state = _state_dict(ctx.session)
        if agent_state != before:
            await broadcaster.broadcast(state)
        return {
            "commentary": commentary,
            "tool_results": tool_results,
            "state": state,
            # Whether the client should voice the commentary (the user's
            # voice_output setting, agent-togglable via set_voice_output).
            # The server owns the decision; the client owns the playback.
            "speak": ctx.settings.voice_output,
        }

    @app.get("/api/settings")
    def get_settings() -> dict[str, Any]:
        """The agent-adjustable settings, for the UI to render its controls
        from (the same truth the tools mutate)."""
        s = ctx.settings
        return {
            "personality": s.personality,
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
        app.mount("/", StaticFiles(directory=static_dir, html=True), name="frontend")

    return app
