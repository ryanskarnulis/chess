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

from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from chessapp.brain import Brain
from chessapp.game import GameSession, MoveResult
from chessapp.tools import UNDO_PLIES_MAX, ToolContext, build_registry


class MoveRequest(BaseModel):
    move: str


class CommandRequest(BaseModel):
    text: str


class UndoRequest(BaseModel):
    plies: int = Field(default=1, ge=1, le=UNDO_PLIES_MAX)


class ResignRequest(BaseModel):
    color: str | None = Field(default=None, pattern="^(white|black)$")


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


def create_app(ctx: ToolContext, brain: Brain | None = None) -> FastAPI:
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
        """
        if brain is None:
            raise HTTPException(status_code=503, detail="agent unavailable: no brain")
        before = _state_dict(ctx.session)
        response = brain.get_agent_response(before, request.text)
        tool_results = [
            {"name": call.name, "result": registry.dispatch(call.name, call.args)}
            for call in response.tool_calls
        ]
        state = _state_dict(ctx.session)
        if tool_results:
            commentary = brain.react(state, tool_results)
        else:
            commentary = response.text
        if state != before:
            await broadcaster.broadcast(state)
        return {
            "commentary": commentary,
            "tool_results": tool_results,
            "state": state,
        }

    @app.get("/api/game/pgn")
    def export_pgn() -> dict[str, Any]:
        return {"pgn": ctx.session.export_pgn()}

    return app
