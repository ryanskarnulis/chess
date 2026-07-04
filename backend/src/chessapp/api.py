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

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from chessapp.game import GameSession, MoveResult
from chessapp.tools import UNDO_PLIES_MAX, ToolContext


class MoveRequest(BaseModel):
    move: str


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


def create_app(ctx: ToolContext) -> FastAPI:
    app = FastAPI(title="chessapp")

    @app.get("/api/state")
    def get_state() -> dict[str, Any]:
        return _state_dict(ctx.session)

    @app.post("/api/game/move")
    def submit_move(request: MoveRequest) -> dict[str, Any]:
        result = ctx.session.submit_move(request.move)
        engine_move: dict[str, Any] | None = None
        if result.legal and ctx.engine is not None and not ctx.session.is_game_over():
            engine_move = _move_dict(ctx.engine.play_move(ctx.session))
        return {
            "legal": result.legal,
            "san": result.san,
            "uci": result.uci,
            "reason": result.reason,
            "engine_move": engine_move,
            "state": _state_dict(ctx.session),
        }

    @app.post("/api/game/new")
    def new_game() -> dict[str, Any]:
        ctx.session.new_game()
        return {"state": _state_dict(ctx.session)}

    @app.post("/api/game/undo")
    def undo(request: UndoRequest) -> dict[str, Any]:
        result = ctx.session.undo(request.plies)
        if not result.ok:
            raise HTTPException(status_code=409, detail=result.reason)
        return {"undone": list(result.undone), "state": _state_dict(ctx.session)}

    @app.post("/api/game/resign")
    def resign(request: ResignRequest) -> dict[str, Any]:
        try:
            outcome = ctx.session.resign(request.color)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {
            "outcome": {
                "termination": outcome.termination,
                "winner": outcome.winner,
                "result": outcome.result,
            },
            "state": _state_dict(ctx.session),
        }

    @app.get("/api/game/pgn")
    def export_pgn() -> dict[str, Any]:
        return {"pgn": ctx.session.export_pgn()}

    return app
