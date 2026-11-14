"""The glance: one small, read-only summary of the app (#536).

`GET /api/agent/glance` answers conductor's `look(chess)` ("how did my last
game go?") without opening the board, per the workspace contract
(`../agent-standard/app-yaml-glance-block.md`): no arguments, no model call,
no engine search, no side effect, about 600 characters.

Everything here is read off state the app already holds: the board (and when
its last move was played), the ledger-fed results log (`results.py`). **No
evaluation is computed.** Chess stores none for a live game (the engine is
asked on demand, and a search cannot promise the 200 ms budget), so the glance
says what the board itself shows, the material, and leaves an evaluation to an
`open`.
"""

import logging
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, HTTPException, status

from chessapp import openings
from chessapp.tools import ToolContext

logger = logging.getLogger(__name__)

APP = "chess"
KIND = "game"

_RESULT_WORDS = {"player": "you won", "opponent": "you lost", None: "drawn"}


def _material_words(pawns: int) -> str:
    if pawns == 0:
        return "material is level"
    unit = "pawn" if abs(pawns) == 1 else "pawns"
    side = "up" if pawns > 0 else "down"
    return f"you are {side} {'a' if abs(pawns) == 1 else abs(pawns)} {unit}"


def _current(ctx: ToolContext) -> dict[str, Any] | None:
    """The game on the board, or None when there is none to speak of: nothing
    played yet, or it already ended (then it is the last finished game)."""
    session = ctx.session
    if session.plies == 0 or session.is_game_over():
        return None
    line = openings.opening_of(session)
    ref: dict[str, Any] = {
        "app": APP,
        "kind": KIND,
        "id": session.game_id,
        "title": "Game in progress",
        "to_move": "you" if session.turn == session.player_color else "Glitch",
        "moves": session.fullmove_number,
        "material": _material_words(session.material_balance()),
        # When the last move was played (#547), so a reader can say how old
        # the game is: UTC ISO 8601, or null for a game saved before moves
        # were timed. Never a guess such as the start date.
        "last_move_at": session.last_move_at,
    }
    if line:
        ref["opening"] = line["name"]
    return ref


def _last_finished(ctx: ToolContext) -> dict[str, Any] | None:
    entry = ctx.results.latest()
    if entry is None:
        return None
    result = _RESULT_WORDS[entry["winner"]]
    ref: dict[str, Any] = {
        "app": APP,
        "kind": KIND,
        "id": entry["game_id"],
        "title": f"Game {result} ({entry['difficulty']})",
        "result": result,
        "score": entry["result"],
        "termination": entry["termination"],
        "date": str(entry["ended_at"])[:10],
    }
    if entry.get("opening"):
        ref["opening"] = entry["opening"]
    return ref


def build_glance(ctx: ToolContext) -> dict[str, Any]:
    return {
        "as_of": datetime.now(UTC).isoformat(timespec="seconds"),
        "current": _current(ctx),
        "last_finished": _last_finished(ctx),
    }


def build_glance_router(glance: Callable[[], dict[str, Any]]) -> APIRouter:
    """`GET /api/agent/glance`, the one route left under `/api/agent` (#565).

    The glance contract (`../agent-standard/app-yaml-glance-block.md`): no
    arguments, no model call, no side effects. Any failure to read the app's
    own state is a 503, which conductor treats as "couldn't look"."""
    router = APIRouter(prefix="/api/agent", tags=["agent"])

    @router.get("/glance")
    def get_glance() -> dict[str, Any]:
        try:
            return glance()
        except Exception as exc:
            logger.warning("glance_failed", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="glance unavailable",
            ) from exc

    return router
