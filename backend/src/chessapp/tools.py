"""Tool layer: the validated boundary the agent talks through.

The LLM never touches `GameSession` or `EnginePlayer` directly — it names a
tool and passes args; `ToolRegistry.dispatch` validates the args against the
tool's JSON schema and runs the handler. Bad tool names, malformed args, and
domain errors (ValueError) all come back as `{"ok": False, "error": ...}` so
the agent loop can feed the failure back to the model instead of crashing.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import jsonschema

from chessapp.engine import EnginePlayer
from chessapp.game import GameSession

GET_BEST_MOVES_MAX = 10


@dataclass(frozen=True)
class Tool:
    """One agent-callable capability: OpenAI-style schema + bound handler."""

    name: str
    description: str
    parameters: dict[str, Any]  # JSON schema for the args object
    handler: Callable[..., dict[str, Any]]


@dataclass
class ToolContext:
    """What tools operate on. `engine` is optional: analysis tools report
    an error result without one; everything else works engine-free."""

    session: GameSession
    engine: EnginePlayer | None = None


@dataclass
class ToolRegistry:
    _tools: dict[str, Tool] = field(default_factory=dict)

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"tool already registered: {tool.name}")
        jsonschema.Draft202012Validator.check_schema(tool.parameters)
        self._tools[tool.name] = tool

    def definitions(self) -> list[dict[str, Any]]:
        """All tools as OpenAI-style function definitions."""
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
        ]

    def dispatch(self, name: str, args: Any) -> dict[str, Any]:
        """Run tool `name` with `args`; never raises on agent-caused faults."""
        tool = self._tools.get(name)
        if tool is None:
            return {"ok": False, "error": f"unknown tool: {name}"}
        try:
            jsonschema.validate(args, tool.parameters)
        except jsonschema.ValidationError as exc:
            return {"ok": False, "error": f"invalid args for {name}: {exc.message}"}
        try:
            return tool.handler(**args)
        except ValueError as exc:
            return {"ok": False, "error": str(exc)}


def _no_args_schema() -> dict[str, Any]:
    return {"type": "object", "properties": {}, "additionalProperties": False}


def _outcome_dict(session: GameSession) -> dict[str, Any] | None:
    outcome = session.outcome()
    if outcome is None:
        return None
    return {
        "termination": outcome.termination,
        "winner": outcome.winner,
        "result": outcome.result,
    }


def _require_engine(ctx: ToolContext) -> EnginePlayer:
    if ctx.engine is None:
        raise ValueError("engine unavailable: analysis tools need Stockfish")
    return ctx.engine


def build_registry(ctx: ToolContext) -> ToolRegistry:
    """All read tools bound to `ctx`. Write and settings tools join in
    later slices of the tool-layer epic."""
    registry = ToolRegistry()

    def get_board_state() -> dict[str, Any]:
        return {
            "ok": True,
            "fen": ctx.session.fen(),
            "turn": ctx.session.turn,
            "game_over": ctx.session.is_game_over(),
            "outcome": _outcome_dict(ctx.session),
        }

    def get_legal_moves() -> dict[str, Any]:
        return {"ok": True, "moves": ctx.session.legal_moves()}

    def get_move_history() -> dict[str, Any]:
        return {"ok": True, "moves": ctx.session.move_history()}

    def get_captured_pieces() -> dict[str, Any]:
        captured = ctx.session.captured_pieces()
        return {"ok": True, "white": captured["white"], "black": captured["black"]}

    def evaluate_position() -> dict[str, Any]:
        evaluation = _require_engine(ctx).evaluate_position(ctx.session)
        return {
            "ok": True,
            "score_cp": evaluation.score_cp,
            "mate_in": evaluation.mate_in,
        }

    def get_best_moves(n: int = 3) -> dict[str, Any]:
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

    registry.register(
        Tool(
            name="get_board_state",
            description=(
                "Current position: FEN, side to move, whether the game is "
                "over, and the outcome if it is."
            ),
            parameters=_no_args_schema(),
            handler=get_board_state,
        )
    )
    registry.register(
        Tool(
            name="get_legal_moves",
            description="All legal moves in the current position, in SAN.",
            parameters=_no_args_schema(),
            handler=get_legal_moves,
        )
    )
    registry.register(
        Tool(
            name="get_move_history",
            description="Moves played so far, in SAN, in order.",
            parameters=_no_args_schema(),
            handler=get_move_history,
        )
    )
    registry.register(
        Tool(
            name="get_captured_pieces",
            description=(
                "Piece symbols each color has captured so far, in capture order."
            ),
            parameters=_no_args_schema(),
            handler=get_captured_pieces,
        )
    )
    registry.register(
        Tool(
            name="evaluate_position",
            description=(
                "Stockfish evaluation of the current position from White's "
                "point of view: centipawns, or mate-in-N."
            ),
            parameters=_no_args_schema(),
            handler=evaluate_position,
        )
    )
    registry.register(
        Tool(
            name="get_best_moves",
            description=(
                "Top n candidate moves from Stockfish (MultiPV), best first, "
                "with SAN, UCI, and White-POV scores."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "n": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": GET_BEST_MOVES_MAX,
                        "description": "How many candidate moves to return.",
                    }
                },
                "additionalProperties": False,
            },
            handler=get_best_moves,
        )
    )
    return registry
