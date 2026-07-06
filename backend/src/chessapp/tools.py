"""Tool layer: the validated boundary the agent talks through.

The LLM never touches `GameSession` or `EnginePlayer` directly — it names a
tool and passes args; `ToolRegistry.dispatch` validates the args against the
tool's JSON schema and runs the handler. Bad tool names, malformed args, and
domain errors (ValueError) all come back as `{"ok": False, "error": ...}` so
the agent loop can feed the failure back to the model instead of crashing.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import jsonschema

from chessapp.engine import ELO_MAX, ELO_MIN, SKILL_MAX, SKILL_MIN, EnginePlayer
from chessapp.game import GameSession

GET_BEST_MOVES_MAX = 10
UNDO_PLIES_MAX = 100
# Save names become filenames: one path segment, no traversal.
SAVE_NAME_PATTERN = r"^[A-Za-z0-9_-]{1,64}$"

# The full personality roster (Phase 1 pair + Phase 3 additions).
PERSONALITIES = (
    "friendly_rival",
    "calm_coach",
    "trash_talker",
    "grandmaster",
    "villain",
    "silent_assassin",
    "beginner_bot",
    "streamer",
)
VERBOSITY_LEVELS = ("low", "normal", "high")


@dataclass
class Settings:
    """Agent-adjustable app settings. Difficulty records exactly one of
    skill_level / elo (the last one set); it is applied to the live engine
    when present and re-applied when an engine attaches later."""

    personality: str = PERSONALITIES[0]
    verbosity: str = "normal"
    hints_mode: bool = False
    voice_output: bool = False
    skill_level: int | None = None
    elo: int | None = None


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
    an error result without one; everything else works engine-free.
    `save_dir` is where save_game/resume_game keep their files; without it
    those tools report an error. `resume_game` replaces `session`, so the
    context — not any captured reference — is the source of truth."""

    session: GameSession
    engine: EnginePlayer | None = None
    save_dir: Path | None = None
    settings: Settings = field(default_factory=Settings)


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


def _save_path(ctx: ToolContext, name: str) -> Path:
    if ctx.save_dir is None:
        raise ValueError("saving unavailable: no save directory configured")
    return ctx.save_dir / f"{name}.json"


def build_registry(ctx: ToolContext) -> ToolRegistry:
    """All read and write tools bound to `ctx`. Settings tools join in the
    final slice of the tool-layer epic."""
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

    def make_move(move: str) -> dict[str, Any]:
        result = ctx.session.submit_move(move)
        if not result.legal:
            return {"ok": True, "legal": False, "reason": result.reason}
        return {
            "ok": True,
            "legal": True,
            "san": result.san,
            "uci": result.uci,
            "game_over": result.game_over,
            "fen": ctx.session.fen(),
            "turn": ctx.session.turn,
        }

    def undo(plies: int = 1) -> dict[str, Any]:
        result = ctx.session.undo(plies)
        if not result.ok:
            return {"ok": False, "error": result.reason}
        return {
            "ok": True,
            "undone": list(result.undone),
            "fen": ctx.session.fen(),
            "turn": ctx.session.turn,
        }

    def new_game() -> dict[str, Any]:
        ctx.session.new_game()
        return {"ok": True, "fen": ctx.session.fen(), "turn": ctx.session.turn}

    def resign(color: str | None = None) -> dict[str, Any]:
        outcome = ctx.session.resign(color)
        return {
            "ok": True,
            "outcome": {
                "termination": outcome.termination,
                "winner": outcome.winner,
                "result": outcome.result,
            },
        }

    def export_pgn() -> dict[str, Any]:
        return {"ok": True, "pgn": ctx.session.export_pgn()}

    def save_game(name: str = "autosave") -> dict[str, Any]:
        path = _save_path(ctx, name)
        ctx.session.save(path)
        return {"ok": True, "name": name}

    def resume_game(name: str = "autosave") -> dict[str, Any]:
        path = _save_path(ctx, name)
        if not path.exists():
            raise ValueError(f"no saved game named {name!r}")
        ctx.session = GameSession.load(path)
        return {
            "ok": True,
            "name": name,
            "fen": ctx.session.fen(),
            "turn": ctx.session.turn,
        }

    def set_difficulty(
        skill_level: int | None = None, elo: int | None = None
    ) -> dict[str, Any]:
        if skill_level is not None:
            if ctx.engine is not None:
                ctx.engine.set_skill_level(skill_level)
            ctx.settings.skill_level = skill_level
            ctx.settings.elo = None
        else:
            if ctx.engine is not None:
                ctx.engine.set_elo(elo)
            ctx.settings.elo = elo
            ctx.settings.skill_level = None
        return {"ok": True, "skill_level": skill_level, "elo": elo}

    def set_personality(personality: str) -> dict[str, Any]:
        ctx.settings.personality = personality
        return {"ok": True, "personality": personality}

    def set_verbosity(verbosity: str) -> dict[str, Any]:
        ctx.settings.verbosity = verbosity
        return {"ok": True, "verbosity": verbosity}

    def set_hints_mode(enabled: bool) -> dict[str, Any]:
        ctx.settings.hints_mode = enabled
        return {"ok": True, "hints_mode": enabled}

    def set_voice_output(enabled: bool) -> dict[str, Any]:
        ctx.settings.voice_output = enabled
        return {"ok": True, "voice_output": enabled}

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
    registry.register(
        Tool(
            name="make_move",
            description=(
                "Submit a move in SAN (e.g. 'Nf3') or UCI (e.g. 'g1f3'). The "
                "engine decides legality: the result says legal or illegal."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "move": {"type": "string", "description": "SAN or UCI move."}
                },
                "required": ["move"],
                "additionalProperties": False,
            },
            handler=make_move,
        )
    )
    registry.register(
        Tool(
            name="undo",
            description=(
                "Take back the last N half-moves (plies). Use plies=2 to "
                "undo both the player's move and the engine's reply."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "plies": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": UNDO_PLIES_MAX,
                        "description": "How many half-moves to take back.",
                    }
                },
                "additionalProperties": False,
            },
            handler=undo,
        )
    )
    registry.register(
        Tool(
            name="new_game",
            description="Reset to the starting position and begin a new game.",
            parameters=_no_args_schema(),
            handler=new_game,
        )
    )
    registry.register(
        Tool(
            name="resign",
            description=(
                "Resign the game. Defaults to the side to move; pass a color "
                "to resign for that side."
            ),
            parameters={
                "type": "object",
                "properties": {"color": {"type": "string", "enum": ["white", "black"]}},
                "additionalProperties": False,
            },
            handler=resign,
        )
    )
    registry.register(
        Tool(
            name="save_game",
            description="Save the current game under a name (default 'autosave').",
            parameters={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "pattern": SAVE_NAME_PATTERN}
                },
                "additionalProperties": False,
            },
            handler=save_game,
        )
    )
    registry.register(
        Tool(
            name="resume_game",
            description="Resume a previously saved game by name (default 'autosave').",
            parameters={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "pattern": SAVE_NAME_PATTERN}
                },
                "additionalProperties": False,
            },
            handler=resume_game,
        )
    )
    registry.register(
        Tool(
            name="export_pgn",
            description="Export the game so far as PGN.",
            parameters=_no_args_schema(),
            handler=export_pgn,
        )
    )
    registry.register(
        Tool(
            name="set_difficulty",
            description=(
                "Set engine strength: pass exactly one of skill_level "
                f"({SKILL_MIN}-{SKILL_MAX}) or elo ({ELO_MIN}-{ELO_MAX})."
            ),
            parameters={
                "type": "object",
                "properties": {
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
                "oneOf": [{"required": ["skill_level"]}, {"required": ["elo"]}],
                "additionalProperties": False,
            },
            handler=set_difficulty,
        )
    )
    registry.register(
        Tool(
            name="set_personality",
            description="Choose the agent's personality.",
            parameters={
                "type": "object",
                "properties": {
                    "personality": {"type": "string", "enum": list(PERSONALITIES)}
                },
                "required": ["personality"],
                "additionalProperties": False,
            },
            handler=set_personality,
        )
    )
    registry.register(
        Tool(
            name="set_verbosity",
            description="How chatty the agent's commentary is.",
            parameters={
                "type": "object",
                "properties": {
                    "verbosity": {"type": "string", "enum": list(VERBOSITY_LEVELS)}
                },
                "required": ["verbosity"],
                "additionalProperties": False,
            },
            handler=set_verbosity,
        )
    )
    registry.register(
        Tool(
            name="set_hints_mode",
            description="Turn move hints on or off.",
            parameters={
                "type": "object",
                "properties": {"enabled": {"type": "boolean"}},
                "required": ["enabled"],
                "additionalProperties": False,
            },
            handler=set_hints_mode,
        )
    )
    registry.register(
        Tool(
            name="set_voice_output",
            description="Turn spoken (TTS) output on or off.",
            parameters={
                "type": "object",
                "properties": {"enabled": {"type": "boolean"}},
                "required": ["enabled"],
                "additionalProperties": False,
            },
            handler=set_voice_output,
        )
    )
    return registry
