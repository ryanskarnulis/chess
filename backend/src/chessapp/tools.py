"""Tool layer: the validated boundary the agent talks through.

The LLM never touches `GameSession` or `EnginePlayer` directly — it names a
tool and passes args; `ToolRegistry.dispatch` validates the args against the
tool's JSON schema and runs the handler. Bad tool names, malformed args, and
domain errors (ValueError) all come back as `{"ok": False, "error": ...}` so
the agent loop can feed the failure back to the model instead of crashing.
"""

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import jsonschema

from chessapp.analysis import analyze_last_move, review_game
from chessapp.conversation import Transcript
from chessapp.engine import (
    DEFAULT_TIER,
    DIFFICULTY_TIERS,
    ELO_MAX,
    ELO_MIN,
    SKILL_MAX,
    SKILL_MIN,
    EnginePlayer,
)
from chessapp.game import GameSession

GET_BEST_MOVES_MAX = 10
UNDO_PLIES_MAX = 100
# Save names become filenames: one path segment, no traversal.
SAVE_NAME_PATTERN = r"^[A-Za-z0-9_-]{1,64}$"

VERBOSITY_LEVELS = ("low", "normal", "high")


@dataclass
class Settings:
    """Agent-adjustable app settings. Difficulty records exactly one of
    tier / skill_level / elo (the last one set); it is applied to the live
    engine when present and applied at assembly when an engine attaches.
    The personality (Glitch) is fixed, not a setting — see
    `chessapp.personality`."""

    verbosity: str = "normal"
    hints_mode: bool = False
    voice_output: bool = False
    tier: str | None = DEFAULT_TIER
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
    those tools report an error. `resume_game` replaces `session` (and
    `transcript`), so the context — not any captured reference — is the
    source of truth. `transcript` is the conversation memory the agent loop
    reads and records; it rides inside save files so a resumed game keeps
    its conversational thread."""

    session: GameSession
    engine: EnginePlayer | None = None
    save_dir: Path | None = None
    settings: Settings = field(default_factory=Settings)
    transcript: Transcript = field(default_factory=Transcript)


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

    def analyze_last_move_tool() -> dict[str, Any]:
        analysis = analyze_last_move(_require_engine(ctx), ctx.session)
        return {
            "ok": True,
            "played": analysis.played_san,
            "played_uci": analysis.played_uci,
            "best": analysis.best_san,
            "best_uci": analysis.best_uci,
            "cp_loss": analysis.cp_loss,
            "classification": analysis.classification,
            "color": analysis.color,
        }

    def review_game_tool() -> dict[str, Any]:
        review = review_game(_require_engine(ctx), ctx.session)
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

    def make_move(move: str) -> dict[str, Any]:
        result = ctx.session.submit_move(move)
        if not result.legal:
            return {"ok": True, "legal": False, "reason": result.reason}
        # Mirror the UI move path: a legal player move gets the engine's
        # reply in the same call, so a move sent through the agent never
        # leaves the player to move for both sides.
        engine_move: dict[str, Any] | None = None
        if ctx.engine is not None and not ctx.session.is_game_over():
            reply = ctx.engine.play_move(ctx.session)
            engine_move = {"san": reply.san, "uci": reply.uci}
        return {
            "ok": True,
            "legal": True,
            "san": result.san,
            "uci": result.uci,
            "engine_move": engine_move,
            "game_over": ctx.session.is_game_over(),
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
        # The transcript rides in the same file under a key GameSession
        # ignores, so game truth and conversation stay in one save and old
        # saves (no transcript key) remain loadable.
        path = _save_path(ctx, name)
        data = ctx.session.to_dict()
        data["transcript"] = ctx.transcript.to_dict()
        path.write_text(json.dumps(data, indent=2))
        return {"ok": True, "name": name}

    def resume_game(name: str = "autosave") -> dict[str, Any]:
        path = _save_path(ctx, name)
        if not path.exists():
            raise ValueError(f"no saved game named {name!r}")
        data = json.loads(path.read_text())
        # Validate both parts before touching the context, so a corrupt file
        # can't leave a restored board with someone else's conversation.
        session = GameSession.from_dict(data)
        transcript = Transcript.from_dict(data.get("transcript", []))
        ctx.session = session
        ctx.transcript = transcript
        return {
            "ok": True,
            "name": name,
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
            name="analyze_last_move",
            description=(
                "Analyze the last move played: how it compares to Stockfish's "
                "best from the same position — centipawn loss, verdict "
                "(good/inaccuracy/mistake/blunder), and what was best. Use "
                "this to answer 'what was my mistake?' and to explain moves."
            ),
            parameters=_no_args_schema(),
            handler=analyze_last_move_tool,
        )
    )
    registry.register(
        Tool(
            name="review_game",
            description=(
                "Review the whole game so far: every move classified "
                "(good/inaccuracy/mistake/blunder) with centipawn loss and "
                "the best alternative, plus per-color accuracy scores."
            ),
            parameters=_no_args_schema(),
            handler=review_game_tool,
        )
    )
    registry.register(
        Tool(
            name="make_move",
            description=(
                "Submit the player's move in SAN (e.g. 'Nf3') or UCI (e.g. "
                "'g1f3'). The engine decides legality: the result says legal "
                "or illegal. When the move is legal, the engine opponent "
                "replies immediately — the result's engine_move is its answer."
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
                "Set engine strength: pass exactly one of tier (named level: "
                "beginner ~500, casual ~1000, intermediate ~1500, advanced "
                f"~2000, maximum = full strength), skill_level ({SKILL_MIN}-"
                f"{SKILL_MAX}), or elo ({ELO_MIN}-{ELO_MAX}). Prefer tier "
                "unless the user asks for a specific number."
            ),
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
            },
            handler=set_difficulty,
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
