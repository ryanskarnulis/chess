"""Stockfish bridge via python-chess UCI.

Stockfish is a calculation tool, not the referee: it proposes moves, but
every move still enters the game through `GameSession.submit_move`, and
`choose_move` works on a copy of the position so the engine can never
touch session truth.
"""

import random
from dataclasses import dataclass

import chess
import chess.engine

from chessapp.game import GameSession, MoveResult

SKILL_MIN, SKILL_MAX = 0, 20
# Stockfish's own UCI_Elo bounds.
ELO_MIN, ELO_MAX = 1320, 3190


@dataclass(frozen=True)
class DifficultyProfile:
    """How one named tier is realized on the engine.

    Stockfish cannot play below ~1300 through UCI options alone (UCI_Elo
    floors at 1320 and Skill Level 0 is still club strength), so the low
    tiers add a weakening layer on top: `max_nodes` starves the search and
    `blunder_chance` is the per-move probability of playing a random legal
    move instead of the engine's choice.
    """

    name: str
    skill_level: int | None = None
    elo: int | None = None
    max_nodes: int | None = None
    blunder_chance: float = 0.0


# Named tiers with human target strengths: beginner ~500, casual ~1000,
# intermediate ~1500, advanced ~2000, maximum = full strength. Node counts
# and blunder rates are calibration knobs — tune from real games.
DIFFICULTY_TIERS = {
    "beginner": DifficultyProfile(
        "beginner", skill_level=0, max_nodes=150, blunder_chance=0.25
    ),
    "casual": DifficultyProfile(
        "casual", skill_level=2, max_nodes=800, blunder_chance=0.10
    ),
    "intermediate": DifficultyProfile("intermediate", elo=1500),
    "advanced": DifficultyProfile("advanced", elo=2000),
    "maximum": DifficultyProfile("maximum", skill_level=20),
}
# What a fresh app plays at. Stockfish's own default is full strength, so an
# engine must never be left unconfigured.
DEFAULT_TIER = "casual"

DEFAULT_MOVE_TIME = 0.1
DEFAULT_ANALYSIS_DEPTH = 12

# A mate scores far beyond any centipawn evaluation; nearer mates score
# higher, so a mate-in-1 beats a mate-in-3.
MATE_CP = 100_000


@dataclass(frozen=True)
class Evaluation:
    """Position score from White's point of view.

    Exactly one of `score_cp` (centipawns) / `mate_in` (signed: positive
    means White mates in N) is set.
    """

    score_cp: int | None
    mate_in: int | None


@dataclass(frozen=True)
class CandidateMove:
    """One MultiPV candidate, best-first; score fields as in Evaluation."""

    uci: str
    san: str
    score_cp: int | None
    mate_in: int | None


def pov_cp(score_cp: int | None, mate_in: int | None, turn: str) -> int:
    """Collapse a White-POV (score_cp, mate_in) pair — the shape every
    analysis result uses — into one comparable centipawn number from `turn`'s
    point of view. Mates map onto the `MATE_CP` scale."""
    if mate_in is not None:
        magnitude = MATE_CP - abs(mate_in)
        cp = magnitude if mate_in > 0 else -magnitude
    else:
        cp = score_cp or 0
    return cp if turn == "white" else -cp


def _score_fields(score: chess.engine.PovScore) -> tuple[int | None, int | None]:
    white = score.white()
    if white.is_mate():
        return None, white.mate()
    return white.score(), None


def validate_skill_level(level: object) -> int:
    if not isinstance(level, int) or isinstance(level, bool):
        raise ValueError(f"skill level must be an int, got {level!r}")
    if not SKILL_MIN <= level <= SKILL_MAX:
        raise ValueError(f"skill level must be {SKILL_MIN}-{SKILL_MAX}, got {level}")
    return level


def validate_elo(elo: object) -> int:
    if not isinstance(elo, int) or isinstance(elo, bool):
        raise ValueError(f"elo must be an int, got {elo!r}")
    if not ELO_MIN <= elo <= ELO_MAX:
        raise ValueError(f"elo must be {ELO_MIN}-{ELO_MAX}, got {elo}")
    return elo


def validate_tier(name: object) -> DifficultyProfile:
    profile = DIFFICULTY_TIERS.get(name) if isinstance(name, str) else None
    if profile is None:
        options = ", ".join(DIFFICULTY_TIERS)
        raise ValueError(f"unknown difficulty tier {name!r}; expected one of {options}")
    return profile


class EnginePlayer:
    """One Stockfish process playing moves at a configurable strength."""

    def __init__(
        self,
        path: str = "stockfish",
        move_time: float = DEFAULT_MOVE_TIME,
        rng: random.Random | None = None,
    ):
        self._engine = chess.engine.SimpleEngine.popen_uci(path)
        self._limit = chess.engine.Limit(time=move_time)
        # Blunder injection randomness; injectable so tests can force or
        # forbid the blunder path deterministically.
        self._rng = rng if rng is not None else random.Random()
        self._max_nodes: int | None = None
        self._blunder_chance = 0.0

    def __enter__(self) -> "EnginePlayer":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        self._engine.quit()

    def set_skill_level(self, level: int) -> None:
        validate_skill_level(level)
        # Raw knobs mean exactly the UCI strength asked for — no leftover
        # tier weakening on top.
        self._max_nodes = None
        self._blunder_chance = 0.0
        self._engine.configure({"UCI_LimitStrength": False, "Skill Level": level})

    def set_elo(self, elo: int) -> None:
        validate_elo(elo)
        self._max_nodes = None
        self._blunder_chance = 0.0
        self._engine.configure({"UCI_LimitStrength": True, "UCI_Elo": elo})

    def set_tier(self, name: str) -> None:
        """Play at a named difficulty tier (see `DIFFICULTY_TIERS`)."""
        profile = validate_tier(name)
        if profile.elo is not None:
            self._engine.configure({"UCI_LimitStrength": True, "UCI_Elo": profile.elo})
        else:
            self._engine.configure(
                {"UCI_LimitStrength": False, "Skill Level": profile.skill_level}
            )
        self._max_nodes = profile.max_nodes
        self._blunder_chance = profile.blunder_chance

    def choose_move(self, session: GameSession) -> str:
        """The engine's move for the current position, as UCI."""
        if session.is_game_over():
            raise ValueError("cannot choose a move: game is over")
        board = chess.Board(session.fen())
        if self._blunder_chance and self._rng.random() < self._blunder_chance:
            return self._rng.choice(list(board.legal_moves)).uci()
        limit = (
            chess.engine.Limit(nodes=self._max_nodes)
            if self._max_nodes is not None
            else self._limit
        )
        result = self._engine.play(board, limit)
        if result.move is None:
            raise ValueError("engine returned no move")
        return result.move.uci()

    def play_move(self, session: GameSession) -> MoveResult:
        """Choose a move and submit it through the session's legality gate."""
        return session.submit_move(self.choose_move(session))

    def evaluate_position(
        self, session: GameSession, depth: int = DEFAULT_ANALYSIS_DEPTH
    ) -> Evaluation:
        if session.is_game_over():
            raise ValueError("cannot evaluate: game is over")
        board = chess.Board(session.fen())
        info = self._engine.analyse(board, chess.engine.Limit(depth=depth))
        score_cp, mate_in = _score_fields(info["score"])
        return Evaluation(score_cp=score_cp, mate_in=mate_in)

    def get_best_moves(
        self, session: GameSession, n: int = 3, depth: int = DEFAULT_ANALYSIS_DEPTH
    ) -> list[CandidateMove]:
        """Top `n` candidate moves (MultiPV), best first. Returns fewer when
        the position has fewer legal moves."""
        if n < 1:
            raise ValueError(f"n must be at least 1, got {n}")
        if session.is_game_over():
            raise ValueError("cannot suggest moves: game is over")
        board = chess.Board(session.fen())
        infos = self._engine.analyse(board, chess.engine.Limit(depth=depth), multipv=n)
        candidates = []
        for info in infos:
            pv = info.get("pv")
            if not pv:
                continue
            move = pv[0]
            score_cp, mate_in = _score_fields(info["score"])
            candidates.append(
                CandidateMove(
                    uci=move.uci(),
                    san=board.san(move),
                    score_cp=score_cp,
                    mate_in=mate_in,
                )
            )
        return candidates
