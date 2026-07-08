"""Stockfish bridge via python-chess UCI.

Stockfish is a calculation tool, not the referee: it proposes moves, but
every move still enters the game through `GameSession.submit_move`, and
`choose_move` works on a copy of the position so the engine can never
touch session truth.
"""

import math
import random
from dataclasses import dataclass

import chess
import chess.engine

from chessapp.game import GameSession, MoveResult

SKILL_MIN, SKILL_MAX = 0, 20
# Stockfish's own UCI_Elo bounds.
ELO_MIN, ELO_MAX = 1320, 3190

# A move's centipawn loss vs the best move is clamped to this before it feeds
# the sampler weight. Without a cap, a move that walks into mate (scored in the
# tens of thousands) underflows exp() to exactly 0 and could never be played;
# with it, even hopeless moves keep a tiny weight, so a weak tier occasionally
# makes the kind of catastrophic blunder a real beginner does.
MAX_SAMPLE_LOSS = 1000


@dataclass(frozen=True)
class DifficultyProfile:
    """How one named tier is realized on the engine.

    Stockfish cannot play below ~1320 through UCI options alone (UCI_Elo
    floors at 1320 and Skill Level 0 still plays ~1300+), so the low tiers
    can't come from a UCI knob. Instead they set `temperature`: the engine
    scores every legal move at `sample_depth` and picks one weighted by how
    much it loses versus the best (see `sample_weighted`). A higher
    temperature flattens that distribution — more, and worse, mistakes — so
    it, not a raw strength setting, is what makes a tier weak.
    """

    name: str
    skill_level: int | None = None
    elo: int | None = None
    temperature: float | None = None
    sample_depth: int = 4


# Named tiers with human target strengths: beginner ~500, casual ~1000,
# intermediate ~1500, advanced ~2000, maximum = full strength. The sampler
# temperatures are calibration knobs — a gauntlet (see scripts/) confirms the
# ordering beginner < casual < the 1320 UCI floor; tune from real games.
DIFFICULTY_TIERS = {
    "beginner": DifficultyProfile("beginner", temperature=700, sample_depth=4),
    "casual": DifficultyProfile("casual", temperature=250, sample_depth=4),
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


def sample_weighted(losses: list[int], temperature: float, rng: random.Random) -> int:
    """Pick an index into `losses` — each the centipawn loss of one candidate
    move versus the best (best-first, so `losses[0]` is 0) — with probability
    proportional to ``exp(-loss / temperature)``.

    A small temperature concentrates almost all weight on the best move; a
    large one flattens toward uniform, so worse moves get played more often.
    This is the knob that makes a tier weak. Losses are clamped to
    `MAX_SAMPLE_LOSS` so a catastrophic move keeps a small nonzero weight.
    """
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    weights = [
        math.exp(-min(max(0, loss), MAX_SAMPLE_LOSS) / temperature) for loss in losses
    ]
    threshold = rng.random() * sum(weights)
    cumulative = 0.0
    for i, weight in enumerate(weights):
        cumulative += weight
        if threshold <= cumulative:
            return i
    return len(weights) - 1


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
        # Move-sampling randomness for the weak tiers; injectable so tests can
        # drive the weighted pick deterministically.
        self._rng = rng if rng is not None else random.Random()
        # Set together by a sampler tier (None => play the engine's move
        # straight, at whatever raw strength is configured).
        self._sample_temperature: float | None = None
        self._sample_depth = 0

    def __enter__(self) -> "EnginePlayer":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        self._engine.quit()

    def _clear_sampler(self) -> None:
        self._sample_temperature = None
        self._sample_depth = 0

    def set_skill_level(self, level: int) -> None:
        validate_skill_level(level)
        # Raw knobs mean exactly the UCI strength asked for — no leftover
        # tier weakening on top.
        self._clear_sampler()
        self._engine.configure({"UCI_LimitStrength": False, "Skill Level": level})

    def set_elo(self, elo: int) -> None:
        validate_elo(elo)
        self._clear_sampler()
        self._engine.configure({"UCI_LimitStrength": True, "UCI_Elo": elo})

    def set_tier(self, name: str) -> None:
        """Play at a named difficulty tier (see `DIFFICULTY_TIERS`)."""
        profile = validate_tier(name)
        if profile.temperature is not None:
            # The sampler does its own weakening, so the engine underneath
            # analyses at full strength for accurate move scores.
            self._engine.configure(
                {"UCI_LimitStrength": False, "Skill Level": SKILL_MAX}
            )
            self._sample_temperature = profile.temperature
            self._sample_depth = profile.sample_depth
        elif profile.elo is not None:
            self._engine.configure({"UCI_LimitStrength": True, "UCI_Elo": profile.elo})
            self._clear_sampler()
        else:
            self._engine.configure(
                {"UCI_LimitStrength": False, "Skill Level": profile.skill_level}
            )
            self._clear_sampler()

    def choose_move(self, session: GameSession) -> str:
        """The engine's move for the current position, as UCI."""
        if session.is_game_over():
            raise ValueError("cannot choose a move: game is over")
        board = chess.Board(session.fen())
        if self._sample_temperature is not None:
            return self._sample_move(board)
        result = self._engine.play(board, self._limit)
        if result.move is None:
            raise ValueError("engine returned no move")
        return result.move.uci()

    def _sample_move(self, board: chess.Board) -> str:
        """Score every legal move at the tier's shallow depth, then pick one
        weighted toward the best (see `sample_weighted`). Sampling over the
        *whole* legal-move pool — not just Stockfish's top few — is what lets
        a weak tier occasionally hang material, the way a real beginner does.
        """
        turn = "white" if board.turn == chess.WHITE else "black"
        legal_count = board.legal_moves.count()
        infos = self._engine.analyse(
            board,
            chess.engine.Limit(depth=self._sample_depth),
            multipv=legal_count,
        )
        moves: list[str] = []
        pov_scores: list[int] = []
        for info in infos:
            pv = info.get("pv")
            if not pv:
                continue
            score_cp, mate_in = _score_fields(info["score"])
            moves.append(pv[0].uci())
            pov_scores.append(pov_cp(score_cp, mate_in, turn))
        if not moves:
            raise ValueError("engine returned no candidate moves")
        best = max(pov_scores)
        losses = [best - score for score in pov_scores]
        return moves[sample_weighted(losses, self._sample_temperature, self._rng)]

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
