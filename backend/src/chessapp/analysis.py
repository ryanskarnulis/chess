""" "What was my mistake?" — deterministic last-move analysis.

Compares the move actually played against Stockfish's best from the same
position and reports the centipawn loss with a classification. The numbers
and the verdict come from here (deterministic code); the agent only
narrates them — it never judges a move in its head.

Works on a replay of the session (root FEN + move stack via `to_dict`), so
analysis can never touch live game truth.
"""

import math
from dataclasses import dataclass

import chess

from chessapp.engine import EnginePlayer, pov_cp
from chessapp.game import GameSession

# Lichess-style thresholds, in centipawns lost vs the best move.
INACCURACY_CP = 50
MISTAKE_CP = 100
BLUNDER_CP = 300


@dataclass(frozen=True)
class MoveAnalysis:
    """The verdict on one played move: what was played, what was best,
    and how much the difference cost (centipawns, mover's point of view)."""

    played_san: str
    played_uci: str
    best_san: str
    best_uci: str
    cp_loss: int
    classification: str
    color: str  # who played the analyzed move


def classify_cp_loss(cp_loss: int) -> str:
    if cp_loss < 0:
        raise ValueError(f"cp_loss cannot be negative, got {cp_loss}")
    if cp_loss < INACCURACY_CP:
        return "good"
    if cp_loss < MISTAKE_CP:
        return "inaccuracy"
    if cp_loss < BLUNDER_CP:
        return "mistake"
    return "blunder"


def analyze_last_move(
    engine: EnginePlayer, session: GameSession, color: str | None = None
) -> MoveAnalysis:
    """Analyze the last move played in `session` — or, with `color`, the last
    move played *by that color*.

    The loss is measured mover-POV: (best candidate's score in the position
    before the move) minus (the position's score after the played move). A
    move that ends the game in the mover's favor costs nothing by
    construction; a drawn end position scores as 0.
    """
    data = session.to_dict()
    moves = data["moves"]
    if not moves:
        raise ValueError("no moves to analyze yet")

    board = chess.Board(data["root_fen"])
    movers = []
    for uci in moves:
        movers.append("white" if board.turn == chess.WHITE else "black")
        board.push(chess.Move.from_uci(uci))

    if color is None:
        index = len(moves) - 1
    else:
        played_by = [i for i, mover in enumerate(movers) if mover == color]
        if not played_by:
            raise ValueError(f"no moves by {color} to analyze yet")
        index = played_by[-1]

    board = chess.Board(data["root_fen"])
    for uci in moves[:index]:
        board.push(chess.Move.from_uci(uci))
    played = chess.Move.from_uci(moves[index])
    mover = movers[index]
    played_san = board.san(played)

    before = GameSession(fen=board.fen())
    best = engine.get_best_moves(before, n=1)[0]
    best_cp = pov_cp(best.score_cp, best.mate_in, mover)

    board.push(played)
    if board.is_game_over():
        # Checkmate by the mover is at least as good as any alternative;
        # stalemate/draw is a dead-equal 0 — either way, no engine call.
        played_cp = best_cp if board.is_checkmate() else 0
    else:
        after = engine.evaluate_position(GameSession(fen=board.fen()))
        played_cp = pov_cp(after.score_cp, after.mate_in, mover)

    cp_loss = max(0, best_cp - played_cp)
    return MoveAnalysis(
        played_san=played_san,
        played_uci=played.uci(),
        best_san=best.san,
        best_uci=best.uci,
        cp_loss=cp_loss,
        classification=classify_cp_loss(cp_loss),
        color=mover,
    )


# --- whole-game review -------------------------------------------------------
#
# Adapted from lichess's published accuracy method (win-percent conversion
# and per-move accuracy curve), applied as glue over our own Stockfish
# bridge rather than depending on the lila codebase.

CLASSIFICATIONS = ("good", "inaccuracy", "mistake", "blunder")


@dataclass(frozen=True)
class ReviewedMove:
    """One move of the review: same verdict shape as `MoveAnalysis`, plus
    the move's accuracy percentage."""

    san: str
    uci: str
    color: str
    cp_loss: int
    classification: str
    best_san: str
    best_uci: str
    accuracy: float


@dataclass(frozen=True)
class GameReview:
    """The whole game reviewed: every move in order, plus per-color
    accuracy (0-100) and per-color classification counts."""

    moves: tuple[ReviewedMove, ...]
    accuracy: dict[str, float]
    counts: dict[str, dict[str, int]]


def win_percent(cp: int) -> float:
    """Mover-POV centipawns → expected win chance in percent (lichess's
    logistic fit). 0 cp is 50%; the `MATE_CP` scale saturates the curve."""
    return 50.0 + 50.0 * (2.0 / (1.0 + math.exp(-0.00368208 * cp)) - 1.0)


def move_accuracy(win_before: float, win_after: float) -> float:
    """Accuracy of one move from the win-percent drop it caused (lichess's
    exponential fit), clamped to 0-100. A move that holds or improves the
    win chance is perfect — the engine can under-promise, never the mover
    over-deliver."""
    if win_after >= win_before:
        return 100.0
    raw = 103.1668 * math.exp(-0.04354 * (win_before - win_after)) - 3.1669
    return max(0.0, min(100.0, raw))


def review_game(engine: EnginePlayer, session: GameSession) -> GameReview:
    """Review every move of `session`'s game.

    One engine analysis per position (a position's evaluation doubles as the
    score of the previous move's outcome), replayed from the root FEN so the
    live session is never touched. Per-color accuracy is the mean of that
    color's move accuracies.
    """
    data = session.to_dict()
    if not data["moves"]:
        raise ValueError("no moves to review yet")

    board = chess.Board(data["root_fen"])
    best = engine.get_best_moves(GameSession(fen=board.fen()), n=1)[0]
    reviewed: list[ReviewedMove] = []
    for uci in data["moves"]:
        move = chess.Move.from_uci(uci)
        mover = "white" if board.turn == chess.WHITE else "black"
        san = board.san(move)
        best_cp = pov_cp(best.score_cp, best.mate_in, mover)
        board.push(move)
        if board.is_game_over():
            played_cp = best_cp if board.is_checkmate() else 0
        else:
            next_best = engine.get_best_moves(GameSession(fen=board.fen()), n=1)[0]
            played_cp = pov_cp(next_best.score_cp, next_best.mate_in, mover)
        cp_loss = max(0, best_cp - played_cp)
        reviewed.append(
            ReviewedMove(
                san=san,
                uci=uci,
                color=mover,
                cp_loss=cp_loss,
                classification=classify_cp_loss(cp_loss),
                best_san=best.san,
                best_uci=best.uci,
                accuracy=move_accuracy(win_percent(best_cp), win_percent(played_cp)),
            )
        )
        if board.is_game_over():
            break
        best = next_best

    accuracy: dict[str, float] = {}
    counts: dict[str, dict[str, int]] = {}
    for color in ("white", "black"):
        color_moves = [m for m in reviewed if m.color == color]
        if not color_moves:
            continue
        accuracy[color] = round(
            sum(m.accuracy for m in color_moves) / len(color_moves), 1
        )
        counts[color] = {
            c: sum(1 for m in color_moves if m.classification == c)
            for c in CLASSIFICATIONS
        }
    return GameReview(moves=tuple(reviewed), accuracy=accuracy, counts=counts)
