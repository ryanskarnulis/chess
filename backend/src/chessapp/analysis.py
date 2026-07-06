""" "What was my mistake?" — deterministic last-move analysis.

Compares the move actually played against Stockfish's best from the same
position and reports the centipawn loss with a classification. The numbers
and the verdict come from here (deterministic code); the agent only
narrates them — it never judges a move in its head.

Works on a replay of the session (root FEN + move stack via `to_dict`), so
analysis can never touch live game truth.
"""

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


def analyze_last_move(engine: EnginePlayer, session: GameSession) -> MoveAnalysis:
    """Analyze the last move played in `session`.

    The loss is measured mover-POV: (best candidate's score in the position
    before the move) minus (the position's score after the played move). A
    move that ends the game in the mover's favor costs nothing by
    construction; a drawn end position scores as 0.
    """
    data = session.to_dict()
    if not data["moves"]:
        raise ValueError("no moves to analyze yet")

    board = chess.Board(data["root_fen"])
    for uci in data["moves"][:-1]:
        board.push(chess.Move.from_uci(uci))
    played = chess.Move.from_uci(data["moves"][-1])
    mover = "white" if board.turn == chess.WHITE else "black"
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
