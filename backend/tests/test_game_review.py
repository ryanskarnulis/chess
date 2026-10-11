"""Game review: per-move classification + accuracy for a whole game.

The review method is adapted from lichess's published approach (win-percent
conversion and per-move accuracy curve) as glue over our own Stockfish
bridge — deterministic code produces every number; the agent only narrates.
Pure math is always tested; whole-game reviews need a live Stockfish and
skip without one.
"""

import shutil
from pathlib import Path

import chess
import chess.pgn
import pytest

from chessapp.analysis import (
    CRITICAL_PER_COLOR,
    EVAL_CEILING_CP,
    GameReview,
    ReviewedMove,
    critical_moves,
    move_accuracy,
    review_game,
    win_percent,
)
from chessapp.engine import CandidateMove, EnginePlayer, Evaluation
from chessapp.facts import analysis_numbers
from chessapp.game import GameSession
from chessapp.tools import ToolContext, build_registry

requires_stockfish = pytest.mark.skipif(
    shutil.which("stockfish") is None, reason="stockfish binary not installed"
)

# 1.e4 e5 2.Bc4 Bc5 3.Qh5 Nf6?? 4.Qxf7# — scholar's mate, one huge black
# blunder, White finishing with mate.
SCHOLARS_MATE = ("e4", "e5", "Bc4", "Bc5", "Qh5", "Nf6", "Qxf7#")

LATE_GAME_PGN = Path(__file__).parent / "late_game_84_plies.pgn"


@pytest.fixture(scope="module")
def engine():
    if shutil.which("stockfish") is None:
        pytest.skip("stockfish binary not installed")
    with EnginePlayer() as player:
        yield player


def play(session, *moves):
    for move in moves:
        assert session.submit_move(move).legal, move
    return session


# --- win percent (pure) -----------------------------------------------------


def test_equal_position_is_fifty_percent():
    assert win_percent(0) == pytest.approx(50.0)


def test_win_percent_is_monotonic_and_bounded():
    values = [win_percent(cp) for cp in (-100_000, -300, 0, 300, 100_000)]
    assert values == sorted(values)
    assert 0.0 <= values[0] < 50.0 < values[-1] <= 100.0


def test_mate_scale_saturates():
    assert win_percent(100_000) > 99.9
    assert win_percent(-100_000) < 0.1


# --- move accuracy (pure) ---------------------------------------------------


def test_no_loss_is_perfect_accuracy():
    assert move_accuracy(55.0, 55.0) == 100.0


def test_improvement_is_perfect_accuracy():
    # The engine can under-promise; a move that improves the win chance is
    # never penalized.
    assert move_accuracy(50.0, 60.0) == 100.0


def test_a_huge_drop_scores_near_zero():
    assert move_accuracy(90.0, 1.0) < 10.0


def test_accuracy_is_clamped_to_0_100():
    assert 0.0 <= move_accuracy(100.0, 0.0) <= 100.0


def test_bigger_drops_score_lower():
    small = move_accuracy(60.0, 55.0)
    large = move_accuracy(60.0, 20.0)
    assert large < small <= 100.0


# --- review_game ------------------------------------------------------------


def test_reviewing_an_empty_game_raises():
    class NeverCalledEngine:
        def get_best_moves(self, session, n=1):  # pragma: no cover
            raise AssertionError("no analysis should happen")

    with pytest.raises(ValueError):
        review_game(NeverCalledEngine(), GameSession())


@requires_stockfish
def test_review_covers_every_move_in_order(engine):
    session = play(GameSession(), *SCHOLARS_MATE)
    review = review_game(engine, session)
    assert isinstance(review, GameReview)
    assert [m.san for m in review.moves] == list(SCHOLARS_MATE)
    assert [m.color for m in review.moves] == [
        "white",
        "black",
    ] * 3 + ["white"]


@requires_stockfish
def test_the_blunder_is_found_and_counted(engine):
    session = play(GameSession(), *SCHOLARS_MATE)
    review = review_game(engine, session)
    nf6 = review.moves[5]
    assert nf6.san == "Nf6"
    assert nf6.classification == "blunder"
    assert nf6.cp_loss >= 300
    assert review.counts["black"]["blunder"] >= 1


@requires_stockfish
def test_delivering_mate_costs_nothing(engine):
    session = play(GameSession(), *SCHOLARS_MATE)
    review = review_game(engine, session)
    mate = review.moves[-1]
    assert mate.san == "Qxf7#"
    assert mate.cp_loss == 0
    assert mate.classification == "good"


@requires_stockfish
def test_the_blundering_side_scores_lower_accuracy(engine):
    session = play(GameSession(), *SCHOLARS_MATE)
    review = review_game(engine, session)
    assert 0.0 <= review.accuracy["black"] < review.accuracy["white"] <= 100.0


@requires_stockfish
def test_review_does_not_mutate_the_session(engine):
    session = play(GameSession(), *SCHOLARS_MATE)
    fen_before = session.fen()
    review_game(engine, session)
    assert session.fen() == fen_before


@requires_stockfish
def test_one_game_reviews_identically_whatever_ran_before():
    """#454: three reviews of one unchanged game gave three accuracy pairs,
    because each depth-limited search leaned on the hash earlier work left,
    and the weak tiers analysed with their handicap on."""
    session = GameSession()
    for move in chess.pgn.read_game(LATE_GAME_PGN.open()).mainline_moves():
        assert session.submit_move(move.uci()).legal
    with EnginePlayer(move_time=0.05) as player:
        player.set_skill_level(0)
        first = review_game(player, session)
        player.choose_move(GameSession())
        player.set_tier("maximum")
        second = review_game(player, session)
    assert second == first


# --- the review's arithmetic, at the analysis boundary (#454) -----------------


class ScriptedEngine:
    """Engine double: each position's best move and White-POV score, by FEN."""

    def __init__(self, by_fen):
        self.by_fen = by_fen

    def get_best_moves(self, session, n=1):
        uci, score_cp, mate_in = self.by_fen[session.fen()]
        board = chess.Board(session.fen())
        san = board.san(chess.Move.from_uci(uci))
        return [CandidateMove(uci, san, score_cp, mate_in)]

    def evaluate_position(self, session):
        _, score_cp, mate_in = self.by_fen[session.fen()]
        return Evaluation(score_cp, mate_in)


def after(*sans):
    board = chess.Board()
    for san in sans:
        board.push_san(san)
    return board.fen()


def test_the_engines_own_move_costs_nothing():
    # Two searches disagree about e4 by a full pawn; it is still the best move.
    engine = ScriptedEngine(
        {after(): ("e2e4", 120, None), after("e4"): ("e7e5", 20, None)}
    )
    review = review_game(engine, play(GameSession(), "e4"))
    (e4,) = review.moves
    assert (e4.san, e4.best_san) == ("e4", "e4")
    assert e4.cp_loss == 0
    assert e4.classification == "good"
    assert e4.accuracy == 100.0


def test_a_missed_mate_costs_a_bounded_loss():
    # White had mate in 3 and played into mate in 2 against: on the raw mate
    # scale that is ~200 000 centipawns, which is not a number anyone can use.
    engine = ScriptedEngine(
        {after(): ("d2d4", None, 3), after("e4"): ("e7e5", None, -2)}
    )
    (e4,) = review_game(engine, play(GameSession(), "e4")).moves
    assert e4.cp_loss == 2 * EVAL_CEILING_CP
    assert e4.classification == "blunder"


def test_a_slower_mate_is_not_a_mistake():
    engine = ScriptedEngine(
        {after(): ("d2d4", None, 2), after("e4"): ("e7e5", None, 5)}
    )
    (e4,) = review_game(engine, play(GameSession(), "e4")).moves
    assert e4.cp_loss == 0
    assert e4.classification == "good"


# --- the tool ---------------------------------------------------------------


def test_tool_without_engine_is_an_error():
    registry = build_registry(ToolContext(session=GameSession()))
    result = registry.dispatch("review_game", {})
    assert result["ok"] is False
    assert "engine" in result["error"]


@requires_stockfish
def test_tool_reviews_the_game(engine):
    session = play(GameSession(), *SCHOLARS_MATE)
    registry = build_registry(ToolContext(session=session, engine=engine))
    result = registry.dispatch("review_game", {})
    assert result["ok"] is True
    assert result["plies"] == len(SCHOLARS_MATE)
    assert list(result["accuracy"]) == ["player", "glitch"]
    assert list(result["counts"]) == ["player", "glitch"]
    # 3...Nf6?? is the game's blunder, named as the player would say it.
    assert {
        "move_number": 3,
        "by": "glitch",
        "color": "black",
        "san": "Nf6",
        "classification": "blunder",
    }.items() <= result["critical"][0].items()
    assert result["critical"][0]["best"]
    # The per-ply table is the UI's, never the model's (#288).
    assert "moves" not in result


@requires_stockfish
@pytest.mark.parametrize("player_color", ["white", "black"])
def test_tool_says_whose_moves_they_are_by_person(engine, player_color):
    """The regression (#455): sides keyed "white"/"black" left the 12B to map
    "my"/"your" onto colors, and it answered "what was my worst move?" with
    Glitch's own blunder. Accuracy, counts and each critical move now say
    whose they are, the player's first, for either color the player holds."""
    session = play(GameSession(player_color=player_color), *SCHOLARS_MATE)
    registry = build_registry(ToolContext(session=session, engine=engine))
    result = registry.dispatch("review_game", {})
    review = review_game(engine, session)
    glitch_color = "black" if player_color == "white" else "white"

    assert result["accuracy"] == {
        "player": review.accuracy[player_color],
        "glitch": review.accuracy[glitch_color],
    }
    assert list(result["accuracy"]) == ["player", "glitch"]
    assert result["counts"] == {
        "player": review.counts[player_color],
        "glitch": review.counts[glitch_color],
    }
    assert list(result["counts"]) == ["player", "glitch"]
    # Black's 3...Nf6?? is the blunder: the player's own when they are Black.
    (blunder,) = [m for m in result["critical"] if m["san"] == "Nf6"]
    assert blunder["color"] == "black"
    assert blunder["by"] == ("player" if player_color == "black" else "glitch")
    for move in result["critical"]:
        assert move["by"] == ("player" if move["color"] == player_color else "glitch")


@requires_stockfish
def test_tool_numbers_back_the_honesty_guard(engine):
    # Every cp_loss the narrator is shown is a number the evaluation class
    # accepts, so trimming the table did not strand a figure it may quote.
    session = play(GameSession(), *SCHOLARS_MATE)
    registry = build_registry(ToolContext(session=session, engine=engine))
    result = registry.dispatch("review_game", {})
    numbers = analysis_numbers([{"name": "review_game", "result": result}])
    for move in result["critical"]:
        assert str(move["cp_loss"]) in numbers


@requires_stockfish
def test_tool_with_no_moves_is_an_error(engine):
    registry = build_registry(ToolContext(session=GameSession(), engine=engine))
    assert registry.dispatch("review_game", {})["ok"] is False


# --- critical moves (pure) --------------------------------------------------


def _move(color, cp_loss, classification="mistake", number=1):
    return ReviewedMove(
        san=f"m{number}",
        uci="a2a3",
        color=color,
        cp_loss=cp_loss,
        classification=classification,
        best_san="b",
        best_uci="a2a4",
        accuracy=50.0,
        move_number=number,
    )


def _review(*moves):
    return GameReview(moves=tuple(moves), accuracy={}, counts={})


def test_critical_moves_are_each_sides_worst_worst_first():
    white = [
        _move("white", cp, number=n) for n, cp in enumerate((90, 400, 150, 60, 300))
    ]
    black = [_move("black", cp, number=n) for n, cp in enumerate((500, 80))]
    picked = critical_moves(_review(*white, *black))
    assert [(m.color, m.cp_loss) for m in picked] == [
        ("black", 500),
        ("white", 400),
        ("white", 300),
        ("white", 150),
        ("black", 80),
    ]


def test_a_lopsided_game_still_names_both_sides():
    # Six worse white moves than black's worst must not crowd black out: a
    # review is asked about from one side.
    white = [_move("white", 900 - n, number=n) for n in range(10)]
    black = [_move("black", 50, number=1)]
    picked = critical_moves(_review(*white, *black))
    assert len(picked) == CRITICAL_PER_COLOR + 1
    assert any(m.color == "black" for m in picked)


def test_good_moves_are_never_critical():
    picked = critical_moves(
        _review(_move("white", 5, "good"), _move("black", 30, "inaccuracy"))
    )
    assert [m.classification for m in picked] == ["inaccuracy"]


def test_a_clean_game_has_no_critical_moves():
    assert critical_moves(_review(_move("white", 0, "good"))) == ()
