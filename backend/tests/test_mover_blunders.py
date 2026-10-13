"""`scripts/mover_blunders.py`: the mover's blunder suite and its history (#495).

The run needs the GPU and Stockfish; everything it decides is tested here off
both: a move judged against the best one (a loss, a blunder at the line), the
corpus checked in (every position legal, the right side to move, undecided),
positions harvested from a game (blunder and free tags, decided positions
left out), arms interleaved per position, a mover failure counted apart and
never judged, a crashed run resumed without mixing, and the history line and
table.
"""

from __future__ import annotations

import json

import chess
import pytest

from chessapp.mover import Mover
from chessapp.profiles import DEFAULT_PROFILE
from chessapp.provider import ProviderError
from fakes import ScriptedProvider, tool_calls_turn
from mover_blunders import (
    BLUNDER_CP,
    CLAMP_CP,
    CORPUS,
    DECIDED_CP,
    HEADER,
    SAMPLE,
    TAG_BLUNDER,
    TAG_FREE,
    Arm,
    Verdict,
    _final_cp,
    ask,
    clamp,
    corpus_sha,
    harvest_game,
    judge_move,
    load_corpus,
    main,
    merge_corpus,
    parse_arm,
    pending,
    rates,
    read_games,
    run,
    schedule,
    session_for,
    summarize,
    trend,
)

START = chess.STARTING_FEN


class TableJudge:
    """Scores from tables: `best` by FEN (else the first legal move at 0),
    `after` by (FEN, UCI) (else 0). Records what it was asked."""

    def __init__(self, best=None, after=None):
        self.best_table = best or {}
        self.after_table = after or {}
        self.after_asked: list[str] = []

    def best(self, board):
        if board.fen() in self.best_table:
            return self.best_table[board.fen()]
        return Verdict(next(iter(board.legal_moves)).uci(), 0)

    def after(self, board, move):
        self.after_asked.append(move.uci())
        return self.after_table.get((board.fen(), move.uci()), 0)


def _play(move: str):
    return tool_calls_turn(("play_move", {"move": move, "intent": "testing"}))


def _arm(*turns, label="m") -> Arm:
    return Arm(
        label=label,
        model="m",
        profile_source="built in",
        profile_sha="abc",
        mover_settings={"max_tokens": 4096},
        mover=Mover(ScriptedProvider(*turns), DEFAULT_PROFILE),
    )


def _position(pid="p1", moves=(), tags=(TAG_BLUNDER,)):
    return {
        "id": pid,
        "source": "test",
        "root_fen": START,
        "moves": list(moves),
        "side": "white" if len(moves) % 2 == 0 else "black",
        "tags": list(tags),
    }


def _header(arms=("m",), **extra):
    return {
        "kind": HEADER,
        "arms": [
            {
                "label": label,
                "model": "m",
                "profile_source": "built in",
                "profile_sha": "abc",
                "mover": {"max_tokens": 4096},
            }
            for label in arms
        ],
        "corpus_sha": "c0ffee",
        "positions": 2,
        "samples": 1,
        "depth": 18,
        "blunder_cp": BLUNDER_CP,
        "git_sha": "deadbee",
        **extra,
    }


def _sample(arm="m", loss=0, outcome="played", tags=(TAG_BLUNDER,), ms=(1000,)):
    record = {
        "kind": SAMPLE,
        "arm": arm,
        "position": "p1",
        "tags": list(tags),
        "sample": 0,
        "outcome": outcome,
        "attempts": ["played"] if outcome == "played" else ["illegal"] * 3,
        "calls": [{"status": "ok", "ms": m} for m in ms],
    }
    if outcome == "played":
        record |= {"loss_cp": loss, "blunder": loss >= BLUNDER_CP}
    return record


# -- judging -----------------------------------------------------------------


def test_the_best_move_loses_nothing_and_is_not_re_scored():
    board = chess.Board()
    judge = TableJudge(best={board.fen(): Verdict("e2e4", 30)})
    judged = judge_move(judge, board, chess.Move.from_uci("e2e4"))
    assert judged == {
        "best_uci": "e2e4",
        "best_san": "e4",
        "best_cp": 30,
        "played_cp": 30,
        "loss_cp": 0,
        "blunder": False,
    }
    # Scoring it again at the same depth could only add noise.
    assert judge.after_asked == []


@pytest.mark.parametrize(
    ("played_cp", "blunder"),
    [(30 - BLUNDER_CP + 1, False), (30 - BLUNDER_CP, True)],
)
def test_a_blunder_gives_up_the_line_or_more(played_cp, blunder):
    board = chess.Board()
    judge = TableJudge(
        best={board.fen(): Verdict("e2e4", 30)},
        after={(board.fen(), "f2f3"): played_cp},
    )
    judged = judge_move(judge, board, chess.Move.from_uci("f2f3"))
    assert judged["loss_cp"] == 30 - played_cp
    assert judged["blunder"] is blunder


def test_a_move_the_judge_scores_above_the_best_loses_nothing():
    board = chess.Board()
    judge = TableJudge(
        best={board.fen(): Verdict("e2e4", 30)},
        after={(board.fen(), "d2d4"): 45},
    )
    assert judge_move(judge, board, chess.Move.from_uci("d2d4"))["loss_cp"] == 0


def test_mate_scores_are_clamped():
    assert clamp(100_000) == CLAMP_CP
    assert clamp(-100_000) == -CLAMP_CP
    assert clamp(42) == 42


def test_a_finished_position_scores_for_the_side_that_just_moved():
    mated = chess.Board()
    for san in ("f3", "e5", "g4", "Qh4#"):
        mated.push_san(san)
    assert _final_cp(mated) == CLAMP_CP
    stalemate = chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
    assert _final_cp(stalemate) == 0


# -- the corpus --------------------------------------------------------------


def test_every_corpus_position_is_legal_undecided_and_its_sides_turn():
    positions = load_corpus(CORPUS)
    assert len({p["id"] for p in positions}) == len(positions)
    for position in positions:
        session = session_for(position)
        assert not session.is_game_over(), position["id"]
        assert session.turn == position["side"], position["id"]
        assert set(position["tags"]) <= {TAG_BLUNDER, TAG_FREE}, position["id"]
        assert position["tags"], position["id"]
        assert abs(position["best_cp"]) <= DECIDED_CP, position["id"]


def test_the_corpus_keeps_the_real_games_misreads():
    # Game 927c238e (2026-10-12), the game that started #494: the knight and
    # the queen given for a defended pawn on d4.
    by_id = {p["id"]: p for p in load_corpus(CORPUS)}
    assert by_id["g927c-13b"]["played"] == "Nxd4"
    assert by_id["g927c-17b"]["played"] == "Qxd4"


def test_a_position_s_session_holds_its_game():
    session = session_for(_position(moves=["e2e4", "e7e5"]))
    assert session.move_history() == ["e4", "e5"]
    assert session.turn == "white"


def test_a_position_with_an_illegal_move_is_refused():
    with pytest.raises(ValueError, match="e2e5"):
        session_for(_position(moves=["e2e5"]))


def test_the_corpus_digest_ignores_notes_and_order():
    a, b = _position("a", ["e2e4"]), _position("b", ["d2d4"])
    noted = {**a, "played": "e5", "source": "elsewhere", "tags": [TAG_FREE]}
    assert corpus_sha([a, b]) == corpus_sha([b, noted])
    assert corpus_sha([a, b]) != corpus_sha([a, _position("b", ["c2c4"])])


def test_merging_skips_a_position_already_held_however_it_was_reached():
    held = _position("a", ["g1f3", "g8f6", "b1c3"])
    transposed = _position("b", ["b1c3", "g8f6", "g1f3"])
    fresh = _position("c", ["e2e4"])
    merged, added = merge_corpus([held], [transposed, fresh])
    assert [p["id"] for p in merged] == ["a", "c"]
    assert added == 1


# -- harvesting --------------------------------------------------------------


def _fools_mate_judge():
    """White's f3 and g4 each give up 200; Black's e5 is best throughout."""
    board = chess.Board()
    best, after = {}, {}
    for uci, best_uci, best_cp, played_cp in (
        ("f2f3", "e2e4", 30, -170),
        ("e7e5", "e7e5", 170, 170),
        ("g2g4", "e2e4", -170, -370),
    ):
        best[board.fen()] = Verdict(best_uci, best_cp)
        after[(board.fen(), uci)] = played_cp
        board.push_uci(uci)
    return TableJudge(best, after)


def test_harvest_tags_a_blunder_and_the_free_position_after_it():
    found = harvest_game(
        _fools_mate_judge(),
        START,
        ["f2f3", "e7e5", "g2g4"],
        sides={chess.WHITE, chess.BLACK},
        source_id="fm",
        source="test",
    )
    assert [(p["id"], p["tags"]) for p in found] == [
        ("fm-1w", [TAG_BLUNDER]),
        ("fm-1b", [TAG_FREE]),
        ("fm-2w", [TAG_BLUNDER]),
    ]
    assert found[1]["moves"] == ["f2f3"]
    assert found[1]["side"] == "black"
    assert (found[2]["played"], found[2]["best"]) == ("g4", "e4")


def test_harvest_never_calls_a_fallback_move_his_blunder():
    found = harvest_game(
        _fools_mate_judge(),
        START,
        ["f2f3", "e7e5", "g2g4"],
        sides={chess.WHITE},
        source_id="fm",
        source="test",
        not_his={2},
    )
    assert [p["id"] for p in found] == ["fm-1w"]


def test_harvest_keeps_only_the_sides_asked():
    found = harvest_game(
        _fools_mate_judge(),
        START,
        ["f2f3", "e7e5", "g2g4"],
        sides={chess.BLACK},
        source_id="fm",
        source="test",
    )
    assert [p["id"] for p in found] == ["fm-1b"]


def test_harvest_leaves_out_decided_positions():
    board = chess.Board()
    judge = TableJudge(
        best={board.fen(): Verdict("e2e4", DECIDED_CP + 1)},
        after={(board.fen(), "f2f3"): -500},
    )
    found = harvest_game(
        judge, START, ["f2f3"], sides={chess.WHITE}, source_id="x", source="t"
    )
    assert found == []


def test_harvest_with_a_cap_keeps_the_biggest_swings_in_game_order():
    found = harvest_game(
        _fools_mate_judge(),
        START,
        ["f2f3", "e7e5", "g2g4"],
        sides={chess.WHITE},
        source_id="fm",
        source="test",
        max_positions=1,
    )
    assert [p["id"] for p in found] == ["fm-1w"]


def test_games_are_read_from_pgn_sessions_and_strength_reports(tmp_path):
    pgn = tmp_path / "g.pgn"
    pgn.write_text("1. e4 e5 2. Nf3 *\n\n1. d4 *\n")
    assert read_games(pgn) == [
        (START, ["e2e4", "e7e5", "g1f3"], frozenset()),
        (START, ["d2d4"], frozenset()),
    ]
    live = tmp_path / "live.json"
    live.write_text(json.dumps({"session": {"root_fen": START, "moves": ["c2c4"]}}))
    assert read_games(live) == [(START, ["c2c4"], frozenset())]
    report = tmp_path / "strength.jsonl"
    game = {
        "kind": "strength_game",
        "colour": "black",
        "sans": ["e4", "e5", "Nf3", "Nc6"],
        "moves": [{"chooser": "glitch"}, {"chooser": "fallback"}],
    }
    report.write_text(
        json.dumps({"kind": "strength_header", "model": "m"})
        + "\n"
        + json.dumps(game)
        + "\n"
    )
    # Black's second move (ply 3) was the fallback's.
    assert read_games(report) == [
        (START, ["e2e4", "e7e5", "g1f3", "b8c6"], frozenset({3}))
    ]


# -- asking ------------------------------------------------------------------


def test_arms_are_interleaved_per_position_and_take_turns_going_first():
    positions = [_position("p1"), _position("p2")]
    assert schedule(positions, ["a", "b"], 2) == [
        (0, "p1", "a"),
        (0, "p1", "b"),
        (0, "p2", "b"),
        (0, "p2", "a"),
        (1, "p1", "b"),
        (1, "p1", "a"),
        (1, "p2", "a"),
        (1, "p2", "b"),
    ]


def test_a_sample_asks_the_mover_and_judges_its_move():
    position = _position(moves=["e2e4"])
    board = chess.Board(session_for(position).fen())
    judge = TableJudge(
        best={board.fen(): Verdict("c7c5", 20)},
        after={(board.fen(), "f7f6"): -160},
    )
    record = ask(_arm(_play("Ke9"), _play("f6")), position, 0, judge)
    assert record["outcome"] == "played"
    assert (record["san"], record["uci"]) == ("f6", "f7f6")
    assert record["attempts"] == ["illegal", "played"]
    assert (record["best_san"], record["loss_cp"], record["blunder"]) == (
        "c5",
        180,
        True,
    )
    assert len(record["calls"]) == 2


def test_a_failed_mover_is_counted_apart_and_never_judged():
    judge = TableJudge()
    record = ask(_arm(ProviderError("down")), _position(), 0, judge)
    assert record["outcome"] == "provider"
    assert "loss_cp" not in record
    assert judge.after_asked == []


def test_a_run_writes_every_sample_and_a_rerun_adds_nothing(tmp_path):
    report = tmp_path / "r.jsonl"
    positions = [_position("p1"), _position("p2", ["e2e4"])]
    header = _header()
    said: list[str] = []
    run(
        header,
        report,
        positions,
        [_arm(_play("e4"), _play("e5"))],
        TableJudge(),
        say=said.append,
    )
    lines = [json.loads(line) for line in report.read_text().splitlines()]
    assert [line["kind"] for line in lines] == [HEADER, SAMPLE, SAMPLE]
    assert len(said) == 2
    run(header, report, positions, [_arm()], TableJudge(), say=said.append)
    assert len(report.read_text().splitlines()) == 3


def test_a_crashed_run_resumes_with_only_the_missing_samples():
    positions = [_position("p1"), _position("p2")]
    records = [_header(), {**_sample(), "position": "p1"}]
    assert pending(records, _header(), positions) == [(0, "p2", "m")]


@pytest.mark.parametrize(
    "change",
    [
        {"corpus_sha": "other"},
        {"samples": 3},
        {"depth": 12},
        {"arms": _header(arms=("m", "n"))["arms"]},
    ],
)
def test_a_report_from_another_run_is_refused(change):
    with pytest.raises(SystemExit, match="use a new --report"):
        pending([_header()], _header(**change), [_position()])


# -- reading -----------------------------------------------------------------


def test_rates_read_blunders_losses_fallbacks_and_time():
    samples = [
        _sample(loss=0, ms=(1000,)),
        _sample(loss=200, tags=(TAG_FREE,), ms=(3000,)),
        _sample(loss=40, ms=(2000,)),
        _sample(outcome="retries", ms=(1000, 1000, 1000)),
    ]
    read = rates(samples)
    assert (read["samples"], read["moves"], read["blunders"]) == (4, 3, 1)
    assert read["blunder_rate"] == 0.333
    assert read["blunder_low"] < 0.333 < read["blunder_high"]
    assert read["mean_loss_cp"] == 80.0
    assert read["median_loss_cp"] == 40
    assert read["by_tag"] == {
        TAG_BLUNDER: {"moves": 2, "blunders": 0, "mean_loss_cp": 20.0},
        TAG_FREE: {"moves": 1, "blunders": 1, "mean_loss_cp": 200.0},
    }
    assert (read["fallbacks"], read["fallback_kinds"]) == (1, {"retries": 1})
    assert (read["attempts"], read["illegal_attempts"]) == (6, 3)
    assert read["seconds_per_move_p50"] == 2.5
    assert read["seconds_per_move_p90"] == 3.0


def test_a_run_with_no_moves_reads_without_dividing_by_zero():
    read = rates([_sample(outcome="deadline")])
    assert read["blunder_rate"] is None
    assert read["mean_loss_cp"] is None


def test_the_history_has_one_line_per_arm_and_the_table_names_them():
    records = [
        _header(arms=("plain", "facts")),
        _sample(arm="plain", loss=300),
        _sample(arm="facts", loss=10),
    ]
    lines = summarize(records, label="baseline", date="2026-10-12")
    assert [(line["arm"], line["blunders"]) for line in lines] == [
        ("plain", 1),
        ("facts", 0),
    ]
    assert lines[0]["corpus_sha"] == "c0ffee"
    assert lines[0]["profile_sha"] == "abc"
    table = trend(lines)
    assert "1/1 100%" in table
    assert "0/1 0%" in table
    assert "[facts]" in table


def test_a_report_without_a_header_is_refused():
    with pytest.raises(SystemExit, match="no header"):
        summarize([_sample()], label="x", date="2026-10-12")


def test_append_then_trend_round_trip(tmp_path, capsys):
    report, history = tmp_path / "r.jsonl", tmp_path / "h.jsonl"
    records = [_header(), _sample(loss=10)]
    report.write_text("".join(json.dumps(r) + "\n" for r in records))
    assert (
        main(["append", str(report), "--label", "base", "--history", str(history)]) == 0
    )
    assert main(["trend", "--history", str(history), "--model", "m"]) == 0
    out = capsys.readouterr().out
    assert out.count("base") == 2


@pytest.mark.parametrize(
    ("text", "parsed"),
    [
        ("gemma-4-26b-a4b", ("gemma-4-26b-a4b", "gemma-4-26b-a4b", None)),
        ("plain=gemma-4-12b", ("plain", "gemma-4-12b", None)),
    ],
)
def test_an_arm_is_a_model_or_a_labelled_model(text, parsed):
    assert parse_arm(text) == parsed


def test_an_arm_can_carry_its_own_profile_file():
    label, model, path = parse_arm("facts=gemma-4-26b-a4b@/tmp/facts.toml")
    assert (label, model, str(path)) == ("facts", "gemma-4-26b-a4b", "/tmp/facts.toml")
