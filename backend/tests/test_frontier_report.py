"""`scripts/frontier_report.py` — the frontier tier's history (#318).

The history is how a hard scenario's number is watched over months, so the
thing that writes it and the thing that reads it are tested off the GPU: the
summary a run becomes, the mark that says a scenario moved (only when the
intervals separate, and never across a change to the scenario), the corpus
total, and the held-out-only solved mark.
"""

from __future__ import annotations

import json
from pathlib import Path

import campaign_report
import frontier_report
from frontier_report import (
    CHANGED,
    SOLVED_RATE,
    moved,
    solved,
    summarize,
    total_row,
    trend,
    wordings,
)


def header(**extra):
    return {
        "kind": "frontier_header",
        "git_sha": "abc1234",
        "model": "gemma-4-12b",
        "planner_temperature": 0.3,
        "planner_prompt_sha": "p",
        "narrator_prompt_sha": "n",
        "offer_sha": "o",
        "runs": 10,
        **extra,
    }


def record(name, split, passed, runs=10, tier=1, rubric=0.5):
    return {
        "kind": "frontier",
        "scenario": name,
        "tier": tier,
        "split": split,
        "passed": passed,
        "runs": runs,
        "rubric": rubric,
        "infra": 0,
        "breaches": [],
        "checkpoint_hits": {"a": passed},
        "samples": [{"turns": []}],
    }


def line(split, cells, date="2026-10-01", sha="s", fingerprint=None):
    """A history line; `fingerprint` None is a line from before #339."""
    return {
        "date": date,
        "git_sha": sha,
        "split": split,
        "scenarios": {
            name: {
                "tier": 1,
                "passed": p,
                "runs": 10,
                "rubric": p / 10,
                **({"fingerprint": fingerprint} if fingerprint else {}),
            }
            for name, p in cells.items()
        },
    }


def test_a_run_becomes_one_line_per_split_with_its_configuration():
    lines = summarize(
        [
            header(),
            record("x", "dev", 3),
            {"kind": "suite"},
            record("x", "heldout", 7),
            record("y", "heldout", 10, tier=3),
        ],
        label="after #340",
        date="2026-10-01",
    )

    assert [entry["split"] for entry in lines] == ["dev", "heldout"]
    held = lines[1]
    assert held["label"] == "after #340" and held["git_sha"] == "abc1234"
    assert held["offer_sha"] == "o" and held["runs"] == 10
    assert held["scenarios"]["y"]["tier"] == 3
    assert held["scenarios"]["x"] == {
        "tier": 1,
        "passed": 7,
        "runs": 10,
        "rubric": 0.5,
        "infra": 0,
        "breaches": 0,
        "checkpoint_hits": {"a": 7},
    }
    assert "samples" not in json.dumps(lines), "the history keeps counts, not traces"


def test_a_line_keeps_what_each_number_measured():
    """#339: the models and crutches behind a run (#298 compares them), and
    per scenario the fingerprint and the passes per wording."""
    per_wording = {"a": {"passed": 1, "runs": 1}, "b": {"passed": 0, "runs": 1}}
    lines = summarize(
        [
            header(
                phase_models={"planner": "m", "narrator": "m", "answer": "m"},
                crutches=["c"],
            ),
            {
                **record("x", "heldout", 1, runs=2),
                "fingerprint": "f1",
                "variants": per_wording,
            },
        ],
        label="v2",
        date="2026-10-03",
    )

    (held,) = lines
    assert held["phase_models"] == {"planner": "m", "narrator": "m", "answer": "m"}
    assert held["crutches"] == ["c"]
    assert held["scenarios"]["x"]["fingerprint"] == "f1"
    assert held["scenarios"]["x"]["variants"] == per_wording


def test_a_mark_needs_the_intervals_to_separate():
    ten = {"passed": 10, "runs": 10}
    assert moved({"passed": 0, "runs": 10}, ten) == "▲"
    assert moved(ten, {"passed": 0, "runs": 10}) == "▼"
    # 5/10 → 8/10 looks like progress and is not evidence of any.
    assert moved({"passed": 5, "runs": 10}, {"passed": 8, "runs": 10}) == ""


def test_solved_reads_the_last_two_heldout_runs_only():
    history = [
        line("heldout", {"solved": 9, "once": 3, "dev_only": 2}),
        line("dev", {"dev_only": 10}),
        line("heldout", {"solved": 8, "once": 10, "dev_only": 2}),
        line("dev", {"dev_only": 10}),
    ]

    assert solved(history) == ["solved"]
    assert SOLVED_RATE == 0.8


def test_one_heldout_run_is_never_enough():
    assert solved([line("heldout", {"x": 10})]) == []


def test_solved_needs_both_runs_on_one_fingerprint():
    """A reworded scenario has one run of its new self, not two."""
    history = [
        line("heldout", {"x": 10}),
        line("heldout", {"x": 10}, fingerprint="v2"),
    ]

    assert solved(history) == []
    assert solved([*history, line("heldout", {"x": 9}, fingerprint="v2")]) == ["x"]


def test_the_trend_table_marks_moves_and_solved_scenarios():
    history = [
        line("heldout", {"x": 0, "y": 9}, date="d1"),
        line("heldout", {"x": 10, "y": 9}, date="d2"),
    ]

    table = trend(history, "heldout")

    assert "10/10 (1.00)▲" in table
    assert table.splitlines()[0].endswith("solved |")
    y_row = next(row for row in table.splitlines() if "`y`" in row)
    assert y_row.endswith("| yes |")
    assert trend(history, "dev") == "no dev runs recorded"


def test_a_changed_scenario_is_never_compared_with_its_old_self():
    history = [
        line("heldout", {"x": 0}, date="d1"),
        line("heldout", {"x": 10}, date="d2", fingerprint="v2"),
        line("heldout", {"x": 0}, date="d3", fingerprint="v2"),
    ]

    x_row = next(r for r in trend(history, "heldout").splitlines() if "`x`" in r)

    assert f"| 0/10 (0.00) | 10/10 (1.00){CHANGED} | 0/10 (0.00)▼ |" in x_row


def test_the_total_is_the_whole_run_marked_only_against_the_same_corpus():
    same = [
        line("heldout", {"x": 0, "y": 0}, date="d1", fingerprint="f"),
        line("heldout", {"x": 10, "y": 10}, date="d2", fingerprint="f"),
    ]
    grown = [
        *same,
        line("heldout", {"x": 10, "y": 10, "z": 0}, date="d3", fingerprint="f"),
    ]

    assert total_row(same) == ["0/20 (0.00)", "20/20 (1.00)▲"]
    assert total_row(grown)[-1] == f"20/30 (0.67){CHANGED}"
    assert "| total | 0/20 (0.00) | 20/20 (1.00)▲ |" in trend(same, "heldout")


def test_append_and_trend_round_trip_through_the_cli(tmp_path: Path, capsys):
    report = tmp_path / "run.jsonl"
    report.write_text(
        "\n".join(json.dumps(r) for r in [header(), record("x", "heldout", 4)])
    )
    history = tmp_path / "history.jsonl"

    assert (
        frontier_report.main(
            ["--history", str(history), "append", str(report), "--label", "t"]
        )
        == 0
    )
    assert frontier_report.main(["--history", str(history), "trend"]) == 0

    out = capsys.readouterr().out
    assert "4/10 (0.50)" in out
    assert len(history.read_text().splitlines()) == 1


def test_the_committed_history_reads():
    """The file the docs point at parses, and the baseline is in it."""
    history = [
        json.loads(text)
        for text in frontier_report.HISTORY.read_text().splitlines()
        if text.strip()
    ]
    assert {entry["split"] for entry in history} >= {"dev", "heldout"}
    for split in ("dev", "heldout"):
        assert "`long_session`" in trend(history, split)


def test_the_campaign_table_joins_frontier_records_as_one_block_each():
    table = campaign_report.aggregate(
        [
            ("a", [record("x", "heldout", 3)]),
            ("b", [record("x", "heldout", 7)]),
            ("a", [record("x", "heldout", 5)]),
        ]
    )

    assert table["x"]["a"] == {"passed": 8, "runs": 20, "blocks": [3, 5]}
    assert table["x"]["b"]["blocks"] == [7]


# --- speech accuracy, trended beside the task scores (#367) ---------------------


def speech(backed, made, **families):
    return {
        "turns": 10,
        "legacy_turns": 0,
        "made": made,
        "backed": backed,
        "accuracy": backed / made if made else None,
        "families": families or {"capture": {"made": made, "backed": backed}},
        "unscored": {},
    }


def test_a_splits_speech_is_its_scenarios_merged():
    lines = summarize(
        [
            header(),
            {**record("a", "dev", 5), "speech": speech(3, 4)},
            {**record("b", "dev", 5), "speech": speech(1, 2)},
            record("c", "dev", 5),  # a report written before speech was
        ],
        label="x",
        date="2026-10-01",
    )

    (dev,) = lines
    assert (dev["speech"]["backed"], dev["speech"]["made"]) == (4, 6)
    assert dev["speech"]["families"] == {"capture": {"made": 6, "backed": 4}}


def test_the_trend_carries_a_marked_speech_row():
    history = [
        {**line("dev", {"s": 5}, date="2026-10-01"), "speech": speech(10, 100)},
        {**line("dev", {"s": 5}, date="2026-10-02"), "speech": speech(95, 100)},
        line("dev", {"s": 5}, date="2026-10-03"),
    ]

    table = trend(history, "dev")

    assert "| speech accuracy | 10/100 (10%) | 95/100 (95%)▲ | — |" in table
    assert "| reply said | — | — | — |" in table, "no run recorded it"


def test_the_trend_carries_the_reply_said_row():
    history = [
        {
            **line("dev", {"s": 5}, date="2026-10-01"),
            "speech": {**speech(1, 1), "replies": {"owed": 20, "announced": 2}},
        },
        {
            **line("dev", {"s": 5}, date="2026-10-02"),
            "speech": {**speech(1, 1), "replies": {"owed": 20, "announced": 19}},
        },
    ]

    table = trend(history, "dev")

    assert "| reply said | 2/20 (10%) | 19/20 (95%)▲ |" in table


def test_speech_is_not_compared_across_a_corpus_change():
    """A different corpus asks different things, so it makes different
    claims: its speech rate is not the same quantity as the last run's."""
    history = [
        {**line("dev", {"s": 5}, date="d1"), "speech": speech(10, 100)},
        {**line("dev", {"s": 5, "t": 5}, date="d2"), "speech": speech(95, 100)},
    ]

    table = trend(history, "dev")

    assert f"| speech accuracy | 10/100 (10%) | 95/100 (95%){CHANGED} |" in table


def test_the_speech_flag_prints_the_families(tmp_path: Path, capsys):
    path = tmp_path / "history.jsonl"
    history = [
        {
            **line("dev", {"s": 5}),
            "speech": speech(
                1,
                3,
                capture={"made": 2, "backed": 1},
                move={"made": 1, "backed": 0},
            ),
        }
    ]
    path.write_text("".join(json.dumps(h) + "\n" for h in history))

    frontier_report.main(
        ["--history", str(path), "trend", "--split", "dev", "--speech"]
    )

    out = capsys.readouterr().out
    assert "| `capture` | 1/2 |" in out
    assert "| `move` | 0/1 |" in out


# --- per wording (#339) ----------------------------------------------------------


def test_the_variants_flag_prints_the_latest_run_per_wording(tmp_path: Path, capsys):
    cell = {
        "tier": 2,
        "passed": 1,
        "runs": 2,
        "rubric": 0.5,
        "fingerprint": "f",
        "variants": {
            "plain": {"passed": 1, "runs": 1},
            "noisy": {"passed": 0, "runs": 1},
        },
    }
    path = tmp_path / "history.jsonl"
    path.write_text(
        json.dumps(
            {"date": "d", "git_sha": "s", "split": "heldout", "scenarios": {"x": cell}}
        )
        + "\n"
    )

    frontier_report.main(
        ["--history", str(path), "trend", "--split", "heldout", "--variants"]
    )

    out = capsys.readouterr().out
    assert "| 2 | `x` | `plain` | 1/1 |" in out
    assert "| 2 | `x` | `noisy` | 0/1 |" in out


def test_a_run_from_before_wordings_were_counted_says_so():
    assert (
        wordings([line("heldout", {"x": 3})], "heldout")
        == "the latest heldout run has no per-wording counts"
    )
    assert wordings([], "dev") == "no dev runs recorded"
