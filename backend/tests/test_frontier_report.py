"""`scripts/frontier_report.py` — the frontier tier's history (#318).

The history is how a hard scenario's number is watched over months, so the
thing that writes it and the thing that reads it are tested off the GPU: the
summary a run becomes, the mark that says a scenario moved (only when the
intervals separate), and the held-out-only graduation rule.
"""

from __future__ import annotations

import json
from pathlib import Path

import campaign_report
import frontier_report
from frontier_report import GRADUATION_RATE, graduates, moved, summarize, trend


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


def line(split, cells, date="2026-10-01", sha="s"):
    return {
        "date": date,
        "git_sha": sha,
        "split": split,
        "scenarios": {
            name: {"tier": 1, "passed": p, "runs": 10, "rubric": p / 10}
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


def test_a_mark_needs_the_intervals_to_separate():
    ten = {"passed": 10, "runs": 10}
    assert moved({"passed": 0, "runs": 10}, ten) == "▲"
    assert moved(ten, {"passed": 0, "runs": 10}) == "▼"
    # 5/10 → 8/10 looks like progress and is not evidence of any.
    assert moved({"passed": 5, "runs": 10}, {"passed": 8, "runs": 10}) == ""


def test_graduation_reads_the_last_two_heldout_runs_only():
    history = [
        line("heldout", {"solved": 9, "once": 3, "dev_only": 2}),
        line("dev", {"dev_only": 10}),
        line("heldout", {"solved": 8, "once": 10, "dev_only": 2}),
        line("dev", {"dev_only": 10}),
    ]

    assert graduates(history) == ["solved"]
    assert GRADUATION_RATE == 0.8


def test_one_heldout_run_is_never_enough():
    assert graduates([line("heldout", {"x": 10})]) == []


def test_the_trend_table_marks_moves_and_candidates():
    history = [
        line("heldout", {"x": 0, "y": 9}, date="d1"),
        line("heldout", {"x": 10, "y": 9}, date="d2"),
    ]

    table = trend(history, "heldout")

    assert "10/10 (1.00)▲" in table
    assert table.splitlines()[0].endswith("gate? |")
    y_row = next(row for row in table.splitlines() if "`y`" in row)
    assert y_row.endswith("| yes |")
    assert trend(history, "dev") == "no dev runs recorded"


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
