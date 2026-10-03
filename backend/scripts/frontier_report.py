"""The frontier tier's history: record a run, and read the trend (#318).

    # after a run with CHESSAPP_EVAL_REPORT=/tmp/frontier.jsonl
    python scripts/frontier_report.py append /tmp/frontier.jsonl --label "after #340"
    python scripts/frontier_report.py trend [--split heldout] [--last 6] [--variants]

`append` turns one run's report (a `frontier_header` and a `frontier` record
per scenario, `tests/test_agent_frontier.py`) into one summary line per split
in `docs/frontier-history.jsonl`: the configuration (the shas, the phase
models and the crutches), and per scenario its fingerprint, passes, runs,
rubric, checkpoint hits and passes per wording, plus the split's speech
accuracy (#367, `docs/speech-accuracy.md`). `trend` prints, per split, a
scenario × run table with a mark where a run moved against the one before it,
a corpus total row and the speech rows under it, and marks solved scenarios;
`--speech` adds the per-family breakdown and `--variants` the latest run
wording by wording.

Three readings keep the table honest (`docs/agent-frontier.md`):

- **A mark means the intervals separated, nothing less.** ▲/▼ appears only
  when the one-sided 95% Wilson intervals of two consecutive runs do not
  overlap. At five samples a scenario needs a large move to earn one, which
  is the point: runs on different days sit on differently-warmed servers, and
  consecutive samples of one prompt are correlated (`docs/agent-evals.md`),
  so the history shows *trend*. A claim that a change moved a scenario is an
  alternating-block A/B (`scripts/eval_campaign.sh --suite frontier`).
- **A changed scenario is never compared with its old self** (#339). Each
  cell carries its scenario's fingerprint for the split (the wordings, the
  checkpoints and the revision), and a cell whose fingerprint differs from the
  previous one is marked `†` instead of compared. The total and speech rows
  are marked only between two runs of the same corpus: the same scenarios on
  the same fingerprints.
- **Solved reads held-out only, and only informs.** A scenario is solved when
  its held-out whole-task rate is at least 0.8 in each of the two most recent
  held-out runs, both on one fingerprint. Dev wordings are what prompts get
  tuned against, so a dev rate is not evidence the task is solved. A solved
  scenario stays in the corpus: it is where a future model shows it holds the
  line (#339).

Pure functions over dicts, tested off the GPU in `tests/test_frontier_report.py`.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))
from chessapp.speech_accuracy import merge  # noqa: E402
from evalstats import wilson_interval  # noqa: E402

HISTORY = Path(__file__).resolve().parents[2] / "docs" / "frontier-history.jsonl"
SOLVED_RATE = 0.8
SOLVED_RUNS = 2
# The mark of a cell that measured something other than the cell before it:
# a reworded or regraded scenario, or a run over a different corpus.
CHANGED = "†"
_CONFIG_KEYS = (
    "git_sha",
    "model",
    "phase_models",
    "crutches",
    "planner_temperature",
    "planner_prompt_sha",
    "narrator_prompt_sha",
    "offer_sha",
)
# What a scenario's record carries since #339. A line summarised from an
# older report lacks them, and reads as one unchanged series.
_SCENARIO_KEYS = ("fingerprint", "variants")


def summarize(
    records: Iterable[dict[str, Any]], *, label: str, date: str
) -> list[dict[str, Any]]:
    """One history line per split present in a run's report records.

    A report may hold several headers (a run per split appended to one file);
    each `frontier` record carries its own split, and the configuration comes
    from the header that opened it.
    """
    lines: dict[str, dict[str, Any]] = {}
    speeches: dict[str, list[dict[str, Any] | None]] = {}
    header: dict[str, Any] = {}
    for record in records:
        if record.get("kind") == "frontier_header":
            header = record
            continue
        if record.get("kind") != "frontier":
            continue
        split = record["split"]
        line = lines.setdefault(
            split,
            {
                "date": date,
                "label": label,
                **{key: header.get(key) for key in _CONFIG_KEYS},
                "split": split,
                "runs": header.get("runs"),
                "scenarios": {},
            },
        )
        cell = {
            "tier": record["tier"],
            "passed": record["passed"],
            "runs": record["runs"],
            "rubric": None if record["rubric"] is None else round(record["rubric"], 3),
            "infra": record["infra"],
            "breaches": len(record["breaches"]),
            "checkpoint_hits": record["checkpoint_hits"],
        }
        cell.update(
            {key: record[key] for key in _SCENARIO_KEYS if record.get(key) is not None}
        )
        line["scenarios"][record["scenario"]] = cell
        speeches.setdefault(split, []).append(record.get("speech"))
    for split, line in lines.items():
        line["speech"] = merge(speeches[split])
    return list(lines.values())


def _rate(cell: dict[str, Any]) -> float:
    return cell["passed"] / cell["runs"] if cell["runs"] else 0.0


def moved(before: dict[str, Any], after: dict[str, Any]) -> str:
    """▲ or ▼ when the two runs' Wilson intervals do not overlap, else ''."""
    low_b, high_b = wilson_interval(before["passed"], before["runs"])
    low_a, high_a = wilson_interval(after["passed"], after["runs"])
    if low_a > high_b:
        return "▲"
    if high_a < low_b:
        return "▼"
    return ""


def mark(previous: dict[str, Any] | None, cell: dict[str, Any]) -> str:
    """A scenario cell's mark against the one before it: `CHANGED` when the
    scenario's wordings or checkpoints changed in between, since the two
    numbers are of different things; else `moved`'s."""
    if previous is None:
        return ""
    if previous.get("fingerprint") != cell.get("fingerprint"):
        return CHANGED
    return moved(previous, cell)


def corpus(line: dict[str, Any]) -> dict[str, str | None]:
    """What a run measured, scenario by scenario. A total or a speech rate is
    comparable between two runs only when this is the same for both: a
    different corpus asks different things and makes different claims."""
    return {name: cell.get("fingerprint") for name, cell in line["scenarios"].items()}


def _marked_row(
    runs: Sequence[dict[str, Any]],
    read: Callable[[dict[str, Any]], dict[str, Any] | None],
    show: Callable[[dict[str, Any]], str],
) -> list[str]:
    """One cell per run for a whole-run row. `read` gives the run's
    `{"passed", "runs"}` (None prints `—`), `show` prints it, and the mark is
    against the last run that had one: `CHANGED` when the two measured
    different corpora, else `moved`'s."""
    cells = []
    previous: tuple[dict[str, Any], dict[str, Any]] | None = None
    for line in runs:
        cell = read(line)
        if cell is None:
            cells.append("—")
            continue
        if previous is None:
            tag = ""
        elif corpus(previous[0]) != corpus(line):
            tag = CHANGED
        else:
            tag = moved(previous[1], cell)
        cells.append(show(cell) + tag)
        previous = (line, cell)
    return cells


def _percent(cell: dict[str, Any]) -> str:
    return f"{cell['passed']}/{cell['runs']} ({_rate(cell):.0%})"


def _total_cell(line: dict[str, Any]) -> dict[str, Any] | None:
    """A run's whole-task passes over every scenario it measured, and the
    scenarios' mean rubric."""
    cells = list(line["scenarios"].values())
    if not cells:
        return None
    rubrics = [c["rubric"] for c in cells if c["rubric"] is not None]
    return {
        "passed": sum(c["passed"] for c in cells),
        "runs": sum(c["runs"] for c in cells),
        "rubric": sum(rubrics) / len(rubrics) if rubrics else None,
    }


def total_row(runs: Sequence[dict[str, Any]]) -> list[str]:
    """One cell per run: the corpus total, `passed/runs (mean rubric)`. The
    headline number at one sample per wording (#339), where a scenario's own
    five samples rarely separate from anything."""

    def show(cell: dict[str, Any]) -> str:
        rubric = "—" if cell["rubric"] is None else f"{cell['rubric']:.2f}"
        return f"{cell['passed']}/{cell['runs']} ({rubric})"

    return _marked_row(runs, _total_cell, show)


def _speech_cell(speech: dict[str, Any] | None) -> dict[str, Any] | None:
    """A run's speech accuracy as the cell `moved` reads: backed of made."""
    if not speech or not speech["made"]:
        return None
    return {"passed": speech["backed"], "runs": speech["made"]}


def speech_row(runs: Sequence[dict[str, Any]]) -> list[str]:
    """One cell per run: `backed/made (pct)`, marked like the total row.
    `—` for a run recorded before speech was, or one that made no claims."""
    return _marked_row(runs, lambda line: _speech_cell(line.get("speech")), _percent)


def _reply_cell(speech: dict[str, Any] | None) -> dict[str, Any] | None:
    """A run's reply-said rate (#365) as the cell `moved` reads: announced of
    owed. None for a run recorded before it was, or one that owed none."""
    replies = (speech or {}).get("replies") or {}
    if not replies.get("owed"):
        return None
    return {"passed": replies["announced"], "runs": replies["owed"]}


def reply_row(runs: Sequence[dict[str, Any]]) -> list[str]:
    """One cell per run: how often Glitch said the engine's move when a turn
    owed it in words (`docs/speech-accuracy.md`), marked like the total row."""
    return _marked_row(runs, lambda line: _reply_cell(line.get("speech")), _percent)


def speech_families(history: Sequence[dict[str, Any]], split: str, last: int) -> str:
    """The split's speech accuracy per claim family, one column per run."""
    runs = [line for line in history if line["split"] == split][-last:]
    names: list[str] = []
    for line in runs:
        for name in (line.get("speech") or {}).get("families", {}):
            if name not in names:
                names.append(name)
    if not names:
        return f"no {split} speech recorded"
    head = ["family"] + [
        f"{line['date']} {line.get('git_sha') or ''}".strip() for line in runs
    ]
    rows = ["| " + " | ".join(head) + " |", "|" + " --- |" * len(head)]
    for name in names:
        cells = []
        for line in runs:
            counts = (line.get("speech") or {}).get("families", {}).get(name)
            cells.append(f"{counts['backed']}/{counts['made']}" if counts else "—")
        rows.append("| " + " | ".join([f"`{name}`", *cells]) + " |")
    return "\n".join(rows)


def solved(history: Sequence[dict[str, Any]]) -> list[str]:
    """Scenarios at or above `SOLVED_RATE` in each of the last `SOLVED_RUNS`
    held-out runs that measured them, all on one fingerprint. A mark, not a
    verdict: a solved scenario stays in the corpus (#339)."""
    heldout = [line for line in history if line["split"] == "heldout"]
    names = {name for line in heldout for name in line["scenarios"]}
    done = []
    for name in sorted(names):
        cells = [
            line["scenarios"][name] for line in heldout if name in line["scenarios"]
        ]
        recent = cells[-SOLVED_RUNS:]
        if (
            len(recent) == SOLVED_RUNS
            and len({c.get("fingerprint") for c in recent}) == 1
            and all(c["runs"] and _rate(c) >= SOLVED_RATE for c in recent)
        ):
            done.append(name)
    return done


def trend(history: Sequence[dict[str, Any]], split: str, last: int = 6) -> str:
    """The split's table: one column per run (oldest first), `passed/runs
    (rubric)` per cell and its mark against the previous run, the corpus total
    and the speech rows under it, and a solved column (held-out only)."""
    runs = [line for line in history if line["split"] == split][-last:]
    if not runs:
        return f"no {split} runs recorded"
    done = set(solved(history)) if split == "heldout" else set()
    names: list[str] = []
    for line in runs:
        for name in line["scenarios"]:
            if name not in names:
                names.append(name)
    head = ["tier", "scenario"] + [
        f"{line['date']} {line.get('git_sha') or ''}".strip() for line in runs
    ]
    if split == "heldout":
        head.append("solved")
    rows = ["| " + " | ".join(head) + " |", "|" + " --- |" * len(head)]
    for name in sorted(
        names,
        key=lambda n: (
            next(r["scenarios"][n]["tier"] for r in runs if n in r["scenarios"]),
            n,
        ),
    ):
        tier = next(
            r["scenarios"][name]["tier"] for r in runs if name in r["scenarios"]
        )
        cells = []
        previous = None
        for line in runs:
            cell = line["scenarios"].get(name)
            if cell is None:
                cells.append("—")
                continue
            rubric = "—" if cell["rubric"] is None else f"{cell['rubric']:.2f}"
            cells.append(
                f"{cell['passed']}/{cell['runs']} ({rubric}){mark(previous, cell)}"
            )
            previous = cell
        row = [str(tier), f"`{name}`", *cells]
        if split == "heldout":
            row.append("yes" if name in done else "")
        rows.append("| " + " | ".join(row) + " |")
    for label, cells in (
        ("total", total_row(runs)),
        ("speech accuracy", speech_row(runs)),
        ("reply said", reply_row(runs)),
    ):
        row = ["", label, *cells]
        if split == "heldout":
            row.append("")
        rows.append("| " + " | ".join(row) + " |")
    return "\n".join(rows)


def wordings(history: Sequence[dict[str, Any]], split: str) -> str:
    """The split's latest run, wording by wording: which phrasings each
    scenario's number came from (#339)."""
    runs = [line for line in history if line["split"] == split]
    if not runs:
        return f"no {split} runs recorded"
    scenarios = runs[-1]["scenarios"]
    rows = ["| tier | scenario | wording | passed |", "| --- | --- | --- | --- |"]
    for name in sorted(scenarios, key=lambda n: (scenarios[n]["tier"], n)):
        cell = scenarios[name]
        for wording, counts in (cell.get("variants") or {}).items():
            rows.append(
                f"| {cell['tier']} | `{name}` | `{wording}` "
                f"| {counts['passed']}/{counts['runs']} |"
            )
    if len(rows) == 2:
        return f"the latest {split} run has no per-wording counts"
    return "\n".join(rows)


def _read(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--history", type=Path, default=HISTORY)
    sub = parser.add_subparsers(dest="command", required=True)
    add = sub.add_parser("append", help="record a run's report in the history")
    add.add_argument("report", type=Path)
    add.add_argument("--label", required=True)
    add.add_argument("--date", default=datetime.now(UTC).date().isoformat())
    show = sub.add_parser("trend", help="print the history as tables")
    show.add_argument("--split", choices=["dev", "heldout"], action="append")
    show.add_argument("--last", type=int, default=6)
    show.add_argument(
        "--speech", action="store_true", help="add speech accuracy per family"
    )
    show.add_argument(
        "--variants", action="store_true", help="add the latest run per wording"
    )
    args = parser.parse_args(argv)

    if args.command == "append":
        lines = summarize(_read(args.report), label=args.label, date=args.date)
        if not lines:
            print(f"no frontier records in {args.report}", file=sys.stderr)
            return 1
        with args.history.open("a") as handle:
            for line in lines:
                handle.write(json.dumps(line) + "\n")
        print(f"appended {len(lines)} line(s) to {args.history}")
        return 0

    history = _read(args.history) if args.history.exists() else []
    # Held-out first: it is the number of record (#339).
    for split in args.split or ["heldout", "dev"]:
        print(f"\n## {split}\n")
        print(trend(history, split, args.last))
        if args.speech:
            print(f"\n### {split} speech accuracy, per family\n")
            print(speech_families(history, split, args.last))
        if args.variants:
            print(f"\n### {split}, the latest run per wording\n")
            print(wordings(history, split))
    return 0


if __name__ == "__main__":
    sys.exit(main())
