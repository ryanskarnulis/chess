"""The frontier tier's history: record a run, and read the trend (#318).

    # after a run with CHESSAPP_EVAL_REPORT=/tmp/frontier.jsonl
    python scripts/frontier_report.py append /tmp/frontier.jsonl --label "after #340"
    python scripts/frontier_report.py trend [--split heldout] [--last 6]

`append` turns one run's report (a `frontier_header` and a `frontier` record
per scenario, `tests/test_agent_frontier.py`) into one summary line per split
in `docs/frontier-history.jsonl`: the configuration shas, and per scenario the
passes, runs, rubric and checkpoint hits. `trend` prints, per split, a
scenario × run table with a mark where the latest run moved against the one
before it, and flags gate candidates.

Two readings keep the table honest (`docs/agent-frontier.md`):

- **A mark means the intervals separated, nothing less.** ▲/▼ appears only
  when the one-sided 95% Wilson intervals of two consecutive runs do not
  overlap. At ten samples that takes a large move, which is the point: runs
  on different days sit on differently-warmed servers, and consecutive
  samples of one prompt are correlated (`docs/agent-evals.md`), so the
  history shows *trend*. A claim that a change moved a scenario is an
  alternating-block A/B (`scripts/eval_campaign.sh --suite frontier`).
- **Graduation reads held-out only.** A scenario is a gate candidate when its
  held-out whole-task rate is at least 0.8 in each of the two most recent
  held-out runs. Dev wordings are what prompts get tuned against, so a dev
  rate is not evidence the task is solved.

Pure functions over dicts, tested off the GPU in `tests/test_frontier_report.py`.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))
from evalstats import wilson_interval  # noqa: E402

HISTORY = Path(__file__).resolve().parents[2] / "docs" / "frontier-history.jsonl"
GRADUATION_RATE = 0.8
GRADUATION_RUNS = 2
_CONFIG_KEYS = (
    "git_sha",
    "model",
    "planner_temperature",
    "planner_prompt_sha",
    "narrator_prompt_sha",
    "offer_sha",
)


def summarize(
    records: Iterable[dict[str, Any]], *, label: str, date: str
) -> list[dict[str, Any]]:
    """One history line per split present in a run's report records.

    A report may hold several headers (a run per split appended to one file);
    each `frontier` record carries its own split, and the configuration comes
    from the header that opened it.
    """
    lines: dict[str, dict[str, Any]] = {}
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
        line["scenarios"][record["scenario"]] = {
            "tier": record["tier"],
            "passed": record["passed"],
            "runs": record["runs"],
            "rubric": None if record["rubric"] is None else round(record["rubric"], 3),
            "infra": record["infra"],
            "breaches": len(record["breaches"]),
            "checkpoint_hits": record["checkpoint_hits"],
        }
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


def graduates(history: Sequence[dict[str, Any]]) -> list[str]:
    """Scenarios at or above the graduation rate in each of the last
    `GRADUATION_RUNS` held-out runs that measured them."""
    heldout = [line for line in history if line["split"] == "heldout"]
    names = {name for line in heldout for name in line["scenarios"]}
    ready = []
    for name in sorted(names):
        cells = [
            line["scenarios"][name] for line in heldout if name in line["scenarios"]
        ]
        recent = cells[-GRADUATION_RUNS:]
        if len(recent) == GRADUATION_RUNS and all(
            c["runs"] and _rate(c) >= GRADUATION_RATE for c in recent
        ):
            ready.append(name)
    return ready


def trend(history: Sequence[dict[str, Any]], split: str, last: int = 6) -> str:
    """The split's table: one column per run (oldest first), `passed/runs
    (rubric)` per cell, the mark against the previous run, and a gate
    candidate column (held-out only)."""
    runs = [line for line in history if line["split"] == split][-last:]
    if not runs:
        return f"no {split} runs recorded"
    ready = set(graduates(history)) if split == "heldout" else set()
    names: list[str] = []
    for line in runs:
        for name in line["scenarios"]:
            if name not in names:
                names.append(name)
    head = ["tier", "scenario"] + [
        f"{line['date']} {line.get('git_sha') or ''}".strip() for line in runs
    ]
    if split == "heldout":
        head.append("gate?")
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
            mark = moved(previous, cell) if previous is not None else ""
            rubric = "—" if cell["rubric"] is None else f"{cell['rubric']:.2f}"
            cells.append(f"{cell['passed']}/{cell['runs']} ({rubric}){mark}")
            previous = cell
        row = [str(tier), f"`{name}`", *cells]
        if split == "heldout":
            row.append("yes" if name in ready else "")
        rows.append("| " + " | ".join(row) + " |")
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
    for split in args.split or ["dev", "heldout"]:
        print(f"\n## {split}\n")
        print(trend(history, split, args.last))
    return 0


if __name__ == "__main__":
    sys.exit(main())
