"""Aggregate the eval harness's per-block JSONL reports into one arm table.

`eval_campaign.sh` runs the suite in alternating blocks of five between two
trees on one server and writes one `CHESSAPP_EVAL_REPORT` file per block. This
joins them: per scenario, per arm, the passes over the runs and the block
sequence, so the record reads "17/20 (5, 3, 4, 5)" the way `docs/agent-evals.md`
has always written it.

    python scripts/campaign_report.py a=out/block-1-a.jsonl b=out/block-1-b.jsonl ...

Pure aggregation over `scenario` records (`evalstats.scenario_record`); header
and suite lines are skipped. Tested off the GPU in `tests/test_probe_planner.py`.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any


def aggregate(
    reports: Iterable[tuple[str, Iterable[dict[str, Any]]]],
) -> dict[str, dict[str, dict[str, Any]]]:
    """`reports` is (arm, records) per block file, in the order the blocks ran.
    Returns {scenario: {arm: {"passed", "runs", "blocks"}}}."""
    table: dict[str, dict[str, dict[str, Any]]] = {}
    for arm, records in reports:
        for record in records:
            if record.get("kind") != "scenario":
                continue
            cell = table.setdefault(record["scenario"], {}).setdefault(
                arm, {"passed": 0, "runs": 0, "blocks": []}
            )
            cell["passed"] += int(record["passed"])
            cell["runs"] += int(record["runs"])
            # One harness report may itself hold several sampling blocks
            # (escalation); keep each block's passes as the harness scored it
            # (`RateResult.blocks` is (passed, runs) per block).
            cell["blocks"].extend(int(block[0]) for block in record.get("blocks", []))
    return table


def format_table(
    table: dict[str, dict[str, dict[str, Any]]], arms: Sequence[str]
) -> str:
    lines = [
        "scenario | " + " | ".join(arms),
        "--- | " + " | ".join("---" for _ in arms),
    ]
    for scenario in sorted(table):
        cells = []
        for arm in arms:
            cell = table[scenario].get(arm)
            if cell is None:
                cells.append("—")
            else:
                blocks = ", ".join(str(b) for b in cell["blocks"])
                cells.append(f"{cell['passed']}/{cell['runs']} ({blocks})")
        lines.append(f"`{scenario}` | " + " | ".join(cells))
    return "\n".join(lines)


def _read(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main(argv: Sequence[str] | None = None) -> int:
    specs = list(sys.argv[1:] if argv is None else argv)
    if not specs:
        print(__doc__)
        return 2
    reports = []
    arms: list[str] = []
    for spec in specs:
        arm, eq, path = spec.partition("=")
        if not eq:
            raise SystemExit(f"expected arm=path, got {spec!r}")
        if arm not in arms:
            arms.append(arm)
        reports.append((arm, _read(Path(path))))
    print(format_table(aggregate(reports), arms))
    return 0


if __name__ == "__main__":
    sys.exit(main())
