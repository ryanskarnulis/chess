"""Aggregate the eval harness's per-block JSONL reports into one arm table.

`eval_campaign.sh` runs the suite in alternating blocks of five between two
trees on one server and writes one `CHESSAPP_EVAL_REPORT` file per block. This
joins them: per scenario, per arm, the passes over the runs and the block
sequence, so the record reads "17/20 (5, 3, 4, 5)" the way `docs/agent-evals.md`
has always written it.

    python scripts/campaign_report.py a=out/block-1-a.jsonl b=out/block-1-b.jsonl ...

**Paired arms** (#363): a seeded campaign runs sample *j* of block *n* with
the same sampling seed in both arms, and each sample record carries its seed.
Samples of two arms that share (block, seed) are a pair, and each later arm
gets a column against the first — the discordant pairs and the exact McNemar
p (`evalstats.mcnemar_exact`). For a small change most pairs agree, so the
paired p resolves what the raw counts cannot. A gate sample pairs on passing
(`PASS`; an infra death was re-taken and never scored), a frontier sample on
`whole`. Unseeded reports print no paired column. The block number comes from
the report's name (`block-<n>-<arm>.jsonl`).

Pure aggregation over `scenario` records (`evalstats.scenario_record`); header
and suite lines are skipped. Tested off the GPU in `tests/test_probe_planner.py`.
"""

from __future__ import annotations

import json
import re
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))
from evalstats import format_paired, paired_counts  # noqa: E402

# The gate's scored outcomes (`evalstats.Outcome`); anything else was not a
# sample — a re-taken infra death, a harness bug — and pairs with nothing.
_SCORED = {"PASS": True, "FAIL": False, "INCONCLUSIVE": False}
_BLOCK = re.compile(r"block-(\d+)-")


def aggregate(
    reports: Iterable[tuple[str, Iterable[dict[str, Any]]]],
) -> dict[str, dict[str, dict[str, Any]]]:
    """`reports` is (arm, records) per block file, in the order the blocks ran.
    Returns {scenario: {arm: {"passed", "runs", "blocks"}}}."""
    table: dict[str, dict[str, dict[str, Any]]] = {}
    for arm, records in reports:
        for record in records:
            # The gate's `scenario` records and the frontier tier's
            # `frontier` ones (#318) — one sampling block each, as run.
            if record.get("kind") not in ("scenario", "frontier"):
                continue
            cell = table.setdefault(record["scenario"], {}).setdefault(
                arm, {"passed": 0, "runs": 0, "blocks": []}
            )
            cell["passed"] += int(record["passed"])
            cell["runs"] += int(record["runs"])
            # One harness report may itself hold several sampling blocks
            # (escalation); keep each block's passes as the harness scored it
            # (`RateResult.blocks` is (passed, runs) per block).
            blocks = record.get("blocks") or [[record["passed"], record["runs"]]]
            cell["blocks"].extend(int(block[0]) for block in blocks)
    return table


def _sample_pass(record: dict[str, Any], sample: dict[str, Any]) -> bool | None:
    if record.get("kind") == "frontier":
        return bool(sample["whole"]) if "whole" in sample else None
    return _SCORED.get(str(sample.get("outcome")))


def sample_outcomes(
    reports: Iterable[tuple[str, int | None, Iterable[dict[str, Any]]]],
) -> dict[str, dict[str, dict[tuple[int | None, int], bool]]]:
    """`reports` is (arm, block, records) per block file. Returns
    {scenario: {arm: {(block, seed): passed}}} over the seeded, scored samples."""
    outcomes: dict[str, dict[str, dict[tuple[int | None, int], bool]]] = {}
    for arm, block, records in reports:
        for record in records:
            if record.get("kind") not in ("scenario", "frontier"):
                continue
            for sample in record.get("samples") or []:
                seed = sample.get("seed")
                passed = _sample_pass(record, sample)
                if seed is None or passed is None:
                    continue
                outcomes.setdefault(record["scenario"], {}).setdefault(arm, {})[
                    (block, int(seed))
                ] = passed
    return outcomes


def paired(
    outcomes: dict[str, dict[str, dict[tuple[int | None, int], bool]]],
    arms: Sequence[str],
) -> dict[str, dict[str, str]]:
    """{scenario: {arm: verdict}} for each arm after the first, against it,
    over the samples both ran with the same seed in the same block."""
    verdicts: dict[str, dict[str, str]] = {}
    for scenario, by_arm in outcomes.items():
        base = by_arm.get(arms[0], {})
        for arm in arms[1:]:
            other = by_arm.get(arm, {})
            keys = sorted(base.keys() & other.keys(), key=lambda k: (k[0] or 0, k[1]))
            if not keys:
                continue
            _, a_only, b_only, _ = paired_counts(
                [base[k] for k in keys], [other[k] for k in keys]
            )
            verdicts.setdefault(scenario, {})[arm] = format_paired(
                a_only, b_only, len(keys)
            )
    return verdicts


def format_table(
    table: dict[str, dict[str, dict[str, Any]]],
    arms: Sequence[str],
    verdicts: dict[str, dict[str, str]] | None = None,
) -> str:
    verdicts = verdicts or {}
    # A paired column per later arm, and only when some scenario paired.
    against = [arm for arm in arms[1:] if any(arm in row for row in verdicts.values())]
    headers = [*arms, *(f"{arms[0]} vs {arm}" for arm in against)]
    lines = [
        "scenario | " + " | ".join(headers),
        "--- | " + " | ".join("---" for _ in headers),
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
        cells.extend(verdicts.get(scenario, {}).get(arm, "—") for arm in against)
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
    blocks = []
    arms: list[str] = []
    for spec in specs:
        arm, eq, path = spec.partition("=")
        if not eq:
            raise SystemExit(f"expected arm=path, got {spec!r}")
        if arm not in arms:
            arms.append(arm)
        records = _read(Path(path))
        reports.append((arm, records))
        match = _BLOCK.search(Path(path).name)
        blocks.append((arm, int(match.group(1)) if match else None, records))
    verdicts = paired(sample_outcomes(blocks), arms)
    print(format_table(aggregate(reports), arms, verdicts))
    return 0


if __name__ == "__main__":
    sys.exit(main())
