"""Speech accuracy over a trace file: how often what Glitch says is true (#367).

    python scripts/speech_report.py turns.jsonl [more.jsonl ...]
        [--since 2026-09-26T00:00] [--route brain] [--json]

    # the deployed app's trace
    docker exec chess-app-1 cat /data/saves/turns.jsonl > /tmp/turns.jsonl
    python scripts/speech_report.py /tmp/turns.jsonl

Re-judges every turn's words against the facts it had (`chessapp.speech_accuracy`)
and prints accuracy overall and per family: claims made, claims backed, and
every unbacked line quoted with the turn it came from. Nothing here is a gate;
it is the number a prompt, context or model change is compared on.

Two readings keep it honest (`docs/speech-accuracy.md`):

- **Unscored is not wrong.** A family the scorer cannot score reliably is
  listed with its claim count and left out of the accuracy, and so is every
  family an older record (schema 1 or 2) cannot decide. The denominator is
  always printed.
- **Legacy records are counted apart.** How many of the scored turns predate
  schema 3 is printed beside the total, because their draft is recovered from
  the commentary rather than read from `draft`.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from chessapp.speech_accuracy import UNSCORED, Tally, tally


def select(
    records: Iterable[dict[str, Any]],
    *,
    since: str | None = None,
    route: str | None = None,
) -> list[dict[str, Any]]:
    """The records a report covers: turns at or after `since` (an ISO prefix
    compared as text, like the trace's own `ts`), on `route` if given."""
    chosen = []
    for record in records:
        if since and str(record.get("ts", "")) < since:
            continue
        if route and record.get("route") != route:
            continue
        chosen.append(record)
    return chosen


def _percent(backed: int, made: int) -> str:
    return f"{backed / made:.1%}" if made else "—"


def render(result: Tally) -> str:
    """The report as Markdown: the headline, the per-family table, and the
    unbacked lines."""
    summary = result.as_dict()
    lines = [
        "# Speech accuracy",
        "",
        f"**{_percent(summary['backed'], summary['made'])}** — "
        f"{summary['backed']}/{summary['made']} claims backed, over "
        f"{summary['turns']} turns ({summary['legacy_turns']} legacy records).",
        "",
        f"Reply said: {summary['replies']['announced']}/"
        f"{summary['replies']['owed']} turns that owed the engine's move in "
        "words named it (#365).",
        "",
        "| family | made | backed | unbacked | accuracy |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, counts in summary["families"].items():
        made, backed = counts["made"], counts["backed"]
        lines.append(
            f"| `{name}` | {made} | {backed} | {made - backed} | "
            f"{_percent(backed, made)} |"
        )
    if summary["unscored"]:
        lines += [
            "",
            "## Unscored",
            "",
            "Counted, never held against the model: the family is unscored "
            "everywhere (reason given) or on legacy records only.",
            "",
            "| family | made | why |",
            "| --- | ---: | --- |",
        ]
        for name, counts in summary["unscored"].items():
            why = UNSCORED.get(name, "not decidable from a legacy record")
            lines.append(f"| `{name}` | {counts['made']} | {why} |")
    lines += ["", "## Unbacked lines", ""]
    if not result.unbacked:
        lines.append("None.")
    for item in result.unbacked:
        lines.append(
            f'- `{item.family}` on {item.said!r} — "{item.sentence}" '
            f"({item.route}, {item.ts[:19]}, asked {item.utterance!r})"
        )
    return "\n".join(lines)


def _read(paths: Sequence[Path]) -> list[dict[str, Any]]:
    records = []
    for path in paths:
        with path.open() as handle:
            records += [json.loads(line) for line in handle if line.strip()]
    return records


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("traces", type=Path, nargs="+")
    parser.add_argument("--since", help="only turns at or after this ISO time")
    parser.add_argument("--route", help="only turns on this route")
    parser.add_argument("--json", action="store_true", help="print the summary dict")
    args = parser.parse_args(argv)

    records = select(_read(args.traces), since=args.since, route=args.route)
    result = tally(records)
    if args.json:
        print(json.dumps(result.as_dict(), indent=2))
    else:
        print(render(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
