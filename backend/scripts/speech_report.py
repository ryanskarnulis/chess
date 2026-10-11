"""Speech accuracy over a trace file: how often what Glitch says is true (#367).

    python scripts/speech_report.py turns.jsonl [more.jsonl ...]
        [--since 2026-09-26T00:00] [--route brain] [--model gemma-4-12b]
        [--chooser glitch|fallback|stockfish] [--json]

    # the deployed app's trace
    docker exec chess-app-1 cat /data/saves/turns.jsonl > /tmp/turns.jsonl
    python scripts/speech_report.py /tmp/turns.jsonl

Re-judges every turn's words against the facts it had (`chessapp.speech_accuracy`)
and prints accuracy overall and per family: claims made, claims backed, and
every unbacked line quoted with the turn it came from. Turns that answered a
reply are also split by who chose it (#474): Stockfish, Glitch himself in the
`glitch` tier, or the fallback, whose drafts are printed in full, because
whether he passed a fallback off as his own idea is read by hand. Nothing here
is a gate; it is the number a prompt, context or model change is compared on.

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

from chessapp.speech_accuracy import UNSCORED, Tally, chooser_of, tally

# Who can choose a reply, in the order the report lists them (#474).
CHOOSERS = ("stockfish", "glitch", "fallback")


def select(
    records: Iterable[dict[str, Any]],
    *,
    since: str | None = None,
    route: str | None = None,
    model: str | None = None,
    chooser: str | None = None,
) -> list[dict[str, Any]]:
    """The records a report covers: turns at or after `since` (an ISO prefix
    compared as text, like the trace's own `ts`), on `route` if given, served
    by `model` if given (`serving.model`; one process can serve several since
    the brain setting, #434), answering a reply `chooser` chose if given
    (`speech_accuracy.chooser_of`, #474)."""
    chosen = []
    for record in records:
        if since and str(record.get("ts", "")) < since:
            continue
        if route and record.get("route") != route:
            continue
        if model and (record.get("serving") or {}).get("model") != model:
            continue
        if chooser and chooser_of(record) != chooser:
            continue
        chosen.append(record)
    return chosen


def _percent(backed: int, made: int) -> str:
    return f"{backed / made:.1%}" if made else "—"


def render(result: Tally, records: Sequence[dict[str, Any]] = ()) -> str:
    """The report as Markdown: the headline, the per-family table, the split
    by who chose the reply and the fallback drafts when `records` are given,
    and the unbacked lines."""
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
    lines += _by_chooser(records)
    lines += ["", "## Unbacked lines", ""]
    if not result.unbacked:
        lines.append("None.")
    for item in result.unbacked:
        lines.append(
            f'- `{item.family}` on {item.said!r} — "{item.sentence}" '
            f"({item.route}, {item.ts[:19]}, asked {item.utterance!r})"
        )
    return "\n".join(lines)


def _by_chooser(records: Sequence[dict[str, Any]]) -> list[str]:
    """Speech accuracy per chooser of the reply (#474), and every fallback
    turn's draft for the hand review; nothing when no turn answered a reply
    in the `glitch` tier, so a Stockfish-only report reads as it always did."""
    split = {name: [r for r in records if chooser_of(r) == name] for name in CHOOSERS}
    if not split["glitch"] and not split["fallback"]:
        return []
    lines = [
        "",
        "## By who chose the reply",
        "",
        "| chooser | turns | made | backed | accuracy | reply said |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, chosen in split.items():
        summary = tally(chosen).as_dict()
        lines.append(
            f"| {name} | {summary['turns']} | {summary['made']} | "
            f"{summary['backed']} | {_percent(summary['backed'], summary['made'])} "
            f"| {summary['replies']['announced']}/{summary['replies']['owed']} |"
        )
    lines += [
        "",
        "## Fallback turns, for hand review",
        "",
        "Stockfish played these replies for him; the brief said so. Read each: "
        "did he pass the move off as his own idea?",
        "",
    ]
    if not split["fallback"]:
        lines.append("None.")
    for record in split["fallback"]:
        san = (record.get("evidence") or {}).get("engine_reply_san")
        lines.append(
            f"- {san} ({record.get('route', '')}, {str(record.get('ts', ''))[:19]}): "
            f"{(record.get('draft') or '')!r}"
        )
    return lines


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
    parser.add_argument("--model", help="only turns this brain served")
    parser.add_argument(
        "--chooser", choices=CHOOSERS, help="only turns whose reply this chose"
    )
    parser.add_argument("--json", action="store_true", help="print the summary dict")
    args = parser.parse_args(argv)

    records = select(
        _read(args.traces),
        since=args.since,
        route=args.route,
        model=args.model,
        chooser=args.chooser,
    )
    result = tally(records)
    if args.json:
        print(json.dumps(result.as_dict(), indent=2))
    else:
        print(render(result, records))
    return 0


if __name__ == "__main__":
    sys.exit(main())
