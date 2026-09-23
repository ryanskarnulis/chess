"""Offline latency report over a trace file (#317).

    python scripts/latency_report.py turns.jsonl [more.jsonl ...]
        [--since 2026-09-23T00:00] [--experiment NAME] [--manifest ID]
        [--cold-gap-ms 5000] [--json]

Reads the records `CHESSAPP_TRACE_PATH` collects — `turn`, `serving`,
`speech` and the browser's `voice` milestones — and prints, per condition:
sample counts, failures, censored waits, and nearest-rank p50/p95/p99.

The rules it keeps, because a latency report that breaks them is worse than
none:

- **Denominators always.** Every percentile is printed beside its `n`, and a
  percentile the sample cannot support (p95 under 20, p99 under 100) prints
  as `–` rather than as the maximum wearing a percentile's name.
- **Failures and censored waits are counted, never hidden.** A failed call
  has no duration worth ranking and a `late` one only a lower bound, so both
  are left out of the percentiles — and both are printed next to them. A
  browser interaction given up on at its ceiling is counted the same way.
- **Conditions are separated, not averaged.** A model call whose wall clock
  exceeds the server's own account of it by `--cold-gap-ms` or more spent
  that time outside the server — a cold load, or a queue — and is `cold`;
  one the server described is `warm`; one it did not is `unknown`. A turn
  that waited on the mutation lock, or ran while another turn was still in
  flight, is `contended` as well. Each is its own row.
- **Overlap is reported, not summed.** The observe beat runs while Stockfish
  computes; speech is its own request after the turn. Spans are printed side
  by side and never added into a serial total the turn did not spend.
- **Clocks are never mixed.** Client milestones are offsets in the browser's
  clock from the moment the player stopped speaking (or submitted); server
  records are in the server's. The report derives segments only within one
  clock and joins the two only by id.

A record older than schema 2 is counted and skipped: it has no phase tags to
read, and guessing them is the thing this report replaces.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

SCHEMA = 2
DEFAULT_COLD_GAP_MS = 5_000

# The smallest sample each percentile is reported for. Nearest-rank p95 of 19
# readings is the maximum; so is p99 of 99. Below these the column says so.
_MIN_N = {50: 1, 95: 20, 99: 100}

# Client segments, each the difference of two marks in the browser's one clock
# (`None` start = the interaction's own origin: speech end, or submit).
_SEGMENTS: tuple[tuple[str, str | None, str], ...] = (
    ("speech_end→transcript", None, "stt_done"),
    ("command→first_board", "command_sent", "first_board_update"),
    ("command→engine_reply", "command_sent", "engine_reply"),
    ("command→response", "command_sent", "command_done"),
    ("tts_request→audio_ready", "tts_requested", "tts_ready"),
    ("start→first_audio", None, "playback_started"),
    ("playback", "playback_started", "playback_ended"),
    ("start→playback_end", None, "playback_ended"),
)


def percentile(values: Sequence[float], q: int) -> float | None:
    """Nearest-rank percentile, or `None` when `values` is too small to support
    it (see `_MIN_N`)."""
    if len(values) < _MIN_N[q]:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(q / 100 * len(ordered)))
    return ordered[rank - 1]


@dataclass
class Series:
    """One row: the readings that finished, and what did not."""

    values: list[float] = field(default_factory=list)
    failures: int = 0
    censored: int = 0

    @property
    def n(self) -> int:
        return len(self.values) + self.failures + self.censored

    def row(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "ok": len(self.values),
            "failed": self.failures,
            "censored": self.censored,
            "p50": percentile(self.values, 50),
            "p95": percentile(self.values, 95),
            "p99": percentile(self.values, 99),
        }


@dataclass
class Report:
    skipped_old: int = 0
    manifests: dict[str, dict[str, Any]] = field(default_factory=dict)
    calls: dict[tuple[str, str, str], Series] = field(
        default_factory=lambda: defaultdict(Series)
    )
    spans: dict[tuple[str, str, str], Series] = field(
        default_factory=lambda: defaultdict(Series)
    )
    planning: dict[str, Any] = field(default_factory=dict)
    speech: dict[str, Series] = field(default_factory=lambda: defaultdict(Series))
    client: dict[tuple[str, str], Series] = field(
        default_factory=lambda: defaultdict(Series)
    )
    outcomes: Counter[str] = field(default_factory=Counter)
    unjoined_voice: int = 0

    def as_dict(self) -> dict[str, Any]:
        def rows(table: dict[Any, Series]) -> list[dict[str, Any]]:
            return [
                {"key": list(key) if isinstance(key, tuple) else key, **s.row()}
                for key, s in sorted(table.items())
            ]

        return {
            "skipped_old_records": self.skipped_old,
            "manifests": self.manifests,
            "model_calls": rows(self.calls),
            "turn_spans": rows(self.spans),
            "planning": self.planning,
            "speech": rows(self.speech),
            "client_segments": rows(self.client),
            "client_outcomes": dict(self.outcomes),
            "voice_reports_without_a_turn": self.unjoined_voice,
        }


def load(paths: Iterable[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        with path.open() as handle:
            for line in handle:
                if line.strip():
                    records.append(json.loads(line))
    return records


def _ts(record: dict[str, Any]) -> datetime | None:
    try:
        return datetime.fromisoformat(record["ts"])
    except (KeyError, TypeError, ValueError):
        return None


def _call_condition(call: dict[str, Any], cold_gap_ms: int) -> str:
    server_ms = call.get("server_ms")
    if server_ms is None:
        return "unknown"
    return "cold" if call["ms"] - server_ms >= cold_gap_ms else "warm"


def _overlapping(turns: list[dict[str, Any]]) -> set[int]:
    """Indices of turns whose wall-clock interval overlapped another turn's —
    two requests in flight at once, whatever the lock made of it. A turn's
    interval ends at its record's `ts` and began `spans_ms.total` earlier."""
    spans: list[tuple[float, float, int]] = []
    for i, turn in enumerate(turns):
        end = _ts(turn)
        total = (turn.get("spans_ms") or {}).get("total")
        if end is None or total is None:
            continue
        stop = end.timestamp() * 1000
        spans.append((stop - total, stop, i))
    spans.sort()
    hit: set[int] = set()
    latest_end, latest_i = -math.inf, -1
    for start, stop, i in spans:
        if start < latest_end:
            hit.update((i, latest_i))
        if stop > latest_end:
            latest_end, latest_i = stop, i
    return hit


def summarize(
    records: Iterable[dict[str, Any]],
    *,
    since: datetime | None = None,
    experiment: str | None = None,
    manifest: str | None = None,
    cold_gap_ms: int = DEFAULT_COLD_GAP_MS,
) -> Report:
    report = Report()
    current: list[dict[str, Any]] = []
    for record in records:
        if record.get("schema", 1) < SCHEMA:
            report.skipped_old += 1
            continue
        stamp = _ts(record)
        if since is not None and stamp is not None and stamp < since:
            continue
        current.append(record)

    for record in current:
        if record.get("kind") == "serving":
            report.manifests[record["manifest_id"]] = {
                "session": record["session"],
                "revision": record["app"]["revision"],
                **{
                    key: record["server"].get(key)
                    for key in ("source", "model_path", "build_info", "n_ctx")
                },
            }

    def wanted(turn: dict[str, Any]) -> bool:
        label = turn.get("serving") or {}
        if experiment is not None and label.get("experiment") != experiment:
            return False
        return manifest is None or label.get("manifest_id") == manifest

    turns = [r for r in current if r.get("kind") == "turn" and wanted(r)]
    filtered = experiment is not None or manifest is not None
    interactions = {t["interaction_id"] for t in turns if t.get("interaction_id")}
    contended = _overlapping(turns)

    overruns: list[float] = []
    phases_run = 0
    for i, turn in enumerate(turns):
        label = (turn.get("serving") or {}).get("manifest_id") or "-"
        spans = turn.get("spans_ms") or {}
        busy = i in contended or spans.get("queue", 0) > 0
        suffix = "+contended" if busy else ""
        conditions: set[str] = set()
        for call in turn.get("calls", ()):
            condition = _call_condition(call, cold_gap_ms)
            conditions.add(condition)
            series = report.calls[(label, condition + suffix, call["phase"])]
            if call["status"] in ("failed", "bad_args"):
                series.failures += 1
            elif call["status"] == "late":
                series.censored += 1
            else:
                series.values.append(call["ms"])
        turn_condition = (
            "cold"
            if "cold" in conditions
            else "warm"
            if "warm" in conditions
            else "unknown"
            if conditions
            else "no_model"
        ) + suffix
        for span, ms in spans.items():
            report.spans[(turn["route"], turn_condition, span)].values.append(ms)
        report.spans[(turn["route"], turn_condition, "model")].values.append(
            turn.get("model_ms", 0)
        )
        planning = turn.get("planning")
        if planning:
            phases_run += 1
            if planning.get("overrun_ms"):
                overruns.append(planning["overrun_ms"])
    report.planning = {
        "phases": phases_run,
        "overruns": len(overruns),
        "overrun_p50_ms": percentile(overruns, 50),
        "overrun_max_ms": max(overruns) if overruns else None,
    }

    def joined(record: dict[str, Any]) -> bool:
        return not filtered or record.get("interaction_id") in interactions

    for record in current:
        if record.get("kind") == "speech" and joined(record):
            series = report.speech[record["op"]]
            if record.get("status") == "ok":
                series.values.append(record["ms"])
            else:
                series.failures += 1

    for record in current:
        if record.get("kind") != "voice" or not joined(record):
            continue
        if record.get("interaction_id") not in interactions:
            report.unjoined_voice += 1
        report.outcomes[record["outcome"]] += 1
        marks = record.get("marks") or {}
        for name, start, end in _SEGMENTS:
            if end not in marks:
                continue
            origin = 0.0 if start is None else marks.get(start)
            if origin is None:
                continue
            series = report.client[(record["origin"], name)]
            if record.get("censored"):
                series.censored += 1
            else:
                series.values.append(marks[end] - origin)
    return report


def _fmt(value: float | None) -> str:
    return "–" if value is None else f"{value:,.0f}"


_STATS = ["n", "ok", "failed", "censored", "p50", "p95", "p99"]


def _table(title: str, header: Sequence[str], rows: list[list[str]]) -> str:
    """A plain-text table; the trailing `_STATS` columns right-aligned."""
    if not rows:
        return f"{title}\n  (none)\n"
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(header)]
    stats = list(header[-len(_STATS) :]) == _STATS
    numeric = len(header) - len(_STATS) if stats else None

    def line(cells: Sequence[str]) -> str:
        return "  " + "  ".join(
            c.rjust(w) if numeric is not None and i >= numeric else c.ljust(w)
            for i, (c, w) in enumerate(zip(cells, widths, strict=True))
        )

    return "\n".join([title, line(header), *(line(r) for r in rows)]) + "\n"


def _stat_cells(series: Series) -> list[str]:
    row = series.row()
    return [
        str(row["n"]),
        str(row["ok"]),
        str(row["failed"]),
        str(row["censored"]),
        _fmt(row["p50"]),
        _fmt(row["p95"]),
        _fmt(row["p99"]),
    ]


def render(report: Report) -> str:
    out = []
    if report.skipped_old:
        out.append(f"skipped {report.skipped_old} record(s) older than schema 2\n")
    out.append(
        _table(
            "serving manifests",
            ["manifest", "revision", "source", "model_path", "build"],
            [
                [
                    mid,
                    str(m["revision"])[:12],
                    str(m["source"]),
                    str(m["model_path"] or "-").rsplit("/", 1)[-1],
                    str(m["build_info"] or "-"),
                ]
                for mid, m in sorted(report.manifests.items())
            ],
        )
    )
    out.append(
        _table(
            "model calls, ms (wall clock; failed and late calls counted, not ranked)",
            ["manifest", "condition", "phase", *_STATS],
            [[*key, *_stat_cells(s)] for key, s in sorted(report.calls.items())],
        )
    )
    out.append(
        _table(
            "turn spans, ms (overlapping phases side by side, never summed)",
            ["route", "condition", "span", *_STATS],
            [[*key, *_stat_cells(s)] for key, s in sorted(report.spans.items())],
        )
    )
    p = report.planning
    out.append(
        f"planning phases {p.get('phases', 0)}, deadline overruns "
        f"{p.get('overruns', 0)} (p50 {_fmt(p.get('overrun_p50_ms'))} ms, "
        f"max {_fmt(p.get('overrun_max_ms'))} ms)\n"
    )
    out.append(
        _table(
            "speech service, ms (server clock)",
            ["op", *_STATS],
            [[op, *_stat_cells(s)] for op, s in sorted(report.speech.items())],
        )
    )
    out.append(
        _table(
            "player milestones, ms (browser clock; censored = given up on)",
            ["origin", "segment", *_STATS],
            [[*key, *_stat_cells(s)] for key, s in sorted(report.client.items())],
        )
    )
    if report.outcomes:
        outcomes = ", ".join(f"{k} {v}" for k, v in sorted(report.outcomes.items()))
        out.append(f"interaction outcomes: {outcomes}\n")
    if report.unjoined_voice:
        out.append(
            f"{report.unjoined_voice} browser report(s) name no traced turn "
            "(nothing transcribed, or the turn was not traced)\n"
        )
    return "\n".join(out)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--since", type=datetime.fromisoformat)
    parser.add_argument("--experiment")
    parser.add_argument("--manifest")
    parser.add_argument("--cold-gap-ms", type=int, default=DEFAULT_COLD_GAP_MS)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    since = args.since
    if since is not None and since.tzinfo is None:
        since = since.astimezone()
    report = summarize(
        load(args.paths),
        since=since,
        experiment=args.experiment,
        manifest=args.manifest,
        cold_gap_ms=args.cold_gap_ms,
    )
    if args.json:
        json.dump(report.as_dict(), sys.stdout, indent=2, default=str)
        print()
    else:
        print(render(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
