"""Watch every model call's exact context while a game is played (#359).

    python scripts/watch_context.py CONTEXT.jsonl [--trace TURNS.jsonl]
        [--phase planner] [--json] [--from-start]

Tails the file `CHESSAPP_CONTEXT_PATH` writes (`chessapp/context_capture.py`)
and prints each model call as it lands, grouped under its turn: the prompt the
server's chat template rendered — the text the model tokenized — then the raw
response body, thought block and tool-call text included. With `--trace`
pointing at the `CHESSAPP_TRACE_PATH` file, the turn's decisions follow its
calls once the turn record is written: the tools run and what they returned,
the engine's reply, the honesty guard, and what the player was told.

The captured content is printed exactly as captured — no wrapping, no
re-indenting, no pretty-printing. Separators and headers are the only
additions, and the decisions block (read from the trace, which is already a
summary) is the only formatted part.

`--phase` follows one phase (planner, closer, reaction, rewrite, answer);
`--json` prints the request body instead of the rendered prompt; `--from-start`
replays the files from the top instead of waiting for new lines.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, TextIO

# Mirrors of `context_capture.KIND_MODEL_CALL` and `trace.KIND_TURN`, kept as
# literals so the watcher runs from any checkout with nothing but the files.
KIND_MODEL_CALL = "model_call"
KIND_TURN = "turn"
SOURCE_CONTEXT = "context"
SOURCE_TRACE = "trace"

_POLL_S = 0.25


class Watcher:
    """Turns captured lines into terminal output. Pure: `feed` a line from
    either file and it writes to `out` — so the grouping and the join are
    tested without a file, a clock or a GPU."""

    def __init__(
        self, out: TextIO, *, phase: str | None = None, show_json: bool = False
    ) -> None:
        self._out = out
        self._phase = phase
        self._show_json = show_json
        self._group: tuple[Any, Any] | None = None

    def feed(self, line: str, source: str) -> None:
        if not line.strip():
            return
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            self._write(f"!! unreadable {source} line skipped\n")
            return
        if not isinstance(record, dict):
            return
        kind = record.get("kind")
        if source == SOURCE_CONTEXT and kind == KIND_MODEL_CALL:
            if self._phase is None or record.get("phase") == self._phase:
                self._header(record)
                self._write(render_call(record, show_json=self._show_json))
        # A trace record with no `kind` is schema 1, which only held turns.
        elif source == SOURCE_TRACE and kind in (KIND_TURN, None):
            self._header(record)
            self._write(render_turn(record))

    def _header(self, record: dict[str, Any]) -> None:
        group = (record.get("correlation_id"), record.get("turn_id"))
        if group == self._group:
            return
        self._group = group
        self._write(render_header(*group))

    def _write(self, text: str) -> None:
        self._out.write(text)
        self._out.flush()


def render_header(correlation_id: str | None, turn_id: int | None) -> str:
    if correlation_id is None:
        title = "outside a turn"
    else:
        title = f"turn {turn_id} · corr {correlation_id}"
    return f"\n{'═' * 4} {title} {'═' * 4}\n"


def render_call(record: dict[str, Any], *, show_json: bool = False) -> str:
    """One call: a separator, then the captured text exactly as captured."""
    parts = [
        str(record.get("phase", "unknown")),
        f"seq {record.get('seq')}",
        f"{record.get('ms')} ms",
    ]
    status = record.get("status_code")
    parts.append("no response" if status is None else f"HTTP {status}")
    lines = [f"── {' · '.join(parts)} ──\n"]

    template = record.get("template") or {}
    prompt = template.get("prompt")
    if show_json or not isinstance(prompt, str):
        if not show_json:
            reason = template.get("error") or "not captured"
            lines.append(f"▸ prompt unavailable ({reason}); request JSON instead\n")
        lines.append("▸ request\n")
        lines.append(_verbatim(record.get("request")))
    else:
        lines.append("▸ prompt\n")
        lines.append(_verbatim(prompt))

    if record.get("error"):
        lines.append(f"▸ error: {record['error']}\n")
    if record.get("response") is not None:
        lines.append("▸ response\n")
        lines.append(_verbatim(record["response"]))
    return "".join(lines)


def render_turn(record: dict[str, Any]) -> str:
    """The turn's decisions, from its trace record."""
    lines = ["── decisions ──\n"]
    if record.get("utterance"):
        lines.append(f"utterance: {record['utterance']}\n")
    route = record.get("route", "?")
    stop = record.get("stop_reason", "?")
    lines.append(f"route: {route} · stop: {stop}")
    if record.get("provider_failure"):
        lines.append(f" ({record['provider_failure']})")
    lines.append("\n")
    for tool in record.get("tools") or []:
        args = json.dumps(tool.get("args"), ensure_ascii=False)
        result = json.dumps(tool.get("result"), ensure_ascii=False)
        lines.append(f"tool {tool.get('name')} {args} → {result}\n")
    reply = record.get("engine_reply")
    if reply:
        lines.append(f"engine reply: {reply.get('san')}\n")
    if record.get("guarded"):
        claims = ", ".join(record.get("guarded_claims") or [])
        lines.append(
            f"guard: cut [{claims}] · rewrite {record.get('rewrite') or '-'}\n"
        )
        lines.append(f"guard suppressed: {record.get('suppressed', '')}\n")
    if record.get("reaction_late"):
        lines.append("reaction: late\n")
    if record.get("engine_failure"):
        lines.append(f"engine failure: {record['engine_failure']}\n")
    if record.get("error"):
        lines.append(f"error: {record['error']}\n")
    lines.append(f"commentary: {record.get('commentary', '')}\n")
    return "".join(lines)


def _verbatim(text: Any) -> str:
    # Exactly as captured; only a missing final newline is supplied, so the
    # next separator starts on its own line.
    text = "" if text is None else str(text)
    return text if text.endswith("\n") else text + "\n"


class Tail:
    """New complete lines of a file, from wherever reading started.

    A file that does not exist yet is waited for; one that shrank (truncated,
    or replaced) is read again from the top. A line still being written is
    left for the next read.
    """

    def __init__(self, path: Path, *, from_start: bool) -> None:
        self.path = path
        self._offset = 0 if from_start else _size(path)
        self._partial = b""

    def read(self) -> list[str]:
        size = _size(self.path)
        if size < self._offset:
            self._offset, self._partial = 0, b""
        if size == self._offset:
            return []
        with self.path.open("rb") as handle:
            handle.seek(self._offset)
            chunk = handle.read()
        self._offset += len(chunk)
        data = self._partial + chunk
        *complete, self._partial = data.split(b"\n")
        return [line.decode("utf-8", errors="replace") for line in complete]


def _written_at(line: str) -> str:
    """When a line's record was written: a capture's `ended_at`, a trace
    record's `ts` — both ISO-8601 UTC, so they compare as strings. A line
    without one sorts first, and the sort is stable, so each file's own order
    always holds."""
    try:
        record = json.loads(line)
    except json.JSONDecodeError:
        return ""
    if not isinstance(record, dict):
        return ""
    return str(record.get("ended_at") or record.get("ts") or "")


def _size(path: Path) -> int:
    try:
        return path.stat().st_size
    except FileNotFoundError:
        return 0


def main(argv: list[str] | None = None, out: TextIO = sys.stdout) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("context", type=Path, help="the CHESSAPP_CONTEXT_PATH file")
    parser.add_argument("--trace", type=Path, help="the CHESSAPP_TRACE_PATH file")
    parser.add_argument("--phase", help="follow one phase only (e.g. planner)")
    parser.add_argument(
        "--json", action="store_true", help="print the request JSON, not the prompt"
    )
    parser.add_argument(
        "--from-start", action="store_true", help="replay the files from the top"
    )
    parser.add_argument("--once", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    watcher = Watcher(out, phase=args.phase, show_json=args.json)
    tails = [(Tail(args.context, from_start=args.from_start), SOURCE_CONTEXT)]
    if args.trace is not None:
        tails.append((Tail(args.trace, from_start=args.from_start), SOURCE_TRACE))
    try:
        while True:
            # Merged by when each record was written, so a replay (or a poll
            # that caught both files moving) puts a turn's decisions after its
            # own calls rather than after every call in the file.
            batch = [(line, source) for tail, source in tails for line in tail.read()]
            for line, source in sorted(batch, key=lambda item: _written_at(item[0])):
                watcher.feed(line, source)
            if args.once:
                return 0
            time.sleep(_POLL_S)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
