"""Turn tracing: the per-command record of what the agent actually did.

The pipeline runs an utterance down one of three roads — a deterministic
confirmation, the deterministic fast path, or the brain's tool loop — and until
this module existed it recorded *which* nowhere. A turn that misfired left
nothing behind to review: no way to tell whether the model was even called, what
tool it picked, what arguments it passed, or whether it ran out of budget. This
writes one JSONL record per turn holding exactly that, so a bad turn becomes a
thing you can read (and, with its `fen_before` + `utterance`, replay as an eval
scenario).

Tracing is diagnostics and nothing more. It is off unless a path is configured
(`CHESSAPP_TRACE_PATH`), and a tracer that fails is swallowed by the pipeline —
losing a diagnostic record must never cost the player their turn.
"""

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

# The three roads an utterance can take through `_run_command`.
ROUTE_CONFIRMATION = "confirmation"
ROUTE_FAST_PATH = "fast_path"
ROUTE_BRAIN = "brain"


class Tracer(Protocol):
    """Whatever records a finished turn. Free to fail: the caller swallows it."""

    def record(self, turn: dict[str, Any]) -> None: ...


def turn_record(
    *,
    utterance: str,
    route: str,
    commentary: str,
    stop_reason: str,
    changed: bool,
    fen_before: str,
    fen_after: str,
    tool_calls: list[dict[str, Any]],
    tool_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """One turn, as the flat record a reviewer (or a replay) reads.

    `tool_calls` and `tool_results` arrive as the pipeline's two parallel lists
    — args and `{"name", "result"}`, parallel by construction — and are zipped
    back into one entry per call, which is the shape a human actually wants:
    what it called, with what, and what came back.
    """
    return {
        "utterance": utterance,
        "route": route,
        "commentary": commentary,
        "stop_reason": stop_reason,
        "changed": changed,
        "fen_before": fen_before,
        "fen_after": fen_after,
        "tools": [
            {"name": result["name"], "args": args, "result": result["result"]}
            for args, result in zip(tool_calls, tool_results, strict=True)
        ],
    }


@dataclass
class JsonlTracer:
    """Appends each turn to a JSONL file, newest last, one line per turn.

    JSONL because turns are appended forever and read back a few at a time: it
    survives a crash mid-write, `tail -f` works during a live game, and a run of
    turns loads with a one-line comprehension.
    """

    path: Path

    def record(self, turn: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps({"ts": datetime.now(UTC).isoformat(), **turn})
        with self.path.open("a") as handle:
            handle.write(line + "\n")
