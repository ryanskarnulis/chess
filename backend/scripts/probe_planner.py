"""Probe the planner directly: one model call per sample, arms interleaved.

The knight-ask campaign (#286) needs a measurement that is cheaper than the
eval harness and honest about the one thing the harness cannot cancel: on one
llama-server, consecutive samples of one prompt are correlated (the unchanged
text once read 5/20, 19/40 and 31/40 in three separate batches). So this probe
runs its arms **round-robin per sample** — control, arm, control, arm — and a
batch of one arm alone is not something it can produce.

What it sends is the shipped planner call, not a copy of it: the app's own
state view (`api._agent_state_dict`), the tool offer `build_app` makes
(`tools.brain_tool_definitions(registry, ctx)`), the messages
`LlamaBrain._messages` opens a run with, and `LlamaCppProvider.chat` with the
planner's own generation ceiling, thinking off. An arm varies exactly one of
its knobs: the planner prompt text, the planner temperature, llama-server's
per-request `cache_prompt`, one tool's description text, the model id, the
shape `legal_moves` is shown in, `make_move`'s schema, or thinking (#351).

Each sample is classified from the wire result alone — the tool calls the
model made, or `no_tool` — and scored against the corpus item's rule. No
language is parsed; a question and a refusal are both `no_tool`, which is the
pass for every ambiguous and every impossible ask here.

Run from `backend/` with llama-swap up:

    python scripts/probe_planner.py --arm control \
        --arm nocache:cache_prompt=false --n 40
    python scripts/probe_planner.py --arm control --items knight_ask --n 20 --fresh
    python scripts/probe_planner.py --arm control --dry-run          # the exact payload
    python scripts/probe_planner.py --preflight-only                 # is the card free?

Arm spec: `NAME[:key=value[,key=value...]]` with keys `prompt=@file`,
`temperature=0.3`, `cache_prompt=false`, `model=<id>`, `tool_text=<tool>@file`,
`drop_tool=<tool>` (repeatable: the offer without that tool),
`state_view=sorted|by_piece|joined` (the shape `legal_moves` is shown in),
`tool_schema=provenance` (`make_move` says where its move came from, scored
as the app would check it) and `thinking=on`.
`control` (no keys) is the shipped planner. `--fresh` calls llama-swap's
`/unload` before the first sample so the session is new; it refuses while a
slot is processing or another job holds the card (the shared-GPU rule).

Every record carries what proves which agent and which server it measured:
git HEAD, the sha of the prompt text and of the tool offer as sent, model,
temperature, `cache_prompt`, llama-swap `/running`, whether the session was
fresh, and the request ordinal since the probe's (un)load.

It is a developer tool, not part of the app or the test suite; its pure parts
(the classifier, the rules, the schedule, the statistics, the pre-flight
decision) are tested off the GPU in `tests/test_probe_planner.py`.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from chessapp import clarification
from chessapp.api import _agent_state_dict, planner_state
from chessapp.coordinator import TurnCoordinator
from chessapp.fastparse import parse_move
from chessapp.game import GameSession
from chessapp.llama_brain import (
    _PLANNER_MAX_TOKENS,
    _PLANNER_TEMPERATURE,
    create_llama_brain,
)
from chessapp.personality import PLANNER_PROMPT
from chessapp.provider import ChatResult, LlamaCppProvider, ProviderError
from chessapp.tools import (
    Settings,
    ToolContext,
    brain_tool_definitions,
    build_registry,
)

DEFAULT_BASE_URL = "http://127.0.0.1:8200/v1"
DEFAULT_MODEL = "gemma-4-12b"

# --- the corpus ---------------------------------------------------------------
#
# Each item is an utterance on a position, plus the rule a pass must satisfy.
# Rules are data, not code, so a record can say which rule scored it:
#   ("no_tool",)                         the model called nothing (asked/refused)
#   ("asks",)                            nothing, or its first call is
#                                        `ask_player` — the two ways to ask (#289)
#   ("no_move",)                         no `make_move` that would land: asking,
#                                        replying and reading all pass (#351)
#   ("first_call", name, constraints)    the first call is `name`; constraints:
#       {"move_in": [...]}  its `move` argument is one of these
#       {"absent": [...]}   these argument names were omitted
# Two ambiguous items are held out of arm design and used only for the final
# confirmation; `--include-held-out` adds them.

Rule = tuple[Any, ...]


@dataclass(frozen=True)
class Question:
    """A clarification record as the planner's state shows it (#319): the
    player's ask and the moves offered. `stale` puts it in `closed_question`
    (the board changed after it was asked) rather than `open_question`."""

    request: str
    candidates: tuple[str, ...]
    stale: bool = False


@dataclass(frozen=True)
class Item:
    name: str
    utterance: str
    moves: tuple[str, ...]
    rule: Rule
    held_out: bool = False
    note: str = ""
    # Prior conversation as (role, content) pairs, final answers only — what
    # `LlamaBrain._messages` puts between the system prompt and the command.
    transcript: tuple[tuple[str, str], ...] = ()
    question: Question | None = None


_CASTLE_BOTH = (
    "e4", "e5", "Nf3", "Nc6", "Bc4", "Bc5", "d3", "d6",
    "Be3", "Be6", "Nc3", "Nf6", "Qd2", "Qd7",
)  # fmt: skip

# The ordinal items (#351). The planner reads "the first one" as
# `legal_moves[0]` whatever it was asked, so each question here offers its
# candidates in an order that is *not* `legal_moves` order (Nh3 comes before
# Nf3 there): a pick that follows the list lands on the wrong move, and a lever
# that simply refuses every ordinal fails the open-question items.
_KNIGHT_ASKED = (
    ("user", "move my kings knight"),
    ("assistant", "King's knight: f3 or h3?"),
)
_KNIGHT_QUESTION = ("move my kings knight", ("Nf3", "Nh3"))
_PLAIN_TURN = (
    ("user", "e4"),
    ("assistant", "e4, and I answer e5. Classic."),
)

CORPUS: tuple[Item, ...] = (
    Item(
        "knight_ask",
        "move my kings knight",
        (),
        ("asks",),
        note="two squares fit (Nf3, Nh3): the campaign's own ask",
    ),
    Item(
        "rook_ask",
        "move the rook",
        ("a4", "a5", "h4", "h5"),
        ("asks",),
        note="four rook moves fit; `ambiguous_move`'s position",
    ),
    Item(
        "bishop_ask",
        "move the bishop",
        ("e4", "e5", "d4", "d5"),
        ("asks",),
        held_out=True,
        note="both bishops mobile; held out of arm design",
    ),
    Item(
        "castle_both",
        "castle",
        _CASTLE_BOTH,
        ("asks",),
        held_out=True,
        note="O-O and O-O-O both legal; held out of arm design",
    ),
    Item(
        "take_pawn",
        "take the pawn",
        (),
        ("no_tool",),
        note="nothing to take on move 1: refused, never asked",
    ),
    Item(
        "bishop_a1",
        "bishop to a1",
        (),
        ("no_tool",),
        note="no piece can make it: refused, never asked",
    ),
    Item(
        "stt_knight",
        "please put my night on f three",
        (),
        ("first_call", "make_move", {"move_in": ["Nf3", "g1f3"]}),
        note="one square fits through the STT slip",
    ),
    Item(
        "castle_one",
        "castle",
        ("e4", "e5", "Nf3", "Nc6", "Bc4", "Bc5"),
        ("first_call", "make_move", {"move_in": ["O-O", "e1g1"]}),
        note="one castling fits (the app's fast path settles this before the planner)",
    ),
    Item(
        "undo_replace",
        "take that bishop move back and play d4 instead",
        ("e4", "b6", "Nf3", "h6", "Bc4", "a5"),
        ("first_call", "undo", {"absent": ["plies"]}),
        note="the first call is the takeback with `plies` left to the app",
    ),
    Item(
        "ordinal_no_question",
        "the first one",
        ("e4", "e5"),
        ("no_move",),
        transcript=_PLAIN_TURN,
        note="nothing was asked: an ordinal names no move (#351; Nh3 20/20)",
    ),
    Item(
        "ordinal_no_question_alt",
        "go with the first option",
        ("e4", "e5"),
        ("no_move",),
        held_out=True,
        transcript=_PLAIN_TURN,
        note="the heldout wording of the no-question ordinal",
    ),
    Item(
        "ordinal_stale",
        "the first one",
        ("d4", "d5"),
        ("no_move",),
        transcript=_KNIGHT_ASKED,
        question=Question(*_KNIGHT_QUESTION, stale=True),
        note="the knight question closed when the board changed under it",
    ),
    Item(
        "ordinal_open_pick",
        "the first one",
        (),
        ("first_call", "make_move", {"move_in": ["Nf3", "g1f3"]}),
        transcript=_KNIGHT_ASKED,
        question=Question(*_KNIGHT_QUESTION),
        note="the open question's first is Nf3; legal_moves[0] is Nh3 (#352)",
    ),
    Item(
        "ordinal_open_second",
        "the second one",
        (),
        ("first_call", "make_move", {"move_in": ["e3", "e2e3"]}),
        held_out=True,
        transcript=(
            ("user", "push my e pawn"),
            ("assistant", "e-pawn: e4 or e3?"),
        ),
        question=Question("push my e pawn", ("e4", "e3")),
        note="the second offered is e3; legal_moves[1] is Nf3",
    ),
)


def corpus(names: Sequence[str] | None, include_held_out: bool) -> list[Item]:
    """The items to run, in corpus order. Naming an item selects it whether or
    not it is held out — naming is the deliberate act."""
    by_name = {item.name: item for item in CORPUS}
    if names:
        unknown = [n for n in names if n not in by_name]
        if unknown:
            raise SystemExit(f"unknown corpus item(s): {', '.join(unknown)}")
        return [by_name[n] for n in names]
    return [item for item in CORPUS if include_held_out or not item.held_out]


# --- arms ---------------------------------------------------------------------


@dataclass(frozen=True)
class Arm:
    """One variant of the planner call. Every field but `name` defaults to the
    shipped value, so an arm differs from `control` in exactly what it names."""

    name: str
    prompt: str = PLANNER_PROMPT
    temperature: float | None = _PLANNER_TEMPERATURE
    cache_prompt: bool | None = None
    model: str | None = None
    tool_text: dict[str, str] = field(default_factory=dict)
    drop_tools: tuple[str, ...] = ()
    state_view: str | None = None
    tool_schema: str | None = None
    thinking: bool = False

    def view(self, state: dict[str, Any]) -> dict[str, Any]:
        """The planner's opening state as this arm shows it."""
        if self.state_view is None:
            return state
        return STATE_VIEWS[self.state_view](state)

    def offer(self, definitions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """The tool offer with this arm's dropped tools removed and its
        description overrides applied."""
        missing_drops = set(self.drop_tools) - {
            d["function"]["name"] for d in definitions
        }
        if missing_drops:
            raise SystemExit(
                f"arm {self.name!r} drops tools not in the offer: "
                f"{', '.join(sorted(missing_drops))}"
            )
        if self.drop_tools:
            definitions = [
                d for d in definitions if d["function"]["name"] not in self.drop_tools
            ]
        if self.tool_schema is not None:
            definitions = TOOL_SCHEMAS[self.tool_schema](copy.deepcopy(definitions))
        if not self.tool_text:
            return definitions
        offered = copy.deepcopy(definitions)
        names = {d["function"]["name"] for d in offered}
        missing = set(self.tool_text) - names
        if missing:
            raise SystemExit(
                f"arm {self.name!r} overrides text of tools not in the offer: "
                f"{', '.join(sorted(missing))}"
            )
        for d in offered:
            text = self.tool_text.get(d["function"]["name"])
            if text is not None:
                d["function"]["description"] = text
        return offered


# Data-shape arms (#351): the same facts, in a shape with or without an order
# an ordinal could index into. Only `legal_moves` changes; `make_move` and
# `ask_player`'s enum still take the flat SAN strings.

_PIECE_NAMES = {"K": "king", "Q": "queen", "R": "rook", "B": "bishop", "N": "knight"}
_PIECE_ORDER = ("king", "queen", "rook", "bishop", "knight", "pawn")


def san_piece(san: str) -> str:
    """The piece a SAN move moves: castling is the king's, a bare square a pawn's."""
    if san.startswith("O-O"):
        return "king"
    return _PIECE_NAMES.get(san[0], "pawn")


def _sorted_view(state: dict[str, Any]) -> dict[str, Any]:
    moves = state["legal_moves"]
    return {
        **state,
        "legal_moves": sorted(
            moves, key=lambda san: (_PIECE_ORDER.index(san_piece(san)), san)
        ),
    }


def _by_piece_view(state: dict[str, Any]) -> dict[str, Any]:
    grouped: dict[str, list[str]] = {}
    for piece in _PIECE_ORDER:
        fitting = sorted(san for san in state["legal_moves"] if san_piece(san) == piece)
        if fitting:
            grouped[piece] = fitting
    return {**state, "legal_moves": grouped}


def _joined_view(state: dict[str, Any]) -> dict[str, Any]:
    return {**state, "legal_moves": " ".join(sorted(state["legal_moves"]))}


STATE_VIEWS: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
    "sorted": _sorted_view,
    "by_piece": _by_piece_view,
    "joined": _joined_view,
}

# Schema arms (#351): a typed claim code can check without reading language.
# `provenance` makes `make_move` say where the move came from; the app would
# refuse `answer_to_open_question` when no question stands or the move is not
# one it offered — `lands` is that check, applied to what the probe scores.
ANSWER = "answer_to_open_question"
NAMED = "player_named_it"


def _provenance_schema(definitions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for d in definitions:
        function = d["function"]
        if function["name"] != "make_move":
            continue
        parameters = function["parameters"]
        parameters["properties"]["source"] = {
            "type": "string",
            "enum": [NAMED, ANSWER],
            "description": (
                f"{NAMED}: the player's own words name this move. "
                f"{ANSWER}: they picked it from the board state's `open_question`."
            ),
        }
        parameters["required"] = [*parameters.get("required", []), "source"]
    return definitions


TOOL_SCHEMAS: dict[str, Callable[[list[dict[str, Any]]], list[dict[str, Any]]]] = {
    "provenance": _provenance_schema,
}


def _read_at(value: str, *, what: str) -> str:
    if not value.startswith("@"):
        raise SystemExit(f"{what} must be a file reference (@path), got {value!r}")
    return Path(value[1:]).read_text()


def parse_arm(spec: str, read: Callable[[str], str] = Path.read_text) -> Arm:
    """`NAME[:key=value[,key=value...]]` → an Arm. Text knobs are files
    (`prompt=@path`, `tool_text=make_move@path`) so a spec never carries prose."""
    name, _, rest = spec.partition(":")
    if not name:
        raise SystemExit(f"arm spec needs a name: {spec!r}")
    kwargs: dict[str, Any] = {}
    tool_text: dict[str, str] = {}
    drop_tools: list[str] = []
    for pair in filter(None, rest.split(",")):
        key, eq, value = pair.partition("=")
        if not eq:
            raise SystemExit(f"arm {name!r}: {pair!r} is not key=value")
        if key == "prompt":
            if not value.startswith("@"):
                raise SystemExit(f"arm {name!r}: prompt must be @path")
            kwargs["prompt"] = read(Path(value[1:]))
        elif key == "temperature":
            kwargs["temperature"] = float(value)
        elif key == "cache_prompt":
            if value.lower() not in ("true", "false"):
                raise SystemExit(f"arm {name!r}: cache_prompt must be true|false")
            kwargs["cache_prompt"] = value.lower() == "true"
        elif key == "model":
            kwargs["model"] = value
        elif key == "tool_text":
            tool, at, path = value.partition("@")
            if not at or not tool:
                raise SystemExit(f"arm {name!r}: tool_text must be <tool>@path")
            tool_text[tool] = read(Path(path))
        elif key == "drop_tool":
            drop_tools.append(value)
        elif key == "state_view":
            if value not in STATE_VIEWS:
                raise SystemExit(
                    f"arm {name!r}: state_view must be one of {', '.join(STATE_VIEWS)}"
                )
            kwargs["state_view"] = value
        elif key == "tool_schema":
            if value not in TOOL_SCHEMAS:
                known = ", ".join(TOOL_SCHEMAS)
                raise SystemExit(f"arm {name!r}: tool_schema must be one of {known}")
            kwargs["tool_schema"] = value
        elif key == "thinking":
            if value.lower() not in ("on", "off"):
                raise SystemExit(f"arm {name!r}: thinking must be on|off")
            kwargs["thinking"] = value.lower() == "on"
        else:
            raise SystemExit(f"arm {name!r}: unknown knob {key!r}")
    return Arm(name=name, tool_text=tool_text, drop_tools=tuple(drop_tools), **kwargs)


# --- classification and scoring -----------------------------------------------

Call = dict[str, Any]  # {"name": str, "args": dict}


def classify(result: ChatResult) -> list[Call]:
    """What the model did, read off the wire result and nothing else."""
    return [{"name": c.name, "args": dict(c.arguments)} for c in result.tool_calls]


def outcome_label(calls: Sequence[Call]) -> str:
    """A short deterministic label for grouping: `no_tool`, or the calls in
    order with the arguments the rules read (`make_move`'s move, `undo`'s plies)."""
    if not calls:
        return "no_tool"
    parts = []
    for call in calls:
        name, args = call["name"], call["args"]
        if name == "make_move" and "move" in args:
            source = f",{args['source']}" if "source" in args else ""
            parts.append(f"make_move({args['move']}{source})")
        elif name == "undo" and "plies" in args:
            parts.append(f"undo(plies={args['plies']})")
        elif name == "ask_player" and "candidates" in args:
            parts.append(f"ask_player({','.join(map(str, args['candidates']))})")
        else:
            parts.append(name)
    return "|".join(parts)


def lands(call: Call, question: Question | None) -> bool:
    """Whether the app would carry `call` out, as far as the provenance check
    goes: a move claimed as an answer lands only on a question that stands and
    only as one of its candidates. Every other call is the shipped behavior."""
    if call["name"] != "make_move" or call["args"].get("source") != ANSWER:
        return True
    if question is None or question.stale:
        return False
    return call["args"].get("move") in question.candidates


def passes(rule: Rule, calls: Sequence[Call]) -> bool:
    """Score one sample's calls against an item's rule."""
    kind = rule[0]
    if kind == "no_tool":
        return not calls
    if kind == "no_move":
        return not any(call["name"] == "make_move" for call in calls)
    if kind == "asks":
        return not calls or calls[0]["name"] == "ask_player"
    if kind == "first_call":
        _, name, constraints = rule
        if not calls or calls[0]["name"] != name:
            return False
        args = calls[0]["args"]
        if "move_in" in constraints and args.get("move") not in constraints["move_in"]:
            return False
        return all(key not in args for key in constraints.get("absent", ()))
    raise ValueError(f"unknown rule kind {kind!r}")


# --- the schedule -------------------------------------------------------------


def schedule(
    n: int, items: Sequence[str], arms: Sequence[str]
) -> Iterator[tuple[int, str, str]]:
    """Sample index, item, arm — arms innermost, so consecutive requests to the
    server alternate arms and any drift lands on all of them alike."""
    for sample in range(n):
        for item in items:
            for arm in arms:
                yield sample, item, arm


# --- statistics ---------------------------------------------------------------


def lag1_agreement(outcomes: Sequence[bool]) -> float | None:
    """The fraction of consecutive same-arm samples with the same outcome.
    Under independence at rate p it should sit near p² + (1-p)²; well above
    that is the clustering the 2026-09-05 batches showed."""
    if len(outcomes) < 2:
        return None
    pairs = list(zip(outcomes, outcomes[1:], strict=False))
    return sum(a == b for a, b in pairs) / len(pairs)


def independent_agreement(rate: float) -> float:
    return rate * rate + (1 - rate) * (1 - rate)


def summarize(
    records: Sequence[dict[str, Any]],
) -> dict[tuple[str, str], dict[str, Any]]:
    """Per (item, arm): scored count, passes, errors, the outcome histogram and
    the lag-1 agreement. Errors (a provider failure) are counted, never scored."""
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for record in records:
        groups.setdefault((record["item"], record["arm"]), []).append(record)
    summary: dict[tuple[str, str], dict[str, Any]] = {}
    for key, rows in groups.items():
        scored = [r for r in rows if r.get("error") is None]
        passed = [bool(r["passed"]) for r in scored]
        rate = (sum(passed) / len(passed)) if passed else None
        summary[key] = {
            "n": len(scored),
            "passed": sum(passed),
            "errors": len(rows) - len(scored),
            "outcomes": dict(Counter(r["outcome"] for r in scored)),
            "lag1": lag1_agreement(passed),
            "lag1_independent": independent_agreement(rate)
            if rate is not None
            else None,
        }
    return summary


def format_summary(summary: dict[tuple[str, str], dict[str, Any]]) -> str:
    lines = [
        "item | arm | passed | lag-1 (indep.) | outcomes",
        "--- | --- | --- | --- | ---",
    ]
    for (item, arm), s in sorted(summary.items()):
        lag = (
            "n/a"
            if s["lag1"] is None
            else f"{s['lag1']:.2f} ({s['lag1_independent']:.2f})"
        )
        errors = f" +{s['errors']} err" if s["errors"] else ""
        outcomes = ", ".join(f"{k} {v}" for k, v in sorted(s["outcomes"].items()))
        lines.append(
            f"{item} | {arm} | {s['passed']}/{s['n']}{errors} | {lag} | {outcomes}"
        )
    return "\n".join(lines)


# --- pre-flight: is the card ours to use? -------------------------------------

# Processes that hold GPU memory without being a job: the desktop.
_DESKTOP_PROCESSES = (
    "kwin",
    "Xorg",
    "Xwayland",
    "gnome-shell",
    "plasmashell",
    "mutter",
)


def busy_slots(slots: Any) -> list[int]:
    """Slot ids llama-server reports as processing (its `/slots` JSON array)."""
    if not isinstance(slots, list):
        return []
    return [
        int(slot.get("id", i))
        for i, slot in enumerate(slots)
        if isinstance(slot, dict) and slot.get("is_processing")
    ]


def foreign_gpu_processes(rows: Sequence[str]) -> list[str]:
    """Process names from `nvidia-smi --query-compute-apps=pid,process_name
    --format=csv,noheader` that are neither llama-server nor the desktop."""
    foreign = []
    for row in rows:
        if not row.strip():
            continue
        name = row.split(",")[-1].strip()
        if "llama-server" in name:
            continue
        if any(marker in name for marker in _DESKTOP_PROCESSES):
            continue
        foreign.append(name)
    return foreign


def preflight_reasons(
    running_models: Sequence[str], model: str, slots: Any, gpu_rows: Sequence[str]
) -> list[str]:
    """Why the probe must not start now; empty means go."""
    reasons = []
    if model in running_models:
        busy = busy_slots(slots)
        if busy:
            reasons.append(f"llama-server slot(s) {busy} are processing")
    foreign = foreign_gpu_processes(gpu_rows)
    if foreign:
        reasons.append(f"another job holds the GPU: {', '.join(foreign)}")
    return reasons


# --- the server ---------------------------------------------------------------


class Swap:
    """The llama-swap endpoints beside `/v1`: identity, slots, unload."""

    def __init__(self, base_url: str) -> None:
        root = base_url.rstrip("/")
        self.root = root[: -len("/v1")] if root.endswith("/v1") else root
        self._client = httpx.Client(timeout=10.0)

    def running(self) -> dict[str, Any]:
        try:
            return self._client.get(f"{self.root}/running").json()
        except (httpx.HTTPError, ValueError):
            return {}

    def running_models(self) -> list[str]:
        return [
            str(entry.get("model"))
            for entry in self.running().get("running", [])
            if isinstance(entry, dict)
        ]

    def slots(self, model: str) -> Any:
        try:
            return self._client.get(f"{self.root}/upstream/{model}/slots").json()
        except (httpx.HTTPError, ValueError):
            return []

    def unload(self) -> None:
        self._client.get(f"{self.root}/unload").raise_for_status()


def gpu_rows() -> list[str]:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    return completed.stdout.splitlines()


def git_head() -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return completed.stdout.strip()


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:12]


# --- one planner call, built the way the app builds it ------------------------


def position(item: Item) -> GameSession:
    session = GameSession()
    for san in item.moves:
        outcome = session.submit_move(san)
        assert outcome.legal, f"{item.name}: setup move {san} is illegal"
    return session


@dataclass
class Prepared:
    """Everything one request needs, before the wire."""

    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]]
    prompt_sha: str
    offer_sha: str
    fen: str
    fast_path: str | None


def question_records(
    question: Question | None,
) -> tuple[clarification.Clarification | None, clarification.Closed | None]:
    """The records `planner_state` takes for an item's question."""
    if question is None:
        return None, None
    record = clarification.ask(
        origin="probe",
        game_id="probe",
        board_version=0,
        request=question.request,
        candidates=question.candidates,
    )
    if not question.stale:
        return record, None
    return None, clarification.Closed(
        record, clarification.INVALIDATED, clarification.BOARD_CHANGED
    )


def prepare(
    item: Item, arm: Arm, provider: LlamaCppProvider, base_url: str, model: str
) -> Prepared:
    session = position(item)
    ctx = ToolContext(session=session, engine=None, settings=Settings())
    coordinator = TurnCoordinator(ctx)
    registry = build_registry(ctx, coordinator, atomic_exchange=False)
    tools = arm.offer(brain_tool_definitions(registry, ctx))
    brain = create_llama_brain(
        base_url=base_url,
        model=model,
        dispatcher=registry,
        tool_definitions=tools,
        planner_prompt_provider=lambda: arm.prompt,
        provider=provider,
    )
    # The shipped opening block (#319), with the item's question if it has one.
    open_record, closed = question_records(item.question)
    state = arm.view(planner_state(_agent_state_dict(ctx), open_record, closed))
    transcript = [{"role": role, "content": text} for role, text in item.transcript]
    messages = brain._messages(state, item.utterance, transcript)
    return Prepared(
        messages=messages,
        tools=tools,
        prompt_sha=sha(arm.prompt),
        offer_sha=sha(json.dumps(tools, sort_keys=True)),
        fen=session.fen(),
        fast_path=parse_move(item.utterance, session.fen()),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--arm", action="append", default=[], help="arm spec; repeatable"
    )
    parser.add_argument("--n", type=int, default=20, help="samples per (item, arm)")
    parser.add_argument(
        "--items", nargs="*", help="corpus items to run (default: all not held out)"
    )
    parser.add_argument("--include-held-out", action="store_true")
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="llama-swap /unload before the first sample",
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--out", type=Path, help="JSONL path (default probe-<utc>.jsonl)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print each arm's first payload; no calls",
    )
    parser.add_argument(
        "--preflight-only", action="store_true", help="exit 0 if the card is free"
    )
    parser.add_argument(
        "--ignore-gpu", action="store_true", help="skip the foreign-process check"
    )
    parser.add_argument("--list", action="store_true", help="print the corpus and exit")
    args = parser.parse_args(argv)

    if args.list:
        for item in CORPUS:
            held = " (held out)" if item.held_out else ""
            print(
                f"{item.name}{held}: {item.utterance!r} after "
                f"{' '.join(item.moves) or 'nothing'} — {item.note}"
            )
        return 0

    swap = Swap(args.base_url)
    rows = [] if args.ignore_gpu else gpu_rows()
    reasons = preflight_reasons(
        swap.running_models(),
        args.model,
        swap.slots(args.model) if args.model in swap.running_models() else [],
        rows,
    )
    if args.preflight_only:
        print("\n".join(reasons) if reasons else "clear")
        return 1 if reasons else 0

    arms = [parse_arm(spec) for spec in (args.arm or ["control"])]
    if len({arm.name for arm in arms}) != len(arms):
        raise SystemExit("arm names must be unique")
    items = corpus(args.items, args.include_held_out)
    providers = {
        model: LlamaCppProvider(args.base_url, model)
        for model in {arm.model or args.model for arm in arms}
    }

    if args.dry_run:
        for arm in arms:
            model = arm.model or args.model
            prepared = prepare(items[0], arm, providers[model], args.base_url, model)
            payload = providers[model]._payload(
                prepared.messages,
                tools=prepared.tools,
                enable_thinking=arm.thinking,
                max_tokens=_PLANNER_MAX_TOKENS,
                temperature=arm.temperature,
                cache_prompt=arm.cache_prompt,
            )
            print(
                f"# arm {arm.name}: prompt {prepared.prompt_sha} "
                f"offer {prepared.offer_sha} item {items[0].name}"
            )
            print(json.dumps(payload, indent=1))
        return 0

    if reasons and not args.fresh:
        # A warm run on a busy card still measures a shared server; say so.
        print("pre-flight: " + "; ".join(reasons), file=sys.stderr)
    if reasons and args.fresh:
        raise SystemExit("refusing to unload: " + "; ".join(reasons))

    session_fresh = False
    if args.fresh:
        swap.unload()
        session_fresh = True
        print("unloaded; the first request reloads the model", file=sys.stderr)

    out = args.out or Path(
        f"probe-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}.jsonl"
    )
    head = git_head()
    by_arm = {arm.name: arm for arm in arms}
    by_item = {item.name: item for item in items}
    records: list[dict[str, Any]] = []
    ordinal = 0
    session_started: str | None = None
    # Server identity is read after the first answer, not before: with the
    # model unloaded `/running` is empty, and the identity worth recording is
    # the server that actually answered (its model and the flags it runs with).
    running_sha: str | None = None
    running_models: list[str] | None = None
    print(f"HEAD {head}; writing {out}", file=sys.stderr)

    with out.open("a") as handle:
        for sample, item_name, arm_name in schedule(
            args.n, [i.name for i in items], [a.name for a in arms]
        ):
            item, arm = by_item[item_name], by_arm[arm_name]
            model = arm.model or args.model
            provider = providers[model]
            prepared = prepare(item, arm, provider, args.base_url, model)
            ordinal += 1
            started = time.monotonic()
            error: str | None = None
            calls: list[Call] = []
            landed: list[Call] = []
            usage: dict[str, Any] | None = None
            try:
                result = provider.chat(
                    prepared.messages,
                    tools=prepared.tools,
                    enable_thinking=arm.thinking,
                    max_tokens=_PLANNER_MAX_TOKENS,
                    temperature=arm.temperature,
                    cache_prompt=arm.cache_prompt,
                )
            except ProviderError as exc:
                error = f"{type(exc).__name__}: {exc}"
            else:
                calls = classify(result)
                landed = [c for c in calls if lands(c, item.question)]
                usage = result.usage.model_dump() if result.usage else None
                if running_sha is None:
                    running_sha = sha(json.dumps(swap.running(), sort_keys=True))
                    running_models = swap.running_models()
                    if session_fresh:
                        session_started = datetime.now(UTC).isoformat()
                    print(f"server {running_sha} {running_models}", file=sys.stderr)
            record = {
                "ts": datetime.now(UTC).isoformat(),
                "sample": sample,
                "ordinal": ordinal,
                "item": item.name,
                "arm": arm.name,
                "utterance": item.utterance,
                "fen": prepared.fen,
                "fast_path": prepared.fast_path,
                "rule": list(item.rule),
                "calls": calls,
                "outcome": outcome_label(calls) if error is None else "error",
                # Scored on what would land: a call the app refuses moves nothing.
                "passed": passes(item.rule, landed) if error is None else None,
                "refused": [c for c in calls if c not in landed],
                "error": error,
                "latency_ms": round((time.monotonic() - started) * 1000),
                "usage": usage,
                "head": head,
                "prompt_sha": prepared.prompt_sha,
                "offer_sha": prepared.offer_sha,
                "model": model,
                "temperature": arm.temperature,
                "cache_prompt": arm.cache_prompt,
                "state_view": arm.state_view,
                "tool_schema": arm.tool_schema,
                "thinking": arm.thinking,
                "running_sha": running_sha,
                "running_models": running_models,
                "session_fresh": session_fresh,
                "session_started": session_started,
            }
            records.append(record)
            handle.write(json.dumps(record) + "\n")
            handle.flush()
            mark = "ERR" if error else ("pass" if record["passed"] else "MISS")
            print(
                f"[{ordinal:4d}] {sample:3d} {item.name:23s} {arm.name:10s} {mark:4s} "
                f"{record['outcome']}  {record['latency_ms']} ms",
                file=sys.stderr,
            )

    print()
    print(format_summary(summarize(records)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
