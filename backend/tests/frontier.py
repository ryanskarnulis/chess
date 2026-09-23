"""The frontier tier: live-model scenarios that are *meant* to be hard (#318).

The gate (`test_agent_evals.py`) holds floors, and its own comment says what
they are: "regression tripwires, not aspirations". A scenario the model
cannot yet do has nowhere to live there — it either fails the gate or gets an
xfail that comes off the day it goes green, and in neither case is anyone
watching the number move. This tier is where those scenarios live. It
**measures and never fails on a score**: a fixed number of samples per
scenario, a pass count with its Wilson interval, and a rubric score for
partial credit, written to the report so a run can be compared with the last.

Three rules keep the number honest.

- **Partial credit is deterministic.** Each scenario names checkpoints — the
  undo landed, the question was asked before anything moved, the named move
  is on the board — and each is a predicate over the board, the tool results
  and the trace. The grader reads no language, exactly as the gate does not; a
  whole-task pass is every checkpoint on one sample. The rubric score is what
  moves while whole-task passes are still near zero, which is the point of a
  scenario built to be hard.
- **A code failure is not a model miss.** Every live sample runs the
  deterministic trajectory invariants (`trajectory.INVARIANTS`) step by step.
  A breach — the board changed with no tool to explain it, an ask that let a
  move through, an offer that disagrees with its board — is a bug in the app,
  so it fails the run loudly, however low the model's score is allowed to be.
- **A scenario must stay a model eval.** A step marked `model` must be routed
  to the planner; a parser that grows to swallow it would turn the scenario
  into a parser test that passes forever. That is a broken scenario, and it
  fails like a harness bug rather than being scored.

Held-out variants are the fourth rule, and it is a discipline, not code: each
scenario carries `dev` variants for iterating on prompts and `heldout`
variants that nobody tunes against (`CHESSAPP_FRONTIER_SPLIT`). An
improvement claimed on `dev` alone is not believed.

Nothing here needs a GPU: the runner takes an app factory, so its wiring is
tested off the GPU over a scripted provider (`test_frontier.py`), and the live
module (`test_agent_frontier.py`) only supplies the real one.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from chessapp.agent_api import reset_rate_limit
from chessapp.trace import ROUTE_BRAIN
from evalstats import (
    DETERMINISTIC_FAILURES,
    STOP_PROVIDER_ERROR,
    failure_signature,
    wilson_interval,
)
from test_agent_evals import _REQUEST_TIMEOUT, EvalApp, _measured
from trajectory import (
    INVARIANTS,
    PANEL,
    InvariantBreach,
    Observed,
    Pending,
    Step,
    next_pending,
)

# Tools whose success is a Stockfish verdict on the position — what a
# judgment question must be answered from (the gate's `_VERDICT_TOOLS`).
VERDICT_TOOLS = frozenset({"evaluate_position", "analyze_last_move"})
# A reply that is a bare answer to a pending question: the trajectory
# invariants read these as the literal confirmation they are.
_LITERAL_ANSWERS = frozenset({"yes", "no"})
# Provider deaths re-taken per scenario before it is abandoned as INFRA.
_INFRA_RETRIES = 5


class ScenarioError(Exception):
    """The scenario is broken, not the model: a checkpoint that raised, or a
    model step the pipeline answered without the planner."""


class Infra(Exception):
    """llama-server died under a step. The sample never happened."""

    def __init__(self, message: str, *, rejected: bool) -> None:
        super().__init__(message)
        self.rejected = rejected


# --- what a scenario is ------------------------------------------------------------


@dataclass(frozen=True)
class Say:
    """One thing the player says. `origin` is the panel or a delegate thread
    (`d0`, `d1`); `model` pins the step to the planner's route."""

    text: str
    origin: str = PANEL
    model: bool = True


@dataclass(frozen=True)
class Variant:
    """One wording (and setup) of a scenario. Samples cycle through a split's
    variants, so a variant is never the whole measurement."""

    name: str
    says: tuple[Say, ...]
    setup: Callable[[EvalApp], None] | None = None


@dataclass(frozen=True)
class Checkpoint:
    """One named, deterministic piece of the task."""

    name: str
    check: Callable[[Episode], bool]


@dataclass(frozen=True)
class Scenario:
    name: str
    tier: int
    why: str
    dev: tuple[Variant, ...]
    heldout: tuple[Variant, ...]
    checkpoints: tuple[Checkpoint, ...]

    def variants(self, split: str) -> tuple[Variant, ...]:
        if split == "dev":
            return self.dev
        if split == "heldout":
            return self.heldout
        raise ValueError(f"unknown split {split!r}: dev or heldout")


# --- what a sample did ---------------------------------------------------------------


@dataclass
class Turn:
    """One step of an episode, as the checkpoints read it."""

    say: Say
    status: int
    route: str | None
    stop_reason: str | None
    # `{"name", "args", "result"}` per tool call, in order, from either seam.
    results: list[dict[str, Any]]
    commentary: str
    before: dict[str, Any]
    after: dict[str, Any]
    model_calls: int
    seconds: float

    def ran(self, name: str) -> list[dict[str, Any]]:
        """Calls to `name` that succeeded (a move `legal`, the rest `ok`)."""
        return [r for r in self.results if r["name"] == name and _ok(r["result"])]

    def succeeded(self) -> list[str]:
        return [r["name"] for r in self.results if _ok(r["result"])]

    @property
    def moved(self) -> bool:
        return (self.before["game_id"], self.before["fen"]) != (
            self.after["game_id"],
            self.after["fen"],
        )

    @property
    def asked(self) -> bool:
        return bool(self.ran("ask_player"))


@dataclass
class Episode:
    """One sample: every turn, and the app it ran on (for file checks)."""

    variant: str
    start: dict[str, Any]
    turns: list[Turn]
    app: EvalApp | None = None

    def turn(self, index: int) -> Turn:
        """Turn `index`, 1-based, the way scenarios are written."""
        return self.turns[index - 1]

    @property
    def final(self) -> dict[str, Any]:
        return self.turns[-1].after if self.turns else self.start

    def history(self, after_turn: int | None = None) -> list[str]:
        state = self.final if after_turn is None else self.turn(after_turn).after
        return list(state["history"])


def _ok(result: dict[str, Any]) -> bool:
    if "legal" in result:
        return result["legal"] is True
    return result.get("ok") is True


# --- grading --------------------------------------------------------------------------


@dataclass(frozen=True)
class Grade:
    hits: dict[str, bool]

    @property
    def whole(self) -> bool:
        return all(self.hits.values())

    @property
    def score(self) -> float:
        return sum(self.hits.values()) / len(self.hits) if self.hits else 0.0

    @property
    def missed(self) -> list[str]:
        return [name for name, hit in self.hits.items() if not hit]


def grade(episode: Episode, checkpoints: Sequence[Checkpoint]) -> Grade:
    """Every checkpoint, each on its own: one that raises is a broken scenario
    and says which, rather than scoring as a miss."""
    hits: dict[str, bool] = {}
    for checkpoint in checkpoints:
        try:
            hits[checkpoint.name] = bool(checkpoint.check(episode))
        except Exception as exc:
            raise ScenarioError(
                f"checkpoint {checkpoint.name} raised: {exc!r}"
            ) from exc
    return Grade(hits)


# --- the result -----------------------------------------------------------------------


@dataclass
class FrontierResult:
    """One scenario's measurement. Never a verdict."""

    scenario: str
    tier: int
    split: str
    checkpoints: tuple[str, ...]
    grades: list[Grade] = field(default_factory=list)
    samples: list[dict[str, Any]] = field(default_factory=list)
    infra: int = 0
    breaches: list[str] = field(default_factory=list)

    @property
    def runs(self) -> int:
        return len(self.grades)

    @property
    def passed(self) -> int:
        return sum(grade.whole for grade in self.grades)

    @property
    def interval(self) -> tuple[float, float]:
        return wilson_interval(self.passed, self.runs)

    @property
    def rubric(self) -> float | None:
        if not self.grades:
            return None
        return sum(grade.score for grade in self.grades) / len(self.grades)

    @property
    def checkpoint_hits(self) -> dict[str, int]:
        return {
            name: sum(grade.hits.get(name, False) for grade in self.grades)
            for name in self.checkpoints
        }

    @property
    def failure_modes(self) -> Counter[str]:
        return Counter(
            failure_signature("missed " + ", ".join(grade.missed))
            for grade in self.grades
            if not grade.whole
        )

    def summary(self) -> str:
        low, high = self.interval
        rubric = "—" if self.rubric is None else f"{self.rubric:.2f}"
        return (
            f"{self.passed}/{self.runs} whole [{low:.2f}, {high:.2f}] "
            f"rubric {rubric} infra {self.infra}"
        )

    def record(self, **meta: Any) -> dict[str, Any]:
        low, high = self.interval
        return {
            "kind": "frontier",
            "scenario": self.scenario,
            "tier": self.tier,
            "split": self.split,
            "runs": self.runs,
            "passed": self.passed,
            "interval": [low, high],
            "rubric": self.rubric,
            "checkpoint_hits": self.checkpoint_hits,
            "failure_modes": dict(self.failure_modes),
            "infra": self.infra,
            "breaches": list(self.breaches),
            "samples": list(self.samples),
            **meta,
        }


# --- running one sample ---------------------------------------------------------------


def _state(app: EvalApp) -> dict[str, Any]:
    return app.client.get("/api/state").json()


def _conversation(app: EvalApp, origin: str, opened: dict[str, int]) -> int:
    if origin not in opened:
        opened[origin] = app.client.post("/api/agent/conversations", json={}).json()[
            "id"
        ]
    return opened[origin]


def _say(
    app: EvalApp, label: str, say: Say, opened: dict[str, int]
) -> tuple[Any, dict[str, Any], list[dict[str, Any]], str]:
    """Send one utterance down its seam; the response, the `_measured` wire
    document, the calls in one shape, and the words."""
    app.provider.reset()
    app.tracer.reset()
    reset_rate_limit()
    started = time.monotonic()
    if say.origin == PANEL:
        response = app.client.post(
            "/api/command", json={"text": say.text}, timeout=_REQUEST_TIMEOUT
        )
    else:
        conversation = _conversation(app, say.origin, opened)
        response = app.client.post(
            f"/api/agent/conversations/{conversation}/messages",
            json={"content": say.text},
            timeout=_REQUEST_TIMEOUT,
        )
    duration = time.monotonic() - started
    results: list[dict[str, Any]] = []
    commentary = ""
    assistant: dict[str, Any] = {"content": "", "tool_calls": []}
    if response.status_code == 200:
        body = response.json()
        if say.origin == PANEL:
            commentary = body["commentary"]
            results = [
                {"name": r["name"], "args": {}, "result": r["result"]}
                for r in body["tool_results"]
            ]
            assistant = {
                "content": commentary,
                "tool_calls": [
                    {
                        "tool": r["name"],
                        "arguments": {},
                        "result": json.dumps(r["result"]),
                        "error": None,
                    }
                    for r in body["tool_results"]
                ],
            }
        else:
            assistant = body["assistant_message"]
            commentary = assistant["content"] or ""
            results = [
                {
                    "name": call["tool"],
                    "args": call.get("arguments") or {},
                    "result": (
                        json.loads(call["result"])
                        if call["result"] is not None
                        else {"ok": False, "error": call["error"]}
                    ),
                }
                for call in assistant.get("tool_calls") or []
            ]
    run = _measured(app, label, assistant, response, duration)
    return response, {"run": run, "duration": duration}, results, commentary


def play(
    app: EvalApp, scenario: Scenario, variant: Variant
) -> tuple[Episode, list[str]]:
    """One sample: the variant's setup, then each utterance, each step checked
    against every trajectory invariant. Returns the episode and any breaches;
    raises `Infra` on a provider death and `ScenarioError` on a broken
    scenario."""
    if variant.setup is not None:
        variant.setup(app)
    start = _state(app)
    episode = Episode(variant=variant.name, start=start, turns=[], app=app)
    breaches: list[str] = []
    pending: Pending | None = None
    opened: dict[str, int] = {}
    for index, say in enumerate(variant.says, start=1):
        label = f"{scenario.name}[{variant.name}]/{index}"
        before = _state(app)
        response, measured, results, commentary = _say(app, label, say, opened)
        run = measured["run"]
        if response.status_code != 200 or run.stop_reason == STOP_PROVIDER_ERROR:
            raise Infra(
                f"{label}: HTTP {response.status_code}, stop {run.stop_reason}, "
                f"{run.provider_failure}",
                rejected=run.provider_failure in DETERMINISTIC_FAILURES,
            )
        if say.model and run.route != ROUTE_BRAIN:
            raise ScenarioError(
                f"{label}: {say.text!r} took the {run.route} route, not the "
                "planner's — the scenario no longer measures the model"
            )
        after = _state(app)
        turn = Turn(
            say=say,
            status=response.status_code,
            route=run.route,
            stop_reason=run.stop_reason,
            results=results,
            commentary=commentary,
            before=before,
            after=after,
            model_calls=len(run.model_calls),
            seconds=round(measured["duration"], 1),
        )
        episode.turns.append(turn)
        observed = Observed(
            step=Step(
                kind=(
                    "literal"
                    if say.text.strip().lower() in _LITERAL_ANSWERS
                    else "command"
                ),
                text=say.text.strip().lower(),
                origin=say.origin,
            ),
            before=before,
            after=after,
            status_code=response.status_code,
            response=response.json(),
            results=[{"name": r["name"], "result": r["result"]} for r in results],
            commentary=commentary,
            provider_calls=list(app.provider.requests),
            seconds=measured["duration"],
            pending_before=pending,
        )
        for invariant in INVARIANTS:
            try:
                invariant(observed)
            except InvariantBreach as breach:
                breaches.append(f"{label}: {invariant.__name__}: {breach}")
        pending = next_pending(observed, pending)
    return episode, breaches


def _sample_record(
    episode: Episode, verdict: Grade, breaches: Sequence[str]
) -> dict[str, Any]:
    return {
        "variant": episode.variant,
        "whole": verdict.whole,
        "score": verdict.score,
        "hits": verdict.hits,
        "breaches": list(breaches),
        "turns": [
            {
                "said": turn.say.text,
                "origin": turn.say.origin,
                "route": turn.route,
                "stop_reason": turn.stop_reason,
                "tools": [
                    r["name"] + ("" if _ok(r["result"]) else "!") for r in turn.results
                ],
                "history": turn.after["history"][-6:],
                "model_calls": turn.model_calls,
                "seconds": turn.seconds,
            }
            for turn in episode.turns
        ],
    }


def measure(
    scenario: Scenario,
    *,
    runs: int,
    split: str,
    app_factory: Callable[[], EvalApp],
    infra_retries: int = _INFRA_RETRIES,
) -> FrontierResult:
    """`runs` samples of `scenario`, cycling through `split`'s variants, each
    on a fresh app. Never judges the count: the caller reports it.

    A provider death is re-taken (bounded by `infra_retries`), a request the
    server rejected ends the scenario at once — it would be refused the same
    way every time — and both are counted rather than scored. A broken
    scenario (`ScenarioError`) propagates: there is nothing to measure.
    """
    variants = scenario.variants(split)
    if not variants:
        raise ScenarioError(f"{scenario.name} has no {split} variants")
    result = FrontierResult(
        scenario=scenario.name,
        tier=scenario.tier,
        split=split,
        checkpoints=tuple(c.name for c in scenario.checkpoints),
    )
    taken = 0
    while result.runs < runs:
        variant = variants[taken % len(variants)]
        taken += 1
        app = app_factory()
        try:
            episode, breaches = play(app, scenario, variant)
        except Infra as death:
            result.infra += 1
            print(f"[frontier]   ⟳ infra ({result.infra}): {death}")
            if death.rejected or result.infra >= infra_retries:
                break
            continue
        finally:
            app.client.close()
        verdict = grade(episode, scenario.checkpoints)
        result.grades.append(verdict)
        result.breaches.extend(breaches)
        result.samples.append(_sample_record(episode, verdict, breaches))
        mark = "✓" if verdict.whole else "·"
        print(
            f"[frontier]   {mark} {scenario.name}[{variant.name}] "
            f"{verdict.score:.2f} missed={verdict.missed}"
        )
    return result
