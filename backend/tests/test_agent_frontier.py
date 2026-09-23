"""The frontier tier, live (#318): hard scenarios, measured and never gated.

Opt-in like the gate, and never in CI — it needs the GPU:

    cd backend
    CHESSAPP_AGENT_FRONTIER=1 CHESSAPP_EVAL_REPORT=/tmp/frontier.jsonl \\
        .venv/bin/pytest tests/test_agent_frontier.py -v -s

A scenario's pass count is **reported, not asserted**: a low number is the
measurement this tier exists for. An item fails only when the number could
not be measured or cannot be trusted — every sample an infrastructure death, a
deterministic invariant breached on a live sample (a bug in the app, not the
model), or a broken scenario (`frontier.ScenarioError`).

Knobs: `CHESSAPP_FRONTIER_RUNS` (samples per scenario, default 10),
`CHESSAPP_FRONTIER_SPLIT` (`dev`, the default, or `heldout`), and the gate's
own `CHESSAPP_EVAL_REPORT`, `LLAMACPP_*` and `CHESSAPP_STOCKFISH`. The report
opens with a header naming the build, model, sampling and the shas of both
prompts and of the tool offer, so each number is tied to what produced it.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Generator
from datetime import UTC, datetime

import pytest

from chessapp.coordinator import TurnCoordinator
from chessapp.game import GameSession
from chessapp.personality import PLANNER_PROMPT, system_prompt_for
from chessapp.tools import ToolContext, brain_tool_definitions, build_registry
from frontier import Scenario, measure
from frontier_corpus import SCENARIOS
from test_agent_evals import (  # noqa: F401 - `engine` is a fixture
    LLAMACPP_MODEL,
    PLANNER_TEMPERATURE,
    _build_eval_app,
    _git_sha,
    _report,
    engine,
)

pytestmark = pytest.mark.skipif(
    os.environ.get("CHESSAPP_AGENT_FRONTIER") != "1",
    reason="live frontier evals: set CHESSAPP_AGENT_FRONTIER=1 (needs the GPU)",
)

RUNS = int(os.environ.get("CHESSAPP_FRONTIER_RUNS", "10"))
SPLIT = os.environ.get("CHESSAPP_FRONTIER_SPLIT", "dev")


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def configuration() -> dict[str, str]:
    """The shas of what the model is shown: both prompts, and the tool offer
    on an opening board (the same `brain_tool_definitions` assembly uses)."""
    ctx = ToolContext(session=GameSession())
    registry = build_registry(ctx, TurnCoordinator(ctx), atomic_exchange=False)
    offer = brain_tool_definitions(registry, ctx)
    return {
        "planner_prompt_sha": _sha(PLANNER_PROMPT),
        "narrator_prompt_sha": _sha(system_prompt_for("normal")),
        "offer_sha": _sha(json.dumps(offer, sort_keys=True)),
    }


@pytest.fixture(scope="module", autouse=True)
def _frontier_header() -> Generator[None, None, None]:
    _report(
        {
            "kind": "frontier_header",
            "started": datetime.now(UTC).isoformat(),
            "git_sha": _git_sha(),
            "model": LLAMACPP_MODEL,
            "planner_temperature": PLANNER_TEMPERATURE,
            "runs": RUNS,
            "split": SPLIT,
            **configuration(),
        }
    )
    yield


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s.name)
def test_frontier(scenario: Scenario, engine) -> None:  # noqa: F811
    result = measure(
        scenario,
        runs=RUNS,
        split=SPLIT,
        app_factory=lambda: _build_eval_app(engine),
    )
    print(
        f"\n[frontier] scenario={scenario.name} tier={scenario.tier} "
        f"split={SPLIT} {result.summary()}"
    )
    for name, hits in result.checkpoint_hits.items():
        print(f"[frontier]   {hits}/{result.runs} {name}")
    for mode, count in result.failure_modes.most_common():
        print(f"[frontier]   ×{count} {mode}")
    _report(result.record(why=scenario.why))
    if result.breaches:
        pytest.fail(
            "the app broke an invariant on a live sample — a bug to fix, not a "
            "score:\n" + "\n".join(result.breaches)
        )
    if result.runs == 0:
        pytest.fail(
            f"{scenario.name}: no sample was measured ({result.infra} "
            "infrastructure deaths) — llama-server is not staying up"
        )
