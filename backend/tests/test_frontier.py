"""The frontier tier's own wiring, tested without a GPU (#318).

`test_agent_frontier.py` needs a live model for every number it reports, but
nothing that turns a sample into that number does: the grader, the result, the
report record, the runner's handling of deaths, breaches and misrouted steps,
and the corpus's own rules. A measurement instrument that could silently lie
about what it measured is worse than none — the gate's harness-bug precedent —
so all of it runs here, in CI, over the shipped app with a scripted provider.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

import frontier
from chessapp.api import create_app, narrator_facts, planner_board_refresh
from chessapp.coordinator import TurnCoordinator
from chessapp.fastparse import parse_confirmation, parse_move, parse_resign
from chessapp.game import GameSession
from chessapp.llama_brain import create_llama_brain
from chessapp.personality import PLANNER_PROMPT, system_prompt_for
from chessapp.provider import ProviderError
from chessapp.tools import ToolContext, brain_tool_definitions, build_registry
from fakes import CountingProvider, ScriptedProvider, text_turn, tool_calls_turn
from frontier import (
    Checkpoint,
    Episode,
    FrontierResult,
    Grade,
    Say,
    Scenario,
    ScenarioError,
    Turn,
    Variant,
    grade,
    measure,
)
from frontier_corpus import SCENARIOS, after_e4_e5
from test_agent_evals import EvalApp, _CollectingTracer
from trajectory import LegalEngine

START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


# --- a scripted app, assembled the way the live one is --------------------------


def scripted_factory(*turns):
    """An `app_factory` whose apps are the eval assembly over a scripted
    provider (every sample replays `turns` from the start)."""

    def build() -> EvalApp:
        ctx = ToolContext(session=GameSession(), engine=LegalEngine())
        coordinator = TurnCoordinator(ctx)
        registry = build_registry(ctx, coordinator, atomic_exchange=False)
        provider = CountingProvider(ScriptedProvider(*turns))
        brain = create_llama_brain(
            base_url="http://unused",
            model="scripted",
            dispatcher=registry,
            tool_definitions=lambda: brain_tool_definitions(registry, ctx),
            system_prompt_provider=lambda: system_prompt_for(ctx.settings.verbosity),
            planner_prompt_provider=lambda: PLANNER_PROMPT,
            provider=provider,
            board_refresh=lambda: planner_board_refresh(ctx, coordinator),
            narrator_facts=lambda: narrator_facts(ctx, coordinator),
        )
        tracer = _CollectingTracer()
        client = TestClient(
            create_app(
                ctx,
                brain=brain,
                registry=registry,
                coordinator=coordinator,
                tracer=tracer,
            )
        )
        return EvalApp(client=client, ctx=ctx, provider=provider, tracer=tracer)

    return build


UNDO_D4_JUDGE = (
    tool_calls_turn(
        ("undo", {}),
        ("make_move", {"move": "d4"}),
        ("evaluate_position", {}),
    ),
    text_turn("Okay."),
)

CHECKS = (
    Checkpoint("took_back", lambda e: bool(e.turn(1).ran("undo"))),
    Checkpoint("played_d4", lambda e: e.history()[:1] == ["d4"]),
)


def scenario(*variants: Variant, checkpoints=CHECKS) -> Scenario:
    return Scenario(
        name="probe",
        tier=1,
        why="test",
        dev=variants,
        heldout=variants,
        checkpoints=checkpoints,
    )


def variant(name="v", text="take it back and play d4, then judge it"):
    return Variant(name, (Say(text),), after_e4_e5)


# --- grading ------------------------------------------------------------------------


def _episode(**after: Any) -> Episode:
    state = {"game_id": "g", "fen": START, "history": after.get("history", [])}
    turn = Turn(
        say=Say("x"),
        status=200,
        route="brain",
        stop_reason="completed",
        results=after.get("results", []),
        commentary="",
        before=state,
        after=state,
        model_calls=2,
        seconds=1.0,
    )
    return Episode(variant="v", start=state, turns=[turn])


def test_a_grade_is_every_checkpoint_on_its_own():
    verdict = grade(
        _episode(history=["d4", "d5"], results=[]),
        CHECKS,
    )

    assert verdict.hits == {"took_back": False, "played_d4": True}
    assert not verdict.whole
    assert verdict.score == 0.5
    assert verdict.missed == ["took_back"]


def test_a_checkpoint_that_raises_is_a_broken_scenario_not_a_miss():
    broken = Checkpoint("broken", lambda e: e.turn(9).asked)

    with pytest.raises(ScenarioError, match="broken"):
        grade(_episode(), (broken,))


def test_the_result_reports_counts_interval_rubric_and_modes():
    result = FrontierResult(
        scenario="s", tier=2, split="heldout", checkpoints=("a", "b")
    )
    result.grades = [
        Grade({"a": True, "b": True}),
        Grade({"a": True, "b": False}),
        Grade({"a": False, "b": False}),
    ]

    assert (result.runs, result.passed) == (3, 1)
    assert result.rubric == pytest.approx(0.5)
    assert result.checkpoint_hits == {"a": 2, "b": 1}
    assert result.failure_modes == {"missed b": 1, "missed a, b": 1}
    low, high = result.interval
    assert 0 < low < 1 / 3 < high < 1
    record = result.record(why="w")
    assert record["kind"] == "frontier"
    assert record["split"] == "heldout" and record["why"] == "w"
    assert record["passed"] == 1 and record["runs"] == 3


def test_an_unknown_split_is_refused():
    with pytest.raises(ValueError):
        scenario(variant()).variants("train")


# --- the runner, end to end over a scripted provider ---------------------------------


def test_measure_takes_exactly_the_runs_asked_for_and_grades_each():
    result = measure(
        scenario(variant()),
        runs=3,
        split="dev",
        app_factory=scripted_factory(*UNDO_D4_JUDGE),
    )

    assert (result.runs, result.passed, result.infra) == (3, 3, 0)
    assert result.breaches == []
    (turn,) = result.samples[0]["turns"]
    assert turn["route"] == "brain"
    assert turn["tools"][:2] == ["undo", "make_move"]
    assert turn["history"][0] == "d4"


def test_samples_cycle_through_the_splits_variants():
    result = measure(
        scenario(variant("a"), variant("b")),
        runs=3,
        split="dev",
        app_factory=scripted_factory(*UNDO_D4_JUDGE),
    )

    assert [s["variant"] for s in result.samples] == ["a", "b", "a"]


def test_a_low_score_is_measured_not_raised():
    wrong = (tool_calls_turn(("make_move", {"move": "Nf3"})), text_turn("Okay."))

    result = measure(
        scenario(variant()), runs=2, split="dev", app_factory=scripted_factory(*wrong)
    )

    assert (result.runs, result.passed) == (2, 0)
    assert result.checkpoint_hits == {"took_back": 0, "played_d4": 0}


def test_a_breached_invariant_is_recorded_against_the_sample(monkeypatch):
    def always(observed):
        raise frontier.InvariantBreach("planted")

    monkeypatch.setattr(frontier, "INVARIANTS", (always,))

    result = measure(
        scenario(variant()),
        runs=1,
        split="dev",
        app_factory=scripted_factory(*UNDO_D4_JUDGE),
    )

    assert len(result.breaches) == 1 and "planted" in result.breaches[0]
    assert result.samples[0]["breaches"] == result.breaches


def test_the_live_invariants_see_what_each_request_sent(monkeypatch):
    """The offer invariant (#315) reads the board a request showed; a run whose
    requests were not recorded would pass it for having looked at nothing."""
    seen: list[int] = []
    monkeypatch.setattr(
        frontier,
        "INVARIANTS",
        (lambda observed: seen.append(len(observed.provider_calls)),),
    )

    measure(
        scenario(variant()),
        runs=1,
        split="dev",
        app_factory=scripted_factory(*UNDO_D4_JUDGE),
    )

    assert seen and seen[0] >= 2, "the planner's requests and the narrator's"


def test_a_model_step_the_parser_answered_is_a_broken_scenario():
    parsed = Variant("fast", (Say("d4"),), after_e4_e5)

    with pytest.raises(ScenarioError, match="route"):
        measure(
            scenario(parsed),
            runs=1,
            split="dev",
            app_factory=scripted_factory(text_turn("Okay.")),
        )


def test_a_provider_death_is_retaken_then_counted_not_scored():
    dead = ProviderError("server gone")

    result = measure(
        scenario(variant()),
        runs=3,
        split="dev",
        app_factory=scripted_factory(dead),
        infra_retries=2,
    )

    assert (result.runs, result.infra) == (0, 2)


# --- the corpus's own rules ---------------------------------------------------------


def test_every_scenario_is_well_formed():
    names = [s.name for s in SCENARIOS]
    assert len(names) == len(set(names))
    for s in SCENARIOS:
        assert s.tier in (1, 2, 3), s.name
        assert s.why, s.name
        assert s.dev and s.heldout, f"{s.name} needs both splits"
        assert s.checkpoints, s.name
        checkpoint_names = [c.name for c in s.checkpoints]
        assert len(checkpoint_names) == len(set(checkpoint_names)), s.name


def test_heldout_wordings_are_not_dev_wordings():
    for s in SCENARIOS:
        # Only what the model reads: a bare "yes" or a SAN is the same on
        # both splits by design, and no parser answer is tuned against.
        dev = {say.text for v in s.dev for say in v.says if say.model}
        heldout = {say.text for v in s.heldout for say in v.says if say.model}
        assert not dev & heldout, s.name


class _SetupOnly:
    """Enough of an `EvalApp` for a variant's setup: a context and nothing
    that talks to a model."""

    def __init__(self) -> None:
        self.ctx = ToolContext(session=GameSession())


def _variants():
    for s in SCENARIOS:
        for v in s.dev + s.heldout:
            yield pytest.param(s, v, id=f"{s.name}[{v.name}]")


@pytest.mark.parametrize(("s", "v"), _variants())
def test_no_model_step_is_a_parser_utterance_on_its_own_board(s, v):
    """The route pin catches this live; this catches it before a GPU is spent,
    on the board the variant's setup leaves (and on the opening, which a reset
    returns to)."""
    app = _SetupOnly()
    if v.setup is not None:
        v.setup(app)
    for fen in {app.ctx.session.fen(), START}:
        for say in v.says:
            if not say.model:
                continue
            assert parse_move(say.text, fen) is None, say.text
            assert not parse_resign(say.text), say.text
            assert parse_confirmation(say.text) is None, say.text


@pytest.mark.parametrize(("s", "v"), _variants())
def test_every_checkpoint_grades_an_episode_of_its_shape(s, v):
    """A checkpoint that raises is a broken scenario (`ScenarioError`), and
    finding that out mid-baseline wastes a GPU run: each one is graded here on
    an episode with the variant's turn count and nothing done."""
    app = _SetupOnly()
    if v.setup is not None:
        v.setup(app)
    state = {
        "game_id": "g",
        "fen": app.ctx.session.fen(),
        "history": app.ctx.session.move_history(),
        "game_over": False,
    }
    settings = app.ctx.settings.snapshot()
    turns = [
        Turn(
            say=say,
            status=200,
            route="brain",
            stop_reason="completed",
            results=[],
            commentary="",
            before=state,
            after=state,
            model_calls=2,
            seconds=1.0,
            settings=settings,
        )
        for say in v.says
    ]
    episode = Episode(variant=v.name, start=state, turns=turns, start_settings=settings)

    verdict = grade(episode, s.checkpoints)

    assert not verdict.whole, f"{s.name}: doing nothing must not pass"
