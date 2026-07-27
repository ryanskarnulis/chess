"""The eval gate's verdict, unit-tested off the GPU (Sprint 5, slice 3 — audit 23).

`test_agent_evals.py` cannot be TDD'd: every one of its assertions needs a live
12B behind llama-swap, which is exactly why its *verdict* was never tested at
all. `assert result.rate >= 0.8` over 5 samples cannot resolve a 60–100% band,
and the record shows it flapping on unchanged builds — `long_capture[poisoned]`
3/5 then 4/5, `main` 5/5 then 3/5, `play_as_black` 5/5, 2/5, 0/5, 5/5 across
four same-build runs (`docs/agent-evals.md`).

So every decision the harness makes moved into `evalstats.py` — pure functions
over counts and strings, no I/O, no model — and this file is their spec. It runs
in ordinary `pytest` on every commit, which means the *gate itself* is now
regression-tested even though what it gates is not.

Three things are pinned here that the harness previously left to a literal:

- **when a rate is evidence of regression** (a one-sided Wilson bound below the
  floor) versus merely a bad sample (`decide`),
- **what kind of sample it was** (`classify`) — a crashed llama-server is
  infrastructure, not behavior, and the harness must retry it rather than score
  it, while a request the server *rejects* is neither and must not be retried at
  all,
- **what the failures had in common** (`failure_signature`, `block_stability`,
  `RateResult.deterministic_suspect`) — five identical failures are a bug and
  five different ones are variance, and today's harness prints both as "2/5".

A fourth joined them (2026-07-26): **which phase spent the turn's milliseconds**
(`split_latencies`). The readings were always there per call; what was missing
was the attribution, and attribution off a route and a stop reason is exactly the
kind of decision that belongs here rather than in a file no test can run.
"""

import json

import pytest

from evalstats import (
    BUDGET_STOPS,
    DETERMINISTIC_FAILURES,
    NARRATE_ROUTES,
    ROUTE_BRAIN,
    Z_ONE_SIDED_95,
    Attribution,
    Decision,
    Outcome,
    RateResult,
    Stability,
    VacuousRun,
    block_stability,
    classify,
    decide,
    failure_signature,
    scenario_record,
    split_latencies,
    wilson_interval,
)

# --- wilson_interval ----------------------------------------------------------
#
# The bound is one-sided by construction: the FAIL test uses one tail (is the
# *upper* end below the floor?), so the z that makes it a 95% test is 1.645.
# Calling a two-tailed 1.96 bound "95%" would overstate the evidence by a full
# tail, which on a gate means calling a bad sample a regression.

# Upper bounds at n=5 for 0..5 passes, at the module's own z. These six numbers
# are the whole reason a 3/5 escalates instead of going red: see the decision
# table below.
_N5_UPPERS = [0.351, 0.565, 0.728, 0.857, 0.954, 1.000]


@pytest.mark.parametrize(("passed", "upper"), list(enumerate(_N5_UPPERS)))
def test_the_n5_upper_bound_table(passed: int, upper: float) -> None:
    assert wilson_interval(passed, 5)[1] == pytest.approx(upper, abs=0.001)


def test_the_default_z_is_the_one_sided_95_percent_point() -> None:
    # Pinned as a value, not just a name: the operating characteristics recorded
    # in docs/agent-evals.md (and every floor set from them) are this z's.
    assert Z_ONE_SIDED_95 == 1.645


@pytest.mark.parametrize(
    ("passed", "upper"),
    [(0, 0.434), (1, 0.624), (2, 0.769), (3, 0.882), (4, 0.964), (5, 1.000)],
)
def test_z_is_a_parameter_and_a_two_tailed_bound_is_wider(
    passed: int, upper: float
) -> None:
    """The same table at z=1.96, which is where the *plan's* printed n=5 numbers
    (0.435/0.625/0.769/0.882/0.964/1.000) came from.

    Kept as a test rather than a footnote because it is the one place a reader
    can see that the choice of tail is worth ~8 points of upper bound at n=5 —
    and because it proves the widening never flips a decision at floor 0.8 (2/5
    is red and 3/5 escalates under both)."""
    assert wilson_interval(passed, 5, 1.96)[1] == pytest.approx(upper, abs=0.001)


def test_twenty_samples_narrow_the_interval() -> None:
    # The two n=20 counts the escalation cap has to adjudicate: 17/20 is the
    # healthy end of the band and 12/20 is a genuine regression.
    assert wilson_interval(17, 20) == pytest.approx((0.678, 0.938), abs=0.001)
    assert wilson_interval(12, 20) == pytest.approx((0.419, 0.758), abs=0.001)


def test_no_samples_is_no_evidence() -> None:
    """n=0 returns the whole unit interval rather than dividing by zero — a
    scenario every sample of which was an infra death knows nothing, and the
    interval has to say so."""
    assert wilson_interval(0, 0) == (0.0, 1.0)


def test_the_interval_never_leaves_the_unit_range() -> None:
    # A Wilson interval can overshoot at the extremes; a probability that reads
    # as 1.04 in a report is a bug in the report, so both ends are clamped.
    low, high = wilson_interval(5, 5)
    assert high == 1.0
    assert 0.0 < low < 1.0
    low, high = wilson_interval(0, 5)
    assert low == 0.0
    assert 0.0 < high < 1.0


# --- decide -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("passed", "expected"),
    [
        (0, Decision.BELOW_FLOOR),
        (1, Decision.BELOW_FLOOR),
        (2, Decision.BELOW_FLOOR),
        (3, Decision.UNDECIDED),
        (4, Decision.ABOVE_FLOOR),
        (5, Decision.ABOVE_FLOOR),
    ],
)
def test_the_n5_decision_table(passed: int, expected: Decision) -> None:
    """The whole cost argument for the slice lives in this table: a healthy
    scenario (4/5 or 5/5) still costs exactly 5 samples, and a hopeless one
    (0–2) still stops at 5. Only the genuinely ambiguous 3/5 — the count that
    made the old gate a coin flip — pays for another block."""
    assert decide(passed, 5, 0.8) == expected


def test_twenty_samples_resolve_the_band() -> None:
    # At the cap the interval is narrow enough to separate the two cases the
    # 5-sample gate could not.
    assert decide(17, 20, 0.8) == Decision.ABOVE_FLOOR
    assert decide(12, 20, 0.8) == Decision.BELOW_FLOOR


def test_the_point_estimate_alone_decides_a_pass() -> None:
    """ABOVE_FLOOR is the point estimate reaching the floor, not the interval's
    lower end clearing it. A gate that demanded the lower bound would call every
    5-sample scenario a regression — 5/5's lower bound is 0.65."""
    assert wilson_interval(5, 5)[0] < 0.8
    assert decide(5, 5, 0.8) == Decision.ABOVE_FLOOR


def test_the_floor_is_a_parameter() -> None:
    # Each scenario carries its own floor (`_FLOORS`), so nothing may bake 0.8
    # in — and a lower floor buys quiet at the cost of power, which is a
    # decision the docs record rather than a constant.
    assert decide(3, 5, 0.6) == Decision.ABOVE_FLOOR
    assert decide(2, 5, 0.5) == Decision.UNDECIDED
    assert decide(0, 5, 0.5) == Decision.BELOW_FLOOR


def test_z_widens_the_undecided_band() -> None:
    # Documented for the same reason as the interval test above: the z is the
    # gate's one tuning knob, and it moves nothing at the counts that matter.
    assert decide(3, 5, 0.8, z=1.96) == Decision.UNDECIDED
    assert decide(2, 5, 0.8, z=1.96) == Decision.BELOW_FLOOR


# --- classify -----------------------------------------------------------------


def test_a_clean_sample_that_passed_its_check_is_a_pass() -> None:
    assert (
        classify(status_code=200, stop_reason="completed", error=None) == Outcome.PASS
    )


def test_a_transport_failure_is_infrastructure() -> None:
    """The old 502: llama-server crash-restarts every 3–8 minutes of sustained
    generation, and the harness counted each one as a failed sample. That is
    what made the 2026-07-25 baseline a hand-assembled composite of re-runs."""
    assert classify(status_code=502, stop_reason=None, error=None) == Outcome.INFRA


def test_a_provider_error_stop_is_infrastructure_too() -> None:
    """Since audit item 20 the brain *catches* `ProviderError` and answers 200
    with `stop_reason="provider_error"`, so a crash stopped looking like a 502
    and started looking like a silent behavioral miss ("expected a resign call:
    nothing"). It is the same dead socket."""
    assert (
        classify(status_code=200, stop_reason="provider_error", error=None)
        == Outcome.INFRA
    )


def test_infrastructure_beats_a_failed_assertion() -> None:
    """Precedence, and the reason `classify` exists at all: when the provider
    died, the check failed *because* it died. Scoring that as behavior is how a
    crash cadence gets written down as a pass rate."""
    assert (
        classify(
            status_code=200,
            stop_reason="provider_error",
            error=AssertionError("expected a resign call: nothing"),
        )
        == Outcome.INFRA
    )


def test_a_failed_assertion_on_a_healthy_turn_is_a_real_failure() -> None:
    assert (
        classify(
            status_code=200,
            stop_reason="completed",
            error=AssertionError("expected Bxe6, board got None"),
        )
        == Outcome.FAIL
    )


def test_any_other_exception_is_a_harness_bug() -> None:
    """A `KeyError` in a check is not data about the model. Today it kills the
    scenario and silently discards the samples already taken; classified as
    HARNESS it is never retried, never scored, and fails the item outright."""
    assert (
        classify(status_code=200, stop_reason="completed", error=KeyError("tool_calls"))
        == Outcome.HARNESS
    )


def test_a_harness_bug_outranks_a_dead_provider() -> None:
    # A retry cannot fix a typo in the check, so HARNESS is checked first: the
    # infra budget must not be burned on a bug in this repo.
    assert (
        classify(status_code=200, stop_reason="provider_error", error=TypeError("nope"))
        == Outcome.HARNESS
    )


def test_a_budget_stop_on_a_commentary_check_is_inconclusive() -> None:
    """The vacuous pass, from the record: three of four `hints_off_no_advice`
    passes in one run were `max_iterations` stops. A budget stop reaches no
    narrator, so there is no commentary to leak and a purely negative check
    scores a pass for never having been tested."""
    assert (
        classify(
            status_code=200,
            stop_reason="max_iterations",
            error=VacuousRun("stopped on max_iterations: no commentary to check"),
        )
        == Outcome.INCONCLUSIVE
    )


def test_a_budget_stop_is_not_vacuous_by_itself() -> None:
    """Only a check that *depends* on commentary can pass vacuously. A scenario
    with a positive assertion (a tool ran, the board reads d4) is answered by
    the tool results, whatever the loop stopped on — so a budget stop alone is
    an ordinary pass and INCONCLUSIVE stays a claim the check has to make."""
    assert (
        classify(status_code=200, stop_reason="max_iterations", error=None)
        == Outcome.PASS
    )


def test_vacuous_still_fails_if_nobody_classifies_it() -> None:
    # `VacuousRun` subclasses AssertionError deliberately: a caller that skips
    # `classify` gets a failed sample rather than a silent pass.
    assert issubclass(VacuousRun, AssertionError)


# --- classify: a death the retry cannot fix -----------------------------------
#
# INFRA is "ask again" — the crash cadence llama-server has under sustained
# load. But a `provider_error` can equally be a request the server refuses
# identically every time (an HTTP 400: a context overrun on a long transcript),
# and asking again just spends the infra budget to arrive at the same place five
# samples later. The brain now names which one it was
# (`AgentResponse.provider_failure`), so `classify` can stop guessing.


def test_a_rejected_request_is_not_retried() -> None:
    assert (
        classify(
            status_code=200,
            stop_reason="provider_error",
            error=None,
            provider_failure="rejected",
        )
        == Outcome.PROVIDER_REJECTED
    )


def test_a_malformed_response_is_not_retried_either() -> None:
    """A 200 whose body fails wire validation is a version skew between this
    app and llama-server, not a crash — the next request gets the same body."""
    assert (
        classify(
            status_code=200,
            stop_reason="provider_error",
            error=None,
            provider_failure="malformed_response",
        )
        == Outcome.PROVIDER_REJECTED
    )


def test_a_dead_server_is_still_retried() -> None:
    for failure in ("unreachable", "server_error"):
        assert (
            classify(
                status_code=200,
                stop_reason="provider_error",
                error=None,
                provider_failure=failure,
            )
            == Outcome.INFRA
        ), failure


def test_an_unnamed_provider_death_is_retried() -> None:
    """The pre-existing shape, and every non-200 (which carries no failure kind
    at all — the app answered, not the provider). Unknown means retry: the
    expensive mistake is aborting a suite on a server that was merely
    restarting."""
    assert (
        classify(status_code=200, stop_reason="provider_error", error=None)
        == Outcome.INFRA
    )
    assert classify(status_code=502, stop_reason=None, error=None) == Outcome.INFRA


def test_a_harness_bug_still_outranks_a_rejected_request() -> None:
    # Same precedence as INFRA: a KeyError in a check is not evidence about
    # anything the provider did.
    assert (
        classify(
            status_code=200,
            stop_reason="provider_error",
            error=TypeError("nope"),
            provider_failure="rejected",
        )
        == Outcome.HARNESS
    )


def test_the_two_failure_vocabularies_agree() -> None:
    """`evalstats` names the non-transient kinds itself rather than importing
    them — it must stay importable with nothing installed, the same reason
    `STOP_PROVIDER_ERROR` is a literal here. That buys a drift risk, so this
    test is the thing that pays for it: the set this module refuses to retry is
    exactly the set `provider.py` calls non-transient, by construction."""
    from chessapp.provider import ProviderFailure

    assert DETERMINISTIC_FAILURES == {
        str(failure) for failure in ProviderFailure if not failure.transient
    }


# --- failure_signature --------------------------------------------------------


def test_failures_differing_only_in_moves_share_a_signature() -> None:
    """Five failures with five different SANs are one failure mode, and the
    harness must be able to say so — "2/5, both the same mode" and "2/5, two
    unrelated modes" are different findings printed identically today."""
    first = failure_signature("expected Bxe6, board got 'Bxh6'")
    second = failure_signature("expected Bxe4, board got 'Nf3'")
    assert first == second


def test_squares_and_counts_collapse_too() -> None:
    assert failure_signature("run 1: analyzed c3 after 12 plies") == failure_signature(
        "run 4: analyzed d4 after 8 plies"
    )


def test_genuinely_different_failures_stay_apart() -> None:
    assert failure_signature("expected a resign call: nothing") != failure_signature(
        "expected Bxe6, board got None"
    )


def test_a_signature_is_stable_and_readable() -> None:
    # It goes in the report and the printed summary, so it has to stay something
    # a human recognises — normalised, not hashed.
    assert "resign" in failure_signature("expected a resign call: nothing")


# --- block_stability ----------------------------------------------------------


def test_a_five_five_then_zero_five_pair_is_unstable() -> None:
    """The clustering confound, made visible. `play_as_black` measured 5/5 in
    isolation and 0/5 mid-suite on one build: 20 consecutive samples of one
    utterance is a cluster of size 1, so the Wilson width understates the truth.
    The slice does not fix that — it flags it."""
    assert block_stability([(5, 5), (0, 5)]) == Stability.UNSTABLE


def test_ordinary_sampling_spread_is_stable() -> None:
    # 5/5 vs 3/5 is what a true rate near the floor looks like
    # (`long_capture[poisoned]`), and flagging that would flag everything.
    assert block_stability([(5, 5), (3, 5)]) == Stability.STABLE


def test_one_block_cannot_be_unstable() -> None:
    assert block_stability([(3, 5)]) == Stability.STABLE
    assert block_stability([]) == Stability.STABLE


def test_empty_blocks_do_not_count_as_zero_rates() -> None:
    # A block every sample of which was retried for infra has no rate; treating
    # it as 0/5 would invent an instability that never happened.
    assert block_stability([(5, 5), (0, 0)]) == Stability.STABLE


# --- RateResult ---------------------------------------------------------------


def test_rate_result_derives_its_own_verdict() -> None:
    """The derived fields are built, never passed: the same rule `turn_record`
    follows for `model_ms`. A report whose interval disagrees with its counts is
    worse than no report."""
    result = RateResult.from_blocks(
        blocks=[(5, 5), (3, 5)],
        failures=["run 7: expected Bxe6, board got None"],
        floor=0.8,
    )
    assert result.passed == 8
    assert result.runs == 10
    assert result.rate == 0.8
    assert result.blocks == ((5, 5), (3, 5))
    assert result.decision == Decision.ABOVE_FLOOR
    assert result.stability == Stability.STABLE
    assert result.interval == wilson_interval(8, 10)


def test_a_scenario_below_the_floor_reads_as_a_regression() -> None:
    result = RateResult.from_blocks(
        blocks=[(1, 5)],
        failures=[f"run {i}: expected Bxe6, board got None" for i in range(1, 5)],
        floor=0.8,
    )
    assert result.decision == Decision.BELOW_FLOOR
    assert result.rate == 0.2


def test_failure_modes_are_a_histogram_on_every_scenario() -> None:
    result = RateResult.from_blocks(
        blocks=[(1, 5)],
        failures=[
            "expected Bxe6, board got 'Bxh6'",
            "expected Bxe6, board got 'Nf3'",
            "expected a resign call: nothing",
            "expected a resign call: nothing",
        ],
        floor=0.8,
    )
    assert sorted(result.failure_modes.values()) == [2, 2]
    assert len(result.failure_modes) == 2


def test_a_uniform_total_failure_is_a_deterministic_suspect() -> None:
    """0/5 with one signature is not sampling noise — it is a broken build, a
    changed tool offer, or a deterministic `provider_error` (a context overrun,
    a malformed request). Naming it is what stops the next reader arguing about
    variance for an afternoon."""
    result = RateResult.from_blocks(
        blocks=[(0, 5)],
        failures=["expected Bxe6, board got 'Bxh6'"] * 5,
        floor=0.8,
    )
    assert result.deterministic_suspect is True


def test_mixed_failures_are_not_a_deterministic_suspect() -> None:
    result = RateResult.from_blocks(
        blocks=[(0, 5)],
        failures=["expected Bxe6, board got None", "expected a resign call: nothing"],
        floor=0.8,
    )
    assert result.deterministic_suspect is False


def test_a_partial_pass_is_not_a_deterministic_suspect() -> None:
    result = RateResult.from_blocks(
        blocks=[(2, 5)],
        failures=["expected Bxe6, board got None"] * 3,
        floor=0.8,
    )
    assert result.deterministic_suspect is False


def test_the_summary_line_carries_the_verdict_and_its_cost() -> None:
    """The `[eval]` line a human reads off the terminal. It has to carry the
    interval (the point of the slice) and the infra retries (a rate taken over a
    crashing server is worth less than the same rate taken clean) — printed even
    at 1, because that is the number the baseline's re-run procedure used to
    hide."""
    result = RateResult.from_blocks(
        blocks=[(4, 5)], failures=["expected Bxe6, board got None"], floor=0.8, infra=1
    )
    summary = result.summary()
    assert "4/5" in summary
    assert "0.44" in summary and "0.95" in summary  # the interval, both ends
    assert "infra=1" in summary
    assert str(Decision.ABOVE_FLOOR) in summary


def test_zero_runs_has_a_rate_of_zero_rather_than_an_exception() -> None:
    # An all-infra scenario is reported, not crashed on: the harness fails the
    # item loudly and the report still has to serialise.
    result = RateResult.from_blocks(blocks=[], failures=[], floor=0.8, infra=5)
    assert result.rate == 0.0
    assert result.runs == 0
    assert result.interval == (0.0, 1.0)


# --- scenario_record ----------------------------------------------------------


def test_a_scenario_record_is_json_and_carries_the_whole_verdict() -> None:
    """One JSONL line per scenario is what stops the baseline tables in
    docs/agent-evals.md being hand-transcribed from terminal scrollback."""
    result = RateResult.from_blocks(
        blocks=[(5, 5), (2, 5)],
        failures=["run 8: expected Bxe6, board got None"],
        floor=0.8,
        infra=2,
        inconclusive=1,
    )
    record = scenario_record("long_capture[poisoned]", result, 0.8, blocking=True)

    decoded = json.loads(json.dumps(record))  # must survive the wire unchanged
    assert decoded["scenario"] == "long_capture[poisoned]"
    assert decoded["floor"] == 0.8
    assert decoded["passed"] == 7
    assert decoded["runs"] == 10
    assert decoded["blocks"] == [[5, 5], [2, 5]]
    assert decoded["decision"] == "UNDECIDED"
    assert decoded["stability"] == "UNSTABLE"
    assert decoded["interval"] == [pytest.approx(x) for x in result.interval]
    assert decoded["infra"] == 2
    assert decoded["inconclusive"] == 1
    assert decoded["failure_modes"] == {"run <n>: expected <san>, board got None": 1}
    assert decoded["blocking"] is True  # arbitrary metadata rides along


def test_metadata_can_annotate_but_the_counts_come_from_the_result() -> None:
    result = RateResult.from_blocks(blocks=[(5, 5)], failures=[], floor=0.8)
    record = scenario_record("play_as_black", result, 0.8, samples=[{"model_ms": 940}])
    assert record["samples"] == [{"model_ms": 940}]
    assert record["rate"] == 1.0


# --- split_latencies ----------------------------------------------------------
#
# `model_ms` is a per-turn *sum*, and the question it cannot answer is the one
# Sprint 5 is asking: a `no_progress` turn was measured narrating 2–3× slower
# than an ordinary one at the same model-call count, and a turn total cannot
# tell a slow planner from a slow narrator. The readings themselves already
# exist per call, in call order, on `AgentResponse.model_latencies_ms` and in
# the trace — so the only thing missing was the attribution, which is what these
# tests pin. It is derived from the route and the stop reason, and it refuses to
# guess: a turn whose phase boundary is not knowable says so.


def test_a_brain_turn_splits_into_the_planner_and_the_narrator() -> None:
    """The narrator is the *last* call on a brain-routed turn that reached one —
    that is the phase order, not a heuristic (`docs/planner-narrator.md`): the
    planner loops, then exactly one tool-free call speaks."""
    latencies = split_latencies(
        (1200, 900, 39100), route="brain", stop_reason="completed"
    )
    assert latencies.attribution is Attribution.SPLIT
    assert latencies.planner_ms == 2100
    assert latencies.narrator_ms == 39100
    assert latencies.total_ms == 41200  # and the parts still sum to the whole


def test_a_repeat_stop_still_reached_its_narrator() -> None:
    """The case this function was written for. `no_progress` ends the *planning*
    phase, and the narrator still runs (CLAUDE.md), so the split is knowable —
    which is what turns "a repeat-stop turn narrates slower" from a hypothesis
    into a measurement."""
    latencies = split_latencies((1100, 1000), route="brain", stop_reason="no_progress")
    assert latencies.attribution is Attribution.SPLIT
    assert latencies.planner_ms == 1100
    assert latencies.narrator_ms == 1000


def test_a_budget_stop_spent_every_millisecond_in_the_planner() -> None:
    """A budget stop reaches no narrator by construction, so attributing its last
    call to one would invent narrator time that was never spent."""
    for stop in sorted(BUDGET_STOPS):
        latencies = split_latencies(
            (900, 800, 850, 870), route="brain", stop_reason=stop
        )
        assert latencies.attribution is Attribution.NO_NARRATOR, stop
        assert latencies.planner_ms == 3420, stop
        assert latencies.narrator_ms is None, stop


def test_a_provider_death_leaves_the_phase_boundary_unknown() -> None:
    """The call that died could have been either phase — a planner round trip or
    the narrator's — and the trace does not say which. Unknown is reported as
    unknown: a median computed over guesses is worse than one computed over
    fewer samples."""
    latencies = split_latencies(
        (1200, 30000), route="brain", stop_reason="provider_error"
    )
    assert latencies.attribution is Attribution.UNKNOWN
    assert latencies.planner_ms is None
    assert latencies.narrator_ms is None
    assert latencies.total_ms == 31200  # the total is still a fact


def test_a_narrate_route_has_no_planner_phase_at_all() -> None:
    """The fast path, a drag, a resignation and a confirmation reach the model
    only through `Brain.narrate` — one narrator call, no planner loop — so all of
    the time is the narrator's and the planner's zero is real, not unknown."""
    for route in sorted(NARRATE_ROUTES):
        latencies = split_latencies((3400,), route=route, stop_reason="completed")
        assert latencies.attribution is Attribution.SPLIT, route
        assert latencies.planner_ms == 0, route
        assert latencies.narrator_ms == 3400, route


def test_a_zero_llm_turn_reports_no_readings_rather_than_zeros() -> None:
    """A canned confirmation at verbosity=low never called the model. An
    unmeasured phase is not a fast one — the same rule `AgentResponse` follows
    for an unmeasured brain."""
    latencies = split_latencies((), route="fast_path", stop_reason="completed")
    assert latencies.attribution is Attribution.NONE
    assert latencies.planner_ms is None
    assert latencies.narrator_ms is None
    assert latencies.total_ms == 0


def test_an_empty_trace_record_is_no_readings_not_a_bad_route() -> None:
    # A request that failed before the pipeline reached its trace point leaves
    # `_CollectingTracer.last` empty, so route and stop reason are both None.
    latencies = split_latencies((), route=None, stop_reason=None)
    assert latencies.attribution is Attribution.NONE


def test_a_route_this_module_does_not_know_is_unknown_not_assumed() -> None:
    """New route, new phase shape — and guessing it would silently mis-attribute
    every sample on it. `test_the_route_vocabularies_agree` is what makes this
    branch reachable only by a *new* route rather than by drift."""
    latencies = split_latencies((500,), route="telepathy", stop_reason="completed")
    assert latencies.attribution is Attribution.UNKNOWN


def test_the_route_vocabularies_agree() -> None:
    """Same drift payment as `test_the_two_failure_vocabularies_agree`: this
    module names the routes itself so it stays importable with nothing
    installed, and this test is what keeps that copy honest. Every route the
    tracer can record is classified — a route in neither set would attribute
    to UNKNOWN forever without anyone noticing."""
    from chessapp import trace

    recorded = {
        value
        for name, value in vars(trace).items()
        if name.startswith("ROUTE_") and isinstance(value, str)
    }
    assert recorded == NARRATE_ROUTES | {ROUTE_BRAIN}
    assert not NARRATE_ROUTES & {ROUTE_BRAIN}


def test_the_summary_names_the_phases_and_the_unknowns() -> None:
    """The `[eval]` line clause a human reads. An unknown prints as `?` rather
    than as a number, because the whole point is that it is not one."""
    split = split_latencies((1200, 900, 39100), route="brain", stop_reason="completed")
    assert split.summary() == (
        "call_ms=[1200,900,39100] planner_ms=2100 narrator_ms=39100 phases=SPLIT"
    )
    stuck = split_latencies((900, 800), route="brain", stop_reason="max_iterations")
    assert stuck.summary() == (
        "call_ms=[900,800] planner_ms=1700 narrator_ms=? phases=NO_NARRATOR"
    )
    silent = split_latencies((), route="fast_path", stop_reason="completed")
    assert silent.summary() == "call_ms=[] planner_ms=? narrator_ms=? phases=NONE"


def test_the_record_is_json_and_carries_the_total_beside_the_parts() -> None:
    """The per-sample half: `docs/agent-evals.md` has claimed per-sample
    `model_ms` since the statistics slice and the report never carried one, so a
    campaign's medians came out of terminal scrollback. The total ships beside
    the parts, derived here for the same reason `turn_record` derives it — a
    record whose total disagrees with its parts is worse than no record."""
    latencies = split_latencies(
        (1200, 900, 39100), route="brain", stop_reason="completed"
    )
    decoded = json.loads(json.dumps(latencies.as_record()))
    assert decoded == {
        "model_ms": 41200,
        "call_ms": [1200, 900, 39100],
        "planner_ms": 2100,
        "narrator_ms": 39100,
        "phases": "SPLIT",
    }
    unknown = split_latencies((1200,), route="brain", stop_reason="provider_error")
    assert json.loads(json.dumps(unknown.as_record()))["narrator_ms"] is None
