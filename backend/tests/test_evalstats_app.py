"""Chess's half of the eval gate, unit-tested off the GPU, and the drift guards
tying the vendored half to chess's own loop and provider.

`evalstats.py` and its spec `test_evalstats.py` are vendored verbatim from
agent-standard and import nothing of the app, so the literals they read — the
loop's failure stops, the statuses, the provider's failure kinds and its error
text — are pinned here against the code that produces them, read off that
code or driven through it rather than restated. Beside them is the spec of
`evalstats_app.py`, the module chess keeps on top of the vendored one:
`VacuousRun` and the INCONCLUSIVE outcome, the SAN-aware failure signature and
the `RateResult` built from it, and the phase attribution of a turn's
milliseconds and tokens (`split_latencies`, `split_tokens`).

Canonical is imported as `evalstats`, the one name chess code uses for it
(`evalstats_app` says why); the last test here holds every chess file to that.
"""

from __future__ import annotations

import ast
import inspect
import json
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest

import evalstats
import evalstats_app
from chessapp import brain, llama_brain, trace
from chessapp.brain import PHASE_NARRATOR, PHASE_PLANNER, PHASE_UNKNOWN
from chessapp.llama_brain import LlamaBrain
from chessapp.provider import LlamaCppProvider, ProviderError, ProviderFailure
from evalstats import (
    DETERMINISTIC_FAILURES,
    INFRA_STATUSES,
    INFRA_STOPS,
    STATUS_RATE_LIMITED,
    Decision,
    Stability,
    provider_failure_kind,
    wilson_interval,
)
from evalstats_app import (
    _KNOWN_PHASES,
    _NARRATOR_PHASES,
    _PLANNER_PHASE,
    BUDGET_STOPS,
    NARRATE_ROUTES,
    ROUTE_BRAIN,
    ROUTE_CONFIRMATION,
    STOP_PROVIDER_ERROR,
    Attribution,
    Outcome,
    RateResult,
    VacuousRun,
    classify,
    failure_signature,
    generation_rate,
    scenario_record,
    split_latencies,
    split_tokens,
)

# --- Outcome and classify -----------------------------------------------------
#
# Canonical's `Outcome` has no INCONCLUSIVE and a StrEnum cannot be extended,
# so chess's is its own class, value for value plus the one, and chess's
# `classify` is canonical's plus the one reading that produces it. Everything
# else must be canonical's verdict exactly.


def test_outcome_is_canonicals_value_for_value_plus_inconclusive() -> None:
    assert {str(outcome) for outcome in Outcome} == {
        str(outcome) for outcome in evalstats.Outcome
    } | {"INCONCLUSIVE"}
    for outcome in evalstats.Outcome:
        # Equal as strings — the report records outcomes as strings — and never
        # the same object, which is why chess takes both names from one module.
        assert Outcome(outcome.value) == outcome
        assert Outcome(outcome.value) is not outcome


_NOT_VACUOUS = [
    pytest.param(200, "completed", None, None, id="pass"),
    pytest.param(200, "completed", AssertionError("missed"), None, id="miss"),
    pytest.param(200, "completed", KeyError("tool_calls"), None, id="harness"),
    pytest.param(200, "max_iterations", AssertionError("x"), None, id="limit-stop"),
    pytest.param(200, STOP_PROVIDER_ERROR, None, None, id="death"),
    pytest.param(200, STOP_PROVIDER_ERROR, None, "rejected", id="rejected"),
    pytest.param(502, None, None, None, id="transport"),
    pytest.param(429, None, None, None, id="rate-limited"),
    pytest.param(500, "completed", None, None, id="server-error"),
]


@pytest.mark.parametrize(("status", "stop", "error", "failure"), _NOT_VACUOUS)
def test_classify_is_canonicals_wherever_nothing_was_vacuous(
    status: int, stop: str | None, error: BaseException | None, failure: str | None
) -> None:
    sample: dict[str, Any] = {
        "status_code": status,
        "stop_reason": stop,
        "error": error,
        "provider_failure": failure,
    }
    outcome = classify(**sample)
    assert isinstance(outcome, Outcome)
    assert outcome == evalstats.classify(**sample)


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
        is Outcome.INCONCLUSIVE
    )


def test_a_budget_stop_is_not_vacuous_by_itself() -> None:
    """Only a check that *depends* on commentary can pass vacuously. A scenario
    with a positive assertion (a tool ran, the board reads d4) is answered by
    the tool results, whatever the loop stopped on — so a budget stop alone is
    an ordinary pass and INCONCLUSIVE stays a claim the check has to make."""
    for stop in sorted(BUDGET_STOPS):
        assert classify(status_code=200, stop_reason=stop, error=None) is Outcome.PASS


def test_vacuous_still_fails_if_nobody_classifies_it() -> None:
    # `VacuousRun` subclasses AssertionError deliberately: a caller that skips
    # chess's `classify` — canonical's included — gets a failed sample rather
    # than a silent pass.
    assert issubclass(VacuousRun, AssertionError)
    vacuous = VacuousRun("no commentary was produced")
    assert (
        evalstats.classify(status_code=200, stop_reason="completed", error=vacuous)
        is evalstats.Outcome.FAIL
    )


def test_infrastructure_still_beats_a_vacuous_check() -> None:
    # The precedence the old chess `classify` stated outright: a provider death
    # outranks whatever the check made of the turn it killed.
    vacuous = VacuousRun("no narrator ran")
    assert (
        classify(status_code=200, stop_reason=STOP_PROVIDER_ERROR, error=vacuous)
        is Outcome.INFRA
    )
    assert classify(status_code=502, stop_reason=None, error=vacuous) is Outcome.INFRA


def test_a_non_200_miss_stays_a_miss_whatever_the_check_said() -> None:
    # Unreachable from the harness (it runs no check on a non-200), and pinned
    # so a vacuity reading can never turn an app failure into an untested one.
    assert (
        classify(status_code=500, stop_reason=None, error=VacuousRun("nothing"))
        is Outcome.FAIL
    )


# --- classify on chess's own shapes -------------------------------------------
#
# Since audit item 20 chess's brain *catches* `ProviderError` and answers 200
# with `stop_reason="provider_error"` and the structured kind it died of
# (`AgentResponse.provider_failure`, a `ProviderFailure` value). The vendored
# spec only ever drives a death as a 502, which classifies on its status alone;
# this is the shape chess's samples actually arrive in.


@pytest.mark.parametrize(
    ("error", "failure", "expected"),
    [
        # A crash stopped looking like a 502 and started looking like a silent
        # behavioral miss ("expected a resign call: nothing"). Same dead socket.
        pytest.param(None, None, Outcome.INFRA, id="unnamed"),
        pytest.param(
            AssertionError("expected a resign call: nothing"),
            None,
            Outcome.INFRA,
            id="infra-beats-assertion",
        ),
        pytest.param(None, "unreachable", Outcome.INFRA, id="unreachable"),
        pytest.param(None, "server_error", Outcome.INFRA, id="server_error"),
        # A request the server refuses identically every time (an HTTP 400 on
        # an overrun context), and a 200 whose body fails wire validation — a
        # version skew, not a crash.
        pytest.param(None, "rejected", Outcome.PROVIDER_REJECTED, id="rejected"),
        pytest.param(
            None, "malformed_response", Outcome.PROVIDER_REJECTED, id="malformed"
        ),
        # A KeyError in a check is not evidence about anything the provider did,
        # and the infra budget must not be burned on a bug in this repo.
        pytest.param(TypeError("nope"), None, Outcome.HARNESS, id="harness"),
        pytest.param(
            TypeError("nope"), "rejected", Outcome.HARNESS, id="harness-rejected"
        ),
    ],
)
def test_a_chess_provider_death_wears_a_200(
    error: BaseException | None, failure: str | None, expected: Outcome
) -> None:
    outcome = classify(
        status_code=200,
        stop_reason=STOP_PROVIDER_ERROR,
        error=error,
        provider_failure=failure,
    )
    assert outcome is expected


@pytest.mark.parametrize("status", [409, 500, 503])
def test_a_non_200_that_is_not_infrastructure_is_a_miss(status: int) -> None:
    """Canonical's reading, adopted with the vendored module (#534). Chess's
    old `classify` retried *any* non-200; canonical retries only 429, 502 and
    504. Chess's seams answer a provider death as a 200 carrying
    `provider_error` (and the delegate route's escaped one as a 502), so any
    other status is the app failing the request — a miss, not a re-take. The
    eval seams cannot produce one today: the harness sends no version, game id
    or Idempotency-Key (no 409/422), always has a brain (no 503), and an
    unhandled exception raises through the TestClient rather than coming back
    as a 500."""
    assert classify(status_code=status, stop_reason=None, error=None) is Outcome.FAIL


@pytest.mark.parametrize("status", sorted(INFRA_STATUSES | {STATUS_RATE_LIMITED}))
def test_the_infra_statuses_are_still_retried(status: int) -> None:
    assert classify(status_code=status, stop_reason=None, error=None) is Outcome.INFRA


# --- the loop's stop vocabulary -----------------------------------------------
#
# Canonical's `INFRA_STOPS` names the fleet's two infrastructure stops,
# `provider_error` and `timed_out`. Chess's loop has no `timed_out`: its wall
# time is a per-turn budget, and it ends as `budget`, scored as behavior. The
# loop keeps no enum of its stops, so the vocabulary is read off its code: every
# string literal `llama_brain.py` passes as the `stop_reason` of something that
# takes one, plus the defaults of those parameters.


def _stop_takers() -> tuple[dict[str, int], set[str]]:
    """Every callable in the brain modules with a `stop_reason` parameter, by
    the name a call site spells it (a class by its own name), with the
    parameter's positional index; and the string defaults they declare."""
    takers: dict[str, int] = {}
    defaults: set[str] = set()
    for module in (brain, llama_brain):
        for value in vars(module).values():
            if getattr(value, "__module__", None) != module.__name__:
                continue
            if inspect.isclass(value):
                members = [
                    (value.__name__ if name == "__init__" else name, member)
                    for name, member in vars(value).items()
                    if inspect.isfunction(member)
                ]
                offset = 1
            elif inspect.isfunction(value):
                members, offset = [(value.__name__, value)], 0
            else:
                continue
            for name, member in members:
                parameters = inspect.signature(member).parameters
                if "stop_reason" not in parameters:
                    continue
                takers[name] = list(parameters).index("stop_reason") - offset
                default = parameters["stop_reason"].default
                if isinstance(default, str):
                    defaults.add(default)
    return takers, defaults


def _loop_stops() -> frozenset[str]:
    takers, stops = _stop_takers()
    tree = ast.parse(Path(llama_brain.__file__).read_text())
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else None
        if isinstance(func, ast.Name):
            name = func.id
        if name not in takers:
            continue
        passed = [kw.value for kw in node.keywords if kw.arg == "stop_reason"]
        if len(node.args) > takers[name]:
            passed.append(node.args[takers[name]])
        stops.update(
            value.value
            for value in passed
            if isinstance(value, ast.Constant) and isinstance(value.value, str)
        )
    return frozenset(stops)


_LOOP_STOPS = _loop_stops()


def test_the_stop_vocabulary_is_read_off_the_loop() -> None:
    # The reading found something to read: the default every closed turn
    # carries, and the death this module names.
    assert {"completed", STOP_PROVIDER_ERROR} <= _LOOP_STOPS


def test_the_only_infrastructure_stop_chess_has_is_provider_error() -> None:
    assert _LOOP_STOPS & INFRA_STOPS == {STOP_PROVIDER_ERROR}
    assert STOP_PROVIDER_ERROR in INFRA_STOPS


def test_the_deadline_is_a_budget_stop_not_timed_out() -> None:
    assert "timed_out" not in _LOOP_STOPS
    assert "budget" in _LOOP_STOPS - INFRA_STOPS


def test_the_budget_stops_are_the_loops_own() -> None:
    assert BUDGET_STOPS <= _LOOP_STOPS - INFRA_STOPS


@pytest.mark.parametrize("stop", sorted(_LOOP_STOPS - INFRA_STOPS))
def test_every_other_stop_is_scored_as_behavior(stop: str) -> None:
    assert classify(status_code=200, stop_reason=stop, error=None) is Outcome.PASS
    assert (
        classify(status_code=200, stop_reason=stop, error=AssertionError(stop))
        is Outcome.FAIL
    )


# --- the provider, over a fake server -----------------------------------------
#
# Chess's real `LlamaCppProvider`, driven over `httpx.MockTransport`. Canonical
# reads a provider death off its *text* (`provider_failure_kind`); chess's brain
# carries the *structured* kind (`ProviderFailure`) and the harness classifies
# on that. The two have to agree on the only question either answers — retry or
# not — and the text has to carry what canonical reads.
#
# Left out on purpose: 408, 409, 425 and 429. Canonical calls those four 4xx
# transient; chess's provider calls every 4xx REJECTED. llama-server does not
# send them, and chess's reading is the fail-fast direction (a scenario fails on
# the first one rather than spending retries), so the gate never spends its
# infra budget on them. Not changed here (#534): a provider change is its own
# slice.

_BAD_ARGUMENTS = {
    "choices": [
        {
            "message": {
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_0",
                        "function": {"name": "make_move", "arguments": "not json"},
                    }
                ],
            },
            "finish_reason": "tool_calls",
        }
    ]
}


def _refused(request: httpx.Request) -> httpx.Response:
    raise httpx.ConnectError("connection refused", request=request)


def _timed_out(request: httpx.Request) -> httpx.Response:
    raise httpx.ReadTimeout("timed out", request=request)


def _status(code: int) -> Callable[[httpx.Request], httpx.Response]:
    return lambda _request: httpx.Response(code, text="context overrun")


def _malformed(_request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json={"nope": 1})


def _bad_arguments(_request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json=_BAD_ARGUMENTS)


def _provider(answer: Callable[[httpx.Request], httpx.Response]) -> LlamaCppProvider:
    client = httpx.Client(transport=httpx.MockTransport(answer))
    return LlamaCppProvider("http://llm.test/v1", "gemma-4-12b", client=client)


def _death(answer: Callable[[httpx.Request], httpx.Response]) -> ProviderError:
    with pytest.raises(ProviderError) as caught:
        _provider(answer).chat([{"role": "user", "content": "hi"}])
    return caught.value


def _agrees_on_retrying(exc: ProviderError) -> None:
    kind = provider_failure_kind(str(exc))
    assert (kind in DETERMINISTIC_FAILURES) == (not exc.failure.transient), kind
    # And through the delegate route's 502 detail, which wraps the message.
    assert provider_failure_kind(f"agent run failed: {exc}") == kind


@pytest.mark.parametrize("code", [400, 500, 502, 503])
def test_an_http_failure_reads_the_same_off_the_text(code: int) -> None:
    exc = _death(_status(code))
    assert f"llama-server returned {code}" in str(exc)
    _agrees_on_retrying(exc)


@pytest.mark.parametrize("answer", [_refused, _timed_out], ids=["refused", "timeout"])
def test_a_dead_socket_reads_as_transport(
    answer: Callable[[httpx.Request], httpx.Response],
) -> None:
    exc = _death(answer)
    assert provider_failure_kind(str(exc)) == "transport"
    _agrees_on_retrying(exc)


def test_an_unparseable_body_reads_as_malformed() -> None:
    exc = _death(_malformed)
    assert "failed wire validation" in str(exc)
    assert provider_failure_kind(str(exc)) == "malformed_response"
    _agrees_on_retrying(exc)


# --- and through the brain ----------------------------------------------------
#
# What the harness actually classifies is not the exception but the brain's
# answer to it: the stop and the kind on `AgentResponse`. So the same fake
# server, behind the real loop.


class _NoTools:
    """A dispatcher for a turn that never gets as far as a tool."""

    def dispatch(self, name: str, args: Any) -> dict[str, Any]:
        raise AssertionError(f"{name} dispatched on a turn that should have died")

    def refusal(self, error: str, retry: str, **details: Any) -> dict[str, Any]:
        return {"ok": False, "error": error, "retry": retry, **details}


_MAKE_MOVE = {
    "type": "function",
    "function": {
        "name": "make_move",
        "description": "Submit a move in SAN or UCI.",
        "parameters": {
            "type": "object",
            "properties": {"move": {"type": "string"}},
            "required": ["move"],
        },
    },
}

_ANSWERS = {
    "400": _status(400),
    "500": _status(500),
    "503": _status(503),
    "refused": _refused,
    "malformed": _malformed,
    "bad_arguments": _bad_arguments,
}


def _turn(answer: Callable[[httpx.Request], httpx.Response]) -> brain.AgentResponse:
    loop = LlamaBrain(
        provider=_provider(answer),
        dispatcher=_NoTools(),
        tool_definitions=[_MAKE_MOVE],
        system_prompt="You are a chess opponent.",
        planner_prompt="Pick the tools.",
    )
    return loop.get_agent_response(board_state={}, command="play e4")


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ("400", Outcome.PROVIDER_REJECTED),
        ("500", Outcome.INFRA),
        ("503", Outcome.INFRA),
        ("refused", Outcome.INFRA),
        ("malformed", Outcome.PROVIDER_REJECTED),
        # Corrected, not a death: the turn ends on its correction budget and is
        # scored like any other turn.
        ("bad_arguments", Outcome.PASS),
    ],
)
def test_what_the_gate_makes_of_each_server_answer(
    answer: str, expected: Outcome
) -> None:
    response = _turn(_ANSWERS[answer])
    outcome = classify(
        status_code=200,
        stop_reason=response.stop_reason,
        error=None,
        provider_failure=response.provider_failure or None,
    )
    assert outcome is expected


def test_the_kinds_a_turn_can_die_of_split_as_canonical_splits_them() -> None:
    """Chess's old set refused to retry `bad_tool_arguments` too. Canonical's
    does not name it, and it needs no name: the brain answers it with a
    correction and never ends a turn on it (`provider.ProviderFailure`). So the
    non-transient kinds that can reach a `provider_error` stop are exactly
    canonical's `DETERMINISTIC_FAILURES`, the transient ones are outside it, and
    every kind but that one ends a turn on some server answer."""
    died = set()
    for answer in _ANSWERS.values():
        response = _turn(answer)
        if response.stop_reason == STOP_PROVIDER_ERROR:
            died.add(ProviderFailure(response.provider_failure))
        else:
            assert response.stop_reason in BUDGET_STOPS, response.stop_reason
    assert {kind for kind in died if not kind.transient} == DETERMINISTIC_FAILURES
    assert not {kind for kind in died if kind.transient} & DETERMINISTIC_FAILURES
    assert set(ProviderFailure) - died == {ProviderFailure.BAD_TOOL_ARGUMENTS}


# --- failure_signature --------------------------------------------------------


def test_failures_differing_only_in_moves_share_a_signature() -> None:
    """Five failures with five different SANs are one failure mode, and the
    harness must be able to say so — "2/5, both the same mode" and "2/5, two
    unrelated modes" are different findings printed identically otherwise."""
    first = failure_signature("expected Bxe6, board got 'Bxh6'")
    second = failure_signature("expected Bxe4, board got 'Nf3'")
    assert first == second == "expected <san>, board got '<san>'"
    # Canonical's own collapses the digits and keeps the pieces apart.
    assert evalstats.failure_signature(
        "expected Bxe6, board got 'Bxh6'"
    ) != evalstats.failure_signature("expected Bxe4, board got 'Nf3'")


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


@pytest.mark.parametrize(
    "message",
    [
        "run 7 [FAIL]: expected Bxe6, board got None",
        "castled O-O-O on move 12, wanted O-O",
        "  e8=Q+  promoted   with 2.5 s left\n",
        "expected a resign call: nothing",
    ],
)
def test_canonicals_signature_leaves_a_chess_one_unchanged(message: str) -> None:
    # What `RateResult.from_blocks` relies on to derive its modes from chess's
    # signature through canonical's code.
    signed = failure_signature(message)
    assert evalstats.failure_signature(signed) == signed


# --- RateResult and scenario_record -------------------------------------------


def test_failure_modes_are_read_by_the_chess_signature() -> None:
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


def test_the_suspect_flag_follows_the_chess_modes_not_canonicals() -> None:
    """The two derived fields that read the signature are derived from chess's,
    together: two failures naming two different moves are one mode and, at 0
    passes, a deterministic suspect — where canonical's number-only signature
    reads two modes and no suspect. The messages themselves stay on the
    record."""
    failures = ["expected Bxe6, board got 'Bxh6'", "expected Bxe4, board got 'Nf3'"]
    chess = RateResult.from_blocks(blocks=[(0, 2)], failures=failures, floor=0.8)
    canonical = evalstats.RateResult.from_blocks(
        blocks=[(0, 2)], failures=failures, floor=0.8
    )

    assert chess.failure_modes == {"expected <san>, board got '<san>'": 2}
    assert chess.deterministic_suspect is True
    assert len(canonical.failure_modes) == 2
    assert canonical.deterministic_suspect is False
    assert chess.failures == tuple(failures)


@pytest.mark.parametrize(
    ("blocks", "failures"),
    [
        ([(5, 5), (3, 5)], ["run 7: expected Bxe6, board got None"] * 2),
        ([(0, 5)], ["run 1: expected Nf3", "run 2: no make_move", "run 3: x"]),
        ([(0, 5)], [f"run {i}: expected Bxe{i}, board got None" for i in range(5)]),
        ([], []),
    ],
)
def test_the_rest_of_the_verdict_is_canonicals(
    blocks: list[tuple[int, int]], failures: list[str]
) -> None:
    chess = RateResult.from_blocks(
        blocks=blocks, failures=failures, floor=0.8, infra=2, inconclusive=1
    )
    canonical = evalstats.RateResult.from_blocks(
        blocks=blocks, failures=failures, floor=0.8, infra=2
    )
    assert isinstance(chess, evalstats.RateResult)
    for name in ("passed", "runs", "blocks", "failures", "infra", "interval"):
        assert getattr(chess, name) == getattr(canonical, name), name
    assert chess.decision is canonical.decision
    assert chess.stability is canonical.stability
    assert chess.rate == canonical.rate
    # And the two signature-derived fields agree with each other and the counts.
    assert chess.failure_modes == Counter(failure_signature(f) for f in failures)
    assert chess.deterministic_suspect == (
        chess.passed == 0 and len(chess.failure_modes) == 1
    )
    assert chess.inconclusive == 1


def test_the_summary_line_carries_the_inconclusive_count() -> None:
    """The `[eval]` line a human reads, exactly as chess has always printed it:
    the inconclusive count straight after the infra one, and the suspect flag
    last."""
    result = RateResult.from_blocks(
        blocks=[(4, 5)],
        failures=["expected Bxe6, board got None"],
        floor=0.8,
        infra=1,
        inconclusive=1,
    )
    assert result.summary() == (
        "4/5 (80%) ci=[0.44,0.95] ABOVE_FLOOR STABLE blocks=4/5 infra=1 inconclusive=1"
    )
    suspect = RateResult.from_blocks(
        blocks=[(0, 5)], failures=["expected Bxe6, board got None"] * 5, floor=0.8
    )
    assert suspect.summary().endswith(
        "blocks=0/5 infra=0 inconclusive=0 DETERMINISTIC_SUSPECT"
    )


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
    assert decoded["failures"] == ["run 8: expected Bxe6, board got None"]
    assert decoded["blocking"] is True  # arbitrary metadata rides along
    # Everything canonical's record says, chess's says the same way.
    assert decoded == {
        **json.loads(json.dumps(evalstats.scenario_record("x", result, 0.8))),
        "scenario": "long_capture[poisoned]",
        "inconclusive": 1,
        "blocking": True,
    }
    assert result.decision is Decision.UNDECIDED
    assert result.stability is Stability.UNSTABLE
    assert result.interval == wilson_interval(7, 10)


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


def test_more_narrator_calls_than_calls_is_unknown_not_a_guess() -> None:
    latencies = split_latencies((3000,), route="brain", stop_reason="completed")
    assert latencies.attribution is Attribution.UNKNOWN


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


def test_one_confirmation_call_is_still_the_whole_turn() -> None:
    """A confirmation turn that spent exactly one round trip is attributed the
    way every other narrate route is — because at one call it does not matter
    which phase made it. That is the common shape: a cancel, or a confirm at
    verbosity `low`, is the reader alone; a literal "yes" at normal verbosity is
    the narrator alone. Either way the turn's whole model time is that call's
    and the planner's zero is real."""
    latencies = split_latencies(
        (900,), route=ROUTE_CONFIRMATION, stop_reason="completed"
    )
    assert latencies.attribution is Attribution.SPLIT
    assert latencies.planner_ms == 0
    assert latencies.narrator_ms == 900


def test_a_two_call_confirmation_cannot_say_which_call_narrated() -> None:
    """The honest correction (audit 2026-09-05, "Statistical interpretation").
    Since the free-text reader landed, a confirmation turn can be reader-only,
    reader-plus-narrator, or narrator-only, and the trace record carries nothing
    that tells them apart. This branch used to call all of it narrator time, so
    every narrator median including a confirmation counted the reader's round
    trip as narration.

    UNKNOWN rather than a reader phase: naming a third phase would need
    production to record which calls were the reader's, and inventing it here
    would be the same guess wearing a label."""
    latencies = split_latencies(
        (300, 2400), route=ROUTE_CONFIRMATION, stop_reason="completed"
    )
    assert latencies.attribution is Attribution.UNKNOWN
    assert latencies.planner_ms is None
    assert latencies.narrator_ms is None
    assert latencies.total_ms == 2700  # the total is still a fact

    tokens = split_tokens(
        ((180, 1), (2600, 90)), route=ROUTE_CONFIRMATION, stop_reason="completed"
    )
    assert tokens.attribution is Attribution.UNKNOWN
    assert tokens.planner_out is None
    assert tokens.narrator_in is None and tokens.narrator_out is None
    assert tokens.completion_tokens == 91


def test_the_other_narrate_routes_keep_their_split_at_any_call_count() -> None:
    """The confirmation carve-out is exactly one route wide. Nothing else in
    `NARRATE_ROUTES` has a second kind of model call in front of the narrator,
    so a multi-call turn on one of them is still all narrator time."""
    for route in sorted(NARRATE_ROUTES - {ROUTE_CONFIRMATION}):
        latencies = split_latencies((300, 2400), route=route, stop_reason="completed")
        assert latencies.attribution is Attribution.SPLIT, route
        assert latencies.planner_ms == 0, route
        assert latencies.narrator_ms == 2700, route


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
    """`evalstats_app` names the routes itself so it stays importable with
    nothing installed, and this test is what keeps that copy honest. Every route
    the tracer can record is classified — a route in neither set would
    attribute to UNKNOWN forever without anyone noticing."""
    recorded = {
        value
        for name, value in vars(trace).items()
        if name.startswith("ROUTE_") and isinstance(value, str)
    }
    assert recorded == NARRATE_ROUTES | {ROUTE_BRAIN}
    assert not NARRATE_ROUTES & {ROUTE_BRAIN}


def test_the_phase_tags_are_the_brains() -> None:
    """Same payment for the phase tags a schema-2 record carries: every phase
    the brain can tag a call with is one the attribution knows, the planner's
    and the narrator's are the ones it splits on, and the tag for a call nobody
    tagged sends the record back to the positional rule."""
    tagged = {
        value
        for name, value in vars(brain).items()
        if name.startswith("PHASE_") and isinstance(value, str)
    }
    assert tagged - {PHASE_UNKNOWN} <= _KNOWN_PHASES
    assert _PLANNER_PHASE == PHASE_PLANNER
    assert PHASE_NARRATOR in _NARRATOR_PHASES
    assert PHASE_UNKNOWN not in _KNOWN_PHASES


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


# --- split_tokens / generation_rate -------------------------------------------
#
# The half `split_latencies` cannot answer. It settled *where* the repeat-stop
# turn's extra 30 s is spent — the narrator's own round trip, p = 0.00088 — and
# left the mechanism open, because "the narrator emitted more tokens" and
# "generation was slower" are the same number of milliseconds. Usage per call
# separates them: tokens say how much was written, the pair says how fast. The
# phase boundary is the *same* rule the latencies use (one derivation, applied
# to whichever readings are being attributed), so the two clauses on a line can
# never disagree about which call was the narrator.


def test_a_brain_turn_attributes_its_tokens_the_way_it_attributes_its_time() -> None:
    tokens = split_tokens(
        ((2100, 8), (2400, 12), (2900, 940)), route="brain", stop_reason="completed"
    )
    assert tokens.attribution is Attribution.SPLIT
    assert tokens.planner_out == 20
    assert tokens.narrator_out == 940
    # The narrator's *prompt* is a candidate mechanism in its own right: a
    # repeat-stop turn dispatched the duplicate, so its narrator reads one more
    # tool result than a completed turn's does.
    assert tokens.narrator_in == 2900
    assert tokens.completion_tokens == 960


def test_a_budget_stop_spent_every_token_in_the_planner() -> None:
    for stop in sorted(BUDGET_STOPS):
        tokens = split_tokens(
            ((2000, 30), (2200, 25), (2400, 28), (2600, 31)),
            route="brain",
            stop_reason=stop,
        )
        assert tokens.attribution is Attribution.NO_NARRATOR, stop
        assert tokens.planner_out == 114, stop
        assert tokens.narrator_out is None, stop
        assert tokens.narrator_in is None, stop


def test_a_narrate_route_spent_every_token_in_the_narrator() -> None:
    for route in sorted(NARRATE_ROUTES):
        tokens = split_tokens(((1800, 240),), route=route, stop_reason="completed")
        assert tokens.attribution is Attribution.SPLIT, route
        assert tokens.planner_out == 0, route
        assert tokens.narrator_out == 240, route


def test_a_provider_death_leaves_the_token_boundary_unknown_too() -> None:
    tokens = split_tokens(
        ((2100, 8), (2400, None)), route="brain", stop_reason="provider_error"
    )
    assert tokens.attribution is Attribution.UNKNOWN
    assert tokens.planner_out is None
    assert tokens.narrator_out is None


def test_a_zero_llm_turn_reports_no_token_readings() -> None:
    tokens = split_tokens((), route="fast_path", stop_reason="completed")
    assert tokens.attribution is Attribution.NONE
    assert tokens.narrator_out is None
    assert tokens.completion_tokens is None


def test_one_unmeasured_call_makes_its_phase_unknown_not_smaller() -> None:
    """A round trip that raised has no usage — the result never came back. Its
    phase total is therefore unknown, and summing the rest would report a
    *smaller* number as if it were the whole: exactly the silent lie the
    instrument exists to avoid."""
    tokens = split_tokens(
        ((2100, 8), (2400, None), (2900, 940)), route="brain", stop_reason="completed"
    )
    assert tokens.planner_out is None  # the raising call was the planner's
    assert tokens.narrator_out == 940  # but the narrator's own reading survives
    assert tokens.completion_tokens is None


def test_the_token_summary_names_the_unknowns_the_same_way() -> None:
    tokens = split_tokens(
        ((2100, 8), (2400, 12), (2900, 940)), route="brain", stop_reason="completed"
    )
    assert tokens.summary() == (
        "call_in=[2100,2400,2900] call_out=[8,12,940] "
        "planner_out=20 narrator_in=2900 narrator_out=940"
    )
    stuck = split_tokens(
        ((2000, 30), (2200, None)), route="brain", stop_reason="max_iterations"
    )
    assert stuck.summary() == (
        "call_in=[2000,2200] call_out=[30,?] planner_out=? narrator_in=? narrator_out=?"
    )
    silent = split_tokens((), route="fast_path", stop_reason="completed")
    assert silent.summary() == (
        "call_in=[] call_out=[] planner_out=? narrator_in=? narrator_out=?"
    )


def test_the_token_record_is_json_and_carries_the_totals() -> None:
    tokens = split_tokens(
        ((2100, 8), (2400, 12), (2900, 940)), route="brain", stop_reason="completed"
    )
    decoded = json.loads(json.dumps(tokens.as_record()))
    assert decoded == {
        "prompt_tokens": 7400,
        "completion_tokens": 960,
        "call_in": [2100, 2400, 2900],
        "call_out": [8, 12, 940],
        "planner_out": 20,
        "narrator_in": 2900,
        "narrator_out": 940,
    }


def test_generation_rate_is_tokens_per_second_of_the_call_that_wrote_them() -> None:
    """The discriminator itself. A narrator that emitted 3× the tokens at the
    same rate is a *thinking* problem (bound the budget); one that emitted the
    same tokens at a third of the rate is a *serving* problem (and a token cap
    would not touch it). One number tells them apart."""
    assert generation_rate(940, 39100) == pytest.approx(24.04, abs=0.01)


def test_generation_rate_refuses_to_divide_by_an_unknown() -> None:
    assert generation_rate(None, 39100) is None
    assert generation_rate(940, None) is None
    # A 0 ms reading is a clock that did not run, not an infinitely fast model.
    assert generation_rate(940, 0) is None


# --- a schema-2 record's phase tags decide the split (#317) ------------------


def test_tagged_calls_split_a_provider_death_the_positions_could_not() -> None:
    """The positional rule has to call a brain turn that died UNKNOWN: the dead
    call could have been either phase. The tags say which it was."""
    untagged = split_latencies((400, 900), route="brain", stop_reason="provider_error")
    assert untagged.attribution == Attribution.UNKNOWN

    tagged = split_latencies(
        (400, 900),
        route="brain",
        stop_reason="provider_error",
        phases=("planner", "closer"),
    )
    assert tagged.attribution == Attribution.SPLIT
    assert (tagged.planner_ms, tagged.narrator_ms) == (400, 900)


def test_tagged_calls_place_the_reader_in_neither_phase() -> None:
    """A confirmation that read the answer and then narrated: positions cannot
    tell the two apart, the tags can — and the reader is neither phase."""
    latencies = split_latencies(
        (150, 800),
        route="confirmation",
        stop_reason="completed",
        phases=("answer", "reaction"),
    )
    assert latencies.attribution == Attribution.SPLIT
    assert (latencies.planner_ms, latencies.narrator_ms) == (0, 800)
    assert latencies.total_ms == 950


def test_a_rewrite_is_narrator_time_by_its_tag() -> None:
    latencies = split_latencies(
        (300, 700, 500),
        route="brain",
        stop_reason="completed",
        phases=("planner", "closer", "rewrite"),
    )
    assert (latencies.planner_ms, latencies.narrator_ms) == (300, 1200)


def test_no_narrator_tag_is_no_narrator() -> None:
    latencies = split_latencies(
        (300, 700), route="brain", stop_reason="budget", phases=("planner", "planner")
    )
    assert latencies.attribution == Attribution.NO_NARRATOR
    assert latencies.narrator_ms is None


def test_an_untagged_or_mismatched_record_falls_back_to_the_positions() -> None:
    fallback = split_latencies((300, 700), route="brain", stop_reason="completed")
    for phases in (("unknown", "unknown"), ("planner",), None):
        assert (
            split_latencies(
                (300, 700), route="brain", stop_reason="completed", phases=phases
            )
            == fallback
        )


def test_tokens_split_by_the_same_tags() -> None:
    tokens = split_tokens(
        [(900, 12), (None, None), (1200, 80)],
        route="confirmation",
        stop_reason="completed",
        phases=("answer", "planner", "closer"),
    )
    assert tokens.attribution == Attribution.SPLIT
    assert tokens.planner_out is None  # the planner's one call reported nothing
    assert (tokens.narrator_in, tokens.narrator_out) == (1200, 80)


# --- one import name ----------------------------------------------------------
#
# `evalstats` and `tests.evalstats` are two module objects with two `Outcome`
# classes, and `evalstats_app` defines its own `Outcome`, `classify`,
# `RateResult`, … over canonical's. A chess file that took one of those from
# canonical would compare `outcome is Outcome.PASS` across two classes and
# never match. So: chess code imports canonical only as `evalstats`, and every
# name `evalstats_app` overrides only from `evalstats_app`.

_BACKEND = Path(__file__).resolve().parent.parent
# The vendored pair reads canonical by its fleet name; these two compare the
# halves on purpose.
_VENDORED = {"evalstats.py", "test_evalstats.py"}
_COMPARERS = {"evalstats_app.py", "test_evalstats_app.py"}


def _overridden() -> set[str]:
    return {
        name
        for name, value in vars(evalstats_app).items()
        if not name.startswith("__")
        and hasattr(evalstats, name)
        and getattr(evalstats, name) is not value
    }


def test_chess_code_takes_each_name_from_the_module_that_owns_it() -> None:
    overridden = _overridden()
    assert {"Outcome", "classify", "RateResult", "failure_signature"} <= overridden
    offenders: list[str] = []
    files = [*(_BACKEND / "tests").glob("*.py"), *(_BACKEND / "scripts").glob("*.py")]
    for path in sorted(files):
        if path.name in _VENDORED:
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            where = f"{path.name}:{getattr(node, 'lineno', '?')}"
            if isinstance(node, ast.ImportFrom) and node.module == "tests.evalstats":
                offenders.append(f"{where} imports tests.evalstats")
            if isinstance(node, ast.Import) and any(
                alias.name == "tests.evalstats" for alias in node.names
            ):
                offenders.append(f"{where} imports tests.evalstats")
            if isinstance(node, ast.ImportFrom) and node.module == "evalstats":
                offenders += [
                    f"{where} takes {alias.name} from canonical"
                    for alias in node.names
                    if alias.name in overridden
                ]
            if (
                path.name not in _COMPARERS
                and isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == "evalstats"
                and node.attr in overridden
            ):
                offenders.append(f"{where} reads evalstats.{node.attr}")
    assert not offenders, offenders
