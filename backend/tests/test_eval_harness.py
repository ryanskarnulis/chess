"""The eval harness's own wiring, tested without a GPU.

`test_agent_evals.py` needs a live 12B for every assertion it makes about the
*model*, but the code that turns a finished turn into the `[eval]` line and the
report's per-sample record needs no model at all — it reads a trace record and a
call meter. That code was untested, and the harness-bug precedent is the reason
that matters: a wrong tool offer in this file once moved a scenario from 5/5 to
0–3/5 with nothing in the output saying the harness had changed
(`docs/agent-evals.md`). A measurement instrument that can silently lie about
the thing it measures is worse than no instrument.

So this file pins the seam `_measured` owns: the latency attribution reaches the
printed line, the `EvalRun`, and the sample record — with the readings and the
attribution coming off the *same* trace record, so they can never describe
different turns. Nothing here calls the model, so it runs in ordinary `pytest`
alongside `test_evalstats.py`.
"""

from __future__ import annotations

from typing import Any

from evalstats import Attribution
from fakes import CountingProvider, ModelCall
from test_agent_evals import EvalApp, _CollectingTracer, _measured


class _Response:
    """The two things `_measured` reads off an HTTP answer."""

    def __init__(self, status_code: int = 200, text: str = "") -> None:
        self.status_code = status_code
        self.text = text


def _app(record: dict[str, Any], calls: int) -> EvalApp:
    """An `EvalApp` with only the two observation seams `_measured` touches
    populated. `client` and `ctx` are None on purpose: reaching for either from
    the reporting path would be a bug this test should catch, not accommodate."""
    tracer = _CollectingTracer()
    tracer.record(record)
    provider = CountingProvider(inner=None)
    provider.calls.extend(ModelCall(thinking=False, seconds=1.0) for _ in range(calls))
    return EvalApp(client=None, ctx=None, provider=provider, tracer=tracer)  # type: ignore[arg-type]


_ASSISTANT: dict[str, Any] = {"content": "Bishop takes. Rude.", "tool_calls": []}


def _traced(**overrides: Any) -> dict[str, Any]:
    """A trace record shaped like `trace.turn_record`'s, cut down to the keys the
    reporting path reads."""
    record: dict[str, Any] = {
        "route": "brain",
        "stop_reason": "completed",
        "model_calls": 3,
        "model_ms": 41200,
        "model_latencies_ms": [1200, 900, 39100],
        "mutations": 2,
        "provider_failure": "",
    }
    return {**record, **overrides}


def test_the_eval_line_carries_the_per_call_split(capsys: Any) -> None:
    """The turn total was already printed; what a reader could not see was which
    call spent it. Both are on the line now, and the total is printed beside the
    parts precisely so a disagreement between them is visible."""
    run = _measured(_app(_traced(), 3), "long_capture", _ASSISTANT, _Response(), 41.5)
    line = capsys.readouterr().out

    assert "model_ms=41200" in line
    assert "call_ms=[1200,900,39100]" in line
    assert "planner_ms=2100" in line
    assert "narrator_ms=39100" in line
    assert "phases=SPLIT" in line
    assert run.latencies.narrator_ms == 39100


def test_a_repeat_stop_reaches_the_line_as_a_knowable_split(capsys: Any) -> None:
    """The measurement the slice exists for: on a `no_progress` turn the narrator
    still ran, so its round trip is separable from the planner's — which is what
    turns TODO.md's "narrates 2–3× slower" into a number instead of a suspicion."""
    app = _app(
        _traced(
            stop_reason="no_progress",
            model_calls=2,
            model_ms=41000,
            model_latencies_ms=[1000, 40000],
        ),
        2,
    )
    run = _measured(app, "hints_off_no_advice", _ASSISTANT, _Response(), 41.2)

    assert "planner_ms=1000 narrator_ms=40000 phases=SPLIT" in capsys.readouterr().out
    assert run.latencies.attribution is Attribution.SPLIT


def test_a_budget_stop_claims_no_narrator_time(capsys: Any) -> None:
    # No narrator ran, so the last call is the planner's and there is no narrator
    # reading — printed as `?`, never as a number.
    app = _app(
        _traced(
            stop_reason="max_iterations",
            model_calls=4,
            model_ms=4000,
            model_latencies_ms=[1000, 1000, 1000, 1000],
        ),
        4,
    )
    run = _measured(app, "hints_off_no_advice", _ASSISTANT, _Response(), 4.4)

    assert "planner_ms=4000 narrator_ms=? phases=NO_NARRATOR" in capsys.readouterr().out
    assert run.latencies.narrator_ms is None


def test_a_turn_that_was_never_traced_reports_no_readings(capsys: Any) -> None:
    """A non-200 leaves `_CollectingTracer.last` empty. The old line printed
    `model_ms=None` there and this one must not improve on that with a
    fabricated zero split."""
    tracer = _CollectingTracer()
    provider = CountingProvider(inner=None)
    app = EvalApp(client=None, ctx=None, provider=provider, tracer=tracer)  # type: ignore[arg-type]
    run = _measured(
        app,
        "resume_not_denied",
        {"content": "", "tool_calls": []},
        _Response(502, "upstream died"),
        12.0,
    )
    out = capsys.readouterr().out

    assert "model_ms=None" in out
    assert "call_ms=[] planner_ms=? narrator_ms=? phases=NONE" in out
    assert "! HTTP 502" in out
    assert run.latencies.attribution is Attribution.NONE


def test_the_sample_record_pairs_the_split_with_the_outcome() -> None:
    """The per-sample half. `docs/agent-evals.md` has claimed per-sample
    `model_ms` since the statistics slice while the report carried none; the
    pairing is the point, because the question is whether the slow samples and
    the `no_progress` samples are the same samples."""
    run = _measured(_app(_traced(), 3), "long_capture", _ASSISTANT, _Response(), 41.5)
    record = run.latencies.as_record()

    assert record == {
        "model_ms": 41200,
        "call_ms": [1200, 900, 39100],
        "planner_ms": 2100,
        "narrator_ms": 39100,
        "phases": "SPLIT",
    }
