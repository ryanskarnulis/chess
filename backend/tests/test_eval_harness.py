"""The eval harness's own wiring, tested without a GPU.

`test_agent_evals.py` needs a live 12B for every assertion it makes about the
*model*, but the code that turns a finished turn into the `[eval]` line and the
report's per-sample record needs no model at all — it reads a trace record and a
call meter. That code was untested, and the harness-bug precedent is the reason
that matters: a wrong tool offer in this file once moved a scenario from 5/5 to
0–3/5 with nothing in the output saying the harness had changed
(`docs/agent-evals.md`). A measurement instrument that can silently lie about
the thing it measures is worse than no instrument.

So this file pins the seam `_measured` owns: the latency attribution and the
per-call token counts reach the printed line, the `EvalRun`, and the sample
record — with each half's readings and its attribution coming off the *same*
seam, so they can never describe different turns. Nothing here calls the model,
so it runs in ordinary `pytest` alongside `test_evalstats.py`.

It also pins the harness's **gates**, which is the same argument one step
further in. The 2026-09-05 audit (`docs/agent-audit-2026-09-05.md`, findings 9
and 10) found three ways a sample could pass without testing anything: a
resignation utterance the fast path settled before the model was built, a
`completed` turn whose text was the pipeline's own canned stuck line, and a
move read off the tool call that claimed it rather than off the board that
kept it. None of those needs a GPU to reproduce — they are conditions on a
finished turn — so each one is a test here, and the scenarios' premises
(which utterance each parser does and does not swallow, which tool names the
generic helpers stand for) are pinned in CI rather than only inside an opt-in
suite nobody runs on a laptop.
"""

from __future__ import annotations

from typing import Any

import pytest

from chessapp.api import STUCK_REPLY
from chessapp.fastparse import parse_confirmation, parse_resign
from chessapp.game import GameSession
from chessapp.tools import DESTRUCTIVE_TOOLS, ToolContext, build_registry
from chessapp.trace import ROUTE_BRAIN, ROUTE_FAST_PATH, ROUTE_RESIGN
from evalstats import (
    STOP_PROVIDER_ERROR,
    Attribution,
    Decision,
    Outcome,
    VacuousRun,
    classify,
    split_latencies,
    split_tokens,
)
from fakes import CountingProvider, FakeEngine, ModelCall
from test_agent_evals import (
    _BLOCK_RUNS,
    _BOARD_TOOLS,
    _FREEFORM_ANSWERS,
    _LIVE_FEN,
    _RESIGN_LITERAL,
    _RESIGN_UTTERANCE,
    _STT_UTTERANCES,
    _VERDICT_TOOLS,
    EvalApp,
    EvalRun,
    _assert_reached_narrator,
    _assert_route,
    _CollectingTracer,
    _expect_san,
    _measured,
    _pass_rate,
    _replay_late_game,
    _run,
    _run_panel,
    _run_steps,
    _seam_name,
    _stays_a_model_eval,
    _step,
    _trajectory,
)


class _Response:
    """The two things `_measured` reads off an HTTP answer."""

    def __init__(self, status_code: int = 200, text: str = "") -> None:
        self.status_code = status_code
        self.text = text


def _app(
    record: dict[str, Any],
    calls: int,
    usage: list[tuple[int | None, int | None]] | None = None,
) -> EvalApp:
    """An `EvalApp` with only the two observation seams `_measured` touches
    populated. `client` and `ctx` are None on purpose: reaching for either from
    the reporting path would be a bug this test should catch, not accommodate.

    `usage` scripts the per-call token counts the meter recorded; omitted, every
    call is unmeasured, which is what an older provider or a died turn looks
    like."""
    tracer = _CollectingTracer()
    tracer.record(record)
    provider = CountingProvider(inner=None)
    readings = usage or [(None, None)] * calls
    provider.calls.extend(
        ModelCall(
            thinking=False,
            seconds=1.0,
            prompt_tokens=prompt,
            completion_tokens=completion,
        )
        for prompt, completion in readings
    )
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


# --- per-call token counts ----------------------------------------------------
#
# The measurement Sprint 5 asked for next. `call_ms` settled *where* a
# repeat-stop turn's extra 30 s goes (the narrator's own round trip); it cannot
# settle *why*, because "emitted three times the tokens" and "generated at a
# third of the rate" are the same milliseconds. The token readings come off the
# call meter rather than the trace — the trace sums the turn — so they are
# printed beside `call_ms`, and `model_calls` beside both is what makes a
# disagreement between the two seams visible rather than silent.


def test_the_eval_line_carries_the_per_call_tokens(capsys: Any) -> None:
    run = _measured(
        _app(_traced(), 3, [(2100, 8), (2400, 12), (2900, 940)]),
        "hints_off_no_advice",
        _ASSISTANT,
        _Response(),
        41.5,
    )
    line = capsys.readouterr().out

    assert "call_in=[2100,2400,2900]" in line
    assert "call_out=[8,12,940]" in line
    assert "planner_out=20" in line
    assert "narrator_in=2900 narrator_out=940" in line
    assert run.tokens.narrator_out == 940


def test_the_line_pairs_the_narrator_s_tokens_with_its_own_clock(
    capsys: Any,
) -> None:
    """The discriminator, computed where both halves are in scope: the narrator
    wrote 940 tokens in the 39.1 s the trace attributes to it, so ~24 tok/s. A
    slow narration at the *same* rate is a thinking-budget problem; the same
    tokens at a third of the rate is a serving problem, and only this number
    tells the two apart."""
    run = _measured(
        _app(_traced(), 3, [(2100, 8), (2400, 12), (2900, 940)]),
        "hints_off_no_advice",
        _ASSISTANT,
        _Response(),
        41.5,
    )

    assert "narrator_tok_s=24.0" in capsys.readouterr().out
    assert run.narrator_tok_s == pytest.approx(24.04, abs=0.01)


def test_an_unmeasured_phase_pairs_into_no_rate_at_all(capsys: Any) -> None:
    """A budget stop reached no narrator: there is no narrator clock and no
    narrator tokens, and a rate over two unknowns would be a fabricated
    number."""
    run = _measured(
        _app(
            _traced(
                stop_reason="max_iterations",
                model_calls=2,
                model_ms=2000,
                model_latencies_ms=[1000, 1000],
            ),
            2,
            [(2100, 8), (2400, 12)],
        ),
        "hints_off_no_advice",
        _ASSISTANT,
        _Response(),
        2.4,
    )

    assert "narrator_out=? " in capsys.readouterr().out
    assert run.tokens.narrator_out is None
    assert run.narrator_tok_s is None


def test_a_provider_that_reported_no_usage_prints_unknowns_not_zeros(
    capsys: Any,
) -> None:
    # The pre-existing shape: a meter with no usage recorded must degrade to `?`
    # rather than claim a turn that cost nothing.
    _measured(_app(_traced(), 3), "long_capture", _ASSISTANT, _Response(), 41.5)
    line = capsys.readouterr().out

    assert "call_in=[?,?,?] call_out=[?,?,?]" in line
    assert "planner_out=? narrator_in=? narrator_out=?" in line
    assert "narrator_tok_s=?" in line


def test_the_sample_record_pairs_the_tokens_with_the_milliseconds() -> None:
    """Both halves in one sample record, which is what makes the campaign
    question answerable off the report instead of off scrollback: are the slow
    narrations the *wordy* ones? That needs `narrator_ms` and `narrator_out` on
    the same line, per sample."""
    run = _measured(
        _app(_traced(), 3, [(2100, 8), (2400, 12), (2900, 940)]),
        "hints_off_no_advice",
        _ASSISTANT,
        _Response(),
        41.5,
    )
    record = {**run.latencies.as_record(), **run.tokens.as_record()}

    assert record["narrator_ms"] == 39100
    assert record["narrator_out"] == 940
    assert record["completion_tokens"] == 960
    assert record["call_out"] == [8, 12, 940]


# --- what each call asked for -------------------------------------------------
#
# The trajectory named the tools and not their arguments, which is a blind spot
# with a live suspect behind it: across four `hints_off_no_advice` runs a
# `set_hints_mode` call turns up on 9/65 turns where the player only asked "what
# should I play here?" (TODO.md). Whether it turned hints **on** — a setting the
# player owns, changed by an agent that was asked a question — is unknowable from
# `trajectory=[get_best_moves → set_hints_mode]`. The arguments are already on
# the wire (`agent_api._tool_call_read` puts them there); only the reporting
# path was dropping them, so this is a harness change with no `src/` in it.


def _call(
    tool: str,
    arguments: dict[str, Any] | None = None,
    *,
    result: str | None = "{}",
    error: str | None = None,
) -> dict[str, Any]:
    """One dispatched call, shaped like the `/api/agent` wire's."""
    return {
        "tool": tool,
        "arguments": arguments or {},
        "result": result,
        "error": error,
    }


def _assistant(*calls: dict[str, Any]) -> dict[str, Any]:
    return {"content": "Bishop takes. Rude.", "tool_calls": list(calls)}


def test_the_trajectory_names_what_each_call_asked_for(capsys: Any) -> None:
    """The measurement the slice exists for, in one line: the suspect call now
    says which way it flipped the setting."""
    _measured(
        _app(_traced(), 3),
        "hints_off_no_advice",
        _assistant(
            _call("get_best_moves", {"count": 3}),
            _call("set_hints_mode", {"enabled": True}),
        ),
        _Response(),
        41.5,
    )

    assert (
        "trajectory=[get_best_moves(count=3) → set_hints_mode(enabled=true)]"
        in capsys.readouterr().out
    )


def test_a_call_that_asked_for_nothing_stays_a_bare_name(capsys: Any) -> None:
    # Most of the suite's tokens are these, and `get_board_state()` would be
    # noise on every line to say nothing. Empty parens are not information.
    _measured(
        _app(_traced(), 3),
        "long_capture",
        _assistant(_call("get_board_state"), _call("undo", {"plies": 2})),
        _Response(),
        41.5,
    )

    assert "trajectory=[get_board_state → undo(plies=2)]" in capsys.readouterr().out


def test_a_string_and_a_boolean_do_not_read_alike() -> None:
    """Values are rendered as JSON, so `"true"` and `true` stay distinguishable.
    A model passing the string where the schema wants the bool is exactly the
    class of mis-invocation this line should not be able to hide — the quotes
    cost two characters and buy the distinction."""
    assert _trajectory(_assistant(_call("set_hints_mode", {"enabled": "true"}))) == (
        'set_hints_mode(enabled="true")'
    )
    assert _trajectory(_assistant(_call("set_hints_mode", {"enabled": False}))) == (
        "set_hints_mode(enabled=false)"
    )


def test_arguments_print_in_a_stable_order() -> None:
    """Sorted by name rather than in the order the model emitted them: the line
    is read by scanning many samples for a difference, and an ordering that
    varies per sample would show up as a difference that isn't one."""
    forward = _trajectory(_assistant(_call("save_game", {"name": "x", "slot": 2})))
    reversed_ = _trajectory(_assistant(_call("save_game", {"slot": 2, "name": "x"})))

    assert forward == reversed_ == 'save_game(name="x", slot=2)'


def test_a_long_value_is_cut_rather_than_swallowing_the_line() -> None:
    # One trajectory sits on one terminal line beside eleven other clauses; a
    # pasted PGN in an argument would cost the whole record's readability. The
    # cut is visible, so a truncated value never reads as a complete one.
    long = _trajectory(_assistant(_call("resume_game", {"name": "a" * 60})))

    assert long == 'resume_game(name="' + "a" * 22 + "…)"
    assert len(long) < len("resume_game") + len("a" * 60)


def test_a_rejected_move_reads_as_rejected_beside_its_arguments() -> None:
    """The marker moved out of the parens it used to sit in (`make_move(illegal)`
    would now be indistinguishable from an argument called `illegal`) and onto a
    `!` suffix — the same `!` a dispatch error already carried, with the reason
    spelled when there is one."""
    rejected = _trajectory(
        _assistant(
            _call("make_move", {"move": "Qh8"}, result='{"legal":false}'),
            _call("save_game", {"name": "x"}, result=None, error="no such slot"),
        )
    )

    assert rejected == 'make_move(move="Qh8")!illegal → save_game(name="x")!'


# --- the gates: what a sample may not pass on ---------------------------------
#
# Everything above is about *reporting* a sample. The rest of this file is about
# refusing one. Three conditions the audit found a sample passing under, none of
# which needs a model to reproduce: the wrong route answered the utterance, the
# reply was the pipeline's canned fallback rather than an answer, and the move
# the wire claimed was not the move the board kept.


def _finished_run(
    *,
    content: str = "Bishop takes. Rude.",
    stop_reason: str = "completed",
    route: str | None = ROUTE_BRAIN,
) -> EvalRun:
    """An `EvalRun` for a turn that finished, carrying what the gates read: the
    reply's text, the stop reason, and the route.

    The cost fields are built by the same splitters `_measured` calls, over
    empty readings — a run assembled here is a turn nobody metered, and
    hand-writing a `TurnLatencies` would let this file's idea of one drift from
    the only place that builds them for real.
    """
    return EvalRun(
        assistant={"content": content, "tool_calls": []},
        duration=1.0,
        model_calls=[],
        status_code=200,
        stop_reason=stop_reason,
        provider_failure=None,
        latencies=split_latencies((), route=route, stop_reason=stop_reason),
        tokens=split_tokens((), route=route, stop_reason=stop_reason),
        route=route,
    )


def test_the_canned_stuck_line_is_not_a_narrator_answer() -> None:
    """Audit finding 10. A truncated narrator keeps the planner's `completed` —
    the loop drops the fragment and no truncation field survives it — and the
    pipeline fills the empty reply with `api.STUCK_REPLY`. So `completed` plus
    nonempty text was not enough to prove an answer was tested: every negative
    commentary check in the suite passes over "I lost the thread on that one",
    and `advice_capture_survives_guard` passes it happily (no guard fired, and
    the line does not say "scratch that")."""
    with pytest.raises(VacuousRun, match="canned stuck line"):
        _assert_reached_narrator(_finished_run(content=STUCK_REPLY))


def test_a_reply_carrying_the_stuck_line_is_rejected_whatever_else_it_says() -> None:
    # Substring rather than equality: the fallback reaches the wire as the
    # commentary of a turn that may also have appended the engine's canned reply
    # line, and a stuck answer with a move announcement after it is still stuck.
    with pytest.raises(VacuousRun):
        _assert_reached_narrator(_finished_run(content=f"{STUCK_REPLY}\n\nNf6."))


def test_ordinary_prose_reaches_the_narrator() -> None:
    # The other half, and the reason the check is a substring of one app-owned
    # constant rather than anything about wording: a real answer passes.
    _assert_reached_narrator(_finished_run(content="You're up a pawn. Keep pushing."))


def test_a_budget_stop_never_reached_a_narrator_at_all() -> None:
    """The gate's original rule, pinned now that it has company: a turn that
    ended on its budget produced no commentary, so the text it carries is not
    the narrator's either."""
    with pytest.raises(VacuousRun, match="no narrator ran"):
        _assert_reached_narrator(_finished_run(stop_reason="max_iterations"))


def test_a_short_circuited_utterance_fails_a_model_scenario() -> None:
    """Audit finding 9, as the assertion that would have caught it: the message
    has to name both routes, because "this scenario measured nothing" is only
    actionable if it says what answered instead."""
    with pytest.raises(AssertionError) as excinfo:
        _assert_route(_finished_run(route=ROUTE_RESIGN), ROUTE_BRAIN)

    message = str(excinfo.value)
    assert ROUTE_RESIGN in message and ROUTE_BRAIN in message


def test_the_expected_route_passes() -> None:
    _assert_route(_finished_run(route=ROUTE_BRAIN), ROUTE_BRAIN)


def test_no_expected_route_asserts_nothing() -> None:
    # For a scenario whose route legitimately varies. Opting out is spelled at
    # the call site (`_pass_rate(route=None)`) rather than being what happens
    # when nobody thought about it.
    _assert_route(_finished_run(route=ROUTE_RESIGN), None)


def test_a_scenario_the_fast_path_answers_comes_back_below_the_floor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The gate end to end, with a scripted runner where the model would be.

    This is finding 9 as a regression test: five samples, every one of them
    answered on another route, and the scenario has to come back BELOW_FLOOR
    naming the route rather than ABOVE_FLOOR having tested nothing. The check
    never runs — a wrong route makes a scenario's own assertions meaningless
    rather than failed, and the resign short-circuit satisfies every one of the
    resignation checks by itself.
    """
    # A deterministic test must never append to a measurement report, even when
    # one is being collected in the same shell.
    monkeypatch.setattr("test_agent_evals._REPORT_PATH", None)
    checked: list[str] = []

    def runner(app: EvalApp, scenario: str, utterance: str) -> EvalRun:
        # What a short-circuit leaves behind: a finished turn on another route.
        app.tracer.record({"route": ROUTE_FAST_PATH, "stop_reason": "completed"})
        return _finished_run(route=ROUTE_FAST_PATH)

    def check(app: EvalApp, assistant: dict[str, Any]) -> None:
        checked.append(assistant["content"])

    result = _pass_rate(
        FakeEngine(),  # type: ignore[arg-type]
        "fast_path_answered_it",
        "I resign",
        check,
        floor=0.8,
        runner=runner,
    )

    assert result.decision is Decision.BELOW_FLOOR
    assert result.runs == _BLOCK_RUNS and result.passed == 0
    assert all(
        ROUTE_FAST_PATH in failure and ROUTE_BRAIN in failure
        for failure in result.failures
    ), result.failures
    assert checked == [], "the route pin must fail before the scenario's check runs"


# --- which move landed --------------------------------------------------------


def _board_app(*sans: str) -> EvalApp:
    """An `EvalApp` whose `ctx` is a real game at `_LIVE_FEN` with `sans` played
    on it, and nothing else wired.

    `client` is None on purpose: `_expect_san` reads the session now, and a
    check that reached for the HTTP seam instead should fail here rather than be
    accommodated. The session is FEN-rooted, like the scenarios', so its move
    history is exactly what the turn under test did.
    """
    ctx = ToolContext(session=GameSession(fen=_LIVE_FEN))
    for san in sans:
        assert ctx.session.submit_move(san).legal
    return EvalApp(
        client=None,  # type: ignore[arg-type]
        ctx=ctx,
        provider=CountingProvider(inner=None),
        tracer=_CollectingTracer(),
    )


# The accepted capture, as the delegate wire carries it: `legal:true` plus the
# SAN the tool reported. This entry alone used to be the whole of the check.
_ACCEPTED_BXE6 = _call(
    "make_move", {"move": "c4e6"}, result='{"legal":true,"san":"Bxe6"}'
)


def test_a_move_undone_in_the_same_turn_is_not_the_move_that_landed() -> None:
    """The audit's example, verbatim: the wire says Bxe6 was accepted and the
    board says nothing was played. `long_capture` — the release-blocking
    scenario — passed this, because the SAN was read out of the tool result by a
    helper whose docstring claimed it read the board back."""
    with pytest.raises(AssertionError, match="only thing that moved a piece"):
        _expect_san("Bxe6")(
            _board_app(),
            _assistant(
                _ACCEPTED_BXE6, _call("undo", {"plies": 2}, result='{"ok":true}')
            ),
        )


def test_a_move_the_engine_never_answered_has_not_settled() -> None:
    """One ply on the board and the engine to move is a turn that stopped
    half-way through the exchange — the pipeline owes a reply it never
    collected (`docs/turn-coordinator.md`), and the board the player is looking
    at is not one they can move on."""
    app = _board_app("Bxe6")
    assert app.ctx.session.turn != app.ctx.session.player_color  # engine's move

    with pytest.raises(AssertionError, match="exactly one engine reply"):
        _expect_san("Bxe6")(app, _assistant(_ACCEPTED_BXE6))


def test_the_move_plus_one_reply_with_the_player_to_move_passes() -> None:
    app = _board_app("Bxe6", "fxe6")

    _expect_san("Bxe6")(app, _assistant(_ACCEPTED_BXE6))

    assert app.ctx.session.move_history() == ["Bxe6", "fxe6"]
    assert app.ctx.session.turn == app.ctx.session.player_color


def test_a_different_move_on_the_board_fails_whatever_the_wire_says() -> None:
    # The plain case the old check did cover, kept: the tool result is not
    # evidence about the board, in either direction. Bd5 is the same bishop
    # going somewhere else — a settled exchange, and the wrong one.
    app = _board_app("Bd5", "Nf6")

    with pytest.raises(AssertionError, match="expected Bxe6 on the board"):
        _expect_san("Bxe6")(app, _assistant(_ACCEPTED_BXE6))


# --- the multi-step runner ----------------------------------------------------
#
# `_run_steps` composes a probe out of several utterances: earlier steps set up
# the state, and the last one is what `_pass_rate` scores. It touches the model
# only through the seam it is given, so all of its own behavior — the order, the
# meter resets, what it snapshots, and what it does when a step dies — is
# testable here with a scripted seam where the wire would be.
#
# The one rule it must not break is that it never asserts: `_sample` calls the
# runner *outside* the try/except that classifies a sample, so an AssertionError
# raised inside it aborts the scenario instead of scoring one miss. Everything
# it learns therefore has to travel in the record.


def _composite_app() -> EvalApp:
    """An `EvalApp` with a real game and real meters, and no wire.

    `client` is None on purpose: `_run_steps` reaches for the HTTP seam only
    through the runner it is handed, so a version that posted for itself should
    fail here rather than be accommodated.
    """
    return EvalApp(
        client=None,  # type: ignore[arg-type]
        ctx=ToolContext(session=GameSession(), engine=FakeEngine()),
        provider=CountingProvider(inner=None),
        tracer=_CollectingTracer(),
    )


def _scripted_seam(
    deaths: dict[str, EvalRun] | None = None,
    plays: dict[str, str] | None = None,
) -> tuple[Any, list[dict[str, Any]]]:
    """A stand-in for `_run_panel`: records what it was handed, spends one model
    round trip and one trace record, and answers.

    Deliberately does **not** reset the meters, which both real seams do — that
    is what makes "the composite resets them itself" an assertion rather than an
    accident of which seam it happens to be given.

    `deaths` maps an utterance to the `EvalRun` it should come back as (a 502, a
    provider death); `plays` maps one to a SAN the step submits, so a step's
    effect on the board is real and the snapshot can be checked against it.
    """
    deaths = deaths or {}
    plays = plays or {}
    seen: list[dict[str, Any]] = []

    def seam(app: EvalApp, scenario: str, utterance: str) -> EvalRun:
        seen.append(
            {
                "scenario": scenario,
                "utterance": utterance,
                # What the composite left standing when this step started.
                "calls_at_entry": len(app.provider.calls),
                "traced_at_entry": len(app.tracer.turns),
            }
        )
        app.provider.calls.append(ModelCall(thinking=False, seconds=0.1))
        app.tracer.record(
            {"route": ROUTE_BRAIN, "stop_reason": "completed", "said": utterance}
        )
        if utterance in plays:
            assert app.ctx.session.submit_move(plays[utterance]).legal
        if utterance in deaths:
            return deaths[utterance]
        # The count comes off the meter, exactly as `_measured` builds it.
        return _finished_run(content=f"said {utterance}")._replace(
            model_calls=list(app.provider.calls)
        )

    return seam, seen


def test_the_steps_run_in_order_and_the_last_one_is_what_is_scored() -> None:
    record: dict[str, Any] = {}
    seam, seen = _scripted_seam()

    final = _run_steps("first", "second", record=record, seam=seam)(
        _composite_app(), "probe", "last"
    )

    assert [entry["utterance"] for entry in seen] == ["first", "second", "last"]
    # Each earlier step gets its own label, so the `[eval]` line and the report
    # can say which step of which sample a reading came from.
    assert [entry["scenario"] for entry in seen] == ["probe/1", "probe/2", "probe"]
    assert [step["utterance"] for step in record["steps"]] == ["first", "second"]
    assert final.assistant["content"] == "said last", (
        "the final step's run is what `_pass_rate` scores"
    )


def test_the_meters_are_reset_between_steps() -> None:
    """Every step is measured on its own, which is the whole reason a
    multi-utterance probe can pin a per-step call count at all. The scripted
    seam never resets, so a running total is what would show up here."""
    record: dict[str, Any] = {}
    seam, seen = _scripted_seam()

    _run_steps("first", "second", record=record, seam=seam)(
        _composite_app(), "probe", "last"
    )

    assert [entry["calls_at_entry"] for entry in seen] == [0, 0, 0]
    assert [entry["traced_at_entry"] for entry in seen] == [0, 0, 0]
    assert [step["model_calls"] for step in record["steps"]] == [1, 1]


def test_each_step_records_the_state_it_left_behind() -> None:
    """The snapshot is taken *after* the step, so a check can say "the first ask
    moved nothing" — and so the board the *next* utterance met is on the record,
    which is where a two-step probe's parser premise has to be asserted."""
    record: dict[str, Any] = {}
    seam, _ = _scripted_seam(plays={"open with e4": "e4"})
    app = _composite_app()

    _run_steps("open with e4", record=record, seam=seam)(app, "probe", "and now?")

    step = _step(record, 1)
    assert step["history"] == ["e4"]
    assert step["fen"] == app.ctx.session.fen()
    assert step["settings"] == app.ctx.settings.snapshot()
    assert step["traced"]["said"] == "open with e4", (
        "the trace record is copied, so the next step's reset cannot take it"
    )
    assert step["pending"] is None


@pytest.mark.parametrize(
    "dead",
    [
        pytest.param(_finished_run()._replace(status_code=502), id="transport"),
        pytest.param(
            _finished_run(stop_reason=STOP_PROVIDER_ERROR), id="provider_error"
        ),
    ],
)
def test_a_dead_step_short_circuits_and_is_returned_as_the_final(
    dead: EvalRun,
) -> None:
    """A step the provider died on is not a miss — the model was never asked.
    So the runner stops there and hands that run back as though it were the
    final one, which is the only way `classify` gets to see the death and
    `_pass_rate` gets to re-take the sample rather than score it."""
    record: dict[str, Any] = {}
    seam, seen = _scripted_seam(deaths={"first": dead})

    final = _run_steps("first", "second", record=record, seam=seam)(
        _composite_app(), "probe", "last"
    )

    assert final is dead
    assert [entry["utterance"] for entry in seen] == ["first"], (
        "nothing after a dead step should have run"
    )
    assert len(record["steps"]) == 1, "the dead step is still on the record"
    assert (
        classify(
            status_code=final.status_code,
            stop_reason=final.stop_reason,
            error=None,
        )
        is Outcome.INFRA
    )


def test_a_composite_runner_is_filed_under_the_seam_it_drove() -> None:
    """The report names each scenario's seam by identity, and a composite is
    neither `_run` nor `_run_panel` — so without the seam it carries, every
    multi-step probe would be recorded against the delegate wire it never
    touched. The report is the baseline's evidence; it does not get to be
    wrong about which seam produced a number."""
    seam, _ = _scripted_seam()

    assert _seam_name(_run_panel) == "panel"
    assert _seam_name(_run) == "delegate"
    assert _seam_name(_run_steps("first", record={})) == "panel"
    assert _seam_name(_run_steps("first", record={}, seam=_run)) == "delegate"
    assert _seam_name(_run_steps("first", record={}, seam=seam)) == "delegate"


def test_the_record_is_cleared_before_every_sample() -> None:
    """A retry or an escalation rebuilds the whole app (`_sample`), and the
    record is the scenario's, not the sample's — so last sample's steps would be
    read as this one's if it were merely appended to."""
    record: dict[str, Any] = {"steps": [{"utterance": "a previous sample"}], "old": 1}
    seam, _ = _scripted_seam()

    _run_steps("first", record=record, seam=seam)(_composite_app(), "probe", "last")

    assert [step["utterance"] for step in record["steps"]] == ["first"]
    assert "old" not in record


def test_a_step_that_never_ran_is_a_harness_bug_not_a_missed_sample() -> None:
    """`_step` raises `LookupError` rather than asserting, and the difference is
    what an xfail or a rate line then means: `classify` reads a non-assertion as
    HARNESS, which is never retried and never scored, while an AssertionError
    would be written down as the model getting it wrong."""
    with pytest.raises(LookupError, match="never recorded"):
        _step({"steps": []}, 1)

    assert (
        classify(
            status_code=200,
            stop_reason="completed",
            error=LookupError("step 1 was never recorded"),
        )
        is Outcome.HARNESS
    )


# --- the scenarios' premises, pinned in CI ------------------------------------
#
# A premise that only holds inside an opt-in suite is a premise nobody checks:
# `parse_resign` grew the clause-and-filler rule and four scenarios silently
# stopped being model evals. These are the statements the resignation scenarios
# and the generic tool helpers are built on, asserted where every commit runs.


def test_the_literal_resignation_belongs_to_the_parser() -> None:
    """What makes `resign_literal_fast_path` a meaningful scenario: this
    utterance really is settled deterministically, so its zero-model assertion
    is a lock on the short-circuit and not a coincidence."""
    assert parse_resign(_RESIGN_LITERAL) is True


def test_the_planner_resignation_reaches_the_planner() -> None:
    """And what makes `resign_never_pretends` and the three `long_resign`
    conditions model evals again. Both boards they run on, because `parse_move`
    answers per position: the six-ply game the fresh scenario replays, and the
    long-transcript family's `_LIVE_FEN`."""
    assert parse_resign(_RESIGN_UTTERANCE) is False
    session = GameSession()
    for san in ("e4", "e5", "Nf3", "Nc6", "Bb5", "a6"):
        assert session.submit_move(san).legal

    _stays_a_model_eval(_RESIGN_UTTERANCE, session.fen())
    _stays_a_model_eval(_RESIGN_UTTERANCE, _LIVE_FEN)


def test_the_setup_guard_names_the_parser_that_would_swallow_the_utterance() -> None:
    """A guard nobody has watched fire is a guard nobody should trust — and this
    one has to say *which* parser, because the two failures need different
    fixes (a new utterance, or a different board)."""
    fresh = GameSession().fen()

    with pytest.raises(AssertionError, match="parse_move"):
        _stays_a_model_eval("e4", fresh)
    with pytest.raises(AssertionError, match="parse_resign"):
        _stays_a_model_eval(_RESIGN_LITERAL, fresh)


def test_the_generic_tool_sets_name_real_tools() -> None:
    """Both helpers are membership tests against tool names, so a rename in
    `tools.py` degrades them silently: `_board_mutations` would stop seeing a
    mutation, and "no verdict tool ran" would pass for a verdict tool nobody
    can spell any more."""
    registry = build_registry(ToolContext(session=GameSession(), engine=FakeEngine()))
    registered = {
        definition["function"]["name"] for definition in registry.definitions()
    }

    assert _BOARD_TOOLS <= registered, sorted(_BOARD_TOOLS - registered)
    assert _VERDICT_TOOLS <= registered, sorted(_VERDICT_TOOLS - registered)


def test_every_way_to_end_a_game_counts_as_a_board_mutation() -> None:
    """`claim_draw` ends a game and was in `tools.DESTRUCTIVE_TOOLS` for a
    release before the harness's board set heard of it, which left every "this
    turn moved nothing" assertion blind to a claimed draw. The relationship is
    pinned rather than the literal, so the next tool that ends a game cannot
    repeat it."""
    assert set(DESTRUCTIVE_TOOLS) <= _BOARD_TOOLS, sorted(
        set(DESTRUCTIVE_TOOLS) - _BOARD_TOOLS
    )
    assert "claim_draw" in _BOARD_TOOLS


def test_the_stt_phrasings_still_reach_the_planner() -> None:
    """`stt_knight_repair` measures the model repairing a mangled transcript, so
    the day `parse_move` learns to read "night" or to skip a filler, the
    scenario silently becomes a parser test. Pinned where every commit runs,
    because the alternative is finding out from a suspiciously perfect rate."""
    fresh = GameSession().fen()
    for utterance, _label in (param.values for param in _STT_UTTERANCES):
        _stays_a_model_eval(utterance, fresh)


def test_the_freeform_confirmation_answers_need_a_model_to_read_them() -> None:
    """And the same premise for the confirmation family, against the third
    parser: `parse_confirmation` is a short list of bare affirmations, and an
    answer it settles costs no reader call at all — which would make every one
    of that scenario's call counts a measurement of the list."""
    fresh = GameSession().fen()
    for answer, _label, _route in (param.values for param in _FREEFORM_ANSWERS):
        assert parse_confirmation(answer) is None, answer
        _stays_a_model_eval(answer, fresh)


def test_the_late_game_fixture_is_the_game_the_scenario_claims() -> None:
    """84 legal plies, White to move at move 43, not over.

    `late_game_tool_composition` asserts these as premises per sample, but the
    fixture is a checked-in file and this is the cheapest place to notice it
    being edited or truncated — and the replay itself is the point: a FEN alone
    would give the same position with an empty move stack, which is exactly the
    condition this fixture exists to escape.
    """
    app = EvalApp(
        client=None,  # type: ignore[arg-type]
        ctx=ToolContext(session=GameSession(), engine=FakeEngine()),
        provider=CountingProvider(inner=None),
        tracer=_CollectingTracer(),
    )

    history = _replay_late_game(app)

    assert len(history) == 84
    assert app.ctx.session.turn == "white" == app.ctx.session.player_color
    _stays_a_model_eval(
        "save this as late_game and tell me the position", app.ctx.session.fen()
    )


def test_a_whole_game_review_is_a_verdict() -> None:
    # `review_game` classifies every move and scores accuracy off the engine —
    # the same "how good is this?" answer `evaluate_position` gives, over the
    # whole game. A description ask answered with one is the walkthrough defect
    # wearing a bigger tool.
    assert "review_game" in _VERDICT_TOOLS
