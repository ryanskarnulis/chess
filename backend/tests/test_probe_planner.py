"""The planner probe's pure parts, off the GPU (#286, Phase 0).

`scripts/probe_planner.py` measures a live 12B and cannot be tested against
it; what *can* be pinned is everything around the one model call — that the
request it builds is the shipped planner call, how a wire result is
classified and scored, the interleaved schedule, the clustering statistic,
the pre-flight decision, and the corpus premises (a position that stops being
ambiguous silently turns the ask into a different measurement). Same rule as
`test_evalstats.py`: every decision the tool makes lives in a function with
no I/O, and this file is its spec.
"""

import json
from pathlib import Path

import pytest

import campaign_report
from chessapp.fastparse import parse_move
from chessapp.personality import PLANNER_PROMPT
from chessapp.provider import ChatResult, LlamaCppProvider, ToolCall
from chessapp.tools import BOARD_STATE_TOOLS
from probe_planner import (
    CORPUS,
    Arm,
    busy_slots,
    classify,
    corpus,
    foreign_gpu_processes,
    format_summary,
    independent_agreement,
    lag1_agreement,
    outcome_label,
    parse_arm,
    passes,
    position,
    preflight_reasons,
    prepare,
    schedule,
    sha,
    summarize,
)


def _result(*calls: tuple[str, dict]) -> ChatResult:
    return ChatResult(
        content=None if calls else "Which one?",
        tool_calls=[
            ToolCall(id=str(i), name=name, arguments=args)
            for i, (name, args) in enumerate(calls)
        ],
        finish_reason="tool_calls" if calls else "stop",
        usage=None,
    )


# --- classification and scoring -----------------------------------------------


def test_a_reply_with_no_calls_is_no_tool() -> None:
    assert classify(_result()) == []
    assert outcome_label([]) == "no_tool"


def test_calls_are_read_off_the_wire_with_the_arguments_the_rules_need() -> None:
    calls = classify(_result(("undo", {"plies": 2}), ("make_move", {"move": "d4"})))
    assert calls == [
        {"name": "undo", "args": {"plies": 2}},
        {"name": "make_move", "args": {"move": "d4"}},
    ]
    assert outcome_label(calls) == "undo(plies=2)|make_move(d4)"
    assert outcome_label([{"name": "undo", "args": {}}]) == "undo"
    assert outcome_label([{"name": "get_best_moves", "args": {"n": 3}}]) == (
        "get_best_moves"
    )


def test_the_no_tool_rule_passes_only_silence() -> None:
    assert passes(("no_tool",), [])
    assert not passes(("no_tool",), classify(_result(("make_move", {"move": "Nf3"}))))


def test_the_first_call_rule_reads_the_move_argument() -> None:
    rule = ("first_call", "make_move", {"move_in": ["Nf3", "g1f3"]})
    assert passes(rule, classify(_result(("make_move", {"move": "g1f3"}))))
    assert not passes(rule, classify(_result(("make_move", {"move": "Nh3"}))))
    assert not passes(rule, classify(_result(("get_legal_moves", {}))))
    assert not passes(rule, [])


def test_the_first_call_rule_can_require_an_argument_to_be_omitted() -> None:
    rule = ("first_call", "undo", {"absent": ["plies"]})
    assert passes(rule, classify(_result(("undo", {}), ("make_move", {"move": "d4"}))))
    assert not passes(rule, classify(_result(("undo", {"plies": 2}))))
    assert not passes(
        rule, classify(_result(("make_move", {"move": "d4"}), ("undo", {})))
    )


def test_an_unknown_rule_is_a_bug_not_a_miss() -> None:
    with pytest.raises(ValueError):
        passes(("sometimes",), [])


# --- the corpus ---------------------------------------------------------------


def _legal(item_name: str) -> set[str]:
    item = next(i for i in CORPUS if i.name == item_name)
    return set(position(item).legal_moves())


def test_every_setup_is_legal_and_every_position_holds_its_premise() -> None:
    for item in CORPUS:
        position(item)  # asserts each setup move is legal
    assert {"Nf3", "Nh3"} <= _legal("knight_ask")
    assert len([m for m in _legal("rook_ask") if m.startswith("R")]) == 4
    assert {"O-O", "O-O-O"} <= _legal("castle_both")
    assert {m for m in _legal("castle_one") if m.startswith("O-O")} == {"O-O"}
    assert len({m[0] for m in _legal("bishop_ask") if m.startswith("B")}) == 1
    assert len([m for m in _legal("bishop_ask") if m.startswith("B")]) >= 2
    assert not any("x" in m for m in _legal("take_pawn"))
    assert "Nf3" in _legal("stt_knight")


def test_the_asks_the_planner_must_answer_are_not_settled_by_the_parser() -> None:
    # An item the fast path settles never reaches the planner in the app, so a
    # probe result on it would measure nothing the player meets. `castle_one`
    # is the documented exception: the probe keeps it as a planner neighbour.
    for item in CORPUS:
        settled = parse_move(item.utterance, position(item).fen())
        if item.name == "castle_one":
            assert settled == "O-O"
        else:
            assert settled is None, (item.name, settled)


def test_held_out_items_stay_out_unless_named_or_included() -> None:
    default = {i.name for i in corpus(None, include_held_out=False)}
    assert "bishop_ask" not in default and "castle_both" not in default
    assert "knight_ask" in default
    assert {i.name for i in corpus(None, include_held_out=True)} == {
        i.name for i in CORPUS
    }
    assert [i.name for i in corpus(["castle_both", "knight_ask"], False)] == [
        "castle_both",
        "knight_ask",
    ]
    with pytest.raises(SystemExit):
        corpus(["queen_ask"], False)


# --- arms ---------------------------------------------------------------------


def test_control_is_the_shipped_planner() -> None:
    arm = parse_arm("control")
    assert arm == Arm(name="control")
    assert arm.prompt == PLANNER_PROMPT
    assert (arm.temperature, arm.cache_prompt, arm.model) == (None, None, None)


def test_an_arm_spec_sets_exactly_the_knobs_it_names() -> None:
    files = {Path("p.txt"): "new prompt", Path("mm.txt"): "new make_move text"}
    arm = parse_arm(
        "x:temperature=0.3,cache_prompt=false,model=qwen,prompt=@p.txt,"
        "tool_text=make_move@mm.txt",
        read=files.__getitem__,
    )
    assert arm.temperature == 0.3
    assert arm.cache_prompt is False
    assert arm.model == "qwen"
    assert arm.prompt == "new prompt"
    assert arm.tool_text == {"make_move": "new make_move text"}
    assert parse_arm("y:cache_prompt=true").cache_prompt is True


@pytest.mark.parametrize(
    "spec",
    [":temperature=0.3", "x:temperature", "x:cache_prompt=maybe", "x:prompt=inline",
     "x:tool_text=nopath", "x:top_k=5"],
)  # fmt: skip
def test_a_malformed_arm_spec_refuses_to_start(spec: str) -> None:
    with pytest.raises(SystemExit):
        parse_arm(spec, read=lambda _p: "")


def test_a_tool_text_override_replaces_one_description_and_copies_the_rest() -> None:
    offer = [
        {"type": "function", "function": {"name": "make_move", "description": "old"}},
        {"type": "function", "function": {"name": "undo", "description": "keep"}},
    ]
    arm = Arm(name="t", tool_text={"make_move": "new"})
    offered = arm.offer(offer)
    assert [d["function"]["description"] for d in offered] == ["new", "keep"]
    assert offer[0]["function"]["description"] == "old", "the input was mutated"
    assert Arm(name="c").offer(offer) is offer
    with pytest.raises(SystemExit):
        Arm(name="t", tool_text={"resign": "x"}).offer(offer)


# --- the schedule -------------------------------------------------------------


def test_arms_alternate_on_consecutive_requests() -> None:
    order = list(schedule(2, ["knight", "rook"], ["control", "arm"]))
    assert order == [
        (0, "knight", "control"),
        (0, "knight", "arm"),
        (0, "rook", "control"),
        (0, "rook", "arm"),
        (1, "knight", "control"),
        (1, "knight", "arm"),
        (1, "rook", "control"),
        (1, "rook", "arm"),
    ]
    arms = [arm for _, _, arm in order]
    assert all(a != b for a, b in zip(arms, arms[1:], strict=False))


# --- statistics ---------------------------------------------------------------


def test_lag1_agreement_needs_two_samples_and_reads_clustering() -> None:
    assert lag1_agreement([]) is None
    assert lag1_agreement([True]) is None
    assert lag1_agreement([True, True, True, True]) == 1.0
    assert lag1_agreement([True, False, True, False]) == 0.0
    assert lag1_agreement([True, True, False, False]) == pytest.approx(2 / 3)


def test_independent_agreement_is_the_binomial_expectation() -> None:
    assert independent_agreement(0.5) == 0.5
    assert independent_agreement(1.0) == 1.0
    assert independent_agreement(0.8) == pytest.approx(0.68)


def _record(item: str, arm: str, outcome: str, passed: bool | None, error=None):
    return {
        "item": item,
        "arm": arm,
        "outcome": outcome,
        "passed": passed,
        "error": error,
    }


def test_summary_scores_clean_samples_and_counts_errors_apart() -> None:
    records = [
        _record("knight_ask", "control", "no_tool", True),
        _record("knight_ask", "arm", "make_move(Nf3)", False),
        _record("knight_ask", "control", "make_move(Nh3)", False),
        _record("knight_ask", "arm", "no_tool", True),
        _record("knight_ask", "control", "error", None, error="ProviderRequestError"),
        _record("knight_ask", "control", "no_tool", True),
    ]
    summary = summarize(records)
    control = summary[("knight_ask", "control")]
    assert (control["n"], control["passed"], control["errors"]) == (3, 2, 1)
    assert control["outcomes"] == {"no_tool": 2, "make_move(Nh3)": 1}
    assert control["lag1"] == 0.0  # pass, miss, pass
    assert control["lag1_independent"] == pytest.approx(independent_agreement(2 / 3))
    arm = summary[("knight_ask", "arm")]
    assert (arm["n"], arm["passed"], arm["errors"]) == (2, 1, 0)
    table = format_summary(summary)
    assert (
        "knight_ask | control | 2/3 +1 err | 0.00 (0.56) | make_move(Nh3) 1, no_tool 2"
        in table
    )


# --- pre-flight ---------------------------------------------------------------


def test_busy_slots_reads_llama_servers_slots_array() -> None:
    slots = [{"id": 0, "is_processing": False}, {"id": 1, "is_processing": True}]
    assert busy_slots(slots) == [1]
    assert busy_slots({"error": "model not loaded"}) == []
    assert busy_slots([]) == []


def test_foreign_gpu_processes_ignore_the_server_and_the_desktop() -> None:
    rows = [
        "25892, /usr/bin/kwin_wayland",
        "389724, /app/llama-server",
        "4242, /usr/bin/python3",
        "",
    ]
    assert foreign_gpu_processes(rows) == ["/usr/bin/python3"]


def test_preflight_refuses_a_busy_slot_or_a_foreign_job_and_nothing_else() -> None:
    busy = [{"id": 0, "is_processing": True}]
    assert preflight_reasons(["gemma-4-12b"], "gemma-4-12b", busy, []) == [
        "llama-server slot(s) [0] are processing"
    ]
    # Not loaded: whatever the slots endpoint said is not about this model.
    assert preflight_reasons([], "gemma-4-12b", busy, []) == []
    reasons = preflight_reasons([], "gemma-4-12b", [], ["1, /usr/bin/python3"])
    assert reasons == ["another job holds the GPU: /usr/bin/python3"]
    assert (
        preflight_reasons(["gemma-4-12b"], "gemma-4-12b", [], ["1, /app/llama-server"])
        == []
    )


# --- the request is the shipped planner call ----------------------------------


def test_prepare_builds_the_planner_call_the_app_makes() -> None:
    knight = next(i for i in CORPUS if i.name == "knight_ask")
    provider = LlamaCppProvider("http://llm.test/v1", "gemma-4-12b")
    prepared = prepare(
        knight, Arm(name="control"), provider, "http://llm.test/v1", "gemma-4-12b"
    )
    system, user = prepared.messages
    assert system == {"role": "system", "content": PLANNER_PROMPT}
    assert user["role"] == "user"
    head, command = user["content"].split("\n\nCommand: ")
    assert command == "move my kings knight"
    state = json.loads(head.removeprefix("Board state:\n"))
    assert {"Nf3", "Nh3"} <= set(state["legal_moves"])
    assert state["fen"] == prepared.fen
    offered = {d["function"]["name"] for d in prepared.tools}
    assert "make_move" in offered and "undo" in offered
    assert not offered & set(BOARD_STATE_TOOLS), "the board is injected, not offered"
    assert "claim_draw" not in offered
    assert prepared.prompt_sha == sha(PLANNER_PROMPT)
    assert prepared.fast_path is None


def test_prepare_applies_the_arm_and_changes_the_recorded_identity() -> None:
    knight = next(i for i in CORPUS if i.name == "knight_ask")
    provider = LlamaCppProvider("http://llm.test/v1", "gemma-4-12b")
    control = prepare(knight, Arm(name="control"), provider, "http://llm.test/v1", "m")
    arm = Arm(name="trim", prompt="shorter", tool_text={"make_move": "terse"})
    prepared = prepare(knight, arm, provider, "http://llm.test/v1", "m")
    assert prepared.messages[0]["content"] == "shorter"
    assert prepared.prompt_sha != control.prompt_sha
    assert prepared.offer_sha != control.offer_sha
    make_move = next(d for d in prepared.tools if d["function"]["name"] == "make_move")
    assert make_move["function"]["description"] == "terse"


# --- the harness aggregator ---------------------------------------------------


def _scenario(name: str, passed: int, runs: int, blocks: list[list[int]]) -> dict:
    return {
        "kind": "scenario",
        "scenario": name,
        "passed": passed,
        "runs": runs,
        "blocks": blocks,
    }


def test_block_reports_join_into_one_arm_table() -> None:
    reports = [
        ("a", [{"kind": "header"}, _scenario("knight", 5, 5, [[5, 5]])]),
        ("b", [_scenario("knight", 3, 5, [[3, 5]]), _scenario("rook", 5, 5, [[5, 5]])]),
        ("b", [_scenario("knight", 4, 5, [[4, 5]]), {"kind": "suite"}]),
        ("a", [_scenario("knight", 8, 10, [[3, 5], [5, 5]])]),
    ]
    table = campaign_report.aggregate(reports)
    assert table["knight"]["a"] == {"passed": 13, "runs": 15, "blocks": [5, 3, 5]}
    assert table["knight"]["b"] == {"passed": 7, "runs": 10, "blocks": [3, 4]}
    assert "a" not in table["rook"]
    text = campaign_report.format_table(table, ["a", "b"])
    assert "`knight` | 13/15 (5, 3, 5) | 7/10 (3, 4)" in text
    assert "`rook` | — | 5/5 (5)" in text
