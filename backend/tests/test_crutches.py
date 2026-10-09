"""The 12B's crutches go through the planner's profile (#375).

Five pieces of guidance exist only because gemma-4-12b needed them
(docs/model-profiles.md). The gemma profile lists all five, and its offer is
pinned to the one recorded from `main` before the switch existed
(`fixtures/gemma_4_12b_offer.json`). A planner whose profile lists none gets
the plain text. The registry, which the MCP server and the delegate wire
read, keeps every word whatever the profile says.
"""

import json
from pathlib import Path

import pytest

from chessapp.app import DEFAULT_MODEL, crutches_from_env
from chessapp.coordinator import TurnCoordinator
from chessapp.game import GameSession
from chessapp.profiles import (
    ASK_PLAYER_EXAMPLES,
    DIFFICULTY_CONSTRAINT_RULE,
    KNOWN_CRUTCHES,
    MOVE_SOURCE_REQUIRED,
    PICK_RESUBMIT_SCRIPT,
    UNDO_CALL_AGAIN,
    load_profile,
)
from chessapp.tools import (
    _TEXT_CRUTCHES,
    ToolContext,
    _text_pattern,
    brain_tool_definitions,
    build_registry,
)
from fakes import FakeEngine

_GOLDEN = json.loads(
    (Path(__file__).parent / "fixtures" / "gemma_4_12b_offer.json").read_text()
)


def _offer(crutches: frozenset[str]) -> tuple[list[dict], list[dict], ToolContext]:
    ctx = ToolContext(session=GameSession(), engine=FakeEngine(), crutches=crutches)
    registry = build_registry(ctx, TurnCoordinator(ctx), atomic_exchange=False)
    return brain_tool_definitions(registry, ctx), registry.definitions(), ctx


def _by_name(definitions: list[dict]) -> dict[str, dict]:
    return {d["function"]["name"]: d["function"] for d in definitions}


def test_the_gemma_offer_is_the_one_recorded_before_the_switch():
    offer, registry, _ = _offer(load_profile("gemma-4-12b").crutches)
    assert offer == _GOLDEN["offer"]
    assert registry == _GOLDEN["registry"]


def test_every_text_crutch_is_still_worded_the_way_it_is_cut():
    # A reworded docstring must fail here, not leave the crutch in silently.
    registry = _by_name(_GOLDEN["registry"])
    for crutch, (tool, text, _) in _TEXT_CRUTCHES.items():
        assert _text_pattern(text).search(registry[tool]["description"]), crutch


def test_without_crutches_the_offer_is_the_plain_text():
    offer, registry, _ = _offer(frozenset())
    tools = _by_name(offer)
    golden = _by_name(_GOLDEN["offer"])

    assert "call this again" not in tools["undo"]["description"]
    assert tools["undo"]["description"].endswith(
        "leaving the player to move again. When they point back to a move by "
        "its number, pass\nbefore_move."
    )
    assert "king's knight" not in tools["ask_player"]["description"]
    assert tools["ask_player"]["description"].endswith("the moves\nthat fit.")
    assert tools["set_difficulty"]["description"].endswith(
        "The setting persists and the player owns it."
    )
    source = tools["make_move"]["parameters"]
    assert "source" not in source.get("required", [])
    assert source["properties"]["source"]["enum"] == [
        "said_the_move",
        "picked_by_position",
    ]
    # Nothing else in the offer moved, and the registry kept every word.
    changed = {name for name in tools if tools[name] != golden[name]}
    assert changed == {"undo", "ask_player", "set_difficulty", "make_move"}
    assert registry == _GOLDEN["registry"]


@pytest.mark.parametrize(
    ("crutch", "tool"),
    [
        (UNDO_CALL_AGAIN, "undo"),
        (ASK_PLAYER_EXAMPLES, "ask_player"),
        (DIFFICULTY_CONSTRAINT_RULE, "set_difficulty"),
        (MOVE_SOURCE_REQUIRED, "make_move"),
    ],
)
def test_dropping_one_crutch_changes_only_its_tool(crutch, tool):
    offer, _, _ = _offer(KNOWN_CRUTCHES - {crutch})
    tools = _by_name(offer)
    golden = _by_name(_GOLDEN["offer"])
    assert {name for name in tools if tools[name] != golden[name]} == {tool}


def _refusal(crutches: frozenset[str]) -> str:
    _, _, ctx = _offer(crutches)
    registry = build_registry(ctx, TurnCoordinator(ctx), atomic_exchange=False)
    asked = registry.dispatch("ask_player", {"piece": "knight", "which": "king's"})
    assert asked["ok"], asked
    refused = registry.dispatch(
        "make_move", {"move": "e4", "source": "picked_by_position"}
    )
    assert not refused["ok"]
    return refused["error"]


def test_the_pick_refusal_scripts_the_resubmit_only_as_a_crutch():
    assert "resubmit" in _refusal(KNOWN_CRUTCHES)
    plain = _refusal(KNOWN_CRUTCHES - {PICK_RESUBMIT_SCRIPT})
    assert "resubmit" not in plain
    # The code check itself stays for every model: the move is still refused,
    # and the moves that stand are still named.
    assert "a pick by position must be one of the moves" in plain


def test_a_bare_context_keeps_every_crutch():
    # MCP and the delegate wire build their own context: today's words.
    assert ToolContext(session=GameSession()).crutches == KNOWN_CRUTCHES


def test_crutches_come_from_the_planner_profile_unless_overridden(monkeypatch):
    monkeypatch.delenv("CHESSAPP_CRUTCHES", raising=False)
    assert crutches_from_env("gemma-4-12b") == KNOWN_CRUTCHES
    assert crutches_from_env("unprofiled-model") == frozenset()
    monkeypatch.setenv("CHESSAPP_CRUTCHES", "none")
    assert crutches_from_env("gemma-4-12b") == frozenset()
    monkeypatch.setenv("CHESSAPP_CRUTCHES", "all")
    assert crutches_from_env("unprofiled-model") == KNOWN_CRUTCHES
    monkeypatch.setenv("CHESSAPP_CRUTCHES", "undo_call_again, ask_player_examples")
    assert crutches_from_env("x") == {UNDO_CALL_AGAIN, ASK_PLAYER_EXAMPLES}
    monkeypatch.setenv("CHESSAPP_CRUTCHES", "undo_call_agian")
    with pytest.raises(ValueError):
        crutches_from_env("x")


def test_the_app_gives_the_context_the_planners_crutches(monkeypatch):
    import chessapp.app

    seen: dict[str, frozenset[str]] = {}
    real = chessapp.app.build_registry

    def spy(ctx, *args, **kwargs):
        seen["crutches"] = ctx.crutches
        return real(ctx, *args, **kwargs)

    monkeypatch.setattr(chessapp.app, "build_registry", spy)
    chessapp.app.build_app(agent_enabled=False)
    assert seen["crutches"] == load_profile(DEFAULT_MODEL).crutches
    chessapp.app.build_app(
        agent_enabled=False, phase_models={"planner": "unprofiled-model"}
    )
    assert seen["crutches"] == frozenset()
