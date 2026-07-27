"""The tool-free closing pass, asserted end-to-end (Sprint 2, audit item 9).

The planner/narrator split made the closing pass tool-free *by construction*:
the phase that writes the player's reply is a separate model call offered no
tools. These tests pin that guarantee at the pipeline level, with the real
registry and turn coordinator behind a real `LlamaBrain`, so no future wiring
change can quietly hand tools back to the phase that talks:

- On every route (brain loop, fast path, board drag), the model call that
  produces player-facing text carries `tools=None`.
- Once the turn's mutation budget is spent — the phase machine's one player
  move, or the command window's one destructive op — the tools the planner
  still holds are dead: a further mutation comes back as error result data and
  the board stands.

`test_llama_brain.py` covers the same properties at the brain seam with a fake
dispatcher; this file is the route-level assertion the audit asked for.
"""

from fastapi.testclient import TestClient

from chessapp.api import create_app
from chessapp.app import BOARD_STATE_TOOLS
from chessapp.coordinator import TurnCoordinator
from chessapp.game import GameSession
from chessapp.llama_brain import LlamaBrain
from chessapp.tools import ToolContext, build_registry
from fakes import FakeEngine, ScriptedProvider, text_turn, tool_calls_turn

PLANNER = "Pick the tools."
PERSONA = "You are a chess opponent."


def make_client(*turns, ctx: ToolContext | None = None):
    """The full pipeline over a real `LlamaBrain` whose provider is scripted —
    the one wiring that lets a route-level test see which calls carried tools,
    exactly as app assembly builds it (shared registry + coordinator,
    `atomic_exchange=False`, board-state reads not offered)."""
    if ctx is None:
        ctx = ToolContext(session=GameSession(), engine=FakeEngine())
    coordinator = TurnCoordinator(ctx)
    registry = build_registry(ctx, coordinator, atomic_exchange=False)
    provider = ScriptedProvider(*turns)
    brain = LlamaBrain(
        provider=provider,
        dispatcher=registry,
        tool_definitions=registry.definitions(exclude=BOARD_STATE_TOOLS),
        system_prompt=PERSONA,
        planner_prompt=PLANNER,
    )
    app = create_app(ctx, brain=brain, registry=registry, coordinator=coordinator)
    return TestClient(app), provider, ctx


def offered_tools(provider: ScriptedProvider) -> list[bool]:
    """Whether each recorded model call held tools, in order."""
    return [call["tools"] is not None for call in provider.calls]


def finished(ctx: ToolContext) -> ToolContext:
    """Fool's mate: a finished game, so the confirmation gate stands aside and
    only the command window's destructive budget is in the way."""
    for san in ("f3", "e5", "g4", "Qh4"):
        ctx.session.submit_move(san)
    return ctx


# --- the brain route: the player-facing call is tool-free, and the tools the
# --- planner keeps cannot mutate past the budget


def test_brain_route_closes_tool_free_and_a_second_move_is_dead():
    client, provider, ctx = make_client(
        tool_calls_turn(("make_move", {"move": "e4"})),
        tool_calls_turn(("make_move", {"move": "d4"})),  # budget already spent
        text_turn("note: played e4, second move refused"),
        text_turn("e4 it is."),
    )

    response = client.post("/api/command", json={"text": "play e4 and d4"}).json()

    # Planner turns hold tools; the narrator — the call whose text the player
    # reads — holds none. Structural, not the model declining.
    assert offered_tools(provider) == [True, True, True, False]
    assert response["commentary"].startswith("e4 it is.")
    # The second mutation was refused as result data, inside the same turn.
    results = [r["result"] for r in response["tool_results"]]
    assert results[0]["ok"] is True
    assert results[1]["ok"] is False
    # One player move, one engine reply — nothing the dead call could add.
    assert ctx.session.move_history() == ["e4", "e5"]


def test_brain_route_destructive_budget_spent_still_closes_tool_free():
    ctx = finished(ToolContext(session=GameSession(), engine=FakeEngine()))
    client, provider, ctx = make_client(
        tool_calls_turn(("new_game", {})),
        tool_calls_turn(("resign", {})),  # second destructive op in one command
        text_turn("note: reset, resign refused"),
        text_turn("Fresh board. Your move."),
        ctx=ctx,
    )

    response = client.post("/api/command", json={"text": "new game"}).json()

    assert offered_tools(provider) == [True, True, True, False]
    results = [r["result"] for r in response["tool_results"]]
    assert results[0]["ok"] is True
    assert results[1]["ok"] is False
    assert not ctx.session.is_game_over(), "the refused resign never ran"
    assert response["commentary"] == "Fresh board. Your move."


def test_brain_route_closes_tool_free_when_the_planner_repeats_itself():
    """The repeat stop is a *third* way into the narrator, and it must arrive
    the same way the other two do: tool-free, with the player reading real
    commentary rather than the pipeline's canned stuck line (which is what a
    budget stop would have produced before the loop learned to end a planner
    that had stopped making progress)."""
    client, provider, _ = make_client(
        tool_calls_turn(("evaluate_position", {})),
        tool_calls_turn(("evaluate_position", {})),  # nothing new — the last turn
        text_turn("Dead even. Your move."),
    )

    response = client.post("/api/command", json={"text": "how's it looking?"}).json()

    # Two planner turns, then the narrator — no third planner turn was bought.
    assert offered_tools(provider) == [True, True, False]
    assert response["commentary"] == "Dead even. Your move."


# --- the routes outside the loop: the model is only ever reached tool-free


def test_fast_path_route_never_reaches_the_model_with_tools():
    client, provider, ctx = make_client(text_turn("The classic."))

    client.post("/api/command", json={"text": "e4"})

    assert len(provider.calls) >= 1, "the observe beat narrated"
    assert offered_tools(provider) == [False] * len(provider.calls)
    assert ctx.session.move_history()[:1] == ["e4"]


def test_board_route_never_reaches_the_model_with_tools():
    client, provider, ctx = make_client(text_turn("Bold opening."))

    client.post("/api/game/move", json={"move": "e2e4"})

    assert len(provider.calls) >= 1, "the observe beat narrated"
    assert offered_tools(provider) == [False] * len(provider.calls)
    assert ctx.session.move_history()[:1] == ["e4"]
