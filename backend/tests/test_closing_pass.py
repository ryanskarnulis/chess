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
- A refusal leaves the turn exactly as it found it — the board it did not
  change, the engine reply the open turn still owes, and nothing armed for a
  later yes to find. The same wiring is what shows it: only a real brain over a
  scripted provider puts a refused call inside a real planner batch, and a real
  `read_answer` in front of a real destructive gate.

`test_llama_brain.py` covers the same properties at the brain seam with a fake
dispatcher; this file is the route-level assertion the audit asked for.
"""

import pytest
from fastapi.testclient import TestClient

from chessapp.api import create_app
from chessapp.coordinator import TurnCoordinator
from chessapp.game import GameSession
from chessapp.llama_brain import LlamaBrain
from chessapp.tools import BOARD_STATE_TOOLS, ToolContext, build_registry
from fakes import FakeEngine, ScriptedProvider, text_turn, tool_calls_turn

PLANNER = "Pick the tools."
PERSONA = "You are a chess opponent."

# A fifty-move claim already available, on a position nobody played into — so
# the confirmation gate stands aside (no player investment) and the command
# window's destructive budget is the only thing left in the way, which is what
# this file is about.
FIFTY_MOVE_FEN = "8/8/8/4k3/8/8/4K3/6R1 w - - 100 80"


class CollectedTurns:
    """The app's own `Tracer` seam (`trace.Tracer`), kept in memory.

    `mutations` — how many times the board actually moved — is stated in the
    trace and nowhere else in the HTTP answer, and it is the whole question
    when a call in the batch was refused.
    """

    def __init__(self) -> None:
        self.records: list[dict] = []

    def record(self, turn: dict) -> None:
        self.records.append(turn)


def make_client(*turns, ctx: ToolContext | None = None, tracer=None):
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
    app = create_app(
        ctx,
        brain=brain,
        registry=registry,
        coordinator=coordinator,
        tracer=tracer,
    )
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


def test_a_claimed_draw_spends_the_budget_and_still_closes_tool_free():
    """Claiming a draw ends the game, so it is one of the ops the command window
    budgets: the second one is dead result data and the narrator that tells the
    player about it still holds no tools."""
    ctx = ToolContext(session=GameSession(fen=FIFTY_MOVE_FEN), engine=FakeEngine())
    client, provider, ctx = make_client(
        tool_calls_turn(("claim_draw", {})),
        tool_calls_turn(("new_game", {})),  # second destructive op in one command
        text_turn("note: draw claimed, reset refused"),
        text_turn("Half a point each. Done."),
        ctx=ctx,
    )

    response = client.post("/api/command", json={"text": "claim the draw"}).json()

    assert offered_tools(provider) == [True, True, True, False]
    results = [r["result"] for r in response["tool_results"]]
    assert results[0]["ok"] is True
    assert results[1]["ok"] is False
    assert ctx.session.outcome().termination == "fifty_moves", "the refused reset"
    assert response["commentary"] == "Half a point each. Done."


# --- a refusal leaves the turn as it found it


def developed(ctx: ToolContext) -> ToolContext:
    """A real game under way: the player has something to lose, so the
    destructive gate arms instead of standing aside."""
    for san in ("e4", "e5", "Nf3", "Nc6"):
        ctx.session.submit_move(san)
    return ctx


@pytest.mark.parametrize(
    ("plies", "took_back", "retry", "history", "announcement"),
    [
        # A hundred half-moves were never played, so nothing is taken back —
        # and the turn that played e4 is still owed the engine's answer to it.
        (100, False, "never", ["e4", "e5"], "\n\ne5."),
        # The same refusal in JSON's one number type. It reaches the same
        # handler as an int (the registry narrows it) and is refused there.
        (100.0, False, "never", ["e4", "e5"], "\n\ne5."),
        # And the same spelling on a count that *is* takeable: 1.0 is one
        # half-move, it pops e4, and now there is nothing left to answer.
        (1.0, True, None, [], ""),
    ],
)
def test_whether_a_reply_is_owed_is_decided_by_whether_the_undo_landed(
    plies, took_back, retry, history, announcement
):
    """`make_move` and `undo` in one planner batch, with the undo refused.

    The tool used to abandon the open turn before finding out whether it could
    take anything back, so a refused undo threw away the engine reply the move
    beside it had just earned: e4 on the board, Black to move, the coordinator
    back awaiting the player, and nobody left to answer (audit 2026-09-05,
    finding 1). A refusal touches neither the coordinator nor the board now, so
    the open turn keeps owning its reply and the pipeline collects it at the
    close — while an undo that *lands* still takes the owed reply with it,
    which is the same rule read the other way.
    """
    turns = CollectedTurns()
    client, _, ctx = make_client(
        tool_calls_turn(("make_move", {"move": "e4"}), ("undo", {"plies": plies})),
        text_turn("played e4; the takeback is another matter"),
        text_turn("e4 is on."),
        tracer=turns,
    )

    response = client.post("/api/command", json={"text": "play e4 and undo that"})

    assert response.status_code == 200
    body = response.json()
    undo_result = body["tool_results"][1]["result"]
    assert undo_result["ok"] is took_back
    assert undo_result.get("retry") == retry
    assert ctx.session.move_history() == history
    assert ctx.session.turn == ctx.session.player_color, "the player is to move"
    # The app's own reply announcement rides on the commentary exactly when a
    # reply was owed and collected.
    assert body["commentary"] == f"e4 is on.{announcement}"
    (record,) = turns.records
    assert record["mutations"] == 2


def test_an_integral_float_argument_lands_instead_of_ending_the_command():
    """The batch that used to escape as a 500 (audit 2026-09-05, finding 5).

    `1.0` passes the `integer` schema — JSON has one number type and the
    validator honors that — and then met a Python slice, which did not: the
    `TypeError` left `dispatch` entirely, taking the rest of the batch and the
    trace with it and reaching the player as a server error on a board that had
    already moved. The value is narrowed to the integer the schema called it,
    so the call it was is the call that runs and every sibling still gets its
    turn.
    """
    client, _, ctx = make_client(
        tool_calls_turn(
            ("make_move", {"move": "e4"}),
            ("undo", {"plies": 1.0}),
            ("set_voice_output", {"enabled": True}),
        ),
        text_turn("moved, took it back, voice on"),
        text_turn("Taken back. Voice is on."),
    )

    response = client.post(
        "/api/command", json={"text": "play e4, undo it, and turn voice on"}
    )

    assert response.status_code == 200
    body = response.json()
    assert [r["name"] for r in body["tool_results"]] == [
        "make_move",
        "undo",
        "set_voice_output",
    ]
    assert body["tool_results"][1]["result"]["undone"] == ["e4"]
    assert ctx.session.move_history() == []
    assert ctx.settings.voice_output is True, "the sibling after it still ran"


def test_a_truncated_reading_leaves_the_armed_resignation_un_run():
    """The confirmation reader's cap fires directly in front of the destructive
    gate, so a fragment read generously would be a game ended by a truncation.

    The planner asks to resign, the gate refuses and arms it, and the player
    answers in their own words — a reply only the model can read. The reading
    comes back cut off, which is not a verdict, so it lands on `unrelated`: the
    op is dropped rather than run, and the utterance falls through to an
    ordinary turn.
    """
    ctx = developed(ToolContext(session=GameSession(), engine=FakeEngine()))
    client, _, ctx = make_client(
        tool_calls_turn(("resign", {})),
        text_turn("asked first"),
        text_turn("That's the game if you mean it. Resign?"),
        # The reading, cut off mid-word by `_ANSWER_MAX_TOKENS`.
        text_turn("confirm", finish_reason="length"),
        text_turn("nothing to do"),
        text_turn("Still your move, then."),
        ctx=ctx,
    )
    client.post("/api/command", json={"text": "I've had enough of this"})
    assert ctx.pending is not None and ctx.pending.name == "resign"

    body = client.post("/api/command", json={"text": "go on then"}).json()

    assert not ctx.session.is_game_over(), "a truncation may not end a game"
    assert ctx.pending is None, "and the op it could not answer for is gone"
    assert body["tool_results"] == []
    assert body["commentary"] == "Still your move, then."


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
