"""A deep takeback goes through the confirmation gate (#543).

There is no redo, so "go back to the start" used to wipe a game without a word
while `new_game` on the same board asked. The rules pinned here, at the tool
boundary and through the surfaces that answer the question:

- the agent's `undo` runs free up to `UNDO_FREE_PLIES` (two full moves) and
  asks past it, counted over the whole command so repeated calls add up;
- the question says how many of the player's moves would go, and the yes runs
  exactly the takeback that was asked about; a no runs nothing;
- a takeback that could not run arms nothing, and a finished game still asks;
- the board UI's own undo button (`/api/game/undo`) never asks;
- an undo ends no game, so it is not in `DESTRUCTIVE_TOOLS`.
"""

from fastapi.testclient import TestClient

from chessapp.api import _DECLINED_REPLY, _destructive_confirmation, create_app
from chessapp.brain import CANCEL, AgentResponse, ToolCall
from chessapp.coordinator import TurnCoordinator
from chessapp.game import GameSession
from chessapp.tools import (
    DESTRUCTIVE_TOOLS,
    GATED_TOOLS,
    PANEL_ORIGIN,
    UNDO_FREE_PLIES,
    PendingOp,
    ToolContext,
    build_registry,
    confirm_pending,
    confirm_question,
    gate_cost,
)
from fakes import FakeEngine, ScriptedBrain, scripted_app

SIX_EXCHANGES = (
    "e4", "e5", "Nf3", "Nc6", "Bc4", "Bc5", "c3", "Nf6", "d3", "d6", "O-O", "O-O",
)  # fmt: skip


def _game(*sans: str, engine: bool = True):
    """`sans` on the board vs the engine (the player is white), or engine-free."""
    ctx = ToolContext(
        session=GameSession(), engine=FakeEngine(reply_uci="e7e5") if engine else None
    )
    for san in sans:
        assert ctx.session.submit_move(san).legal
    coordinator = TurnCoordinator(ctx)
    return build_registry(ctx, coordinator), ctx, coordinator


def _asked(result: dict) -> bool:
    return result["ok"] is False and result["error"].startswith("confirmation required")


# --- the tables ----------------------------------------------------------------


def test_undo_is_gated_but_ends_no_game():
    assert "undo" in GATED_TOOLS
    assert "undo" not in DESTRUCTIVE_TOOLS
    assert UNDO_FREE_PLIES == 4, "two full moves: the player's and the reply, twice"


# --- the threshold -------------------------------------------------------------


def test_two_full_moves_back_runs_without_asking():
    registry, ctx, _ = _game(*SIX_EXCHANGES)

    result = registry.dispatch("undo", {"plies": UNDO_FREE_PLIES})

    assert result["ok"] is True
    assert ctx.pending is None
    assert ctx.session.move_history() == list(SIX_EXCHANGES[:8])


def test_one_ply_past_two_full_moves_asks_and_nothing_moves():
    registry, ctx, _ = _game(*SIX_EXCHANGES)

    result = registry.dispatch("undo", {"plies": UNDO_FREE_PLIES + 1})

    assert _asked(result)
    assert result["retry"] == "never"
    assert ctx.session.move_history() == list(SIX_EXCHANGES), "nothing ran on the ask"
    armed = ctx.live_pending(PANEL_ORIGIN)
    assert armed is not None and armed.name == "undo"
    assert armed.args == {"plies": UNDO_FREE_PLIES + 1}


def test_back_to_a_move_number_is_free_two_moves_back_and_asks_three():
    """The planner's own form: `before_move`, the move `review_game` names."""
    registry, ctx, _ = _game(*SIX_EXCHANGES)
    assert registry.dispatch("undo", {"before_move": 5})["ok"] is True  # 4 plies
    assert ctx.pending is None

    registry, ctx, _ = _game(*SIX_EXCHANGES)
    assert _asked(registry.dispatch("undo", {"before_move": 4}))  # 6 plies
    assert ctx.session.move_history() == list(SIX_EXCHANGES)


def test_an_ordinary_takeback_never_asks():
    registry, ctx, _ = _game(*SIX_EXCHANGES)

    result = registry.dispatch("undo", {})

    assert result["ok"] is True
    assert result["undone"] == ["O-O", "O-O"]
    assert ctx.pending is None


# --- the question and the answer -----------------------------------------------


def test_the_question_counts_the_players_moves_and_names_where_it_lands():
    registry, ctx, _ = _game(*SIX_EXCHANGES)

    result = registry.dispatch("undo", {"before_move": 4})

    assert (
        "undo would take back 3 of the player's moves, back to move 4"
        in (result["error"])
    )
    armed = ctx.live_pending(PANEL_ORIGIN)
    assert confirm_question(armed, ctx) == (
        "That takes back 3 of your moves, back to move 4. Go back that far?"
    )


def test_a_question_the_board_cannot_count_falls_back_to_the_apps_words():
    """Never reached through the gate, which arms only a takeback that runs;
    the wording still never claims a count it could not make."""
    _, ctx, _ = _game()
    op = PendingOp("undo", {"before_move": 3}, ctx.board_version, PANEL_ORIGIN)

    assert confirm_question(op, ctx) == (
        "That takes back several of your moves. Go back that far?"
    )
    assert gate_cost(op, ctx) == "undo would take back several of the player's moves"
    assert "end the current game" not in gate_cost(op, ctx)


def test_go_back_to_the_start_asks_and_the_yes_runs_it():
    registry, ctx, _ = _game(*SIX_EXCHANGES)

    assert _asked(registry.dispatch("undo", {"before_move": 1}))
    assert confirm_question(ctx.live_pending(PANEL_ORIGIN), ctx) == (
        "That takes back 6 of your moves, back to move 1. Go back that far?"
    )
    confirmed = confirm_pending(registry, ctx, PANEL_ORIGIN)

    assert confirmed is not None
    name, result = confirmed
    assert name == "undo" and result["ok"] is True
    assert len(result["undone"]) == len(SIX_EXCHANGES)
    assert ctx.session.move_history() == []
    assert ctx.pending is None


def test_the_yes_runs_the_takeback_it_was_asked_about():
    registry, ctx, _ = _game(*SIX_EXCHANGES)
    registry.dispatch("undo", {"before_move": 4})

    _, result = confirm_pending(registry, ctx, PANEL_ORIGIN)

    assert result["ok"] is True
    assert ctx.session.move_history() == list(SIX_EXCHANGES[:6])
    assert ctx.session.turn == ctx.session.player_color


def test_a_yes_after_the_board_moved_runs_nothing():
    registry, ctx, _ = _game(*SIX_EXCHANGES)
    registry.dispatch("undo", {"before_move": 1})
    assert ctx.session.submit_move("a3").legal  # a dragged move

    assert confirm_pending(registry, ctx, PANEL_ORIGIN) is None
    assert len(ctx.session.move_history()) == len(SIX_EXCHANGES) + 1


# --- what arms and what does not -----------------------------------------------


def test_a_finished_game_still_asks():
    """The stake is the moves themselves: no redo brings back a won game either."""
    registry, ctx, _ = _game(
        "e4", "e5", "Bc4", "Nc6", "Qh5", "Nf6", "Qxf7#", engine=False
    )
    assert ctx.session.is_game_over()

    assert _asked(registry.dispatch("undo", {"plies": 5}))
    assert ctx.session.is_game_over()


def test_a_takeback_that_cannot_run_arms_nothing():
    registry, ctx, _ = _game(*SIX_EXCHANGES)

    result = registry.dispatch("undo", {"plies": 20})

    assert result["ok"] is False
    assert "only 12 played" in result["error"]
    assert ctx.pending is None


def test_a_resigned_game_arms_nothing():
    registry, ctx, _ = _game(*SIX_EXCHANGES)
    ctx.session.resign()

    result = registry.dispatch("undo", {"before_move": 1})

    assert result["ok"] is False
    assert "resignation" in result["error"]
    assert ctx.pending is None


# --- one command, one count ----------------------------------------------------


def test_takebacks_add_up_inside_a_command():
    """ "Undo my last three moves" is three calls (#394); the third reaches
    past two full moves from where the command found the board, so it asks
    exactly as one call three moves deep does."""
    registry, ctx, coordinator = _game(*SIX_EXCHANGES)
    coordinator.begin_command()
    try:
        assert registry.dispatch("undo", {})["ok"] is True
        assert registry.dispatch("undo", {})["ok"] is True
        third = registry.dispatch("undo", {})
    finally:
        coordinator.end_command()

    assert _asked(third)
    assert ctx.session.move_history() == list(SIX_EXCHANGES[:8])
    assert ctx.live_pending(PANEL_ORIGIN).args == {}

    _, result = confirm_pending(registry, ctx, PANEL_ORIGIN)

    assert result["ok"] is True
    assert ctx.session.move_history() == list(SIX_EXCHANGES[:6])


def test_a_new_command_starts_the_count_again():
    registry, ctx, coordinator = _game(*SIX_EXCHANGES)
    for _ in range(3):
        coordinator.begin_command()
        try:
            assert registry.dispatch("undo", {})["ok"] is True
        finally:
            coordinator.end_command()
    assert ctx.pending is None
    assert ctx.session.move_history() == list(SIX_EXCHANGES[:6])


def test_a_deep_undo_after_another_question_does_not_replace_it():
    registry, ctx, coordinator = _game(*SIX_EXCHANGES)
    coordinator.begin_command()
    try:
        assert _asked(registry.dispatch("new_game", {}))
        result = registry.dispatch("undo", {"before_move": 1})
    finally:
        coordinator.end_command()

    assert result["ok"] is False
    assert result["pending"] == "new_game"
    assert ctx.pending.name == "new_game"
    assert ctx.session.move_history() == list(SIX_EXCHANGES)


# --- the surfaces --------------------------------------------------------------


def _spoken(text: str = "you sure?"):
    """Engine-free, so every ply is the player's and `before_move=1` is all 12."""
    ctx = ToolContext(session=GameSession())
    for san in SIX_EXCHANGES:
        assert ctx.session.submit_move(san).legal
    return ctx, AgentResponse(
        text=text, tool_calls=(ToolCall(name="undo", args={"before_move": 1}),)
    )


def test_spoken_go_back_to_the_start_asks_then_yes_runs_it():
    ctx, response = _spoken()
    app, brain = scripted_app(ctx, response)
    client = TestClient(app)

    asked = client.post("/api/command", json={"text": "go back to the start"}).json()

    assert ctx.session.move_history() == list(SIX_EXCHANGES), "nothing ran on the ask"
    assert asked["tool_results"][0]["result"]["ok"] is False
    assert ctx.pending is not None and ctx.pending.name == "undo"
    calls_after_ask = len(brain.calls)

    body = client.post("/api/command", json={"text": "yes"}).json()

    assert ctx.session.move_history() == []
    assert len(brain.calls) == calls_after_ask, "a confirmation needs no model turn"
    assert [r["name"] for r in body["tool_results"]] == ["undo"]
    assert ctx.pending is None


def test_spoken_no_drops_the_deep_undo():
    ctx, response = _spoken()
    app, _ = scripted_app(ctx, response)
    client = TestClient(app)
    client.post("/api/command", json={"text": "go back to the start"})

    body = client.post("/api/command", json={"text": "no"}).json()

    assert ctx.session.move_history() == list(SIX_EXCHANGES)
    assert ctx.pending is None
    assert body["commentary"] == _DECLINED_REPLY
    assert body["tool_results"] == []


def test_the_reader_is_asked_the_counted_question():
    ctx, response = _spoken()
    brain = ScriptedBrain(response, answers=(CANCEL,))
    app, brain = scripted_app(ctx, brain=brain)
    client = TestClient(app)
    client.post("/api/command", json={"text": "go back to the start"})

    client.post("/api/command", json={"text": "hold on"})

    assert brain.answer_calls == [
        (
            "That takes back 12 of your moves, back to move 1. Go back that far?",
            "hold on",
        )
    ]
    assert ctx.session.move_history() == list(SIX_EXCHANGES)


def test_the_low_verbosity_line_reports_a_takeback_not_a_game_over():
    line = _destructive_confirmation(
        "undo", {"ok": True, "undone": ["O-O", "O-O", "d6"]}, GameSession()
    )
    assert line == "Took back 3 half-moves."


def test_the_board_undo_button_never_asks():
    """A person pressing the button is deliberate: only the agent's tool asks."""
    ctx = ToolContext(session=GameSession())
    for san in SIX_EXCHANGES:
        assert ctx.session.submit_move(san).legal
    client = TestClient(create_app(ctx))

    response = client.post("/api/game/undo", json={"plies": 12})

    assert response.status_code == 200
    assert ctx.session.move_history() == []
    assert ctx.pending is None
