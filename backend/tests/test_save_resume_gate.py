"""Resume and overwrite go through the confirmation gate (#291).

`resume_game` throws the game on the board away as surely as a reset does, and
`save_game` over an existing name throws away a save the player made earlier.
Both used to happen without a word. The rules pinned here, at the tool
boundary:

- a resume asks whenever a reset would (the player has moved, the game is not
  over), takes the destructive budget, and never arms for a save that is
  missing or unreadable;
- an overwrite asks whenever the name exists — whatever the board — except for
  `autosave` and for a name this same command wrote;
- neither is a game *ending*, so neither is in `DESTRUCTIVE_TOOLS`.
"""

import json

from chessapp.api import _destructive_confirmation
from chessapp.coordinator import TurnCoordinator
from chessapp.facts import destructive_succeeded
from chessapp.game import GameSession
from chessapp.tools import (
    CONFIRM_QUESTIONS,
    DESTRUCTIVE_TOOLS,
    GATED_TOOLS,
    PANEL_ORIGIN,
    ToolContext,
    _save_path,
    build_registry,
    confirm_pending,
    delegate_origin,
)
from fakes import FakeEngine


def _app(tmp_path, *moves: str):
    ctx = ToolContext(session=GameSession(), save_dir=tmp_path)
    coordinator = TurnCoordinator(ctx)
    registry = build_registry(ctx, coordinator)
    for san in moves:
        assert ctx.session.submit_move(san).legal
    return registry, ctx, coordinator


def _write_save(ctx: ToolContext, name: str, *moves: str) -> None:
    saved = GameSession()
    for san in moves:
        assert saved.submit_move(san).legal
    path = _save_path(ctx, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    saved.save(path)


def _refused_for_confirmation(result: dict) -> bool:
    return result["ok"] is False and result["error"].startswith("confirmation required")


# --- the tables ----------------------------------------------------------------


def test_every_gated_tool_has_its_question_and_none_else_does():
    assert set(CONFIRM_QUESTIONS) == set(GATED_TOOLS)


def test_resume_and_save_are_gated_but_end_no_game():
    for name in ("resume_game", "save_game"):
        assert name in GATED_TOOLS
        assert name not in DESTRUCTIVE_TOOLS
    assert set(DESTRUCTIVE_TOOLS) < set(GATED_TOOLS)


# --- resume --------------------------------------------------------------------


def test_resume_on_a_fresh_board_runs_without_asking(tmp_path):
    registry, ctx, _ = _app(tmp_path)
    _write_save(ctx, "scholars", "e4", "e5")

    result = registry.dispatch("resume_game", {"name": "scholars"})

    assert result["ok"] is True
    assert ctx.pending is None
    assert ctx.session.move_history() == ["e4", "e5"]


def test_resume_on_a_finished_game_runs_without_asking(tmp_path):
    registry, ctx, _ = _app(tmp_path, "f3", "e5", "g4", "Qh4")
    assert ctx.session.is_game_over()
    _write_save(ctx, "scholars", "e4", "e5")

    assert registry.dispatch("resume_game", {"name": "scholars"})["ok"] is True
    assert ctx.pending is None


def test_resume_mid_game_asks_and_the_yes_runs_it(tmp_path):
    registry, ctx, _ = _app(tmp_path, "d4", "d5")
    _write_save(ctx, "scholars", "e4", "e5")

    refused = registry.dispatch("resume_game", {"name": "scholars"})

    assert _refused_for_confirmation(refused)
    assert "resume_game would end the current game" in refused["error"]
    assert ctx.session.move_history() == ["d4", "d5"], "nothing ran on the ask"
    armed = ctx.live_pending(PANEL_ORIGIN)
    assert armed is not None and armed.name == "resume_game"
    assert armed.args == {"name": "scholars"}

    confirmed = confirm_pending(registry, ctx, PANEL_ORIGIN)

    assert confirmed is not None
    name, result = confirmed
    assert name == "resume_game" and result["ok"] is True
    assert ctx.session.move_history() == ["e4", "e5"]


def test_a_missing_save_arms_nothing(tmp_path):
    registry, ctx, _ = _app(tmp_path, "d4", "d5")

    result = registry.dispatch("resume_game", {"name": "nope"})

    assert result["ok"] is False
    assert "no saved game named" in result["error"]
    assert ctx.pending is None


def test_a_corrupt_save_arms_nothing(tmp_path):
    registry, ctx, _ = _app(tmp_path, "d4", "d5")
    path = _save_path(ctx, "broken")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json")

    result = registry.dispatch("resume_game", {"name": "broken"})

    assert result["ok"] is False
    assert ctx.pending is None
    assert ctx.session.move_history() == ["d4", "d5"]


def test_a_resume_spends_the_commands_destructive_budget(tmp_path):
    """Nothing is at stake on a fresh board, so the resume runs — and having
    thrown one board away, the command may not throw away another."""
    registry, ctx, coordinator = _app(tmp_path)
    _write_save(ctx, "scholars", "e4", "e5")
    coordinator.begin_command()

    assert registry.dispatch("resume_game", {"name": "scholars"})["ok"] is True
    after = registry.dispatch("new_game", {})

    assert after["ok"] is False
    assert "destructive operation already ran" in after["error"]
    assert ctx.session.move_history() == ["e4", "e5"]


def test_a_resume_asked_in_a_delegate_thread_is_not_the_panels_to_answer(tmp_path):
    registry, ctx, _ = _app(tmp_path, "d4", "d5")
    _write_save(ctx, "scholars", "e4", "e5")
    ctx.origin = delegate_origin(7)

    assert _refused_for_confirmation(
        registry.dispatch("resume_game", {"name": "scholars"})
    )

    assert confirm_pending(registry, ctx, PANEL_ORIGIN) is None
    assert ctx.session.move_history() == ["d4", "d5"]
    assert confirm_pending(registry, ctx, delegate_origin(7)) is not None
    assert ctx.session.move_history() == ["e4", "e5"]


# --- save ----------------------------------------------------------------------


def test_a_new_name_saves_without_asking(tmp_path):
    registry, ctx, _ = _app(tmp_path, "d4", "d5")

    result = registry.dispatch("save_game", {"name": "fresh"})

    assert result["ok"] is True
    assert result["replaced"] is False
    assert _save_path(ctx, "fresh").exists()


def test_an_existing_name_asks_before_it_is_replaced(tmp_path):
    registry, ctx, _ = _app(tmp_path, "d4", "d5")
    _write_save(ctx, "scholars", "e4", "e5")
    before = _save_path(ctx, "scholars").read_text()

    refused = registry.dispatch("save_game", {"name": "scholars"})

    assert _refused_for_confirmation(refused)
    assert "replace the existing save 'scholars'" in refused["error"]
    assert "end the current game" not in refused["error"]
    assert _save_path(ctx, "scholars").read_text() == before, "file untouched"

    name, result = confirm_pending(registry, ctx, PANEL_ORIGIN)

    assert name == "save_game"
    assert result["ok"] is True and result["replaced"] is True
    saved = json.loads(_save_path(ctx, "scholars").read_text())
    assert GameSession.from_dict(saved).move_history() == ["d4", "d5"]


def test_an_overwrite_asks_even_with_no_game_at_stake(tmp_path):
    """The stake is the file, not the board: a fresh board still asks."""
    registry, ctx, _ = _app(tmp_path)
    _write_save(ctx, "scholars", "e4", "e5")

    assert _refused_for_confirmation(
        registry.dispatch("save_game", {"name": "scholars"})
    )


def test_autosave_is_overwritten_without_asking(tmp_path):
    registry, ctx, _ = _app(tmp_path, "d4", "d5")
    _write_save(ctx, "autosave", "e4", "e5")

    result = registry.dispatch("save_game", {})

    assert result["ok"] is True and result["replaced"] is True
    assert ctx.pending is None


def test_a_name_this_command_wrote_is_rewritten_without_asking(tmp_path):
    """ "Save as checkpoint, undo, save it again" is one ask: the second save
    replaces nothing the player had before they asked (audit finding 8)."""
    registry, ctx, coordinator = _app(tmp_path, "d4", "d5", "Nf3", "Nc6")
    coordinator.begin_command()

    assert registry.dispatch("save_game", {"name": "checkpoint"})["ok"] is True
    assert registry.dispatch("undo", {})["ok"] is True
    again = registry.dispatch("save_game", {"name": "checkpoint"})

    assert again["ok"] is True and again["replaced"] is True
    assert ctx.pending is None

    # The next command's save of that name is a save the player now has.
    coordinator.end_command()
    coordinator.begin_command()
    assert _refused_for_confirmation(
        registry.dispatch("save_game", {"name": "checkpoint"})
    )


def test_a_save_outside_a_command_never_counts_as_this_commands(tmp_path):
    """The MCP server and the board buttons dispatch one call per interaction:
    two separate saves of one name are two asks, and the second one asks."""
    registry, ctx, _ = _app(tmp_path, "d4", "d5")

    assert registry.dispatch("save_game", {"name": "checkpoint"})["ok"] is True
    assert _refused_for_confirmation(
        registry.dispatch("save_game", {"name": "checkpoint"})
    )


def test_an_overwrite_spends_no_destructive_budget(tmp_path):
    registry, ctx, coordinator = _app(tmp_path)
    _write_save(ctx, "scholars", "e4", "e5")
    ctx.pending = None
    coordinator.begin_command()
    assert _refused_for_confirmation(
        registry.dispatch("save_game", {"name": "scholars"})
    )
    confirm_pending(registry, ctx, PANEL_ORIGIN)

    # The board is fresh, so the reset runs: a save threw no game away.
    assert registry.dispatch("new_game", {})["ok"] is True


# --- what the pipeline says about them -------------------------------------------


def test_a_confirmed_resume_is_not_a_game_ending():
    results = [{"name": "resume_game", "result": {"ok": True, "name": "x"}}]
    assert destructive_succeeded(results) is False
    saves = [{"name": "save_game", "result": {"ok": True, "name": "x"}}]
    assert destructive_succeeded(saves) is False


def test_the_low_verbosity_lines_name_what_ran():
    session = GameSession()
    assert (
        _destructive_confirmation("resume_game", {"ok": True, "name": "x"}, session)
        == "Loaded x."
    )
    assert (
        _destructive_confirmation("save_game", {"ok": True, "name": "x"}, session)
        == "Saved over x."
    )


def test_a_resume_that_hands_the_engine_the_move_is_settled(tmp_path):
    ctx = ToolContext(
        session=GameSession(), engine=FakeEngine("e7e5"), save_dir=tmp_path
    )
    registry = build_registry(ctx, TurnCoordinator(ctx))
    _write_save(ctx, "half", "e4")
    for san in ("d4", "d5"):
        assert ctx.session.submit_move(san).legal

    assert _refused_for_confirmation(registry.dispatch("resume_game", {"name": "half"}))
    _, result = confirm_pending(registry, ctx, PANEL_ORIGIN)

    assert result["engine_move"]["san"] == "e5"
    assert ctx.session.move_history() == ["e4", "e5"]
