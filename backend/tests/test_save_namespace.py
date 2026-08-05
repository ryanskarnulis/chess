"""Game saves and application metadata occupy distinct namespaces."""

import json

from chessapp.game import GameSession
from chessapp.tools import (
    GAME_SAVE_DIRNAME,
    SETTINGS_FILENAME,
    ToolContext,
    build_registry,
    saved_game_names,
)


def test_settings_metadata_is_never_listed_as_a_saved_game(tmp_path):
    ctx = ToolContext(session=GameSession(), save_dir=tmp_path)

    # The metadata file is created lazily by the first real setting mutation.
    ctx.settings.verbosity = "high"

    assert (tmp_path / SETTINGS_FILENAME).exists()
    assert saved_game_names(ctx) == []


def test_game_named_settings_is_distinct_from_application_settings(tmp_path):
    ctx = ToolContext(session=GameSession(), save_dir=tmp_path)
    ctx.settings.verbosity = "low"
    registry = build_registry(ctx)
    registry.dispatch("make_move", {"move": "e4"})

    assert registry.dispatch("save_game", {"name": "settings"})["ok"] is True

    metadata = json.loads((tmp_path / SETTINGS_FILENAME).read_text())
    game = json.loads((tmp_path / GAME_SAVE_DIRNAME / "settings.json").read_text())
    assert metadata["verbosity"] == "low"
    assert game["moves"] == ["e2e4"]
    assert saved_game_names(ctx) == ["settings"]


def test_setting_change_after_saving_settings_cannot_overwrite_the_game(tmp_path):
    ctx = ToolContext(session=GameSession(), save_dir=tmp_path)
    registry = build_registry(ctx)
    registry.dispatch("make_move", {"move": "e4"})
    assert registry.dispatch("save_game", {"name": "settings"})["ok"] is True

    ctx.settings.hints_mode = True

    fresh = ToolContext(session=GameSession(), save_dir=tmp_path)
    resumed = build_registry(fresh).dispatch("resume_game", {"name": "settings"})
    assert resumed["ok"] is True
    assert fresh.session.move_history() == ["e4"]
    assert fresh.settings.hints_mode is True


def test_restart_preserves_settings_and_a_game_with_the_same_name(tmp_path):
    first = ToolContext(session=GameSession(), save_dir=tmp_path)
    first.settings.verbosity = "low"
    first_registry = build_registry(first)
    first_registry.dispatch("make_move", {"move": "d4"})
    assert first_registry.dispatch("save_game", {"name": "settings"})["ok"] is True

    second = ToolContext(session=GameSession(), save_dir=tmp_path)

    assert second.settings.verbosity == "low"
    assert saved_game_names(second) == ["settings"]
    assert (
        build_registry(second).dispatch("resume_game", {"name": "settings"})["ok"]
        is True
    )
    assert second.session.move_history() == ["d4"]


def test_restart_migrates_legacy_top_level_saves_without_orphaning_them(tmp_path):
    legacy = GameSession()
    legacy.submit_move("c4")
    legacy.save(tmp_path / "weekend.json")

    ctx = ToolContext(session=GameSession(), save_dir=tmp_path)

    assert not (tmp_path / "weekend.json").exists()
    assert (tmp_path / GAME_SAVE_DIRNAME / "weekend.json").exists()
    assert saved_game_names(ctx) == ["weekend"]
    assert (
        build_registry(ctx).dispatch("resume_game", {"name": "weekend"})["ok"] is True
    )
    assert ctx.session.move_history() == ["c4"]


def test_restart_recovers_a_legacy_game_that_collided_with_settings(tmp_path):
    legacy = GameSession()
    legacy.submit_move("Nf3")
    legacy.save(tmp_path / SETTINGS_FILENAME)

    ctx = ToolContext(session=GameSession(), save_dir=tmp_path)

    assert ctx.settings.verbosity == "normal"
    assert not (tmp_path / SETTINGS_FILENAME).exists()
    assert (tmp_path / GAME_SAVE_DIRNAME / "settings.json").exists()
    assert saved_game_names(ctx) == ["settings"]
    assert (
        build_registry(ctx).dispatch("resume_game", {"name": "settings"})["ok"] is True
    )
    assert ctx.session.move_history() == ["Nf3"]
