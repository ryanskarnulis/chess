"""Settings survive a restart.

`Settings` lives in memory, so without a write-through the difficulty a
player dialed in (by voice or the options sheet) silently reverts to the
default whenever the backend restarts. Persistence hangs off
`ToolContext.save_dir` — the same volume the save files ride — written
through on every mutation at the dataclass's own `__setattr__` chokepoint
(so no `set_*` site can forget), and restored at context construction so
assembly applies the loaded difficulty to the engine like any other startup
value. Best-effort by design: a missing, corrupt, or invalid file must never
stop assembly, and a failed write must never fail the mutation.
"""

import json

from fastapi.testclient import TestClient

from chessapp.app import build_app
from chessapp.game import GameSession
from chessapp.tools import SETTINGS_FILENAME, Settings, ToolContext
from fakes import FakeEngine, ScriptedBrain


def settings_file(tmp_path):
    return tmp_path / SETTINGS_FILENAME


def test_mutating_a_setting_writes_the_settings_file(tmp_path):
    ctx = ToolContext(session=GameSession(), save_dir=tmp_path)
    ctx.settings.tier = "advanced"
    data = json.loads(settings_file(tmp_path).read_text())
    assert data["tier"] == "advanced"


def test_settings_are_restored_by_a_new_context(tmp_path):
    ctx = ToolContext(session=GameSession(), save_dir=tmp_path)
    ctx.settings.elo = 1500
    ctx.settings.tier = None
    ctx.settings.verbosity = "low"
    reborn = ToolContext(session=GameSession(), save_dir=tmp_path)
    assert reborn.settings.elo == 1500
    assert reborn.settings.tier is None
    assert reborn.settings.skill_level is None
    assert reborn.settings.verbosity == "low"


def test_difficulty_survives_a_restart_end_to_end(tmp_path):
    first = TestClient(
        build_app(engine=FakeEngine(), save_dir=tmp_path, brain=ScriptedBrain())
    )
    response = first.post("/api/game/difficulty", json={"tier": "advanced"})
    assert response.status_code == 200

    engine = FakeEngine()
    second = TestClient(
        build_app(engine=engine, save_dir=tmp_path, brain=ScriptedBrain())
    )
    assert second.get("/api/settings").json()["tier"] == "advanced"
    # Assembly applied the restored tier to the fresh engine — the strength
    # actually played, not just the one reported.
    assert engine.tiers == ["advanced"]


def test_missing_file_means_defaults(tmp_path):
    ctx = ToolContext(session=GameSession(), save_dir=tmp_path)
    assert ctx.settings == Settings()
    assert not settings_file(tmp_path).exists()


def test_corrupt_file_is_ignored_then_overwritten(tmp_path):
    settings_file(tmp_path).write_text("not json {{{")
    ctx = ToolContext(session=GameSession(), save_dir=tmp_path)
    assert ctx.settings == Settings()
    # The next mutation replaces the corrupt file with a clean snapshot.
    ctx.settings.verbosity = "high"
    data = json.loads(settings_file(tmp_path).read_text())
    assert data["verbosity"] == "high"


def test_invalid_values_in_the_file_are_ignored(tmp_path):
    settings_file(tmp_path).write_text(
        json.dumps(
            {
                "tier": "banana",
                "skill_level": 999,
                "elo": "high",
                "verbosity": "shouting",
                "hints_mode": "yes",
                "unknown_key": True,
            }
        )
    )
    ctx = ToolContext(session=GameSession(), save_dir=tmp_path)
    assert ctx.settings == Settings()


def test_no_save_dir_means_no_persistence():
    ctx = ToolContext(session=GameSession())
    # Still a plain in-memory mutation — nothing to write, nothing to fail.
    ctx.settings.tier = "advanced"
    assert ctx.settings.tier == "advanced"
