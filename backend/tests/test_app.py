"""App-assembly entrypoint: wire a real ToolContext + brain into one app.

`build_app` is where the pieces finally meet — session, settings, optional
engine, and a `Brain` all share one context. This is also where live prompt
settings land: `set_verbosity`/`set_hints_mode` record onto `ctx.settings`,
and the brain resolves its system prompt from those settings per command, so
a change takes effect on the next command.

Exercised only through a `ScriptedProvider` that records the `chat()` requests
and returns scripted `ChatResult`s — never a live LLM.
"""

import pytest
from fastapi.testclient import TestClient

import chessapp.app
from chessapp.app import (
    DEFAULT_LLAMA_BASE_URL,
    DEFAULT_MODEL,
    build_app,
    build_app_from_env,
)
from chessapp.brain import AgentResponse
from chessapp.engine import DEFAULT_TIER
from chessapp.personality import planner_prompt_for, system_prompt_for
from chessapp.tools import BOARD_STATE_TOOLS
from fakes import (
    FakeEngine,
    ScriptedBrain,
    ScriptedProvider,
    text_turn,
    tool_calls_turn,
)


def test_build_app_serves_state_from_a_fresh_game():
    app = build_app(brain=ScriptedBrain(AgentResponse(text="hi")))
    client = TestClient(app)
    state = client.get("/api/state").json()
    assert state["turn"] == "white"
    assert state["history"] == []


def test_build_app_applies_the_default_difficulty_to_the_engine():
    # Stockfish's own default is full strength (Skill Level 20); assembling
    # the app must configure the attached engine to the settings default so
    # the strength the UI reports is the strength that actually plays.
    engine = FakeEngine()
    app = build_app(brain=ScriptedBrain(), engine=engine)
    assert engine.tiers == [DEFAULT_TIER]
    settings = TestClient(app).get("/api/settings").json()
    assert settings["tier"] == DEFAULT_TIER
    assert settings["skill_level"] is None
    assert settings["elo"] is None


def test_the_brain_is_not_offered_the_redundant_read_tools():
    """Every turn hands the brain the board state; the read tools only return
    subsets of it. Offering them buys a wasted round trip out of a 4-iteration
    budget, so assembly narrows what the brain sees — while the registry keeps
    them for callers with no such injection (MCP, the delegate wire)."""
    provider = ScriptedProvider(text_turn("hi"))
    app = build_app(provider=provider)
    TestClient(app).post("/api/command", json={"text": "how am I doing?"})
    offered = {t["function"]["name"] for t in provider.calls[0]["tools"]}
    assert not offered & set(BOARD_STATE_TOOLS)
    assert {"make_move", "undo", "evaluate_position"} <= offered


def test_build_app_runs_a_command_through_the_assembled_pipeline():
    brain = ScriptedBrain(AgentResponse(text="Which knight did you mean?"))
    client = TestClient(build_app(brain=brain))
    body = client.post("/api/command", json={"text": "move the knight"}).json()
    assert body["commentary"] == "Which knight did you mean?"


def test_build_app_serves_the_frontend_when_a_static_dir_is_configured(tmp_path):
    # The compose stack serves the built frontend from the app container:
    # same-origin, so the UI's relative /api + /ws URLs just work.
    (tmp_path / "index.html").write_text("<html><body>chess ui</body></html>")
    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "app.js").write_text("console.log('ui')")

    app = build_app(brain=ScriptedBrain(AgentResponse(text="hi")), static_dir=tmp_path)
    client = TestClient(app)

    root = client.get("/")
    assert root.status_code == 200
    assert "chess ui" in root.text

    asset = client.get("/assets/app.js")
    assert asset.status_code == 200
    assert "console.log" in asset.text

    # The API keeps priority over the static mount.
    assert client.get("/api/state").status_code == 200


def test_static_wasm_assets_get_the_wasm_mime_type(tmp_path):
    # The hands-free VAD ships onnxruntime as WASM under /vad/. Browsers
    # compile it with instantiateStreaming, which hard-fails unless the
    # response is application/wasm — and Python's mimetypes table does not
    # know .wasm on every platform, so the app must register it.
    (tmp_path / "index.html").write_text("<html></html>")
    vad = tmp_path / "vad"
    vad.mkdir()
    (vad / "ort-wasm-simd-threaded.wasm").write_bytes(b"\x00asm")

    app = build_app(brain=ScriptedBrain(AgentResponse(text="hi")), static_dir=tmp_path)
    client = TestClient(app)

    asset = client.get("/vad/ort-wasm-simd-threaded.wasm")
    assert asset.status_code == 200
    assert asset.headers["content-type"] == "application/wasm"


def test_build_app_without_a_static_dir_serves_no_frontend():
    app = build_app(brain=ScriptedBrain(AgentResponse(text="hi")))
    client = TestClient(app)
    assert client.get("/").status_code == 404
    assert client.get("/api/state").status_code == 200


def test_build_app_wires_the_planner_and_narrator_prompts():
    # The assembled shape of the split (docs/planner-narrator.md): the loop's
    # turns run on the compact planner contract and the closing turn runs on the
    # personality — one command, both prompts, from real app assembly.
    fake = ScriptedProvider(
        tool_calls_turn(("make_move", {"move": "e4"})),
        text_turn("played e4"),
        text_turn("e4 it is."),
    )
    client = TestClient(build_app(model="gemma", provider=fake))

    body = client.post("/api/command", json={"text": "play the king's pawn"}).json()

    assert [call["messages"][0]["content"] for call in fake.calls] == [
        planner_prompt_for(),
        planner_prompt_for(),
        system_prompt_for(),
    ]
    assert [call["tools"] is None for call in fake.calls] == [False, False, True]
    assert body["commentary"] == "e4 it is."  # the narrator's words, not the note


def test_build_app_wires_live_verbosity_switching():
    # "Talk less": a set_verbosity command must change the system prompt the
    # brain gets on the *next* command — end to end, through the assembled app
    # (settings mutated by the tool, read by the brain), so voice commands
    # like "talk more/less" are real. Verbosity is the narrator's layer, so it
    # is the closing call of each command that has to move.
    fake = ScriptedProvider(
        # cmd 1, planner turn one: the verbosity switch.
        tool_calls_turn(("set_verbosity", {"verbosity": "low"})),
        # the planner's handoff note, then every call after: plain text.
        text_turn("ok"),
    )
    client = TestClient(build_app(model="gemma", provider=fake))

    client.post("/api/command", json={"text": "talk less"})
    client.post("/api/command", json={"text": "hello"})

    # The command that made the switch planned under the plain planner prompt...
    assert fake.calls[0]["messages"][0]["content"] == planner_prompt_for()
    # ...and the last command's narrator gets the low-verbosity layer.
    assert fake.calls[-1]["messages"][0]["content"] == system_prompt_for(
        verbosity="low"
    )


def test_build_app_wires_live_hints_switching():
    # "Give me hints": set_hints_mode must change *both* prompts on the next
    # command — the narrator's tone layer and the planner's permission to reach
    # for get_best_moves — the same live-settings seam as verbosity.
    fake = ScriptedProvider(
        tool_calls_turn(("set_hints_mode", {"enabled": True})),
        text_turn("ok"),
    )
    client = TestClient(build_app(model="gemma", provider=fake))

    client.post("/api/command", json={"text": "give me hints"})
    client.post("/api/command", json={"text": "hello"})

    assert fake.calls[0]["messages"][0]["content"] == planner_prompt_for()
    # cmd 2: the planner turn, then the narrator, both with hints on.
    assert fake.calls[-2]["messages"][0]["content"] == planner_prompt_for(
        hints_mode=True
    )
    assert fake.calls[-1]["messages"][0]["content"] == system_prompt_for(
        hints_mode=True
    )


def test_build_app_gates_get_best_moves_on_hints_mode():
    # Audit 11, code-owned: hints off is a capability restriction, not a prompt
    # line. With hints off (the default) the brain is not *offered*
    # `get_best_moves` at all; turning hints on offers it on the very next
    # command — the same live seam the prompts already use. The registry keeps
    # the tool for the trusted callers (MCP, the delegate wire, /api/game/hint).
    fake = ScriptedProvider(
        tool_calls_turn(("set_hints_mode", {"enabled": True})),
        text_turn("ok"),
    )
    client = TestClient(build_app(model="gemma", provider=fake))

    client.post("/api/command", json={"text": "give me hints"})
    client.post("/api/command", json={"text": "what should I play?"})

    def offered(call) -> set[str]:
        return {t["function"]["name"] for t in call["tools"] or []}

    # cmd 1 planned with hints off: no get_best_moves on the table.
    assert "get_best_moves" not in offered(fake.calls[0])
    # cmd 2's planner turn: the switch took effect without a rebuild.
    assert "get_best_moves" in offered(fake.calls[-2])
    # The other analysis reads are ungated — hints only gates move advice.
    assert {"evaluate_position", "analyze_last_move"} <= offered(fake.calls[0])


def test_build_app_from_env_reads_the_planner_temperature(monkeypatch):
    # The sampling experiment's knob: a number in the environment, so a
    # measurement run needs no code change. Unset means the provider's default.
    captured: dict[str, object] = {}

    def fake_create_llama_brain(*, planner_temperature=None, **kwargs):
        captured["planner_temperature"] = planner_temperature
        return ScriptedBrain(AgentResponse(text="hi"))

    monkeypatch.setattr(chessapp.app, "create_llama_brain", fake_create_llama_brain)

    monkeypatch.delenv("CHESSAPP_PLANNER_TEMPERATURE", raising=False)
    build_app_from_env()
    assert captured["planner_temperature"] is None

    monkeypatch.setenv("CHESSAPP_PLANNER_TEMPERATURE", "0.3")
    build_app_from_env()
    assert captured["planner_temperature"] == 0.3


def test_build_app_from_env_honors_the_llamacpp_env_vars(monkeypatch):
    # Workspace agent-standard env names (LLAMACPP_BASE_URL / LLAMACPP_MODEL,
    # not the old CHESSAPP_LLAMA_URL / CHESSAPP_MODEL): build_app_from_env
    # must read these and pass them straight through to the brain factory.
    captured: dict[str, str] = {}

    def fake_create_llama_brain(*, base_url, model, **kwargs):
        captured["base_url"] = base_url
        captured["model"] = model
        return ScriptedBrain(AgentResponse(text="hi"))

    monkeypatch.setattr(chessapp.app, "create_llama_brain", fake_create_llama_brain)
    monkeypatch.setenv("LLAMACPP_BASE_URL", "http://llama-test:9999/v1")
    monkeypatch.setenv("LLAMACPP_MODEL", "test-model")

    build_app_from_env()

    assert captured["base_url"] == "http://llama-test:9999/v1"
    assert captured["model"] == "test-model"


def test_build_app_from_env_defaults_match_the_agent_standard(monkeypatch):
    # No env set: falls back to the workspace-standard llama-swap endpoint
    # and model name (127.0.0.1:8200/v1, gemma-4-12b), not the old
    # single-server localhost:8080 defaults.
    captured: dict[str, str] = {}

    def fake_create_llama_brain(*, base_url, model, **kwargs):
        captured["base_url"] = base_url
        captured["model"] = model
        return ScriptedBrain(AgentResponse(text="hi"))

    monkeypatch.setattr(chessapp.app, "create_llama_brain", fake_create_llama_brain)
    monkeypatch.delenv("LLAMACPP_BASE_URL", raising=False)
    monkeypatch.delenv("LLAMACPP_MODEL", raising=False)

    build_app_from_env()

    assert captured["base_url"] == DEFAULT_LLAMA_BASE_URL == "http://127.0.0.1:8200/v1"
    assert captured["model"] == DEFAULT_MODEL == "gemma-4-12b"


# --- Direct mode is selectable ----------------------------------------------
#
# `brain is None` is the whole of direct mode and every seam downstream of it
# already behaves (`/api/command` 503s, the drag runs the atomic exchange,
# `agent_available` reports the mode). What was missing was a way for a
# deployment to *reach* that state: `build_app(brain=None)` means "construct
# one", so the LLM-off invariant was unreachable from configuration.


def test_build_app_assembles_a_playable_game_with_the_agent_disabled():
    # The binding invariant, end to end: no brain, and a full exchange still
    # plays. `/api/command` is the documented 503 and the mode is visible.
    app = build_app(agent_enabled=False, engine=FakeEngine(reply_uci="e7e5"))
    client = TestClient(app)

    assert client.get("/api/settings").json()["agent_available"] is False
    assert client.post("/api/command", json={"text": "hi"}).status_code == 503

    move = client.post("/api/game/move", json={"move": "e2e4"}).json()
    assert move["legal"] is True
    assert move["san"] == "e4"
    # The engine answered inside the same request: direct mode runs the
    # coordinator's atomic exchange, so a drag is a whole turn.
    assert move["engine_move"]["san"] == "e5"
    assert move["state"]["history"] == ["e4", "e5"]


def test_the_disabled_agent_never_constructs_a_brain(monkeypatch):
    # The point of the switch, and the sharpest assertion available for it:
    # disabled must mean *no provider client is ever built*, not one built and
    # left unused. A factory that detonates proves nothing reached for it.
    def exploding_create_llama_brain(**kwargs):
        raise AssertionError("built a brain with the agent disabled")

    monkeypatch.setattr(
        chessapp.app, "create_llama_brain", exploding_create_llama_brain
    )

    client = TestClient(build_app(agent_enabled=False))

    assert client.get("/api/settings").json()["agent_available"] is False


def test_build_app_rejects_an_injected_brain_with_the_agent_disabled():
    # An incoherent pair, and silently letting one win would make a confusing
    # test someday: say so at assembly.
    with pytest.raises(ValueError, match="agent_enabled"):
        build_app(
            brain=ScriptedBrain(AgentResponse(text="hi")),
            agent_enabled=False,
        )


def test_build_app_from_env_disables_the_agent(monkeypatch):
    # The deployment switch: CHESSAPP_AGENT=off is what makes the LLM-off
    # invariant selectable in the actual container.
    def exploding_create_llama_brain(**kwargs):
        raise AssertionError("built a brain with CHESSAPP_AGENT=off")

    monkeypatch.setattr(
        chessapp.app, "create_llama_brain", exploding_create_llama_brain
    )
    monkeypatch.setenv("CHESSAPP_AGENT", "off")

    client = TestClient(build_app_from_env())

    assert client.get("/api/settings").json()["agent_available"] is False
    assert client.post("/api/command", json={"text": "hi"}).status_code == 503


def test_build_app_from_env_keeps_the_agent_on_by_default(monkeypatch):
    # Agent-on stays the no-config default: a missing variable must never
    # quietly change the operating mode, and neither the endpoint nor the
    # model defaults move (pinned above).
    calls: list[str] = []

    def fake_create_llama_brain(**kwargs):
        calls.append("built")
        return ScriptedBrain(AgentResponse(text="hi"))

    monkeypatch.setattr(chessapp.app, "create_llama_brain", fake_create_llama_brain)

    monkeypatch.delenv("CHESSAPP_AGENT", raising=False)
    assert (
        TestClient(build_app_from_env()).get("/api/settings").json()["agent_available"]
    )
    monkeypatch.setenv("CHESSAPP_AGENT", "on")
    assert (
        TestClient(build_app_from_env()).get("/api/settings").json()["agent_available"]
    )
    assert calls == ["built", "built"]


def test_build_app_from_env_rejects_an_unrecognized_agent_switch(monkeypatch):
    # Refuse to start rather than fall back to "on". The bug this switch
    # exists to fix *is* the app advertising an agent it hasn't got, and a
    # permissive parse reproduces it from a typo — loudly, where someone is
    # watching, is the only safe direction to fail.
    monkeypatch.setenv("CHESSAPP_AGENT", "of")

    with pytest.raises(ValueError, match="CHESSAPP_AGENT"):
        build_app_from_env()


def test_uvicorn_has_a_websocket_protocol_available():
    # /ws works in pytest through the in-process TestClient, which never
    # touches uvicorn's protocol layer — so CI stayed green while the Docker
    # image served 404 on every WebSocket handshake (uvicorn without a ws
    # library installed). Pin that the environment the app actually runs
    # under can negotiate WebSockets.
    from uvicorn.protocols.websockets.auto import AutoWebSocketsProtocol

    assert AutoWebSocketsProtocol is not None
