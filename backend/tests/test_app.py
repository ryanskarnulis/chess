"""App-assembly entrypoint: wire a real ToolContext + brain into one app.

`build_app` is where the pieces finally meet — session, settings, optional
engine, and a `Brain` all share one context. This is also where live prompt
settings land: `set_verbosity`/`set_hints_mode` record onto `ctx.settings`,
and the brain resolves its system prompt from those settings per command, so
a change takes effect on the next command.

Exercised only through a `ScriptedProvider` that records the `chat()` requests
and returns scripted `ChatResult`s — never a live LLM.
"""

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
from chessapp.personality import system_prompt_for
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


def test_build_app_wires_live_verbosity_switching():
    # "Talk less": a set_verbosity command must change the system prompt the
    # brain gets on the *next* command — end to end, through the assembled app
    # (settings mutated by the tool, read by the brain), so voice commands
    # like "talk more/less" are real.
    fake = ScriptedProvider(
        # cmd 1, phase one: the verbosity switch.
        tool_calls_turn(("set_verbosity", {"verbosity": "low"})),
        # cmd 1, phase two (react) + everything after: plain text, no tools.
        text_turn("ok"),
    )
    client = TestClient(build_app(model="gemma", provider=fake))

    client.post("/api/command", json={"text": "talk less"})
    client.post("/api/command", json={"text": "hello"})

    # Before the switch the first command used the plain prompt...
    assert fake.calls[0]["messages"][0]["content"] == system_prompt_for()
    # ...and the last command, after the switch, gets the low-verbosity layer.
    assert fake.calls[-1]["messages"][0]["content"] == system_prompt_for(
        verbosity="low"
    )


def test_build_app_wires_live_hints_switching():
    # "Give me hints": set_hints_mode must change the system prompt the brain
    # gets on the next command, same live-settings seam as verbosity.
    fake = ScriptedProvider(
        tool_calls_turn(("set_hints_mode", {"enabled": True})),
        text_turn("ok"),
    )
    client = TestClient(build_app(model="gemma", provider=fake))

    client.post("/api/command", json={"text": "give me hints"})
    client.post("/api/command", json={"text": "hello"})

    assert fake.calls[0]["messages"][0]["content"] == system_prompt_for()
    assert fake.calls[-1]["messages"][0]["content"] == system_prompt_for(
        hints_mode=True
    )


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


def test_uvicorn_has_a_websocket_protocol_available():
    # /ws works in pytest through the in-process TestClient, which never
    # touches uvicorn's protocol layer — so CI stayed green while the Docker
    # image served 404 on every WebSocket handshake (uvicorn without a ws
    # library installed). Pin that the environment the app actually runs
    # under can negotiate WebSockets.
    from uvicorn.protocols.websockets.auto import AutoWebSocketsProtocol

    assert AutoWebSocketsProtocol is not None
