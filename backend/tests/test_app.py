"""App-assembly entrypoint: wire a real ToolContext + brain into one app.

`build_app` is where the pieces finally meet — session, settings, optional
engine, and a `Brain` all share one context. This is also where live prompt
settings land: `set_verbosity` records onto `ctx.settings`, and the brain
resolves its system prompt from those settings per command, so a change takes
effect on the next command.

Exercised only through a `ScriptedProvider` that records the `chat()` requests
and returns scripted `ChatResult`s — never a live LLM.
"""

import json

import pytest
from fastapi.testclient import TestClient

import chessapp.app
from chessapp.app import (
    DEFAULT_LLAMA_BASE_URL,
    DEFAULT_MODEL,
    build_app,
    build_app_from_env,
    serve,
)
from chessapp.brain import AgentResponse
from chessapp.engine import DEFAULT_TIER
from chessapp.llama_brain import _PLANNER_TEMPERATURE, _REFRESH_LABEL
from chessapp.personality import PLANNER_PROMPT, system_prompt_for
from chessapp.tools import BOARD_STATE_TOOLS
from fakes import (
    FakeEngine,
    ScriptedBrain,
    ScriptedProvider,
    text_turn,
    tool_calls_turn,
)


def offered(call) -> set[str]:
    """The tool names one recorded `chat()` call carried — empty for a call
    offered none (the narrator)."""
    return {t["function"]["name"] for t in call["tools"] or []}


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
        PLANNER_PROMPT,
        PLANNER_PROMPT,
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
    assert fake.calls[0]["messages"][0]["content"] == PLANNER_PROMPT
    # ...and the last command's narrator gets the low-verbosity layer.
    assert fake.calls[-1]["messages"][0]["content"] == system_prompt_for(
        verbosity="low"
    )


def test_build_app_offers_get_best_moves_every_command():
    # Hints are on-request (the mode retired 2026-09-01): the tool that answers
    # an advice ask is on the table from the first command, no setting to flip
    # first — what keeps advice honest is the pipeline's evidence guard, not a
    # capability cut.
    fake = ScriptedProvider(text_turn("ok"))
    client = TestClient(build_app(model="gemma", provider=fake))

    client.post("/api/command", json={"text": "what should I play?"})

    assert "get_best_moves" in offered(fake.calls[0])
    # The rest of the analysis roster rides along, as it always did.
    assert {"evaluate_position", "analyze_last_move"} <= offered(fake.calls[0])


def test_build_app_offers_claim_draw_only_when_a_draw_is_claimable():
    # The offer's one remaining live gate, off board truth: whether a draw can
    # be claimed is something the app knows, so the tool is simply absent until
    # it can be used and the model is never asked to judge it. Every turn with
    # no claim available plans against the unchanged schema.
    fake = ScriptedProvider(text_turn("ok"))
    client = TestClient(build_app(model="gemma", provider=fake))

    client.post("/api/command", json={"text": "how's it looking?"})
    assert "claim_draw" not in offered(fake.calls[0]), "nothing to claim yet"

    # Repeat the position into a claimable threefold draw. Each of these is a
    # fast-path move (no planner turn), so the next planner call is the one that
    # sees the new offer.
    for san in ("Nf3", "Nf6", "Ng1", "Ng8") * 2:
        client.post("/api/command", json={"text": san})
    client.post("/api/command", json={"text": "can we call it a draw?"})

    # -2 is that command's planner turn; -1 is the tool-free narrator.
    assert "claim_draw" in offered(fake.calls[-2])
    assert offered(fake.calls[-1]) == set(), "the narrator still gets no tools"


def test_build_app_from_env_reads_the_planner_temperature(monkeypatch):
    # The sampling experiment's knob: a number in the environment, so a
    # measurement run needs no code change. Unset means the brain's shipped
    # planner temperature (#286), and the provider's own 1.0 is one more
    # number to ask for, not a special case.
    captured: dict[str, object] = {}

    def fake_create_llama_brain(*, planner_temperature=None, **kwargs):
        captured["planner_temperature"] = planner_temperature
        return ScriptedBrain(AgentResponse(text="hi"))

    monkeypatch.setattr(chessapp.app, "create_llama_brain", fake_create_llama_brain)

    monkeypatch.delenv("CHESSAPP_PLANNER_TEMPERATURE", raising=False)
    build_app_from_env()
    assert captured["planner_temperature"] == _PLANNER_TEMPERATURE == 0.3

    monkeypatch.setenv("CHESSAPP_PLANNER_TEMPERATURE", "1.0")
    build_app_from_env()
    assert captured["planner_temperature"] == 1.0


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


# --- shutdown (walkthrough #4) -------------------------------------------------
#
# python-chess drives Stockfish from a *non-daemon* thread it starts per engine,
# so an engine nobody closed keeps the interpreter alive after the server has
# stopped. In the container that was "Application shutdown complete", then the
# full ten-second grace period, then SIGKILL — exit 137, on every deploy, with
# the API already gone for those ten seconds.


def _served(monkeypatch, engine, runner):
    """Run `serve` over a fake engine and a stub server."""
    monkeypatch.setattr(chessapp.app, "_engine_from_env", lambda: engine)
    monkeypatch.setenv("CHESSAPP_AGENT", "off")
    serve(runner)


def test_serve_closes_the_engine_when_the_server_stops(monkeypatch):
    engine = FakeEngine()
    served: list[tuple[str, int]] = []

    _served(
        monkeypatch,
        engine,
        lambda app, host, port: served.append((host, port)) or None,
    )

    assert served, "the runner must actually be handed the assembled app"
    assert engine.closed


def test_serve_closes_the_engine_even_when_serving_raises(monkeypatch):
    """A crash must not be the one path that leaves the process unkillable."""
    engine = FakeEngine()

    def boom(_app, _host, _port):
        raise RuntimeError("bind failed")

    with pytest.raises(RuntimeError, match="bind failed"):
        _served(monkeypatch, engine, boom)

    assert engine.closed


def test_serve_hands_the_runner_the_configured_host_and_port(monkeypatch):
    engine = FakeEngine()
    served: list[tuple[str, int]] = []
    monkeypatch.setenv("CHESSAPP_HOST", "127.0.0.1")
    monkeypatch.setenv("CHESSAPP_PORT", "8123")

    _served(
        monkeypatch,
        engine,
        lambda app, host, port: served.append((host, port)) or None,
    )

    assert served == [("127.0.0.1", 8123)]


def test_serve_runs_the_engine_it_closes(monkeypatch):
    """The app must be built around the same engine `serve` owns — closing a
    different one would end the process and leave the game's engine open."""
    engine = FakeEngine(reply_uci="e7e5")
    apps: list[object] = []

    _served(monkeypatch, engine, lambda app, _host, _port: apps.append(app))

    body = TestClient(apps[0]).post("/api/game/move", json={"move": "e4"}).json()
    assert body["legal"]
    assert body["state"]["history"] == ["e4", "e5"], "the owned engine replied"


# --- the planner's board, mid-command (#282) --------------------------------
#
# Assembly is where the loop meets a real board, so this is where the refresh
# is worth asserting in chess rather than in dicts: the tools that run are the
# tools that ship, `make_move` leaves the turn open the way it does in
# production, and what the planner is handed is read off the recorded request.
# The loop's own share of the policy is pinned in test_llama_brain.py.


def refreshed(call) -> list[dict]:
    """Every board-refresh block one recorded planner call carried, parsed."""
    blocks = []
    for message in call["messages"]:
        if message["role"] != "user":
            continue
        for part in message["content"].split(_REFRESH_LABEL)[1:]:
            blocks.append(json.loads(part.split("\n\n")[0].strip()))
    return blocks


def _seed(client, provider, *moves: str) -> None:
    """Seed a position through the board endpoint, not the agent — the fast
    path plays the whole exchange and leaves the turn closed. It also narrates,
    so the calls it recorded are dropped: what these tests index is the planner
    calls of the one command under test.
    """
    for move in moves:
        assert client.post("/api/game/move", json={"move": move}).status_code == 200
    provider.calls.clear()


def test_a_replacement_move_is_planned_against_the_board_the_undo_left():
    """The bug this exists for. "Undo that and play e4 again" on a board where
    e4 is *on* e4: the opening state block cannot list it, and the planner's
    contract says a move that fits no entry is illegal and must be refused. The
    takeback makes it legal again, and the second decision has to be able to
    see that."""
    provider = ScriptedProvider(
        tool_calls_turn(("undo", {})),
        tool_calls_turn(("make_move", {"move": "e4"})),
        text_turn("took it back and played e4"),
        text_turn("Back to square one, then e4."),
    )
    client = TestClient(build_app(provider=provider, engine=FakeEngine()))
    _seed(client, provider, "e4")

    client.post("/api/command", json={"text": "take that back and play e4 again"})

    opening = json.loads(
        provider.calls[0]["messages"][-1]["content"]
        .split("Board state:\n")[1]
        .split("\n\nCommand:")[0]
    )
    assert "e4" not in opening["legal_moves"], "the list the first decision read"

    blocks = refreshed(provider.calls[1])
    assert len(blocks) == 1, "the planner is told once, after the takeback"
    assert "e4" in blocks[0]["legal_moves"], "the move the stale list could not offer"
    assert blocks[0]["turn"] == "white"
    # The menu and what qualifies it, and nothing about what the turn has
    # already done — see `api._REFRESH_KEYS` for why `history` is left out.
    assert "history" not in blocks[0]
    assert client.get("/api/state").json()["history"][0] == "e4"


def test_a_reset_that_switches_sides_is_planned_against_the_new_board():
    """`new_game(player_color="black")` moves more than the pieces: the engine
    opens, and the moves the player may now make are Black's. Both facts are in
    the block the next decision reads."""
    provider = ScriptedProvider(
        tool_calls_turn(("new_game", {"player_color": "black"})),
        text_turn("reset as black"),
        text_turn("You're black. I've opened."),
    )
    # An engine whose scripted reply is a legal *opening* move, so the side
    # switch really costs a ply — FakeEngine's default answers 1...e5.
    engine = FakeEngine(reply_uci="e2e4")
    client = TestClient(build_app(provider=provider, engine=engine))

    client.post("/api/command", json={"text": "new game, I'll take black"})

    blocks = refreshed(provider.calls[1])
    assert len(blocks) == 1
    assert blocks[0]["player_color"] == "black"
    assert blocks[0]["turn"] == "black"
    assert blocks[0]["legal_moves"], "Black has moves, and they are Black's"


def test_a_resumed_game_is_planned_against_the_restored_board(tmp_path):
    """A resume replaces the session outright, so the block the next decision
    reads has to come from the *new* one — a view closed over the old session
    would describe a game that is gone."""
    provider = ScriptedProvider(
        tool_calls_turn(("save_game", {"name": "checkpoint"})),
        text_turn("saved"),
        text_turn("Saved as checkpoint."),
    )
    engine = FakeEngine()
    client = TestClient(build_app(provider=provider, engine=engine, save_dir=tmp_path))
    _seed(client, provider, "e4")
    client.post("/api/command", json={"text": "save this as checkpoint"})
    # Diverge, so the board the resume restores is visibly a different one: a
    # new game as black, where the engine's opening move is on the board with
    # Black to move. The player has not moved on it, so there is nothing for
    # the resume's confirmation gate to protect and it runs inside the command
    # (#291) — which is the path whose refresh block this pins.
    engine.reply_uci = "d2d4"  # the engine opens, as White
    assert client.post("/api/game/new", json={"color": "black"}).status_code == 409
    assert client.post("/api/game/confirm", json={"confirm": True}).status_code == 200
    diverged = client.get("/api/state").json()["history"]
    assert len(diverged) == 1

    provider.rescript(
        tool_calls_turn(("resume_game", {"name": "checkpoint"})),
        text_turn("resumed"),
        text_turn("Loaded checkpoint."),
    )
    client.post("/api/command", json={"text": "load the game named checkpoint"})

    restored = client.get("/api/state").json()["history"]
    assert restored != diverged, "the resume really replaced the game"
    blocks = refreshed(provider.calls[1])
    assert len(blocks) == 1
    # Read off the restored session, not the one the command started on: the
    # board the command started from had Black to move, the checkpoint has
    # White — and Nf3, which only White can play.
    assert blocks[0]["turn"] == "white"
    assert "Nf3" in blocks[0]["legal_moves"]


def test_a_move_mid_exchange_hands_the_planner_no_board_at_all():
    """`make_move` applies the player's move and stops — the engine's reply is
    the pipeline's to collect. So the board it leaves has the engine to move
    and the engine's legal moves, and handing those to a phase that chooses
    moves is #193 one layer up. Nothing is sent instead."""
    provider = ScriptedProvider(
        tool_calls_turn(("make_move", {"move": "e4"})),
        text_turn("played e4"),
        text_turn("e4 it is."),
    )
    client = TestClient(build_app(provider=provider, engine=FakeEngine()))

    client.post("/api/command", json={"text": "go on and play e4 for me"})

    assert refreshed(provider.calls[1]) == []
    planner = " ".join(str(m["content"]) for m in provider.calls[1]["messages"])
    assert "e5" not in planner, "no menu of the engine's replies"


def test_a_settings_change_alone_appends_no_refresh():
    """The trigger is the board, and deliberately: every settings tool answers
    with its own new value, so the next decision already holds it. A second
    copy would be the ageing duplicate `docs/turn-memory.md` forbids."""
    provider = ScriptedProvider(
        tool_calls_turn(("set_verbosity", {"verbosity": "high"})),
        text_turn("set it"),
        text_turn("Chattier from here."),
    )
    client = TestClient(build_app(provider=provider, engine=FakeEngine()))

    client.post("/api/command", json={"text": "talk more"})

    assert refreshed(provider.calls[1]) == []


def test_a_saved_game_alone_appends_no_refresh(tmp_path):
    """Same rule: a save writes a file, not a position, and `save_game` reports
    the name it wrote."""
    provider = ScriptedProvider(
        tool_calls_turn(("save_game", {"name": "checkpoint"})),
        text_turn("saved"),
        text_turn("Saved as checkpoint."),
    )
    client = TestClient(
        build_app(provider=provider, engine=FakeEngine(), save_dir=tmp_path)
    )

    client.post("/api/command", json={"text": "save this as checkpoint"})

    assert refreshed(provider.calls[1]) == []


def test_the_narrator_still_sees_no_side_to_play_for():
    """The refresh is the planner's alone. The narrator reacts mid-turn and a
    narrator that can see whose move it is announces one (#188/#193), so none
    of it may reach the closing pass."""
    provider = ScriptedProvider(
        tool_calls_turn(("undo", {})),
        tool_calls_turn(("make_move", {"move": "e4"})),
        text_turn("took it back and played e4"),
        text_turn("Back to square one, then e4."),
    )
    client = TestClient(build_app(provider=provider, engine=FakeEngine()))
    _seed(client, provider, "e4")

    client.post("/api/command", json={"text": "take that back and play e4 again"})

    narrator = " ".join(str(m["content"]) for m in provider.calls[-1]["messages"])
    assert _REFRESH_LABEL not in narrator
    # The menu, specifically: a narrator handed the moves it could make treats
    # the reaction beat as a move-selection beat. (`undo`'s own result reports
    # `fen`/`turn` for the settled board it left, which is a separate and
    # deliberate decision — what must never reach here is the list.)
    for leak in ("legal_moves", "captures", "saved_games"):
        assert leak not in narrator, leak
