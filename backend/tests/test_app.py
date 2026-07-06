"""App-assembly entrypoint: wire a real ToolContext + brain into one app.

`build_app` is where the pieces finally meet — session, settings, optional
engine, and a `Brain` all share one context. This is also where live
personality switching lands: `set_personality` records
`ctx.settings.personality`, and the brain resolves its system prompt from
that setting per command, so the change takes effect on the next command.

Exercised only through a fake OpenAI client that records request kwargs and
returns scripted completions — never a live LLM.
"""

from types import SimpleNamespace

from fastapi.testclient import TestClient

from chessapp.app import build_app
from chessapp.brain import AgentResponse
from chessapp.personality import system_prompt_for
from fakes import ScriptedBrain


def _tool_call(name: str, arguments: str, call_id: str = "id0"):
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def _completion(*, content=None, tool_calls=None):
    message = SimpleNamespace(
        role="assistant",
        content=content,
        tool_calls=tool_calls,
        reasoning_content=None,
    )
    return SimpleNamespace(choices=[SimpleNamespace(index=0, message=message)])


class FakeOpenAIClient:
    """Records create() kwargs; returns scripted completions in sequence
    (repeating the last), mirroring llama-server's response shape."""

    def __init__(self, *completions):
        self._completions = list(completions)
        self.calls: list[dict] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        i = min(len(self.calls), len(self._completions) - 1)
        self.calls.append(kwargs)
        return self._completions[i]


def test_build_app_serves_state_from_a_fresh_game():
    app = build_app(brain=ScriptedBrain(AgentResponse(text="hi")))
    client = TestClient(app)
    state = client.get("/api/state").json()
    assert state["turn"] == "white"
    assert state["history"] == []


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


def test_build_app_without_a_static_dir_serves_no_frontend():
    app = build_app(brain=ScriptedBrain(AgentResponse(text="hi")))
    client = TestClient(app)
    assert client.get("/").status_code == 404
    assert client.get("/api/state").status_code == 200


def test_build_app_wires_live_personality_switching():
    # The whole point of item 3: a set_personality command changes which
    # system prompt the brain uses on the *next* command — end to end, through
    # the assembled app (settings mutated by the tool, read by the brain).
    switch = _tool_call("set_personality", '{"personality":"calm_coach"}')
    fake = FakeOpenAIClient(
        # cmd 1, phase one: switch personality to calm_coach.
        _completion(tool_calls=[switch]),
        # cmd 1, phase two (react) + everything after: plain text, no tools.
        _completion(content="ok"),
    )
    app = build_app(model="gemma", openai_client=fake)
    client = TestClient(app)

    client.post("/api/command", json={"text": "be a calm coach"})
    client.post("/api/command", json={"text": "hello"})

    # Before the switch the first command used the default (friendly_rival)...
    assert fake.calls[0]["messages"][0]["content"] == system_prompt_for(
        "friendly_rival"
    )
    # ...and the last command, after the switch, uses calm_coach's prompt.
    assert fake.calls[-1]["messages"][0]["content"] == system_prompt_for("calm_coach")
