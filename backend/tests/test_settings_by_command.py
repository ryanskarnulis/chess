"""Settings by natural speech — acceptance at the tool boundary.

A spoken command is transcribed and enters `/api/command` exactly like typed
text, so "settings by voice" is settings-by-command: the brain maps the
utterance to a `set_*` tool call and the validated registry mutates the one
shared `Settings`. Per the project's testing rule the brain is a scripted
fake — we pin that every setting is reachable through the single pipeline
and lands in the same truth `GET /api/settings` serves, never what a live
LLM would say.
"""

from fastapi.testclient import TestClient

from chessapp.brain import AgentResponse, ToolCall
from chessapp.game import GameSession
from chessapp.tools import ToolContext
from fakes import scripted_app


class RecordingEngine:
    """Engine double that records difficulty configuration."""

    def __init__(self):
        self.skill_levels: list[int] = []
        self.elos: list[int] = []
        self.tiers: list[str] = []

    def set_skill_level(self, level: int) -> None:
        self.skill_levels.append(level)

    def set_elo(self, elo: int) -> None:
        self.elos.append(elo)

    def set_tier(self, tier: str) -> None:
        self.tiers.append(tier)


def make_client_for(ctx: ToolContext, *responses: AgentResponse):
    app, brain = scripted_app(ctx, *responses)
    return TestClient(app), brain


def make_client(*tool_calls: ToolCall, engine=None):
    ctx = ToolContext(session=GameSession(), engine=engine)
    client, _ = make_client_for(ctx, AgentResponse(text="done", tool_calls=tool_calls))
    return client, ctx


def command(client, text):
    response = client.post("/api/command", json={"text": text})
    assert response.status_code == 200
    return response.json()


def test_difficulty_by_speech_reaches_the_live_engine():
    engine = RecordingEngine()
    client, ctx = make_client(
        ToolCall(name="set_difficulty", args={"skill_level": 15}), engine=engine
    )
    body = command(client, "make it harder, skill fifteen")
    assert body["tool_results"][0]["result"]["ok"] is True
    assert ctx.settings.skill_level == 15
    assert engine.skill_levels == [15]
    assert client.get("/api/settings").json()["skill_level"] == 15


def test_difficulty_by_elo_by_speech():
    engine = RecordingEngine()
    client, ctx = make_client(
        ToolCall(name="set_difficulty", args={"elo": 1500}), engine=engine
    )
    command(client, "play at about 1500 strength")
    assert ctx.settings.elo == 1500
    assert ctx.settings.skill_level is None
    assert engine.elos == [1500]


def test_difficulty_by_speech_lands_in_the_command_response():
    # The UI's difficulty selector reflects only what the server confirms —
    # /api/settings on load, then each mutation response after. An agent-side
    # change must ride the command response the same way voice_output already
    # does (`speak`), or the selector keeps showing a strength the engine is
    # no longer playing.
    engine = RecordingEngine()
    client, _ = make_client(
        ToolCall(name="set_difficulty", args={"tier": "advanced"}), engine=engine
    )
    body = command(client, "make it harder")
    assert body["tier"] == "advanced"
    assert engine.tiers == ["advanced"]


def test_out_of_tier_difficulty_reports_a_null_tier():
    # A raw elo unsets the named tier; the response says so, rather than
    # leaving a stale tier highlighted in the selector.
    engine = RecordingEngine()
    client, _ = make_client(
        ToolCall(name="set_difficulty", args={"elo": 1500}), engine=engine
    )
    body = command(client, "play at about 1500 strength")
    assert body["tier"] is None


def test_verbosity_by_speech():
    client, ctx = make_client(ToolCall(name="set_verbosity", args={"verbosity": "low"}))
    body = command(client, "talk less")
    assert body["tool_results"][0]["result"]["ok"] is True
    assert ctx.settings.verbosity == "low"
    assert client.get("/api/settings").json()["verbosity"] == "low"


def test_hints_mode_by_speech():
    client, ctx = make_client(ToolCall(name="set_hints_mode", args={"enabled": True}))
    command(client, "give me hints")
    assert ctx.settings.hints_mode is True
    assert client.get("/api/settings").json()["hints_mode"] is True


def test_voice_output_by_speech_flips_the_speak_flag():
    client, ctx = make_client(ToolCall(name="set_voice_output", args={"enabled": True}))
    body = command(client, "turn on voice")
    assert ctx.settings.voice_output is True
    # The same response already tells the client to start voicing replies.
    assert body["speak"] is True


def test_bad_setting_from_the_brain_is_data_not_an_error():
    # An out-of-enum verbosity is an error *result* the loop reads and gives up
    # on in words — never an HTTP failure or a corrupted setting.
    ctx = ToolContext(session=GameSession())
    client, _ = make_client_for(
        ctx,
        AgentResponse(
            text="I can't talk like that.",
            tool_calls=(
                ToolCall(name="set_verbosity", args={"verbosity": "shouting"}),
            ),
        ),
    )
    body = command(client, "start shouting")
    assert body["tool_results"][0]["result"]["ok"] is False
    assert ctx.settings.verbosity == "normal"


def test_several_settings_in_one_utterance():
    # "Talk less and give me hints" — one utterance, two tool calls, both
    # land.
    client, ctx = make_client(
        ToolCall(name="set_verbosity", args={"verbosity": "low"}),
        ToolCall(name="set_hints_mode", args={"enabled": True}),
    )
    command(client, "talk less and give me hints")
    assert ctx.settings.verbosity == "low"
    assert ctx.settings.hints_mode is True
