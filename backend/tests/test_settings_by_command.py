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

from chessapp.api import create_app
from chessapp.brain import AgentResponse, ToolCall
from chessapp.game import GameSession
from chessapp.tools import ToolContext
from fakes import ScriptedBrain


class RecordingEngine:
    """Engine double that records difficulty configuration."""

    def __init__(self):
        self.skill_levels: list[int] = []
        self.elos: list[int] = []

    def set_skill_level(self, level: int) -> None:
        self.skill_levels.append(level)

    def set_elo(self, elo: int) -> None:
        self.elos.append(elo)


def make_client(*tool_calls: ToolCall, engine=None):
    ctx = ToolContext(session=GameSession(), engine=engine)
    brain = ScriptedBrain(
        AgentResponse(text="on it", tool_calls=tool_calls),
        reactions=("done",),
    )
    return TestClient(create_app(ctx, brain=brain)), ctx


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


def test_personality_by_speech():
    client, ctx = make_client(
        ToolCall(name="set_personality", args={"personality": "villain"})
    )
    body = command(client, "be the villain")
    assert body["tool_results"][0]["result"]["ok"] is True
    assert ctx.settings.personality == "villain"
    assert client.get("/api/settings").json()["personality"] == "villain"


def test_verbosity_by_speech():
    client, ctx = make_client(ToolCall(name="set_verbosity", args={"verbosity": "low"}))
    command(client, "talk less")
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
    # An out-of-enum personality is an error *result* the agent can react to,
    # never an HTTP failure or a corrupted setting. The failure earns a retry
    # round; here the brain concedes in words.
    ctx = ToolContext(session=GameSession())
    brain = ScriptedBrain(
        AgentResponse(
            text="on it",
            tool_calls=(
                ToolCall(name="set_personality", args={"personality": "chaos_gremlin"}),
            ),
        ),
        AgentResponse(text="I don't know that personality."),
    )
    client = TestClient(create_app(ctx, brain=brain))
    body = command(client, "be a chaos gremlin")
    assert body["tool_results"][0]["result"]["ok"] is False
    assert ctx.settings.personality == "friendly_rival"


def test_several_settings_in_one_utterance():
    # "Be the grandmaster and talk less" — one utterance, two tool calls,
    # both land.
    client, ctx = make_client(
        ToolCall(name="set_personality", args={"personality": "grandmaster"}),
        ToolCall(name="set_verbosity", args={"verbosity": "low"}),
    )
    command(client, "be the grandmaster and talk less")
    assert ctx.settings.personality == "grandmaster"
    assert ctx.settings.verbosity == "low"
