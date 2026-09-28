"""The difficulty a conversation opened at, shown to the planner (#338).

"Put the difficulty back to what it was when we started" names a value only
the conversation remembers, and the planner reads the words spoken, never an
old tool result. So the app records the difficulty when a conversation's first
command arrives and shows it once it differs from the current one.
"""

from chessapp.conversation import Transcript
from chessapp.game import GameSession
from chessapp.tools import PANEL_ORIGIN, ToolContext
from fakes import FakeEngine, text_turn, tool_calls_turn
from test_clarification_state import _opening_states
from test_closing_pass import make_client

_KEY = "difficulty_when_this_conversation_began"


def _at_beginner() -> ToolContext:
    ctx = ToolContext(session=GameSession(), engine=FakeEngine())
    ctx.settings.tier = "beginner"
    return ctx


def test_the_start_is_shown_once_the_difficulty_has_moved():
    client, provider, ctx = make_client(
        tool_calls_turn(("set_difficulty", {"step": "harder"})),
        text_turn("stepped up"),
        text_turn("Casual now."),
        text_turn("nothing to do"),
        text_turn("Where we started was beginner."),
        ctx=_at_beginner(),
    )

    client.post("/api/command", json={"text": "make the engine harder"})
    first = _opening_states(provider)[0]
    assert _KEY not in first["settings"], "unchanged: the opening view is too"
    assert ctx.settings.tier == "casual"

    provider.calls.clear()
    client.post("/api/command", json={"text": "what did we start on?"})
    second = _opening_states(provider)[0]
    assert second["settings"]["difficulty"] == {"tier": "casual"}
    assert second["settings"][_KEY] == {"tier": "beginner"}


def test_a_conversation_already_under_way_has_no_start_to_report():
    """A transcript this process did not see open (a restart, a resumed save)
    records nothing: its first command here is not where it started."""
    ctx = _at_beginner()
    ctx.transcript.record("hi", "Hey.")
    client, provider, ctx = make_client(
        tool_calls_turn(("set_difficulty", {"step": "harder"})),
        text_turn("stepped up"),
        text_turn("Casual now."),
        text_turn("nothing to do"),
        text_turn("No idea."),
        ctx=ctx,
    )

    client.post("/api/command", json={"text": "make the engine harder"})
    client.post("/api/command", json={"text": "what did we start on?"})

    for state in _opening_states(provider):
        assert _KEY not in state["settings"]


def test_a_resumed_game_forgets_the_panels_start():
    ctx = _at_beginner()
    ctx.opening_difficulty[PANEL_ORIGIN] = {"tier": "beginner"}
    ctx.replace_session(GameSession(), Transcript())
    assert PANEL_ORIGIN not in ctx.opening_difficulty
