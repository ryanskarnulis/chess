"""A destructive question is answered only where it was asked (#281).

The gate arms `new_game`/`resign`/`claim_draw` for a yes, and until this slice
the armed op was bound to the *board* alone. So any later caller of the shared
pipeline whose text read as an affirmation ran it: "I resign" in one delegate
conversation and a "yes" posted to another ended the game, and so did a "yes"
typed into the web panel — the second thread's stored history had never held
the question, and the panel had never seen it either. No board version could
catch that, because nothing has to move between the question and the wrong
thread's answer.

An op now carries the origin the gate armed it for (`tools.PendingOp.origin`),
and the three origins are the panel (the player's own screen: `/api/command`,
the board buttons and the confirm dialog, one origin on purpose), one per
delegate conversation, and the standalone MCP server's own context. The rule
for a foreign answer is the one the pipeline already had for every other
utterance: **the newest interaction wins.** A command from any origin disarms
whatever is pending on its way in, so a "yes" from elsewhere answers nothing,
is never even shown to the free-text reader, and travels on as an ordinary
utterance — and the origin that *was* asked no longer has a question to answer
either, because the intervening command dropped it.

Tested at the pipeline boundary with the repo's doubles: a `ScriptedBrain` and
no engine, so nothing here touches a live model. Verbosity is `low` throughout,
which makes a confirmed op the app's own canned line and keeps the model out of
the half of the turn under test. The registry's half of this lives in
`test_tools.py`, the MCP surface's in `test_mcp_server.py`.
"""

import json

import pytest
from fastapi.testclient import TestClient

from chessapp.agent_api import reset_rate_limit
from chessapp.brain import AgentResponse
from chessapp.game import GameSession
from chessapp.tools import ToolContext
from chessapp.trace import JsonlTracer
from fakes import ScriptedBrain, scripted_app


@pytest.fixture(autouse=True)
def _reset_rate_limit():
    """The delegate limiter is module-global; one test's posts must not count
    against another's."""
    reset_rate_limit()
    yield
    reset_rate_limit()


# `fastparse.parse_resign` settles this one, so the question is armed through
# the app's own road at zero model calls: dispatched through the registry,
# refused by the gate, armed for whoever said it.
RESIGN = "I resign"


def make_client(*responses: AgentResponse, tracer=None):
    """A game worth losing, on an app carrying both surfaces.

    Four plies of the player's own (no engine, so every move on the board is
    theirs), which is the investment the gate guards — without it a resignation
    simply runs and there is no question to misdirect.
    """
    ctx = ToolContext(session=GameSession())
    for san in ("e4", "e5", "Nf3", "Nc6"):
        assert ctx.session.submit_move(san).legal
    ctx.settings.verbosity = "low"
    app, brain = scripted_app(ctx, brain=ScriptedBrain(*responses), tracer=tracer)
    return TestClient(app), brain, ctx


# The two delegate helpers `test_agent_api.py` uses, copied rather than imported
# across test modules.


def new_conversation(client, **body):
    return client.post("/api/agent/conversations", json=body).json()["id"]


def send(client, conversation_id, content, **kwargs):
    return client.post(
        f"/api/agent/conversations/{conversation_id}/messages",
        json={"content": content},
        **kwargs,
    )


def arm_in(client, conversation_id, ctx):
    """Arm a resignation inside one delegate conversation, and prove it."""
    response = send(client, conversation_id, RESIGN)
    assert response.status_code == 200, response.text
    assert ctx.pending is not None and ctx.pending.name == "resign"
    assert not ctx.session.is_game_over(), "the ask must not end the game"
    return response


# --- a yes from another conversation ------------------------------------------


def test_a_yes_in_another_conversation_runs_nothing():
    """The issue's own reproduction: two threads, one board. B was never asked."""
    client, brain, ctx = make_client(AgentResponse(text="yes to what?"))
    a, b = new_conversation(client), new_conversation(client)
    arm_in(client, a, ctx)

    answer = send(client, b, "yes")

    assert answer.status_code == 200, answer.text
    assert not ctx.session.is_game_over(), (
        "B's yes ended a game B was never asked about"
    )
    assert ctx.session.move_history() == ["e4", "e5", "Nf3", "Nc6"]


def test_a_foreign_yes_is_an_utterance_like_any_other():
    """Not an answer — so not swallowed either. It goes down the ordinary road
    as the fresh intent it is, exactly as a "yes" with nothing armed does."""
    client, brain, ctx = make_client(AgentResponse(text="yes to what?"))
    a, b = new_conversation(client), new_conversation(client)
    arm_in(client, a, ctx)

    answer = send(client, b, "yes").json()

    assert answer["assistant_message"]["content"] == "yes to what?"
    assert brain.calls[-1][1] == "yes", "it reached the brain as a plain utterance"


def test_a_foreign_free_text_answer_is_never_shown_to_the_reader():
    """The sharp version: "just do it" is one no parser settles, so on the
    origin that *was* asked it costs a `read_answer` round trip and can confirm.
    Asked of a thread that holds no question, the reader is not consulted at
    all — there is nothing for it to judge the words against."""
    client, brain, ctx = make_client(AgentResponse(text="do what?"))
    a, b = new_conversation(client), new_conversation(client)
    arm_in(client, a, ctx)

    send(client, b, "just do it")

    assert brain.answer_calls == [], "the reader judged an answer to nobody's question"
    assert not ctx.session.is_game_over()


def test_a_conversation_answers_its_own_question():
    """The other half, and the one that matters most: nothing was broken for the
    thread that was actually asked."""
    client, brain, ctx = make_client()
    a = new_conversation(client)
    arm_in(client, a, ctx)

    answered = send(client, a, "yes")

    assert answered.status_code == 200, answered.text
    assert ctx.session.is_game_over(), "the thread that was asked said yes"
    assert ctx.session.outcome().termination == "resignation"
    assert ctx.pending is None, "the op is spent"


def test_a_foreign_command_in_between_drops_the_question():
    """The newest interaction wins, which is the rule every command already
    followed: an unrelated turn from any origin disarms on its way in. The cost
    is deliberate and is the whole of the trade — the conversation that was
    asked has to ask again, rather than the question sitting armed across other
    surfaces' commands where the gate would refuse a thread that never asked."""
    client, brain, ctx = make_client(
        AgentResponse(text="nice board"), AgentResponse(text="yes to what?")
    )
    a = new_conversation(client)
    arm_in(client, a, ctx)

    client.post("/api/command", json={"text": "how does it look?"})
    assert ctx.pending is None, "the panel's turn disarmed it on the way in"

    answered = send(client, a, "yes")

    assert answered.status_code == 200, answered.text
    assert not ctx.session.is_game_over(), "the question was gone before the yes"
    assert brain.answer_calls == [], "and no reader was asked to find one"


def test_board_version_staleness_still_applies_within_one_origin():
    """The older half of the same rule, unchanged: the right conversation's yes
    about the wrong board answers nothing either."""
    client, brain, ctx = make_client(AgentResponse(text="yes to what?"))
    a = new_conversation(client)
    arm_in(client, a, ctx)

    assert ctx.session.submit_move("Bb5").legal  # another client moves

    answered = send(client, a, "yes")

    assert answered.status_code == 200, answered.text
    assert not ctx.session.is_game_over()
    assert ctx.pending is None


# --- the panel and its buttons are one origin ---------------------------------


def test_the_panel_cannot_answer_a_delegate_question():
    """The variant that needs no second conversation at all: the human's own
    screen saying yes to a question a delegate thread was asked."""
    client, brain, ctx = make_client(AgentResponse(text="yes to what?"))
    a = new_conversation(client)
    arm_in(client, a, ctx)

    answered = client.post("/api/command", json={"text": "yes"})

    assert answered.status_code == 200, answered.text
    assert not ctx.session.is_game_over(), "the panel answered someone else's question"
    assert answered.json()["commentary"] == "yes to what?"


def test_the_button_will_not_confirm_a_delegate_armed_op():
    """A click with nothing of its own to confirm is not a new command: it is
    the same 409 a click on a stale question gets, and the delegate's question
    is left standing for the thread that was asked."""
    client, brain, ctx = make_client()
    a = new_conversation(client)
    arm_in(client, a, ctx)

    clicked = client.post("/api/game/confirm", json={"confirm": True})

    assert clicked.status_code == 409
    assert clicked.json()["detail"] == "nothing to confirm"
    assert not ctx.session.is_game_over()
    assert ctx.pending is not None, "the delegate's question is still standing"

    assert send(client, a, "yes").status_code == 200
    assert ctx.session.is_game_over(), "and the thread that was asked can still answer"


def test_the_button_still_confirms_a_panel_armed_op():
    """The hand-off that must keep working: the panel's free text and the board
    dialog are one person at one screen."""
    client, brain, ctx = make_client()
    client.post("/api/command", json={"text": RESIGN})
    assert ctx.pending is not None and ctx.pending.name == "resign"

    clicked = client.post("/api/game/confirm", json={"confirm": True})

    assert clicked.status_code == 200, clicked.text
    assert clicked.json()["confirmed"] is True
    assert ctx.session.is_game_over()


def test_the_trace_names_the_conversation_a_turn_came_from(tmp_path):
    """One line in the record, and it is the one a bug of this shape is read
    off: two turns that both said "yes" are told apart by nothing else."""
    path = tmp_path / "turns.jsonl"
    client, _, ctx = make_client(tracer=JsonlTracer(path))
    a = new_conversation(client)
    arm_in(client, a, ctx)
    client.post("/api/command", json={"text": RESIGN})

    asked, panel = [json.loads(line) for line in path.read_text().splitlines()]

    assert asked["origin"] == f"delegate:{a}"
    assert panel["origin"] == "panel"


def test_a_typed_panel_yes_still_answers_a_button_armed_op():
    """And the reverse hand-off, which is the same origin seen from the other
    side: the button asked, the keyboard answered."""
    client, brain, ctx = make_client()
    asked = client.post("/api/game/resign", json={})
    assert asked.status_code == 409, asked.text
    assert ctx.pending is not None and ctx.pending.name == "resign"

    answered = client.post("/api/command", json={"text": "yes"})

    assert answered.status_code == 200, answered.text
    assert ctx.session.is_game_over()
    assert ctx.pending is None
