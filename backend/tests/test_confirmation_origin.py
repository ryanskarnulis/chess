"""A destructive question is answered only where it was asked (#281).

The gate arms `new_game`/`resign`/`claim_draw` for a yes, and until #281 the
armed op was bound to the *board* alone. So any later caller of the shared
pipeline whose text read as an affirmation ran it: "I resign" in one delegate
conversation and a "yes" posted to another ended the game. No board version
could catch that, because nothing has to move between the question and the
wrong surface's answer.

An op now carries the origin the gate armed it for (`tools.PendingOp.origin`).
The app's own origin is the panel (the player's own screen: `/api/command`, the
board buttons and the confirm dialog, one origin on purpose); the standalone
MCP server's context is another. The delegate conversations that used to be
the second in-app origin were retired in #565, so a foreign question is armed
here directly on the context, as the MCP surface would arm it. The rule for a
foreign answer is the one the pipeline already had for every other utterance:
**the newest interaction wins.** A command disarms whatever is pending on its
way in, so a "yes" to someone else's question answers nothing, is never even
shown to the free-text reader, and travels on as an ordinary utterance.

Tested at the pipeline boundary with the repo's doubles: a `ScriptedBrain` and
no engine, so nothing here touches a live model. Verbosity is `low` throughout,
which makes a confirmed op the app's own canned line and keeps the model out of
the half of the turn under test. The registry's half of this lives in
`test_tools.py`, the MCP surface's in `test_mcp_server.py`.
"""

import json

from fastapi.testclient import TestClient

from chessapp.brain import AgentResponse
from chessapp.game import GameSession
from chessapp.tools import MCP_ORIGIN, PendingOp, ToolContext
from chessapp.trace import JsonlTracer
from fakes import ScriptedBrain, scripted_app

# `fastparse.parse_resign` settles this one, so the question is armed through
# the app's own road at zero model calls: dispatched through the registry,
# refused by the gate, armed for whoever said it.
RESIGN = "I resign"


def make_client(*responses: AgentResponse, tracer=None):
    """A game worth losing.

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


def arm_elsewhere(ctx):
    """A resignation the gate armed for another surface, about this board."""
    ctx.pending = PendingOp(
        name="resign", args={}, board_version=ctx.board_version, origin=MCP_ORIGIN
    )
    assert ctx.live_pending(MCP_ORIGIN) is not None


def command(client, text, **body):
    return client.post("/api/command", json={"text": text, **body})


# --- a yes to someone else's question -----------------------------------------


def test_a_yes_to_another_surfaces_question_runs_nothing():
    client, brain, ctx = make_client(AgentResponse(text="yes to what?"))
    arm_elsewhere(ctx)

    answered = command(client, "yes")

    assert answered.status_code == 200, answered.text
    assert not ctx.session.is_game_over(), "the panel answered someone else's question"
    assert ctx.session.move_history() == ["e4", "e5", "Nf3", "Nc6"]


def test_a_foreign_yes_is_an_utterance_like_any_other():
    """Not an answer — so not swallowed either. It goes down the ordinary road
    as the fresh intent it is, exactly as a "yes" with nothing armed does, and
    disarms the other surface's question on its way in."""
    client, brain, ctx = make_client(AgentResponse(text="yes to what?"))
    arm_elsewhere(ctx)

    answered = command(client, "yes").json()

    assert answered["commentary"] == "yes to what?"
    assert brain.calls[-1][1] == "yes", "it reached the brain as a plain utterance"
    assert ctx.pending is None, "the newest interaction wins"


def test_a_foreign_free_text_answer_is_never_shown_to_the_reader():
    """The sharp version: "just do it" is one no parser settles, so on the
    origin that *was* asked it costs a `read_answer` round trip and can confirm.
    Said where no question stands, the reader is not consulted at all — there
    is nothing for it to judge the words against."""
    client, brain, ctx = make_client(AgentResponse(text="do what?"))
    arm_elsewhere(ctx)

    command(client, "just do it")

    assert brain.answer_calls == [], "the reader judged an answer to nobody's question"
    assert not ctx.session.is_game_over()


def test_board_version_staleness_still_applies_within_one_origin():
    """The older half of the same rule: the right surface's yes about the wrong
    board answers nothing either."""
    client, brain, ctx = make_client(AgentResponse(text="yes to what?"))
    command(client, RESIGN)
    assert ctx.pending is not None and ctx.pending.name == "resign"

    assert ctx.session.submit_move("Bb5").legal  # another client moves

    answered = command(client, "yes")

    assert answered.status_code == 200, answered.text
    assert not ctx.session.is_game_over()
    assert ctx.pending is None


# --- the panel and its buttons are one origin ---------------------------------


def test_the_button_will_not_confirm_another_surfaces_op():
    """A click with nothing of its own to confirm is not a new command: it is
    the same 409 a click on a stale question gets, and the other surface's
    question is left standing."""
    client, brain, ctx = make_client()
    arm_elsewhere(ctx)

    clicked = client.post("/api/game/confirm", json={"confirm": True})

    assert clicked.status_code == 409
    assert clicked.json()["detail"] == "nothing to confirm"
    assert not ctx.session.is_game_over()
    assert ctx.pending is not None, "the other surface's question is still standing"


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


def test_a_handoff_label_changes_nothing_about_who_may_answer(tmp_path):
    """`origin: handoff` on `/api/command` is a trace label (#546), not another
    origin: the turn is the panel's, so it answers what the panel was asked and
    cannot answer another surface's question any differently."""
    path = tmp_path / "turns.jsonl"
    client, brain, ctx = make_client(tracer=JsonlTracer(path))
    command(client, RESIGN)
    assert ctx.pending is not None and ctx.pending.name == "resign"

    command(client, "yes", origin="handoff")
    assert ctx.session.is_game_over(), "the panel's own question, answered"

    client2, brain2, ctx2 = make_client(AgentResponse(text="yes to what?"))
    arm_elsewhere(ctx2)
    answered = command(client2, "yes", origin="handoff")
    assert answered.status_code == 200, answered.text
    assert not ctx2.session.is_game_over(), "a handoff answered a foreign question"
    assert answered.json()["commentary"] == "yes to what?"

    asked, confirmed = [json.loads(line) for line in path.read_text().splitlines()]
    assert (asked["origin"], confirmed["origin"]) == ("panel", "handoff")


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
