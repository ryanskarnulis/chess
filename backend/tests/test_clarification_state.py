"""The open clarification through the real pipeline (#319).

A question the planner asked with `ask_player` is kept as a record bound to
the conversation it was asked in, the game and the board — and these tests
drive it through every road that can open, keep, answer or end it: the panel
and delegate command pipelines, the fast path, a board drag, the buttons, a
resume. The brain is a real `LlamaBrain` over a scripted provider
(`test_closing_pass.make_client`), so the handoff that opens a question is the
one the shipped loop builds; nothing here reads the model's words.
"""

import json

from chessapp.clarification import (
    ANSWERED,
    ASKED_AGAIN,
    BOARD_CHANGED,
    GAME_CHANGED,
    INVALIDATED,
    OTHER_MOVE,
    SUPERSEDED,
)
from chessapp.conversation import Transcript
from chessapp.game import GameSession
from chessapp.tools import PANEL_ORIGIN, ToolContext, delegate_origin
from fakes import FakeEngine, text_turn, tool_calls_turn
from test_closing_pass import CollectedTurns, make_client

KNIGHT_ASK = "move my kings knight"
ASK = tool_calls_turn(("ask_player", {"candidates": ["Nf3", "Nh3"]}))
QUESTION = text_turn("Nf3 or Nh3?")


def _asked(ctx: ToolContext | None = None):
    """A client whose panel has just been asked "Nf3 or Nh3?"."""
    turns = CollectedTurns()
    client, provider, ctx = make_client(ASK, QUESTION, ctx=ctx, tracer=turns)
    client.post("/api/command", json={"text": KNIGHT_ASK})
    return client, provider, ctx, turns


def _quiet(provider) -> None:
    """Script the next command as one that does nothing: no tool, a reply."""
    provider.rescript(text_turn("the player asked a question"), text_turn("Sure."))


def test_an_ask_opens_a_question_on_its_origin_game_and_board():
    client, _, ctx, turns = _asked()

    question = ctx.clarifications[PANEL_ORIGIN]
    assert question.candidates == ("Nf3", "Nh3")
    assert question.request == KNIGHT_ASK
    assert question.game_id == ctx.session.game_id
    assert question.board_version == ctx.board_version
    (record,) = turns.records
    assert record["clarification"] == {
        "open": None,
        "expired": None,
        "created": question.trace(),
        "closed": None,
    }


def test_an_aside_that_moves_nothing_keeps_it_open():
    """The continuation policy: a question about a board nobody touched is
    still the open decision, however many asides come between."""
    client, provider, ctx, turns = _asked()
    question = ctx.clarifications[PANEL_ORIGIN]

    for aside in ("what difficulty am I on?", "talk less", "nice weather"):
        _quiet(provider)
        client.post("/api/command", json={"text": aside})
        assert ctx.clarifications[PANEL_ORIGIN] is question
        assert turns.records[-1]["clarification"]["open"] == question.id
        assert turns.records[-1]["clarification"]["closed"] is None


def test_a_candidate_played_by_the_planner_answers_it_once():
    client, provider, ctx, turns = _asked()
    question = ctx.clarifications[PANEL_ORIGIN]

    provider.rescript(
        tool_calls_turn(("make_move", {"move": "Nf3"})), text_turn("Hop.")
    )
    client.post("/api/command", json={"text": "the one to f3"})

    assert ctx.session.move_history() == ["Nf3", "e5"]
    assert PANEL_ORIGIN not in ctx.clarifications
    closed = turns.records[-1]["clarification"]["closed"]
    assert (closed["id"], closed["status"], closed["move"]) == (
        question.id,
        ANSWERED,
        "Nf3",
    )
    # Once: the next turn finds nothing open, and nothing expired either.
    _quiet(provider)
    client.post("/api/command", json={"text": "good move?"})
    assert turns.records[-1]["clarification"]["open"] is None
    assert turns.records[-1]["clarification"]["expired"] is None


def test_a_candidate_typed_on_the_fast_path_answers_it():
    client, _, ctx, turns = _asked()
    ctx.settings.verbosity = "low"  # a zero-model fast-path turn

    client.post("/api/command", json={"text": "Nh3"})

    closed = turns.records[-1]["clarification"]["closed"]
    assert (closed["status"], closed["move"]) == (ANSWERED, "Nh3")
    assert PANEL_ORIGIN not in ctx.clarifications


def test_a_different_move_supersedes_it():
    client, _, ctx, turns = _asked()
    ctx.settings.verbosity = "low"

    client.post("/api/command", json={"text": "e4"})

    closed = turns.records[-1]["clarification"]["closed"]
    assert (closed["status"], closed["reason"], closed["move"]) == (
        SUPERSEDED,
        OTHER_MOVE,
        "e4",
    )
    assert PANEL_ORIGIN not in ctx.clarifications


def test_a_dragged_candidate_answers_the_panels_question():
    client, _, ctx, turns = _asked()
    ctx.settings.verbosity = "low"

    response = client.post("/api/game/move", json={"move": "g1f3"})

    assert response.status_code == 200
    closed = turns.records[-1]["clarification"]["closed"]
    assert (closed["status"], closed["move"]) == (ANSWERED, "Nf3")
    assert PANEL_ORIGIN not in ctx.clarifications


def test_asking_again_replaces_it():
    client, provider, ctx, turns = _asked()
    first = ctx.clarifications[PANEL_ORIGIN]

    provider.rescript(
        tool_calls_turn(("ask_player", {"candidates": ["e3", "e4"]})),
        text_turn("One step or two?"),
    )
    client.post("/api/command", json={"text": "push the king pawn"})

    second = ctx.clarifications[PANEL_ORIGIN]
    assert second.candidates == ("e3", "e4")
    assert second.id != first.id
    record = turns.records[-1]["clarification"]
    assert (record["closed"]["id"], record["closed"]["status"]) == (
        first.id,
        SUPERSEDED,
    )
    assert record["closed"]["reason"] == ASKED_AGAIN
    assert record["created"]["id"] == second.id


def test_a_move_from_another_conversation_invalidates_it():
    """Another client plays: the question is about a board that is gone. The
    panel's next turn finds it expired, once, and nothing of it stands."""
    client, provider, ctx, turns = _asked()
    question = ctx.clarifications[PANEL_ORIGIN]
    ctx.settings.verbosity = "low"
    thread = client.post("/api/agent/conversations", json={}).json()["id"]

    client.post(f"/api/agent/conversations/{thread}/messages", json={"content": "e4"})
    # The delegate's own turn left the panel's question where it was: it is
    # not that thread's to read, answer or drop.
    assert turns.records[-1]["clarification"]["open"] is None
    assert turns.records[-1]["clarification"]["closed"] is None
    assert ctx.clarifications[PANEL_ORIGIN] is question

    _quiet(provider)
    client.post("/api/command", json={"text": "the one to f3"})

    expired = turns.records[-1]["clarification"]["expired"]
    assert (expired["id"], expired["status"], expired["reason"]) == (
        question.id,
        INVALIDATED,
        BOARD_CHANGED,
    )
    assert PANEL_ORIGIN not in ctx.clarifications
    _quiet(provider)
    client.post("/api/command", json={"text": "hm"})
    assert turns.records[-1]["clarification"]["expired"] is None


def test_the_undo_button_invalidates_it():
    ctx = ToolContext(session=GameSession(), engine=FakeEngine())
    for san in ("e4", "e5"):
        ctx.session.submit_move(san)
    client, provider, ctx, turns = _asked(ctx)
    assert PANEL_ORIGIN in ctx.clarifications

    assert client.post("/api/game/undo", json={}).status_code == 200
    _quiet(provider)
    client.post("/api/command", json={"text": "the one to f3"})

    expired = turns.records[-1]["clarification"]["expired"]
    assert (expired["status"], expired["reason"]) == (INVALIDATED, BOARD_CHANGED)


def test_a_resumed_game_invalidates_it():
    client, provider, ctx, turns = _asked()

    ctx.replace_session(GameSession(), Transcript())  # what resume_game does
    _quiet(provider)
    client.post("/api/command", json={"text": "the one to f3"})

    expired = turns.records[-1]["clarification"]["expired"]
    assert expired["reason"] == GAME_CHANGED


def test_one_thread_never_reads_anothers_question():
    """Two delegate conversations with the same ask: each question is its
    own thread's, and a turn in one neither sees nor closes the other's."""
    turns = CollectedTurns()
    client, provider, ctx = make_client(ASK, QUESTION, tracer=turns)
    first = client.post("/api/agent/conversations", json={}).json()["id"]
    second = client.post("/api/agent/conversations", json={}).json()["id"]

    client.post(
        f"/api/agent/conversations/{first}/messages", json={"content": KNIGHT_ASK}
    )
    asked = ctx.clarifications[delegate_origin(first)]
    _quiet(provider)
    client.post(
        f"/api/agent/conversations/{second}/messages",
        json={"content": "the one to f3"},
    )

    assert set(ctx.clarifications) == {delegate_origin(first)}
    assert ctx.clarifications[delegate_origin(first)] is asked
    record = turns.records[-1]["clarification"]
    assert record == {"open": None, "expired": None, "created": None, "closed": None}


def test_a_yes_after_a_question_confirms_nothing():
    """A clarification is not a confirmation: nothing is armed, so a "yes"
    goes down the planner's road like any utterance, runs no destructive op,
    and leaves the question standing."""
    client, provider, ctx, turns = _asked()
    question = ctx.clarifications[PANEL_ORIGIN]
    version = ctx.board_version

    _quiet(provider)
    body = client.post("/api/command", json={"text": "yes"}).json()

    assert body["tool_results"] == []
    assert ctx.pending is None
    assert ctx.board_version == version
    assert turns.records[-1]["route"] == "brain"
    assert ctx.clarifications[PANEL_ORIGIN] is question


# --- what the planner is shown (PR 2) ----------------------------------------------


def _opening_states(provider) -> list[dict]:
    """The opening board state of every planner request the provider saw —
    the `Board state:` block `LlamaBrain._messages` writes, parsed. Planner
    requests are the ones offered tools; the narrator's never are."""
    states = []
    for call in provider.calls:
        if call["tools"] is None:
            continue
        opening = next(
            m["content"]
            for m in reversed(call["messages"])
            if m["role"] == "user" and m["content"].startswith("Board state:\n")
        )
        block = opening.removeprefix("Board state:\n").split("\n\nCommand: ")[0]
        states.append(json.loads(block))
    return states


def _narrator_text(provider) -> str:
    return "\n".join(
        str(m.get("content"))
        for call in provider.calls
        if call["tools"] is None
        for m in call["messages"]
    )


def test_the_planner_is_shown_the_open_question():
    """The record, not the narrator's words: the player's own ask and the
    validated candidates, on every planner request while it stands."""
    client, provider, ctx, _ = _asked()

    for aside in ("what difficulty am I on?", "the one to f3"):
        _quiet(provider)
        client.post("/api/command", json={"text": aside})
        (state,) = _opening_states(provider)
        assert state["open_question"] == {
            "player_asked": KNIGHT_ASK,
            "choose_between": ["Nf3", "Nh3"],
        }
        assert "closed_question" not in state
        # The planner's only: the narrator never reads the record.
        assert "open_question" not in _narrator_text(provider)


def test_no_question_no_key():
    """A turn with nothing asked carries neither key — which is every turn of
    every gated scenario that asks nothing, so their prompts are unchanged."""
    turns = CollectedTurns()
    client, provider, _ = make_client(
        text_turn("the player said hi"), text_turn("Hi."), tracer=turns
    )

    client.post("/api/command", json={"text": "hello"})

    (state,) = _opening_states(provider)
    assert "open_question" not in state and "closed_question" not in state


def test_a_stale_question_is_named_closed_once_then_forgotten():
    client, provider, ctx, _ = _asked()
    ctx.settings.verbosity = "low"
    thread = client.post("/api/agent/conversations", json={}).json()["id"]
    client.post(f"/api/agent/conversations/{thread}/messages", json={"content": "e4"})

    _quiet(provider)
    client.post("/api/command", json={"text": "the first one"})
    (state,) = _opening_states(provider)
    assert state["closed_question"] == {
        "player_asked": KNIGHT_ASK,
        "why": "the board changed after it was asked",
    }
    assert "open_question" not in state

    _quiet(provider)
    client.post("/api/command", json={"text": "hm?"})
    (state,) = _opening_states(provider)
    assert "open_question" not in state and "closed_question" not in state


def test_a_different_game_is_said_as_one():
    client, provider, ctx, _ = _asked()

    ctx.replace_session(GameSession(), Transcript())
    _quiet(provider)
    client.post("/api/command", json={"text": "the first one"})

    (state,) = _opening_states(provider)
    assert state["closed_question"]["why"] == "a different game is on the board now"


def test_another_threads_question_is_never_shown():
    turns = CollectedTurns()
    client, provider, _ = make_client(ASK, QUESTION, tracer=turns)
    first = client.post("/api/agent/conversations", json={}).json()["id"]
    second = client.post("/api/agent/conversations", json={}).json()["id"]
    client.post(
        f"/api/agent/conversations/{first}/messages", json={"content": KNIGHT_ASK}
    )

    _quiet(provider)
    client.post(
        f"/api/agent/conversations/{second}/messages",
        json={"content": "the one to f3"},
    )

    (state,) = _opening_states(provider)
    assert "open_question" not in state and "closed_question" not in state


def test_the_question_survives_the_input_budget_trimming_the_conversation():
    """The acceptance criterion the record exists for: a conversation long
    enough that the input budget drops its oldest exchanges — the question
    among them — still hands the planner the open question, because the state
    block is never trimmed. And it carries no board fact of its own, so what
    is kept cannot be a stale copy of the position."""
    turns = CollectedTurns()
    client, provider, ctx = make_client(
        ASK, QUESTION, tracer=turns, input_budget_tokens=10_000
    )
    client.post("/api/command", json={"text": KNIGHT_ASK})
    chatter = "tell me more about the history of this opening " * 160
    for _ in range(5):
        _quiet(provider)
        client.post("/api/command", json={"text": chatter})

    _quiet(provider)
    client.post("/api/command", json={"text": "the one to f3"})

    assert turns.records[-1]["input_trimmed"] > 0
    (state,) = _opening_states(provider)
    assert state["open_question"] == {
        "player_asked": KNIGHT_ASK,
        "choose_between": ["Nf3", "Nh3"],
    }
    # Nothing of the question is left in the conversation the planner reads.
    planner = next(call for call in provider.calls if call["tools"] is not None)
    conversation = [m["content"] for m in planner["messages"][1:-1]]
    assert not any("Nf3 or Nh3?" in str(content) for content in conversation)
