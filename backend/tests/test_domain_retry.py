"""Domain-retry loop: failed tool calls get bounded self-correction.

When a dispatched call comes back `ok: false` — or `make_move` says
`legal: false` — the pipeline feeds the error (plus the current legal-move
list for rejected moves) back to the brain and lets it retry before falling
through to react. Bounded rounds, so a hopeless brain can never loop; a
retry that answers in words instead of tools is a final reply, exactly like
a no-tools turn. All exercised with the scripted brain — never a live LLM.
"""

from fastapi.testclient import TestClient

from chessapp.api import MAX_TOOL_RETRY_ROUNDS, create_app
from chessapp.brain import AgentResponse, ToolCall
from chessapp.game import GameSession
from chessapp.tools import ToolContext
from fakes import ScriptedBrain


def make_client(*responses: AgentResponse, reactions: tuple[str, ...] = ()):
    brain = ScriptedBrain(*responses, reactions=reactions)
    ctx = ToolContext(session=GameSession())
    return TestClient(create_app(ctx, brain=brain)), brain


def move(san: str) -> AgentResponse:
    return AgentResponse(
        text="on it", tool_calls=(ToolCall(name="make_move", args={"move": san}),)
    )


def test_illegal_move_is_retried_and_the_correction_lands():
    """An illegal-move guess no longer ends the turn: the brain gets the
    rejection back and its corrected move is what actually happens."""
    client, brain = make_client(
        move("Nf6"),  # illegal for White from the start
        move("Nf3"),
        reactions=("Knight out, as intended.",),
    )
    body = client.post("/api/command", json={"text": "knight to f6"}).json()

    assert body["state"]["history"] == ["Nf3"]
    assert body["commentary"] == "Knight out, as intended."
    # Both attempts are visible in the turn's tool results, in order.
    assert [r["name"] for r in body["tool_results"]] == ["make_move", "make_move"]
    assert body["tool_results"][0]["result"]["legal"] is False
    assert body["tool_results"][1]["result"]["legal"] is True
    assert len(brain.calls) == 2


def test_retry_command_carries_the_error_and_the_legal_moves():
    client, brain = make_client(
        move("Nf6"),
        move("Nf3"),
    )
    client.post("/api/command", json={"text": "knight to f6"})

    _, retry_command = brain.calls[1]
    assert "make_move" in retry_command
    assert body_mentions_rejection(retry_command)
    # Rejected moves come with the current legal-move list to pick from.
    assert "Nf3" in retry_command
    assert "e4" in retry_command


def body_mentions_rejection(command: str) -> bool:
    return "illegal" in command.lower() or "rejected" in command.lower()


def test_retry_sees_fresh_board_state():
    """The retry call reasons from the state *after* the earlier calls ran,
    not the pre-command snapshot."""
    client, brain = make_client(
        AgentResponse(
            text="on it",
            tool_calls=(
                ToolCall(name="make_move", args={"move": "e4"}),
                ToolCall(name="make_move", args={"move": "e4"}),  # now illegal
            ),
        ),
        AgentResponse(text="never mind"),
    )
    client.post("/api/command", json={"text": "double push"})

    retry_state, _ = brain.calls[1]
    assert retry_state["history"] == ["e4"]
    assert retry_state["turn"] == "black"


def test_ok_false_domain_error_is_retried():
    client, brain = make_client(
        AgentResponse(text="undoing", tool_calls=(ToolCall(name="undo", args={}),)),
        AgentResponse(text="Nothing to undo yet — the board is fresh."),
    )
    body = client.post("/api/command", json={"text": "take that back"}).json()

    assert len(brain.calls) == 2
    _, retry_command = brain.calls[1]
    assert body["tool_results"][0]["result"]["ok"] is False
    assert body["tool_results"][0]["result"]["error"] in retry_command


def test_wordy_retry_reply_is_the_commentary_and_skips_react():
    """A retry that answers in words (clarification, giving up) is a final
    reply — the user sees it verbatim, and react never runs."""
    client, brain = make_client(
        move("Nf6"),
        AgentResponse(text="f6 isn't reachable — did you mean Nf3?"),
    )
    body = client.post("/api/command", json={"text": "knight to f6"}).json()

    assert body["commentary"] == "f6 isn't reachable — did you mean Nf3?"
    assert brain.react_calls == []
    assert body["state"]["history"] == []


def test_retries_are_bounded_then_fall_through_to_react():
    """A brain that never self-corrects gets exactly MAX_TOOL_RETRY_ROUNDS
    extra chances, then the turn ends with a reaction to the failures."""
    client, brain = make_client(
        move("Nf6"),
        *[move("Nf6") for _ in range(MAX_TOOL_RETRY_ROUNDS)],
        reactions=("I couldn't find that move.",),
    )
    body = client.post("/api/command", json={"text": "knight to f6"}).json()

    assert len(brain.calls) == 1 + MAX_TOOL_RETRY_ROUNDS
    assert body["commentary"] == "I couldn't find that move."
    assert len(brain.react_calls) == 1
    assert all(r["result"]["legal"] is False for r in body["tool_results"])
    assert body["state"]["history"] == []


def test_successful_calls_do_not_trigger_a_retry():
    client, brain = make_client(
        move("e4"),
        reactions=("A fine start.",),
    )
    body = client.post("/api/command", json={"text": "play e4"}).json()

    assert len(brain.calls) == 1
    assert body["state"]["history"] == ["e4"]


def test_only_failed_calls_appear_in_the_correction():
    client, brain = make_client(
        AgentResponse(
            text="on it",
            tool_calls=(
                ToolCall(name="get_legal_moves", args={}),
                ToolCall(name="make_move", args={"move": "Nf6"}),
            ),
        ),
        move("Nf3"),
        reactions=("Done.",),
    )
    client.post("/api/command", json={"text": "options, then knight f6"})

    _, retry_command = brain.calls[1]
    assert "get_legal_moves" not in retry_command
    assert "make_move" in retry_command


def test_transcript_records_the_users_command_not_the_correction():
    """The retry correction is plumbing, not conversation: the next turn's
    transcript shows the user's own words and the commentary they saw."""
    client, brain = make_client(
        move("Nf6"),
        move("Nf3"),
        AgentResponse(text="noted"),
        reactions=("Fixed: Nf3.",),
    )
    client.post("/api/command", json={"text": "knight to f6"})
    client.post("/api/command", json={"text": "thanks"})

    assert brain.transcripts[2] == [
        {"role": "user", "content": "knight to f6"},
        {"role": "assistant", "content": "Fixed: Nf3."},
    ]


def test_mutation_via_retry_still_broadcasts():
    client, _ = make_client(
        move("Nf6"),
        move("Nf3"),
        reactions=("There.",),
    )
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()  # connect snapshot
        client.post("/api/command", json={"text": "knight to f6"})
        message = ws.receive_json()
    assert message["state"]["history"] == ["Nf3"]
