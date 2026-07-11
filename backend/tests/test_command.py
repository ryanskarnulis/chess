"""Text command endpoint: user string in → brain → tool calls → new state.

The brain is exercised only through its interface with a scripted fake —
never a live LLM. The endpoint dispatches whatever tool calls the brain
returns through the validated registry, so a misbehaving brain surfaces
as error results, never as corrupted state or a 500.
"""

from fastapi.testclient import TestClient

from chessapp.api import create_app
from chessapp.brain import AgentResponse, ToolCall
from chessapp.game import GameSession
from chessapp.tools import ToolContext
from fakes import ScriptedBrain

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def make_client(
    *responses: AgentResponse,
    reactions: tuple[str, ...] = (),
    brain: ScriptedBrain | None = None,
):
    brain = (
        brain if brain is not None else ScriptedBrain(*responses, reactions=reactions)
    )
    ctx = ToolContext(session=GameSession())
    return TestClient(create_app(ctx, brain=brain)), brain


def test_no_brain_is_503():
    client = TestClient(create_app(ToolContext(session=GameSession())))
    response = client.post("/api/command", json={"text": "play e4"})
    assert response.status_code == 503


def test_missing_text_is_422():
    client, _ = make_client(AgentResponse(text="hi"))
    assert client.post("/api/command", json={}).status_code == 422


def test_brain_receives_board_state_and_command():
    client, brain = make_client(AgentResponse(text="hello"))
    client.post("/api/command", json={"text": "how does it look?"})
    board_state, command = brain.calls[0]
    assert command == "how does it look?"
    assert board_state["fen"] == START_FEN
    assert board_state["turn"] == "white"


def test_brain_view_is_agent_facing_not_the_ui_state():
    """The brain reasons from a purpose-made view — board truth plus the
    player's color and check status — never the UI state document, whose
    per-ply `fens` and `dests` are prompt noise that grows every move."""
    client, brain = make_client(AgentResponse(text="hello"))
    client.post("/api/command", json={"text": "how does it look?"})
    board_state, _ = brain.calls[0]
    assert board_state["fen"] == START_FEN
    assert board_state["turn"] == "white"
    assert board_state["player_color"] == "white"
    assert board_state["in_check"] is False
    assert board_state["game_over"] is False
    assert board_state["outcome"] is None
    assert board_state["history"] == []
    assert board_state["captured"] == {"white": [], "black": []}
    assert "e4" in board_state["legal_moves"]
    assert "fens" not in board_state
    assert "dests" not in board_state


def test_react_gets_the_same_agent_facing_view():
    """Phase two reads the same slim view, rebuilt from the post-move state;
    player_color is the side the command was issued for, so it survives the
    move (and the engine's reply) unchanged."""
    client, brain = make_client(
        AgentResponse(
            text="on it",
            tool_calls=(ToolCall(name="make_move", args={"move": "e4"}),),
        ),
        reactions=("A fine start.",),
    )
    client.post("/api/command", json={"text": "play e4"})
    react_state, _ = brain.react_calls[0]
    assert react_state["player_color"] == "white"
    assert react_state["turn"] == "black"
    assert react_state["in_check"] is False
    assert react_state["history"] == ["e4"]
    assert "e5" in react_state["legal_moves"]
    assert "fens" not in react_state
    assert "dests" not in react_state


def test_command_with_no_tool_calls_returns_commentary_only():
    client, _ = make_client(AgentResponse(text="Which knight did you mean?"))
    body = client.post("/api/command", json={"text": "move the knight"}).json()
    assert body["commentary"] == "Which knight did you mean?"
    assert body["tool_results"] == []
    assert body["state"]["fen"] == START_FEN


def test_command_tool_calls_mutate_state_through_registry():
    client, _ = make_client(
        AgentResponse(
            # Pre-action text: the model's utterance before the tool ran. When
            # tools run the user-facing commentary comes from the reaction, not
            # from this.
            text="on it",
            tool_calls=(ToolCall(name="make_move", args={"move": "e4"}),),
        ),
        reactions=("e4 — the classic King's pawn.",),
    )
    body = client.post("/api/command", json={"text": "play e4"}).json()
    assert body["commentary"] == "e4 — the classic King's pawn."
    assert body["tool_results"][0]["name"] == "make_move"
    assert body["tool_results"][0]["result"]["legal"] is True
    assert body["state"]["history"] == ["e4"]
    assert body["state"]["turn"] == "black"


def test_reaction_reads_new_state_and_changes_not_raw_utterance():
    """The heart of the game loop: after tools run, the agent reacts from the
    *new* game state plus what the tools returned — never from the raw
    utterance. This keeps a future deterministic fast-parse path free to add."""
    client, brain = make_client(
        AgentResponse(
            text="on it",
            tool_calls=(ToolCall(name="make_move", args={"move": "e4"}),),
        ),
        reactions=("Nice opening.",),
    )
    body = client.post("/api/command", json={"text": "play e4"}).json()

    assert body["commentary"] == "Nice opening."
    assert len(brain.react_calls) == 1
    react_state, react_changes = brain.react_calls[0]
    # React sees the post-move board (Black to move, e4 played)...
    assert react_state["history"] == ["e4"]
    assert react_state["turn"] == "black"
    # ...and the tool results (what changed), but not the raw command.
    assert react_changes == body["tool_results"]
    assert react_changes[0]["name"] == "make_move"


def test_no_tool_calls_skips_reaction():
    """A pure question / clarifying reply changes nothing, so there is nothing
    to react to: the direct answer stands and react is never called."""
    client, brain = make_client(AgentResponse(text="Which knight did you mean?"))
    body = client.post("/api/command", json={"text": "move the knight"}).json()
    assert body["commentary"] == "Which knight did you mean?"
    assert brain.react_calls == []


def test_reaction_runs_for_read_only_tools_and_grounds_the_answer():
    """Even a read-only tool triggers the reaction: the answer the user sees is
    grounded in the tool result, not in the model's pre-execution guess."""
    client, brain = make_client(
        AgentResponse(
            text="let me check",
            tool_calls=(ToolCall(name="get_legal_moves", args={}),),
        ),
        reactions=("You have 20 legal moves.",),
    )
    body = client.post("/api/command", json={"text": "what are my options?"}).json()
    assert body["commentary"] == "You have 20 legal moves."
    assert len(brain.react_calls) == 1
    _, react_changes = brain.react_calls[0]
    assert "e4" in react_changes[0]["result"]["moves"]


def test_multiple_tool_calls_run_in_order():
    client, _ = make_client(
        AgentResponse(
            text="Fresh board, and I checked your options.",
            tool_calls=(
                ToolCall(name="new_game", args={}),
                ToolCall(name="get_legal_moves", args={}),
            ),
        )
    )
    body = client.post("/api/command", json={"text": "start over"}).json()
    names = [r["name"] for r in body["tool_results"]]
    assert names == ["new_game", "get_legal_moves"]
    assert "e4" in body["tool_results"][1]["result"]["moves"]


def test_unknown_tool_from_brain_is_error_result_not_500():
    # The failure earns a retry round; here the brain concedes in words.
    client, _ = make_client(
        AgentResponse(
            text="doing something odd",
            tool_calls=(ToolCall(name="launch_rocket", args={}),),
        ),
        AgentResponse(text="I can't do that."),
    )
    response = client.post("/api/command", json={"text": "do it"})
    assert response.status_code == 200
    result = response.json()["tool_results"][0]["result"]
    assert result["ok"] is False
    assert "unknown tool" in result["error"]
    assert response.json()["commentary"] == "I can't do that."


def test_invalid_args_from_brain_is_error_result_not_500():
    client, _ = make_client(
        AgentResponse(
            text="moving",
            tool_calls=(ToolCall(name="make_move", args={"move": 42}),),
        ),
        AgentResponse(text="Sorry, I fumbled that one."),
    )
    response = client.post("/api/command", json={"text": "play something"})
    assert response.status_code == 200
    assert response.json()["tool_results"][0]["result"]["ok"] is False


def test_command_mutation_broadcasts_state_to_ws():
    client, _ = make_client(
        AgentResponse(
            text="done",
            tool_calls=(ToolCall(name="make_move", args={"move": "e4"}),),
        )
    )
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()  # connect snapshot
        client.post("/api/command", json={"text": "play e4"})
        message = ws.receive_json()
    assert message["state"]["history"] == ["e4"]


def test_first_command_sees_empty_transcript():
    client, brain = make_client(AgentResponse(text="hello"))
    client.post("/api/command", json={"text": "hi there"})
    assert brain.transcripts == [[]]


def test_transcript_carries_prior_turns_to_the_brain():
    """The conversation memory: turn N+1's brain call includes turn N's
    user command and the commentary the user actually saw."""
    client, brain = make_client(
        AgentResponse(text="Which knight did you mean?"),
        AgentResponse(text="noted"),
    )
    client.post("/api/command", json={"text": "move the knight"})
    client.post("/api/command", json={"text": "the queenside one"})
    assert brain.transcripts[1] == [
        {"role": "user", "content": "move the knight"},
        {"role": "assistant", "content": "Which knight did you mean?"},
    ]


def test_reaction_commentary_is_what_the_transcript_records():
    """When tools run, the user sees the reaction — so that, not the brain's
    pre-action text, is what the next turn remembers. React itself also gets
    the prior turns (not including the in-flight one)."""
    client, brain = make_client(
        AgentResponse(
            text="on it",
            tool_calls=(ToolCall(name="make_move", args={"move": "e4"}),),
        ),
        AgentResponse(text="you did"),
        reactions=("A bold king's pawn!",),
    )
    client.post("/api/command", json={"text": "play e4"})
    assert brain.react_transcripts == [[]]
    client.post("/api/command", json={"text": "did I open well?"})
    assert brain.transcripts[1] == [
        {"role": "user", "content": "play e4"},
        {"role": "assistant", "content": "A bold king's pawn!"},
    ]


def test_read_only_command_does_not_broadcast():
    client, _ = make_client(
        AgentResponse(
            text="You have 20 moves.",
            tool_calls=(ToolCall(name="get_legal_moves", args={}),),
        ),
        AgentResponse(
            text="done",
            tool_calls=(ToolCall(name="make_move", args={"move": "e4"}),),
        ),
    )
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()
        client.post("/api/command", json={"text": "what are my options?"})
        client.post("/api/command", json={"text": "play e4"})
        message = ws.receive_json()  # first broadcast is the move, not the read
    assert message["state"]["history"] == ["e4"]
