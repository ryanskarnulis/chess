"""Text command endpoint: user string in → the brain's tool loop → new state.

The pipeline's half of the game loop, not the loop itself (that is pinned in
test_llama_brain.py). The brain is exercised only through its interface with a
scripted fake — never a live LLM — but the tool calls it scripts run through the
*real* registry against the real session, so a misbehaving brain surfaces as
error results, never as corrupted state or a 500. What this file pins: the
agent-facing view the brain is handed, that the commentary the user sees is the
loop's own closing turn, the transcript, the broadcast, and the deterministic
fast path.
"""

from fastapi.testclient import TestClient

from chessapp.api import create_app
from chessapp.brain import AgentResponse, ToolCall
from chessapp.game import GameSession
from chessapp.tools import ToolContext
from fakes import FakeEngine, ScriptedBrain, scripted_app

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def make_client(*responses: AgentResponse, brain: ScriptedBrain | None = None):
    app, brain = scripted_app(
        ToolContext(session=GameSession()), *responses, brain=brain
    )
    return TestClient(app), brain


def move(san: str, text: str = "on it") -> AgentResponse:
    return AgentResponse(
        text=text, tool_calls=(ToolCall(name="make_move", args={"move": san}),)
    )


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


def test_one_brain_call_does_the_whole_turn():
    """No phase two: the loop already ran the tools and produced the closing
    comment, so the pipeline consults the brain exactly once."""
    client, brain = make_client(move("e4", text="e4 — the classic King's pawn."))
    body = client.post("/api/command", json={"text": "play e4"}).json()
    assert len(brain.calls) == 1
    assert body["commentary"] == "e4 — the classic King's pawn."
    assert body["tool_results"][0]["name"] == "make_move"
    assert body["tool_results"][0]["result"]["legal"] is True
    assert body["state"]["history"] == ["e4"]
    assert body["state"]["turn"] == "black"


def test_command_with_no_tool_calls_returns_commentary_only():
    client, _ = make_client(AgentResponse(text="Which knight did you mean?"))
    body = client.post("/api/command", json={"text": "move the knight"}).json()
    assert body["commentary"] == "Which knight did you mean?"
    assert body["tool_results"] == []
    assert body["state"]["fen"] == START_FEN


def test_read_only_tool_results_reach_the_ui():
    client, _ = make_client(
        AgentResponse(
            text="You have 20 legal moves.",
            tool_calls=(ToolCall(name="get_legal_moves", args={}),),
        )
    )
    body = client.post("/api/command", json={"text": "what are my options?"}).json()
    assert body["commentary"] == "You have 20 legal moves."
    assert "e4" in body["tool_results"][0]["result"]["moves"]


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
    client, _ = make_client(
        AgentResponse(
            text="I can't do that.",
            tool_calls=(ToolCall(name="launch_rocket", args={}),),
        )
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
            text="Sorry, I fumbled that one.",
            tool_calls=(ToolCall(name="make_move", args={"move": 42}),),
        )
    )
    response = client.post("/api/command", json={"text": "play something"})
    assert response.status_code == 200
    assert response.json()["tool_results"][0]["result"]["ok"] is False


def test_a_rejected_move_leaves_the_board_alone():
    """An illegal move the loop could not recover from is still just data: the
    result says `legal: false` and the board is untouched."""
    client, _ = make_client(move("Nf6", text="That one's not legal for White."))
    body = client.post("/api/command", json={"text": "knight to f6"}).json()
    assert body["tool_results"][0]["result"]["legal"] is False
    assert body["state"]["history"] == []


def test_a_budget_stop_still_says_something():
    """When the loop runs out of iterations or corrections it has no closing
    turn, so the pipeline supplies a line rather than an empty bubble."""
    client, _ = make_client(
        AgentResponse(text="", stop_reason="max_iterations"),
    )
    body = client.post("/api/command", json={"text": "play the best move"}).json()
    assert body["commentary"]


def test_command_mutation_broadcasts_state_to_ws():
    client, _ = make_client(move("e4"))
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()  # connect snapshot
        client.post("/api/command", json={"text": "play e4"})
        message = ws.receive_json()
    assert message["state"]["history"] == ["e4"]


def test_a_move_the_agent_corrected_its_way_to_still_broadcasts():
    """The loop's self-correction is invisible to the pipeline — it sees only
    the calls that ran — but the board change it produced must still reach the
    UI. (Both calls come back in `tool_results`, rejection first.)"""
    client, _ = make_client(
        AgentResponse(
            text="Meant Nf3.",
            tool_calls=(
                ToolCall(name="make_move", args={"move": "Nf6"}),  # rejected
                ToolCall(name="make_move", args={"move": "Nf3"}),  # lands
            ),
        )
    )
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()
        client.post("/api/command", json={"text": "knight to f6"})
        message = ws.receive_json()
    assert message["state"]["history"] == ["Nf3"]


def test_read_only_command_does_not_broadcast():
    client, _ = make_client(
        AgentResponse(
            text="You have 20 moves.",
            tool_calls=(ToolCall(name="get_legal_moves", args={}),),
        ),
        move("e4", text="done"),
    )
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()
        client.post("/api/command", json={"text": "what are my options?"})
        client.post("/api/command", json={"text": "play e4"})
        message = ws.receive_json()  # first broadcast is the move, not the read
    assert message["state"]["history"] == ["e4"]


# --- conversation transcript -------------------------------------------------


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


def test_the_transcript_records_the_users_words_and_the_closing_comment():
    """What the next turn remembers is the conversation the user had — their own
    words and the comment they saw — never the loop's internal tool traffic."""
    client, brain = make_client(
        move("e4", text="A bold king's pawn!"),
        AgentResponse(text="you did"),
    )
    client.post("/api/command", json={"text": "play e4"})
    client.post("/api/command", json={"text": "did I open well?"})
    assert brain.transcripts[1] == [
        {"role": "user", "content": "play e4"},
        {"role": "assistant", "content": "A bold king's pawn!"},
    ]


# --- deterministic fast-parse path (the seam BRIEF reserves) ------------------
#
# An utterance that is exactly one unambiguous legal move skips the model
# entirely and goes straight to make_move through the same registry — one road
# in, one pipeline, minus the LLM. Because it never enters the loop there is no
# closing turn to comment with, so `narrate` supplies one; at verbosity=low a
# canned confirmation stands in for that too, making a plain move a zero-LLM
# turn. Anything ambiguous or non-move reaches the brain unchanged. A
# ScriptedBrain with no scripted responses fails loudly (IndexError) if the
# loop is ever consulted.


def make_fast_client(
    *responses: AgentResponse,
    narrations: tuple[str, ...] = (),
    verbosity: str = "normal",
    engine=None,
):
    ctx = ToolContext(session=GameSession(), engine=engine)
    ctx.settings.verbosity = verbosity
    brain = ScriptedBrain(*responses, narrations=narrations)
    app, _ = scripted_app(ctx, brain=brain)
    return TestClient(app), brain, ctx


def test_plain_move_skips_the_llm_loop():
    client, brain, _ = make_fast_client(narrations=("Pawn out.",))
    body = client.post("/api/command", json={"text": "e4"}).json()
    assert brain.calls == []  # the loop was never entered
    assert body["commentary"] == "Pawn out."
    assert body["tool_results"][0]["name"] == "make_move"
    assert body["tool_results"][0]["result"]["legal"] is True
    assert body["state"]["history"] == ["e4"]


def test_fast_path_covers_spoken_phrases():
    client, brain, _ = make_fast_client()
    body = client.post("/api/command", json={"text": "knight to f3"}).json()
    assert brain.calls == []
    assert body["state"]["history"] == ["Nf3"]


def test_fast_path_narrates_from_the_new_state():
    client, brain, _ = make_fast_client(narrations=("Classic.",))
    body = client.post("/api/command", json={"text": "e4"}).json()
    state, changes = brain.narrate_calls[0]
    assert state["history"] == ["e4"]
    assert state["player_color"] == "white"
    assert changes == body["tool_results"]


def test_low_verbosity_fast_move_is_a_zero_llm_turn():
    client, brain, _ = make_fast_client(verbosity="low")
    body = client.post("/api/command", json={"text": "e4"}).json()
    assert brain.calls == []
    assert brain.narrate_calls == []
    assert body["commentary"] == "e4."


def test_low_verbosity_confirmation_includes_the_engine_reply():
    client, _, _ = make_fast_client(verbosity="low", engine=FakeEngine("e7e5"))
    body = client.post("/api/command", json={"text": "e4"}).json()
    assert body["commentary"] == "e4. e5."
    assert body["state"]["history"] == ["e4", "e5"]


def test_low_verbosity_confirmation_reports_game_over():
    client, _, ctx = make_fast_client(verbosity="low")
    for san in ("e4", "f6", "d3", "g5"):
        ctx.session.submit_move(san)
    body = client.post("/api/command", json={"text": "queen to h5"}).json()
    assert body["commentary"] == "Qh5#. Game over: 1-0 (checkmate)."


def test_non_move_text_reaches_the_brain_unchanged():
    client, brain, _ = make_fast_client(AgentResponse(text="hi!"))
    body = client.post("/api/command", json={"text": "hello"}).json()
    assert brain.calls[0][1] == "hello"
    assert body["commentary"] == "hi!"


def test_illegal_move_text_reaches_the_brain_unchanged():
    # Nf6 is Black's move — nothing legal matches, so the utterance falls
    # through to the agent (whose loop owns illegal-move recovery).
    client, brain, _ = make_fast_client(AgentResponse(text="That's not legal."))
    client.post("/api/command", json={"text": "knight to f6"})
    assert [command for _, command in brain.calls] == ["knight to f6"]


def test_fast_path_turn_is_recorded_in_the_transcript():
    client, brain, _ = make_fast_client(
        AgentResponse(text="you did"), narrations=("Sharp.",)
    )
    client.post("/api/command", json={"text": "e4"})
    client.post("/api/command", json={"text": "did I open well?"})
    assert brain.transcripts[0] == [
        {"role": "user", "content": "e4"},
        {"role": "assistant", "content": "Sharp."},
    ]


def test_fast_move_broadcasts_state_to_ws():
    client, _, _ = make_fast_client(narrations=("ok",))
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()  # connect snapshot
        client.post("/api/command", json={"text": "e4"})
        message = ws.receive_json()
    assert message["state"]["history"] == ["e4"]
