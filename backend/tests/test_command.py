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


def test_player_color_is_the_sessions_not_whoever_is_to_move():
    """The player's color is session state the session already owns — it must
    not be re-derived from whose turn it is. A command can legitimately arrive
    on the engine's turn (asking a question while the engine thinks, or any
    command in an engine-less session where the player has both sides), and
    `turn` names the wrong side there."""
    session = GameSession(player_color="black")  # engine is white, and moves first
    assert session.turn == "white"  # not the player's turn
    app, brain = scripted_app(ToolContext(session=session), AgentResponse(text="hi"))
    client = TestClient(app)

    client.post("/api/command", json={"text": "how does it look?"})

    board_state, _ = brain.calls[0]
    assert board_state["player_color"] == "black"
    assert board_state["turn"] == "white"  # board truth is still board truth


def test_brain_is_told_which_saved_games_exist():
    """Whether a save exists is a question the filesystem answers, so the brain
    is told — it must never have to infer it from the conversation."""
    client, brain = make_client(AgentResponse(text="hi"))
    client.post("/api/command", json={"text": "how does it look?"})
    assert brain.calls[0][0]["saved_games"] == []


def test_saved_games_lists_what_is_on_disk(tmp_path):
    session = GameSession()
    session.save(tmp_path / "scholars.json")
    app, brain = scripted_app(
        ToolContext(session=GameSession(), save_dir=tmp_path), AgentResponse(text="hi")
    )
    TestClient(app).post("/api/command", json={"text": "load the game I saved"})
    assert brain.calls[0][0]["saved_games"] == ["scholars"]


def test_a_past_failure_in_the_transcript_cannot_suppress_the_fresh_fact(tmp_path):
    """The self-poisoning bug (trace review 2026-07-13, finding 5). Live, one
    prior assistant turn saying saving had failed made the model stop calling
    `resume_game` entirely and invent a reason — "it hasn't been saved yet" —
    about a file sitting on disk. The transcript carries only prose, and prose
    was the model's only source of truth about saves.

    It no longer is. Whatever the thread says happened, the state the brain is
    handed still reports what is actually on disk, this turn."""
    session = GameSession()
    session.save(tmp_path / "scholars.json")
    ctx = ToolContext(session=GameSession(), save_dir=tmp_path)
    ctx.transcript.record(
        "save this game as testgame and give me the pgn",
        "I can't save the game right now because the save directory isn't set up.",
    )
    app, brain = scripted_app(ctx, AgentResponse(text="hi"))

    TestClient(app).post("/api/command", json={"text": "load up the game I saved"})

    assert brain.calls[0][0]["saved_games"] == ["scholars"]


def test_saved_games_is_read_fresh_every_turn(tmp_path):
    """Per turn, not cached at assembly: a game saved *during* this session is
    there on the next one."""
    app, brain = scripted_app(
        ToolContext(session=GameSession(), save_dir=tmp_path),
        AgentResponse(
            text="saved",
            tool_calls=(ToolCall(name="save_game", args={"name": "midgame"}),),
        ),
        AgentResponse(text="hi"),
    )
    client = TestClient(app)

    client.post("/api/command", json={"text": "save this as midgame"})
    assert brain.calls[0][0]["saved_games"] == []

    client.post("/api/command", json={"text": "what have I got saved?"})
    assert brain.calls[1][0]["saved_games"] == ["midgame"]


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


def test_fast_path_covers_a_capture_that_names_no_square():
    # "bishop takes" has exactly one legal referent here (Bxh6), so the board
    # settles it and the model is never asked — the failure the trace review
    # caught (finding 3) is now structurally impossible.
    client, brain, ctx = make_fast_client()
    for san in ("e4", "h6", "d4", "d6", "e5", "e6"):
        ctx.session.submit_move(san)
    body = client.post("/api/command", json={"text": "bishop takes"}).json()
    assert brain.calls == []
    assert body["state"]["history"][-1] == "Bxh6"


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


# --- The destructive-op confirmation gate, at the pipeline.
#
# The tool boundary refuses an unconfirmed new_game/resign and arms it (see
# test_tools.py). The pipeline owns the other half: the *answer*. A bare "yes"
# executes the armed op deterministically — no model call, because there is
# nothing left to decide — and "no" drops it. Anything else is a new intent and
# reaches the brain normally, taking the armed op down with it.


def destructive(name: str, text: str = "you sure about that?") -> AgentResponse:
    return AgentResponse(text=text, tool_calls=(ToolCall(name=name, args={}),))


def developed(ctx: ToolContext) -> ToolContext:
    for san in ("e4", "e5", "Nf3", "Nc6"):
        ctx.session.submit_move(san)
    return ctx


def make_developed_client(*responses: AgentResponse):
    ctx = developed(ToolContext(session=GameSession()))
    app, brain = scripted_app(ctx, *responses)
    return TestClient(app), brain, ctx


def test_bare_new_game_asks_instead_of_resetting():
    client, brain, ctx = make_developed_client(destructive("new_game"))
    fen_before = ctx.session.fen()

    response = client.post("/api/command", json={"text": "new game"}).json()

    assert response["state"]["fen"] == fen_before, "board must not reset on the ask"
    assert response["commentary"] == "you sure about that?"
    assert ctx.pending is not None and ctx.pending.name == "new_game"
    # The agent saw the refusal as a result and asked from it.
    assert response["tool_results"][0]["result"]["ok"] is False


def test_yes_executes_the_armed_op_with_no_model_call():
    client, brain, ctx = make_developed_client(destructive("new_game"))
    client.post("/api/command", json={"text": "new game"})
    calls_after_ask = len(brain.calls)

    response = client.post("/api/command", json={"text": "yes"}).json()

    assert response["state"]["fen"] == GameSession().fen(), "confirmed: reset"
    assert len(brain.calls) == calls_after_ask, "a confirmation needs no model turn"
    assert [r["name"] for r in response["tool_results"]] == ["new_game"]
    assert response["tool_results"][0]["result"]["ok"] is True
    assert ctx.pending is None


def test_yes_confirms_a_resign_and_ends_the_game():
    client, brain, ctx = make_developed_client(destructive("resign"))
    client.post("/api/command", json={"text": "i resign"})

    response = client.post("/api/command", json={"text": "yes"}).json()

    assert ctx.session.is_game_over()
    assert response["state"]["game_over"] is True


def test_no_drops_the_armed_op_and_the_game_stands():
    client, brain, ctx = make_developed_client(destructive("new_game"))
    client.post("/api/command", json={"text": "new game"})
    fen_before = ctx.session.fen()

    response = client.post("/api/command", json={"text": "no"}).json()

    assert ctx.session.fen() == fen_before
    assert ctx.pending is None, "declined: the op is gone, not still armed"
    assert response["tool_results"] == []
    assert response["commentary"]


def test_a_later_yes_cannot_revive_a_declined_op():
    """The gate must not leave a loaded gun lying around: once declined, a bare
    "yes" is just an utterance for the brain, not a licence to reset."""
    client, brain, ctx = make_developed_client(
        destructive("new_game"), AgentResponse(text="yes to what?")
    )
    client.post("/api/command", json={"text": "new game"})
    client.post("/api/command", json={"text": "no"})
    fen_before = ctx.session.fen()

    response = client.post("/api/command", json={"text": "yes"}).json()

    assert ctx.session.fen() == fen_before
    assert response["commentary"] == "yes to what?", "it reached the brain"


def test_an_unrelated_command_drops_the_armed_op():
    client, brain, ctx = make_developed_client(destructive("new_game"))
    client.post("/api/command", json={"text": "new game"})

    client.post("/api/command", json={"text": "e5"})  # fast path: a plain move

    assert ctx.pending is None, "changing the subject disarms the op"
    assert ctx.session.fen() != GameSession().fen()
