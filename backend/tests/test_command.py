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

from chessapp.api import (
    MOVE_ADVICE_REPLY,
    PROVIDER_LOST_RETRY,
    PROVIDER_LOST_TURN_STANDS,
    UNTRUE_CLAIM_REPLY,
    UNVERIFIED_CLAIM_REPLY,
    create_app,
)
from chessapp.brain import AgentResponse, ToolCall
from chessapp.conversation import RECENT_TURNS
from chessapp.engine import DEFAULT_TIER, CandidateMove, Evaluation
from chessapp.game import GameSession
from chessapp.provider import ProviderError
from chessapp.tools import ToolContext
from fakes import FakeEngine, ScriptedBrain, receive_state, scripted_app

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


def test_brain_is_told_the_live_settings():
    """Difficulty and voice-output are facts the app holds. Without them in the
    state block the model's only source for "how hard am I playing?" is its own
    past prose — the same self-poisoning shape as `saved_games`."""
    client, brain = make_client(AgentResponse(text="hi"))
    client.post("/api/command", json={"text": "how hard are you playing?"})
    assert brain.calls[0][0]["settings"] == {
        "difficulty": {"tier": DEFAULT_TIER},
        "voice_output": False,
    }


def test_settings_block_shows_only_the_difficulty_field_that_is_set():
    """`Settings` records exactly one of tier / skill_level / elo; the block
    must not imply two difficulties are in force at once."""
    ctx = ToolContext(session=GameSession())
    ctx.settings.tier = None
    ctx.settings.elo = 1400
    app, brain = scripted_app(ctx, AgentResponse(text="hi"))
    TestClient(app).post("/api/command", json={"text": "how does it look?"})
    assert brain.calls[0][0]["settings"]["difficulty"] == {"elo": 1400}


def test_settings_are_read_fresh_every_turn():
    """Changed mid-session by the agent's own tool call, the *next* turn sees
    the new value — not the assembly-time one."""
    app, brain = scripted_app(
        ToolContext(session=GameSession(), engine=FakeEngine()),
        AgentResponse(
            text="ok",
            tool_calls=(
                ToolCall(name="set_difficulty", args={"skill_level": 7}),
                ToolCall(name="set_voice_output", args={"enabled": True}),
            ),
        ),
        AgentResponse(text="hi"),
    )
    client = TestClient(app)

    client.post("/api/command", json={"text": "play harder and talk to me"})
    assert brain.calls[0][0]["settings"] == {
        "difficulty": {"tier": DEFAULT_TIER},
        "voice_output": False,
    }

    client.post("/api/command", json={"text": "how hard are you playing?"})
    assert brain.calls[1][0]["settings"] == {
        "difficulty": {"skill_level": 7},
        "voice_output": True,
    }


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


def test_a_repeat_stop_speaks_for_itself_instead_of_the_stuck_line():
    """`no_progress` is the loop ending a planner that had started repeating
    itself — the narrator still ran, so the turn has real commentary and the
    pipeline must not paper over it with the canned stuck reply (the whole point
    of the stop: the hints-off ask used to reach the player as "I lost the
    thread" for want of one more iteration it had no use for)."""
    client, _ = make_client(
        AgentResponse(text="Hints are off. Figure it out.", stop_reason="no_progress"),
    )
    body = client.post("/api/command", json={"text": "what should I play?"}).json()
    assert body["commentary"] == "Hints are off. Figure it out."


def test_command_mutation_broadcasts_state_to_ws():
    client, _ = make_client(move("e4"))
    with client.websocket_connect("/ws") as ws:
        receive_state(ws)  # connect snapshot
        client.post("/api/command", json={"text": "play e4"})
        message = receive_state(ws)
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
        receive_state(ws)
        client.post("/api/command", json={"text": "knight to f6"})
        message = receive_state(ws)
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
        receive_state(ws)
        client.post("/api/command", json={"text": "what are my options?"})
        client.post("/api/command", json={"text": "play e4"})
        message = receive_state(ws)  # first broadcast is the move, not the read
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


def test_older_turns_reach_the_brain_condensed_not_verbatim():
    """The memory policy at the boundary (`docs/turn-memory.md`): what the brain
    is handed is the last few turns verbatim behind a digest of what the player
    asked for earlier — not twenty turns of Glitch's prose."""
    ctx = ToolContext(session=GameSession())
    ctx.transcript.record("save this as testgame", "I can't save right now, bro.")
    app, brain = scripted_app(
        ctx, *[AgentResponse(text=f"reply {i}") for i in range(7)]
    )
    client = TestClient(app)
    for i in range(RECENT_TURNS):
        client.post("/api/command", json={"text": f"filler {i}"})
    client.post("/api/command", json={"text": "load the game I saved"})

    transcript = brain.transcripts[-1]
    assert len(transcript) == 2 + 2 * RECENT_TURNS
    digest = transcript[0]["content"]
    assert '"save this as testgame"' in digest
    # The stale claim itself is gone — that sentence is precisely what taught the
    # model a save it had made didn't exist (trace review 2026-07-13).
    assert "can't save right now" not in digest
    assert transcript[-2:] == [
        {"role": "user", "content": f"filler {RECENT_TURNS - 1}"},
        {"role": "assistant", "content": f"reply {RECENT_TURNS - 1}"},
    ]


def test_a_recent_turn_still_reaches_the_brain_word_for_word():
    """The digest must not eat the turns references point at: "the queenside
    one" only resolves against the question that prompted it."""
    client, brain = make_client(*[AgentResponse(text=f"reply {i}") for i in range(9)])
    for i in range(6):
        client.post("/api/command", json={"text": f"filler {i}"})
    client.post("/api/command", json={"text": "move the knight"})
    client.post("/api/command", json={"text": "the queenside one"})
    assert brain.transcripts[-1][-2:] == [
        {"role": "user", "content": "move the knight"},
        {"role": "assistant", "content": "reply 6"},
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
        receive_state(ws)  # connect snapshot
        client.post("/api/command", json={"text": "e4"})
        message = receive_state(ws)
    assert message["state"]["history"] == ["e4"]


# --- The observe beat: a reaction to the player's move, then the reply.
#
# `make_move` applies the player's move and stops (audit items 2/5), so the
# narration Glitch produces on any route is now a reaction to a *verified player
# move* rather than to a finished exchange. The pipeline runs the beats around
# it: the reaction (which overlaps Stockfish, already computing since the move
# landed), then collecting the reply, then a deterministic line announcing it.
# The close beat is canned on purpose — the narration has already been paid for,
# and a second round trip to react to the reply is the latency the acceptance
# criterion forbids.


def observing_client(
    *responses: AgentResponse,
    narrations: tuple = (),
    verbosity: str = "normal",
    reply_uci: str = "e7e5",
):
    ctx = ToolContext(session=GameSession(), engine=FakeEngine(reply_uci))
    ctx.settings.verbosity = verbosity
    brain = ScriptedBrain(*responses, narrations=narrations)
    app, _ = scripted_app(ctx, brain=brain)
    return TestClient(app), brain, ctx


def test_the_observation_reacts_to_the_player_move_alone():
    """The board the narrator is handed has the player's move on it and nothing
    else: the reply does not exist yet, so Glitch cannot be asked to react to
    something it has not seen."""
    client, brain, _ = observing_client(narrations=("Bold opener.",))

    body = client.post("/api/command", json={"text": "e4"}).json()

    assert len(brain.narrate_calls) == 1
    state, changes = brain.narrate_calls[0]
    assert state["history"] == ["e4"], "the engine has not replied yet"
    assert state["turn"] == "black"
    result = changes[0]["result"]
    assert result["san"] == "e4"
    assert "engine_move" not in result
    assert body["state"]["history"] == ["e4", "e5"], "and then it replied"


def test_the_observation_is_handed_the_facts_about_the_move():
    """The structured facts the audit asks for: what moved, what it took, and
    whether it checks — all from the session, none from the model."""
    client, brain, ctx = observing_client(narrations=("Mine.",), reply_uci="b8c6")
    for san in ("e4", "d5"):
        ctx.session.submit_move(san)

    client.post("/api/command", json={"text": "exd5"})

    result = brain.narrate_calls[0][1][0]["result"]
    assert result["capture"] == "p"
    assert result["check"] is False


def test_the_reply_is_announced_after_the_reaction():
    client, _, _ = observing_client(narrations=("Bold opener.",))
    body = client.post("/api/command", json={"text": "e4"}).json()
    assert body["commentary"] == "Bold opener.\n\ne5."


def test_the_brain_routes_move_gets_the_reply_appended_too():
    """One convergent close beat: the loop's own closing narration is the observe
    beat on that route, and the reply announcement follows it the same way."""
    ctx = ToolContext(session=GameSession(), engine=FakeEngine("e7e5"))
    app, _ = scripted_app(ctx, move("e4", text="King's pawn, obviously."))

    body = TestClient(app).post("/api/command", json={"text": "play e4"}).json()

    assert body["commentary"] == "King's pawn, obviously.\n\ne5."
    assert body["state"]["history"] == ["e4", "e5"]
    # What the brain saw is what it reacted to: no reply in the tool result.
    assert "engine_move" not in body["tool_results"][0]["result"]


def test_a_turn_that_moved_nothing_announces_nothing():
    ctx = ToolContext(session=GameSession(), engine=FakeEngine())
    app, _ = scripted_app(ctx, AgentResponse(text="Twenty moves, take your pick."))

    body = TestClient(app).post("/api/command", json={"text": "how's it look?"}).json()

    assert body["commentary"] == "Twenty moves, take your pick."
    assert body["state"]["history"] == []


def test_a_game_ending_player_move_has_no_reply_to_announce():
    client, _, ctx = observing_client(narrations=("Called it.",))
    for san in ("e4", "f6", "d3", "g5"):
        ctx.session.submit_move(san)

    body = client.post("/api/command", json={"text": "queen to h5"}).json()

    assert ctx.session.is_game_over()
    assert body["commentary"] == "Called it.", "nothing replied, so nothing is added"


def test_a_reply_that_ends_the_game_says_so():
    """The close beat carries the outcome, because the reply is the one move
    Glitch's reaction could not have seen coming."""
    client, _, ctx = observing_client(narrations=("Your funeral.",), reply_uci="d8h4")
    for san in ("f3", "e5"):
        ctx.session.submit_move(san)

    body = client.post("/api/command", json={"text": "g4"}).json()

    assert ctx.session.is_game_over()
    assert body["commentary"] == "Your funeral.\n\nQh4#. Game over: 0-1 (checkmate)."


def test_a_failed_observation_still_gets_the_reply_and_the_turn():
    """Direct-mode degradation (audit item 20): the reaction is optional by
    construction, the engine's reply is not. A provider failure costs the words
    and nothing else — the move it was about has already been played."""
    client, _, ctx = observing_client(
        narrations=(ProviderError("llama-server went away"),)
    )

    body = client.post("/api/command", json={"text": "e4"}).json()

    assert body["commentary"] == "e4. e5.", "the canned confirmation stands in"
    assert ctx.session.move_history() == ["e4", "e5"]
    # And the turn really closed: the next move is accepted straight away.
    second = client.post("/api/command", json={"text": "Nf3"}).json()
    assert second["state"]["history"][:3] == ["e4", "e5", "Nf3"]


def test_low_verbosity_keeps_the_zero_llm_move_with_the_reply_in_one_line():
    """The latency floor the acceptance criterion protects: verbosity=low skips
    the reaction exactly as it always skipped narration, and the whole turn is
    one canned line assembled from the two results."""
    client, brain, _ = observing_client(verbosity="low")

    body = client.post("/api/command", json={"text": "e4"}).json()

    assert brain.calls == [] and brain.narrate_calls == []
    assert body["commentary"] == "e4. e5."


def test_an_undo_inside_the_turn_abandons_the_owed_reply():
    """undo-then-replace in one turn: the undo abandons the coordinator's turn,
    so the reply that was being computed for the undone move never lands."""
    ctx = ToolContext(session=GameSession(), engine=FakeEngine("e7e5"))
    app, _ = scripted_app(
        ctx,
        AgentResponse(
            text="Fine, taken back.",
            tool_calls=(
                ToolCall(name="make_move", args={"move": "e4"}),
                ToolCall(name="undo", args={}),
            ),
        ),
    )

    body = (
        TestClient(app)
        .post("/api/command", json={"text": "e4 — no wait, take that back"})
        .json()
    )

    assert ctx.session.move_history() == []
    assert body["commentary"] == "Fine, taken back.", "no reply to announce"


def test_a_false_ending_in_the_observation_is_still_guarded():
    """The guard runs on the assembled commentary against the post-reply board,
    so a reaction that invents an ending is caught on the new road too."""
    client, _, ctx = observing_client(narrations=("That's the game. Game over.",))

    body = client.post("/api/command", json={"text": "e4"}).json()

    assert not ctx.session.is_game_over()
    assert body["commentary"] == UNTRUE_CLAIM_REPLY
    assert "e5" not in body["commentary"], "the lie takes the reply line with it"


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


def make_developed_client(*responses: AgentResponse, narrations: tuple[str, ...] = ()):
    ctx = developed(ToolContext(session=GameSession()))
    brain = ScriptedBrain(*responses, narrations=narrations)
    app, brain = scripted_app(ctx, brain=brain)
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


# --- One destructive op per command (audit item 6).
#
# The gate stands aside when there is nothing to lose, which is right — and left
# a hole the width of one command: the brain loop is the one surface that can
# chain several dispatches inside a single user interaction, so it could reset a
# finished game and then resign the fresh board, both past the gate. The
# coordinator now holds a command-scoped budget the pipeline opens and closes.


def finished(ctx: ToolContext) -> ToolContext:
    """Fool's mate: the game is over, so the gate has nothing to guard."""
    for san in ("f3", "e5", "g4", "Qh4"):
        ctx.session.submit_move(san)
    return ctx


def test_only_the_first_destructive_op_in_a_command_runs():
    ctx = finished(ToolContext(session=GameSession()))
    twice = AgentResponse(
        text="board's fresh",
        tool_calls=(
            ToolCall(name="new_game", args={}),
            ToolCall(name="new_game", args={"player_color": "black"}),
        ),
    )
    app, _ = scripted_app(ctx, twice)
    client = TestClient(app)

    response = client.post("/api/command", json={"text": "new game"}).json()

    results = [r["result"] for r in response["tool_results"]]
    assert results[0]["ok"] is True
    # The refusal is ordinary result data, on the same road as an illegal move:
    # the loop reads it and reports what happened rather than crashing.
    assert results[1]["ok"] is False
    assert "one per turn" in results[1]["error"]
    assert ctx.session.player_color == "white", "the second reset never ran"
    assert ctx.session.move_history() == []
    assert not ctx.session.is_game_over(), "one live fresh game came out of it"


def test_the_command_window_closes_so_the_buttons_still_work():
    """The window is the pipeline's, and it must not outlive the command: a
    button press is its own interaction and spends its own budget."""
    ctx = finished(ToolContext(session=GameSession()))
    app, _ = scripted_app(
        ctx,
        AgentResponse(text="fresh", tool_calls=(ToolCall(name="new_game", args={}),)),
    )
    client = TestClient(app)
    client.post("/api/command", json={"text": "new game"})  # spends the budget

    response = client.post("/api/game/new", json={"color": "black"})

    assert response.status_code == 200
    assert ctx.session.player_color == "black"


# --- The honesty guard: the player is never told something that didn't happen.
#
# The deepest version of the house rule. The closing turn is produced from a
# context ending in the new board and it still invents endings — "Word. Game
# over." on a live board (trace review, finding 6). The board knows whether the
# game ended, so the board, not the model, gets the last word on saying so.


def test_commentary_may_not_announce_an_ending_that_never_happened():
    client, _, ctx = make_developed_client(AgentResponse(text="Word. Game over."))

    response = client.post("/api/command", json={"text": "i'm done with this"}).json()

    assert not ctx.session.is_game_over(), "the board never ended the game"
    assert "Game over" not in response["commentary"]
    assert response["commentary"] == UNTRUE_CLAIM_REPLY


def test_a_real_ending_is_narrated_as_it_stands():
    """The guard checks the board, not the vocabulary: the same words are fine
    when they are true."""
    ctx = ToolContext(session=GameSession())
    for san in ("e4", "e5", "Bc4", "Nc6", "Qh5", "Nf6"):
        ctx.session.submit_move(san)
    client = TestClient(
        scripted_app(
            ctx,
            AgentResponse(
                text="Checkmate. Game over.",
                tool_calls=(ToolCall(name="make_move", args={"move": "Qxf7#"}),),
            ),
        )[0]
    )

    # Not a phrasing `parse_move` settles, so the brain's turn is the commentary.
    response = client.post("/api/command", json={"text": "finish him"}).json()

    assert ctx.session.is_game_over()
    assert response["commentary"] == "Checkmate. Game over."


def test_a_confirmed_resignation_may_say_the_game_is_over():
    """A destructive op that really ran licenses the claim even though it is the
    tool, not the board's checkmate detection, that ended things."""
    client, _, ctx = make_developed_client(narrations=("Done. Game over.",))
    client.post("/api/command", json={"text": "i resign"})

    response = client.post("/api/command", json={"text": "yes"}).json()

    assert ctx.session.is_game_over()
    assert response["commentary"] == "Done. Game over."


# --- The advice guard: hints off means no move is handed over.
#
# Audit item 11's second half. The capability cut (get_best_moves withheld from
# the offer) killed the tool half of the leak, but the model still invents a
# move from its own head — measured live at 2/5–3/5 after the cut. Same shape
# as the honesty guard: the board knows what is playable and the settings know
# whether help was wanted, so the code, not the prompt, gets the last word.


def test_hints_off_commentary_may_not_hand_over_a_move():
    ctx = ToolContext(session=GameSession())
    app, _ = scripted_app(ctx, AgentResponse(text="Easy: play Nf3 and thank me later."))
    client = TestClient(app)

    response = client.post("/api/command", json={"text": "what should I play?"}).json()

    assert ctx.settings.hints_mode is False  # the default
    assert "Nf3" not in response["commentary"]
    assert response["commentary"] == MOVE_ADVICE_REPLY


def test_hints_on_lets_advice_through():
    ctx = ToolContext(session=GameSession())
    ctx.settings.hints_mode = True
    app, _ = scripted_app(ctx, AgentResponse(text="Play Nf3."))
    client = TestClient(app)

    response = client.post("/api/command", json={"text": "what should I play?"}).json()

    assert response["commentary"] == "Play Nf3."


def test_hints_off_analysis_the_player_asked_for_keeps_its_moves():
    """The guard checks evidence, not vocabulary: a move a successful analysis
    tool reported this turn is a verified fact the commentary may repeat —
    "what was my mistake?" works with hints off, and its answer names moves."""
    ctx = ToolContext(session=GameSession())
    ctx.engine = FakeEngine(
        best_moves=(CandidateMove(uci="g1f3", san="Nf3", score_cp=30, mate_in=None),)
    )
    app, _ = scripted_app(
        ctx,
        AgentResponse(
            text="Nf3 was the better try.",
            tool_calls=(ToolCall(name="get_best_moves", args={"n": 1}),),
        ),
    )
    client = TestClient(app)

    body = {"text": "what's the best move?"}
    response = client.post("/api/command", json=body).json()

    assert response["commentary"] == "Nf3 was the better try."


def test_analysis_licenses_only_the_moves_it_reported():
    """The evidence test is scoped to the analysis's *own* moves. Live, the
    planner answered "what should I play here?" with `evaluate_position` +
    `analyze_last_move`, which switched the whole guard off and let the
    narrator hand over a shopping list of moves the analysis never mentioned
    (`['Bc4', 'c3', 'd3']` — docs/agent-evals.md, 2026-07-25). A tool result
    licenses repeating what it reported; it does not license the rest of the
    legal-move list."""
    ctx = ToolContext(session=GameSession())
    ctx.engine = FakeEngine(
        best_moves=(CandidateMove(uci="g1f3", san="Nf3", score_cp=30, mate_in=None),)
    )
    app, _ = scripted_app(
        ctx,
        AgentResponse(
            text="Nf3 is the engine's pick, but honestly just play e4.",
            tool_calls=(ToolCall(name="get_best_moves", args={"n": 1}),),
        ),
    )
    client = TestClient(app)

    body = {"text": "what should I play here?"}
    response = client.post("/api/command", json=body).json()

    assert response["commentary"] == MOVE_ADVICE_REPLY


def test_the_model_cannot_turn_hints_on_to_license_its_own_advice():
    """The setting that governs a turn is the one in force when the player
    asked. Live, "what should I play here?" got answered by a planner that
    called `set_hints_mode(True)` and then handed over moves — the model
    granting itself the permission the player had switched off. The offer is
    resolved once per command for the same reason; the guard reads the same
    snapshot, so a settings change takes effect from the *next* turn."""
    ctx = ToolContext(session=GameSession())
    app, _ = scripted_app(
        ctx,
        AgentResponse(
            text="Hints on. Play Nc3.",
            tool_calls=(ToolCall(name="set_hints_mode", args={"enabled": True}),),
        ),
    )
    client = TestClient(app)

    body = {"text": "what should I play here?"}
    response = client.post("/api/command", json=body).json()

    assert ctx.settings.hints_mode is True, "the setting change itself stands"
    assert response["commentary"] == MOVE_ADVICE_REPLY


def test_a_mistake_analysis_licenses_its_played_and_best_moves():
    """The exemption the scoping must not break: "what was my mistake?" names
    the move played and the move that was better, and both are facts the
    result reported — even when the better one is still playable now."""
    session = GameSession()
    for san in ("e4", "e5", "Nf3", "Nc6"):
        assert session.submit_move(san).legal
    ctx = ToolContext(session=session)
    ctx.engine = FakeEngine(
        best_moves=(CandidateMove(uci="f1c4", san="Bc4", score_cp=30, mate_in=None),)
    )
    app, _ = scripted_app(
        ctx,
        AgentResponse(
            text="Nf3 was fine; Bc4 was the sharper try.",
            tool_calls=(ToolCall(name="analyze_last_move", args={}),),
        ),
    )
    client = TestClient(app)

    body = {"text": "how was my last move?"}
    response = client.post("/api/command", json=body).json()

    assert "Bc4" in session.legal_moves(), "the licensed move is playable now"
    assert response["commentary"] == "Nf3 was fine; Bc4 was the sharper try."


def test_a_played_move_may_be_named_with_hints_off():
    """Reacting to what just happened is not advice: the turn changed the
    board, and the commentary describes it."""
    client, _ = make_client(move("e4", text="e4. Predictable."))
    response = client.post("/api/command", json={"text": "king's pawn"}).json()
    assert response["commentary"].startswith("e4. Predictable.")


# --- The honesty guard, generalized: every operational claim needs evidence.
#
# Audit item 13. The ending class above is the same rule at its most severe;
# these are the rest of the facts a turn produces. The evidence is assembled
# from the turn's tool results, the engine's reply and the board, so the check
# is against what happened, never against what the model remembers saying.


def test_commentary_may_not_announce_a_capture_that_never_happened():
    client, _, ctx = make_developed_client(AgentResponse(text="Snagged your bishop."))

    response = client.post("/api/command", json={"text": "your move"}).json()

    assert ctx.session.captured_pieces() == {"white": [], "black": []}
    assert response["commentary"] == UNVERIFIED_CLAIM_REPLY


def test_a_real_capture_is_narrated_as_it_stands():
    """The guard checks the board, not the vocabulary — the twin of the
    ending class's test, one claim class down."""
    client, _, ctx = make_developed_client(narrations=("You took my pawn. Cute.",))

    response = client.post("/api/command", json={"text": "knight takes pawn"}).json()

    assert ctx.session.captured_pieces()["white"] == ["p"]
    assert response["commentary"].startswith("You took my pawn. Cute.")


def test_commentary_may_not_invent_a_move_that_was_never_on_the_board():
    client, _, _ = make_developed_client(AgentResponse(text="Rough. Qxh7 ends you."))

    response = client.post("/api/command", json={"text": "how bad is it?"}).json()

    assert response["commentary"] == UNVERIFIED_CLAIM_REPLY


def test_the_move_the_player_missed_is_still_sayable():
    """The false positive that would cost the most: post-hoc commentary about
    a move that *was* playable when the turn started. The facts hold both
    positions, so "Bb5 was better" survives a turn that made it illegal."""
    client, _, ctx = make_developed_client(narrations=("Bb5 was better.",))

    response = client.post("/api/command", json={"text": "bishop to c4"}).json()

    assert "Bb5" not in ctx.session.legal_moves(), "no longer playable"
    assert response["commentary"].startswith("Bb5 was better.")


def test_reading_the_move_list_back_is_not_an_invention():
    """The other false positive the recorded turns caught: "read me the move
    list" is answered with moves that are long past being legal. The game's
    history is board truth, so the facts hold it."""
    client, _, ctx = make_developed_client(
        AgentResponse(
            text="Move list: e4, e5, Nf3, Nc6.",
            tool_calls=(ToolCall(name="get_move_history", args={}),),
        )
    )

    body = {"text": "read me the move list"}
    response = client.post("/api/command", json=body).json()

    assert "Nf3" not in ctx.session.legal_moves(), "already played, not playable"
    assert response["commentary"] == "Move list: e4, e5, Nf3, Nc6."


def test_a_settings_claim_is_checked_against_the_live_settings():
    """A setting the model announces must be the setting the app is actually
    on — the same rule as the board, applied to the other state the agent
    can change. Nothing set the difficulty to maximum, so nothing may say so."""
    client, _, ctx = make_developed_client(
        AgentResponse(text="Difficulty is maximum now.")
    )

    response = client.post("/api/command", json={"text": "make it harder"}).json()

    assert ctx.settings.tier == DEFAULT_TIER, "the setting never changed"
    assert response["commentary"] == UNVERIFIED_CLAIM_REPLY


def test_a_settings_change_that_ran_may_be_announced():
    client, _, ctx = make_developed_client(
        AgentResponse(
            text="Difficulty is advanced now.",
            tool_calls=(ToolCall(name="set_difficulty", args={"tier": "advanced"}),),
        )
    )

    response = client.post("/api/command", json={"text": "make it harder"}).json()

    assert ctx.settings.tier == "advanced"
    assert response["commentary"] == "Difficulty is advanced now."


def test_an_unbacked_engine_number_is_guarded():
    """Engine numbers come from the engine. With no analysis in the turn's
    results there is nothing for a score to derive from."""
    client, _, _ = make_developed_client(AgentResponse(text="You're at -3.5 here."))

    response = client.post("/api/command", json={"text": "who's winning?"}).json()

    assert response["commentary"] == UNVERIFIED_CLAIM_REPLY


def test_an_evaluation_the_engine_reported_may_be_quoted():
    ctx = developed(ToolContext(session=GameSession()))
    ctx.engine = FakeEngine(evaluation=Evaluation(score_cp=150, mate_in=None))
    app, _ = scripted_app(
        ctx,
        AgentResponse(
            text="+1.5 for me. Comfortable.",
            tool_calls=(ToolCall(name="evaluate_position", args={}),),
        ),
    )
    client = TestClient(app)

    response = client.post("/api/command", json={"text": "who's winning?"}).json()

    assert response["commentary"] == "+1.5 for me. Comfortable."


def test_an_invented_material_count_is_guarded():
    """The claim class the board answers on its own: nothing has been traded in
    a developed opening, so nobody is up a piece."""
    client, _, ctx = make_developed_client(AgentResponse(text="You're up a piece."))

    response = client.post("/api/command", json={"text": "how's it look?"}).json()

    assert ctx.session.material_balance() == 0, "a level board"
    assert response["commentary"] == UNVERIFIED_CLAIM_REPLY


def test_a_material_count_the_board_backs_survives():
    """The other half of the wiring, and the half a guarded test cannot prove:
    the balance really reaches the facts, so a true count is still sayable."""
    ctx = ToolContext(session=GameSession())
    for san in ("e4", "Nf6", "Nc3", "Nxe4", "Nxe4", "d5"):
        assert ctx.session.submit_move(san).legal
    app, _ = scripted_app(ctx, AgentResponse(text="You're up a knight. Enjoy it."))
    client = TestClient(app)

    response = client.post("/api/command", json={"text": "how's it look?"}).json()

    assert ctx.session.material_balance() == 2, "a knight for a pawn"
    assert response["commentary"] == "You're up a knight. Enjoy it."


def test_the_ending_class_keeps_its_own_correction():
    """The two substitutions are not interchangeable: an invented ending is
    answered by the line that says the game is still live, which is the fact
    the player most needs back."""
    client, _, _ = make_developed_client(AgentResponse(text="Word. Game over."))
    response = client.post("/api/command", json={"text": "i'm done"}).json()
    assert response["commentary"] == UNTRUE_CLAIM_REPLY


# --- The resign route: the player conceding is not the model's call.
#
# Live, "you know what, I give up. I resign" got *"Word. Game over."* with zero
# tool calls on a live board, and the eval measured the path as a coin flip
# (docs/agent-evals.md). An explicit resignation is deterministic text, so the
# pipeline settles it — through the same registry and the same gate, so the road
# stays one road, minus the model.


def test_an_explicit_resignation_calls_resign_without_the_model():
    client, brain, ctx = make_developed_client()
    fen_before = ctx.session.fen()

    response = client.post(
        "/api/command", json={"text": "you know what, I give up. I resign"}
    ).json()

    assert brain.calls == [], "the model does not get a vote on a resignation"
    assert [r["name"] for r in response["tool_results"]] == ["resign"]
    assert ctx.pending is not None and ctx.pending.name == "resign"
    assert ctx.session.fen() == fen_before, "gated: it asks before it ends the game"
    assert response["commentary"]


def test_a_resignation_resigns_the_player_not_the_side_to_move():
    """The player is black and it is white's move: the side to move is only
    coincidentally the player (trace review, finding 8)."""
    ctx = ToolContext(session=GameSession(player_color="black"))
    ctx.session.submit_move("e4")
    ctx.session.submit_move("e5")
    ctx.session.submit_move("Nf3")
    client = TestClient(scripted_app(ctx)[0])

    client.post("/api/command", json={"text": "i resign"})
    client.post("/api/command", json={"text": "yes"})

    assert ctx.session.is_game_over()
    assert ctx.session.outcome().winner == "white", "black (the player) resigned"


def test_a_resignation_confirmed_by_yes_ends_the_game():
    client, _, ctx = make_developed_client()
    client.post("/api/command", json={"text": "i resign"})

    response = client.post("/api/command", json={"text": "yes"}).json()

    assert ctx.session.is_game_over()
    assert response["state"]["game_over"] is True


def test_an_unrelated_command_drops_the_armed_op():
    client, brain, ctx = make_developed_client(destructive("new_game"))
    client.post("/api/command", json={"text": "new game"})

    client.post("/api/command", json={"text": "e5"})  # fast path: a plain move

    assert ctx.pending is None, "changing the subject disarms the op"
    assert ctx.session.fen() != GameSession().fen()


# --- Recovery: the provider dies mid-turn (audit item 20).
#
# The rule: the position stays valid and resumable, a move that landed is never
# silently replayed (or invited to be), and no route 500s after the board
# changed. The brain converts a mid-loop ProviderError into
# stop_reason="provider_error" carrying whatever verifiably ran
# (test_llama_brain.py); the pipeline's half is here — it still closes the
# turn, broadcasts, and says the truth. The two canned lines differ on the one
# fact that matters for retry advice: whether anything changed.


def provider_died(*tool_calls: ToolCall) -> AgentResponse:
    """What the real brain returns when llama-server dies mid-loop: no text,
    the tool results of whatever already ran (filled in by ScriptedBrain's
    dispatcher), and the provider_error stop."""
    return AgentResponse(text="", tool_calls=tool_calls, stop_reason="provider_error")


def test_a_provider_failure_after_the_move_keeps_it_and_closes_the_turn():
    client, _, ctx = observing_client(
        provider_died(ToolCall(name="make_move", args={"move": "e4"}))
    )

    response = client.post("/api/command", json={"text": "push the king pawn"})

    assert response.status_code == 200, "the move landed; this is not a failure"
    body = response.json()
    assert ctx.session.move_history() == ["e4", "e5"], "move kept, reply collected"
    assert body["commentary"] == f"{PROVIDER_LOST_TURN_STANDS}\n\ne5."
    # Resumable: the next turn plays normally, and the landed move was not
    # replayed by anything.
    second = client.post("/api/command", json={"text": "Nf3"}).json()
    assert second["state"]["history"] == ["e4", "e5", "Nf3"]


def test_a_provider_failure_before_anything_ran_invites_a_retry():
    client, _, ctx = observing_client(provider_died())
    fen_before = ctx.session.fen()

    body = client.post("/api/command", json={"text": "how does it look?"}).json()

    assert ctx.session.fen() == fen_before
    assert body["commentary"] == PROVIDER_LOST_RETRY, "nothing ran, so retry is safe"


def test_a_narrator_failure_after_a_confirmed_reset_degrades_to_the_canned_line():
    """The confirmation route's narration runs after the destructive op already
    ran; a provider failure there must cost the words, never a 500 after the
    board changed."""
    client, _, ctx = make_developed_client(
        destructive("new_game"), narrations=(ProviderError("llama-server died"),)
    )
    client.post("/api/command", json={"text": "new game"})

    response = client.post("/api/command", json={"text": "yes"})

    assert response.status_code == 200
    assert response.json()["commentary"] == "New game."
    assert ctx.session.fen() == GameSession().fen(), "the confirmed reset stands"


def test_a_narrator_failure_after_a_confirmed_resign_degrades_the_same_way():
    client, _, ctx = make_developed_client(
        destructive("resign"), narrations=(ProviderError("llama-server died"),)
    )
    client.post("/api/command", json={"text": "i resign"})

    response = client.post("/api/command", json={"text": "yes"})

    assert response.status_code == 200
    assert ctx.session.is_game_over()
    assert response.json()["commentary"].startswith("Game over:")
