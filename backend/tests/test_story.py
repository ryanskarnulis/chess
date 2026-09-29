"""The story of the game: notes, the summarizer call, the keeper (#372, PR 2).

Trace-only in this PR: the story is written off the turns' path and recorded,
and nothing reads it. So the tests here pin what goes *in* (code-written notes
from the ledger, Glitch's words labelled and never the app's), how the one call
is made, that nothing about it can reach or delay a turn, and that the offline
scorer reads a third-person story against the evidence it covered.
"""

import json
import threading
from typing import Any

from fastapi.testclient import TestClient

from chessapp.agent_api import ConversationStore
from chessapp.brain import (
    CALL_FAILED,
    CALL_OK,
    CALL_TRUNCATED,
    AgentResponse,
    ModelCall,
    ToolCall,
)
from chessapp.game import GameSession
from chessapp.ledger import Ledger
from chessapp.personality import SUMMARIZER_PROMPT
from chessapp.provider import ProviderError, ProviderFailure
from chessapp.speech_accuracy import score_story, story_reading, story_tally
from chessapp.story import (
    STORY_MAX_TOKENS,
    ChatSummarizer,
    StoryKeeper,
    StoryState,
    Summary,
    TurnNote,
    accumulate,
    render_events,
    render_note,
    render_request,
)
from chessapp.tools import LIVE_CHECKPOINT_FILENAME, Settings, ToolContext
from fakes import CollectedTurns, FakeEngine, ScriptedProvider, scripted_app, text_turn

SETTINGS = Settings().snapshot()


class FakeSummarizer:
    """Writes a story that lists what it was given; can be made to fail or to
    block until released."""

    def __init__(self, *, fail: bool = False, gate: threading.Event | None = None):
        self.fail = fail
        self.gate = gate
        self.calls: list[tuple[str, list[TurnNote], int]] = []
        self.entered = threading.Event()

    def summarize(self, previous, notes, first):
        self.calls.append((previous, list(notes), first))
        self.entered.set()
        if self.gate is not None:
            self.gate.wait(5)
        if self.fail:
            return Summary(None, (ModelCall("summarizer", CALL_FAILED, 3),))
        told = " ".join(n.words or "(board)" for n in notes)
        return Summary(
            f"{previous} | {told}".strip(" |"), (ModelCall("summarizer", CALL_OK, 5),)
        )


def _note(seq: int, words: str | None = "hi", **kwargs: Any) -> TurnNote:
    fields = dict(
        seq=seq,
        correlation_id=f"c{seq}",
        turn_id=seq,
        route="brain",
        words=words,
        tools=[],
        draft="",
        events=[],
    )
    fields.update(kwargs)
    return TurnNote(**fields)


def _watched(*sans: str) -> tuple[GameSession, Ledger]:
    session, ledger = GameSession(), Ledger()
    ledger.observe(session, SETTINGS)
    for san in sans:
        session.submit_move(san)
        ledger.observe(session, SETTINGS)
    return session, ledger


# --- what the summarizer reads --------------------------------------------------------


def test_events_read_as_plain_sentences():
    session, ledger = _watched("e4", "d5", "exd5")
    session.undo(1)
    ledger.observe(session, {**SETTINGS})
    ledger.observe(session, {**SETTINGS, "verbosity": "low"})
    lines = render_events([e.to_dict() for e in ledger.current()])
    assert lines == [
        "a new game began; the player has white",
        "1. e4 by the player",
        "1... d5 by Glitch",
        "2. exd5 by the player, taking a pawn (the player up 1)",
        "taken back: exd5 (1 move)",
        "verbosity changed from normal to low",
    ]


def test_a_resumed_line_is_one_line_of_moves():
    events = [
        {"kind": "resumed", "name": "keep", "player_color": "white"},
        {
            "kind": "move",
            "san": "e4",
            "color": "white",
            "move_number": 1,
            "restored": True,
        },
        {
            "kind": "move",
            "san": "e5",
            "color": "black",
            "move_number": 1,
            "restored": True,
        },
    ]
    assert render_events(events) == [
        "the saved game 'keep' was resumed; the player has white",
        "its moves so far: 1. e4 e5",
    ]


def test_draw_offers_and_endings_read_from_the_players_side():
    lines = render_events(
        [
            {"kind": "draw_offer", "accepted": False, "reason": "engine_ahead"},
            {
                "kind": "game_end",
                "termination": "checkmate",
                "winner": "opponent",
                "result": "0-1",
            },
        ]
    )
    assert lines == [
        "the player offered a draw, and Glitch declined (engine ahead)",
        "the game ended by checkmate: Glitch won (0-1)",
    ]


def test_glitch_is_labelled_as_what_he_said_and_silence_is_said():
    spoke = render_note(_note(0, "push the e pawn", draft="Bet. e5 back at you."), 1)
    assert 'The player said: "push the e pawn"' in spoke
    assert "Glitch said (his words only" in spoke and '"Bet. e5 back at you."' in spoke
    assert "What happened: nothing on the board." in spoke
    silent = render_note(_note(1, None), 2)
    assert "The player moved a piece on the board." in silent
    assert "Glitch said nothing." in silent


def test_tools_read_as_what_they_did():
    note = _note(
        0,
        tools=[
            {
                "name": "make_move",
                "args": {"move": "e4"},
                "result": {"legal": True, "san": "e4"},
            },
            {
                "name": "save_game",
                "args": {"name": "x"},
                "result": {"ok": False, "error": "no dir"},
            },
            {
                "name": "ask_player",
                "args": {"piece": "knight"},
                "result": {"ok": True, "candidates": ["Nf3", "Nh3"]},
            },
        ],
    )
    text = render_note(note, 1)
    assert "- make_move(move=e4): played e4" in text
    assert "- save_game(name=x): refused — no dir" in text
    assert "asked the player to choose between Nf3, Nh3" in text


def test_the_request_numbers_turns_and_starts_an_empty_story():
    request = render_request("", [_note(4), _note(5)], 5)
    assert request.startswith("The story so far:\n(nothing yet")
    assert "Turn 5" in request and "Turn 6" in request


# --- the call -------------------------------------------------------------------------


def test_the_summarizer_is_one_tool_free_cool_capped_call():
    provider = ScriptedProvider(text_turn("The player opened 1. e4."))
    summary = ChatSummarizer(provider).summarize("", [_note(0)], 1)
    assert summary.text == "The player opened 1. e4."
    assert summary.call.phase == "summarizer" and summary.call.status == CALL_OK
    [call] = provider.calls
    assert call["messages"][0] == {"role": "system", "content": SUMMARIZER_PROMPT}
    assert call["tools"] is None and call["enable_thinking"] is False
    assert call["max_tokens"] == STORY_MAX_TOKENS
    assert call["temperature"] == 0.3


def test_a_story_too_long_is_asked_for_again_shorter_once():
    provider = ScriptedProvider(
        text_turn("The pla", finish_reason="length"), text_turn("Short story.")
    )
    summary = ChatSummarizer(provider).summarize("", [_note(0)], 1)
    assert summary.text == "Short story."
    assert [c.status for c in summary.calls] == [CALL_TRUNCATED, CALL_OK]
    assert "shorter" in provider.calls[1]["messages"][1]["content"]


def test_a_truncated_or_dead_call_writes_no_story():
    cut = ChatSummarizer(ScriptedProvider(text_turn("The pla", finish_reason="length")))
    summary = cut.summarize("", [_note(0)], 1)
    assert summary.text is None
    assert [c.status for c in summary.calls] == [CALL_TRUNCATED, CALL_TRUNCATED]
    dead = ChatSummarizer(
        ScriptedProvider(ProviderError("down", failure=ProviderFailure.UNREACHABLE))
    )
    summary = dead.summarize("", [_note(0)], 1)
    assert summary.text is None and summary.call.status == CALL_FAILED


# --- the keeper -----------------------------------------------------------------------


def _keeper(summarizer, *, state=None, ledger=None, records=None):
    state = state or StoryState()
    ledger = ledger or _watched()[1]
    records = records if records is not None else []
    keeper = StoryKeeper(
        summarizer,
        states=lambda origin: state if origin == "panel" else None,
        ledger=lambda: ledger,
        record=records.append,
    )
    return keeper, state, records


def _enqueue(keeper, words="hi", draft=""):
    keeper.enqueue(
        "panel",
        correlation_id="c",
        turn_id=1,
        route="brain",
        words=words,
        tools=[],
        draft=draft,
    )


def test_a_note_is_told_off_the_turns_path_and_recorded():
    session, ledger = _watched("e4", "e5")
    keeper, state, records = _keeper(FakeSummarizer(), ledger=ledger)
    _enqueue(keeper, "play e4")
    assert keeper.wait_idle(5)
    assert state.text == "play e4"
    assert state.covered_through == 0 and state.pending == []
    [record] = records
    assert record["status"] == CALL_OK
    assert [e["kind"] for e in record["notes"][0]["events"]] == [
        "new_game",
        "move",
        "move",
    ]
    assert record["evidence"]["by_player"] == ["e4"]
    assert record["evidence"]["by_opponent"] == ["e5"]


def test_each_note_carries_only_the_events_since_the_last():
    session, ledger = _watched("e4")
    keeper, state, _ = _keeper(FakeSummarizer(), ledger=ledger)
    _enqueue(keeper)
    session.submit_move("e5")
    ledger.observe(session, SETTINGS)
    _enqueue(keeper)
    keeper.wait_idle(5)
    notes = [n for (_, batch, _) in keeper._summarizer.calls for n in batch]
    assert [len(n.events) for n in notes] == [2, 1]


def test_a_backlog_is_caught_up_in_one_call_and_the_wait_is_measured():
    gate = threading.Event()
    summarizer = FakeSummarizer(gate=gate)
    keeper, state, records = _keeper(summarizer)
    _enqueue(keeper, "one")
    # Held inside its first call, so the next two pile up behind it.
    assert summarizer.entered.wait(5)
    _enqueue(keeper, "two")
    _enqueue(keeper, "three")
    # A turn begins while the story is behind: it is remembered.
    started = keeper.turn_started("panel", "waiting-turn")
    assert started["pending"] == 3
    gate.set()
    assert keeper.wait_idle(5)
    assert state.text == "one | two three"
    assert [len(batch) for _, batch, _ in summarizer.calls] == [1, 2]
    [waited] = [w for r in records for w in r["waited"]]
    assert waited["correlation_id"] == "waiting-turn" and waited["lag_ms"] >= 0


def test_a_caught_up_story_makes_a_turn_wait_for_nothing():
    keeper, _, _ = _keeper(FakeSummarizer())
    _enqueue(keeper)
    keeper.wait_idle(5)
    assert keeper.turn_started("panel", "next") == {"pending": 0, "covered_through": 0}


def test_a_failed_call_keeps_the_notes_and_the_old_story():
    summarizer = FakeSummarizer(fail=True)
    keeper, state, records = _keeper(summarizer)
    state.text = "Earlier."
    _enqueue(keeper)
    keeper.wait_idle(5)
    assert state.text == "Earlier." and len(state.pending) == 1
    assert records[0]["story"] is None and records[0]["status"] == CALL_FAILED
    summarizer.fail = False
    _enqueue(keeper, "again")
    keeper.wait_idle(5)
    assert state.pending == [] and state.text.endswith("hi again")


def test_a_summarizer_that_raises_never_reaches_the_turn():
    class Boom:
        def summarize(self, *args):
            raise RuntimeError("boom")

    keeper, state, _ = _keeper(Boom())
    _enqueue(keeper)
    assert keeper.wait_idle(5)
    assert len(state.pending) == 1


def test_a_conversation_with_no_story_is_left_alone():
    keeper, _, records = _keeper(FakeSummarizer())
    keeper.enqueue(
        "mcp",
        correlation_id="c",
        turn_id=0,
        route="brain",
        words="x",
        tools=[],
        draft="",
    )
    assert keeper.turn_started("mcp", "c") == {"pending": 0}
    keeper.wait_idle(5)
    assert records == []


def test_the_state_round_trips_and_a_bad_one_is_empty():
    state = StoryState(cursor=4)
    state.text, state.pending = "So far.", [_note(0, draft="Word.")]
    back = StoryState.from_dict(json.loads(json.dumps(state.to_dict())))
    assert back.text == "So far." and back.cursor == 4
    assert back.pending[0].draft == "Word."
    assert StoryState.from_dict({"text": 3}, cursor=2).cursor == 2
    assert StoryState.from_dict("nope").text == ""


# --- the evidence and the scorer ------------------------------------------------------


def test_evidence_is_historical_and_game_spanning():
    session, ledger = _watched("f3", "e5", "g4", "Qh4#")
    evidence = accumulate({}, _note(0, events=[e.to_dict() for e in ledger.current()]))
    session.new_game()
    ledger.observe(session, SETTINGS)
    evidence = accumulate(evidence, _note(1, events=[ledger.current()[0].to_dict()]))
    assert evidence["ended"] is True and evidence["winner"] == "opponent"
    assert evidence["restarted"] is True
    assert "Qh4#" in evidence["by_opponent"]


def test_the_story_is_read_in_the_readings_own_persons():
    assert story_reading("The player played Nf3 and Glitch's knight took it.") == (
        "you played Nf3 and my knight took it."
    )


def _story_record(story: str, *sans: str, undo: bool = False) -> dict:
    session, ledger = _watched(*sans)
    if undo:
        session.undo(1)
        ledger.observe(session, SETTINGS)
    evidence = accumulate({}, _note(0, events=[e.to_dict() for e in ledger.current()]))
    return {"kind": "story", "story": story, "evidence": evidence}


def test_move_numbers_do_not_split_a_move_from_its_sentence():
    assert story_reading("Glitch took a pawn with 6... exf4.") == (
        "I took a pawn with exf4."
    )


def test_a_true_story_is_backed():
    record = _story_record(
        "The player played e4 and Glitch played d5. The player took on d5 with exd5.",
        "e4",
        "d5",
        "exd5",
    )
    score = score_story(record)
    assert score is not None and score.claims
    assert all(claim.backed for claim in score.claims)


def test_a_false_story_is_caught():
    record = _story_record(
        "Glitch played Nc6. Then the player took back the last move.", "e4", "e5"
    )
    unbacked = {c.claim for c in score_story(record).claims if not c.backed}
    assert {"owned_move", "takeback"} <= unbacked


def test_the_story_tally_counts_stories_not_turns():
    records = [
        _story_record("The player played e4.", "e4"),
        {"kind": "turn", "draft": "x"},
        {"kind": "story", "story": None},
    ]
    assert story_tally(records).turns == 1


# --- the app --------------------------------------------------------------------------


def _app(ctx, *responses, summarizer=None, tracer=None):
    app, brain = scripted_app(
        ctx, *responses, summarizer=summarizer or FakeSummarizer(), tracer=tracer
    )
    return TestClient(app), brain


def test_a_turn_is_noted_with_glitchs_words_and_never_the_apps():
    ctx = ToolContext(session=GameSession(), engine=FakeEngine())
    ctx.settings.verbosity = "low"  # the app says "e4. e5." and Glitch says nothing
    tracer = CollectedTurns()
    summarizer = FakeSummarizer()
    client, _ = _app(ctx, summarizer=summarizer, tracer=tracer)
    body = client.post("/api/command", json={"text": "e4"}).json()
    assert body["commentary"], "the app spoke"
    _wait_for(lambda: any(r.get("kind") == "story" for r in tracer.records))
    [story] = [r for r in tracer.records if r.get("kind") == "story"]
    [note] = story["notes"]
    assert note["draft"] == "" and note["words"] == "e4"
    moves = [e["san"] for e in note["events"] if e["kind"] == "move"]
    assert moves == ["e4", "e5"], "the engine's reply is in the turn's note"
    [turn] = [r for r in tracer.records if r.get("kind", "turn") == "turn"]
    assert turn["story"] == {"pending": 0, "covered_through": -1}


def test_a_brain_turn_notes_its_tools_and_draft():
    ctx = ToolContext(session=GameSession(), engine=FakeEngine())
    tracer = CollectedTurns()
    client, _ = _app(
        ctx,
        AgentResponse(
            text="Bet, e4. I answer e5.",
            tool_calls=(ToolCall("make_move", {"move": "e4"}),),
            stop_reason="completed",
        ),
        tracer=tracer,
    )
    client.post("/api/command", json={"text": "push the king pawn"})
    _wait_for(lambda: any(r.get("kind") == "story" for r in tracer.records))
    [story] = [r for r in tracer.records if r.get("kind") == "story"]
    [note] = story["notes"]
    assert note["draft"] == "Bet, e4. I answer e5."
    assert note["tools"][0]["name"] == "make_move"
    assert "legal_moves" not in note["tools"][0]["result"]


def test_the_panel_story_survives_a_restart(tmp_path):
    ctx = ToolContext(session=GameSession(), engine=FakeEngine(), save_dir=tmp_path)
    ctx.settings.verbosity = "low"
    gate = threading.Event()  # held, so the note is still pending at the checkpoint
    client, _ = _app(ctx, summarizer=FakeSummarizer(gate=gate))
    client.post("/api/command", json={"text": "e4"})
    checkpoint = json.loads((tmp_path / LIVE_CHECKPOINT_FILENAME).read_text())
    gate.set()
    assert len(checkpoint["story"]["pending"]) == 1

    fresh = ToolContext(session=GameSession(), engine=FakeEngine(), save_dir=tmp_path)
    from chessapp.tools import restore_live_checkpoint

    assert restore_live_checkpoint(fresh)
    summarizer = FakeSummarizer()
    _app(fresh, summarizer=summarizer)  # a restart with a backlog catches up
    _wait_for(lambda: fresh.story.pending == [])
    assert fresh.story.text == "e4"


def test_a_delegate_thread_keeps_its_own_story(tmp_path):
    store = ConversationStore(tmp_path / "c.json")
    conversation = store.create()
    conversation.story = StoryState(cursor=0)
    conversation.story.text = "The player asked for a hint."
    store.append_user_message(conversation, "hi")
    reread = ConversationStore(tmp_path / "c.json").get(conversation.id)
    assert reread.story is not None
    assert reread.story.text == "The player asked for a hint."


def test_a_resumed_save_brings_its_story_back(tmp_path):
    from chessapp.coordinator import TurnCoordinator
    from chessapp.tools import build_registry

    ctx = ToolContext(session=GameSession(), engine=FakeEngine(), save_dir=tmp_path)
    registry = build_registry(ctx, TurnCoordinator(ctx), atomic_exchange=False)
    ctx.story.text = "The player opened with e4."
    registry.dispatch("save_game", {"name": "keep"})
    ctx.story = StoryState()
    assert registry.dispatch("resume_game", {"name": "keep"})["ok"] is True
    assert ctx.story.text == "The player opened with e4."
    assert ctx.story.cursor <= ctx.ledger.current()[0].seq


def test_the_app_without_a_summarizer_keeps_no_story():
    ctx = ToolContext(session=GameSession(), engine=FakeEngine())
    tracer = CollectedTurns()
    app, _ = scripted_app(ctx, tracer=tracer)
    TestClient(app).post("/api/game/move", json={"move": "e4"})
    assert ctx.story.pending == [] and ctx.story.text == ""
    assert all(r.get("story") is None for r in tracer.records)


def _wait_for(condition, timeout: float = 5.0) -> None:
    done = threading.Event()
    deadline = threading.Timer(timeout, done.set)
    deadline.start()
    try:
        while not done.is_set():
            if condition():
                return
            done.wait(0.01)
        raise AssertionError("condition never held")
    finally:
        deadline.cancel()


# A labelled corpus for the story's reading: lines a story of this game
# (1. e4 d5 2. exd5 Qxd5, then 2... Qxd5 taken back, verbosity to low) may say,
# and lines it may not. The reading is the narrator's; these pin that the
# third-person map lands it on the right side.
_TRUE_LINES = (
    "The player opened with e4 and Glitch answered d5.",
    "The player took on d5 with exd5, winning a pawn.",
    "After 1. e4 d5, the player captured a pawn with 2. exd5.",
    "Glitch recaptured a pawn with 2... Qxd5.",
    "Glitch played Qxd5, and the player took the last move back.",
    "The player took back Glitch's queen move.",
    "The player asked for less talking, and verbosity went to low.",
)
_FALSE_LINES = (
    "Glitch played Nf6.",
    "The player played Bb5.",
    "Glitch took the player's knight.",
    "The player started a fresh game.",
    "The player saved the game as testgame.",
)


def _corpus_record(story: str) -> dict:
    session, ledger = _watched("e4", "d5", "exd5", "Qxd5")
    session.undo(1)
    ledger.observe(session, SETTINGS)
    ledger.observe(session, {**SETTINGS, "verbosity": "low"})
    evidence = accumulate({}, _note(0, events=[e.to_dict() for e in ledger.current()]))
    return {"kind": "story", "story": story, "evidence": evidence}


def test_true_story_lines_are_backed():
    for line in _TRUE_LINES:
        score = score_story(_corpus_record(line))
        unbacked = [c for c in score.claims if not c.backed]
        assert unbacked == [], (line, unbacked)


def test_false_story_lines_are_caught():
    for line in _FALSE_LINES:
        score = score_story(_corpus_record(line))
        assert any(not c.backed for c in score.claims), (line, score.claims)
