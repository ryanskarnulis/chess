"""The open clarification's pure half (#319, `clarification.py`).

What a question is bound to, when it stops standing, and what closes it —
decided from the record, the board version, the game id and the turn's tool
results. Nothing here reads language, and nothing here plays a move.
"""

from chessapp import clarification
from chessapp.clarification import (
    ANSWERED,
    BOARD_CHANGED,
    GAME_CHANGED,
    INVALIDATED,
    OTHER_MOVE,
    SUPERSEDED,
    TURN_CHANGED_BOARD,
    ask,
    settle,
    staleness,
)
from chessapp.conversation import Transcript
from chessapp.coordinator import TurnCoordinator
from chessapp.game import GameSession
from chessapp.tools import (
    ToolContext,
    build_registry,
    live_checkpoint,
    restore_live_checkpoint,
    write_live_checkpoint,
)
from fakes import FakeEngine


def _question(**overrides) -> clarification.Clarification:
    fields = {
        "origin": "panel",
        "game_id": "g1",
        "board_version": 7,
        "request": "move my kings knight",
        "candidates": ["Nf3", "Nh3"],
    } | overrides
    return ask(**fields)


def _move(san: str, legal: bool = True) -> dict:
    return {"name": "make_move", "result": {"ok": True, "legal": legal, "san": san}}


# --- the record ------------------------------------------------------------------


def test_a_question_keeps_what_it_was_asked_about():
    question = _question(candidates=["Nf3", "Nh3", "Nf3"])

    assert question.origin == "panel"
    assert question.game_id == "g1"
    assert question.board_version == 7
    assert question.request == "move my kings knight"
    assert question.candidates == ("Nf3", "Nh3")  # deduped, order kept
    assert question.trace() == {
        "id": question.id,
        "origin": "panel",
        "game_id": "g1",
        "board_version": 7,
        "candidates": ["Nf3", "Nh3"],
    }


def test_every_question_has_its_own_id():
    assert _question().id != _question().id


# --- staleness -------------------------------------------------------------------


def test_a_question_stands_on_its_own_game_and_board():
    assert staleness(_question(), game_id="g1", board_version=7) == ""


def test_a_moved_board_ends_the_question():
    assert staleness(_question(), game_id="g1", board_version=8) == BOARD_CHANGED


def test_a_different_game_is_named_before_a_different_board():
    """A new game or a resume moves the version too; "a different game" is
    the more exact reason, so it wins."""
    assert staleness(_question(), game_id="g2", board_version=8) == GAME_CHANGED
    assert staleness(_question(), game_id="g2", board_version=7) == GAME_CHANGED


# --- settle ----------------------------------------------------------------------


def test_a_candidate_played_answers_the_question():
    closed = settle(_question(), [_move("Nf3")])

    assert (closed.status, closed.reason, closed.move) == (ANSWERED, "", "Nf3")
    assert closed.trace()["status"] == ANSWERED


def test_a_move_outside_the_candidates_supersedes_it():
    closed = settle(_question(), [_move("e4")])

    assert (closed.status, closed.reason, closed.move) == (SUPERSEDED, OTHER_MOVE, "e4")


def test_reads_settings_and_refusals_before_the_move_do_not_count():
    """Only the first *board change* is the answer: a look at the board, a
    setting, a declined draw offer or a refused move in front of it moved
    nothing."""
    results = [
        {"name": "get_legal_moves", "result": {"ok": True}},
        {"name": "set_verbosity", "result": {"ok": True, "verbosity": "low"}},
        {"name": "offer_draw", "result": {"ok": True, "accepted": False}},
        {"name": "undo", "result": {"ok": False, "error": "nothing to undo"}},
        _move("Nd4", legal=False),
        _move("Nh3"),
    ]

    closed = settle(_question(), results)

    assert (closed.status, closed.move) == (ANSWERED, "Nh3")


def test_a_candidate_played_after_an_undo_is_not_an_answer():
    """The undo changed the position first; a candidate played on the board
    it left is a move on a different position, not the player's choice
    between the two they were asked about."""
    results = [
        {"name": "undo", "result": {"ok": True, "undone": ["e4", "e5"]}},
        _move("Nf3"),
    ]

    closed = settle(_question(), results)

    assert (closed.status, closed.reason) == (SUPERSEDED, TURN_CHANGED_BOARD)


def test_a_change_with_no_move_supersedes_it():
    closed = settle(_question(), [{"name": "new_game", "result": {"ok": True}}])

    assert (closed.status, closed.reason, closed.move) == (
        SUPERSEDED,
        TURN_CHANGED_BOARD,
        "",
    )


def test_every_registered_tool_is_classified_for_settle():
    """A tool added later lands on the "changes the board" side unless it is
    named neutral — which supersedes a question rather than answering it. This
    pins the neutral list to real tool names, and the rest to the tools that
    really do move the board, so a new tool is a decision, not an accident."""
    ctx = ToolContext(session=GameSession(), engine=FakeEngine())
    registry = build_registry(ctx, TurnCoordinator(ctx), atomic_exchange=False)
    names = {d["function"]["name"] for d in registry.definitions()}
    neutral = clarification._BOARD_NEUTRAL

    assert neutral <= names
    assert names - neutral == {
        "make_move",
        "undo",
        "new_game",
        "resume_game",
        "resign",
        "claim_draw",
        "offer_draw",
    }


# --- on the context --------------------------------------------------------------


def _ctx_with_question(origin: str = "panel") -> ToolContext:
    ctx = ToolContext(session=GameSession(), engine=FakeEngine())
    ctx.clarifications[origin] = ask(
        origin=origin,
        game_id=ctx.session.game_id,
        board_version=ctx.board_version,
        request="move my kings knight",
        candidates=["Nf3", "Nh3"],
    )
    return ctx


def test_a_standing_question_is_read_live_and_kept():
    ctx = _ctx_with_question()
    question = ctx.clarifications["panel"]

    assert ctx.live_clarification("panel") == (question, None)
    assert ctx.clarifications["panel"] is question


def test_another_origin_reads_nothing_and_leaves_it_alone():
    ctx = _ctx_with_question("delegate:1")

    assert ctx.live_clarification("delegate:2") == (None, None)
    assert ctx.live_clarification("panel") == (None, None)
    assert "delegate:1" in ctx.clarifications


def test_a_stale_question_is_reported_once_then_gone():
    ctx = _ctx_with_question()
    question = ctx.clarifications["panel"]
    ctx.session.submit_move("e4")

    live, expired = ctx.live_clarification("panel")

    assert live is None
    assert expired is not None
    assert (expired.record, expired.status, expired.reason) == (
        question,
        INVALIDATED,
        BOARD_CHANGED,
    )
    assert ctx.live_clarification("panel") == (None, None)


def test_a_resumed_game_invalidates_it_as_a_different_game():
    ctx = _ctx_with_question()
    ctx.replace_session(GameSession(), Transcript())

    _, expired = ctx.live_clarification("panel")

    assert expired is not None and expired.reason == GAME_CHANGED


def test_a_restart_discards_the_question(tmp_path):
    """Restart policy: discard safely. The checkpoint does not carry it, and a
    question stamped before the restart would read stale on the board after
    it anyway, because the restore bumps the version past the checkpoint."""
    ctx = _ctx_with_question()
    ctx.save_dir = tmp_path
    question = ctx.clarifications["panel"]
    data = live_checkpoint(ctx)
    write_live_checkpoint(ctx, data)

    assert "clarifications" not in data
    restarted = ToolContext(session=GameSession(), save_dir=tmp_path)
    assert restore_live_checkpoint(restarted)
    assert restarted.clarifications == {}
    assert (
        staleness(
            question,
            game_id=restarted.session.game_id,
            board_version=restarted.board_version,
        )
        == BOARD_CHANGED
    )
