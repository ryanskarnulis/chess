"""The typed handoff (#289): the kind derivation, the narrator projection, and
the brief it renders. Pure functions over result dicts — no brain, no app."""

import pytest

from chessapp import api
from chessapp.coordinator import TurnCoordinator
from chessapp.game import GameSession
from chessapp.handoff import (
    NARRATOR_HIDDEN_KEYS,
    READ_TOOLS,
    Entry,
    build,
    narrator_result_view,
    render,
)
from chessapp.tools import ToolContext, build_registry
from fakes import FakeEngine

MOVED = {"name": "make_move", "result": {"ok": True, "legal": True, "san": "e4"}}
ILLEGAL = {
    "name": "make_move",
    "result": {"ok": True, "legal": False, "reason": "illegal", "alternatives": []},
}
REFUSED = {"name": "new_game", "result": {"ok": False, "error": "confirm first"}}
LOOKED = {"name": "evaluate_position", "result": {"ok": True, "score_cp": 20}}
SET = {"name": "set_verbosity", "result": {"ok": True, "verbosity": "high"}}


@pytest.mark.parametrize(
    ("results", "stop", "kind"),
    [
        ([], "completed", "reply"),
        ([], "no_progress", "reply"),
        ([MOVED], "completed", "completed"),
        ([MOVED, SET], "completed", "completed"),
        ([LOOKED], "completed", "completed"),
        ([LOOKED], "no_progress", "completed"),
        ([REFUSED], "completed", "declined"),
        ([ILLEGAL], "completed", "declined"),
        ([LOOKED, REFUSED], "completed", "declined"),
        ([MOVED, REFUSED], "completed", "partial"),
        ([MOVED, ILLEGAL], "completed", "partial"),
        ([MOVED], "no_progress", "partial"),
        ([MOVED], "length", "partial"),
        # #288: a budget ends the phase before the planner said it was done.
        ([MOVED], "budget", "partial"),
        ([MOVED], "max_iterations", "partial"),
        ([MOVED], "correction_limit", "partial"),
        ([LOOKED], "budget", "completed"),
    ],
)
def test_the_kind_comes_from_the_results_and_the_stop_alone(results, stop, kind):
    assert build(results, stop, note="I did everything you asked").kind == kind


def test_the_entries_point_at_their_results():
    handoff = build([LOOKED, MOVED, REFUSED])

    assert handoff.consulted == (Entry(1, "evaluate_position"),)
    assert handoff.performed == (Entry(2, "make_move"),)
    assert handoff.refused == (Entry(3, "new_game", "confirm first"),)


def test_an_illegal_move_is_refused_with_its_reason():
    assert build([ILLEGAL]).refused == (Entry(1, "make_move", "illegal"),)


def test_the_note_never_decides_the_kind():
    lying = build([], note="Undid your last move and played d4.")
    assert lying.kind == "reply"
    assert lying.performed == ()


def test_every_registered_tool_is_classified_as_a_read_or_not():
    """`READ_TOOLS` names the reads; everything else is an action. A read
    that is not listed would be reported as done, so the list must hold every
    registered read — pinned against the registry's own tools."""
    registry = build_registry(ToolContext(session=GameSession(), engine=FakeEngine()))
    names = {d["function"]["name"] for d in registry.definitions()}
    assert READ_TOOLS <= names, "a read that no longer exists"
    actions = names - READ_TOOLS
    assert actions == {
        "make_move",
        "undo",
        "new_game",
        "resign",
        "claim_draw",
        "offer_draw",
        "save_game",
        "resume_game",
        "set_difficulty",
        "set_verbosity",
        "set_voice_output",
    }


def test_the_projection_hides_what_the_state_view_hides():
    """One list of what a narrator never reads: in the facts every narration
    is handed, and in every result it reads (#193, #289, #369)."""
    ctx = ToolContext(session=GameSession(), engine=FakeEngine())
    facts = api.narrator_facts(ctx, TurnCoordinator(ctx))
    assert set(NARRATOR_HIDDEN_KEYS) <= set(api._agent_state_dict(ctx))
    assert not set(facts) & set(NARRATOR_HIDDEN_KEYS)


def test_the_projection_drops_the_side_to_move_and_keeps_the_rest():
    undo = {
        "name": "undo",
        "result": {
            "ok": True,
            "undone": ["e5", "e4"],
            "engine_move": None,
            "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "turn": "white",
        },
    }

    view = narrator_result_view(undo)

    assert view == {
        "name": "undo",
        "result": {"ok": True, "undone": ["e5", "e4"], "engine_move": None},
    }
    assert "fen" in undo["result"], "the record itself is untouched"


def test_a_turn_with_no_tools_says_nothing_was_done():
    brief = render(build([], note="undid it"), "undo that", [])

    assert "No tool was called this turn." in brief
    assert "Done this turn: nothing." in brief
    assert (
        "The planner's reading of what the player wants (not a record of what "
        "happened):\nundid it" in brief
    )


def test_the_brief_lists_results_by_id_and_sorts_them():
    results = [LOOKED, MOVED, REFUSED]
    brief = render(build(results), "eval, e4, and reset", results)

    assert '#1 {"name": "evaluate_position"' in brief
    assert "Done this turn: #2 make_move." in brief
    assert "Refused: #3 new_game (confirm first)." in brief
    assert "Looked up: #1 evaluate_position." in brief


def test_the_brief_never_carries_a_side_to_move():
    undo = {"name": "undo", "result": {"ok": True, "fen": "x w - - 0 1", "turn": "w"}}
    brief = render(build([undo]), "undo", [undo])

    assert '"fen"' not in brief and '"turn"' not in brief


REPLY = {"san": "Nf6", "capture": "knight", "check": True}


def test_the_engines_reply_is_the_narrators_to_say():
    """#365: the narrator speaks after the reply is on the board, is handed
    it, and is told the player learns it from what he says — the exact line,
    pinned, because it is the whole of the prompt change."""
    brief = render(build([MOVED], engine_reply=REPLY), "e4", [MOVED])
    assert (
        "Your reply, already on the board: Nf6, taking their knight, check. "
        "The player learns your move only from what you say; say it however "
        "you like."
    ) in brief
    assert "Your reply" not in render(build([MOVED]), "e4", [MOVED])
    assert "and tell them your move, Nf6, your way." in brief


def test_a_turn_with_no_reply_keeps_the_closing_sentence_as_it_was():
    """Pinned so a turn with no move — the most common brain turn — measures
    the prompt it always did."""
    brief = render(build([MOVED]), "e4", [MOVED])
    assert brief.endswith(
        "Reply to the player in character. Say only what the record above "
        "shows was done; if it shows nothing done, do not say anything was. "
        "When the player has to choose, ask them, naming the options."
    )


def test_a_quiet_reply_is_just_the_move():
    quiet = {"san": "e5", "capture": None, "check": False}
    brief = render(build([MOVED], engine_reply=quiet), "e4", [MOVED])
    assert "Your reply, already on the board: e5. " in brief


def test_a_reply_the_engine_died_on_is_said_never_to_have_come():
    brief = render(build([MOVED], reply_owed=True), "e4", [MOVED])
    assert "Your reply to the player's move never came" in brief
    assert "already on the board" not in brief


def test_the_facts_ride_along_and_an_empty_note_leaves_its_heading_out():
    facts = {"player_color": "white", "game_over": False}
    brief = render(build([MOVED], facts=facts), "e4", [MOVED])

    assert 'The game now:\n{"player_color": "white", "game_over": false}' in brief
    assert "planner's reading" not in brief


def test_the_trace_names_tools_and_not_results():
    handoff = build([LOOKED, MOVED, REFUSED], engine_reply=REPLY)
    assert handoff.trace() == {
        "kind": "partial",
        "performed": ["make_move"],
        "refused": ["new_game"],
        "consulted": ["evaluate_position"],
        "reply_owed": False,
        "engine_reply": "Nf6",
        "candidates": [],
    }


# --- the typed clarification (#289, PR 2) ------------------------------------

ASKED = {"name": "ask_player", "result": {"ok": True, "candidates": ["Nf3", "Nh3"]}}
ASK_REFUSED = {
    "name": "ask_player",
    "result": {"ok": False, "error": "not legal here: Qh5", "retry": "different_args"},
}


def test_an_ask_that_landed_is_a_clarification_with_its_candidates():
    handoff = build([ASKED], note="asked which knight")

    assert handoff.kind == "clarify"
    assert handoff.candidates == ("Nf3", "Nh3")
    assert handoff.performed == () and handoff.consulted == ()


def test_a_refused_ask_is_a_refusal_and_not_a_clarification():
    handoff = build([ASK_REFUSED])

    assert handoff.kind == "declined"
    assert handoff.candidates == ()


def test_the_clarification_brief_names_every_candidate():
    brief = render(build([ASKED]), "move my kings knight", [ASKED])

    assert "Done this turn: nothing." in brief
    assert "The player has to choose between: Nf3, Nh3." in brief
    assert "naming each" in brief


def test_the_split_registry_classifies_ask_player_apart():
    registry = build_registry(
        ToolContext(session=GameSession(), engine=FakeEngine()), atomic_exchange=False
    )
    names = {d["function"]["name"] for d in registry.definitions()}
    assert "ask_player" in names and "ask_player" not in READ_TOOLS
