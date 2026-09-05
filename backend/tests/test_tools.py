"""Tool layer: registry + validated dispatch + read tools.

This is the boundary the LLM talks through. Dispatch must be un-crashable:
unknown tools, malformed args, and domain errors all come back as
`{"ok": False, "error": ...}` — never an exception, never corrupted state.
Analysis tools need a live stockfish and skip without one (CI installs it).
"""

import json
import shutil
from pathlib import Path

import pytest

from chessapp.coordinator import TurnCoordinator, TurnPhase
from chessapp.engine import DEFAULT_TIER
from chessapp.game import GameSession
from chessapp.tools import (
    BOARD_STATE_TOOLS,
    GAME_SAVE_DIRNAME,
    Settings,
    Tool,
    ToolContext,
    ToolRegistry,
    brain_tool_exclusions,
    build_registry,
    confirm_pending,
    saved_game_names,
)
from fakes import FakeEngine

requires_stockfish = pytest.mark.skipif(
    shutil.which("stockfish") is None, reason="stockfish binary not installed"
)

# White to move, Qxf7# available (scholar's mate pattern).
WHITE_MATE_IN_1 = "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 0 1"

# A fifty-move claim standing on a position nobody played into: the halfmove
# clock came in with the FEN, so the confirmation gate has no player investment
# to guard and a claim runs straight through it (`_player_has_moved`). That is
# what makes it the setup for testing everything *past* the gate.
FIFTY_MOVE_FEN = "8/8/8/4k3/8/8/4K3/6R1 w - - 100 80"

REPETITION = ("Nf3", "Nf6", "Ng1", "Ng8")


@pytest.fixture
def session():
    return GameSession()


@pytest.fixture
def registry(session):
    return build_registry(ToolContext(session=session))


@pytest.fixture(scope="module")
def live_engine():
    if shutil.which("stockfish") is None:
        pytest.skip("stockfish binary not installed")
    from chessapp.engine import EnginePlayer

    with EnginePlayer() as player:
        yield player


# --- registry ---------------------------------------------------------------


def test_registry_lists_all_read_tools(registry):
    names = {d["function"]["name"] for d in registry.definitions()}
    assert names >= {
        "get_board_state",
        "get_legal_moves",
        "get_move_history",
        "get_captured_pieces",
        "evaluate_position",
        "get_best_moves",
    }


def test_definitions_can_exclude_tools(registry):
    """The brain is offered a subset; the registry still holds everything.

    `BOARD_STATE_TOOLS` answer with strict subsets of the board state the brain
    is handed in its prompt every turn, so offering them to the brain only buys
    a wasted round trip out of a 4-iteration budget. Other callers (MCP, the
    delegate wire) have no such injection, so the tools stay registered and
    runnable — only the brain's *offer* narrows.
    """
    names = {
        d["function"]["name"] for d in registry.definitions(exclude=BOARD_STATE_TOOLS)
    }
    assert not names & set(BOARD_STATE_TOOLS)
    assert {"make_move", "undo", "evaluate_position"} <= names


def test_excluded_tools_are_still_dispatchable(registry):
    assert registry.dispatch("get_legal_moves", {})["ok"] is True


# --- what the brain is offered, resolved off live state ----------------------
#
# One policy, in one place, because two copies of it drift: assembly and the
# eval harness both ask this function what to withhold, so what is measured is
# what ships. The reads are always out (the state block already answers them);
# the rest is a *capability* the code withholds when the app knows the answer —
# `claim_draw` with no claim to make.


def test_the_brain_offer_always_withholds_the_board_state_reads(session):
    excluded = set(brain_tool_exclusions(ToolContext(session=session)))
    assert set(BOARD_STATE_TOOLS) <= excluded


def test_the_brain_offer_always_carries_get_best_moves(session):
    # Hints are on-request (the mode was retired 2026-09-01): the tool that
    # answers an advice ask must be in the offer on every turn, or the planner
    # is back to burning iterations hunting for a capability that isn't there.
    assert "get_best_moves" not in brain_tool_exclusions(ToolContext(session=session))


def test_the_brain_offer_always_carries_describe_position(session):
    """A read, but not one of `BOARD_STATE_TOOLS`. Those are withheld because
    their answers are already in the planner's state block; this one's consumer
    is the *narrator*, which is handed no board at all (#193) — withhold it and
    a description ask has nothing to describe from, which is exactly how it
    came back as an eval."""
    excluded = brain_tool_exclusions(ToolContext(session=session))
    assert "describe_position" not in excluded
    assert "describe_position" not in BOARD_STATE_TOOLS


def test_the_brain_offer_withholds_claim_draw_until_a_draw_is_claimable(session):
    """The tool exists only when the rules allow the claim, so the model is
    never asked to judge whether one is available — and the schema it plans
    against is unchanged on every turn where none is."""
    ctx = ToolContext(session=session)
    assert "claim_draw" in brain_tool_exclusions(ctx)

    for san in REPETITION * 2:
        session.submit_move(san)

    assert "claim_draw" not in brain_tool_exclusions(ctx)


def test_a_claimable_draw_adds_claim_draw_to_the_offer_and_nothing_else(session):
    """Byte-for-byte, the offer gains exactly one tool: the schema the planner
    reads is measurably sensitive to churn, so a claim must not reshuffle it."""
    ctx = ToolContext(session=session)
    registry = build_registry(ctx)
    before = registry.definitions(exclude=brain_tool_exclusions(ctx))

    for san in REPETITION * 2:
        session.submit_move(san)
    after = registry.definitions(exclude=brain_tool_exclusions(ctx))

    added = [d for d in after if d not in before]
    assert [d["function"]["name"] for d in added] == ["claim_draw"]
    assert [d for d in before if d not in after] == []


def test_definitions_are_openai_style_and_json_serializable(registry):
    definitions = registry.definitions()
    json.dumps(definitions)  # must not raise
    for definition in definitions:
        assert definition["type"] == "function"
        fn = definition["function"]
        assert fn["name"]
        assert fn["description"]
        assert fn["parameters"]["type"] == "object"
        # Closed schemas: the LLM cannot smuggle extra args past validation.
        assert fn["parameters"]["additionalProperties"] is False


# --- guidance carried on the tool descriptions ------------------------------
#
# A rule the model must follow belongs with the capability it governs, not in a
# distant system-prompt block (a prompt rule is honored ~half the time — the
# gate finding). These pin the hot-path move guidance and the destructive-op
# dance where the model actually reads them: the tool descriptions. Pinned as
# substantive tokens, not exact wording.


def _description(registry, name: str) -> str:
    for d in registry.definitions():
        if d["function"]["name"] == name:
            return d["function"]["description"]
    raise AssertionError(f"tool {name!r} not registered")


def _flat(description: str) -> str:
    """One line, for asserting on a sentence the docstring's wrapping breaks."""
    return " ".join(description.split())


def test_make_move_anchors_moves_to_provided_legal_moves(registry):
    # The move string must come from the board state's legal_moves list, never
    # be invented — and what reaches the model is looser than notation, so the
    # description carries descriptive examples ("queen's bishop pawn" → c3).
    desc = _description(registry, "make_move")
    assert "legal_moves" in desc
    assert "never invent" in desc.lower()
    assert "queen's bishop pawn" in desc  # a descriptive example, from real traces


def test_make_move_leaves_the_mutation_limit_to_code(session):
    """The once-per-turn rule is the phase machine's — a second player move
    mid-turn is refused — so it is not *also* prose. A rule code owns does not
    live in the prompt (the house rule), and a page of tone competes with the
    tool decision for a 12B's attention. What stays is what the model cannot
    derive: that the engine answers on its own, and that a proposal the player
    accepted has to be *called*."""
    ctx = ToolContext(session=session)
    atomic = _flat(_description(build_registry(ctx), "make_move"))
    split = _flat(_description(build_registry(ctx, atomic_exchange=False), "make_move"))

    for desc in (atomic, split):
        assert "once per player turn" not in desc.lower()
        assert "announcing a move in words is not making it" in desc
        assert "the player accepts" in desc
    assert "The engine plays its reply inside the same call" in atomic
    assert "The engine answers as soon as your move lands" in split


def test_make_move_requires_acting_on_an_accepted_proposal(registry):
    # Propose a move → player says yes → CALL make_move, don't announce it in
    # words (seen live: "moving forward with dxe5", no tool call, no move).
    desc = _description(registry, "make_move").lower()
    assert "accept" in desc
    assert "announc" in desc


def test_make_move_warns_about_mangled_voice_transcripts(registry):
    # Transcribed speech arrives mangled ("e 4"); repair obvious slips before
    # matching, rather than failing the move.
    desc = _description(registry, "make_move").lower()
    assert "voice" in desc
    assert '"e 4"' in desc


def test_evaluate_position_routes_who_is_winning(registry):
    # Live game: "who's winning?" was answered from vibes (wrongly), no tool
    # call. The judgment question is a read like any other.
    desc = _description(registry, "evaluate_position").lower()
    assert "who's winning" in desc


def test_describe_position_routes_whats_the_position(registry):
    """The other half of the same routing decision. Live, "what's the
    position?" reached `evaluate_position` — whose description was the only
    tool text in the registry containing the word — and came back as a verdict
    ("You're cooked") twice. The description ask now has a tool of its own, and
    it says so in the words the player uses."""
    desc = _flat(_description(registry, "describe_position")).lower()
    assert "what's the position" in desc
    assert "where each side's pieces stand" in desc


def test_evaluate_position_hands_the_description_ask_over(registry):
    """A verdict is not a description, and the tool that gives verdicts is the
    one that has to say so: two tools whose text both invited "what's the
    position?" is the ambiguity that produced the miss."""
    desc = _flat(_description(registry, "evaluate_position"))
    assert "`describe_position`" in desc
    assert "It says nothing about what is on the board" in desc


def test_analyze_last_move_routes_how_good_was_that_move(registry):
    desc = _description(registry, "analyze_last_move").lower()
    assert "how good was that move" in desc or "what was my mistake" in desc


def test_destructive_tools_carry_the_call_then_relay_dance(registry):
    # The gate owns the confirmation (tools.py `_gate`); the tool's job is only
    # to tell the model not to pre-ask, and to relay the refusal when it comes.
    for name in ("new_game", "resign", "claim_draw"):
        desc = _description(registry, name).lower()
        assert "confirm" in desc
        assert "relay" in desc


def test_register_duplicate_name_raises():
    registry = ToolRegistry()
    tool = Tool(
        name="t",
        description="d",
        parameters={"type": "object", "properties": {}, "additionalProperties": False},
        handler=lambda: {"ok": True},
    )
    registry.register(tool)
    with pytest.raises(ValueError):
        registry.register(tool)


# --- dispatch boundary ------------------------------------------------------


def test_dispatch_unknown_tool_is_error_not_exception(registry):
    result = registry.dispatch("no_such_tool", {})
    assert result["ok"] is False
    assert "no_such_tool" in result["error"]


def test_dispatch_rejects_non_dict_args(registry):
    result = registry.dispatch("get_board_state", "not a dict")
    assert result["ok"] is False


def test_dispatch_rejects_extra_properties(registry):
    result = registry.dispatch("get_board_state", {"bogus": 1})
    assert result["ok"] is False


def test_dispatch_rejects_wrong_arg_type(registry):
    result = registry.dispatch("get_best_moves", {"n": "three"})
    assert result["ok"] is False


def test_dispatch_rejects_out_of_range_args(registry):
    assert registry.dispatch("get_best_moves", {"n": 0})["ok"] is False
    assert registry.dispatch("get_best_moves", {"n": 11})["ok"] is False


def test_dispatch_reports_the_tool_that_is_about_to_run(registry):
    """Live progress reads the dispatch chokepoint (audit item 19): a tool call
    the UI hears about is one that really ran, because this is the same road
    every call takes."""
    seen: list[str] = []
    registry.on_tool = seen.append
    registry.dispatch("get_board_state", {})
    registry.dispatch("get_legal_moves", {})
    assert seen == ["get_board_state", "get_legal_moves"]


def test_a_call_that_never_runs_is_not_reported(registry):
    """An unknown name and args the schema rejects never reach a handler, so
    reporting them would put a label on screen for work nobody did."""
    seen: list[str] = []
    registry.on_tool = seen.append
    registry.dispatch("no_such_tool", {})
    registry.dispatch("get_best_moves", {"n": "three"})
    assert seen == []


def test_a_failing_tool_observer_never_costs_the_call(registry, session):
    def explode(_name):
        raise RuntimeError("socket went away")

    registry.on_tool = explode
    assert registry.dispatch("get_board_state", {})["ok"] is True


def test_dispatch_reports_a_mutation_once_it_has_happened(registry):
    """The board document's seam: a call that moved the board is one the UI must
    hear about immediately, and `board_version` is what says it moved."""
    seen: list[int] = []
    registry.on_mutation = lambda: seen.append(registry.context.board_version)
    registry.dispatch("make_move", {"move": "e4"})
    assert seen == [registry.context.board_version]


def test_a_call_that_moved_nothing_reports_no_mutation(registry):
    """A read and a rejected move leave the board where it was, and a state
    frame for a board that did not change is a frame nobody needed."""
    seen: list[int] = []
    registry.on_mutation = lambda: seen.append(1)
    registry.dispatch("get_board_state", {})
    registry.dispatch("make_move", {"move": "e5"})  # illegal for white
    assert seen == []


def test_a_mutation_is_reported_even_when_the_call_then_failed(registry, session):
    """The board is the board: a handler that moved it and *then* refused has
    still moved it, and a client left holding the old position would be wrong
    about the game rather than merely late."""
    seen: list[int] = []

    def refuse_after_moving() -> dict:
        session.submit_move("e4")
        raise ValueError("changed my mind")

    registry.register(
        Tool(
            name="half_done",
            description="mutates, then refuses",
            parameters={"type": "object", "properties": {}},
            handler=refuse_after_moving,
        )
    )
    registry.on_mutation = lambda: seen.append(1)
    assert registry.dispatch("half_done", {})["ok"] is False
    assert seen == [1]


def test_a_failing_mutation_observer_never_costs_the_call(registry):
    """Same rule the tool observer has: a lost frame is not a lost move."""

    def explode():
        raise RuntimeError("socket went away")

    registry.on_mutation = explode
    assert registry.dispatch("make_move", {"move": "e4"})["legal"] is True


def test_legal_moves_after_game_over_is_empty(registry, session):
    session.resign("white")
    result = registry.dispatch("get_legal_moves", {})
    assert result["ok"] is True
    assert result["moves"] == []


def test_all_dispatch_results_are_json_serializable(registry):
    json.dumps(registry.dispatch("get_board_state", {}))
    json.dumps(registry.dispatch("get_legal_moves", {}))
    json.dumps(registry.dispatch("nope", {}))
    json.dumps(registry.dispatch("get_best_moves", {"n": -1}))


def test_dispatch_never_mutates_session(registry, session):
    before = session.fen()
    registry.dispatch("get_board_state", {})
    registry.dispatch("get_legal_moves", {})
    registry.dispatch("get_move_history", {})
    registry.dispatch("get_captured_pieces", {})
    registry.dispatch("unknown", {"x": 1})
    assert session.fen() == before
    assert session.move_history() == []


# --- read tools -------------------------------------------------------------


def test_get_board_state_fresh_game(registry, session):
    result = registry.dispatch("get_board_state", {})
    assert result["ok"] is True
    assert result["fen"] == session.fen()
    assert result["turn"] == "white"
    assert result["game_over"] is False
    assert result["outcome"] is None


def test_get_board_state_after_checkmate(session):
    for move in ["e4", "e5", "Bc4", "Nc6", "Qf3", "d6", "Qxf7#"]:
        assert session.submit_move(move).legal
    registry = build_registry(ToolContext(session=session))
    result = registry.dispatch("get_board_state", {})
    assert result["game_over"] is True
    assert result["outcome"] == {
        "termination": "checkmate",
        "winner": "white",
        "result": "1-0",
    }


def test_get_legal_moves_start_position(registry):
    result = registry.dispatch("get_legal_moves", {})
    assert result["ok"] is True
    assert len(result["moves"]) == 20
    assert "e4" in result["moves"]
    assert "Nf3" in result["moves"]


def test_get_move_history_and_captures(session):
    for move in ["e4", "d5", "exd5", "Qxd5"]:
        assert session.submit_move(move).legal
    registry = build_registry(ToolContext(session=session))
    history = registry.dispatch("get_move_history", {})
    assert history["ok"] is True
    assert history["moves"] == ["e4", "d5", "exd5", "Qxd5"]
    captures = registry.dispatch("get_captured_pieces", {})
    assert captures["ok"] is True
    assert captures["white"] == ["p"]
    assert captures["black"] == ["p"]


# --- describe_position: the board in words ----------------------------------
#
# The read whose result *is* a description. It exists because the phase that
# speaks is handed no board (`api._narrator_state_dict`, #193), so "what's the
# position?" had nothing to answer from and came back as an eval ("You're
# cooked") twice in the 2026-09-04 walkthrough. Engine-free on purpose — the
# `registry` fixture below has no Stockfish and every test here still runs.


def _describe(registry) -> dict:
    result = registry.dispatch("describe_position", {})
    assert result["ok"] is True
    return result


def _black_registry() -> ToolRegistry:
    """A registry over a game the *player* is playing black in — the swap that
    decides which side the summary calls "you"."""
    return build_registry(ToolContext(session=GameSession(player_color="black")))


def test_describe_position_reports_the_board_as_data_and_as_prose(registry, session):
    result = _describe(registry)
    assert result["pieces"] == session.piece_placement()
    assert result["castling"] == session.castling_status()
    assert result["material"] == {"player_advantage": 0}
    assert result["in_check"] is False
    assert result["last_move"] is None
    assert result["move_number"] == 1
    assert result["plies"] == 0
    assert result["game_over"] is False
    assert result["outcome"] is None
    assert result["summary"]
    json.dumps(result)  # must not raise


def test_describe_position_offers_the_narrator_no_side_to_play_for(session):
    """The invariant the whole tool is shaped around (#193): a narrator that
    can see whose move it is announces one, so this result carries no `turn`,
    no `fen` (the string names the side to move) and no `legal_moves` (the menu
    to pick from) — the three fields `_narrator_state_dict` deletes for the
    same reason.

    And no top-level `san`: `api._verified_facts` reads that key off every tool
    result as a move the turn played, so the last ply rides under `last_move`,
    SAN only and with nobody's name on it.
    """
    for move in ("e4", "e5", "Nf3", "Nc6"):
        assert session.submit_move(move).legal
    result = _describe(build_registry(ToolContext(session=session)))

    assert {"turn", "fen", "legal_moves", "san"} & result.keys() == set()
    assert result["last_move"] == "Nc6"
    summary = result["summary"]
    assert "Last move: Nc6." in summary
    assert "your move" not in summary.lower()
    assert "to move" not in summary.lower()


def test_the_summary_addresses_the_player_and_the_narrator(registry):
    """Same second person as a move summary: the paragraph is written *to* the
    narrator, so "the player" is the human and "you" is Glitch."""
    summary = _describe(registry)["summary"]
    assert "The player (White) has: king e1; queen d1; rooks a1, h1;" in summary
    assert "You (Black) have: king e8; queen d8; rooks a8, h8;" in summary
    assert "pawns a2, b2, c2, d2, e2, f2, g2, h2." in summary


def test_the_summary_swaps_the_voices_with_the_players_color():
    """The one fact that decides who "you" is — read from the session, never
    from the board, which cannot know which side the human is on."""
    summary = _describe(_black_registry())["summary"]
    assert "The player (Black) has: king e8;" in summary
    assert "You (White) have: king e1;" in summary


def test_the_summary_says_nothing_has_been_played_on_a_fresh_board(registry):
    assert _describe(registry)["summary"].startswith(
        "Nothing has been played yet — the starting position."
    )


def test_the_summary_counts_the_moves_once_a_game_is_under_way(session, registry):
    for move in ("e4", "e5", "Nf3"):
        assert session.submit_move(move).legal
    assert _describe(registry)["summary"].startswith("Move 2, 3 plies played.")


def test_a_level_board_is_reported_as_level(registry):
    assert "Material is level." in _describe(registry)["summary"]


def test_the_material_sentence_follows_the_players_side(session, registry):
    """The direction is the point: the same three moves put the *player* a pawn
    up playing white and Glitch a pawn up playing black. Phrased as "up N
    pawns" because that is the shape the honesty guard's material class
    verifies against the board."""
    for move in ("e4", "d5", "exd5"):
        assert session.submit_move(move).legal
    assert "The player is up 1 pawn of material." in _describe(registry)["summary"]

    black = _black_registry()
    for move in ("e4", "d5", "exd5"):
        assert black.context.session.submit_move(move).legal
    assert "You are up 1 pawn of material." in _describe(black)["summary"]


def test_the_castling_sentences_report_each_side(session, registry):
    for move in ("e4", "e5", "Nf3", "Nc6", "Bc4", "Bc5", "O-O"):
        assert session.submit_move(move).legal
    summary = _describe(registry)["summary"]
    assert "The player has castled kingside." in summary
    assert "You can still castle either side." in summary


def test_a_side_that_lost_its_rights_without_castling_says_so(session, registry):
    for move in ("e4", "e5", "Ke2", "Ke7"):
        assert session.submit_move(move).legal
    summary = _describe(registry)["summary"]
    assert "The player can no longer castle." in summary
    assert "You can no longer castle." in summary


def test_one_remaining_right_is_named(session, registry):
    for move in ("Nf3", "Nf6", "Rg1", "Ng8"):
        assert session.submit_move(move).legal
    assert (
        "The player can still castle queenside only." in _describe(registry)["summary"]
    )


def test_the_check_sentence_names_a_color_not_a_person():
    """A king in check is the side to move's, so "you are in check" would be
    one more way of telling the narrator it is holding a turn (#193). The
    colour says the same fact without handing over a move."""
    session = GameSession(fen="4k3/8/8/8/8/8/8/4K2R w K - 0 1")
    assert session.submit_move("Rh8+").legal
    result = _describe(build_registry(ToolContext(session=session)))
    assert result["in_check"] is True
    assert "The black king is in check." in result["summary"]
    assert "you are in check" not in result["summary"].lower()


def test_no_check_sentence_when_nobody_is_in_check(registry):
    assert "in check" not in _describe(registry)["summary"]


def test_the_summary_reports_a_finished_game_first():
    session = GameSession(fen=WHITE_MATE_IN_1)
    assert session.submit_move("Qxf7#").legal
    result = _describe(build_registry(ToolContext(session=session)))
    assert result["game_over"] is True
    assert result["summary"].startswith("The game is over: checkmate, white wins.")


def test_a_drawn_game_is_reported_as_drawn():
    session = GameSession(fen=FIFTY_MOVE_FEN)
    session.claim_draw()
    result = _describe(build_registry(ToolContext(session=session)))
    assert result["summary"].startswith("The game is over: fifty moves, drawn.")


# --- write tools ------------------------------------------------------------


def test_registry_lists_all_write_tools(registry):
    names = {d["function"]["name"] for d in registry.definitions()}
    assert names >= {
        "make_move",
        "undo",
        "new_game",
        "resign",
        "claim_draw",
        "save_game",
        "resume_game",
        "export_pgn",
    }


def test_make_move_legal(registry, session):
    result = registry.dispatch("make_move", {"move": "e4"})
    assert result["ok"] is True
    assert result["legal"] is True
    assert result["san"] == "e4"
    assert result["uci"] == "e2e4"
    assert result["game_over"] is False
    assert result["fen"] == session.fen()
    assert result["turn"] == "black"


def test_make_move_illegal_is_ok_result_with_legal_false(registry, session):
    before = session.fen()
    result = registry.dispatch("make_move", {"move": "e5"})
    assert result["ok"] is True
    assert result["legal"] is False
    assert result["reason"]
    assert session.fen() == before


def test_make_move_requires_move_arg(registry):
    assert registry.dispatch("make_move", {})["ok"] is False
    assert registry.dispatch("make_move", {"move": 4})["ok"] is False


def test_make_move_with_engine_triggers_reply(session):
    """The agent path mirrors the UI path: a legal player move gets the
    engine's reply in the same tool call, so texting 'e4' never leaves the
    player to move for both sides."""
    registry = build_registry(ToolContext(session=session, engine=FakeEngine()))
    result = registry.dispatch("make_move", {"move": "e4"})
    assert result["legal"] is True
    assert result["engine_move"]["san"] == "e5"
    assert result["engine_move"]["uci"] == "e7e5"
    assert result["turn"] == "white"
    assert session.move_history() == ["e4", "e5"]
    assert result["fen"] == session.fen()


def test_make_move_reply_never_detours_through_multipv(session):
    # The reply is the engine's own move at the configured strength, never a
    # MultiPV detour around the difficulty (a personality move-bias layer was
    # tried in 2026-07 and removed for exactly this).
    engine = FakeEngine()
    ctx = ToolContext(session=session, engine=engine)
    registry = build_registry(ctx)
    result = registry.dispatch("make_move", {"move": "e4"})
    assert result["engine_move"]["uci"] == "e7e5"
    assert engine.multipv_requests == []


def test_make_move_illegal_gets_no_engine_reply(session):
    registry = build_registry(ToolContext(session=session, engine=FakeEngine()))
    result = registry.dispatch("make_move", {"move": "e5"})
    assert result["legal"] is False
    assert "engine_move" not in result
    assert session.move_history() == []


def test_make_move_no_engine_reply_once_game_is_over():
    class MustNotPlay:
        def play_move(self, session):
            raise AssertionError("engine must not reply to a game-ending move")

    session = GameSession(WHITE_MATE_IN_1)
    registry = build_registry(ToolContext(session=session, engine=MustNotPlay()))
    result = registry.dispatch("make_move", {"move": "Qxf7#"})
    assert result["game_over"] is True
    assert result["engine_move"] is None


def test_make_move_without_engine_has_no_reply(registry):
    result = registry.dispatch("make_move", {"move": "e4"})
    assert result["engine_move"] is None
    assert result["turn"] == "black"


# --- make_move, the two sequencing modes -------------------------------------
#
# The boundary is the same either way; who *sequences* the turn differs. Built
# with `atomic_exchange=False` (app assembly), the tool applies the player's move
# and stops, because the pipeline owns the beats that follow — the observation
# reaction, then collecting the engine's reply. Built atomically (the default,
# and what the MCP server gets), it runs the whole exchange itself, because an
# MCP caller has no pipeline and its game must not stall half-way through a turn.


def _split_registry(ctx: ToolContext):
    coordinator = TurnCoordinator(ctx)
    return build_registry(ctx, coordinator, atomic_exchange=False), coordinator


def test_split_make_move_applies_the_player_move_only(session):
    ctx = ToolContext(session=session, engine=FakeEngine())
    registry, coordinator = _split_registry(ctx)

    result = registry.dispatch("make_move", {"move": "e4"})

    assert result["legal"] is True
    assert result["san"] == "e4"
    assert "engine_move" not in result, "the reply is not this call's to report"
    assert session.move_history() == ["e4"], "the engine has not moved yet"
    assert coordinator.phase == TurnPhase.PLAYER_MOVE_APPLIED
    assert coordinator.turn_id == 1, "the turn is still open"


def test_split_make_move_reports_the_facts_the_narrator_reacts_to(session):
    """The observation beat's structured facts (audit item 5): what moved, what
    it took, and whether it checks — and deliberately not the board it left
    behind. The split payload's `fen`/`turn` described a mid-exchange position
    the pipeline is about to supersede, and `turn` named a side to play for to
    every narrator that read the result (#193): it is the engine's color for
    exactly as long as the reply is still being computed. The atomic payload
    keeps both — a caller with no pipeline gets the settled position."""
    for san in ("e4", "d5"):
        session.submit_move(san)
    ctx = ToolContext(session=session, engine=FakeEngine())
    registry, _ = _split_registry(ctx)

    result = registry.dispatch("make_move", {"move": "exd5"})

    assert result["san"] == "exd5"
    assert result["uci"] == "e4d5"
    assert result["capture"] == "p"
    assert result["check"] is False
    assert result["game_over"] is False
    assert "fen" not in result
    assert "turn" not in result


def test_split_make_move_reports_a_check(session):
    ctx = ToolContext(session=GameSession("4k3/8/8/8/8/8/8/4K2R w K - 0 1"))
    registry, _ = _split_registry(ctx)
    result = registry.dispatch("make_move", {"move": "Rh8"})
    assert result["check"] is True
    assert result["capture"] is None


def test_split_make_move_on_a_game_ending_move_closes_the_turn():
    session = GameSession(WHITE_MATE_IN_1)
    ctx = ToolContext(session=session, engine=FakeEngine())
    registry, coordinator = _split_registry(ctx)

    result = registry.dispatch("make_move", {"move": "Qxf7#"})

    assert result["game_over"] is True
    # Nothing is owed, so there is no beat left to run and no turn left open.
    assert coordinator.phase == TurnPhase.AWAITING_PLAYER
    assert coordinator.turn_id == 2


def test_split_make_move_illegal_leaves_the_turn_untouched(session):
    ctx = ToolContext(session=session, engine=FakeEngine())
    registry, coordinator = _split_registry(ctx)

    result = registry.dispatch("make_move", {"move": "e5"})

    assert result["legal"] is False
    assert coordinator.phase == TurnPhase.AWAITING_PLAYER
    assert session.move_history() == []


def test_atomic_make_move_keeps_the_whole_exchange_and_the_new_facts(session):
    """The default. An MCP caller dispatches this with nothing behind it to
    collect the reply, so the tool finishes the turn itself — and gains the same
    move facts the split result carries."""
    for san in ("e4", "d5"):
        session.submit_move(san)
    ctx = ToolContext(session=session, engine=FakeEngine("b8c6"))
    coordinator = TurnCoordinator(ctx)
    registry = build_registry(ctx, coordinator)

    result = registry.dispatch("make_move", {"move": "exd5"})

    assert result["engine_move"]["san"] == "Nc6"
    assert result["engine_move"]["uci"] == "b8c6"
    assert result["capture"] == "p"
    assert result["check"] is False
    assert coordinator.phase == TurnPhase.AWAITING_PLAYER
    assert coordinator.turn_id == 2
    assert session.move_history() == ["e4", "d5", "exd5", "Nc6"]


def test_the_two_modes_describe_themselves_differently(session):
    """The docstring is the model's only account of what the call does, so it
    cannot promise a reply inside the call in the mode that has none."""
    ctx = ToolContext(session=session)

    def description(registry):
        return next(
            d["function"]["description"]
            for d in registry.definitions()
            if d["function"]["name"] == "make_move"
        )

    atomic = description(build_registry(ctx))
    split = description(build_registry(ctx, atomic_exchange=False))
    assert "inside the same call" in atomic
    assert "inside the same call" not in split
    # Neither mode asks the model to do anything about the reply.
    assert "Map loose phrasing" in atomic and "Map loose phrasing" in split


# --- a move in the record says whose move it was -----------------------------
#
# The narrator is the player's *opponent*, and it never sees the board — it sees
# these results. Reporting a move with no actor left attribution to be inferred
# from history parity plus `player_color`, which a 12B fumbles: live, Glitch
# claimed the player's capture as his own ("I took your knight" for a piece the
# player had just taken off him). Whose move it was is deterministic state, so
# code states it: `mover`, and an English `summary` that spells out actor, move,
# whose piece fell and whether it checks.


def _summary(registry, move: str) -> str:
    return registry.dispatch("make_move", {"move": move})["summary"]


def test_make_move_names_the_mover(registry):
    result = registry.dispatch("make_move", {"move": "e4"})
    assert result["mover"] == "player"


def test_move_summary_states_the_actor_on_a_quiet_move(registry):
    assert _summary(registry, "e4") == "The player played e4."


def test_move_summary_says_whose_piece_the_capture_took(session):
    # The captured piece belonged to the side that did not move — the narrator's
    # own side. "your pawn", never a bare symbol to be attributed by guesswork.
    for san in ("e4", "d5"):
        session.submit_move(san)
    registry, _ = _split_registry(ToolContext(session=session, engine=FakeEngine()))

    assert _summary(registry, "exd5") == "The player played exd5, capturing your pawn."


def test_move_summary_names_the_piece_in_words():
    # A knight is a knight, not an "n": the summary is prose the narrator reads.
    knight_on_c6 = "r1bqkbnr/pppppppp/2n5/1B6/8/8/PPPPPPPP/RNBQK1NR w KQkq - 0 1"
    registry, _ = _split_registry(ToolContext(session=GameSession(knight_on_c6)))

    assert _summary(registry, "Bxc6") == (
        "The player played Bxc6, capturing your knight."
    )


def test_move_summary_reports_a_check():
    ctx = ToolContext(session=GameSession("4k3/8/8/8/8/8/8/4K2R w K - 0 1"))
    registry, _ = _split_registry(ctx)
    assert _summary(registry, "Rh8") == "The player played Rh8+, putting you in check."


def test_move_summary_combines_a_capture_and_a_check():
    ctx = ToolContext(session=GameSession("4k2r/8/8/8/8/8/8/4K2R w K - 0 1"))
    registry, _ = _split_registry(ctx)
    assert _summary(registry, "Rxh8") == (
        "The player played Rxh8+, capturing your rook and putting you in check."
    )


def test_move_summary_handles_an_en_passant_capture():
    # The taken pawn is not on the destination square, so a summary that read the
    # board after the push would name nothing. It reads `MoveResult.capture`,
    # which the session already derives en-passant-aware.
    ctx = ToolContext(session=GameSession("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1"))
    registry, _ = _split_registry(ctx)
    assert _summary(registry, "exd6") == "The player played exd6, capturing your pawn."


def test_the_engine_reply_carries_its_own_summary(session):
    """The atomic result holds two moves, so both say who made them — and the
    reply's wording is neutral, because it is the narrator's own move."""
    for san in ("e4", "d5"):
        session.submit_move(san)
    ctx = ToolContext(session=session, engine=FakeEngine("d5e4"))
    registry = build_registry(ctx, TurnCoordinator(ctx))

    result = registry.dispatch("make_move", {"move": "Nc3"})

    assert result["summary"] == "The player played Nc3."
    assert result["engine_move"]["summary"] == (
        "The engine played dxe4, capturing the player's pawn."
    )
    # Additive: the fields the reply already promised are untouched.
    assert result["engine_move"]["san"] == "dxe4"
    assert result["engine_move"]["uci"] == "d5e4"


def test_the_engine_summary_fits_an_opening_move_too():
    """`_engine_move_dict` also reports `new_game`'s opening move, which answers
    nothing — so the engine's wording says "played", never "replied"."""
    session = GameSession(player_color="black")
    registry = build_registry(
        ToolContext(session=session, engine=FakeEngine(reply_uci="e2e4"))
    )

    result = registry.dispatch("new_game", {})

    assert result["engine_move"]["summary"] == "The engine played e4."


def test_a_rejected_move_has_no_mover_or_summary(registry):
    # Nothing happened, so there is nothing to attribute; the refusal keeps
    # carrying only what correcting it needs.
    result = registry.dispatch("make_move", {"move": "e5"})
    assert result["legal"] is False
    assert "mover" not in result
    assert "summary" not in result


# --- non-move mutations end the open turn ------------------------------------
#
# A turn is about a position. Undo, a new game, a resignation and a resumed save
# all replace that position, so the coordinator's open turn — and any reply being
# computed for it — goes with them. Otherwise the next collect would apply a
# move decided for a board that no longer exists.


def test_undo_abandons_the_open_turn(session):
    ctx = ToolContext(session=session, engine=FakeEngine())
    registry, coordinator = _split_registry(ctx)
    registry.dispatch("make_move", {"move": "e4"})
    assert coordinator.phase == TurnPhase.PLAYER_MOVE_APPLIED

    result = registry.dispatch("undo", {})

    assert result["ok"] is True
    assert session.move_history() == []
    assert coordinator.phase == TurnPhase.AWAITING_PLAYER


def test_new_game_abandons_the_open_turn(session):
    ctx = ToolContext(session=session, engine=FakeEngine())
    ctx._confirming = True  # past the gate; this is about the turn, not the ask
    registry, coordinator = _split_registry(ctx)
    registry.dispatch("make_move", {"move": "e4"})

    result = registry.dispatch("new_game", {})

    assert result["ok"] is True, "a fresh board mid-turn is not a turn-state error"
    assert session.move_history() == []
    assert coordinator.phase == TurnPhase.AWAITING_PLAYER


def test_resign_abandons_the_open_turn(session):
    ctx = ToolContext(session=session, engine=FakeEngine())
    ctx._confirming = True  # past the gate; this is about the turn, not the ask
    registry, coordinator = _split_registry(ctx)
    registry.dispatch("make_move", {"move": "e4"})

    assert registry.dispatch("resign", {})["ok"] is True
    assert coordinator.phase == TurnPhase.AWAITING_PLAYER


def test_claim_draw_abandons_the_open_turn():
    """A claim is a non-move mutation like a resignation: it ends the game the
    open turn was about, so no reply is owed to it."""
    session = GameSession(fen=FIFTY_MOVE_FEN)
    ctx = ToolContext(session=session, engine=FakeEngine("e5e6"))
    ctx._confirming = True  # past the gate; this is about the turn, not the ask
    registry, coordinator = _split_registry(ctx)
    registry.dispatch("make_move", {"move": "Rh1"})  # quiet, so the clock runs on
    assert coordinator.phase == TurnPhase.PLAYER_MOVE_APPLIED

    assert registry.dispatch("claim_draw", {})["ok"] is True
    assert coordinator.phase == TurnPhase.AWAITING_PLAYER


def test_resume_game_abandons_the_open_turn(session, tmp_path):
    GameSession().save(tmp_path / "saved.json")
    ctx = ToolContext(session=session, engine=FakeEngine(), save_dir=tmp_path)
    registry, coordinator = _split_registry(ctx)
    registry.dispatch("make_move", {"move": "e4"})

    assert registry.dispatch("resume_game", {"name": "saved"})["ok"] is True
    assert coordinator.phase == TurnPhase.AWAITING_PLAYER


def test_make_move_reports_checkmate(registry):
    for move in ["f3", "e5", "g4"]:
        assert registry.dispatch("make_move", {"move": move})["legal"]
    result = registry.dispatch("make_move", {"move": "Qh4"})
    assert result["legal"] is True
    assert result["game_over"] is True


def test_undo_reverts_last_ply(registry, session):
    registry.dispatch("make_move", {"move": "e4"})
    result = registry.dispatch("undo", {})
    assert result["ok"] is True
    assert result["undone"] == ["e4"]
    assert session.move_history() == []


def test_undo_two_plies_for_engine_pair(registry, session):
    registry.dispatch("make_move", {"move": "e4"})
    registry.dispatch("make_move", {"move": "e5"})
    result = registry.dispatch("undo", {"plies": 2})
    assert result["ok"] is True
    assert result["undone"] == ["e5", "e4"]
    assert session.move_history() == []


def test_undo_defaults_to_the_whole_exchange_vs_engine(session):
    """A bare `undo()` vs the engine takes back the pair, not the lone reply.

    The pairing rule is the REST endpoint's, and it belongs to the caller of
    `GameSession.undo` — not to the model. Popping one ply here would leave the
    player's move on the board with the engine to move and nothing to move it.
    """
    ctx = ToolContext(session=session, engine=FakeEngine(reply_uci="e7e5"))
    registry = build_registry(ctx)
    registry.dispatch("make_move", {"move": "e4"})  # engine replies e5
    result = registry.dispatch("undo", {})
    assert result["ok"] is True
    assert result["undone"] == ["e5", "e4"]
    assert session.move_history() == []
    assert session.turn == session.player_color


def test_undo_defaults_to_one_ply_without_an_engine(registry, session):
    registry.dispatch("make_move", {"move": "e4"})
    assert registry.dispatch("undo", {})["undone"] == ["e4"]
    assert session.move_history() == []


def test_undo_default_never_takes_back_the_engines_lone_opening(session):
    """Player is black: the engine's opening move is not theirs to take back."""
    session.new_game(player_color="black")
    ctx = ToolContext(session=session, engine=FakeEngine(reply_uci="e2e4"))
    registry = build_registry(ctx)
    ctx.engine.play_move(session)  # engine (white) opens
    result = registry.dispatch("undo", {})
    assert result["ok"] is False
    assert session.move_history() == ["e4"]


def test_undo_honors_an_explicit_ply_count(session):
    ctx = ToolContext(session=session, engine=FakeEngine(reply_uci="e7e5"))
    registry = build_registry(ctx)
    registry.dispatch("make_move", {"move": "e4"})
    result = registry.dispatch("undo", {"plies": 1})
    assert result["undone"] == ["e5"]


def test_undo_with_nothing_to_undo_is_error(registry):
    result = registry.dispatch("undo", {})
    assert result["ok"] is False


def test_undo_rejects_bad_plies(registry):
    assert registry.dispatch("undo", {"plies": 0})["ok"] is False
    assert registry.dispatch("undo", {"plies": "two"})["ok"] is False


def test_new_game_resets(session):
    # A game is under way, so this goes through the gate: the call arms it, the
    # confirmation runs it. (The refusal itself is pinned further down.)
    ctx = ToolContext(session=session)
    registry = build_registry(ctx)
    registry.dispatch("make_move", {"move": "e4"})

    registry.dispatch("new_game", {})
    _, result = confirm_pending(registry, ctx)

    assert result["ok"] is True
    assert session.move_history() == []
    assert session.turn == "white"


def test_resign_defaults_to_the_player_not_the_side_to_move(session):
    """An unqualified "I resign" is the *player's* resignation, and the session
    knows which side they are. The old default — the side to move — was only
    coincidentally the player (trace review, finding 8): here the player is
    white, it is black's move, and the game must still end 0-1."""
    ctx = ToolContext(session=session)
    registry = build_registry(ctx)
    registry.dispatch("make_move", {"move": "e4"})
    assert session.turn == "black" and session.player_color == "white"

    registry.dispatch("resign", {})
    _, result = confirm_pending(registry, ctx)

    assert result["ok"] is True
    assert result["outcome"] == {
        "termination": "resignation",
        "winner": "black",
        "result": "0-1",
    }
    assert session.is_game_over()


def test_resign_explicit_color(registry, session):
    result = registry.dispatch("resign", {"color": "black"})
    assert result["ok"] is True
    assert result["outcome"]["winner"] == "white"


def test_resign_rejects_bad_color_and_finished_game(registry, session):
    assert registry.dispatch("resign", {"color": "green"})["ok"] is False
    registry.dispatch("resign", {})
    assert registry.dispatch("resign", {})["ok"] is False


# --- claim_draw: the other way a player ends a game ---------------------------
#
# Threefold repetition and the fifty-move rule are *claims*, so ending the game
# on one is a mutation like resigning — same gate, same budget — and whether a
# claim exists is board truth the tool checks before it asks anybody anything.


def test_claim_draw_ends_the_game_in_a_draw():
    session = GameSession(fen=FIFTY_MOVE_FEN)
    registry = build_registry(ToolContext(session=session))

    result = registry.dispatch("claim_draw", {})

    assert result["ok"] is True
    assert result["outcome"] == {
        "termination": "fifty_moves",
        "winner": None,
        "result": "1/2-1/2",
    }
    assert session.is_game_over()


def test_claim_draw_reports_the_rule_the_claim_landed_under():
    session = GameSession()
    for san in REPETITION * 2:
        session.submit_move(san)
    ctx = ToolContext(session=session)
    registry = build_registry(ctx)
    registry.dispatch("claim_draw", {})  # gated: a real game is on the board

    _, result = confirm_pending(registry, ctx)

    assert result["outcome"]["termination"] == "threefold_repetition"


def test_claim_draw_with_nothing_to_claim_is_a_refusal_not_a_crash(registry, session):
    """No draw available is a domain "no": the board never moved, and no
    argument would have changed the answer, so it comes back `retry: never` for
    the player to hear rather than for the loop to retry."""
    for san in ("e4", "e5"):
        session.submit_move(san)
    fen_before = session.fen()

    result = registry.dispatch("claim_draw", {})

    assert result["ok"] is False
    assert result["retry"] == "never"
    assert "no draw" in result["error"]
    assert session.fen() == fen_before
    assert not session.is_game_over()


def test_an_unclaimable_call_never_arms_the_gate(session):
    """Checked before the gate, so the player is never asked to confirm an op
    that could not run — a "yes" to that question would fail."""
    for san in ("e4", "e5"):
        session.submit_move(san)
    ctx = ToolContext(session=session)
    registry = build_registry(ctx)

    assert registry.dispatch("claim_draw", {})["ok"] is False

    assert ctx.pending is None


def test_export_pgn(registry):
    registry.dispatch("make_move", {"move": "e4"})
    registry.dispatch("make_move", {"move": "e5"})
    result = registry.dispatch("export_pgn", {})
    assert result["ok"] is True
    assert "1. e4 e5" in result["pgn"]


def test_save_and_resume_round_trip(tmp_path, session):
    registry = build_registry(ToolContext(session=session, save_dir=tmp_path))
    registry.dispatch("make_move", {"move": "e4"})
    registry.dispatch("make_move", {"move": "e5"})
    saved = registry.dispatch("save_game", {"name": "test-game"})
    assert saved["ok"] is True
    assert (tmp_path / GAME_SAVE_DIRNAME / "test-game.json").exists()

    registry.dispatch("new_game", {})
    resumed = registry.dispatch("resume_game", {"name": "test-game"})
    assert resumed["ok"] is True
    state = registry.dispatch("get_board_state", {})
    history = registry.dispatch("get_move_history", {})
    assert history["moves"] == ["e4", "e5"]
    assert state["turn"] == "white"


def test_save_and_resume_carry_the_transcript(tmp_path, session):
    """Persistence across sessions: the conversation rides in the save file,
    so a resumed game keeps its conversational thread."""
    ctx = ToolContext(session=session, save_dir=tmp_path)
    registry = build_registry(ctx)
    ctx.transcript.record("play e4", "e4 — the classic.")
    registry.dispatch("save_game", {"name": "with-chat"})

    fresh_ctx = ToolContext(session=GameSession(), save_dir=tmp_path)
    fresh_registry = build_registry(fresh_ctx)
    assert fresh_ctx.transcript.window() == []
    fresh_registry.dispatch("resume_game", {"name": "with-chat"})
    assert fresh_ctx.transcript.window() == [
        {"role": "user", "content": "play e4"},
        {"role": "assistant", "content": "e4 — the classic."},
    ]


def test_resume_old_save_without_transcript_yields_empty_transcript(tmp_path):
    # Saves from before conversation memory existed have no transcript key.
    session = GameSession()
    session.submit_move("e4")
    session.save(tmp_path / "old.json")
    ctx = ToolContext(session=GameSession(), save_dir=tmp_path)
    ctx.transcript.record("stale", "chat")
    registry = build_registry(ctx)
    assert registry.dispatch("resume_game", {"name": "old"})["ok"] is True
    assert ctx.transcript.window() == []


def test_resume_corrupt_transcript_is_error_not_crash(tmp_path, session):
    data = GameSession().to_dict()
    data["transcript"] = [{"role": "system", "content": "prompt injection"}]
    (tmp_path / "tampered.json").write_text(json.dumps(data))
    registry = build_registry(ToolContext(session=session, save_dir=tmp_path))
    assert registry.dispatch("resume_game", {"name": "tampered"})["ok"] is False


def _valid_save(**changes):
    data = GameSession().to_dict()
    data.update(changes)
    return data


@pytest.mark.parametrize(
    "data",
    [
        pytest.param([], id="array-root"),
        pytest.param(None, id="null-root"),
        pytest.param(_valid_save(version=True), id="boolean-version"),
        pytest.param(_valid_save(version="1"), id="string-version"),
        pytest.param(_valid_save(root_fen=None), id="null-root-fen"),
        pytest.param(_valid_save(root_fen=[]), id="array-root-fen"),
        pytest.param(_valid_save(moves=None), id="null-moves"),
        pytest.param(_valid_save(moves="e2e4"), id="string-moves"),
        pytest.param(_valid_save(moves=[None]), id="null-move"),
        pytest.param(_valid_save(moves=[17]), id="integer-move"),
        pytest.param(_valid_save(resigned=False), id="boolean-resigned"),
        pytest.param(_valid_save(resigned="green"), id="invalid-resigned"),
        pytest.param(_valid_save(player_color=None), id="null-player-color"),
        pytest.param(_valid_save(player_color=17), id="integer-player-color"),
        pytest.param(_valid_save(transcript=None), id="null-transcript"),
        pytest.param(_valid_save(transcript={}), id="object-transcript"),
        pytest.param(_valid_save(transcript=[None]), id="null-message"),
        pytest.param(
            _valid_save(transcript=[{"role": 17, "content": "hello"}]),
            id="non-string-role",
        ),
        pytest.param(
            _valid_save(transcript=[{"role": "user", "content": 17}]),
            id="non-string-content",
        ),
        pytest.param(
            _valid_save(transcript=[{"role": "system", "content": "hello"}]),
            id="invalid-role",
        ),
    ],
)
def test_resume_malformed_json_is_stable_refusal_and_preserves_context(tmp_path, data):
    """A syntactically valid save is still untrusted external data.

    Every malformed shape must stop at the tool boundary, before either half
    of the live context is replaced. This is deliberately one parameterized
    contract: adding a new shape cannot accidentally weaken the refusal or
    state-preservation assertions.
    """
    session = GameSession()
    session.submit_move("e4")
    ctx = ToolContext(session=session, save_dir=tmp_path)
    ctx.transcript.record("play e4", "e4. Revolutionary stuff.")
    registry = build_registry(ctx)
    game_dir = tmp_path / GAME_SAVE_DIRNAME
    game_dir.mkdir()
    path = game_dir / "malformed.json"
    path.write_text(json.dumps(data))

    session_before = ctx.session
    game_before = ctx.session.to_dict()
    transcript_before = ctx.transcript
    messages_before = ctx.transcript.to_dict()
    version_before = ctx.board_version

    result = registry.dispatch("resume_game", {"name": "malformed"})

    assert set(result) == {"ok", "error", "retry", "board_version"}
    assert result["ok"] is False
    assert result["error"]
    assert result["retry"] == "never"
    assert result["board_version"] == version_before
    assert ctx.session is session_before
    assert ctx.session.to_dict() == game_before
    assert ctx.transcript is transcript_before
    assert ctx.transcript.to_dict() == messages_before
    assert ctx.board_version == version_before


def test_save_game_default_name_is_autosave(tmp_path, session):
    registry = build_registry(ToolContext(session=session, save_dir=tmp_path))
    result = registry.dispatch("save_game", {})
    assert result["ok"] is True
    assert (tmp_path / GAME_SAVE_DIRNAME / "autosave.json").exists()


def test_resume_missing_save_is_error(tmp_path, session):
    registry = build_registry(ToolContext(session=session, save_dir=tmp_path))
    result = registry.dispatch("resume_game", {"name": "nope"})
    assert result["ok"] is False


def test_save_tools_reject_path_traversal_names(tmp_path, session):
    registry = build_registry(ToolContext(session=session, save_dir=tmp_path))
    for name in ("../evil", "a/b", "", "x" * 65):
        assert registry.dispatch("save_game", {"name": name})["ok"] is False
        assert registry.dispatch("resume_game", {"name": name})["ok"] is False


def test_save_tools_without_save_dir_are_error(registry):
    assert registry.dispatch("save_game", {})["ok"] is False
    assert registry.dispatch("resume_game", {})["ok"] is False


def test_resume_corrupt_save_is_error(tmp_path, session):
    (tmp_path / "bad.json").write_text("{not json")
    registry = build_registry(ToolContext(session=session, save_dir=tmp_path))
    result = registry.dispatch("resume_game", {"name": "bad"})
    assert result["ok"] is False


# --- a failed save must not cost the save it was replacing -------------------
#
# `save_game` overwrites a named save in place, so the write is the moment the
# last good game is at risk: a write that truncates the file and then dies (a
# full disk, an I/O error, the host going away) would take the previous save
# with it and hand the raw OSError up through the tool boundary. The bytes
# therefore go to a sibling temp file and the destination is replaced only once
# they are all down — the pattern `_write_settings_file` has always used — and
# the failure comes back as a refusal like every other "no" this layer produces.


def _partial_write_then_fail(target, data, *_args, **_kwargs):
    """A `write_text` that truncates, lands one byte, then dies.

    The nastiest shape of the real thing: the bytes it did write are already on
    disk when the OSError lands, so a save written straight to its destination
    has destroyed the previous one before it can report anything.
    """
    Path(target).write_bytes(data[:1].encode())
    raise OSError(28, "No space left on device")


def _save_through_a_dying_disk(registry, monkeypatch, name="autosave"):
    """Dispatch `save_game` with every file write failing mid-write."""
    with monkeypatch.context() as failing:
        failing.setattr(Path, "write_text", _partial_write_then_fail)
        return registry.dispatch("save_game", {"name": name})


def test_failed_overwrite_leaves_the_previous_save_loadable(
    tmp_path, monkeypatch, session
):
    ctx = ToolContext(session=session, save_dir=tmp_path)
    registry = build_registry(ctx)
    registry.dispatch("make_move", {"move": "e4"})
    assert registry.dispatch("save_game", {"name": "autosave"})["ok"] is True
    good_bytes = (tmp_path / GAME_SAVE_DIRNAME / "autosave.json").read_bytes()

    registry.dispatch("make_move", {"move": "e5"})
    assert _save_through_a_dying_disk(registry, monkeypatch)["ok"] is False

    # Byte-for-byte, and still a save the app can actually load.
    assert (tmp_path / GAME_SAVE_DIRNAME / "autosave.json").read_bytes() == good_bytes
    reader = ToolContext(session=GameSession(), save_dir=tmp_path)
    resumed = build_registry(reader).dispatch("resume_game", {"name": "autosave"})
    assert resumed["ok"] is True
    assert reader.session.move_history() == ["e4"]


def test_failed_overwrite_leaves_no_temp_file_behind(tmp_path, monkeypatch, session):
    ctx = ToolContext(session=session, save_dir=tmp_path)
    registry = build_registry(ctx)
    assert registry.dispatch("save_game", {"name": "autosave"})["ok"] is True

    assert _save_through_a_dying_disk(registry, monkeypatch)["ok"] is False

    game_dir = tmp_path / GAME_SAVE_DIRNAME
    assert sorted(path.name for path in game_dir.iterdir()) == ["autosave.json"]


def test_failed_first_save_leaves_no_file_behind(tmp_path, monkeypatch, session):
    """Nothing to preserve, so the bar is that nothing is created: a partial
    file under the requested name is a save the agent would be told exists."""
    ctx = ToolContext(session=session, save_dir=tmp_path)
    registry = build_registry(ctx)

    assert _save_through_a_dying_disk(registry, monkeypatch)["ok"] is False

    game_dir = tmp_path / GAME_SAVE_DIRNAME
    assert not game_dir.exists() or list(game_dir.iterdir()) == []
    assert saved_game_names(ctx) == []


def test_failed_save_is_a_stable_refusal_and_preserves_the_session(
    tmp_path, monkeypatch, session
):
    ctx = ToolContext(session=session, save_dir=tmp_path)
    registry = build_registry(ctx)
    registry.dispatch("make_move", {"move": "e4"})
    ctx.transcript.record("save this", "Saved. Probably.")
    game_before = ctx.session.to_dict()
    transcript_before = ctx.transcript.to_dict()
    version_before = ctx.board_version

    result = _save_through_a_dying_disk(registry, monkeypatch, "keeper")

    assert set(result) == {"ok", "error", "retry", "board_version"}
    assert result["ok"] is False
    assert "keeper" in result["error"]
    # The app's own words, not the platform's errno string: the agent relays
    # this to the player, and a message that varies by host is not a contract.
    assert "No space left" not in result["error"]
    # A disk that cannot take the bytes will not take them under another name,
    # so retrying is a wasted round trip out of the iteration budget.
    assert result["retry"] == "never"
    assert result["board_version"] == version_before
    # Saving is a read of the session; a failed one must not have touched it.
    assert ctx.session is session
    assert ctx.session.to_dict() == game_before
    assert ctx.transcript.to_dict() == transcript_before


# --- what saves exist, as deterministic state -------------------------------
#
# The agent must never have to *infer* whether a saved game exists: it is a
# question the filesystem answers. `saved_game_names` is the one reader, and the
# tool layer owns it because the tool layer owns the `games/{name}.json`
# convention.


def test_saved_game_names_without_save_dir_is_empty(session):
    assert saved_game_names(ToolContext(session=session)) == []


def test_saved_game_names_with_missing_dir_is_empty(tmp_path, session):
    ctx = ToolContext(session=session, save_dir=tmp_path / "not-created")
    assert saved_game_names(ctx) == []


def test_saved_game_names_lists_saves_sorted(tmp_path, session):
    ctx = ToolContext(session=session, save_dir=tmp_path)
    registry = build_registry(ctx)
    registry.dispatch("save_game", {"name": "scholars"})
    registry.dispatch("save_game", {"name": "blitz"})
    assert saved_game_names(ctx) == ["blitz", "scholars"]


def test_saved_game_names_ignores_non_saves(tmp_path, session):
    (tmp_path / "notes.txt").write_text("not a save")
    ctx = ToolContext(session=session, save_dir=tmp_path)
    build_registry(ctx).dispatch("save_game", {"name": "real"})
    assert saved_game_names(ctx) == ["real"]


# --- settings tools ---------------------------------------------------------


def test_registry_lists_all_settings_tools(registry):
    names = {d["function"]["name"] for d in registry.definitions()}
    assert names >= {
        "set_difficulty",
        "set_verbosity",
        "set_voice_output",
    }
    # The personality is fixed (Glitch); there is no set_personality tool.
    assert "set_personality" not in names
    # Hints are on-request, not a mode (retired 2026-09-01): there is nothing
    # for a setter to set, and the flip-it-on-unasked failure dies with it.
    assert "set_hints_mode" not in names


def test_settings_defaults(session):
    ctx = ToolContext(session=session)
    assert ctx.settings == Settings()
    assert ctx.settings.verbosity == "normal"
    assert ctx.settings.voice_output is False
    # A real default strength, not None: without one the engine silently
    # plays at Stockfish's full-strength default.
    assert ctx.settings.tier == DEFAULT_TIER
    assert ctx.settings.skill_level is None
    assert ctx.settings.elo is None


def test_set_difficulty_skill_level_recorded(session):
    ctx = ToolContext(session=session)
    registry = build_registry(ctx)
    result = registry.dispatch("set_difficulty", {"skill_level": 5})
    assert result["ok"] is True
    assert ctx.settings.skill_level == 5
    assert ctx.settings.elo is None
    # A raw knob replaces the named tier.
    assert ctx.settings.tier is None


def test_set_difficulty_by_tier_reaches_engine_and_clears_raw_knobs(session):
    engine = FakeEngine()
    ctx = ToolContext(session=session, engine=engine)
    registry = build_registry(ctx)
    registry.dispatch("set_difficulty", {"skill_level": 5})
    result = registry.dispatch("set_difficulty", {"tier": "beginner"})
    assert result["ok"] is True
    assert result["tier"] == "beginner"
    assert ctx.settings.tier == "beginner"
    assert ctx.settings.skill_level is None
    assert ctx.settings.elo is None
    assert engine.tiers == ["beginner"]


def test_set_difficulty_rejects_unknown_tier(registry):
    assert registry.dispatch("set_difficulty", {"tier": "impossible"})["ok"] is False


def test_set_difficulty_elo_recorded_and_clears_skill(session):
    ctx = ToolContext(session=session)
    registry = build_registry(ctx)
    registry.dispatch("set_difficulty", {"skill_level": 5})
    result = registry.dispatch("set_difficulty", {"elo": 1500})
    assert result["ok"] is True
    assert ctx.settings.elo == 1500
    assert ctx.settings.skill_level is None


def test_set_difficulty_requires_exactly_one_of_skill_or_elo(registry):
    assert registry.dispatch("set_difficulty", {})["ok"] is False
    assert (
        registry.dispatch("set_difficulty", {"skill_level": 5, "elo": 1500})["ok"]
        is False
    )


def test_set_difficulty_rejects_out_of_range(registry):
    assert registry.dispatch("set_difficulty", {"skill_level": 21})["ok"] is False
    assert registry.dispatch("set_difficulty", {"skill_level": -1})["ok"] is False
    assert registry.dispatch("set_difficulty", {"elo": 100})["ok"] is False
    assert registry.dispatch("set_difficulty", {"elo": 4000})["ok"] is False


@requires_stockfish
def test_set_difficulty_configures_live_engine(session, live_engine):
    ctx = ToolContext(session=session, engine=live_engine)
    registry = build_registry(ctx)
    assert registry.dispatch("set_difficulty", {"skill_level": 3})["ok"] is True
    assert registry.dispatch("set_difficulty", {"elo": 1400})["ok"] is True


def test_set_verbosity(session):
    ctx = ToolContext(session=session)
    registry = build_registry(ctx)
    result = registry.dispatch("set_verbosity", {"verbosity": "low"})
    assert result["ok"] is True
    assert ctx.settings.verbosity == "low"
    assert registry.dispatch("set_verbosity", {"verbosity": "shouty"})["ok"] is False


def test_set_voice_output(session):
    ctx = ToolContext(session=session)
    registry = build_registry(ctx)
    assert registry.dispatch("set_voice_output", {"enabled": True})["ok"] is True
    assert ctx.settings.voice_output is True
    assert registry.dispatch("set_voice_output", {})["ok"] is False
    assert registry.dispatch("set_voice_output", {"enabled": "yes"})["ok"] is False


def test_settings_results_are_json_serializable(registry):
    json.dumps(registry.dispatch("set_difficulty", {"skill_level": 5}))
    json.dumps(registry.dispatch("set_verbosity", {"verbosity": "high"}))
    json.dumps(registry.dispatch("set_voice_output", {"enabled": False}))


# --- analysis tools (engine-backed) ----------------------------------------


def test_analysis_tools_without_engine_return_error(registry):
    for name in ("evaluate_position", "get_best_moves"):
        result = registry.dispatch(name, {})
        assert result["ok"] is False
        assert "engine" in result["error"]


@requires_stockfish
def test_evaluate_position_start(session, live_engine):
    registry = build_registry(ToolContext(session=session, engine=live_engine))
    result = registry.dispatch("evaluate_position", {})
    assert result["ok"] is True
    assert result["mate_in"] is None
    assert abs(result["score_cp"]) < 150


@requires_stockfish
def test_get_best_moves_mate_position(live_engine):
    session = GameSession(fen=WHITE_MATE_IN_1)
    registry = build_registry(ToolContext(session=session, engine=live_engine))
    result = registry.dispatch("get_best_moves", {"n": 2})
    assert result["ok"] is True
    best = result["moves"][0]
    assert best == {"uci": "f3f7", "san": "Qxf7#", "score_cp": None, "mate_in": 1}
    json.dumps(result)


@requires_stockfish
def test_analysis_on_finished_game_is_error_result(session, live_engine):
    session.resign("black")
    registry = build_registry(ToolContext(session=session, engine=live_engine))
    for name in ("evaluate_position", "get_best_moves"):
        result = registry.dispatch(name, {})
        assert result["ok"] is False
        assert "over" in result["error"]


def test_new_game_plays_engine_opening_when_player_is_black():
    # A voice/text "new game" while playing black must not leave the board
    # stuck waiting for white: the engine opens, mirroring the UI path.
    session = GameSession(player_color="black")
    engine = FakeEngine(reply_uci="e2e4")
    registry = build_registry(ToolContext(session=session, engine=engine))
    result = registry.dispatch("new_game", {})
    assert result["ok"] is True
    assert result["engine_move"]["san"] == "e4"
    assert result["engine_move"]["uci"] == "e2e4"
    assert session.move_history() == ["e4"]
    assert session.turn == "black"


def test_new_game_as_white_has_no_engine_opening():
    session = GameSession()
    engine = FakeEngine(reply_uci="e2e4")
    registry = build_registry(ToolContext(session=session, engine=engine))
    result = registry.dispatch("new_game", {})
    assert result["ok"] is True
    assert result["engine_move"] is None
    assert session.move_history() == []


# --- "let's play chess as black": new_game has to be able to say it.
#
# `new_game()` took no arguments, so the agent had no way to assign a side and
# the model fabricated compliance instead (trace review, finding 2). This is
# also the exact intent string in the advertised conductor deep link
# (/?intent=let's+play+chess+as+black), so the handoff was broken end to end.


def test_new_game_assigns_the_requested_side():
    session = GameSession()  # the player is white today
    engine = FakeEngine(reply_uci="e2e4")
    registry = build_registry(ToolContext(session=session, engine=engine))

    result = registry.dispatch("new_game", {"player_color": "black"})

    assert result["ok"] is True
    assert session.player_color == "black"
    # Owning white, the engine must open — or the board sits waiting on a move
    # only it can make.
    assert result["engine_move"]["san"] == "e4"
    assert result["engine_move"]["uci"] == "e2e4"
    assert session.move_history() == ["e4"]
    assert session.turn == "black"


def test_new_game_can_switch_back_to_white():
    session = GameSession(player_color="black")
    engine = FakeEngine(reply_uci="e2e4")
    registry = build_registry(ToolContext(session=session, engine=engine))

    result = registry.dispatch("new_game", {"player_color": "white"})

    assert result["ok"] is True
    assert session.player_color == "white"
    assert result["engine_move"] is None
    assert session.move_history() == []


def test_new_game_without_a_color_keeps_the_current_side():
    session = GameSession(player_color="black")
    engine = FakeEngine(reply_uci="e2e4")
    registry = build_registry(ToolContext(session=session, engine=engine))

    registry.dispatch("new_game", {})

    assert session.player_color == "black"


def test_new_game_rejects_a_color_that_is_not_a_side():
    session = GameSession()
    registry = build_registry(ToolContext(session=session))

    result = registry.dispatch("new_game", {"player_color": "red"})

    assert result["ok"] is False


def test_the_requested_side_survives_the_confirmation_gate():
    """The gate arms the op and the player's "yes" replays it later. If the
    requested color didn't ride along in the pending args, "new game, I'll take
    black" would be confirmed into a game as *white* — the gate would silently
    drop the only thing the player asked for."""
    session = GameSession(player_color="white")
    engine = FakeEngine(reply_uci="e2e4")
    ctx = ToolContext(session=session, engine=engine)
    registry = build_registry(ctx)
    played(session, "e4", "e5")  # a real game stands to be lost

    refused = registry.dispatch("new_game", {"player_color": "black"})
    assert refused["ok"] is False
    assert session.player_color == "white", "the gate must not mutate"

    name, result = confirm_pending(registry, ctx)

    assert name == "new_game"
    assert result["ok"] is True
    assert session.player_color == "black", "the player's yes lost their side"
    assert session.move_history() == ["e4"], "the engine owns white and must open"


# --- The destructive-op confirmation gate.
#
# `new_game` and `resign` throw a real game away. The prompt asks the agent to
# confirm first, but gemma-4-12b honors that only ~half the time (docs/agent-evals
# .md), so the rule is enforced here instead: an unconfirmed call does not mutate,
# it arms a pending op and comes back as a rejection *result* the agent reads and
# asks from. The model cannot talk its way past this — confirmation is not a tool
# argument (see test_confirmation_is_not_a_tool_argument), it is pipeline-owned.


def played(session, *sans):
    """Put a real game on the board — something that stands to be lost."""
    for san in sans:
        session.submit_move(san)
    return session


def test_unconfirmed_new_game_does_not_reset(session, registry):
    played(session, "e4", "e5")
    fen_before = session.fen()

    result = registry.dispatch("new_game", {})

    assert result["ok"] is False
    assert "confirm" in result["error"].lower()
    assert session.fen() == fen_before, "the board must not move on an unconfirmed call"


def test_unconfirmed_resign_does_not_end_the_game(session, registry):
    played(session, "e4", "e5")

    result = registry.dispatch("resign", {})

    assert result["ok"] is False
    assert "confirm" in result["error"].lower()
    assert not session.is_game_over()


def test_unconfirmed_call_arms_the_pending_op(session):
    ctx = ToolContext(session=played(GameSession(), "e4", "e5"))
    registry = build_registry(ctx)
    assert ctx.pending is None

    registry.dispatch("resign", {"color": "white"})

    assert ctx.pending is not None
    assert ctx.pending.name == "resign"
    assert ctx.pending.args == {"color": "white"}


def test_confirm_pending_executes_the_armed_op(session):
    ctx = ToolContext(session=played(GameSession(), "e4", "e5"))
    registry = build_registry(ctx)
    registry.dispatch("new_game", {})

    name, result = confirm_pending(registry, ctx)

    assert name == "new_game"
    assert result["ok"] is True
    assert ctx.session.fen() == GameSession().fen(), "confirmed: the board resets"
    assert ctx.pending is None, "the op is spent"


def test_confirm_pending_with_nothing_armed_is_a_no_op(session, registry):
    ctx = ToolContext(session=session)
    assert confirm_pending(build_registry(ctx), ctx) is None


def test_confirmation_is_not_a_tool_argument(session, registry):
    """The gate would be worthless if the model could open it: `confirm` is not
    in either schema, and the schemas are closed, so a model that invents the
    argument is rejected on args — the board still does not move."""
    played(session, "e4", "e5")
    fen_before = session.fen()

    for name in ("new_game", "resign", "claim_draw"):
        result = registry.dispatch(name, {"confirm": True})
        assert result["ok"] is False
        assert "invalid args" in result["error"]
        assert session.fen() == fen_before


def test_unconfirmed_claim_draw_does_not_end_the_game(session):
    """A claim ends a real game, so it earns the same question a resignation
    does — a mis-parsed "let's call it a draw" must cost a question, never the
    game."""
    ctx = ToolContext(session=session)
    registry = build_registry(ctx)
    for san in REPETITION * 2:
        session.submit_move(san)

    result = registry.dispatch("claim_draw", {})

    assert result["ok"] is False
    assert "confirm" in result["error"].lower()
    assert not session.is_game_over()
    assert ctx.pending is not None and ctx.pending.name == "claim_draw"
    assert ctx.pending.args == {}


def test_a_confirmed_claim_draw_ends_the_game(session):
    ctx = ToolContext(session=session)
    registry = build_registry(ctx)
    for san in REPETITION * 2:
        session.submit_move(san)
    registry.dispatch("claim_draw", {})

    name, result = confirm_pending(registry, ctx)

    assert name == "claim_draw"
    assert result["ok"] is True
    assert session.outcome().result == "1/2-1/2"
    assert ctx.pending is None, "the op is spent"


def test_a_second_unconfirmed_call_still_does_not_fire(session):
    """The agent retrying inside one turn must not self-confirm: re-arming is
    not confirming. Only the pipeline, on a new user turn, can open the gate."""
    ctx = ToolContext(session=played(GameSession(), "e4", "e5"))
    registry = build_registry(ctx)
    fen_before = ctx.session.fen()

    assert registry.dispatch("new_game", {})["ok"] is False
    assert registry.dispatch("new_game", {})["ok"] is False

    assert ctx.session.fen() == fen_before


def test_an_armed_op_records_the_board_it_is_about(session):
    """A destructive question is a question about a position, so the position
    rides along in the armed op."""
    ctx = ToolContext(session=played(GameSession(), "e4", "e5"))
    registry = build_registry(ctx)

    registry.dispatch("resign", {"color": "white"})

    assert ctx.pending.board_version == ctx.board_version


def test_live_pending_drops_an_op_the_board_has_moved_past(session):
    """The read that makes the stamp matter: once the board moves, the question
    is about a game that is no longer on it, so there is nothing to confirm."""
    ctx = ToolContext(session=played(GameSession(), "e4", "e5"))
    registry = build_registry(ctx)
    registry.dispatch("resign", {"color": "white"})
    assert ctx.live_pending() is not None, "the same board: still answerable"

    ctx.session.submit_move("Nf3")

    assert ctx.live_pending() is None
    assert ctx.pending is None, "and it is not left lying around to go stale twice"


def test_confirm_pending_will_not_run_an_op_about_another_board(session):
    """The last gate before a game is thrown away makes the check itself rather
    than trusting each caller to have made it."""
    ctx = ToolContext(session=played(GameSession(), "e4", "e5"))
    registry = build_registry(ctx)
    registry.dispatch("resign", {"color": "white"})
    ctx.session.submit_move("Nf3")

    assert confirm_pending(registry, ctx) is None
    assert not ctx.session.is_game_over()


def test_restamping_points_the_question_at_the_board_now_on_screen(session):
    """The gate arms mid-turn and the turn can still mutate after it (the
    engine's reply to a move played in the same command). What the player is
    asked about is the board when the turn ends."""
    ctx = ToolContext(session=played(GameSession(), "e4", "e5"))
    registry = build_registry(ctx)
    registry.dispatch("resign", {"color": "white"})
    ctx.session.submit_move("Nf3")

    ctx.restamp_pending()

    assert ctx.live_pending() is not None
    assert ctx.pending.board_version == ctx.board_version


def test_restamping_nothing_is_a_no_op(session):
    ctx = ToolContext(session=session)
    ctx.restamp_pending()
    assert ctx.pending is None


def test_new_game_after_game_over_needs_no_confirmation(session, registry):
    """The standing exception: game_over means there is no game left to lose."""
    # Fool's mate — game over, nothing to protect.
    played(session, "f3", "e5", "g4", "Qh4")
    assert session.is_game_over()

    result = registry.dispatch("new_game", {})

    assert result["ok"] is True
    assert session.fen() == GameSession().fen()


# --- The destructive-op budget: one per command.
#
# The gate guards the *player's* investment, so it stands aside when there is
# none — on a finished game, and on a board nobody has moved yet. That left a
# hole one command wide: a planner could reset a finished game and then resign
# the fresh board it had just created, both ops past the gate, both inside one
# user interaction. Nothing code-owned said "one destructive op per command", so
# now the coordinator does (audit item 6). A window is open below because the
# pipeline is the only surface that chains dispatches; the last test here is the
# other shape, and it is deliberately unconstrained.


def finished(session):
    """Fool's mate: a game that is over, so the gate stands aside."""
    return played(session, "f3", "e5", "g4", "Qh4")


def _windowed_registry(ctx: ToolContext):
    """A registry over a coordinator with a command window open — the pipeline's
    shape, which is the only one that can spend a budget twice."""
    coordinator = TurnCoordinator(ctx)
    coordinator.begin_command()
    return build_registry(ctx, coordinator), coordinator


def test_only_one_new_game_runs_per_command():
    ctx = ToolContext(session=finished(GameSession()))
    registry, _ = _windowed_registry(ctx)

    assert registry.dispatch("new_game", {})["ok"] is True
    refused = registry.dispatch("new_game", {"player_color": "black"})

    assert refused["ok"] is False
    assert "one per turn" in refused["error"]
    # The color is the tell: a second reset that ran would have switched sides.
    assert ctx.session.player_color == "white", "the refused reset must not have run"
    assert ctx.session.move_history() == []


def test_a_resign_after_a_new_game_is_refused_in_the_same_command():
    """The exact live hole: the fresh board `new_game` just made has no player
    move on it, so the gate would wave the resignation straight through and the
    new game would die the moment it was born."""
    ctx = ToolContext(session=finished(GameSession()))
    registry, _ = _windowed_registry(ctx)
    assert registry.dispatch("new_game", {})["ok"] is True

    refused = registry.dispatch("resign", {})

    assert refused["ok"] is False
    assert "one per turn" in refused["error"]
    assert not ctx.session.is_game_over(), "the fresh game is still live"
    assert ctx.pending is None, "a budget refusal never arms the gate"


def test_a_gate_refusal_does_not_burn_the_budget():
    """Check then record: the budget pays for an op that *ran*. A refused one
    left the board alone, so the player's yes on the next turn — or the model's
    corrected call — must still find the budget there."""
    ctx = ToolContext(session=played(GameSession(), "e4", "e5"))
    registry, coordinator = _windowed_registry(ctx)

    refused = registry.dispatch("new_game", {})

    assert refused["ok"] is False and ctx.pending is not None
    coordinator.require_destructive_budget()  # must not raise


def test_a_failed_destructive_op_does_not_burn_the_budget():
    """Same rule one layer down: `resign` on a finished game raises, so nothing
    was thrown away and nothing was spent."""
    ctx = ToolContext(session=finished(GameSession()))
    registry, coordinator = _windowed_registry(ctx)

    assert registry.dispatch("resign", {})["ok"] is False

    coordinator.require_destructive_budget()  # must not raise


def test_a_claim_spends_the_destructive_budget():
    """Ending the game by claiming a draw is one of the ops the window budgets:
    it threw the game away, so the command's one destructive op is gone."""
    ctx = ToolContext(session=GameSession(fen=FIFTY_MOVE_FEN))
    registry, _ = _windowed_registry(ctx)

    assert registry.dispatch("claim_draw", {})["ok"] is True
    refused = registry.dispatch("new_game", {})

    assert refused["ok"] is False
    assert "one per turn" in refused["error"]
    assert ctx.session.outcome().termination == "fifty_moves", "the draw stands"


def test_a_refused_claim_does_not_burn_the_budget():
    """Check then record, as everywhere else: nothing was claimable, so nothing
    was thrown away and the command's budget is still there for the real op."""
    ctx = ToolContext(session=played(GameSession(), "e4", "e5"))
    registry, coordinator = _windowed_registry(ctx)

    assert registry.dispatch("claim_draw", {})["ok"] is False

    coordinator.require_destructive_budget()  # must not raise


def test_a_windowless_registry_is_unconstrained():
    """MCP's shape (`build_registry` with no coordinator of its own, nothing
    opening a window): one dispatch per call by construction, so a client may
    start as many games across a session as it likes."""
    ctx = ToolContext(session=GameSession())
    registry = build_registry(ctx)

    assert registry.dispatch("new_game", {"player_color": "black"})["ok"] is True
    assert registry.dispatch("new_game", {"player_color": "white"})["ok"] is True
    assert ctx.session.player_color == "white"
