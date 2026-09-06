"""Draw offers: the player offers, code decides for the engine (`docs/draw-offer.md`).

Three layers, each pinned where it lives. `GameSession.agree_draw` is the
session-level ending beside resignation and a claimed draw. `draw_offer.
judge_draw_offer` is the rule — a pure function of Stockfish's number and the
material, so every branch is pinned here with a synthetic evaluation and no
engine, and each constant has a position at its edge. The `offer_draw` tool and
`/api/game/offer-draw` are the two surfaces, tested at the tool boundary with a
`FakeEngine` whose evaluation is scripted; the four fixtures are also run
against the real Stockfish once each (skipped without the binary) so the rule's
inputs are known to be what the engine really says about them.
"""

import shutil

import pytest
from fastapi.testclient import TestClient

from chessapp.api import create_app
from chessapp.coordinator import TurnCoordinator, TurnPhase
from chessapp.draw_offer import (
    DRAW_OFFER_BAND_CP,
    ENDGAME_MAX_NON_PAWN_MATERIAL,
    ENGINE_AHEAD,
    NOT_AN_ENDGAME,
    PLAYER_AHEAD,
    TOO_EARLY,
    is_endgame,
    judge_draw_offer,
)
from chessapp.engine import MATE_CP, EnginePlayer, Evaluation
from chessapp.game import GameSession
from chessapp.tools import ToolContext, brain_tool_exclusions, build_registry
from fakes import FakeEngine

requires_stockfish = pytest.mark.skipif(
    shutil.which("stockfish") is None, reason="stockfish binary not installed"
)

# --- the four fixtures -------------------------------------------------------
#
# Each is a position *before* two plies are played on it, so the player has
# moved (the rule's first premise) and the position judged is the one after
# them. The moves are quiet and keep the character of the position.

# Rook and three pawns each, symmetrical: dead level, an endgame by any rule.
# (Rooks on different files, so no side has a capture on the move.)
DEAD_DRAWN_ROOK_ENDGAME = ("4k3/r4ppp/8/8/8/8/2R2PPP/4K3 w - - 0 40", ("Kd2", "Kd7"))
# The Ruy Lopez after 3...a6: level, and every piece still on the board.
FLAT_MIDDLEGAME = (
    "r1bqkbnr/1ppp1ppp/p1n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 4",
    ("Ba4", "Nf6"),
)
# Rook and pawns against a lone king's pawns: the engine (black) is winning.
ENGINE_WINNING_ENDGAME = ("4k3/r4ppp/8/8/8/8/5PPP/4K3 w - - 0 40", ("Kd1", "Ra1+"))
# The same, mirrored: the player (white) has the rook.
PLAYER_WINNING_ENDGAME = ("4k3/5ppp/8/8/8/8/R4PPP/4K3 w - - 0 40", ("Rb2", "Kd8"))

LEVEL = Evaluation(score_cp=0, mate_in=None)


def positioned(fixture, player_color="white") -> GameSession:
    fen, plies = fixture
    session = GameSession(fen=fen, player_color=player_color)
    for san in plies:
        assert session.submit_move(san).legal, san
    return session


# --- the session ------------------------------------------------------------


def test_agree_draw_ends_the_game_by_agreement():
    session = GameSession()
    session.submit_move("e4")
    outcome = session.agree_draw()
    assert outcome.termination == "agreement"
    assert outcome.winner is None
    assert outcome.result == "1/2-1/2"
    assert session.is_game_over()
    assert session.outcome() == outcome
    assert session.legal_moves() == []
    assert session.claimable_draws() == ()


def test_agree_draw_bumps_the_revision_once():
    session = GameSession()
    before = session.revision
    session.agree_draw()
    assert session.revision == before + 1


def test_agree_draw_refuses_a_finished_game():
    session = GameSession()
    session.resign("white")
    before = session.revision
    with pytest.raises(ValueError):
        session.agree_draw()
    assert session.revision == before
    assert session.outcome().termination == "resignation"


def test_moves_and_undo_are_refused_after_agreement():
    session = GameSession()
    session.submit_move("e4")
    session.agree_draw()
    assert not session.submit_move("e5").legal
    result = session.undo()
    assert not result.ok
    assert "agreement" in result.reason


def test_new_game_clears_the_agreement():
    session = GameSession()
    session.agree_draw()
    session.new_game()
    assert not session.is_game_over()
    assert session.outcome() is None


def test_an_agreed_draw_round_trips_through_a_save():
    session = GameSession()
    session.submit_move("e4")
    session.agree_draw()
    data = session.to_dict()
    assert data["draw_agreed"] is True
    restored = GameSession.from_dict(data)
    assert restored.is_game_over()
    assert restored.outcome() == session.outcome()


def test_a_save_without_the_agreement_flag_is_a_live_game():
    data = GameSession().to_dict()
    del data["draw_agreed"]
    assert not GameSession.from_dict(data).is_game_over()


def test_from_dict_rejects_a_non_bool_agreement_flag():
    data = GameSession().to_dict()
    data["draw_agreed"] = "yes"
    with pytest.raises(ValueError):
        GameSession.from_dict(data)


def test_from_dict_rejects_an_agreement_on_a_finished_game():
    """Replayed through the same refusal a live agreement gets: a file carrying
    a checkmate *and* an agreement is not a game that was played."""
    session = GameSession()
    for san in ("f3", "e5", "g4", "Qh4#"):
        session.submit_move(san)
    data = session.to_dict()
    data["draw_agreed"] = True
    with pytest.raises(ValueError):
        GameSession.from_dict(data)


def test_the_pgn_records_the_half_point():
    session = GameSession()
    session.submit_move("e4")
    session.agree_draw()
    assert '[Result "1/2-1/2"]' in session.export_pgn()


# --- material_profile -------------------------------------------------------


def test_material_profile_of_the_start_position():
    profile = GameSession().material_profile()
    assert profile["queens"] is True
    assert profile["non_pawn"] == {"white": 31, "black": 31}
    assert profile["balance"] == 0


def test_material_profile_counts_a_promoted_queen():
    session = GameSession(fen="8/P3k3/8/8/8/8/8/4K3 w - - 0 1")
    assert session.material_profile()["queens"] is False
    session.submit_move("a8=Q")
    profile = session.material_profile()
    assert profile["queens"] is True
    assert profile["non_pawn"] == {"white": 9, "black": 0}
    assert profile["balance"] == 9


# --- the rule, without an engine ---------------------------------------------


def test_the_constants_are_what_the_note_records():
    assert DRAW_OFFER_BAND_CP == 50
    assert ENDGAME_MAX_NON_PAWN_MATERIAL == 8


def test_a_level_rook_endgame_is_accepted():
    verdict = judge_draw_offer(
        positioned(DEAD_DRAWN_ROOK_ENDGAME), LEVEL, player_has_moved=True
    )
    assert verdict.accepted is True
    assert verdict.reason is None
    assert verdict.cp_engine_pov == 0
    assert verdict.material["queens"] is False
    assert verdict.material["non_pawn"] == {"white": 5, "black": 5}


def test_a_flat_middlegame_is_declined_as_not_an_endgame():
    verdict = judge_draw_offer(
        positioned(FLAT_MIDDLEGAME), LEVEL, player_has_moved=True
    )
    assert verdict.accepted is False
    assert verdict.reason == NOT_AN_ENDGAME


def test_an_endgame_the_engine_is_winning_is_declined_as_engine_ahead():
    # White-POV −300: the engine has black, so +300 from its side.
    verdict = judge_draw_offer(
        positioned(ENGINE_WINNING_ENDGAME),
        Evaluation(score_cp=-300, mate_in=None),
        player_has_moved=True,
    )
    assert verdict.accepted is False
    assert verdict.reason == ENGINE_AHEAD
    assert verdict.cp_engine_pov == 300


def test_a_position_the_player_is_winning_is_declined_as_player_ahead():
    """A draw is the player's to offer, not Glitch's to grab."""
    verdict = judge_draw_offer(
        positioned(PLAYER_WINNING_ENDGAME),
        Evaluation(score_cp=400, mate_in=None),
        player_has_moved=True,
    )
    assert verdict.accepted is False
    assert verdict.reason == PLAYER_AHEAD
    assert verdict.cp_engine_pov == -400


def test_an_offer_before_the_player_has_moved_is_too_early():
    verdict = judge_draw_offer(
        positioned(DEAD_DRAWN_ROOK_ENDGAME), LEVEL, player_has_moved=False
    )
    assert verdict.accepted is False
    assert verdict.reason == TOO_EARLY


def test_the_evaluation_is_read_from_the_engines_side():
    """The same White-POV number is the engine's advantage when the engine has
    white and the player's when it has black."""
    ahead_for_white = Evaluation(score_cp=200, mate_in=None)
    as_white_player = judge_draw_offer(
        positioned(DEAD_DRAWN_ROOK_ENDGAME, "white"),
        ahead_for_white,
        player_has_moved=True,
    )
    as_black_player = judge_draw_offer(
        positioned(DEAD_DRAWN_ROOK_ENDGAME, "black"),
        ahead_for_white,
        player_has_moved=True,
    )
    assert as_white_player.reason == PLAYER_AHEAD
    assert as_black_player.reason == ENGINE_AHEAD
    assert as_black_player.cp_engine_pov == 200


@pytest.mark.parametrize(
    ("score_cp", "reason"),
    [
        (DRAW_OFFER_BAND_CP, None),
        (-DRAW_OFFER_BAND_CP, None),
        (DRAW_OFFER_BAND_CP + 1, PLAYER_AHEAD),
        (-(DRAW_OFFER_BAND_CP + 1), ENGINE_AHEAD),
    ],
)
def test_the_band_is_inclusive_at_both_edges(score_cp, reason):
    # The player has white here, so a positive White-POV score is theirs.
    verdict = judge_draw_offer(
        positioned(DEAD_DRAWN_ROOK_ENDGAME),
        Evaluation(score_cp=score_cp, mate_in=None),
        player_has_moved=True,
    )
    assert verdict.reason == reason


@pytest.mark.parametrize(
    ("mate_in", "reason"),
    [(3, PLAYER_AHEAD), (-3, ENGINE_AHEAD)],
)
def test_a_forced_mate_either_way_is_a_decline(mate_in, reason):
    """No band is wide enough for a mate: `pov_cp` puts it on the `MATE_CP`
    scale, so "no forced mate either way" is the same check as the band."""
    verdict = judge_draw_offer(
        positioned(DEAD_DRAWN_ROOK_ENDGAME),
        Evaluation(score_cp=None, mate_in=mate_in),
        player_has_moved=True,
    )
    assert verdict.reason == reason
    assert abs(verdict.cp_engine_pov) == MATE_CP - 3
    assert verdict.mate_in == mate_in


def test_the_eval_reasons_outrank_the_material_one():
    """A middlegame the player is winning is declined as `player_ahead`, not as
    `not_an_endgame` — the more useful thing to hear."""
    verdict = judge_draw_offer(
        positioned(FLAT_MIDDLEGAME),
        Evaluation(score_cp=300, mate_in=None),
        player_has_moved=True,
    )
    assert verdict.reason == PLAYER_AHEAD


def test_the_endgame_threshold_at_its_edge():
    """Rook plus a minor piece each (8) is an endgame; rook plus two minors
    (11) is not; a queen alone (9) is not, whatever the sum says."""
    assert is_endgame({"queens": False, "non_pawn": {"white": 8, "black": 8}})
    assert not is_endgame({"queens": False, "non_pawn": {"white": 11, "black": 8}})
    assert not is_endgame({"queens": True, "non_pawn": {"white": 9, "black": 0}})
    assert not is_endgame({"queens": True, "non_pawn": {"white": 0, "black": 9}})


# --- the rule, against Stockfish --------------------------------------------
#
# One run per fixture: the synthetic evaluations above pin the rule's branches;
# these pin that the fixtures really are what they are called.


@pytest.fixture(scope="module")
def stockfish():
    if shutil.which("stockfish") is None:
        pytest.skip("stockfish binary not installed")
    with EnginePlayer() as engine:
        yield engine


@requires_stockfish
@pytest.mark.parametrize(
    ("fixture", "reason"),
    [
        pytest.param(DEAD_DRAWN_ROOK_ENDGAME, None, id="dead-drawn-rook-endgame"),
        pytest.param(FLAT_MIDDLEGAME, NOT_AN_ENDGAME, id="flat-middlegame"),
        pytest.param(ENGINE_WINNING_ENDGAME, ENGINE_AHEAD, id="engine-winning"),
        pytest.param(PLAYER_WINNING_ENDGAME, PLAYER_AHEAD, id="player-winning"),
    ],
)
def test_the_fixtures_judge_as_named_under_stockfish(stockfish, fixture, reason):
    session = positioned(fixture)
    verdict = judge_draw_offer(
        session, stockfish.evaluate_position(session), player_has_moved=True
    )
    assert verdict.reason == reason, verdict


# --- the tool ---------------------------------------------------------------


def tooled(fixture, evaluation=LEVEL, *, engine=None, player_color="white"):
    """A context on a fixture, with a coordinator and a registry as app assembly
    builds them."""
    engine = engine or FakeEngine(evaluation=evaluation)
    ctx = ToolContext(session=positioned(fixture, player_color), engine=engine)
    coordinator = TurnCoordinator(ctx)
    registry = build_registry(ctx, coordinator, atomic_exchange=False)
    return ctx, coordinator, registry


def test_offer_draw_is_offered_to_the_brain_on_every_live_turn():
    ctx = ToolContext(session=GameSession())
    assert "offer_draw" not in brain_tool_exclusions(ctx)
    for san in ("Nf3", "Nf6", "Ng1", "Ng8") * 2:
        ctx.session.submit_move(san)
    assert "offer_draw" not in brain_tool_exclusions(ctx)


def test_offer_draw_is_not_gated():
    """No question to ask: a decline changes nothing and an acceptance ends a
    position the rule has already judged drawn."""
    from chessapp.tools import CONFIRM_QUESTIONS, DESTRUCTIVE_TOOLS

    assert "offer_draw" not in DESTRUCTIVE_TOOLS
    assert "offer_draw" not in CONFIRM_QUESTIONS


def test_an_accepted_offer_ends_the_game_and_reports_every_fact():
    ctx, _, registry = tooled(DEAD_DRAWN_ROOK_ENDGAME)
    result = registry.dispatch("offer_draw", {})
    assert result["ok"] is True
    assert result["accepted"] is True
    assert result["reason"] is None
    assert result["evaluation"] == {"cp_engine_pov": 0, "mate_in": None}
    assert result["material"]["non_pawn"] == {"white": 5, "black": 5}
    assert result["outcome"] == {
        "termination": "agreement",
        "winner": None,
        "result": "1/2-1/2",
    }
    assert ctx.session.is_game_over()
    assert ctx.pending is None


def test_a_declined_offer_changes_nothing():
    ctx, coordinator, registry = tooled(FLAT_MIDDLEGAME)
    fen = ctx.session.fen()
    revision = ctx.session.revision
    result = registry.dispatch("offer_draw", {})
    assert result["ok"] is True
    assert result["accepted"] is False
    assert result["reason"] == NOT_AN_ENDGAME
    assert "outcome" not in result
    assert not ctx.session.is_game_over()
    assert ctx.session.fen() == fen
    assert ctx.session.revision == revision
    assert coordinator.turn_id == 1


def test_an_offer_on_a_finished_game_is_refused_for_good():
    ctx, _, registry = tooled(DEAD_DRAWN_ROOK_ENDGAME)
    ctx.session.resign("white")
    result = registry.dispatch("offer_draw", {})
    assert result["ok"] is False
    assert result["retry"] == "never"
    assert "already over" in result["error"]


def test_an_offer_with_no_engine_is_refused():
    ctx = ToolContext(session=positioned(DEAD_DRAWN_ROOK_ENDGAME))
    registry = build_registry(ctx)
    result = registry.dispatch("offer_draw", {})
    assert result["ok"] is False
    assert "engine" in result["error"]
    assert not ctx.session.is_game_over()


def test_the_tool_reads_investment_the_way_the_gate_does():
    """The gate's own notion of investment: as black, the engine's opening move
    is not the player's."""
    ctx = ToolContext(
        session=GameSession(fen=DEAD_DRAWN_ROOK_ENDGAME[0], player_color="black"),
        engine=FakeEngine(evaluation=LEVEL),
    )
    ctx.session.submit_move("Rb2")  # the engine's ply
    registry = build_registry(ctx)
    result = registry.dispatch("offer_draw", {})
    assert result["accepted"] is False
    assert result["reason"] == TOO_EARLY


def test_an_accepted_offer_spends_the_commands_destructive_budget():
    """One ending per command: an offer accepted after a resignation, or a
    resignation after an accepted offer, is refused by the budget."""
    ctx, coordinator, registry = tooled(DEAD_DRAWN_ROOK_ENDGAME)
    coordinator.begin_command()
    try:
        first = registry.dispatch("offer_draw", {})
        assert first["accepted"] is True
        second = registry.dispatch("resign", {})
    finally:
        coordinator.end_command()
    assert second["ok"] is False
    assert "already ran" in second["error"]
    assert ctx.session.outcome().termination == "agreement"


def test_a_declined_offer_spends_no_budget():
    ctx, coordinator, registry = tooled(FLAT_MIDDLEGAME)
    coordinator.begin_command()
    try:
        registry.dispatch("offer_draw", {})
        again = registry.dispatch("offer_draw", {})
    finally:
        coordinator.end_command()
    assert again["ok"] is True
    assert again["accepted"] is False


def test_an_offer_after_a_reset_in_the_same_command_is_refused():
    """ "New game and call it a draw" cannot end the fresh game: the budget is
    checked before anything is evaluated."""
    ctx, coordinator, registry = tooled(DEAD_DRAWN_ROOK_ENDGAME)
    coordinator.begin_command()
    try:
        ctx._confirming = True  # stand in for the player's yes to the reset
        registry.dispatch("new_game", {})
        ctx._confirming = False
        result = registry.dispatch("offer_draw", {})
    finally:
        coordinator.end_command()
    assert result["ok"] is False
    assert not ctx.session.is_game_over()


def test_an_accepted_offer_with_a_turn_open_abandons_the_owed_reply():
    ctx, coordinator, registry = tooled(
        DEAD_DRAWN_ROOK_ENDGAME, engine=FakeEngine("a7a6", evaluation=LEVEL)
    )
    assert coordinator.apply_player_move("Rc3").legal  # a turn is open
    assert coordinator.phase is TurnPhase.PLAYER_MOVE_APPLIED
    result = registry.dispatch("offer_draw", {})
    assert result["accepted"] is True
    assert coordinator.phase is TurnPhase.AWAITING_PLAYER
    assert coordinator.turn_id == 2
    assert ctx.session.move_history()[-1] == "Rc3", "no reply was played"


def test_a_declined_offer_with_a_turn_open_leaves_the_reply_owed():
    ctx, coordinator, registry = tooled(
        FLAT_MIDDLEGAME, engine=FakeEngine("a6a5", evaluation=LEVEL)
    )
    coordinator.apply_player_move("O-O")
    result = registry.dispatch("offer_draw", {})
    assert result["accepted"] is False
    assert coordinator.phase is TurnPhase.PLAYER_MOVE_APPLIED
    assert coordinator.turn_id == 1
    reply = coordinator.collect_engine_reply()
    assert reply is not None and reply.san == "a5"


def test_the_description_carries_the_relay_and_never_twice_guidance():
    registry = build_registry(ToolContext(session=GameSession()))
    for d in registry.definitions():
        if d["function"]["name"] == "offer_draw":
            desc = " ".join(d["function"]["description"].lower().split())
            break
    else:
        raise AssertionError("offer_draw not registered")
    assert "as soon as the player offers" in desc
    assert "do not decide the answer yourself" in desc
    assert "relay" in desc
    assert "never call it twice" in desc
    assert "claim_draw" in desc
    parameters = d["function"]["parameters"]
    assert parameters["properties"] == {}, "no arguments: the answer is the rule's"
    assert parameters["additionalProperties"] is False


# --- the endpoint -----------------------------------------------------------


def endpoint_client(fixture, evaluation=LEVEL):
    ctx = ToolContext(
        session=positioned(fixture), engine=FakeEngine(evaluation=evaluation)
    )
    return TestClient(create_app(ctx)), ctx


def test_offer_draw_endpoint_accepts_and_returns_outcome_and_state():
    client, ctx = endpoint_client(DEAD_DRAWN_ROOK_ENDGAME)
    body = client.post("/api/game/offer-draw", json={}).json()
    assert body["accepted"] is True
    assert body["reason"] is None
    assert body["outcome"]["termination"] == "agreement"
    assert body["state"]["game_over"] is True
    assert body["state"]["outcome"]["result"] == "1/2-1/2"
    assert body["evaluation"] == {"cp_engine_pov": 0, "mate_in": None}
    assert body["material"]["queens"] is False


def test_offer_draw_endpoint_declines_with_the_reason_and_an_unchanged_board():
    client, ctx = endpoint_client(FLAT_MIDDLEGAME)
    before = client.get("/api/state").json()
    body = client.post("/api/game/offer-draw", json={}).json()
    assert body["accepted"] is False
    assert body["reason"] == NOT_AN_ENDGAME
    assert "outcome" not in body
    assert body["state"] == before


def test_offer_draw_endpoint_is_409_on_a_finished_game():
    client, ctx = endpoint_client(DEAD_DRAWN_ROOK_ENDGAME)
    ctx.session.resign("white")
    response = client.post("/api/game/offer-draw", json={})
    assert response.status_code == 409
    assert "already over" in response.json()["detail"]


def test_offer_draw_endpoint_is_409_without_an_engine():
    ctx = ToolContext(session=positioned(DEAD_DRAWN_ROOK_ENDGAME))
    response = TestClient(create_app(ctx)).post("/api/game/offer-draw", json={})
    assert response.status_code == 409


def test_offer_draw_endpoint_rejects_a_stale_version():
    client, ctx = endpoint_client(DEAD_DRAWN_ROOK_ENDGAME)
    stale = client.get("/api/state").json()["version"]
    assert ctx.session.submit_move("Rc3").legal
    response = client.post("/api/game/offer-draw", json={"version": stale})
    assert response.status_code == 409
    assert response.json()["stale"] is True
    assert not ctx.session.is_game_over()


def test_an_accepted_offer_bumps_the_version_and_a_decline_does_not():
    client, _ = endpoint_client(FLAT_MIDDLEGAME)
    before = client.get("/api/state").json()["version"]
    client.post("/api/game/offer-draw", json={})
    assert client.get("/api/state").json()["version"] == before

    client, _ = endpoint_client(DEAD_DRAWN_ROOK_ENDGAME)
    before = client.get("/api/state").json()["version"]
    client.post("/api/game/offer-draw", json={})
    assert client.get("/api/state").json()["version"] == before + 1


def test_an_accepted_offer_drops_an_armed_question():
    """Whatever was armed was a question about a game that is now over."""
    client, ctx = endpoint_client(DEAD_DRAWN_ROOK_ENDGAME)
    assert client.post("/api/game/resign", json={}).status_code == 409  # armed
    assert ctx.pending is not None
    client.post("/api/game/offer-draw", json={})
    assert ctx.pending is None
    assert ctx.session.outcome().termination == "agreement"


# --- the pipeline -----------------------------------------------------------
#
# The coordinator interaction `docs/draw-offer.md` works out: an offer can
# arrive in the same command as a move, with the engine's reply owed.


def pipeline_client(fixture, *responses, evaluation=LEVEL, reply_uci):
    from fakes import scripted_app

    ctx = ToolContext(
        session=positioned(fixture), engine=FakeEngine(reply_uci, evaluation=evaluation)
    )
    app, _ = scripted_app(ctx, *responses)
    return TestClient(app), ctx


def move_then_offer(san: str, text: str):
    from chessapp.brain import AgentResponse, ToolCall

    return AgentResponse(
        text=text,
        tool_calls=(
            ToolCall(name="make_move", args={"move": san}),
            ToolCall(name="offer_draw", args={}),
        ),
    )


def test_a_declined_offer_beside_a_move_still_gets_the_engines_reply():
    """The reply is still owed, and the pipeline collects and announces it."""
    client, ctx = pipeline_client(
        FLAT_MIDDLEGAME,
        move_then_offer("O-O", "Nah, too much on the board still."),
        reply_uci="a6a5",
    )
    body = client.post(
        "/api/command", json={"text": "castle, and wanna call it a draw?"}
    ).json()
    assert body["tool_results"][1]["result"]["accepted"] is False
    assert ctx.session.move_history()[-2:] == ["O-O", "a5"]
    assert not ctx.session.is_game_over()
    assert body["commentary"] == "Nah, too much on the board still.\n\na5."


def test_an_accepted_offer_beside_a_move_ends_the_game_with_no_reply():
    client, ctx = pipeline_client(
        DEAD_DRAWN_ROOK_ENDGAME,
        move_then_offer("Rc3", "Sure, half a point each."),
        reply_uci="a7a6",
    )
    body = client.post(
        "/api/command", json={"text": "Rc3, and let's call it a draw"}
    ).json()
    assert body["tool_results"][1]["result"]["accepted"] is True
    assert ctx.session.move_history()[-1] == "Rc3", "no reply on a game that ended"
    assert ctx.session.outcome().termination == "agreement"
    assert body["state"]["game_over"] is True
    assert body["commentary"] == "Sure, half a point each."


def test_a_decline_narrated_as_a_draw_is_guarded():
    """The honesty guard: "we drew" on a decline is an ending the board does not
    back, and the player is told the truth instead."""
    from chessapp.api import UNTRUE_CLAIM_REPLY
    from chessapp.brain import AgentResponse, ToolCall

    client, ctx = pipeline_client(
        FLAT_MIDDLEGAME,
        AgentResponse(
            text="Game over, we drew.",
            tool_calls=(ToolCall(name="offer_draw", args={}),),
        ),
        reply_uci="a6a5",
    )
    body = client.post("/api/command", json={"text": "call it a draw?"}).json()
    assert not ctx.session.is_game_over()
    assert body["commentary"] == UNTRUE_CLAIM_REPLY


def test_the_verdicts_number_may_be_quoted_from_either_side():
    """`_analysis_numbers` learns the offer's evaluation, both signs: an honest
    "up three" survives whichever side it is said from, an invented number
    does not."""
    from chessapp.api import UNVERIFIED_CLAIM_REPLY
    from chessapp.brain import AgentResponse, ToolCall

    def narrated(text: str) -> str:
        client, _ = pipeline_client(
            ENGINE_WINNING_ENDGAME,
            AgentResponse(
                text=text, tool_calls=(ToolCall(name="offer_draw", args={}),)
            ),
            evaluation=Evaluation(score_cp=-300, mate_in=None),
            reply_uci="a1a2",
        )
        return client.post("/api/command", json={"text": "draw?"}).json()["commentary"]

    mine = "Nope. I'm up 3.0 here, play on."
    yours = "Nope. You're down 3.0, play on."
    assert narrated(mine) == mine
    assert narrated(yours) == yours
    assert narrated("Nope. I'm up 7.0 here, play on.") == UNVERIFIED_CLAIM_REPLY
