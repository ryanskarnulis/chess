"""`facts`: a turn's facts, assembled from evidence a trace can carry (#367).

The live path (`api._verified_facts`) and the offline scorer both go through
`facts.assemble`, so the property worth pinning is that the evidence holds
everything the assembly reads: written to JSON and read back, it assembles to
the same facts. The facts themselves are specified in `test_api.py` and
`test_honesty.py`.
"""

import json

import chess

from chessapp.facts import (
    TurnEvidence,
    analysis_moves,
    analysis_numbers,
    assemble,
    settings_of,
)
from chessapp.game import GameSession


def _evidence(session: GameSession, **overrides) -> TurnEvidence:
    fields = {
        "session": session.to_dict(),
        "settings": settings_of(False, "normal", "casual"),
        "tool_results": [],
        "engine_reply_san": None,
        "fen_before": chess.STARTING_FEN,
    }
    return TurnEvidence(**{**fields, **overrides})


def _through_json(evidence: TurnEvidence) -> TurnEvidence:
    """What a trace record carries, and what a reader rebuilds from it."""
    tools = [
        {"name": r["name"], "args": {}, "result": r["result"]}
        for r in evidence.tool_results
    ]
    record = json.loads(json.dumps({"evidence": evidence.as_trace(), "tools": tools}))
    return TurnEvidence.from_trace(record["evidence"], record["tools"])


def test_evidence_survives_the_trace_and_assembles_to_the_same_facts():
    session = GameSession(player_color="black")
    for san in ("e4", "d5", "exd5"):
        assert session.submit_move(san).legal
    between = session.fen()
    assert session.submit_move("Qxd5").legal
    evidence = _evidence(
        session,
        tool_results=[
            {"name": "make_move", "result": {"ok": True, "legal": True, "san": "d5"}},
            {"name": "set_verbosity", "result": {"ok": True}},
        ],
        engine_reply_san="exd5",
        fens_observed=(between,),
        pending_reply_fen=between,
    )

    rebuilt = _through_json(evidence)

    assert rebuilt == evidence
    assert assemble(rebuilt) == assemble(evidence)


def test_the_session_carries_the_players_side_and_the_history():
    """What a FEN alone cannot say: who the player is, and who played what."""
    session = GameSession(player_color="black")
    for san in ("e4", "d5", "exd5"):
        assert session.submit_move(san).legal

    facts = assemble(_evidence(session))

    assert facts.moves_by_player == {"d5"}
    assert facts.moves_by_opponent == {"e4", "exd5"}
    assert facts.captured_by_opponent == {"pawn"}
    assert facts.material[0] == -1, "the player, playing black, is a pawn down"


def test_a_resignation_in_the_session_is_the_ending():
    session = GameSession()
    session.resign("black")

    facts = assemble(_evidence(session))

    assert (facts.ended, facts.winner, facts.termination) == (
        True,
        "player",
        "resignation",
    )


def test_only_a_named_tier_is_a_claimable_difficulty():
    assert settings_of(True, "low", None) == {"voice": "on", "verbosity": "low"}
    assert settings_of(False, "high", "advanced") == {
        "voice": "off",
        "verbosity": "high",
        "difficulty": "advanced",
    }


# --- #320: analysis from the player's side -----------------------------------


def test_a_player_side_score_backs_both_directions():
    """ "You're up 1.5" and "I'm down 1.5" quote one fact, so both signs are
    reported, and a mate's distance whoever delivers it."""
    numbers = analysis_numbers(
        [
            {
                "name": "evaluate_position",
                "result": {"ok": True, "player_advantage_cp": 150, "mate": None},
            },
            {
                "name": "get_best_moves",
                "result": {
                    "ok": True,
                    "moves": [
                        {
                            "san": "Qh5",
                            "player_advantage_cp": None,
                            "mate": {"in": 3, "for": "glitch"},
                        },
                    ],
                },
            },
        ]
    )
    assert {"150", "1.5", "+1.5", "-150", "-1.5", "3"} <= numbers


def test_a_pre_320_trace_still_reads_its_white_pov_numbers():
    numbers = analysis_numbers(
        [
            {
                "name": "evaluate_position",
                "result": {"ok": True, "score_cp": -80, "mate_in": None},
            }
        ]
    )
    assert {"-80", "-0.8"} <= numbers
    assert "0.8" not in numbers


def test_the_engines_line_is_a_reported_move():
    moves = analysis_moves(
        [
            {
                "name": "evaluate_position",
                "result": {"ok": True, "line": ["Nf6", "Bc4"]},
            },
            {
                "name": "get_best_moves",
                "result": {"ok": True, "moves": [{"san": "e5", "line": ["e5", "Nf3"]}]},
            },
        ]
    )
    assert moves == {"Nf6", "Bc4", "e5", "Nf3"}


def test_the_turns_verdicts_are_the_players_side():
    from chessapp.facts import analysis_advantages

    results = [
        {
            "name": "evaluate_position",
            "result": {"ok": True, "player_advantage_cp": 40},
        },
        {
            "name": "evaluate_position",
            "result": {
                "ok": True,
                "player_advantage_cp": None,
                "mate": {"in": 2, "for": "glitch"},
            },
        },
        {
            "name": "get_best_moves",
            "result": {
                "ok": True,
                "moves": [
                    {"san": "e5", "player_advantage_cp": -20},
                    {"san": "a6", "player_advantage_cp": -90},
                ],
            },
        },
        # Before #320: White-POV, turned with the player's color.
        {
            "name": "evaluate_position",
            "result": {"ok": True, "score_cp": 70, "mate_in": None},
        },
        {"name": "evaluate_position", "result": {"ok": False, "error": "engine"}},
    ]
    assert analysis_advantages(results, "black") == (40, -(100_000 - 2), -20, -70)
