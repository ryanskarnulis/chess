"""`facts`: a turn's facts, assembled from evidence a trace can carry (#367).

The live path (`api._verified_facts`) and the offline scorer both go through
`facts.assemble`, so the property worth pinning is that the evidence holds
everything the assembly reads: written to JSON and read back, it assembles to
the same facts. The facts themselves are specified in `test_api.py` and
`test_honesty.py`.
"""

import json

import chess

from chessapp.facts import TurnEvidence, assemble, settings_of
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
