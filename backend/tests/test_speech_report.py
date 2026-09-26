"""`scripts/speech_report.py`: the report's selection and its rendering (#367).

The counting is `test_speech_accuracy.py`'s; this pins what a reader sees.
"""

import json

import chess

import speech_report
from chessapp.facts import TurnEvidence, settings_of
from chessapp.game import GameSession


def _turn(draft: str, ts: str, route: str = "brain") -> dict:
    session = GameSession()
    for san in ("e4", "d5", "exd5"):
        assert session.submit_move(san).legal
    evidence = TurnEvidence(
        session=session.to_dict(),
        settings=settings_of(False, "normal", None),
        tool_results=[],
        engine_reply_san=None,
        fen_before=chess.STARTING_FEN,
    )
    return {
        "ts": ts,
        "kind": "turn",
        "schema": 3,
        "route": route,
        "utterance": "take it",
        "draft": draft,
        "evidence": evidence.as_trace(),
        "tools": [],
    }


LEGACY = {
    "ts": "2026-09-05T10:00:00",
    "kind": "turn",
    "schema": 1,
    "route": "fast_path",
    "utterance": "e4",
    "model_calls": 1,
    "commentary": "Took your rook, easy.",
    "guarded": False,
    "suppressed": "",
    "fen_before": chess.STARTING_FEN,
    "fen_after": chess.STARTING_FEN,
    "engine_reply": None,
    "tools": [],
}


def test_selection_by_time_and_route():
    records = [
        _turn("You took my pawn.", "2026-09-26T09:00:00"),
        _turn("You took my queen.", "2026-09-26T11:00:00", route="fast_path"),
    ]

    assert speech_report.select(records, since="2026-09-26T10") == records[1:]
    assert speech_report.select(records, route="brain") == records[:1]


def test_the_report_prints_the_denominator_the_families_and_the_lies():
    tally = speech_report.tally(
        [
            _turn("You took my pawn.", "2026-09-26T09:00:00"),
            _turn("You took my queen.", "2026-09-26T11:00:00"),
            LEGACY,
        ]
    )

    text = speech_report.render(tally)

    assert "**50.0%** — 1/2 claims backed, over 3 turns (1 legacy records)." in text
    assert "| `capture` | 2 | 1 | 1 | 50.0% |" in text
    assert "| `capture` | 1 | not decidable from a legacy record |" in text
    assert '"You took my queen."' in text


def test_a_trace_with_no_claims_says_so_rather_than_scoring_perfect():
    text = speech_report.render(speech_report.tally([_turn("Nice.", "2026")]))

    assert "**—** — 0/0 claims backed" in text
    assert "None." in text


def test_main_reads_trace_files_and_prints_json(tmp_path, capsys):
    path = tmp_path / "turns.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(r)
            for r in (
                _turn("You took my queen.", "2026-09-26T11:00:00"),
                {"kind": "serving", "schema": 3},
            )
        )
        + "\n"
    )

    assert speech_report.main([str(path), "--json"]) == 0

    summary = json.loads(capsys.readouterr().out)
    assert summary["families"] == {"capture": {"made": 1, "backed": 0}}
