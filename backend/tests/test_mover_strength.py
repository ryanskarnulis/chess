"""`scripts/mover_strength.py` — the mover's strength and its history (#476).

The run itself needs the GPU; everything it decides is tested here off it:
games played through the app's own move source (an illegal attempt counted,
a fallback marked, a long game adjudicated), a crashed run resumed without
replaying or mixing, the Elo read from a score, the history line and table,
and the busy-server check.
"""

from __future__ import annotations

import json

import httpx
import pytest

from chessapp.move_source import GlitchSource
from chessapp.mover import Mover
from chessapp.profiles import DEFAULT_PROFILE
from chessapp.provider import ProviderError
from fakes import ScriptedProvider, tool_calls_turn
from mover_strength import (
    GAME,
    HEADER,
    busy,
    estimate_elo,
    expected,
    glitch_score,
    main,
    pending,
    play_game,
    rates,
    run,
    schedule,
    summarize,
    trend,
)


class ScriptedEngine:
    """Plays its moves in order, whatever the position; records levels."""

    def __init__(self, *moves: str) -> None:
        self.moves = list(moves)
        self.elos: list[int] = []

    def choose_move(self, session) -> str:
        return self.moves.pop(0)

    def set_elo(self, elo: int) -> None:
        self.elos.append(elo)


def _play(move: str):
    return tool_calls_turn(("play_move", {"move": move, "intent": "testing"}))


def _source(*turns, fallback: ScriptedEngine | None = None) -> GlitchSource:
    mover = Mover(ScriptedProvider(*turns), DEFAULT_PROFILE)
    return GlitchSource(mover, fallback or ScriptedEngine())


def _header(**extra):
    return {
        "kind": HEADER,
        "model": "m",
        "profile_sha": "abc",
        "mover": {"max_tokens": 4096},
        "git_sha": "deadbee",
        "levels": [1320],
        "games_per_level": 2,
        "max_plies": 200,
        **extra,
    }


def _game(index, level, score, moves=()):
    return {
        "kind": GAME,
        "index": index,
        "level": level,
        "score": score,
        "adjudicated": False,
        "moves": list(moves),
    }


def _move(chooser="glitch", attempts=("played",), ms=(1000,), reason=""):
    return {
        "chooser": chooser,
        "fallback_reason": reason,
        "attempts": list(attempts),
        "calls": [{"status": "ok", "ms": m} for m in ms],
    }


def test_a_game_plays_through_the_move_source_and_counts_an_illegal_try():
    # Glitch as White walks into the fool's mate: his first reply is illegal
    # and refused, the second plays f3, the third g4.
    source = _source(_play("Ke2"), _play("f3"), _play("g4"))
    game = play_game(
        source, ScriptedEngine("e7e5", "d8h4"), index=0, level=1320, colour="white"
    )
    assert game["result"] == "0-1"
    assert game["termination"] == "checkmate"
    assert game["score"] == 0.0
    assert game["sans"] == ["f3", "e5", "g4", "Qh4#"]
    assert [m["attempts"] for m in game["moves"]] == [["illegal", "played"], ["played"]]
    assert [m["chooser"] for m in game["moves"]] == ["glitch", "glitch"]


def test_a_failed_mover_is_the_apps_fallback_and_is_marked():
    source = _source(ProviderError("down"), fallback=ScriptedEngine("f2f3", "g2g4"))
    game = play_game(
        source, ScriptedEngine("e7e5", "d8h4"), index=0, level=1320, colour="white"
    )
    assert game["result"] == "0-1"
    assert [m["chooser"] for m in game["moves"]] == ["fallback", "fallback"]
    assert {m["fallback_reason"] for m in game["moves"]} == {"provider"}


def test_glitch_as_black_scores_from_his_side():
    # White blunders into the fool's mate against Glitch.
    source = _source(_play("e5"), _play("Qh4#"))
    game = play_game(
        source, ScriptedEngine("f2f3", "g2g4"), index=1, level=1600, colour="black"
    )
    assert game["result"] == "0-1"
    assert game["score"] == 1.0


def test_a_game_past_the_ply_cap_is_an_adjudicated_draw():
    source = _source(_play("Nf3"), _play("Ng1"))
    game = play_game(
        source,
        ScriptedEngine("g8f6", "f6g8"),
        index=0,
        level=1320,
        colour="white",
        max_plies=4,
    )
    assert game["result"] == "1/2-1/2"
    assert game["adjudicated"] is True
    assert game["termination"] == "max_plies"
    assert game["score"] == 0.5


def test_scores():
    assert glitch_score("1-0", "white") == 1.0
    assert glitch_score("1-0", "black") == 0.0
    assert glitch_score("0-1", "black") == 1.0
    assert glitch_score("1/2-1/2", "white") == 0.5


def test_the_schedule_alternates_colours_within_each_level():
    assert schedule([1320, 1600], 2) == [
        (0, 1320, "white"),
        (1, 1320, "black"),
        (2, 1600, "white"),
        (3, 1600, "black"),
    ]


def test_a_resumed_run_plays_only_the_missing_games(tmp_path):
    report = tmp_path / "r.jsonl"
    report.write_text(
        json.dumps(_header()) + "\n" + json.dumps(_game(0, 1320, 1.0)) + "\n"
    )
    opponent = ScriptedEngine("f2f3", "g2g4")
    run(
        _header(),
        report,
        _source(_play("e5"), _play("Qh4#")),
        opponent,
        say=lambda _: None,
    )
    records = [json.loads(line) for line in report.read_text().splitlines()]
    assert [r["kind"] for r in records] == [HEADER, GAME, GAME]
    assert records[2]["index"] == 1
    assert records[2]["colour"] == "black"
    assert opponent.elos == [1320]


def test_a_new_report_starts_with_its_header(tmp_path):
    report = tmp_path / "r.jsonl"
    run(
        _header(games_per_level=1),
        report,
        _source(_play("f3"), _play("g4")),
        ScriptedEngine("e7e5", "d8h4"),
        say=lambda _: None,
    )
    kinds = [json.loads(line)["kind"] for line in report.read_text().splitlines()]
    assert kinds == [HEADER, GAME]


def test_a_report_from_another_profile_is_refused_not_mixed():
    with pytest.raises(SystemExit, match="profile_sha"):
        pending([_header(profile_sha="old")], _header(profile_sha="new"))


def test_an_even_score_at_one_level_is_that_level():
    games = [_game(0, 1500, 1.0), _game(1, 1500, 0.0)]
    elo = estimate_elo(games)
    assert elo["elo"] == 1500
    assert elo["low"] < 1500 < elo["high"]


def test_the_rating_makes_the_expected_score_the_actual_one():
    games = [_game(0, 1320, 1.0), _game(1, 1320, 0.5), _game(2, 1600, 0.0)]
    elo = estimate_elo(games)["elo"]
    total = sum(expected(elo, g["level"]) for g in games)
    assert total == pytest.approx(1.5, abs=0.01)


def test_a_whitewash_is_a_ceiling_only():
    games = [_game(i, level, 0.0) for i, level in enumerate([1320] * 4 + [1600] * 4)]
    elo = estimate_elo(games)
    assert elo["elo"] is None
    assert elo["low"] is None
    assert elo["high"] < 1320


def test_a_sweep_is_a_floor_only():
    elo = estimate_elo([_game(i, 1320, 1.0) for i in range(4)])
    assert elo["elo"] is None and elo["high"] is None
    assert elo["low"] > 1320 - 400


def test_rates_count_illegal_tries_fallbacks_and_time():
    games = [
        _game(
            0,
            1320,
            0.0,
            [
                _move(attempts=("illegal", "played"), ms=(1000, 3000)),
                _move(chooser="fallback", attempts=(), ms=(), reason="deadline"),
                _move(ms=(2000,)),
            ],
        )
    ]
    found = rates(games)
    assert found["moves"] == 3
    assert found["attempts"] == 3
    assert found["illegal_attempts"] == 1
    assert found["moves_with_illegal"] == 1
    assert found["fallbacks"] == 1
    assert found["fallback_kinds"] == {"deadline": 1}
    assert found["seconds_per_move_p50"] == 2.0


def test_a_report_becomes_one_history_line_and_a_row():
    records = [
        _header(levels=[1320, 1600]),
        _game(0, 1320, 0.5, [_move()]),
        _game(1, 1600, 0.0, [_move()]),
    ]
    line = summarize(records, label="first", date="2026-10-11")
    assert line["model"] == "m"
    assert line["profile_sha"] == "abc"
    assert line["games"] == 2
    assert line["by_level"] == {
        "1320": {"points": 0.5, "games": 1},
        "1600": {"points": 0.0, "games": 1},
    }
    assert line["fallbacks"] == 0
    table = trend([line, {**line, "model": "other"}], model="m")
    assert "1320:0.5/1 1600:0/1" in table
    assert "other" not in table
    assert table.count("\n") == 1


def test_append_writes_the_history(tmp_path, capsys):
    report = tmp_path / "r.jsonl"
    history = tmp_path / "h.jsonl"
    report.write_text(
        "\n".join(json.dumps(r) for r in [_header(), _game(0, 1320, 0.0, [_move()])])
    )
    assert main(["append", str(report), "--label", "x", "--history", str(history)]) == 0
    assert json.loads(history.read_text())["label"] == "x"
    assert main(["trend", "--history", str(history)]) == 0
    assert "m " in capsys.readouterr().out


def _server(routes):
    def handler(request: httpx.Request) -> httpx.Response:
        body = routes.get(request.url.path)
        if body is None:
            return httpx.Response(404)
        return httpx.Response(200, json=body)

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_a_generating_slot_behind_llama_swap_is_busy():
    client = _server(
        {
            "/running": {"running": [{"model": "g", "state": "ready"}]},
            "/upstream/g/slots": [
                {"id": 0, "is_processing": False},
                {"id": 1, "is_processing": True},
            ],
        }
    )
    assert busy(client, "http://x/v1") == ["g: slot 1 is generating"]


def test_an_idle_bare_server_is_not_busy():
    client = _server({"/slots": [{"id": 0, "is_processing": False}]})
    assert busy(client, "http://x/v1") == []
