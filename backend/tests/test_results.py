"""The results log, and the tally it keeps across games (#373)."""

import json

from chessapp.api import TurnCoordinator, _agent_state_dict, narrator_facts
from chessapp.game import GameSession
from chessapp.results import RESULTS_FILENAME, ResultsLog, counts_of
from chessapp.tools import (
    LIVE_CHECKPOINT_FILENAME,
    ToolContext,
    live_checkpoint,
    restore_live_checkpoint,
)
from fakes import FakeEngine

# The fool's mate: black mates on move two.
FOOLS_MATE = ("f3", "e5", "g4", "Qh4#")
# The scholar's mate: white mates on move four.
SCHOLARS_MATE = ("e4", "e5", "Bc4", "Nc6", "Qh5", "Nf6", "Qxf7#")


def _record(log: ResultsLog, game_id: str, winner, difficulty="casual") -> None:
    log.record(
        game_id,
        player_color="white",
        difficulty=difficulty,
        result={"player": "1-0", "opponent": "0-1", None: "1/2-1/2"}[winner],
        winner=winner,
        termination="checkmate" if winner else "stalemate",
        from_setup=False,
    )


def _ctx(tmp_path=None, **kwargs) -> ToolContext:
    return ToolContext(
        session=GameSession(**kwargs), engine=FakeEngine(), save_dir=tmp_path
    )


def _play(ctx: ToolContext, *sans: str) -> None:
    """Moves one at a time, observed after each, the way dispatch sees them."""
    for san in sans:
        assert ctx.session.submit_move(san).legal, san
        ctx.observe_ledger()


def _lines(tmp_path) -> list[dict]:
    path = tmp_path / RESULTS_FILENAME
    return [json.loads(line) for line in path.read_text().splitlines()]


# --- the log ----------------------------------------------------------------------


def test_an_empty_log_tallies_zeros():
    assert ResultsLog().tally() == {
        "games": 0,
        "player_won": 0,
        "engine_won": 0,
        "drawn": 0,
        "by_difficulty": {},
    }


def test_the_tally_counts_overall_and_per_difficulty():
    log = ResultsLog()
    _record(log, "a" * 32, "player")
    _record(log, "b" * 32, "opponent")
    _record(log, "c" * 32, "opponent", difficulty="advanced")
    _record(log, "d" * 32, None, difficulty="advanced")
    tally = log.tally()
    assert (tally["games"], tally["player_won"], tally["engine_won"]) == (4, 1, 2)
    assert tally["drawn"] == 1
    assert tally["by_difficulty"] == {
        "casual": {"games": 2, "player_won": 1, "engine_won": 1, "drawn": 0},
        "advanced": {"games": 2, "player_won": 0, "engine_won": 1, "drawn": 1},
    }


def test_the_latest_line_for_a_game_wins():
    log = ResultsLog()
    _record(log, "a" * 32, "player")
    log.withdraw("a" * 32)
    assert log.tally()["games"] == 0
    _record(log, "a" * 32, "opponent")
    assert log.tally()["engine_won"] == 1 and log.tally()["games"] == 1


def test_withdrawing_a_game_with_no_result_writes_nothing(tmp_path):
    log = ResultsLog(tmp_path / RESULTS_FILENAME)
    log.withdraw("a" * 32)
    assert not (tmp_path / RESULTS_FILENAME).exists()


def test_the_log_survives_a_reload_and_skips_a_line_it_cannot_read(tmp_path):
    path = tmp_path / RESULTS_FILENAME
    log = ResultsLog(path)
    _record(log, "a" * 32, "player")
    _record(log, "b" * 32, "opponent")
    log.withdraw("b" * 32)
    with path.open("a") as out:
        out.write("not json\n")
        out.write(json.dumps({"game_id": "c" * 32, "winner": "nobody"}) + "\n")
    assert ResultsLog.load(path).tally() == log.tally()


def test_a_log_that_cannot_be_written_still_counts(tmp_path):
    log = ResultsLog(tmp_path / "missing-dir" / RESULTS_FILENAME)
    _record(log, "a" * 32, "player")
    assert log.tally()["player_won"] == 1


def test_counts_of_names_every_count_the_tally_states():
    log = ResultsLog()
    _record(log, "a" * 32, "player")
    _record(log, "b" * 32, "opponent", difficulty="advanced")
    counts = counts_of(log.tally())
    assert ("games", 2) in counts and ("games", 1) in counts
    assert ("player_won", 1) in counts and ("player_won", 0) in counts


# --- fed from the ledger ------------------------------------------------------------


def test_a_checkmate_is_recorded_with_the_game_it_ended(tmp_path):
    ctx = _ctx(tmp_path)
    ctx.settings.tier = "advanced"
    _play(ctx, *SCHOLARS_MATE)
    [line] = _lines(tmp_path)
    assert line["game_id"] == ctx.session.game_id
    assert (line["winner"], line["result"], line["termination"]) == (
        "player",
        "1-0",
        "checkmate",
    )
    assert (line["player_color"], line["difficulty"]) == ("white", "advanced")
    assert line["from_setup"] is False


def test_a_resignation_and_an_agreed_draw_are_recorded():
    ctx = _ctx()
    _play(ctx, "e4")
    ctx.session.resign("white")
    ctx.observe_ledger()
    ctx.session.new_game()
    ctx.observe_ledger()
    _play(ctx, "d4")
    ctx.session.agree_draw()
    ctx.observe_ledger()
    tally = ctx.results.tally()
    assert (tally["games"], tally["engine_won"], tally["drawn"]) == (2, 1, 1)


def test_a_takeback_of_the_ending_withdraws_it_and_a_second_ending_counts_once(
    tmp_path,
):
    ctx = _ctx(tmp_path)
    _play(ctx, *SCHOLARS_MATE)
    assert ctx.results.tally()["player_won"] == 1
    ctx.session.undo(1)
    ctx.observe_ledger()
    assert ctx.results.tally()["games"] == 0
    _play(ctx, "Qxf7#")
    assert ctx.results.tally()["games"] == 1
    assert [line.get("withdrawn", False) for line in _lines(tmp_path)] == [
        False,
        True,
        False,
    ]


def test_a_takeback_in_a_live_game_writes_nothing(tmp_path):
    ctx = _ctx(tmp_path)
    _play(ctx, "e4", "e5")
    ctx.session.undo(2)
    ctx.observe_ledger()
    assert not (tmp_path / RESULTS_FILENAME).exists()


def test_a_game_lost_as_black_is_the_engines_win():
    ctx = _ctx(player_color="black")
    _play(ctx, *SCHOLARS_MATE)
    assert ctx.results.tally()["engine_won"] == 1


def test_a_finished_save_resumed_is_not_counted_again():
    ctx = _ctx()
    _play(ctx, *FOOLS_MATE)
    assert ctx.results.tally()["games"] == 1
    finished = GameSession.from_dict(ctx.session.to_dict())
    finished.renew_game_id()
    ctx.ledger.expect_resume("fools")
    ctx.replace_session(finished, ctx.transcript)
    ctx.observe_ledger()
    assert ctx.results.tally()["games"] == 1


def test_a_restart_restoring_a_finished_game_counts_it_once(tmp_path):
    ctx = _ctx(tmp_path)
    _play(ctx, *FOOLS_MATE)
    (tmp_path / LIVE_CHECKPOINT_FILENAME).write_text(json.dumps(live_checkpoint(ctx)))

    restarted = _ctx(tmp_path)
    assert restore_live_checkpoint(restarted)
    restarted.observe_ledger()
    assert restarted.results.tally()["games"] == 1
    assert len(_lines(tmp_path)) == 1


def test_a_restart_then_a_takeback_withdraws_the_restored_ending(tmp_path):
    ctx = _ctx(tmp_path)
    _play(ctx, *FOOLS_MATE)
    (tmp_path / LIVE_CHECKPOINT_FILENAME).write_text(json.dumps(live_checkpoint(ctx)))

    restarted = _ctx(tmp_path)
    assert restore_live_checkpoint(restarted)
    restarted.session.undo(1)
    restarted.observe_ledger()
    assert restarted.results.tally()["games"] == 0


def test_without_a_save_dir_nothing_is_written(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ctx = _ctx()
    _play(ctx, *FOOLS_MATE)
    assert ctx.results.tally()["games"] == 1
    assert list(tmp_path.iterdir()) == []


def test_both_phases_are_shown_the_tally():
    ctx = _ctx()
    _play(ctx, *FOOLS_MATE)
    assert _agent_state_dict(ctx)["results"]["engine_won"] == 1
    assert narrator_facts(ctx, TurnCoordinator(ctx))["results"]["engine_won"] == 1
