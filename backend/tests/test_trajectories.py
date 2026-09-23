"""Composed interaction trajectories, in CI (#318).

`trajectory.py` is the machinery: a seeded walk through the shipped app with a
scripted planner and a deterministic engine, checking every invariant after
every step. This file is what runs:

- **The corpus.** A fixed range of seeds, each one walk. Bounded so the whole
  corpus costs CI about as much as one of the larger test files; widen it
  locally with `CHESSAPP_TRAJ_SEEDS=5000` for a soak. Measured when it was
  sized, each bug put back one at a time over 100–300 walks: #314's fix
  reverted, 239/300 walks red; #315's, 176/300; #316's closer left unbounded,
  35/100; #281's origin check dropped from the confirmation read, 7/100;
  #291's idempotent replay dropped, 45/100.
- **Named regressions.** The shortest step sequences that reproduced a real
  bug, each tagged with what it is: `reproduces` (went red on the build that
  had the bug — the fix reverted, or its check removed) or `prevents` (a guard
  against a shape nothing has broken yet). An open bug the walks found is a
  strict xfail on its issue, so fixing it fails the test until the marker
  comes off.
- **The instrument's own checks.** Every invariant must be able to fail, and a
  walk must be a pure function of its seed. A checker that cannot fire is
  worse than none: the corpus would read green for having looked at nothing.
- **Replay.** `CHESSAPP_TRAJ_SEED=<n> CHESSAPP_TRAJ_LENGTH=<k> pytest
  tests/test_trajectories.py -k replay -s` reruns one walk and prints it, the
  command every breach report carries.
"""

from __future__ import annotations

import os
from pathlib import Path

import chess.engine
import pytest

from trajectory import (
    DELEGATES,
    INVARIANTS,
    LATE_WORDS,
    PANEL,
    InvariantBreach,
    Observed,
    Pending,
    Step,
    check_budgets_cap_the_turn,
    check_clarification_moves_nothing,
    check_confirmation_is_answered_by_its_asker,
    check_late_words_land_nowhere,
    check_no_reply_left_owed,
    check_no_unexplained_mutation,
    check_offer_follows_board,
    check_owed_reply_settled_once,
    check_retry_never_acts_twice,
    check_stall_is_bounded,
    late_game_session,
    run_steps,
    walk,
)

CORPUS_SEEDS = int(os.environ.get("CHESSAPP_TRAJ_SEEDS", "48"))
CORPUS_LENGTH = 20


@pytest.mark.parametrize("seed", range(CORPUS_SEEDS))
def test_corpus_walk_keeps_every_invariant(seed: int, tmp_path: Path) -> None:
    walk(seed, CORPUS_LENGTH, tmp_path)


# --- named regressions -------------------------------------------------------------


def command(*rounds: tuple[tuple[str, dict], ...], **fields) -> Step:
    return Step(kind="command", text="do the thing", rounds=tuple(rounds), **fields)


D0, D1 = DELEGATES

REGRESSIONS: dict[str, tuple[str, tuple[Step, ...]]] = {
    # An ask and a move in one response: the move used to run under a
    # `clarify` handoff, choosing for the player (#314).
    "ask_then_move_in_one_batch": (
        "reproduces:#314",
        (
            command(
                (
                    ("ask_player", {"candidates": ["e4", "d4"]}),
                    ("make_move", {"move": "e4"}),
                )
            ),
        ),
    ),
    # An undo, then an ask about the restored board in the next round: the
    # offer used to stay on the pre-undo menu (#315).
    "undo_then_ask_next_round": (
        "reproduces:#315",
        (
            Step(kind="literal", text="d4"),
            command(
                (("undo", {"plies": 2}),),
                (("ask_player", {"candidates": ["e4", "d4"]}),),
            ),
        ),
    ),
    # The planner plays a move and the closer stalls: the turn used to wait on
    # the words with the reply ready, and then say them (#316).
    "move_then_stalled_closer": (
        "reproduces:#316",
        (command((("make_move", {"move": "e4"}),), stall=True),),
    ),
    # A reset armed by the panel's button, answered "yes" in a delegate thread
    # that was never asked: it used to run (#281).
    "a_thread_answers_the_panels_question": (
        "reproduces:#281",
        (
            Step(kind="literal", text="e4"),
            Step(kind="control", endpoint="/api/game/new", body={"color": "white"}),
            Step(kind="literal", text="yes", origin=D0),
        ),
    ),
    # A keyed delegate move, sent again: the retry must be the stored answer,
    # never a second move (#291).
    "a_keyed_move_retried": (
        "reproduces:#291",
        (
            Step(kind="literal", text="e4", origin=D1, key="k1"),
            Step(kind="retry", text="e4", origin=D1, key="k1"),
        ),
    ),
    # A takeback of one ply leaves the engine to move; the step must still end
    # with the player to move, and a restart on top of it must keep the game.
    "odd_undo_then_restart": (
        "prevents",
        (
            Step(kind="literal", text="e4"),
            Step(kind="literal", text="Nf3"),
            command((("undo", {"plies": 1}),)),
            Step(kind="restart"),
        ),
    ),
    # A destructive op armed by the planner, answered by a board endpoint and
    # then by a literal: neither may run it twice or run it for the wrong ask.
    "armed_reset_crossed_by_a_board_move": (
        "prevents",
        (
            Step(kind="literal", text="e4"),
            command((("new_game", {}),)),
            Step(kind="control", endpoint="/api/game/move", body={"move": "d2d4"}),
            Step(kind="literal", text="yes"),
        ),
    ),
    # Stockfish dies under the player's move (#284's own case), then comes
    # back: the move stands, and the next command collects the reply once.
    "engine_dies_on_the_reply_then_recovers": (
        "prevents",
        (
            Step(kind="literal", text="e4", engine_dies=True),
            command((("describe_position", {}),)),
        ),
    ),
    # llama-server dies after the planner's move landed: the move stands and
    # its reply is collected; the delegate says 502, a retry says 409.
    "provider_dies_after_a_move": (
        "prevents",
        (
            command(
                (("make_move", {"move": "e4"}),),
                (("describe_position", {}),),
                origin=D0,
                key="k1",
                provider_dies=1,
            ),
            Step(kind="retry", text="do the thing", origin=D0, key="k1"),
        ),
    ),
    # A batch past the per-turn cap on a real 84-ply game.
    "overlong_batch_on_the_late_game": (
        "prevents",
        (command(tuple([("evaluate_position", {})] * 5 + [("undo", {})] * 5)),),
    ),
}

# The #329 sites: Stockfish dying inside a tool the planner called. Each is a
# strict xfail, so the fix fails these until the marker comes off — and then
# `_ENGINE_DEATH_UNSAFE` in trajectory.py goes too.
OPEN_BUGS: dict[str, tuple[Step, ...]] = {
    "engine_dies_during_evaluate_position": (
        command(
            (("make_move", {"move": "e4"}), ("evaluate_position", {})),
            engine_dies=True,
        ),
    ),
    "engine_dies_during_get_best_moves": (
        command(
            (("make_move", {"move": "e4"}), ("get_best_moves", {"n": 2})),
            engine_dies=True,
        ),
    ),
    "engine_dies_while_an_undo_settles": (
        Step(kind="literal", text="e4"),
        command((("undo", {"plies": 1}),), engine_dies=True),
        Step(kind="literal", text="d4"),
    ),
}


@pytest.mark.parametrize("name", sorted(REGRESSIONS))
def test_named_regression(name: str, tmp_path: Path) -> None:
    _, steps = REGRESSIONS[name]
    start = late_game_session() if "late_game" in name else None
    run_steps(steps, tmp_path, start=start)


@pytest.mark.xfail(
    strict=True,
    raises=(chess.engine.EngineTerminatedError, InvariantBreach),
    reason="#329: engine death inside a planner tool",
)
@pytest.mark.parametrize("name", sorted(OPEN_BUGS))
def test_open_bug(name: str, tmp_path: Path) -> None:
    run_steps(OPEN_BUGS[name], tmp_path)


def test_every_regression_says_what_it_is() -> None:
    for tag, _ in REGRESSIONS.values():
        assert tag == "prevents" or tag.startswith("reproduces:#")


# --- the instrument's own checks ---------------------------------------------------


START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def _state(history: list[str], fen: str = START, **extra) -> dict:
    return {
        "version": extra.get("version", 0),
        "game_id": extra.get("game_id", "g"),
        "fen": fen,
        "history": history,
        "fens": [fen] * (len(history) + 1),
        "turn": extra.get("turn", "white"),
        "player_color": "white",
        "game_over": False,
        "legal_moves": extra.get("legal_moves", []),
    }


def _observed(step: Step, before=None, after=None, **fields) -> Observed:
    before = before or _state([])
    return Observed(step=step, before=before, after=after or before, **fields)


def test_the_clarification_check_fires_on_a_move_after_an_ask() -> None:
    observed = _observed(
        command(),
        status_code=200,
        results=[
            {"name": "ask_player", "result": {"ok": True}},
            {"name": "make_move", "result": {"legal": True}},
        ],
    )
    with pytest.raises(InvariantBreach):
        check_clarification_moves_nothing(observed)


def test_the_offer_check_fires_on_a_stale_enum() -> None:
    ask = {
        "function": {
            "name": "ask_player",
            "parameters": {
                "properties": {"candidates": {"items": {"enum": ["d4", "Nf3"]}}}
            },
        }
    }
    observed = _observed(
        command(),
        provider_calls=[
            {
                "tools": [ask],
                "messages": [
                    {
                        "role": "user",
                        "content": 'Board state:\n{"legal_moves": ["d4", "Nf3"]}\n\n'
                        "Command: x",
                    },
                    {
                        "role": "user",
                        "content": "Board state after those tool calls:\n"
                        '{"legal_moves": ["e4", "d4"]}',
                    },
                ],
            }
        ],
    )
    with pytest.raises(InvariantBreach, match="planner call 0"):
        check_offer_follows_board(observed)


def test_the_mutation_check_fires_on_a_silent_change() -> None:
    observed = _observed(
        command(),
        after=_state(["e4", "e5"]),
        status_code=200,
        results=[{"name": "describe_position", "result": {"ok": True}}],
    )
    with pytest.raises(InvariantBreach):
        check_no_unexplained_mutation(observed)


def test_the_owed_reply_check_fires_on_the_engine_to_move() -> None:
    observed = _observed(command(), after=_state(["e4"], turn="black"))
    with pytest.raises(InvariantBreach):
        check_no_reply_left_owed(observed)


def test_the_settle_check_fires_on_a_reply_played_twice() -> None:
    observed = _observed(
        Step(kind="literal", text="d4"),
        before=_state(["e4"], turn="black"),
        after=_state(["e4", "e5", "d4"], turn="black"),
        status_code=200,
    )
    with pytest.raises(InvariantBreach, match="exactly once"):
        check_owed_reply_settled_once(observed)


def test_the_confirmation_check_fires_on_a_yes_from_elsewhere() -> None:
    observed = _observed(
        Step(kind="literal", text="yes", origin=DELEGATES[0]),
        status_code=200,
        results=[{"name": "new_game", "result": {"ok": True}}],
        pending_before=Pending("new_game", PANEL, 0),
    )
    with pytest.raises(InvariantBreach):
        check_confirmation_is_answered_by_its_asker(observed)


def test_the_confirmation_check_fires_on_an_ignored_yes() -> None:
    observed = _observed(
        Step(kind="literal", text="yes"),
        status_code=200,
        results=[],
        pending_before=Pending("resign", PANEL, 0),
    )
    with pytest.raises(InvariantBreach):
        check_confirmation_is_answered_by_its_asker(observed)


def test_the_retry_check_fires_on_a_second_run() -> None:
    first = _observed(
        Step(kind="literal", text="e4", origin=DELEGATES[0], key="k"),
        status_code=200,
        response={"a": 1},
    )
    retry = _observed(
        Step(kind="retry", text="e4", origin=DELEGATES[0], key="k"),
        status_code=200,
        response={"a": 1},
        provider_calls=[{"tools": None, "messages": []}],
        original=first,
    )
    with pytest.raises(InvariantBreach, match="reached the model"):
        check_retry_never_acts_twice(retry)


def test_the_late_words_check_fires_on_remembered_words() -> None:
    observed = _observed(
        command(),
        provider_calls=[{"tools": [], "messages": [{"content": f"x {LATE_WORDS}"}]}],
    )
    with pytest.raises(InvariantBreach, match="remembers"):
        check_late_words_land_nowhere(observed)


def test_the_stall_check_fires_on_a_turn_that_waited() -> None:
    observed = _observed(command(stall=True), seconds=60.0)
    with pytest.raises(InvariantBreach):
        check_stall_is_bounded(observed)


def test_the_budget_check_fires_on_a_fourth_search() -> None:
    observed = _observed(
        command(),
        results=[{"name": "evaluate_position", "result": {"ok": True}}] * 4,
    )
    with pytest.raises(InvariantBreach):
        check_budgets_cap_the_turn(observed)


def test_every_invariant_is_run() -> None:
    assert len(INVARIANTS) == len({invariant.__name__ for invariant in INVARIANTS})
    assert {
        "check_clarification_moves_nothing",
        "check_offer_follows_board",
        "check_no_unexplained_mutation",
        "check_no_reply_left_owed",
        "check_owed_reply_settled_once",
        "check_restart_restores",
        "check_confirmation_is_answered_by_its_asker",
        "check_retry_never_acts_twice",
        "check_late_words_land_nowhere",
        "check_stall_is_bounded",
        "check_budgets_cap_the_turn",
    } <= {invariant.__name__ for invariant in INVARIANTS}


def test_a_walk_is_a_pure_function_of_its_seed(tmp_path: Path) -> None:
    def trace(directory: Path) -> list[tuple[str, int | None, tuple[str, ...]]]:
        return [
            (o.step.describe(), o.status_code, tuple(o.after["history"]))
            for o in walk(7, 12, directory)
        ]

    assert trace(tmp_path / "a") == trace(tmp_path / "b")


# --- replay ------------------------------------------------------------------------


@pytest.mark.skipif(
    "CHESSAPP_TRAJ_SEED" not in os.environ, reason="set CHESSAPP_TRAJ_SEED to replay"
)
def test_replay(tmp_path: Path) -> None:
    seed = int(os.environ["CHESSAPP_TRAJ_SEED"])
    length = int(os.environ.get("CHESSAPP_TRAJ_LENGTH", str(CORPUS_LENGTH)))
    for index, observed in enumerate(walk(seed, length, tmp_path)):
        print(f"{index:>2}. {observed.step.describe()} -> {observed.after['history']}")
