"""Composed interaction trajectories, in CI (#318).

`trajectory.py` is the machinery: a seeded walk through the shipped app with a
scripted planner and a deterministic engine, checking every invariant after
every step. This file is what runs:

- **The corpus.** A fixed range of seeds, each one walk. Bounded so the whole
  corpus costs CI about as much as one of the larger test files; widen it
  locally with `CHESSAPP_TRAJ_SEEDS=5000` for a soak. Measured when it was
  sized: with the #314 fix reverted 157 of 200 walks broke an invariant, with
  #315's reverted 121 of 200 — so a corpus of this size cannot miss either.
- **Named regressions.** The shortest step sequences that reproduced a real
  bug, each tagged with what it is: `reproduces` (went red on the build that
  had the bug, verified by reverting the fix) or `prevents` (a guard against a
  shape nothing has broken yet).
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

import pytest

from trajectory import (
    INVARIANTS,
    InvariantBreach,
    Observed,
    Step,
    check_clarification_moves_nothing,
    check_no_reply_left_owed,
    check_no_unexplained_mutation,
    check_offer_follows_board,
    run_steps,
    walk,
)

CORPUS_SEEDS = int(os.environ.get("CHESSAPP_TRAJ_SEEDS", "64"))
CORPUS_LENGTH = 20


@pytest.mark.parametrize("seed", range(CORPUS_SEEDS))
def test_corpus_walk_keeps_every_invariant(seed: int, tmp_path: Path) -> None:
    walk(seed, CORPUS_LENGTH, tmp_path)


# --- named regressions -------------------------------------------------------------


def command(*rounds: tuple[tuple[str, dict], ...]) -> Step:
    return Step(kind="command", text="do the thing", rounds=tuple(rounds))


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
}


@pytest.mark.parametrize("name", sorted(REGRESSIONS))
def test_named_regression(name: str, tmp_path: Path) -> None:
    _, steps = REGRESSIONS[name]
    run_steps(steps, tmp_path)


def test_every_regression_says_what_it_is() -> None:
    for tag, _ in REGRESSIONS.values():
        assert tag == "prevents" or tag.startswith("reproduces:#")


# --- the instrument's own checks ---------------------------------------------------


def _state(history: list[str], fen: str, **extra) -> dict:
    return {
        "game_id": "g",
        "fen": fen,
        "history": history,
        "fens": [fen] * (len(history) + 1),
        "turn": extra.get("turn", "white"),
        "player_color": "white",
        "game_over": False,
        "legal_moves": extra.get("legal_moves", []),
    }


START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def test_the_clarification_check_fires_on_a_move_after_an_ask() -> None:
    observed = Observed(
        step=command(),
        before=_state([], START),
        after=_state([], START),
        status_code=200,
        response={
            "tool_results": [
                {"name": "ask_player", "result": {"ok": True}},
                {"name": "make_move", "result": {"legal": True}},
            ]
        },
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
    observed = Observed(
        step=command(),
        before=_state([], START),
        after=_state([], START),
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
    observed = Observed(
        step=command(),
        before=_state([], START),
        after=_state(["e4"], START),
        status_code=200,
        response={
            "tool_results": [{"name": "describe_position", "result": {"ok": True}}]
        },
    )
    with pytest.raises(InvariantBreach):
        check_no_unexplained_mutation(observed)


def test_the_owed_reply_check_fires_on_the_engine_to_move() -> None:
    observed = Observed(
        step=command(),
        before=_state([], START),
        after=_state(["e4"], START, turn="black"),
    )
    with pytest.raises(InvariantBreach):
        check_no_reply_left_owed(observed)


def test_every_invariant_is_run() -> None:
    names = {invariant.__name__ for invariant in INVARIANTS}
    assert {
        "check_clarification_moves_nothing",
        "check_offer_follows_board",
        "check_no_unexplained_mutation",
        "check_no_reply_left_owed",
        "check_restart_restores",
    } <= names


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
