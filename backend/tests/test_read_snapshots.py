"""The read endpoints describe one real board, never a mid-mutation one (#230).

Every mutating surface runs under `ToolContext.mutation_lock`, and #200/#213
gave `/api/state` a coherent published document — but `/api/game/pgn`,
`/api/game/review` and `/api/game/hint` still serialized the *live* session with
no lock and no snapshot. All three are sync `def` routes, so they run in the
threadpool genuinely concurrent with a locked mutation, and a read that lands
between a mutation's two steps walks a half-applied board: a `to_dict()` inside
`undo`'s pop found an empty move stack and raised `IndexError` out of
`board.root()`. On the wire that is a 500, and the UI maps any non-OK to null —
so "Review unavailable", indistinguishable from having no engine at all.

Two properties hold this together, and they pull against each other:

- a read may not observe a half-applied board, so it joins the same mutation
  boundary the writers already take; but
- a turn holds that lock while the brain thinks — seconds — so nothing may hold
  it *across engine work*. The review is a multi-second Stockfish sweep and the
  hint is a search; blocking every concurrent drag behind one would be a worse
  bug than the tear it fixed.

So the boundary buys a **copy** (the version plus `to_dict()`, both cheap and
pure) and the slow work runs against that copy with the lock released. The copy
is also what makes the hint's #218 version binding exact rather than merely
fail-safe: the search runs on the very board the version names, so the number
and the analyzed position cannot come apart.

Engine-free by construction — the tearing regression needs no Stockfish, and
the two search endpoints are pinned through a paused engine double, so CI
(which has no binary) runs all of it.
"""

import io
import threading
import time
from typing import Any

import chess.pgn
import pytest
from fastapi.testclient import TestClient

from chessapp.api import _session_snapshot, create_app
from chessapp.engine import CandidateMove
from chessapp.game import GameSession
from chessapp.tools import ToolContext
from fakes import FakeEngine

# --- helpers ----------------------------------------------------------------

# One repeatable opening, played and taken back: at every instant the move
# history is exactly a prefix of this, which is what a coherent read must show.
STORM_MOVES = ("e4", "e5", "Nf3", "Nc6", "Bb5", "a6", "Ba4", "Nf6")

BEST = CandidateMove(uci="e2e4", san="e4", score_cp=30, mate_in=None)


@pytest.fixture
def ctx():
    return ToolContext(session=GameSession())


def endpoint_for(app: Any, path: str) -> Any:
    """The route function itself, so a test can call it from its own thread —
    the same boundary the threadpool calls it on. (`getattr`: the delegate
    router is mounted among the routes and carries no path of its own.)"""
    return next(
        route.endpoint for route in app.routes if getattr(route, "path", None) == path
    )


def developed(ctx: ToolContext) -> None:
    """A game with moves in it: the review refuses an empty one, and the hint
    wants a live position."""
    for san in ("e4", "e5", "Nf3", "Nc6"):
        ctx.session.submit_move(san)


class PausingEngine(FakeEngine):
    """An engine double whose search blocks until the test lets it finish.

    Stands in for the seconds a real search or review sweep takes. It records
    the position it was actually handed *after* the pause, which is the window
    that matters: on the live session a mutation landing mid-search moves the
    board out from under the analysis, so what the answer is about stops being
    what its `version` says it is about.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.searching = threading.Event()
        self.finish = threading.Event()
        self.analyzed_fens: list[str] = []

    def get_best_moves(self, session: GameSession, n: int = 3) -> list[CandidateMove]:
        self.searching.set()
        assert self.finish.wait(5), "the test never released the search"
        self.analyzed_fens.append(session.fen())
        return super().get_best_moves(session, n)


def read_while_a_mutation_holds_the_lock(
    ctx: ToolContext, endpoint: Any, move: str = "d4"
) -> tuple[bool, dict[str, Any]]:
    """Call `endpoint` from another thread while a mutation owns the lock.

    Returns whether the read answered *during* the mutation, plus its result. A
    read that shares the mutation boundary cannot: it waits, and then describes
    the finished board. An unguarded one answers straight out of the session
    that is being written.
    """
    result: dict[str, Any] = {}
    failure: list[BaseException] = []
    answered = threading.Event()

    def read() -> None:
        try:
            result.update(endpoint())
        except BaseException as exc:  # re-raised in the test thread below
            failure.append(exc)
        finally:
            answered.set()

    reader = threading.Thread(target=read)
    with ctx.mutation_lock:
        ctx.session.submit_move(move)
        reader.start()
        answered_during = answered.wait(0.25)
    reader.join(5)
    assert not reader.is_alive(), "the read deadlocked"
    if failure:
        raise failure[0]
    return answered_during, result


def search_in_flight(ctx: ToolContext, endpoint: Any) -> tuple[PausingEngine, Any]:
    """Start `endpoint` on a paused engine and return once the search is running
    — the window in which the lock must be free."""
    engine = PausingEngine(best_moves=(BEST,))
    ctx.engine = engine
    result: dict[str, Any] = {}
    failure: list[BaseException] = []

    def call() -> None:
        try:
            result.update(endpoint())
        except BaseException as exc:
            failure.append(exc)

    caller = threading.Thread(target=call)
    caller.start()
    if not engine.searching.wait(5):
        engine.finish.set()
        caller.join(5)
        if failure:
            raise failure[0]
        raise AssertionError("the endpoint never reached the engine")
    return engine, (caller, result, failure)


def finish_search(engine: PausingEngine, pending: Any) -> dict[str, Any]:
    caller, result, failure = pending
    engine.finish.set()
    caller.join(5)
    assert not caller.is_alive(), "the endpoint deadlocked"
    if failure:
        raise failure[0]
    return result


# --- the boundary: a read never straddles a mutation ------------------------


def test_pgn_shares_the_mutation_boundary(ctx):
    """`export_pgn` walks the move stack; walking one another thread is popping
    is what made "Copy PGN" a 500."""
    developed(ctx)
    pgn = endpoint_for(create_app(ctx), "/api/game/pgn")

    answered_during, result = read_while_a_mutation_holds_the_lock(ctx, pgn)

    assert not answered_during, "the PGN was serialized off a mid-mutation board"
    assert "d4" in result["pgn"]


def test_review_shares_the_mutation_boundary(ctx):
    ctx.engine = FakeEngine(best_moves=(BEST,))
    developed(ctx)
    review = endpoint_for(create_app(ctx), "/api/game/review")

    answered_during, result = read_while_a_mutation_holds_the_lock(ctx, review)

    assert not answered_during, "the review read a mid-mutation board"
    assert [m["san"] for m in result["moves"]][:2] == ["e4", "e5"]


def test_hint_shares_the_mutation_boundary(ctx):
    ctx.engine = FakeEngine(best_moves=(BEST,))
    developed(ctx)
    hint = endpoint_for(create_app(ctx), "/api/game/hint")

    answered_during, result = read_while_a_mutation_holds_the_lock(ctx, hint)

    assert not answered_during, "the hint read a mid-mutation board"
    assert result["version"] == ctx.board_version


# --- but the boundary is never held across engine work ----------------------


def test_the_hint_search_does_not_hold_the_mutation_lock(ctx):
    """The liveness half, and the reason the fix is a copy rather than a longer
    lock: a turn already holds this lock while the brain thinks, so a search
    holding it too would stall every concurrent drag."""
    developed(ctx)
    hint = endpoint_for(create_app(ctx), "/api/game/hint")
    engine, pending = search_in_flight(ctx, hint)

    free = ctx.mutation_lock.acquire(timeout=1)
    if free:
        ctx.mutation_lock.release()
    finish_search(engine, pending)

    assert free, "the hint held the mutation lock across the engine search"


def test_the_review_sweep_does_not_hold_the_mutation_lock(ctx):
    developed(ctx)
    review = endpoint_for(create_app(ctx), "/api/game/review")
    engine, pending = search_in_flight(ctx, review)

    free = ctx.mutation_lock.acquire(timeout=1)
    if free:
        ctx.mutation_lock.release()
    finish_search(engine, pending)

    assert free, "the review held the mutation lock across the Stockfish sweep"


def test_a_move_lands_while_the_review_sweep_runs(ctx):
    """The same property from the player's side: a drag during a review is not
    made to wait for it, and the review still answers about the game it took."""
    developed(ctx)
    review = endpoint_for(create_app(ctx), "/api/game/review")
    engine, pending = search_in_flight(ctx, review)

    with ctx.mutation_lock:
        assert ctx.session.submit_move("d4").legal
    result = finish_search(engine, pending)

    assert [m["san"] for m in result["moves"]] == ["e4", "e5", "Nf3", "Nc6"]


# --- the hint's version names the board the engine actually saw (#218) -------


def test_a_hint_analyzes_the_board_its_version_names(ctx):
    """#218 put the analyzed board's version in the payload so a late answer
    cannot paint its arrow onto a newer position. Reading the version off the
    live session merely failed *safe* (an old number labelling a newer board,
    which the client discards); searching the snapshot the version was taken
    with makes the two the same board by construction.
    """
    developed(ctx)
    hint = endpoint_for(create_app(ctx), "/api/game/hint")
    engine, pending = search_in_flight(ctx, hint)
    analyzed = ctx.session.fen()
    version = ctx.board_version

    # A drag lands while the search is in flight — the race #218 is about.
    with ctx.mutation_lock:
        assert ctx.session.submit_move("d4").legal
    result = finish_search(engine, pending)

    assert ctx.board_version != version, "the board never moved; nothing was raced"
    assert engine.analyzed_fens == [analyzed], "the search followed the live board"
    assert result["version"] == version


# --- the issue's repro, bounded ---------------------------------------------


def test_reads_survive_a_storm_of_locked_mutations(ctx):
    """A writer doing what a locked turn does, a reader doing what the read
    endpoints do — the issue's reproduction with a deadline on it.

    Probabilistic, so the deterministic tests above are the gate; this is the
    shape the failure was actually found in (~1 tear per 20k reads), and it
    pins the property the others assert about one interleaving over a few
    thousand: no exception, and every answer is a real position. The writer
    takes the lock per mutation because that is what the app does — the fix is
    the reader joining a boundary the writers were already keeping.
    """
    pgn = endpoint_for(create_app(ctx), "/api/game/pgn")
    stop = threading.Event()
    failures: list[BaseException] = []
    reads = 0
    deadline = time.monotonic() + 1.0

    def writer() -> None:
        try:
            while not stop.is_set() and time.monotonic() < deadline:
                for san in STORM_MOVES:
                    with ctx.mutation_lock:
                        assert ctx.session.submit_move(san).legal
                with ctx.mutation_lock:
                    assert ctx.session.undo(len(STORM_MOVES)).ok
        except BaseException as exc:
            failures.append(exc)
        finally:
            stop.set()

    def reader() -> None:
        nonlocal reads
        try:
            while not stop.is_set():
                game = chess.pgn.read_game(io.StringIO(pgn()["pgn"]))
                assert game is not None and not game.errors
                played = tuple(
                    node.san() for node in game.mainline() if node.move is not None
                )
                assert played == STORM_MOVES[: len(played)], played
                _, snapshot = _session_snapshot(ctx)
                history = tuple(snapshot.move_history())
                assert history == STORM_MOVES[: len(history)], history
                reads += 1
        except BaseException as exc:
            failures.append(exc)
        finally:
            stop.set()

    threads = [threading.Thread(target=writer), threading.Thread(target=reader)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(30)
        assert not thread.is_alive(), "the storm never finished"

    if failures:
        raise failures[0]
    assert reads > 100, f"the storm barely ran ({reads} reads)"


def test_a_snapshot_is_detached_from_the_live_session(ctx):
    """The copy is a copy: the slow work runs on it for seconds while the game
    goes on, so a later mutation must not reach back into it."""
    developed(ctx)
    version, snapshot = _session_snapshot(ctx)

    ctx.session.submit_move("d4")

    assert snapshot is not ctx.session
    assert snapshot.move_history() == ["e4", "e5", "Nf3", "Nc6"]
    assert version == ctx.board_version - 1


def test_a_snapshot_carries_the_endings_a_board_cannot_hold(ctx):
    """Resignation and a claimed draw are session state, not board state, so a
    snapshot that dropped them would report a finished game as playable — the
    hint would answer one, and the PGN would lose its result."""
    developed(ctx)
    ctx.session.resign("black")

    _, snapshot = _session_snapshot(ctx)

    assert snapshot.is_game_over()
    assert snapshot.outcome() == ctx.session.outcome()
    assert snapshot.export_pgn() == ctx.session.export_pgn()


def test_the_read_endpoints_keep_their_failure_shapes(ctx):
    """503 without an engine, 409 for a domain refusal: taking the reads off a
    snapshot must not move where those answers come from."""
    client = TestClient(create_app(ctx))
    assert client.get("/api/game/review").status_code == 503
    assert client.get("/api/game/hint").status_code == 503

    ctx.engine = FakeEngine(best_moves=(BEST,))
    client = TestClient(create_app(ctx))
    assert client.get("/api/game/review").status_code == 409  # no moves yet

    ctx.session.resign()
    assert client.get("/api/game/hint").status_code == 409  # game is over
    assert client.get("/api/game/pgn").status_code == 200
