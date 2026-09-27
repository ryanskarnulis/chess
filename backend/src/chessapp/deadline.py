"""Waiting on optional words for a bounded time (#283, #316).

Shared by the pipeline, which bounds the observe beat and the other `narrate`
sites, and by the brain, which bounds its own closing narration
(`llama_brain._close`). Both wait by one
policy (#369): `NARRATION_BUDGET_S` when something computed or locked waits on
the words and the narrator is not thinking, `NARRATION_CEILING_S` otherwise. A
module of its own so the brain can use it without importing the app.
"""

import contextvars
import queue
import threading
from collections.abc import Callable

from chessapp.provider import ProviderError

# How long Glitch's words may take when something is waiting on them: a
# computed engine reply held behind the observe beat or the brain route's
# closer, or the mutation lock a confirmed-op or resign beat holds. Measured,
# not derived from the token cap (which bounds generation, not queueing or a
# dead server). Across 58 observe beats in the deployed trace (routes
# `fast_path` and `board`, one thinking-off call each) the reaction took
# 0.7–2.1 s, median ~1.5 s, with a single 7.5 s outlier; across the 31
# brain-route closers that spoke before a reply (2026-09-04 → 09-18) it took
# 0.8–2.0 s, median 1.3 s. Ten seconds clears every one of them with room for
# the shared GPU having a bad minute, so it fires on a model that is stuck,
# never on one that is merely talking, and it is still far below the point
# where a player decides the board is frozen. A cold llama-swap load (~100 s to
# first byte, first move after a reboot) is over it and loses that one
# narration to the app's own line; hanging up does not unload the upstream, so
# the next turn is warm (#283, #316).
NARRATION_BUDGET_S = 10.0

# The wait when nothing is held, or when the narrator thinks: a question, an
# analysis, a setting, a move played on the engine's advice. A stall backstop
# and not a budget, sized so it never cuts a thoughtful answer. After an
# analysis tool the narrator reasons before it speaks, and on the gate's
# move-plus-analysis scenarios (`move_and_judgment`, `best_move_then_play`,
# 2026-09-23) that took 6–10 s and more, so half the samples were cut at 10 s.
# The slowest such narration in the deployed trace took 15 s and thinking-on
# evals reach 30 s and more; 60 s is the planning phase's own wall clock
# (`llama_brain._DEFAULT_PLANNING_DEADLINE_S`), so a turn at worst waits as
# long for its words as it may spend deciding what to do (#316).
NARRATION_CEILING_S = 60.0


class LateReaction(ProviderError):
    """The app stopped waiting for a narration — the model may still be writing.

    A `ProviderError` on purpose, and the same design choice `TurnStateError`
    makes by subclassing `ValueError`: every branch that already treats the
    words as the one thing a failure may cost — the observe beat, the
    confirmed-op and resign narrations — handles lateness with no new failure
    shape to learn. The distinct type is what keeps the record honest about
    which of the two happened, since the player hears the same deterministic
    line either way.

    Deliberately *not* a `ProviderFailure`: that vocabulary answers "would
    asking again work", and nothing here says the provider failed. It answered
    too late for a turn that had already gone on without it, which is the app's
    judgment about this beat and not a fact about the server.
    """


def within_budget[T](work: Callable[[], T], budget: float) -> T:
    """Run `work` on its own thread and give up on it after `budget` seconds.

    Giving up is all this does: a model round trip cannot be cancelled, so the
    thread runs on and whatever it produces is dropped — the same shape
    `coordinator._PendingReply` uses for an engine computation the board moved
    out from under, and safe for the same reason. The work handed here touches
    no session: the narrator is given a board view snapshotted before the call
    and answers with words, so a late thread has nothing to land. The one piece
    of machine it can reach is the observation *phase* (a brain reports
    `narrating`, and assembly reads that report as the beat opening), and that
    is a mark on a turn the board is already waiting on: no mutation, and
    collecting the reply is legal from either phase by construction.

    The call's own exceptions cross back to the caller's thread unchanged; only
    the deadline is this function's own answer.

    It runs in a *copy* of the caller's context, which is what keeps the live
    progress stream working: which interaction an event belongs to is a
    `ContextVar` (`progress._CURRENT`), a bare thread starts from an empty
    context, and the beat's own "narrating" frame would simply stop being sent
    — the one thing the UI shows while Glitch is writing. A copy rather than
    the context itself because the caller goes on without this thread and must
    be free to close the interaction out from under it.
    """
    # Either the value in a 1-tuple or the exception that replaced it — never
    # bare, so a `T` that is itself an exception could not be mistaken for one.
    settled: queue.Queue[tuple[T] | BaseException] = queue.Queue(maxsize=1)
    context = contextvars.copy_context()

    def _run() -> None:
        try:
            settled.put((context.run(work),))
        except BaseException as exc:  # noqa: BLE001 — re-raised on the caller's thread
            settled.put(exc)

    threading.Thread(target=_run, name="bounded-words", daemon=True).start()
    try:
        outcome = settled.get(timeout=budget)
    except queue.Empty:
        raise LateReaction(f"no answer within {budget:.1f}s") from None
    if isinstance(outcome, BaseException):
        raise outcome
    return outcome[0]
