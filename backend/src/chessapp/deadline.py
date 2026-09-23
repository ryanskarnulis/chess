"""Waiting on optional words for a bounded time (#283, #316).

Shared by the pipeline, which bounds the observe beat and the other `narrate`
sites (`api._REACTION_BUDGET_S`), and by the brain, which bounds its own
closing narration (`llama_brain._close`). A module of its own so the brain can
use it without importing the app.
"""

import contextvars
import queue
import threading
from collections.abc import Callable

from chessapp.provider import ProviderError


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
