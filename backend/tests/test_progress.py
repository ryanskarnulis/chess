"""The live-progress reporter: what a turn says about itself while it runs.

Unit level — no app, no websocket. What is pinned here is the reporter's own
contract: an event belongs to an interaction or it does not exist, the
interaction is bracketed whatever happens inside it, and reporting is never
allowed to cost the turn it is reporting on.
"""

import contextvars
import threading

import pytest

from chessapp.progress import (
    BRAIN_NARRATING,
    BRAIN_PLANNING,
    KIND_BEGIN,
    KIND_BRAIN,
    KIND_END,
    KIND_PHASE,
    KIND_TOOL,
    ProgressEvent,
    ProgressReporter,
)


def collecting() -> tuple[ProgressReporter, list[ProgressEvent]]:
    events: list[ProgressEvent] = []
    reporter = ProgressReporter()
    reporter.bind(events.append)
    return reporter, events


def kinds(events: list[ProgressEvent]) -> list[tuple[str, str]]:
    return [(event.kind, event.name) for event in events]


# --- what an event belongs to ------------------------------------------------


def test_events_outside_an_interaction_are_dropped():
    """Progress is *about* an interaction. Nothing outside one has a turn to
    label, so a tool dispatched by MCP or a phase moved by a button reports
    nothing rather than an event nobody can place."""
    reporter, events = collecting()
    reporter.tool("make_move")
    reporter.phase("engine_calculating")
    reporter.brain(BRAIN_PLANNING)
    assert events == []


def test_an_interaction_brackets_its_events():
    reporter, events = collecting()
    with reporter.interaction("abc123", 7):
        reporter.tool("make_move")
    assert kinds(events) == [(KIND_BEGIN, ""), (KIND_TOOL, "make_move"), (KIND_END, "")]


def test_every_event_carries_the_interaction_it_belongs_to():
    """`correlation_id` is the key the trace record already uses, so a progress
    line and the record of the turn that produced it are one search apart."""
    reporter, events = collecting()
    with reporter.interaction("abc123", 7):
        reporter.phase("engine_calculating")
    assert all(event.correlation_id == "abc123" for event in events)
    assert all(event.turn_id == 7 for event in events)


def test_each_source_names_what_it_did():
    reporter, events = collecting()
    with reporter.interaction("abc123", 1):
        reporter.tool("get_best_moves")
        reporter.phase("engine_calculating")
        reporter.brain(BRAIN_NARRATING)
    assert kinds(events)[1:-1] == [
        (KIND_TOOL, "get_best_moves"),
        (KIND_PHASE, "engine_calculating"),
        (KIND_BRAIN, BRAIN_NARRATING),
    ]


def test_the_end_event_fires_even_when_the_turn_raises():
    """A turn that blew up is exactly the one whose progress line must not be
    left spinning on the player's screen."""
    reporter, events = collecting()
    with pytest.raises(ValueError), reporter.interaction("abc123", 1):
        raise ValueError("boom")
    assert kinds(events) == [(KIND_BEGIN, ""), (KIND_END, "")]


def test_an_interaction_leaves_nothing_behind_it():
    reporter, events = collecting()
    with reporter.interaction("abc123", 1):
        pass
    reporter.tool("make_move")
    assert kinds(events) == [(KIND_BEGIN, ""), (KIND_END, "")]


# --- reporting never costs the turn ------------------------------------------


def test_an_unbound_reporter_reports_nothing():
    """No sink is the ordinary state of a reporter that was built but never
    wired (a unit test, an MCP process) — it is not a failure."""
    reporter = ProgressReporter()
    with reporter.interaction("abc123", 1):
        reporter.tool("make_move")  # must not raise


def test_a_failing_sink_never_costs_the_turn():
    """Same rule as the tracer: a diagnostic that fails is a lost diagnostic,
    never a lost turn."""

    def explode(_event: ProgressEvent) -> None:
        raise RuntimeError("socket went away")

    reporter = ProgressReporter()
    reporter.bind(explode)
    with reporter.interaction("abc123", 1):
        reporter.tool("make_move")


# --- the observation beat ----------------------------------------------------


def test_narrating_opens_the_observation_beat():
    """The narrator turn *is* the observe beat, and this report is the only
    signal the app gets that it started — so it is where the beat opens."""
    opened: list[bool] = []
    reporter = ProgressReporter(on_narrating=lambda: opened.append(True))
    with reporter.interaction("abc123", 1):
        reporter.brain(BRAIN_NARRATING)
    assert opened == [True]


def test_planning_does_not_open_the_observation_beat():
    opened: list[bool] = []
    reporter = ProgressReporter(on_narrating=lambda: opened.append(True))
    with reporter.interaction("abc123", 1):
        reporter.brain(BRAIN_PLANNING)
    assert opened == []


def test_the_beat_opens_even_with_no_sink_bound():
    """Marking the phase is game-machine business; reporting is decoration.
    An unwired reporter must not silently stop the machine advancing."""
    opened: list[bool] = []
    reporter = ProgressReporter(on_narrating=lambda: opened.append(True))
    with reporter.interaction("abc123", 1):
        reporter.brain(BRAIN_NARRATING)
    assert opened == [True]


def test_a_failing_beat_hook_never_costs_the_turn():
    def explode() -> None:
        raise RuntimeError("wrong phase")

    reporter = ProgressReporter(on_narrating=explode)
    reporter.bind(lambda _event: None)
    with reporter.interaction("abc123", 1):
        reporter.brain(BRAIN_NARRATING)


# --- across threads ----------------------------------------------------------


def test_a_worker_thread_inherits_the_interaction_it_was_started_in():
    """The pipeline's blocking steps run off the event loop (the loop has to
    stay free or nothing arrives *live*), so the interaction has to travel with
    the copied context rather than living on the reporter."""
    reporter, events = collecting()
    with reporter.interaction("abc123", 4):
        context = contextvars.copy_context()
        worker = threading.Thread(
            target=lambda: context.run(reporter.tool, "make_move")
        )
        worker.start()
        worker.join()
    assert (KIND_TOOL, "make_move") in kinds(events)
    assert events[1].correlation_id == "abc123"


# --- the wire shape ----------------------------------------------------------


def test_as_dict_is_the_wire_shape():
    event = ProgressEvent(
        correlation_id="abc123", turn_id=4, kind=KIND_PHASE, name="engine_calculating"
    )
    assert event.as_dict() == {
        "correlation_id": "abc123",
        "turn_id": 4,
        "kind": "phase",
        "name": "engine_calculating",
    }
