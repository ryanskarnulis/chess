"""A slow narrator may not hold the engine's reply, or the lock (#283).

The observe beat is optional by construction — Stockfish starts thinking the
moment the player's move lands, and collecting the reply is legal with or
without a reaction — but "optional" used to mean only that the beat could be
*skipped*. A narrator that was merely slow still held an answer that was already
computed and sitting in memory, kept the turn in `agent_observing`, and, because
a command runs under the app's mutation lock, parked every other road onto the
board behind it: a drag, an undo, a delegate's move. The only bounds were the
token cap (which bounds generation, not queueing or a stalled server) and the
provider's 300 s read timeout.

What this file pins: gameplay proceeds inside the budget on every route that
narrates, the late words land nowhere — not on the board, not in the player's
commentary, not in Glitch's memory — the lock is free the moment the turn
closes, and a reaction that arrives *inside* the budget is spoken exactly as it
always was, because react-before-reply is the beat's acceptance criterion and
this fix is not allowed to buy latency with it.

The narrator here is a double that blocks on an event, so nothing sleeps for
real and nothing depends on the wall clock being kind: the apps are built with a
budget of a few hundredths of a second, and every assertion about lateness is
made while that double is provably still writing.
"""

import json
import threading

import pytest
from fastapi.testclient import TestClient

from chessapp import api
from chessapp.api import (
    _REACTION_BUDGET_S,
    PROVIDER_LOST_TURN_STANDS,
    STUCK_REPLY,
    _late_close_words,
    create_app,
)
from chessapp.brain import AgentResponse, Narration
from chessapp.coordinator import TurnCoordinator, TurnPhase
from chessapp.game import GameSession, MoveResult
from chessapp.llama_brain import _NARRATE_TIMEOUT, LlamaBrain
from chessapp.tools import ToolContext, brain_tool_definitions, build_registry
from chessapp.trace import JsonlTracer
from fakes import (
    BlockingNarratorProvider,
    FakeEngine,
    ScriptedBrain,
    ScriptedProvider,
    text_turn,
    tool_calls_turn,
)

# Small enough that the suite never waits on it, and that the double never
# finishes inside it by accident.
BUDGET = 0.05
# How long a test is willing to wait on the double's own events before calling
# the fix broken. Generous: it only ever elapses on a failure.
PATIENCE = 5.0
# What the late narrator eventually writes. It must appear nowhere.
LATE_WORDS = "Late words nobody should ever hear."


class SlowNarrator(ScriptedBrain):
    """A brain whose reaction is still being written when the turn moves on.

    `entered` says the beat was reached, `release` is the test letting it
    finish, and `finished` says it did — so "the turn came back while this was
    still writing" is a fact a test can assert rather than a race it hopes for.
    """

    def __init__(self, *responses, **kwargs):
        super().__init__(*responses, **kwargs)
        self.entered = threading.Event()
        self.release = threading.Event()
        self.finished = threading.Event()

    def narrate(self, board_state, changes, transcript=()):
        self.narrate_calls.append((board_state, changes))
        self.entered.set()
        self.release.wait(timeout=PATIENCE)
        self.finished.set()
        return Narration(text=LATE_WORDS)


@pytest.fixture
def slow():
    """Makes blocked narrators, and releases every one of them at teardown so
    no test leaves a thread parked behind it."""
    made: list[SlowNarrator] = []

    def make(*responses: AgentResponse) -> SlowNarrator:
        brain = SlowNarrator(*responses)
        made.append(brain)
        return brain

    yield make
    for brain in made:
        brain.release.set()
        if brain.entered.is_set():
            brain.finished.wait(timeout=PATIENCE)


def build(brain, *, budget=BUDGET, tracer=None, verbosity="normal"):
    """An agent-mode app on a real registry and coordinator, with a budget a
    test can outlive. The coordinator is handed in rather than built inside
    (which is what `scripted_app` does) so a test can read the phase the turn
    was left in."""
    ctx = ToolContext(session=GameSession(), engine=FakeEngine())
    ctx.settings.verbosity = verbosity
    coordinator = TurnCoordinator(ctx)
    registry = build_registry(ctx, coordinator, atomic_exchange=False)
    brain.dispatcher = registry
    app = create_app(
        ctx,
        brain=brain,
        registry=registry,
        coordinator=coordinator,
        tracer=tracer,
        reaction_budget=budget,
    )
    return TestClient(app), ctx, coordinator


def still_writing(brain: SlowNarrator) -> bool:
    """The double reached the beat and has not come back — so whatever the turn
    just did, it did without waiting for these words."""
    return brain.entered.wait(timeout=PATIENCE) and not brain.finished.is_set()


# --- the observe beat: the reply is ready, so it is played ------------------


def test_the_fast_path_plays_the_ready_reply_without_waiting_for_the_words(slow):
    brain = slow()
    client, ctx, coordinator = build(brain)

    body = client.post("/api/command", json={"text": "e4"}).json()

    assert still_writing(brain), "the turn must not have waited for the reaction"
    # The reply Stockfish computed during the beat is on the board, and the
    # turn closed on it rather than parking in `agent_observing`.
    assert ctx.session.move_history() == ["e4", "e5"]
    assert body["state"]["history"] == ["e4", "e5"]
    assert coordinator.phase is TurnPhase.AWAITING_PLAYER
    # With no reaction to show, the app says its own deterministic line — the
    # same one verbosity=low and a dead provider already get.
    assert body["commentary"] == "e4. e5."


def test_the_lock_is_free_while_the_narrator_is_still_writing(slow):
    """The audit's probe hung here: a concurrent mutation parked on the
    mutation lock for as long as the narration took, because the lock is held
    across the whole command and the command was waiting on words."""
    brain = slow()
    client, ctx, _ = build(brain)
    client.post("/api/command", json={"text": "e4"})
    assert still_writing(brain)

    undone = client.post("/api/game/undo", json={})

    assert undone.status_code == 200
    assert ctx.session.move_history() == []


def test_the_late_words_land_nowhere(slow):
    """A late result carries no authority: not on the board, not in the
    player's commentary, and not in what Glitch is told he said."""
    brain = slow()
    client, ctx, _ = build(brain)
    body = client.post("/api/command", json={"text": "e4"}).json()
    assert still_writing(brain)
    version, history = ctx.board_version, ctx.session.move_history()

    brain.release.set()
    assert brain.finished.wait(timeout=PATIENCE), "the double never finished"

    assert ctx.board_version == version, "a late narration moved nothing"
    assert ctx.session.move_history() == history
    assert LATE_WORDS not in body["commentary"]
    remembered = " ".join(message["content"] for message in ctx.transcript.memory())
    assert LATE_WORDS not in remembered
    assert LATE_WORDS not in json.dumps(client.get("/api/state").json())


def test_a_dragged_move_is_bounded_the_same_way(slow):
    """Agent mode runs the same beats for a drag as for a spoken move
    (`api._play_move`), so the budget reaches both or neither."""
    brain = slow()
    client, ctx, coordinator = build(brain)

    body = client.post("/api/game/move", json={"move": "e2e4"}).json()

    assert still_writing(brain)
    assert body["legal"] is True
    assert body["engine_move"]["san"] == "e5"
    assert ctx.session.move_history() == ["e4", "e5"]
    assert coordinator.phase is TurnPhase.AWAITING_PLAYER
    assert body["commentary"] == "e4. e5."


def test_a_confirmed_destructive_op_falls_back_to_the_apps_own_line(slow):
    """The close beat narrates too, and holds the lock while it does. No
    computed reply is waiting here — the op has already run — so what the budget
    buys is the lock, and the turn coming back at all."""
    brain = slow()
    client, ctx, _ = build(brain)
    # A game worth throwing away, so the gate has something to ask about, armed
    # from the button and answered in words (the surfaces are independent).
    ctx.session.submit_move("e4")
    ctx.session.submit_move("e5")
    client.post("/api/game/new", json={"color": "white"})
    assert ctx.pending is not None, "the gate armed the op and asked"

    body = client.post("/api/command", json={"text": "yes"}).json()

    assert still_writing(brain)
    assert body["commentary"] == "New game."
    assert ctx.session.move_history() == []


# --- a reaction inside the budget is untouched ------------------------------


def test_a_reaction_inside_the_budget_is_spoken_exactly_as_ever():
    """React-before-reply is the beat's acceptance criterion: a narrator that
    answers in time still gets its words in front of the app's announcement,
    and the move still costs the one model call it always cost."""
    brain = ScriptedBrain(narrations=("Classic opener.",))
    client, ctx, _ = build(brain, budget=5.0)

    body = client.post("/api/command", json={"text": "e4"}).json()

    assert body["commentary"] == "Classic opener.\n\ne5."
    assert ctx.session.move_history() == ["e4", "e5"]
    assert len(brain.narrate_calls) == 1


def test_the_budget_clears_every_reaction_the_deployed_app_has_measured():
    """The number is measured, not guessed (`api._REACTION_BUDGET_S`): across 58
    observe beats in the deployed trace the reaction took 0.7–2.1 s, median
    ~1.5 s, with one 7.5 s outlier. The floor this may never sink under is that
    worst observed beat — cutting a healthy reaction is the failure this fix is
    not allowed to trade for."""
    assert _REACTION_BUDGET_S >= 7.5


def test_the_narrators_own_read_ceiling_clears_the_pipelines_budget():
    """Two layers, and they may not race for which gives up first. The pipeline
    stops waiting and plays the reply; the brain's socket ceiling is the
    backstop underneath it that ends the abandoned round trip, so a
    llama-server slot is not held for words nobody will hear."""
    assert _NARRATE_TIMEOUT > _REACTION_BUDGET_S


# --- the record -------------------------------------------------------------


def test_the_trace_says_the_reaction_was_late(tmp_path, slow):
    """A cut reaction is invisible in the commentary — it reads exactly like
    verbosity=low, like a dead provider, like a beat that never opened — so the
    turn record is the only place the budget can be tuned from."""
    path = tmp_path / "turns.jsonl"
    brain = slow()
    client, _, _ = build(brain, tracer=JsonlTracer(path))

    client.post("/api/command", json={"text": "e4"})

    assert still_writing(brain)
    record = json.loads(path.read_text().splitlines()[0])
    assert record["reaction_late"] is True
    assert record["route"] == "fast_path"
    # Not a failure: nothing died, the turn completed, and the board moved twice.
    assert record["stop_reason"] == "completed"
    assert record["engine_failure"] == ""
    assert record["provider_failure"] == ""
    assert record["mutations"] == 2


def test_a_dragged_move_records_its_lateness_too(tmp_path, slow):
    path = tmp_path / "turns.jsonl"
    brain = slow()
    client, _, _ = build(brain, tracer=JsonlTracer(path))

    client.post("/api/game/move", json={"move": "e2e4"})

    assert still_writing(brain)
    record = json.loads(path.read_text().splitlines()[0])
    assert (record["route"], record["reaction_late"]) == ("board", True)


def test_a_late_reaction_is_still_a_call_on_the_turn(tmp_path, slow):
    """The turn waited out the budget on a round trip it made (#290): one call,
    its latency the wait, its tokens unknown — never a turn that cost nothing."""
    path = tmp_path / "turns.jsonl"
    brain = slow()
    client, _, _ = build(brain, tracer=JsonlTracer(path))

    client.post("/api/command", json={"text": "e4"})

    assert still_writing(brain)
    record = json.loads(path.read_text().splitlines()[0])
    assert record["model_calls"] == 1
    assert record["unmetered_calls"] == 1
    (waited,) = record["model_latencies_ms"]
    assert waited >= BUDGET * 1000 * 0.9
    # #317: the call says which beat it was, that it was cut rather than
    # failed, and what its censored reading was censored at.
    (call,) = record["calls"]
    assert call["phase"] == "reaction"
    assert call["status"] == "late"
    assert call["budget_ms"] == round(BUDGET * 1000)
    assert call["failure"] == ""


def test_a_turn_that_spoke_in_time_records_no_lateness(tmp_path):
    path = tmp_path / "turns.jsonl"
    client, _, _ = build(
        ScriptedBrain(narrations=("Classic opener.",)),
        budget=5.0,
        tracer=JsonlTracer(path),
    )

    client.post("/api/command", json={"text": "e4"})

    record = json.loads(path.read_text().splitlines()[0])
    assert record["reaction_late"] is False
    assert record["commentary"].startswith("Classic opener.")


# --- a beat that never ran is not a late one --------------------------------


def test_a_silent_turn_is_not_recorded_as_late(tmp_path):
    """verbosity=low never opens the beat at all. Nothing was cut, so nothing
    is reported cut: the flag names one thing, and not "there were no words"."""
    path = tmp_path / "turns.jsonl"
    client, _, _ = build(ScriptedBrain(), verbosity="low", tracer=JsonlTracer(path))

    body = client.post("/api/command", json={"text": "e4"}).json()

    assert body["commentary"] == "e4. e5."
    record = json.loads(path.read_text().splitlines()[0])
    assert record["reaction_late"] is False


def test_a_turn_with_no_move_in_it_never_reaches_the_budget(slow):
    """The brain route's commentary is its loop's, not `narrate`'s: a question
    that moves nothing opens no observe beat, so there is nothing here for a
    budget to cut."""
    brain = slow(AgentResponse(text="You're fine, bro."))
    client, _, _ = build(brain)

    body = client.post("/api/command", json={"text": "how am I doing?"}).json()

    assert body["commentary"] == "You're fine, bro."
    assert not brain.entered.is_set(), (
        "no observation beat on a turn that moved nothing"
    )


# --- the brain route's own closer (#316) ------------------------------------
#
# Everything above goes through `api._narrate`. The brain route does not: its
# words come from the loop's closing narrator, *inside* `get_agent_response`,
# and the planner has already played the player's move by then — so the same
# ready reply and the same lock waited on a phase no budget reached. These run
# the real `LlamaBrain` over a scripted provider whose narrator call blocks, and
# bound it inside the brain: the plan's record comes back on time, the words
# are dropped, and nothing that can act outlives the turn.


def plays_e4(**kwargs) -> BlockingNarratorProvider:
    """The issue's repro script: the planner plays e4, then hands off; the
    narrator call blocks until released."""
    return BlockingNarratorProvider(
        tool_calls_turn(("make_move", {"move": "e4", "source": "said_the_move"})),
        text_turn("played e4"),
        words=LATE_WORDS,
        patience=PATIENCE,
        **kwargs,
    )


@pytest.fixture
def blocked():
    """Makes blocking providers and releases each at teardown, so no test
    leaves a narrator thread parked behind it."""
    made: list[BlockingNarratorProvider] = []

    def make() -> BlockingNarratorProvider:
        provider = plays_e4()
        made.append(provider)
        return provider

    yield make
    for provider in made:
        provider.release.set()
        if provider.entered.is_set():
            provider.finished.wait(timeout=PATIENCE)


def build_brain_route(provider, *, closing=BUDGET, ceiling=BUDGET, tracer=None):
    """The full pipeline over a real `LlamaBrain`, wired as app assembly wires
    it (split registry, shared coordinator, the narrator-facts seam that says
    whether the reply is owed), with closer budgets a test can outlive."""
    ctx = ToolContext(session=GameSession(), engine=FakeEngine())
    coordinator = TurnCoordinator(ctx)
    registry = build_registry(ctx, coordinator, atomic_exchange=False)
    brain = LlamaBrain(
        provider=provider,
        dispatcher=registry,
        tool_definitions=lambda: brain_tool_definitions(registry, ctx),
        system_prompt="You are Glitch.",
        planner_prompt="Route the ask to tools.",
        narrator_facts=lambda: api.narrator_facts(ctx, coordinator),
        closing_budget_s=closing,
        closing_ceiling_s=ceiling,
    )
    app = create_app(
        ctx, brain=brain, registry=registry, coordinator=coordinator, tracer=tracer
    )
    return TestClient(app), ctx, coordinator


def closer_still_writing(provider: BlockingNarratorProvider) -> bool:
    return provider.entered.wait(timeout=PATIENCE) and not provider.finished.is_set()


def test_a_stalled_closer_no_longer_holds_the_ready_reply(blocked):
    """The issue's reproduction, inverted: the request finishes, the reply is
    on the board and the turn is closed while the narrator is still writing."""
    provider = blocked()
    client, ctx, coordinator = build_brain_route(provider)

    body = client.post("/api/command", json={"text": "push the king pawn"}).json()

    assert closer_still_writing(provider), "the turn must not wait for the words"
    assert ctx.session.move_history() == ["e4", "e5"]
    assert body["state"]["history"] == ["e4", "e5"]
    assert coordinator.phase is TurnPhase.AWAITING_PLAYER
    # The fast path's late line, not "say it again" over a move that landed.
    assert body["commentary"] == "e4. e5."


def test_a_mutation_queued_behind_the_turn_runs_while_the_words_are_late(blocked):
    """The lock half: an undo sent while the closer is stalled parks on the
    mutation lock, and gets it when the brain stops waiting — not when the
    model finally answers."""
    provider = blocked()
    client, ctx, _ = build_brain_route(provider)
    done = threading.Event()

    def command() -> None:
        client.post("/api/command", json={"text": "push the king pawn"})
        done.set()

    threading.Thread(target=command, daemon=True).start()
    assert provider.entered.wait(timeout=PATIENCE)

    undone = client.post("/api/game/undo", json={})

    assert closer_still_writing(provider)
    assert done.wait(timeout=PATIENCE)
    assert undone.status_code == 200
    assert ctx.session.move_history() == []


def test_the_late_closers_words_land_nowhere(blocked):
    provider = blocked()
    client, ctx, _ = build_brain_route(provider)
    body = client.post("/api/command", json={"text": "push the king pawn"}).json()
    assert closer_still_writing(provider)
    version, history = ctx.board_version, ctx.session.move_history()

    provider.release.set()
    assert provider.finished.wait(timeout=PATIENCE)

    assert ctx.board_version == version, "a late closer moved nothing"
    assert ctx.session.move_history() == history
    assert LATE_WORDS not in body["commentary"]
    remembered = " ".join(message["content"] for message in ctx.transcript.memory())
    assert LATE_WORDS not in remembered
    assert LATE_WORDS not in json.dumps(client.get("/api/state").json())
    # Nor in what the next turn's planner is shown.
    provider.rescript(text_turn("nothing to do"))
    provider.words = "All good."
    client.post("/api/command", json={"text": "how am I doing?"})
    assert provider.calls, "the next turn reached the model"
    assert LATE_WORDS not in json.dumps([call["messages"] for call in provider.calls])


def test_a_delegate_turn_is_bounded_the_same_way(blocked):
    """The conductor's route runs the same pipeline (`_run_command`), so the
    bound lives in the brain and reaches it without a second fix."""
    provider = blocked()
    client, ctx, _ = build_brain_route(provider)
    conversation = client.post("/api/agent/conversations", json={}).json()["id"]

    exchange = client.post(
        f"/api/agent/conversations/{conversation}/messages",
        json={"content": "push the king pawn"},
    ).json()

    assert closer_still_writing(provider)
    assert ctx.session.move_history() == ["e4", "e5"]
    assert exchange["assistant_message"]["content"] == "e4. e5."


def test_the_trace_records_one_late_brain_turn(tmp_path, blocked):
    path = tmp_path / "turns.jsonl"
    provider = blocked()
    client, _, _ = build_brain_route(provider, tracer=JsonlTracer(path))

    client.post("/api/command", json={"text": "push the king pawn"})
    assert closer_still_writing(provider)
    provider.release.set()
    assert provider.finished.wait(timeout=PATIENCE)

    lines = path.read_text().splitlines()
    assert len(lines) == 1, "the late thread writes no record of its own"
    record = json.loads(lines[0])
    assert record["route"] == "brain"
    assert record["reaction_late"] is True
    assert record["stop_reason"] == "completed"
    assert record["provider_failure"] == ""
    assert record["mutations"] == 2
    # Planner, handoff, and the closer the turn stopped waiting for.
    assert record["model_calls"] == 3
    assert record["model_latencies_ms"][-1] >= BUDGET * 1000 * 0.9
    assert [(c["phase"], c["status"]) for c in record["calls"]] == [
        ("planner", "ok"),
        ("planner", "ok"),
        ("closer", "late"),
    ]
    assert record["calls"][-1]["budget_ms"] == round(BUDGET * 1000)
    assert record["planning"]["deadline_ms"] is not None


def test_a_closer_inside_the_budget_is_spoken_before_the_reply():
    """React-before-reply, untouched: Glitch's words, then the app's
    announcement, and nothing recorded as late."""
    provider = ScriptedProvider(
        tool_calls_turn(("make_move", {"move": "e4", "source": "said_the_move"})),
        text_turn("played e4"),
        text_turn("King pawn, straight down the middle."),
    )
    client, ctx, _ = build_brain_route(provider, closing=5.0, ceiling=5.0)

    body = client.post("/api/command", json={"text": "push the king pawn"}).json()

    assert body["commentary"] == "King pawn, straight down the middle.\n\ne5."
    assert ctx.session.move_history() == ["e4", "e5"]


def test_a_question_waits_past_the_tight_budget():
    """Nothing is held behind a question's answer, so it gets the stall
    ceiling, not the move budget: a slow thoughtful answer is still spoken."""
    provider = BlockingNarratorProvider(
        text_turn("they want an assessment"), words="You're fine, bro."
    )
    client, _, _ = build_brain_route(provider, closing=0.01, ceiling=PATIENCE)
    threading.Timer(0.2, provider.release.set).start()

    body = client.post("/api/command", json={"text": "how am I doing?"}).json()

    assert body["commentary"] == "You're fine, bro."


# --- what a late closer says -------------------------------------------------


def played(san: str) -> dict:
    return {"name": "make_move", "result": {"ok": True, "legal": True, "san": san}}


def test_a_late_move_turn_says_the_moves_and_the_reply():
    session = GameSession()
    session.submit_move("e4")
    session.submit_move("e5")
    words = _late_close_words(
        [played("e4")], MoveResult(legal=True, san="e5"), True, True, session
    )
    assert words == "e4. e5."


def test_a_late_turn_that_changed_something_else_says_it_stands():
    # Never "say it again" over a change that landed: repeating the ask could
    # replay it.
    words = _late_close_words(
        [{"name": "set_verbosity", "result": {"ok": True}}],
        None,
        False,
        True,
        GameSession(),
    )
    assert words == PROVIDER_LOST_TURN_STANDS


def test_a_late_turn_that_changed_nothing_asks_again():
    words = _late_close_words([], None, False, False, GameSession())
    assert words == STUCK_REPLY


def test_a_rejected_move_is_not_announced():
    rejected = {"name": "make_move", "result": {"ok": True, "legal": False}}
    words = _late_close_words([rejected], None, False, False, GameSession())
    assert words == STUCK_REPLY
