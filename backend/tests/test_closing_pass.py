"""The tool-free closing pass, asserted end-to-end (Sprint 2, audit item 9).

The planner/narrator split made the closing pass tool-free *by construction*:
the phase that writes the player's reply is a separate model call offered no
tools. These tests pin that guarantee at the pipeline level, with the real
registry and turn coordinator behind a real `LlamaBrain`, so no future wiring
change can quietly hand tools back to the phase that talks:

- On every route (brain loop, fast path, board drag), the model call that
  produces player-facing text carries `tools=None`.
- Once the turn's mutation budget is spent — the phase machine's one player
  move, or the command window's one destructive op — the tools the planner
  still holds are dead: a further mutation comes back as error result data and
  the board stands.
- A refusal leaves the turn exactly as it found it — the board it did not
  change, the engine reply the open turn still owes, and nothing armed for a
  later yes to find. The same wiring is what shows it: only a real brain over a
  scripted provider puts a refused call inside a real planner batch, and a real
  `read_answer` in front of a real destructive gate.
- What the narrator says about a batch is checked against every board that
  batch actually held, not just the two ends of the command.

`test_llama_brain.py` covers the same properties at the brain seam with a fake
dispatcher; this file is the route-level assertion the audit asked for.
"""

import json

import pytest
from fastapi.testclient import TestClient

from chessapp import api
from chessapp.api import MOVE_ADVICE_REPLY, UNVERIFIED_CLAIM_REPLY, create_app
from chessapp.coordinator import TurnCoordinator, TurnPhase
from chessapp.game import GameSession
from chessapp.llama_brain import LlamaBrain
from chessapp.tools import BOARD_STATE_TOOLS, ToolContext, build_registry
from chessapp.trace import JsonlTracer
from fakes import FakeEngine, ScriptedProvider, text_turn, tool_calls_turn

PLANNER = "Pick the tools."
PERSONA = "You are a chess opponent."

# A fifty-move claim already available, on a position nobody played into — so
# the confirmation gate stands aside (no player investment) and the command
# window's destructive budget is the only thing left in the way, which is what
# this file is about.
FIFTY_MOVE_FEN = "8/8/8/4k3/8/8/4K3/6R1 w - - 100 80"


class CollectedTurns:
    """The app's own `Tracer` seam (`trace.Tracer`), kept in memory.

    `mutations` — how many times the board actually moved — is stated in the
    trace and nowhere else in the HTTP answer, and it is the whole question
    when a call in the batch was refused.
    """

    def __init__(self) -> None:
        self.records: list[dict] = []

    def record(self, turn: dict) -> None:
        self.records.append(turn)


def make_client(
    *turns,
    ctx: ToolContext | None = None,
    tracer=None,
    coordinator: TurnCoordinator | None = None,
):
    """The full pipeline over a real `LlamaBrain` whose provider is scripted —
    the one wiring that lets a route-level test see which calls carried tools,
    exactly as app assembly builds it (shared registry + coordinator,
    `atomic_exchange=False`, board-state reads not offered).

    `coordinator` is the same escape hatch `ctx` is: pass one to keep a handle
    on the turn machine when what the test is about is the phase the command
    left it in, rather than the board or the words. `tracer` is the third, for
    the refusal and guard tests below: whether the guard fired, and how many
    times the board moved, are fields of the turn record, and reading them
    there rather than inferring them from the text is what tells "the model
    said this" from "the model said this and the app let it through"."""
    if ctx is None:
        ctx = ToolContext(session=GameSession(), engine=FakeEngine())
    if coordinator is None:
        coordinator = TurnCoordinator(ctx)
    registry = build_registry(ctx, coordinator, atomic_exchange=False)
    provider = ScriptedProvider(*turns)
    brain = LlamaBrain(
        provider=provider,
        dispatcher=registry,
        tool_definitions=registry.definitions(exclude=BOARD_STATE_TOOLS),
        system_prompt=PERSONA,
        planner_prompt=PLANNER,
    )
    app = create_app(
        ctx,
        brain=brain,
        registry=registry,
        coordinator=coordinator,
        tracer=tracer,
    )
    return TestClient(app), provider, ctx


@pytest.fixture
def trace_path(tmp_path):
    return tmp_path / "turns.jsonl"


def last_turn(trace_path) -> dict:
    """The turn record the command just wrote."""
    lines = [line for line in trace_path.read_text().splitlines() if line.strip()]
    return json.loads(lines[-1])


def offered_tools(provider: ScriptedProvider) -> list[bool]:
    """Whether each recorded model call held tools, in order."""
    return [call["tools"] is not None for call in provider.calls]


def finished(ctx: ToolContext) -> ToolContext:
    """Fool's mate: a finished game, so the confirmation gate stands aside and
    only the command window's destructive budget is in the way."""
    for san in ("f3", "e5", "g4", "Qh4"):
        ctx.session.submit_move(san)
    return ctx


# --- the brain route: the player-facing call is tool-free, and the tools the
# --- planner keeps cannot mutate past the budget


def test_brain_route_closes_tool_free_and_a_second_move_is_dead():
    client, provider, ctx = make_client(
        tool_calls_turn(("make_move", {"move": "e4"})),
        tool_calls_turn(("make_move", {"move": "d4"})),  # budget already spent
        text_turn("note: played e4, second move refused"),
        text_turn("e4 it is."),
    )

    response = client.post("/api/command", json={"text": "play e4 and d4"}).json()

    # Planner turns hold tools; the narrator — the call whose text the player
    # reads — holds none. Structural, not the model declining.
    assert offered_tools(provider) == [True, True, True, False]
    assert response["commentary"].startswith("e4 it is.")
    # The second mutation was refused as result data, inside the same turn.
    results = [r["result"] for r in response["tool_results"]]
    assert results[0]["ok"] is True
    assert results[1]["ok"] is False
    # One player move, one engine reply — nothing the dead call could add.
    assert ctx.session.move_history() == ["e4", "e5"]


def test_brain_route_destructive_budget_spent_still_closes_tool_free():
    ctx = finished(ToolContext(session=GameSession(), engine=FakeEngine()))
    client, provider, ctx = make_client(
        tool_calls_turn(("new_game", {})),
        tool_calls_turn(("resign", {})),  # second destructive op in one command
        text_turn("note: reset, resign refused"),
        text_turn("Fresh board. Your move."),
        ctx=ctx,
    )

    response = client.post("/api/command", json={"text": "new game"}).json()

    assert offered_tools(provider) == [True, True, True, False]
    results = [r["result"] for r in response["tool_results"]]
    assert results[0]["ok"] is True
    assert results[1]["ok"] is False
    assert not ctx.session.is_game_over(), "the refused resign never ran"
    assert response["commentary"] == "Fresh board. Your move."


def test_brain_route_closes_tool_free_when_the_planner_repeats_itself():
    """The repeat stop is a *third* way into the narrator, and it must arrive
    the same way the other two do: tool-free, with the player reading real
    commentary rather than the pipeline's canned stuck line (which is what a
    budget stop would have produced before the loop learned to end a planner
    that had stopped making progress)."""
    client, provider, _ = make_client(
        tool_calls_turn(("evaluate_position", {})),
        tool_calls_turn(("evaluate_position", {})),  # nothing new — the last turn
        text_turn("Dead even. Your move."),
    )

    response = client.post("/api/command", json={"text": "how's it looking?"}).json()

    # Two planner turns, then the narrator — no third planner turn was bought.
    assert offered_tools(provider) == [True, True, False]
    assert response["commentary"] == "Dead even. Your move."


def test_a_claimed_draw_spends_the_budget_and_still_closes_tool_free():
    """Claiming a draw ends the game, so it is one of the ops the command window
    budgets: the second one is dead result data and the narrator that tells the
    player about it still holds no tools."""
    ctx = ToolContext(session=GameSession(fen=FIFTY_MOVE_FEN), engine=FakeEngine())
    client, provider, ctx = make_client(
        tool_calls_turn(("claim_draw", {})),
        tool_calls_turn(("new_game", {})),  # second destructive op in one command
        text_turn("note: draw claimed, reset refused"),
        text_turn("Half a point each. Done."),
        ctx=ctx,
    )

    response = client.post("/api/command", json={"text": "claim the draw"}).json()

    assert offered_tools(provider) == [True, True, True, False]
    results = [r["result"] for r in response["tool_results"]]
    assert results[0]["ok"] is True
    assert results[1]["ok"] is False
    assert ctx.session.outcome().termination == "fifty_moves", "the refused reset"
    assert response["commentary"] == "Half a point each. Done."


# --- a refusal leaves the turn as it found it


def developed(ctx: ToolContext) -> ToolContext:
    """A real game under way: the player has something to lose, so the
    destructive gate arms instead of standing aside."""
    for san in ("e4", "e5", "Nf3", "Nc6"):
        ctx.session.submit_move(san)
    return ctx


@pytest.mark.parametrize(
    ("plies", "took_back", "retry", "history", "announcement"),
    [
        # A hundred half-moves were never played, so nothing is taken back —
        # and the turn that played e4 is still owed the engine's answer to it.
        (100, False, "never", ["e4", "e5"], "\n\ne5."),
        # The same refusal in JSON's one number type. It reaches the same
        # handler as an int (the registry narrows it) and is refused there.
        (100.0, False, "never", ["e4", "e5"], "\n\ne5."),
        # And the same spelling on a count that *is* takeable: 1.0 is one
        # half-move, it pops e4, and now there is nothing left to answer.
        (1.0, True, None, [], ""),
    ],
)
def test_whether_a_reply_is_owed_is_decided_by_whether_the_undo_landed(
    plies, took_back, retry, history, announcement
):
    """`make_move` and `undo` in one planner batch, with the undo refused.

    The tool used to abandon the open turn before finding out whether it could
    take anything back, so a refused undo threw away the engine reply the move
    beside it had just earned: e4 on the board, Black to move, the coordinator
    back awaiting the player, and nobody left to answer (audit 2026-09-05,
    finding 1). A refusal touches neither the coordinator nor the board now, so
    the open turn keeps owning its reply and the pipeline collects it at the
    close — while an undo that *lands* still takes the owed reply with it,
    which is the same rule read the other way.
    """
    turns = CollectedTurns()
    client, _, ctx = make_client(
        tool_calls_turn(("make_move", {"move": "e4"}), ("undo", {"plies": plies})),
        text_turn("played e4; the takeback is another matter"),
        text_turn("e4 is on."),
        tracer=turns,
    )

    response = client.post("/api/command", json={"text": "play e4 and undo that"})

    assert response.status_code == 200
    body = response.json()
    undo_result = body["tool_results"][1]["result"]
    assert undo_result["ok"] is took_back
    assert undo_result.get("retry") == retry
    assert ctx.session.move_history() == history
    assert ctx.session.turn == ctx.session.player_color, "the player is to move"
    # The app's own reply announcement rides on the commentary exactly when a
    # reply was owed and collected.
    assert body["commentary"] == f"e4 is on.{announcement}"
    (record,) = turns.records
    assert record["mutations"] == 2


def test_an_integral_float_argument_lands_instead_of_ending_the_command():
    """The batch that used to escape as a 500 (audit 2026-09-05, finding 5).

    `1.0` passes the `integer` schema — JSON has one number type and the
    validator honors that — and then met a Python slice, which did not: the
    `TypeError` left `dispatch` entirely, taking the rest of the batch and the
    trace with it and reaching the player as a server error on a board that had
    already moved. The value is narrowed to the integer the schema called it,
    so the call it was is the call that runs and every sibling still gets its
    turn.
    """
    client, _, ctx = make_client(
        tool_calls_turn(
            ("make_move", {"move": "e4"}),
            ("undo", {"plies": 1.0}),
            ("set_voice_output", {"enabled": True}),
        ),
        text_turn("moved, took it back, voice on"),
        text_turn("Taken back. Voice is on."),
    )

    response = client.post(
        "/api/command", json={"text": "play e4, undo it, and turn voice on"}
    )

    assert response.status_code == 200
    body = response.json()
    assert [r["name"] for r in body["tool_results"]] == [
        "make_move",
        "undo",
        "set_voice_output",
    ]
    assert body["tool_results"][1]["result"]["undone"] == ["e4"]
    assert ctx.session.move_history() == []
    assert ctx.settings.voice_output is True, "the sibling after it still ran"


def test_a_truncated_reading_leaves_the_armed_resignation_un_run():
    """The confirmation reader's cap fires directly in front of the destructive
    gate, so a fragment read generously would be a game ended by a truncation.

    The planner asks to resign, the gate refuses and arms it, and the player
    answers in their own words — a reply only the model can read. The reading
    comes back cut off, which is not a verdict, so it lands on `unrelated`: the
    op is dropped rather than run, and the utterance falls through to an
    ordinary turn.
    """
    ctx = developed(ToolContext(session=GameSession(), engine=FakeEngine()))
    client, _, ctx = make_client(
        tool_calls_turn(("resign", {})),
        text_turn("asked first"),
        text_turn("That's the game if you mean it. Resign?"),
        # The reading, cut off mid-word by `_ANSWER_MAX_TOKENS`.
        text_turn("confirm", finish_reason="length"),
        text_turn("nothing to do"),
        text_turn("Still your move, then."),
        ctx=ctx,
    )
    client.post("/api/command", json={"text": "I've had enough of this"})
    assert ctx.pending is not None and ctx.pending.name == "resign"

    body = client.post("/api/command", json={"text": "go on then"}).json()

    assert not ctx.session.is_game_over(), "a truncation may not end a game"
    assert ctx.pending is None, "and the op it could not answer for is gone"
    assert body["tool_results"] == []
    assert body["commentary"] == "Still your move, then."


# --- a restored position is settled, and the app says what it played
#
# A takeback or a restore can hand back a board with the engine to move and no
# turn open over it — nothing was owed, because nothing was asked. The
# coordinator settles it inside the tool, and the pipeline announces the move it
# made with the same deterministic line every other engine move gets: a
# voice-first player who asked to load a game must not be left with a board that
# moved twice in silence.


def test_a_resumed_mid_exchange_save_finishes_the_exchange(tmp_path):
    """The audit's second core probe, end to end (finding 2). "play e4 and save
    this as half" writes the board *between* the player's move and the reply —
    the live game then settles at [e4, e5], but the file holds one ply — and
    loading it used to hand back [e4] with Black to move and nobody to move it.
    """
    ctx = ToolContext(session=GameSession(), engine=FakeEngine(), save_dir=tmp_path)
    coordinator = TurnCoordinator(ctx)
    client, _, ctx = make_client(
        tool_calls_turn(("make_move", {"move": "e4"}), ("save_game", {"name": "half"})),
        text_turn("moved and saved"),
        text_turn("Saved."),
        tool_calls_turn(("resume_game", {"name": "half"})),
        text_turn("loaded it"),
        text_turn("Back where you left it."),
        ctx=ctx,
        coordinator=coordinator,
    )

    client.post("/api/command", json={"text": "play e4 and save this as half"})
    assert ctx.session.move_history() == ["e4", "e5"], "the live game settled"

    body = client.post("/api/command", json={"text": "load half"}).json()

    assert ctx.session.move_history() == ["e4", "e5"], "the save, plus a fresh reply"
    assert ctx.session.turn == ctx.session.player_color == "white"
    assert coordinator.phase == TurnPhase.AWAITING_PLAYER
    resumed = body["tool_results"][0]["result"]
    assert resumed["engine_move"]["san"] == "e5"
    assert body["commentary"] == "Back where you left it.\n\ne5."


def test_an_odd_takeback_is_announced_like_any_other_engine_move():
    """The other restore, and the one that makes the announcement's case: the
    player asked for one half-move back and the board moved twice. The narrator
    naming the settled move survives the honesty guard — the `engine_move` the
    result carries is evidence like the reply's own is."""
    ctx = ToolContext(session=GameSession(), engine=FakeEngine("e7e5"))
    for san in ("e4", "c5"):
        assert ctx.session.submit_move(san).legal
    turns = CollectedTurns()
    client, _, ctx = make_client(
        tool_calls_turn(("undo", {"plies": 1})),
        text_turn("popped one half-move"),
        text_turn("Rolled it back, and I'm on e5 again."),
        ctx=ctx,
        tracer=turns,
    )

    body = client.post(
        "/api/command", json={"text": "take back just that last half-move"}
    ).json()

    assert body["tool_results"][0]["result"]["engine_move"]["san"] == "e5"
    assert body["commentary"] == "Rolled it back, and I'm on e5 again.\n\ne5."
    assert ctx.session.move_history() == ["e4", "e5"]
    assert ctx.session.turn == ctx.session.player_color
    (record,) = turns.records
    assert record["mutations"] == 2, "the takeback and the move that answered it"
    assert record["guarded"] is False


class FailEngine(FakeEngine):
    """An engine that dies when asked to think. The background computation
    swallows its own failure by design, so this surfaces in the collector's
    synchronous retry — where the turn is."""

    def choose_move(self, session):
        raise ValueError("engine died")


def test_an_engine_that_died_mid_command_is_healed_by_the_next_one():
    """`docs/turn-coordinator.md` has always said the next command heals a turn
    a failure left open, and for an engine failure it was not true: the phase
    stayed at `engine_calculating`, where `_require` refuses every ordinary
    player move, so only an undo, a reset or a resume could dig the game out.
    The failure is loud — nothing swallows it, and the command that hit it ends
    in the error it raised — but the player's move stands and the reply is still
    owed, so the next command pays one utterance and the game goes on."""
    ctx = ToolContext(session=GameSession(), engine=FailEngine())
    coordinator = TurnCoordinator(ctx)
    client, _, ctx = make_client(
        tool_calls_turn(("make_move", {"move": "e4"})),
        text_turn("played e4"),
        text_turn("e4, then."),
        tool_calls_turn(("make_move", {"move": "d4"})),
        text_turn("that one did not land"),
        text_turn("Couldn't play that."),
        ctx=ctx,
        coordinator=coordinator,
    )

    with pytest.raises(ValueError, match="engine died"):
        client.post("/api/command", json={"text": "play e4"})
    assert ctx.session.move_history() == ["e4"], "the player's move stands"
    assert coordinator.phase == TurnPhase.PLAYER_MOVE_APPLIED, "the reply is owed"

    ctx.engine = FakeEngine("e7e5")
    body = client.post("/api/command", json={"text": "play d4"}).json()

    assert body["tool_results"][0]["result"]["ok"] is False, "refused as mid-turn"
    assert ctx.session.move_history() == ["e4", "e5"], "one reply, to e4"
    assert ctx.session.turn == ctx.session.player_color
    assert coordinator.phase == TurnPhase.AWAITING_PLAYER
    assert body["commentary"].startswith("Couldn't play that.")


# --- the routes outside the loop: the model is only ever reached tool-free


def test_fast_path_route_never_reaches_the_model_with_tools():
    client, provider, ctx = make_client(text_turn("The classic."))

    client.post("/api/command", json={"text": "e4"})

    assert len(provider.calls) >= 1, "the observe beat narrated"
    assert offered_tools(provider) == [False] * len(provider.calls)
    assert ctx.session.move_history()[:1] == ["e4"]


def test_board_route_never_reaches_the_model_with_tools():
    client, provider, ctx = make_client(text_turn("Bold opening."))

    client.post("/api/game/move", json={"move": "e2e4"})

    assert len(provider.calls) >= 1, "the observe beat narrated"
    assert offered_tools(provider) == [False] * len(provider.calls)
    assert ctx.session.move_history()[:1] == ["e4"]


# --- the honesty guard's evidence: every board the batch really held.
#
# Audit findings 6 and 7 (2026-09-05), both reproduced here from the audit's own
# probes. Both are the guard firing on a *correct* answer, so both are answered
# by loosening the guard rather than by scripting what the narrator says.
#
# Finding 7 is the batch's boards. The narrator reads the batch's tool results,
# and those were produced on the boards the tools ran on — but the guard used to
# stand on `fen_before` and the final position only, so `make_move(exd5)` +
# `describe_position()` reporting a pawn was called a lie the moment the engine
# recaptured. The command now keeps the position each mutating dispatch left
# behind, which is the same fact the fast path has always handed over as its
# observation board.
#
# Finding 6 is the clarifying question. "Do you mean Nf3 or Nh3?" names two
# playable moves and hands over neither; it is the answer an ambiguous request
# deserves and the advice guard ate it whole.
#
# The converses are the point of the section: an invented fact is invented from
# every board the turn held, and one named move is still a hint.


def exchange_client(narration: str, tracer=None):
    """A brain batch that captures, describes, and is recaptured: after `e4 d5`
    the planner plays exd5 and reads the position — the player is a pawn up
    there — and the engine answers Qxd5 before the guard ever looks."""
    ctx = ToolContext(session=GameSession(), engine=FakeEngine(reply_uci="d8d5"))
    for san in ("e4", "d5"):
        assert ctx.session.submit_move(san).legal
    return make_client(
        tool_calls_turn(("make_move", {"move": "exd5"}), ("describe_position", {})),
        text_turn("note: took on d5, player a pawn up"),
        text_turn(narration),
        ctx=ctx,
        tracer=tracer,
    )


def test_a_count_from_a_board_the_batch_held_survives_the_guard(trace_path):
    client, _, ctx = exchange_client(
        "You are up a pawn.", tracer=JsonlTracer(trace_path)
    )

    response = client.post(
        "/api/command", json={"text": "grab the pawn on d5 and tell me the material"}
    ).json()

    assert ctx.session.move_history() == ["e4", "d5", "exd5", "Qxd5"]
    assert ctx.session.material_balance() == 0, "the recapture levelled it again"
    assert response["commentary"] == "You are up a pawn.\n\nQxd5."
    assert last_turn(trace_path)["guarded"] is False


def test_a_count_no_board_the_batch_held_backs_is_still_guarded(trace_path):
    """Widening to the batch's own boards is not the same as waving counts
    through: nobody was ever a rook up in that exchange."""
    client, _, _ = exchange_client("You are up a rook.", tracer=JsonlTracer(trace_path))

    response = client.post(
        "/api/command", json={"text": "grab the pawn on d5 and tell me the material"}
    ).json()

    assert response["commentary"] == f"{UNVERIFIED_CLAIM_REPLY}\n\nQxd5."
    assert last_turn(trace_path)["guarded"] is True


def test_a_move_only_a_board_the_batch_never_held_makes_legal_is_guarded(trace_path):
    """The same line for the move class. Nxd5 is a move some position would
    make legal — a white knight on c3 or f4 — and none of the three this
    command actually passed through is that position."""
    client, _, ctx = exchange_client(
        "Nxd5 was cleaner.", tracer=JsonlTracer(trace_path)
    )

    response = client.post(
        "/api/command", json={"text": "grab the pawn on d5 and tell me the material"}
    ).json()

    assert "Nxd5" not in ctx.session.legal_moves()
    assert response["commentary"] == f"{UNVERIFIED_CLAIM_REPLY}\n\nQxd5."
    assert last_turn(trace_path)["guarded"] is True


def test_the_command_trail_holds_the_board_after_each_mutating_call(monkeypatch):
    """The trail itself, read at the seam that consumes it: one FEN per call
    that moved the board, in the order the batch ran them.

    Filled from the registry's mutation hook, which is the one road every
    model-initiated mutation takes — so the boards the guard is handed are
    boards the game really reached, and a call that changed nothing adds
    nothing."""
    ctx = ToolContext(session=GameSession(), engine=FakeEngine())
    for san in ("e4", "e5"):
        assert ctx.session.submit_move(san).legal
    observed: list[list[str]] = []
    real = api._verified_facts

    def spy(ctx, tool_results, engine_reply, fen_before, fens_observed=()):
        observed.append(list(fens_observed))
        return real(ctx, tool_results, engine_reply, fen_before, fens_observed)

    monkeypatch.setattr(api, "_verified_facts", spy)
    client, _, ctx = make_client(
        tool_calls_turn(("undo", {}), ("make_move", {"move": "d4"})),
        text_turn("note: took it back and played d4"),
        text_turn("Queen's pawn instead."),
        ctx=ctx,
    )
    # What the two mutating calls leave behind: the board the takeback restores,
    # then the board d4 stands on. The engine's reply lands after, through the
    # coordinator rather than the registry, so it is the *final* board and not a
    # step on the trail.
    replay = GameSession()
    after_undo = replay.fen()
    assert replay.submit_move("d4").legal
    after_d4 = replay.fen()

    client.post("/api/command", json={"text": "take that back and play d4"})

    assert ctx.session.move_history() == ["d4", "e5"]
    assert observed == [[after_undo, after_d4]]


def test_a_read_only_command_leaves_the_trail_empty(monkeypatch):
    """Nothing moved, so there is no board between the two ends to remember."""
    observed: list[list[str]] = []
    real = api._verified_facts

    def spy(ctx, tool_results, engine_reply, fen_before, fens_observed=()):
        observed.append(list(fens_observed))
        return real(ctx, tool_results, engine_reply, fen_before, fens_observed)

    monkeypatch.setattr(api, "_verified_facts", spy)
    client, _, ctx = make_client(
        tool_calls_turn(("describe_position", {})),
        text_turn("note: described it"),
        text_turn("Even material, nothing developed."),
    )

    client.post("/api/command", json={"text": "what's the position?"})

    assert ctx.session.move_history() == []
    assert observed == [[]]


def test_a_clarifying_question_naming_two_moves_survives_the_advice_guard(trace_path):
    """Audit finding 6, from its own probe. "move my kings knight" is genuinely
    ambiguous on a fresh board, the narrator asks the right question, and the
    advice guard replaced the whole thing with the unbacked-move correction."""
    client, _, ctx = make_client(
        text_turn("note: ambiguous, ask which knight"),
        text_turn("Do you mean Nf3 or Nh3?"),
        tracer=JsonlTracer(trace_path),
    )

    response = client.post("/api/command", json={"text": "move my kings knight"}).json()

    assert {"Nf3", "Nh3"} <= set(ctx.session.legal_moves()), "both really playable"
    assert ctx.session.move_history() == [], "asking is not moving"
    assert response["commentary"] == "Do you mean Nf3 or Nh3?"
    assert last_turn(trace_path)["guarded"] is False


def test_handing_over_one_unbacked_move_is_still_advice(trace_path):
    """The converse, and the reason the loosening is a sentence rule rather
    than an exemption for knights: naming one playable move nothing checked is
    the leak the guard was built for, question mark or not."""
    client, _, _ = make_client(
        text_turn("note: tell them to play Nf3"),
        text_turn("Nf3 is the move."),
        tracer=JsonlTracer(trace_path),
    )

    response = client.post("/api/command", json={"text": "move my kings knight"}).json()

    assert response["commentary"] == MOVE_ADVICE_REPLY
    traced = last_turn(trace_path)
    assert traced["guarded"] is True
    assert traced["guarded_claims"] == ["move_advice"]
