"""The mover at its tool boundary (#472, docs/glitch-difficulty.md).

Against a scripted provider: a legal move comes back with its intent, an
illegal one is refused with the reason and retried under the budget, and every
way of running out (attempts, deadline, token cap, a dead provider) is a typed
failure, never a move. Plus what the mover is shown: the board and nothing
else, append-only per game. The exact bytes are `test_mover_bytes.py`'s.
"""

import inspect
import json

import pytest

from chessapp.brain import CALL_BAD_ARGS, CALL_FAILED, CALL_OK, CALL_TRUNCATED
from chessapp.game import GameSession
from chessapp.mover import (
    BAD_ARGS,
    ILLEGAL,
    MAX_ATTEMPTS,
    NO_CALL,
    PLAY_MOVE,
    PLAY_MOVE_TOOL,
    PLAYED,
    Mover,
    MoverFailed,
    MoverFailureKind,
    mover_messages,
)
from chessapp.profiles import DEFAULT_PROFILE, MOVER, parse_profile
from chessapp.provider import (
    ProviderFailure,
    ProviderRequestError,
    ToolCallArgumentsError,
    Usage,
)
from fakes import ScriptedProvider, text_turn, tool_calls_turn


def _play(move: str, intent: str = "fighting for the centre"):
    return tool_calls_turn(
        (PLAY_MOVE, {"move": move, "intent": intent}),
        usage=Usage(prompt_tokens=300, completion_tokens=40),
    )


def _after_e4() -> GameSession:
    session = GameSession()
    session.submit_move("e4")
    return session


class _Clock:
    """A clock that moves only when told to, or by `step` on every read."""

    def __init__(self, step: float = 0.0) -> None:
        self.now = 100.0
        self.step = step

    def __call__(self) -> float:
        self.now += self.step
        return self.now


def _mover(provider, profile=DEFAULT_PROFILE, clock=None) -> Mover:
    return Mover(provider, profile, clock=clock or _Clock())


# --- the move --------------------------------------------------------------


def test_the_profiles_thinking_budget_reaches_every_call():
    profile = parse_profile(
        "p", "[phases.mover]\nmax_tokens = 4096\nthinking_budget = 1024\n", "test"
    )
    provider = ScriptedProvider(_play("Ke2"), _play("e5"))
    _mover(provider, profile).choose(_after_e4())
    assert [call["thinking_budget"] for call in provider.calls] == [1024, 1024]
    provider = ScriptedProvider(_play("e5"))
    _mover(provider).choose(_after_e4())
    assert provider.calls[0]["thinking_budget"] is None


def test_the_profiles_effort_reaches_the_call():
    profile = parse_profile(
        "p",
        'effort_kwarg = "reasoning_effort"\n[phases.mover]\neffort = "low"\n',
        "test",
    )
    provider = ScriptedProvider(_play("e5"))
    _mover(provider, profile).choose(_after_e4())
    assert provider.calls[0]["effort"] == "low"


def test_a_legal_move_comes_back_with_its_intent_and_the_session_is_untouched():
    session = _after_e4()
    fen = session.fen()
    provider = ScriptedProvider(_play("e5"))
    choice = _mover(provider).choose(session)
    assert (choice.uci, choice.san, choice.intent) == (
        "e7e5",
        "e5",
        "fighting for the centre",
    )
    assert [a.verdict for a in choice.attempts] == [PLAYED]
    assert [c.status for c in choice.calls] == [CALL_OK]
    assert choice.calls[0].phase == MOVER
    assert choice.calls[0].prompt_tokens == 300
    # The coordinator submits the move (#471); the mover only names it.
    assert session.fen() == fen
    assert len(provider.calls) == 1


def test_uci_is_accepted_and_named_in_san():
    choice = _mover(ScriptedProvider(_play("g8f6"))).choose(_after_e4())
    assert (choice.uci, choice.san) == ("g8f6", "Nf6")


def test_the_intent_is_the_models_words_only_stripped():
    intent = "  hitting e4 —\nand getting castled soon, I hope  "
    choice = _mover(ScriptedProvider(_play("Nf6", intent))).choose(_after_e4())
    assert choice.intent == intent.strip()


def test_an_illegal_move_is_refused_with_the_reason_and_retried():
    provider = ScriptedProvider(_play("Nf3"), _play("Nf6"))
    choice = _mover(provider).choose(_after_e4())
    assert choice.san == "Nf6"
    assert [a.verdict for a in choice.attempts] == [ILLEGAL, PLAYED]
    assert "Nf3 is not a legal move" in choice.attempts[0].reason
    retry = provider.calls[1]["messages"]
    # The refusal reaches the model as its call's answer, after its own reply
    # (final answer only; thinking never round-trips).
    assert retry[-2]["role"] == "assistant"
    assert "reasoning_content" not in retry[-2]
    refusal = json.loads(retry[-1]["content"])
    assert retry[-1]["role"] == "tool"
    assert refusal["ok"] is False
    assert "Nf3 is not a legal move" in refusal["error"]
    # The opening request is the retry's prefix: the history only grows.
    assert retry[: len(provider.calls[0]["messages"])] == provider.calls[0]["messages"]


def test_an_ambiguous_move_is_refused_with_the_moves_it_could_be():
    session = GameSession(fen="4k3/8/8/8/8/8/8/1N2KN2 w - - 0 1")
    provider = ScriptedProvider(_play("Nd2"), _play("Nbd2"))
    choice = _mover(provider).choose(session)
    assert choice.uci == "b1d2"
    assert "ambiguous" in choice.attempts[0].reason
    assert "Nbd2" in choice.attempts[0].reason


def test_attempts_run_out_as_a_failure_never_a_move():
    provider = ScriptedProvider(_play("Ke2"))
    with pytest.raises(MoverFailed) as failed:
        _mover(provider).choose(_after_e4())
    assert failed.value.kind is MoverFailureKind.RETRIES
    assert len(provider.calls) == MAX_ATTEMPTS
    assert [a.verdict for a in failed.value.attempts] == [ILLEGAL] * MAX_ATTEMPTS
    assert len(failed.value.calls) == MAX_ATTEMPTS
    trace = failed.value.as_trace()
    assert trace["outcome"] == "retries"
    assert trace["uci"] is None
    assert len(trace["attempts"]) == MAX_ATTEMPTS


def test_a_reply_without_a_call_is_asked_to_call_and_counts_as_an_attempt():
    provider = ScriptedProvider(text_turn("I think e5."), _play("e5"))
    choice = _mover(provider).choose(_after_e4())
    assert [a.verdict for a in choice.attempts] == [NO_CALL, PLAYED]
    nudge = provider.calls[1]["messages"][-1]
    assert nudge["role"] == "user"
    assert PLAY_MOVE in nudge["content"]


def test_bad_arguments_are_refused_and_retried():
    provider = ScriptedProvider(
        tool_calls_turn((PLAY_MOVE, {"move": "e5"})),
        ToolCallArgumentsError(PLAY_MOVE, "not a JSON object"),
        tool_calls_turn(("make_move", {"move": "e5"})),
    )
    with pytest.raises(MoverFailed) as failed:
        _mover(provider).choose(_after_e4())
    assert failed.value.kind is MoverFailureKind.RETRIES
    assert [a.verdict for a in failed.value.attempts] == [BAD_ARGS] * 3
    assert "intent" in failed.value.attempts[0].reason
    assert [c.status for c in failed.value.calls] == [
        CALL_OK,
        CALL_BAD_ARGS,
        CALL_OK,
    ]


def test_an_empty_intent_is_refused():
    provider = ScriptedProvider(_play("e5", "  "), _play("e5"))
    choice = _mover(provider).choose(_after_e4())
    assert [a.verdict for a in choice.attempts] == [BAD_ARGS, PLAYED]


def test_the_first_legal_call_of_several_is_the_move_and_each_is_answered():
    provider = ScriptedProvider(
        tool_calls_turn(
            (PLAY_MOVE, {"move": "Ke2", "intent": "a"}),
            (PLAY_MOVE, {"move": "d5", "intent": "b"}),
            (PLAY_MOVE, {"move": "e5", "intent": "c"}),
        )
    )
    choice = _mover(provider).choose(_after_e4())
    assert (choice.san, choice.intent) == ("d5", "b")


# --- running out -------------------------------------------------------------


def test_a_reply_cut_off_by_the_token_cap_is_a_failure():
    provider = ScriptedProvider(text_turn(None, finish_reason="length"))
    with pytest.raises(MoverFailed) as failed:
        _mover(provider).choose(_after_e4())
    assert failed.value.kind is MoverFailureKind.TRUNCATED
    assert [c.status for c in failed.value.calls] == [CALL_TRUNCATED]
    assert len(provider.calls) == 1


def test_a_dead_provider_is_a_failure():
    provider = ScriptedProvider(
        ProviderRequestError("connection refused", ProviderFailure.UNREACHABLE)
    )
    with pytest.raises(MoverFailed) as failed:
        _mover(provider).choose(_after_e4())
    assert failed.value.kind is MoverFailureKind.PROVIDER
    assert [(c.status, c.failure) for c in failed.value.calls] == [
        (CALL_FAILED, "unreachable")
    ]


def test_each_call_gets_what_is_left_of_the_deadline():
    clock = _Clock(step=10.0)
    provider = ScriptedProvider(_play("Ke2"), _play("Ke7"), _play("e5"))
    choice = _mover(provider, clock=clock).choose(_after_e4())
    assert choice.san == "e5"
    timeouts = [call["timeout"] for call in provider.calls]
    assert timeouts == sorted(timeouts, reverse=True)
    assert all(0 < t < 90.0 for t in timeouts)
    assert [c.budget_ms for c in choice.calls] == [round(t * 1000) for t in timeouts]


def test_a_passed_deadline_is_a_failure_before_another_call():
    clock = _Clock(step=40.0)
    provider = ScriptedProvider(_play("Ke2"))
    with pytest.raises(MoverFailed) as failed:
        _mover(provider, clock=clock).choose(_after_e4())
    assert failed.value.kind is MoverFailureKind.DEADLINE
    assert len(provider.calls) < MAX_ATTEMPTS


def test_a_provider_timeout_past_the_deadline_is_the_deadline():
    clock = _Clock()

    class _Hangs(ScriptedProvider):
        def chat(self, messages, **kwargs):
            clock.now += 120.0
            return super().chat(messages, **kwargs)

    provider = _Hangs(ProviderRequestError("read timed out"))
    with pytest.raises(MoverFailed) as failed:
        _mover(provider, clock=clock).choose(_after_e4())
    assert failed.value.kind is MoverFailureKind.DEADLINE


# --- what the call sends ---------------------------------------------------


def test_the_call_carries_the_mover_phase_settings():
    profile = parse_profile(
        "m",
        "[phases.mover]\nmax_tokens = 512\nthinking = false\n"
        "temperature = 0.6\ndeadline_s = 5\n",
        "test",
    )
    provider = ScriptedProvider(_play("e5"))
    _mover(provider, profile).choose(_after_e4())
    (call,) = provider.calls
    assert call["tools"] == [PLAY_MOVE_TOOL]
    assert call["max_tokens"] == 512
    assert call["enable_thinking"] is False
    assert call["temperature"] == 0.6
    assert call["timeout"] == 5.0


def test_by_default_the_mover_thinks():
    provider = ScriptedProvider(_play("e5"))
    _mover(provider).choose(_after_e4())
    assert provider.calls[0]["enable_thinking"] is True
    assert provider.calls[0]["max_tokens"] == 4096
    # No temperature of its own: the profile's, here the server's.
    assert provider.calls[0]["temperature"] is None


def test_a_finished_game_is_not_asked():
    session = GameSession(fen="7k/5QQ1/8/8/8/8/8/K7 b - - 0 1")
    provider = ScriptedProvider(_play("Kh7"))
    with pytest.raises(ValueError):
        _mover(provider).choose(session)
    assert provider.calls == []


# --- what the mover sees ---------------------------------------------------


def test_the_mover_is_given_a_session_and_nothing_else():
    # Decision 2 is held by construction: no conversation, settings, analysis
    # or notes can reach a call that is handed only the game.
    assert list(inspect.signature(Mover.choose).parameters) == ["self", "session"]
    assert list(inspect.signature(mover_messages).parameters) == ["session"]


def test_the_request_is_the_board_and_only_the_board():
    messages = mover_messages(_after_e4())
    assert [m["role"] for m in messages] == ["system", "user"]
    assert "as Black" in messages[0]["content"]
    user = messages[1]["content"]
    assert user.startswith("Moves so far: 1. e4\n\nPosition:\n")
    assert "FEN: rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1" in user
    assert "4 . . . . P . . ." in user
    assert user.endswith("Legal moves: " + ", ".join(_after_e4().legal_moves()))


def test_what_is_not_the_board_does_not_change_the_request():
    plain = _after_e4()
    busy = GameSession(player_color="black")
    busy.submit_move("e4")
    busy.note_difficulty({"tier": "maximum"})
    busy.renew_game_id()
    assert mover_messages(busy) == mover_messages(plain)


def test_a_game_from_a_set_up_position_numbers_its_moves_from_there():
    session = GameSession(
        fen="r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 2 3"
    )
    session.submit_move("Nf6")
    session.submit_move("Bc4")
    user = mover_messages(session)[1]["content"]
    assert user.startswith(
        "Starting position: r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R"
        " b KQkq - 2 3\nMoves so far: 3... Nf6 4. Bc4\n"
    )


def test_the_prompt_is_append_only_across_a_game():
    session = GameSession()
    session.submit_move("e4")
    first = mover_messages(session)
    session.submit_move("e5")
    session.submit_move("Nf3")
    later = mover_messages(session)
    assert later[0] == first[0]
    before, _, _ = first[1]["content"].partition("\n\nPosition:")
    assert later[1]["content"].startswith(before + " ")


def test_the_trace_says_what_was_played_and_tried():
    provider = ScriptedProvider(_play("Ke2"), _play("e5", "mirror"))
    trace = _mover(provider).choose(_after_e4()).as_trace()
    assert trace["outcome"] == PLAYED
    assert (trace["uci"], trace["san"], trace["intent"]) == ("e7e5", "e5", "mirror")
    assert [a["verdict"] for a in trace["attempts"]] == [ILLEGAL, PLAYED]
    assert trace["attempts"][0]["move"] == "Ke2"
