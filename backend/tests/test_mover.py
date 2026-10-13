"""The mover at its tool boundary (#472, docs/glitch-difficulty.md).

Against a scripted provider: a legal move comes back with its intent, an
illegal one is refused with the reason and retried under the budget, and every
way of running out (attempts, deadline, token cap, a dead provider) is a typed
failure, never a move. Looking ahead with `try_line` (#498): lines played on a
copy, capped by the profile, never an attempt. Plus what the mover is shown:
the board and nothing else, append-only per game. The exact bytes are
`test_mover_bytes.py`'s.
"""

import inspect
import json

import chess
import pytest

from chessapp.board_facts import board_facts
from chessapp.brain import CALL_BAD_ARGS, CALL_FAILED, CALL_OK, CALL_TRUNCATED
from chessapp.game import GameSession
from chessapp.mover import (
    BAD_ARGS,
    ILLEGAL,
    MAX_ATTEMPTS,
    MAX_LINE_PLIES,
    MAX_LINES,
    NO_CALL,
    NO_LOOKS,
    PLAY_MOVE,
    PLAY_MOVE_TOOL,
    PLAYED,
    TRY_LINE,
    TRY_LINE_TOOL,
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
    assert list(inspect.signature(mover_messages).parameters) == [
        "session",
        "board_facts",
        "look_calls",
    ]


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


def test_board_facts_sit_between_the_board_and_the_legal_moves():
    session = _after_e4()
    plain = mover_messages(session)[1]["content"]
    seen = mover_messages(session, board_facts=True)[1]["content"]
    before, _, legal = plain.partition("Legal moves: ")
    assert seen == (
        before
        + board_facts(chess.Board(session.fen())).text()
        + "\n\nLegal moves: "
        + legal
    )
    assert "Black: Ke8, Qd8" in seen


def test_the_profile_turns_board_facts_on():
    profile = parse_profile("p", "[phases.mover]\nboard_facts = true\n", "test")
    provider = ScriptedProvider(_play("e5"))
    _mover(provider, profile).choose(_after_e4())
    assert "Board facts (" in provider.calls[0]["messages"][1]["content"]
    provider = ScriptedProvider(_play("e5"))
    _mover(provider).choose(_after_e4())
    assert "Board facts (" not in provider.calls[0]["messages"][1]["content"]


def test_the_prompt_with_board_facts_is_append_only_across_a_game():
    session = GameSession()
    session.submit_move("e4")
    first = mover_messages(session, board_facts=True)
    session.submit_move("e5")
    session.submit_move("Nf3")
    later = mover_messages(session, board_facts=True)
    before, _, _ = first[1]["content"].partition("\n\nPosition:")
    assert later[1]["content"].startswith(before + " ")


def test_the_trace_says_what_was_played_and_tried():
    provider = ScriptedProvider(_play("Ke2"), _play("e5", "mirror"))
    trace = _mover(provider).choose(_after_e4()).as_trace()
    assert trace["outcome"] == PLAYED
    assert (trace["uci"], trace["san"], trace["intent"]) == ("e7e5", "e5", "mirror")
    assert [a["verdict"] for a in trace["attempts"]] == [ILLEGAL, PLAYED]
    assert trace["attempts"][0]["move"] == "Ke2"


# --- looking ahead (#498) ----------------------------------------------------


def _looks(calls: int = 2, extra: str = ""):
    return parse_profile("p", f"[phases.mover]\nlook_calls = {calls}\n{extra}", "t")


def _try(*lines):
    return tool_calls_turn((TRY_LINE, {"lines": list(lines)}))


def _answer(call: dict, index: int = -1) -> str:
    return call["messages"][index]["content"]


def test_without_looks_the_tool_and_its_instructions_are_left_out():
    provider = ScriptedProvider(_play("e5"))
    _mover(provider).choose(_after_e4())
    assert provider.calls[0]["tools"] == [PLAY_MOVE_TOOL]
    assert TRY_LINE not in provider.calls[0]["messages"][0]["content"]


def test_with_looks_the_tool_is_listed_and_explained():
    provider = ScriptedProvider(_play("e5"))
    _mover(provider, _looks(2)).choose(_after_e4())
    assert provider.calls[0]["tools"] == [PLAY_MOVE_TOOL, TRY_LINE_TOOL]
    system = provider.calls[0]["messages"][0]["content"]
    assert system.startswith(mover_messages(_after_e4())[0]["content"])
    assert "calling try_line, up to 2 times" in system
    assert f"up to {MAX_LINES} lines" in system


def test_a_look_plays_the_line_on_a_copy_and_is_not_an_attempt():
    session = _after_e4()
    fen = session.fen()
    provider = ScriptedProvider(_try("e5 Nf3 Nc6"), _play("e5"))
    choice = _mover(provider, _looks()).choose(session)
    assert choice.san == "e5"
    assert session.fen() == fen
    assert [a.verdict for a in choice.attempts] == [PLAYED]
    (look,) = choice.looks
    (line,) = look.lines
    assert line.played == ("e5", "Nf3", "Nc6")
    assert line.stopped == ""
    assert line.fen.startswith("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/")
    answer = _answer(provider.calls[1])
    assert provider.calls[1]["messages"][-1]["role"] == "tool"
    assert answer.startswith("Played on a copy of the board; nothing was played")
    assert "Line 1: 1... e5 2. Nf3 Nc6\nPosition at its end:\nFEN: r1bqkbnr/" in answer
    assert "White to move." in answer
    assert "Looks left: 1." in answer
    # The history only grows.
    first = provider.calls[0]["messages"]
    assert provider.calls[1]["messages"][: len(first)] == first


def test_the_end_of_a_line_shows_its_facts_and_material_whatever_the_prompt():
    # The prompt's facts are off; at a line's end they are what he looked for.
    provider = ScriptedProvider(_try("Nf6 e5"), _play("e5"))
    _mover(provider, _looks()).choose(_after_e4())
    answer = _answer(provider.calls[1])
    assert "Board facts (" not in provider.calls[0]["messages"][1]["content"]
    assert "Board facts (" in answer
    assert "Nf6: attacked by pawn e5" in answer
    assert "Material (1/3/3/5/9): White 39, Black 39; before the line" in answer


def test_a_line_that_takes_material_shows_both_counts():
    # 17... Qxd4 18. cxd4 from game 927c238e: the queen is gone at the end.
    session = GameSession(
        fen="r2r2k1/pp1qbppp/2p1p3/8/3PN2P/2P3B1/PP2QPP1/2KRR3 b - - 0 17"
    )
    provider = ScriptedProvider(_try("17... Qxd4 18. cxd4"), _play("Rf8"))
    _mover(provider, _looks()).choose(session)
    answer = _answer(provider.calls[1])
    assert "Line 1: 17... Qxd4 18. cxd4\n" in answer
    assert "White 31, Black 20; before the line White 32, Black 29." in answer


def test_an_illegal_ply_stops_the_line_with_the_reason():
    provider = ScriptedProvider(_try("e5 Nf3 Nf3", "Ke2", "Nf6"), _play("e5"))
    choice = _mover(provider, _looks()).choose(_after_e4())
    stopped, never, fine = choice.looks[0].lines
    assert stopped.played == ("e5", "Nf3")
    assert "Nf3 is not a legal move there" in stopped.stopped
    assert never.played == ()
    assert fine.played == ("Nf6",)
    answer = _answer(provider.calls[1])
    assert "Line 1: 1... e5 2. Nf3\nStopped there: Nf3 is not a legal move" in answer
    assert "Line 2: not played: Ke2 is not a legal move there; legal" in answer


def test_an_ambiguous_ply_names_the_moves_it_could_be():
    session = GameSession(fen="4k3/8/8/8/8/8/8/1N2KN2 w - - 0 1")
    provider = ScriptedProvider(_try("Nd2"), _play("Nbd2"))
    choice = _mover(provider, _looks()).choose(session)
    assert "ambiguous" in choice.looks[0].lines[0].stopped
    assert "Nbd2" in choice.looks[0].lines[0].stopped


@pytest.mark.parametrize(
    "lines",
    [
        ["1... e5 2. Nf3"],
        ["e5! Nf3!?"],
        ["e7e5 g1f3"],
        ["e5, Nf3"],
        "e5 Nf3",
        [["e5", "Nf3"]],
        ["1…e5 2.Nf3"],
    ],
)
def test_a_line_may_be_written_as_a_score_sheet_writes_it(lines):
    provider = ScriptedProvider(
        tool_calls_turn((TRY_LINE, {"lines": lines})), _play("e5")
    )
    choice = _mover(provider, _looks()).choose(_after_e4())
    assert choice.looks[0].lines[0].played == ("e5", "Nf3")


def test_a_look_plays_only_so_many_lines_and_plies():
    long = "e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O"
    provider = ScriptedProvider(_try(long, "d5", "c5", "e6"), _play("e5"))
    choice = _mover(provider, _looks()).choose(_after_e4())
    look = choice.looks[0]
    assert len(look.lines) == MAX_LINES
    assert look.dropped == ("e6",)
    assert len(look.lines[0].played) == MAX_LINE_PLIES
    assert "only the first 6 moves" in look.lines[0].stopped
    answer = _answer(provider.calls[1])
    assert "Only 3 lines a look; not played: e6" in answer


def test_a_line_ending_in_mate_says_so_and_plays_nothing_past_it():
    session = GameSession(
        fen="r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4"
    )
    provider = ScriptedProvider(_try("Qxf7# Ke7"), _play("Qxf7#"))
    choice = _mover(provider, _looks()).choose(session)
    (line,) = choice.looks[0].lines
    assert line.played == ("Qxf7#",)
    assert line.status == "checkmate, White wins"
    assert "the game is over there" in line.stopped
    assert "The game: checkmate, White wins." in _answer(provider.calls[1])


def test_a_line_sees_the_games_own_repetitions():
    session = GameSession()
    for move in ("Nf3", "Nf6", "Ng1", "Ng8", "Nf3", "Nf6", "Ng1"):
        session.submit_move(move)
    provider = ScriptedProvider(_try("Ng8"), _play("Ng8"))
    choice = _mover(provider, _looks()).choose(session)
    status = choice.looks[0].lines[0].status
    assert status == "a draw may be claimed by threefold repetition"


def test_a_look_past_the_cap_is_refused_and_is_an_attempt():
    provider = ScriptedProvider(_try("e5"), _try("d5"), _play("e5"))
    choice = _mover(provider, _looks(1)).choose(_after_e4())
    assert choice.san == "e5"
    assert len(choice.looks) == 1
    assert [a.verdict for a in choice.attempts] == [NO_LOOKS, PLAYED]
    assert "No looks left; play your move next." in _answer(provider.calls[1])
    refusal = json.loads(_answer(provider.calls[2]))
    assert refusal["ok"] is False
    assert refusal["error"] == "no looks left; play your move"
    # Still listed, so the request's prefix still caches.
    assert provider.calls[2]["tools"] == [PLAY_MOVE_TOOL, TRY_LINE_TOOL]


def test_looks_do_not_spend_attempts_but_the_run_still_ends():
    provider = ScriptedProvider(
        _try("e5"), _play("Ke2"), _try("d5"), _play("Ke2"), _try("c5"), _play("Ke2")
    )
    with pytest.raises(MoverFailed) as failed:
        _mover(provider, _looks(2)).choose(_after_e4())
    assert failed.value.kind is MoverFailureKind.RETRIES
    # Two looks, then three replies that are attempts (one of them a refused
    # third look): MAX_ATTEMPTS + look_calls calls at most.
    assert len(provider.calls) == MAX_ATTEMPTS + 2
    assert len(failed.value.looks) == 2
    assert [a.verdict for a in failed.value.attempts] == [ILLEGAL, ILLEGAL, NO_LOOKS]
    assert len(failed.value.as_trace()["looks"]) == 2


def test_a_look_and_a_move_in_one_reply_are_both_answered_in_order():
    provider = ScriptedProvider(
        tool_calls_turn(
            (TRY_LINE, {"lines": ["d5 exd5"]}),
            (PLAY_MOVE, {"move": "e5", "intent": "centre"}),
            (TRY_LINE, {"lines": ["c5"]}),
        )
    )
    choice = _mover(provider, _looks()).choose(_after_e4())
    assert choice.san == "e5"
    assert len(choice.looks) == 1
    assert len(provider.calls) == 1


def test_a_look_with_no_lines_is_refused_as_bad_arguments():
    provider = ScriptedProvider(
        tool_calls_turn((TRY_LINE, {"lines": ["  "]})),
        tool_calls_turn((TRY_LINE, {"moves": "e5"})),
        _play("e5"),
    )
    choice = _mover(provider, _looks()).choose(_after_e4())
    assert [a.verdict for a in choice.attempts] == [BAD_ARGS, BAD_ARGS, PLAYED]
    assert choice.looks == ()


def test_try_line_without_looks_is_no_tool():
    provider = ScriptedProvider(_try("e5"), _play("e5"))
    choice = _mover(provider).choose(_after_e4())
    assert [a.verdict for a in choice.attempts] == [BAD_ARGS, PLAYED]
    assert "there is no tool 'try_line'" in choice.attempts[0].reason


def test_the_deadline_covers_looks():
    clock = _Clock(step=20.0)
    provider = ScriptedProvider(_try("e5"))
    with pytest.raises(MoverFailed) as failed:
        _mover(provider, _looks(8), clock=clock).choose(_after_e4())
    assert failed.value.kind is MoverFailureKind.DEADLINE
    assert failed.value.looks
    assert failed.value.attempts == ()


def test_the_trace_carries_each_look():
    provider = ScriptedProvider(_try("e5 Nf3", "Ke2"), _play("e5"))
    trace = _mover(provider, _looks()).choose(_after_e4()).as_trace()
    (look,) = trace["looks"]
    assert [line["played"] for line in look["lines"]] == [["e5", "Nf3"], []]
    assert look["lines"][0]["asked"] == "e5 Nf3"
    assert "not a legal move" in look["lines"][1]["stopped"]
    plain = _mover(ScriptedProvider(_play("e5"))).choose(_after_e4()).as_trace()
    assert plain["looks"] == []
