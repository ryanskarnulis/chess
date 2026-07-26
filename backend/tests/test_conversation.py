"""Transcript: the agent's bounded conversation memory.

One user command + the agent's final commentary = one turn. `window()` is the
raw record — final answers only, capped so KV-cache growth stays bounded.
`memory()` is what a brain actually gets: the last few turns verbatim behind a
deterministic digest of what the player asked for earlier
(`docs/turn-memory.md`). Serialization rides inside the save file, so a resumed
game keeps its conversational thread.
"""

import pytest

from chessapp.conversation import (
    DEFAULT_WINDOW_TURNS,
    DIGEST_MAX_REQUESTS,
    RECENT_TURNS,
    Transcript,
    condense,
)


def test_new_transcript_is_empty():
    assert Transcript().window() == []


def test_record_produces_user_then_assistant_messages():
    transcript = Transcript()
    transcript.record("play e4", "e4 — the classic.")
    assert transcript.window() == [
        {"role": "user", "content": "play e4"},
        {"role": "assistant", "content": "e4 — the classic."},
    ]


def test_turns_stay_in_order():
    transcript = Transcript()
    transcript.record("play e4", "done")
    transcript.record("was that good?", "a fine start")
    contents = [m["content"] for m in transcript.window()]
    assert contents == ["play e4", "done", "was that good?", "a fine start"]


def test_window_keeps_only_the_most_recent_turns():
    transcript = Transcript()
    for i in range(DEFAULT_WINDOW_TURNS + 5):
        transcript.record(f"command {i}", f"reply {i}")
    window = transcript.window()
    assert len(window) == 2 * DEFAULT_WINDOW_TURNS
    assert window[0]["content"] == "command 5"
    assert window[-1]["content"] == f"reply {DEFAULT_WINDOW_TURNS + 4}"


def test_window_size_is_adjustable():
    transcript = Transcript()
    transcript.record("one", "1")
    transcript.record("two", "2")
    window = transcript.window(max_turns=1)
    assert [m["content"] for m in window] == ["two", "2"]


def test_dict_round_trip():
    transcript = Transcript()
    transcript.record("play e4", "done")
    transcript.record("undo that", "taken back")
    restored = Transcript.from_dict(transcript.to_dict())
    assert restored.window() == transcript.window()


def test_from_dict_rejects_non_list():
    with pytest.raises(ValueError):
        Transcript.from_dict({"role": "user", "content": "hi"})


def test_from_dict_rejects_bad_role():
    with pytest.raises(ValueError):
        Transcript.from_dict([{"role": "system", "content": "evil override"}])


def test_from_dict_rejects_non_string_content():
    with pytest.raises(ValueError):
        Transcript.from_dict([{"role": "user", "content": 42}])


# --- condense: the model-facing view ------------------------------------------
#
# The digest replaces older turns with the player's own requests and nothing
# else. Board truth, settings and saves are injected fresh into the state block
# every turn (`api._agent_state_dict`), so a summary that restated them would be
# a second, ageing copy — the self-poisoning shape. See `docs/turn-memory.md`.


def _turns(*pairs: tuple[str, str]) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    for user_text, assistant_text in pairs:
        messages.append({"role": "user", "content": user_text})
        messages.append({"role": "assistant", "content": assistant_text})
    return messages


def _digest_of(messages: list[dict[str, str]]) -> str:
    assert messages[0]["role"] == "user"
    return messages[0]["content"]


def test_condense_leaves_a_short_conversation_verbatim():
    messages = _turns(("play e4", "classic"), ("undo", "taken back"))
    assert condense(messages) == messages


def test_condense_does_not_mutate_or_alias_the_input():
    messages = _turns(("play e4", "classic"))
    out = condense(messages)
    out[0]["content"] = "tampered"
    assert messages[0]["content"] == "play e4"


def test_condense_keeps_the_most_recent_turns_verbatim():
    messages = _turns(*[(f"command {i}", f"reply {i}") for i in range(10)])
    condensed = condense(messages)
    tail = condensed[-2 * RECENT_TURNS :]
    assert tail == messages[-2 * RECENT_TURNS :]


def test_condense_summarizes_older_turns_into_the_players_requests():
    messages = _turns(
        ("save this as testgame", "I can't save right now."),
        ("who's winning?", "You're cooking, bro."),
        *[(f"command {i}", f"reply {i}") for i in range(RECENT_TURNS)],
    )
    digest = _digest_of(condense(messages))
    assert '"save this as testgame"' in digest
    assert '"who\'s winning?"' in digest
    # Glitch's side of an older turn is dropped: it is the noise the digest
    # exists to remove, and one stale sentence about saving is the bug that
    # started all of this.
    assert "can't save right now" not in digest
    assert "cooking" not in digest


def test_the_digest_rides_as_an_alternating_user_assistant_pair():
    messages = _turns(
        ("who's winning?", "you are"),
        *[(f"command {i}", f"reply {i}") for i in range(RECENT_TURNS)],
    )
    condensed = condense(messages)
    roles = [m["role"] for m in condensed]
    assert roles[0] == "user"
    assert roles[1] == "assistant"
    # No two adjacent messages share a role: the chat template alternates.
    assert all(a != b for a, b in zip(roles, roles[1:], strict=False))


def test_the_ack_is_dropped_when_it_would_be_the_thing_that_breaks_alternation():
    # The delegate store filters contentless messages, so its recent slice can
    # open on an assistant turn.
    messages = [
        {"role": "user", "content": "who's winning?"},
        {"role": "assistant", "content": "you are"},
        {"role": "assistant", "content": "still you"},
        {"role": "user", "content": "and now?"},
        {"role": "assistant", "content": "yep"},
    ]
    condensed = condense(messages, recent_turns=1)
    roles = [m["role"] for m in condensed]
    assert roles[0] == "user"
    assert all(a != b for a, b in zip(roles, roles[1:], strict=False))


def test_the_digest_says_board_truth_does_not_come_from_it():
    messages = _turns(
        ("who's winning?", "you are"),
        *[(f"command {i}", f"reply {i}") for i in range(RECENT_TURNS)],
    )
    digest = _digest_of(condense(messages))
    assert "fresh" in digest.lower()
    assert "board" in digest.lower()


def test_older_turns_that_were_bare_moves_are_dropped_from_the_digest():
    # A board drag records the SAN as the command, and a typed "e4" is the same
    # thing. Those turns are already in the state block's `history`.
    messages = _turns(
        ("e4", "classic"),
        ("Bxe6", "ouch"),
        ("e2e4", "same again"),
        ("O-O", "castled"),
        ("who's winning?", "you are"),
        *[(f"command {i}", f"reply {i}") for i in range(RECENT_TURNS)],
    )
    digest = _digest_of(condense(messages))
    assert '"who\'s winning?"' in digest
    for move in ("e4", "Bxe6", "e2e4", "O-O"):
        assert f'"{move}"' not in digest


def test_a_history_of_nothing_but_moves_produces_no_digest_at_all():
    messages = _turns(
        *[("e4", "classic")] * 6,
        *[(f"command {i}", f"reply {i}") for i in range(RECENT_TURNS)],
    )
    # Those older turns hold nothing the state block doesn't already carry, so
    # the model gets the recent turns and no synthetic preamble.
    assert condense(messages) == messages[-2 * RECENT_TURNS :]


def test_the_request_list_is_capped_and_says_what_it_dropped():
    older = [(f"please do the thing number {i}", f"reply {i}") for i in range(30)]
    messages = _turns(
        *older,
        *[(f"command {i}", f"reply {i}") for i in range(RECENT_TURNS)],
    )
    digest = _digest_of(condense(messages))
    assert digest.count("\n- ") == DIGEST_MAX_REQUESTS
    dropped = len(older) - DIGEST_MAX_REQUESTS
    assert f"+{dropped}" in digest
    # Oldest dropped first: the newest older requests are the ones kept.
    assert "thing number 29" in digest
    assert "thing number 0" not in digest


def test_a_long_request_is_truncated_on_a_word_boundary():
    long_ask = "castle kingside and " + "then think about it " * 20
    messages = _turns(
        (long_ask, "sure"),
        *[(f"command {i}", f"reply {i}") for i in range(RECENT_TURNS)],
    )
    digest = _digest_of(condense(messages))
    line = next(li for li in digest.splitlines() if li.startswith("- "))
    assert len(line) < len(long_ask)
    assert line.startswith('- "castle kingside and')
    assert line.endswith('…"')
    kept = line.removeprefix('- "').removesuffix('…"')
    # Cut between words, not through one: every word kept is a whole word.
    assert long_ask.startswith(kept)
    assert not kept.endswith(" ")
    assert long_ask[len(kept)] == " "


def test_whitespace_in_a_request_is_collapsed_so_one_request_is_one_line():
    messages = _turns(
        ("undo that\nand play d4 instead", "bet"),
        *[(f"command {i}", f"reply {i}") for i in range(RECENT_TURNS)],
    )
    digest = _digest_of(condense(messages))
    assert '"undo that and play d4 instead"' in digest


def test_memory_is_the_condensed_view_of_the_raw_window():
    transcript = Transcript()
    transcript.record("who's winning?", "you are")
    for i in range(RECENT_TURNS):
        transcript.record(f"command {i}", f"reply {i}")
    assert transcript.memory() == condense(transcript.window(DEFAULT_WINDOW_TURNS))
    assert _digest_of(transcript.memory()).count('"who\'s winning?"') == 1


def test_memory_is_bounded_however_long_the_game_runs():
    # The old window handed over 40 messages of prose at this point. The
    # digest pair plus the recent turns is the ceiling, forever.
    transcript = Transcript()
    for i in range(200):
        transcript.record(f"question number {i}?", f"answer {i}")
    memory = transcript.memory()
    assert len(memory) == 2 + 2 * RECENT_TURNS
    assert len(transcript.window()) == 2 * DEFAULT_WINDOW_TURNS
