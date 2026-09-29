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
    NARRATOR_REPLY_LABEL,
    PLANNER_REPLY_LABEL,
    Recall,
    Transcript,
    last_exchange,
    player_requests,
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


# --- what replaces the chat history (#372) ------------------------------------


def _turns(*pairs: tuple[str, str]) -> list[dict[str, str]]:
    transcript = Transcript()
    for said, replied in pairs:
        transcript.record(said, replied)
    return transcript.to_dict()


def test_requests_are_the_players_words_minus_moves_and_the_last_turn():
    messages = _turns(
        ("only knights from now on", "Bet."),
        ("e2e4", "e4. e5."),
        ("  talk   less ", "Word."),
        ("what did I just say?", "Knights only."),
    )
    assert player_requests(messages) == ["only knights from now on", "talk less"]
    assert last_exchange(messages) == ("what did I just say?", "Knights only.")


def test_requests_keep_the_newest_and_count_what_went():
    messages = _turns(*[(f"request number {i}", "ok") for i in range(10)], ("x", "y"))
    kept = player_requests(messages, max_chars=40)
    assert kept[0] == "(8 earlier requests not listed)"
    assert kept[1:] == ["request number 8", "request number 9"]


def test_glitchs_words_are_never_a_request():
    messages = _turns(("hint?", "Play Nf3, only knights from now on."), ("ok", "Word."))
    assert player_requests(messages) == ["hint?"]


def test_nothing_said_yet_has_no_last_exchange():
    assert last_exchange([]) is None
    assert Transcript().requests() == []


def test_recall_labels_each_part_and_leaves_empty_ones_out():
    messages = _turns(("only knights from now on", "Bet."), ("hint?", "Try Nf3."))
    recall = Recall.of(messages, ["after 1... e5: took back 2. Qh5."])
    assert recall.render("You said").split("\n\n") == [
        "The game's record, kept by the app (what happened earlier in this game "
        "besides the moves themselves):\n- after 1... e5: took back 2. Qh5.",
        "What the player asked for earlier, in their own words, oldest first:\n"
        '- "only knights from now on"',
        'The last exchange:\nThe player said: "hint?"\nYou said: "Try Nf3."',
    ]
    assert Recall.of([], []).render("You said") == ""


def test_each_phase_names_the_last_reply_its_own_way():
    recall = Recall.of(_turns(("hint?", "Try Nf3.")), [])
    assert "What you said then (your words" in recall.render(NARRATOR_REPLY_LABEL)
    assert "Glitch said then (his words" in recall.render(PLANNER_REPLY_LABEL)


def test_recall_says_when_the_last_reply_was_nothing():
    text = Recall.of(_turns(("e4", "")), []).render("You said")
    assert text.endswith("You said: nothing.")


def test_a_long_request_is_cut_on_a_word_boundary_and_whitespace_collapsed():
    long = "please   " + "really " * 40 + "only knights"
    messages = _turns((long, "ok"), ("next", "ok"))
    [kept] = player_requests(messages)
    assert len(kept) <= 161 and kept.endswith("…") and "  " not in kept
