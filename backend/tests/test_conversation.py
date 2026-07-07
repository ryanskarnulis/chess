"""Transcript: the agent's bounded conversation memory.

One user command + the agent's final commentary = one turn. The window is
what a brain gets to see — final answers only, capped so KV-cache growth
stays bounded. Serialization rides inside the save file, so a resumed game
keeps its conversational thread.
"""

import pytest

from chessapp.conversation import DEFAULT_WINDOW_TURNS, Transcript


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
