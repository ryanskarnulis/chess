"""Personality system prompts: the agent's selectable tone + behavioral contract.

Tested as data — never a live LLM. We pin that every selectable personality has
a prompt, that the prompts encode the non-negotiable contract every personality
inherits (act only through tools; ask instead of guessing when a command is
ambiguous), that the two tones actually differ, and that lookup falls back
safely for an unknown name.
"""

import pytest

from chessapp.personality import (
    DEFAULT_PERSONALITY,
    SYSTEM_PROMPTS,
    system_prompt_for,
)
from chessapp.tools import PERSONALITIES


def test_full_phase3_roster_is_selectable():
    # The complete personality roster from the brief: the Phase 1 pair plus
    # the Phase 3 additions. Every one must be selectable via set_personality.
    assert set(PERSONALITIES) == {
        "friendly_rival",
        "calm_coach",
        "trash_talker",
        "grandmaster",
        "villain",
        "silent_assassin",
        "beginner_bot",
        "streamer",
    }


def test_prompts_and_personality_names_stay_in_lockstep():
    # No personality selectable without a prompt, and no orphan prompt for a
    # name `set_personality` would reject.
    assert set(SYSTEM_PROMPTS) == set(PERSONALITIES)


def test_default_personality_is_a_real_one():
    assert DEFAULT_PERSONALITY in SYSTEM_PROMPTS


@pytest.mark.parametrize("name", PERSONALITIES)
def test_lookup_returns_that_personalitys_prompt(name):
    assert system_prompt_for(name) == SYSTEM_PROMPTS[name]
    assert system_prompt_for(name).strip()  # non-empty


def test_personalities_have_distinct_prompts():
    # Two personalities, two distinct prompts — the tone actually varies.
    assert len(set(SYSTEM_PROMPTS.values())) == len(SYSTEM_PROMPTS)


@pytest.mark.parametrize("name", PERSONALITIES)
def test_prompt_encodes_the_behavioral_contract(name):
    prompt = system_prompt_for(name).lower()
    # The two guarantees the brief pins on every personality: act only through
    # tools, and ask a clarifying question instead of guessing when ambiguous.
    assert "tool" in prompt
    assert "clarif" in prompt or "ambiguous" in prompt


def test_unknown_personality_falls_back_to_default():
    # `set_personality` is enum-guarded, but the lookup must never leave the
    # agent without a prompt.
    assert system_prompt_for("nonexistent") == SYSTEM_PROMPTS[DEFAULT_PERSONALITY]


# --- hot-path move guidance (agent-reliability epic) --------------------------
#
# Voice games die when a spoken move doesn't land as a tool call. The base
# prompt — inherited by every personality — must teach the speech→make_move
# mapping, anchor move strings to the provided legal_moves, explain the
# one-call-per-turn contract, and make destructive tools require confirmation.
# Pinned as data: substantive tokens, not exact wording.


@pytest.mark.parametrize("name", PERSONALITIES)
def test_prompt_has_speech_to_tool_examples(name):
    prompt = system_prompt_for(name)
    # Few-shot mapping from spoken phrasing to make_move strings, including
    # castling and promotion syntax.
    assert "make_move" in prompt
    assert "pawn to e4" in prompt.lower()
    assert '"e4"' in prompt
    assert "O-O" in prompt
    assert "=Q" in prompt


@pytest.mark.parametrize("name", PERSONALITIES)
def test_prompt_anchors_moves_to_provided_legal_moves(name):
    # The move string must come from the board state's legal_moves list,
    # never be invented.
    assert "legal_moves" in system_prompt_for(name)


@pytest.mark.parametrize("name", PERSONALITIES)
def test_prompt_states_the_player_color_is_provided(name):
    assert "player_color" in system_prompt_for(name)


@pytest.mark.parametrize("name", PERSONALITIES)
def test_prompt_explains_one_move_per_turn_and_engine_reply(name):
    prompt = system_prompt_for(name).lower()
    # One make_move per player turn; the engine answers inside the same call,
    # so the agent must never move for the engine's side.
    assert "once per player turn" in prompt
    assert "engine" in prompt
    assert "never" in prompt


@pytest.mark.parametrize("name", PERSONALITIES)
def test_prompt_teaches_the_file_capture_form(name):
    # "d takes e5" is how players pronounce dxe5; without the example the
    # model has rejected the spoken form while offering the SAN back.
    prompt = system_prompt_for(name)
    assert "d takes e5" in prompt.lower()
    assert '"dxe5"' in prompt


@pytest.mark.parametrize("name", PERSONALITIES)
def test_prompt_requires_acting_on_an_accepted_proposal(name):
    # Propose a move → player says yes → the agent must CALL make_move, not
    # announce the move in words (seen live: "moving forward with dxe5",
    # no tool call, no move).
    prompt = system_prompt_for(name).lower()
    assert "accept" in prompt
    assert "announc" in prompt


@pytest.mark.parametrize("name", PERSONALITIES)
def test_prompt_requires_confirmation_before_destructive_tools(name):
    prompt = system_prompt_for(name)
    assert "resign" in prompt
    assert "new_game" in prompt
    assert "confirm" in prompt.lower()


@pytest.mark.parametrize("name", PERSONALITIES)
def test_prompt_warns_about_mangled_voice_transcripts(name):
    prompt = system_prompt_for(name).lower()
    # Transcribed speech arrives mangled ("e 4", "night to f3"); the agent
    # should repair obvious slips instead of failing the move.
    assert "transcri" in prompt  # transcript / transcribed / transcription
    assert '"e 4"' in prompt
    assert "night" in prompt


# --- verbosity ("talk more / talk less") -------------------------------------


def test_normal_verbosity_leaves_the_prompt_unchanged():
    assert (
        system_prompt_for("calm_coach", verbosity="normal")
        == SYSTEM_PROMPTS["calm_coach"]
    )


@pytest.mark.parametrize("verbosity", ["low", "high"])
def test_low_and_high_verbosity_append_an_instruction(verbosity):
    base = SYSTEM_PROMPTS["calm_coach"]
    prompt = system_prompt_for("calm_coach", verbosity=verbosity)
    assert prompt.startswith(base)
    assert len(prompt) > len(base)


def test_low_and_high_verbosity_instructions_differ():
    low = system_prompt_for("calm_coach", verbosity="low")
    high = system_prompt_for("calm_coach", verbosity="high")
    assert low != high


def test_unknown_verbosity_falls_back_to_normal():
    assert (
        system_prompt_for("calm_coach", verbosity="shouting")
        == SYSTEM_PROMPTS["calm_coach"]
    )


# --- hints mode ---------------------------------------------------------------


def test_hints_off_leaves_the_prompt_unchanged():
    assert (
        system_prompt_for("calm_coach", hints_mode=False)
        == SYSTEM_PROMPTS["calm_coach"]
    )


def test_hints_on_appends_a_hint_instruction():
    base = SYSTEM_PROMPTS["calm_coach"]
    prompt = system_prompt_for("calm_coach", hints_mode=True)
    assert prompt.startswith(base)
    assert "hint" in prompt[len(base) :].lower()


def test_hints_and_verbosity_layer_together():
    prompt = system_prompt_for("calm_coach", verbosity="low", hints_mode=True)
    base = SYSTEM_PROMPTS["calm_coach"]
    assert prompt.startswith(base)
    assert "hint" in prompt.lower()
    assert prompt != system_prompt_for("calm_coach", verbosity="low")
