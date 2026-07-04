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
