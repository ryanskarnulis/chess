"""Prompt wiring: composition, layering, and the verbosity mechanics.

Tested as data — never a live LLM. Prompt *content* (tone, contract wording)
is deliberately not pinned here: behavior is gated by the eval suite and the
honesty/advice guards, and tone is tuned by ear. What stays pinned is the
machinery — the vendored global layer loads and composes in order, and the
verbosity layers actually do what the setting says.
"""

from pathlib import Path

import pytest

from chessapp.personality import SYSTEM_PROMPT, system_prompt_for


def test_default_lookup_returns_the_glitch_prompt():
    assert system_prompt_for() == SYSTEM_PROMPT
    assert SYSTEM_PROMPT.strip()  # non-empty


# --- layered composition (agent-standard Phase 2 slice 2) ---------------------
#
# The prompt is composed base -> global Glitch -> chess flavor -> dynamic
# layers (agent-standard/STANDARD.md §5). The global layer is a vendored
# verbatim copy of agent-standard/personality-global.md, never hand-edited
# here; drift is caught by agent-standard/check-sync.sh.


def test_vendored_global_personality_file_exists():
    vendored = (
        Path(__file__).parent.parent / "src" / "chessapp" / "personality-global.md"
    )
    assert vendored.is_file()
    assert vendored.read_text(encoding="utf-8").startswith("<!-- vendored")


def test_vendored_header_is_stripped_from_the_composed_prompt():
    assert "<!-- vendored" not in SYSTEM_PROMPT


def test_composed_prompt_contains_the_app_base_layer():
    # Distinctive to chess's own base prompt, not the global personality.
    assert "You are not the referee" in SYSTEM_PROMPT


def test_composed_prompt_contains_the_global_personality_layer():
    # Distinctive line from the canonical agent-standard/personality-global.md
    # (generic house-wide Glitch tone, not chess-specific).
    assert "Whatever app you're working in right now" in SYSTEM_PROMPT


def test_composed_prompt_contains_the_chess_flavor_layer():
    # Distinctive to chess's own flavor block, not the vendored global text.
    assert "salty-but-obvious line of cope" in SYSTEM_PROMPT


def test_composition_order_is_base_then_global_then_flavor():
    base_marker = SYSTEM_PROMPT.index("You are not the referee")
    global_marker = SYSTEM_PROMPT.index("Whatever app you're working in right now")
    flavor_marker = SYSTEM_PROMPT.index("salty-but-obvious line of cope")
    assert base_marker < global_marker < flavor_marker


# --- verbosity ("talk more / talk less") -------------------------------------


def test_normal_verbosity_leaves_the_prompt_unchanged():
    assert system_prompt_for(verbosity="normal") == SYSTEM_PROMPT


@pytest.mark.parametrize("verbosity", ["low", "high"])
def test_low_and_high_verbosity_append_an_instruction(verbosity):
    prompt = system_prompt_for(verbosity=verbosity)
    assert prompt.startswith(SYSTEM_PROMPT)
    assert len(prompt) > len(SYSTEM_PROMPT)


def test_low_and_high_verbosity_instructions_differ():
    assert system_prompt_for(verbosity="low") != system_prompt_for(verbosity="high")


def test_unknown_verbosity_falls_back_to_normal():
    # `set_verbosity` is enum-guarded, but the lookup must never leave the
    # agent without a valid prompt.
    assert system_prompt_for(verbosity="shouting") == SYSTEM_PROMPT
