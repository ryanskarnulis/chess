"""Glitch's system prompt: the agent's tone and behavioral contract.

Tested as data — never a live LLM. There is exactly one personality (Glitch,
decided 2026-07: the selectable roster was collapsed to one dialed-in
character). We pin that the prompt encodes the non-negotiable contract (act
only through tools; ask instead of guessing when a command is ambiguous), the
hot-path move guidance, and the character beats that were chosen by hand —
low-key trolling, help that stays real, cope-then-props when beaten, and
explicit permission to swear (the serving model is safety-tuned and won't
without it).
"""

from pathlib import Path

import pytest

from chessapp.personality import (
    PLANNER_PROMPT,
    SYSTEM_PROMPT,
    system_prompt_for,
)


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


def test_prompt_encodes_the_behavioral_contract():
    prompt = system_prompt_for().lower()
    # The two guarantees the brief pins on the personality: act only through
    # tools, and ask a clarifying question instead of guessing when ambiguous.
    assert "tool" in prompt
    assert "clarif" in prompt or "ambiguous" in prompt


# --- character: Glitch --------------------------------------------------------
#
# The tone block was dialed in by hand (2026-07-11); these pin its load-bearing
# beats as substantive tokens, not exact wording, so the block can be reworded
# without losing a trait.


def test_prompt_names_glitch():
    assert "glitch" in system_prompt_for().lower()


def test_prompt_authorizes_swearing():
    # Gemma is safety-tuned; without explicit permission it softens every
    # curse to "dang". The authorization is load-bearing — and the first live
    # game (2026-07-11) showed permission alone isn't enough: eight earned
    # moments, zero swears. The prompt must also forbid the softening itself.
    prompt = system_prompt_for().lower()
    assert "swear" in prompt
    assert "censor" in prompt


def test_prompt_keeps_the_trolling_low_key():
    prompt = system_prompt_for().lower()
    assert "troll" in prompt
    assert "never explain a joke" in prompt


def test_prompt_makes_trolling_occasional_not_constant():
    # Ryan's live feedback (2026-07-11): "the personality is still weird…
    # more chill". The troll-move list read as a bit to perform every turn;
    # the default turn is now quiet.
    assert "mostly you just play" in system_prompt_for().lower()


def test_prompt_carries_ryans_lexicon():
    # The slang whitelist came from Ryan directly (2026-07-11). Quoted forms
    # so prose words don't false-match.
    prompt = system_prompt_for()
    for term in (
        '"word"',
        '"bet"',
        '"ight"',
        '"for sure"',  # acknowledgment
        '"clean"',
        '"nasty"',
        '"filthy"',
        '"sheesh"',
        '"goes hard"',  # props
        '"cooked"',  # someone losing
        '"fr"',
        '"deadass"',  # emphasis
        '"bro"',
        '"dude"',
        '"man"',  # address
    ):
        assert term in prompt, f"missing lexicon term {term}"


def test_prompt_keeps_help_real():
    # The jokes ride on top of real competence: asked for help, Glitch gives
    # genuinely good help (plus one jab), never sandbags the player.
    prompt = system_prompt_for().lower()
    assert "help is always real" in prompt
    assert "never troll the player into worse chess" in prompt


def test_prompt_gives_props_then_cope_when_beaten():
    # Live game: forking and winning the queen earned confusion instead of
    # the one honest beat. The condensed flavor keeps all three beats without
    # the scripted lines: props (the honest beat), cope, and "material" named
    # as a trigger ("beats you" alone read as game-over only).
    prompt = system_prompt_for().lower()
    assert "props" in prompt  # the one honest beat
    assert "cope" in prompt  # the cope
    assert "material" in prompt  # the trigger that actually happens


def test_prompt_caps_reaction_length():
    # The spec is underreaction; live reactions ran 3-4 sentences even under
    # "one or two sentences", so the default is now inverted: one short line
    # is the norm and two sentences the hard ceiling.
    prompt = system_prompt_for().lower()
    assert "one short line" in prompt
    assert "two sentences is the ceiling" in prompt


# Hot-path move guidance and the destructive-op confirm dance used to live in
# the base prompt; they now live on the tools themselves — make_move,
# new_game, resign, evaluate_position, analyze_last_move — and are tested in
# test_tools.py against the tool descriptions the model actually reads (a rule
# belongs with the capability it governs). The base prompt keeps only the
# non-negotiable contract below.


def test_prompt_forbids_claiming_the_players_move():
    # Live game: Glitch narrated the player's capture as his own ("I took your
    # knight" for a piece the player had just taken off him). The tools state the
    # attribution now, but the narrator's contract must say it too: a move it
    # carried out for the player belongs to the player.
    prompt = system_prompt_for().lower()
    assert "the player's move, not yours" in prompt
    assert "opponent" in prompt


def test_prompt_gives_the_narrator_no_moves_of_its_own():
    # The first cut of the rule above added "Only the engine's replies are your
    # own moves" — ownership the narrator acted on: reacting mid-turn, before
    # Stockfish had answered, every reaction in the 2026-07-28 game announced a
    # reply it had invented ("i'll go with d6", then Bb4 was played). The rule
    # is purely negative; nothing in the prompt assigns the narrator moves.
    prompt = system_prompt_for().lower()
    assert "your own moves" not in prompt


def test_prompt_forbids_inventing_events():
    # Live game: the react step narrated a capture that never happened
    # ("you actually took my pawn" on a quiet knight move). Commentary must
    # stick to the moves the tools reported.
    assert "never invent" in system_prompt_for().lower()


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


# --- the planner prompt (the other half of the planner/narrator split) --------
#
# Two model phases, two prompts (docs/planner-narrator.md). The planner picks
# tools and never speaks; the narrator above speaks and holds no tools. These
# pin what the planner prompt must and must not contain: the whole point of the
# split is that no persona token competes with the tool decision, so a token
# leaking back in is a regression, not a style choice.


def test_planner_prompt_carries_no_persona():
    # The diagnosis the split exists to fix: on a 12B, tone instructions
    # compete with tool selection for attention. None of Glitch reaches here.
    prompt = PLANNER_PROMPT.lower()
    for token in ("glitch", "troll", "swear", "cope", "props", "bro"):
        assert token not in prompt, f"persona token leaked into the planner: {token}"


def test_planner_prompt_carries_no_verbosity_layer():
    # Verbosity shapes what the player is *told*, and the planner tells them
    # nothing — so it takes no verbosity layer at any setting.
    assert "talk less" not in PLANNER_PROMPT.lower()


def test_planner_prompt_is_compact():
    # Compactness is the mechanism, not a nicety: measurably smaller than the
    # personality prompt the same decision used to be made under.
    assert len(PLANNER_PROMPT) < len(SYSTEM_PROMPT) / 2


def test_planner_prompt_keeps_the_load_bearing_rules():
    # Persona-free, not contract-free. Everything the base layer guaranteed
    # about *acting* has to survive the compaction.
    prompt = PLANNER_PROMPT.lower()
    assert "referee" in prompt  # the board and engine own legality
    assert "tool" in prompt  # act only through tools
    assert "legal_moves" in prompt  # map phrasing onto the injected list
    assert "never invent" in prompt  # and never onto something else
    assert "ambiguous" in prompt  # ask rather than guess
    assert "no tool" in prompt  # by declining to call anything


def test_planner_prompt_states_the_handoff_contract():
    # Its closing text is an internal note, not commentary: a separate voice
    # writes what the player reads.
    prompt = PLANNER_PROMPT.lower()
    assert "separate voice" in prompt
    assert "never address the player" in prompt


def test_planner_prompt_teaches_the_advice_path():
    # Hints are on-request (hints mode was retired 2026-09-01): an ask for
    # advice routes to `get_best_moves` on every turn, so the line that used to
    # arrive with hints-on is part of the standing contract now.
    assert "get_best_moves" in PLANNER_PROMPT


# --- hints retirement ----------------------------------------------------------


def test_the_narrator_prompt_never_tells_glitch_to_volunteer_hints():
    # With hints mode retired there is no state in which Glitch is told to
    # volunteer suggestions: a hint exists only as an answer to an ask, and the
    # advice guard holds him to moves a tool actually reported.
    for verbosity in ("low", "normal", "high"):
        assert "volunteer" not in system_prompt_for(verbosity).lower()
