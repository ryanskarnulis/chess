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
    assert "self-hosted chess app" in SYSTEM_PROMPT


def test_composed_prompt_contains_the_global_personality_layer():
    # Distinctive line from the canonical agent-standard/personality-global.md
    # (generic house-wide Glitch tone, not chess-specific).
    assert "Whatever app you're working in right now" in SYSTEM_PROMPT


def test_composed_prompt_contains_the_chess_flavor_layer():
    # Distinctive to chess's own flavor block, not the vendored global text.
    assert "Trolling (occasional, earned" in SYSTEM_PROMPT


def test_composition_order_is_base_then_global_then_flavor():
    base_marker = SYSTEM_PROMPT.index("self-hosted chess app")
    global_marker = SYSTEM_PROMPT.index("Whatever app you're working in right now")
    flavor_marker = SYSTEM_PROMPT.index("Trolling (occasional, earned")
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
    assert "most turns: no troll" in system_prompt_for().lower()


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
    # the one honest beat. "Wins material off you" names the common trigger
    # explicitly — "beats you" alone read as game-over only.
    prompt = system_prompt_for().lower()
    assert "letting them cook" in prompt  # the cope
    assert "drop the act" in prompt  # the one honest beat
    assert "wins material" in prompt  # the trigger that actually happens


def test_prompt_caps_reaction_length():
    # The spec is underreaction; live reactions ran 3-4 sentences even under
    # "one or two sentences", so the default is now inverted: one short line
    # is the norm and two sentences the hard ceiling.
    prompt = system_prompt_for().lower()
    assert "one short line" in prompt
    assert "two sentences is the ceiling" in prompt


# --- hot-path move guidance (agent-reliability epic) --------------------------
#
# Voice games die when a spoken move doesn't land as a tool call. The base
# prompt must teach the speech→make_move mapping, anchor move strings to the
# provided legal_moves, explain the one-call-per-turn contract, and make
# destructive tools require confirmation. Pinned as data: substantive tokens,
# not exact wording.


def test_prompt_has_speech_to_tool_examples():
    prompt = system_prompt_for()
    # Few-shot mapping from spoken phrasing to make_move strings, including
    # castling and promotion syntax.
    assert "make_move" in prompt
    assert "pawn to e4" in prompt.lower()
    assert '"e4"' in prompt
    assert "O-O" in prompt
    assert "=Q" in prompt


def test_prompt_anchors_moves_to_provided_legal_moves():
    # The move string must come from the board state's legal_moves list,
    # never be invented.
    assert "legal_moves" in system_prompt_for()


def test_prompt_states_the_player_color_is_provided():
    assert "player_color" in system_prompt_for()


def test_prompt_explains_one_move_per_turn_and_engine_reply():
    prompt = system_prompt_for().lower()
    # One make_move per player turn; the engine answers inside the same call,
    # so the agent must never move for the engine's side.
    assert "once per player turn" in prompt
    assert "engine" in prompt
    assert "never" in prompt


def test_prompt_teaches_the_file_capture_form():
    # "d takes e5" is how players pronounce dxe5; without the example the
    # model has rejected the spoken form while offering the SAN back.
    prompt = system_prompt_for()
    assert "d takes e5" in prompt.lower()
    assert '"dxe5"' in prompt


def test_prompt_requires_acting_on_an_accepted_proposal():
    # Propose a move → player says yes → the agent must CALL make_move, not
    # announce the move in words (seen live: "moving forward with dxe5",
    # no tool call, no move).
    prompt = system_prompt_for().lower()
    assert "accept" in prompt
    assert "announc" in prompt


def test_prompt_requires_confirmation_before_destructive_tools():
    prompt = system_prompt_for()
    assert "resign" in prompt
    assert "new_game" in prompt
    assert "confirm" in prompt.lower()


def test_prompt_skips_new_game_confirmation_once_game_is_over():
    # A finished game has nothing left to lose: "new game" after checkmate
    # must start immediately, not trigger the destructive-tool confirmation
    # question (seen live: agent asks "are you sure?" after the game ended).
    prompt = system_prompt_for()
    assert "game_over" in prompt
    assert "without asking" in prompt.lower()


def test_prompt_routes_eval_questions_through_tools():
    # Live game: "who's winning?" was answered from vibes (wrongly), no
    # evaluate_position call. Judgment questions are reads like any other.
    prompt = system_prompt_for().lower()
    assert "who's winning" in prompt
    assert "evaluate_position" in prompt


def test_prompt_forbids_inventing_events():
    # Live game: the react step narrated a capture that never happened
    # ("you actually took my pawn" on a quiet knight move). Commentary must
    # stick to the moves the tools reported.
    assert "never invent" in system_prompt_for().lower()


def test_prompt_warns_about_mangled_voice_transcripts():
    prompt = system_prompt_for().lower()
    # Transcribed speech arrives mangled ("e 4", "night to f3"); the agent
    # should repair obvious slips instead of failing the move.
    assert "transcri" in prompt  # transcript / transcribed / transcription
    assert '"e 4"' in prompt
    assert "night" in prompt


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


# --- hints mode ---------------------------------------------------------------


def test_hints_off_leaves_the_prompt_unchanged():
    assert system_prompt_for(hints_mode=False) == SYSTEM_PROMPT


def test_hints_on_appends_a_hint_instruction():
    prompt = system_prompt_for(hints_mode=True)
    assert prompt.startswith(SYSTEM_PROMPT)
    assert "hint" in prompt[len(SYSTEM_PROMPT) :].lower()


def test_hints_and_verbosity_layer_together():
    prompt = system_prompt_for(verbosity="low", hints_mode=True)
    assert prompt.startswith(SYSTEM_PROMPT)
    assert "hint" in prompt.lower()
    assert prompt != system_prompt_for(verbosity="low")
