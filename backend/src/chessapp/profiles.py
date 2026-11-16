"""Chess's model profiles: its phases, crutches and files (#375).

The profile format and the strict loader are the fleet's, vendored verbatim
in `agentkit/model_profile.py` (canonical in `../agent-standard/agentkit/`;
re-copy, never edit). What is chess's lives here:

- the phases (`PHASES`), their default caps, and the keys each one takes
  (`SPEC`). The planner alone takes `thinking`: whether it reasons before
  every call, not only once an analysis tool has answered (#298), and
  `effort` (#490). The narrator alone takes `analysis_thinking` (#440). The
  mover alone (#472) takes `thinking`, `effort`, `deadline_s`,
  `thinking_budget` (#476), `board_facts` (#497) and `look_calls` (#498).
- the crutch names (`KNOWN_CRUTCHES`; docs/model-profiles.md says what each
  one is and what it was measured to fix).
- the brains a player may pick (`BRAIN_CHOICES`).
- where the files are (`data/profiles/`, one per llama-swap model id), the
  cache, and the warning for a model with no file: it runs on
  `DEFAULT_PROFILE`, because it is a model nobody has measured here yet.

The gemma-4-12b file is today's behaviour. Its sampling is the fleet's
(`../agent-standard/model-profile.md`, copied, never edited here), and
`tests/test_profile_bytes.py` pins that the default config still sends the
bytes it sent before profiles existed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import cache
from importlib import resources

from chessapp.agentkit import model_profile as _kit
from chessapp.agentkit.model_profile import (
    DEFAULT_NAME,
    ModelProfile,
    ProfileError,
    ProfileSpec,
)

__all__ = [
    "ANSWER",
    "ASK_PLAYER_EXAMPLES",
    "BRAIN_CHOICES",
    "DEFAULT_NAME",
    "DEFAULT_PROFILE",
    "DIFFICULTY_CONSTRAINT_RULE",
    "KNOWN_CRUTCHES",
    "MOVER",
    "MOVE_SOURCE_REQUIRED",
    "NARRATOR",
    "PHASES",
    "PICK_RESUBMIT_SCRIPT",
    "PLANNER",
    "SPEC",
    "UNDO_CALL_AGAIN",
    "ModelProfile",
    "PhaseSettings",
    "ProfileError",
    "load_profile",
    "parse_profile",
]

logger = logging.getLogger(__name__)

# The phases a profile can tune, by the names `brain.PHASE_*` give them.
PLANNER = "planner"
NARRATOR = "narrator"
ANSWER = "answer"
# Glitch's own move, in the tier where the model picks it
# (docs/glitch-difficulty.md, #472).
MOVER = "mover"
PHASES = (PLANNER, NARRATOR, ANSWER, MOVER)

# The 12B's crutches (2026-09-29 scope comment on #375). Named here so a
# profile that misspells one fails to load rather than silently running
# without it.
MOVE_SOURCE_REQUIRED = "move_source_required"
UNDO_CALL_AGAIN = "undo_call_again"
ASK_PLAYER_EXAMPLES = "ask_player_examples"
PICK_RESUBMIT_SCRIPT = "pick_resubmit_script"
DIFFICULTY_CONSTRAINT_RULE = "difficulty_constraint_rule"
KNOWN_CRUTCHES = frozenset(
    {
        MOVE_SOURCE_REQUIRED,
        UNDO_CALL_AGAIN,
        ASK_PLAYER_EXAMPLES,
        PICK_RESUBMIT_SCRIPT,
        DIFFICULTY_CONSTRAINT_RULE,
    }
)

# The brains a player may pick for Glitch (#434), as llama-swap ids, each
# with a profile and a pinned request fixture. Code-owned: nothing outside
# the list can be chosen. qwen38-27b, the slow "Deep" brain, joined as an
# option without a gate run (#436).
BRAIN_CHOICES = ("gemma-4-12b", "gemma-4-26b-a4b", "qwen38-27b")


@dataclass(frozen=True)
class PhaseSettings(_kit.PhaseSettings):
    """A phase's knobs, with chess's own beside the fleet's.
    `analysis_thinking` is the narrator's: the brain's rule turns its thinking
    on once an analysis tool has answered, and a profile can switch that off.
    `board_facts` is the mover's (#497): whether its request carries the board
    facts (`board_facts.py`, docs/glitch-sight.md) between the board and the
    legal moves, and so is `look_calls` (#498): how many `try_line` calls one
    run may make, `0` leaving the tool out of the request."""

    analysis_thinking: bool = True
    board_facts: bool = False
    look_calls: int = 0


SPEC = ProfileSpec(
    # The caps every phase needs whatever the model: without one llama-server
    # runs n_predict -1 and a thought loop generates until the read timeout.
    # The sizing is gemma-4-12b's (see its profile); a new model starts there.
    phases={
        PLANNER: PhaseSettings(max_tokens=2048),
        NARRATOR: PhaseSettings(max_tokens=4096),
        ANSWER: PhaseSettings(max_tokens=16),
        # A thinking move choice: about 70 s at the 12B's and the 26B's
        # ~60 tok/s, inside the deadline. Starting values, not measured (#472).
        MOVER: PhaseSettings(max_tokens=4096, thinking=True, deadline_s=90.0),
    },
    # Beyond `max_tokens`. The order is the order `describe()` lists them in,
    # which the serving manifest and the trend scripts key on.
    phase_keys={
        PLANNER: ("temperature", "thinking", "effort"),
        NARRATOR: ("temperature", "analysis_thinking"),
        ANSWER: ("temperature",),
        MOVER: (
            "temperature",
            "thinking",
            "effort",
            "deadline_s",
            "thinking_budget",
            "board_facts",
            "look_calls",
        ),
    },
    known_crutches=KNOWN_CRUTCHES,
)

DEFAULT_PROFILE = SPEC.default_profile()


def parse_profile(name: str, text: str, source: str) -> ModelProfile:
    """A profile from its TOML text, strict (`ProfileSpec.parse`)."""
    return SPEC.parse(name, text, source)


@cache
def load_profile(model: str) -> ModelProfile:
    """The profile for `model`, or `DEFAULT_PROFILE` with a warning when no
    file describes it. Cached: a profile is read once per process."""
    directory = resources.files("chessapp") / "data" / "profiles"
    profile = SPEC.load(directory, model, "data/profiles")
    if profile is None:
        logger.warning("no_model_profile model=%s using=%s", model, DEFAULT_NAME)
        return DEFAULT_PROFILE
    return profile
