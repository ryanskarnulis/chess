"""Model profiles: what the app knows about one model, kept out of the code (#375).

A profile is one TOML file in `data/profiles/`, named for the llama-swap model
id it describes. It holds what changes with the model and nothing else:

- `sampling` — the request's `temperature`, `top_p` and `top_k`. A key left
  out is a field the request leaves out, so the server's own default applies.
- `thinking_kwarg` — the `chat_template_kwargs` key that turns the model's
  thinking channel on and off, or absent for a model with no toggle (the
  request then carries no `chat_template_kwargs` at all).
- `[phases.<phase>]` — per-phase `temperature` (a phase without one samples at
  `sampling.temperature`) and `max_tokens`, for the phases in `PHASES`.
- `crutches` — the guidance that exists only because this model needed it
  (`KNOWN_CRUTCHES`; docs/model-profiles.md says what each one is and what it
  was measured to fix). A better model runs without them.
- `quirks` — standing measured facts about the model, as text. Nothing reads
  them but the serving manifest and the person changing the model.

A model with no file runs on `DEFAULT_PROFILE`: no sampling opinion, today's
token caps, the common thinking toggle, no crutches — and a warning, because
it is a model nobody has measured here yet.

The gemma-4-12b file is today's behaviour. Its sampling is the fleet's
(`../agent-standard/model-profile.md`, copied, never edited here), and
`tests/test_profile_bytes.py` pins that the default config still sends the
bytes it sent before profiles existed.
"""

from __future__ import annotations

import logging
import tomllib
from dataclasses import dataclass, field
from functools import cache
from importlib import resources
from types import MappingProxyType
from typing import Any

logger = logging.getLogger(__name__)

# The phases a profile can tune, by the names `brain.PHASE_*` give them.
PLANNER = "planner"
NARRATOR = "narrator"
ANSWER = "answer"
PHASES = (PLANNER, NARRATOR, ANSWER)

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

# What a profile is named when no file described the model.
DEFAULT_NAME = "default"

_SAMPLING_KEYS = ("temperature", "top_p", "top_k")
_TOP_LEVEL_KEYS = {"sampling", "thinking_kwarg", "phases", "crutches", "quirks"}


class ProfileError(ValueError):
    """A profile file that does not say what a profile must."""


@dataclass(frozen=True)
class PhaseSettings:
    """One phase's knobs. `temperature=None` samples at the profile's own."""

    max_tokens: int
    temperature: float | None = None


@dataclass(frozen=True)
class ModelProfile:
    name: str
    # Where the numbers came from, for the reader (a file path, or "built in").
    source: str
    # Only the keys the request sends; an absent key is the server's default.
    sampling: MappingProxyType[str, float | int] = field(
        default_factory=lambda: MappingProxyType({})
    )
    thinking_kwarg: str | None = "enable_thinking"
    phases: MappingProxyType[str, PhaseSettings] = field(
        default_factory=lambda: MappingProxyType({})
    )
    crutches: frozenset[str] = frozenset()
    quirks: tuple[str, ...] = ()

    def phase(self, name: str) -> PhaseSettings:
        return self.phases[name]

    def temperature_for(self, phase: str) -> float | None:
        """What `phase` samples at: its own temperature, else the profile's,
        else `None` — the server's."""
        own = self.phases[phase].temperature
        return own if own is not None else self.sampling.get("temperature")

    def describe(self) -> dict[str, Any]:
        """The profile as the serving manifest records it."""
        return {
            "name": self.name,
            "source": self.source,
            "sampling": dict(self.sampling),
            "thinking_kwarg": self.thinking_kwarg,
            "phases": {
                name: {
                    "max_tokens": settings.max_tokens,
                    "temperature": settings.temperature,
                }
                for name, settings in self.phases.items()
            },
            "crutches": sorted(self.crutches),
            "quirks": list(self.quirks),
        }


# The caps every phase needs whatever the model: without one llama-server
# runs n_predict -1 and a thought loop generates until the read timeout. The
# sizing is gemma-4-12b's (see its profile); a new model starts there.
_DEFAULT_PHASES = MappingProxyType(
    {
        PLANNER: PhaseSettings(max_tokens=2048),
        NARRATOR: PhaseSettings(max_tokens=4096),
        ANSWER: PhaseSettings(max_tokens=16),
    }
)

DEFAULT_PROFILE = ModelProfile(
    name=DEFAULT_NAME,
    source="built in",
    phases=_DEFAULT_PHASES,
)


def parse_profile(name: str, text: str, source: str) -> ModelProfile:
    """A profile from its TOML text. Strict: an unknown key, phase or crutch
    is an error, because a typo in a profile is a model quietly run on
    settings nobody chose."""
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise ProfileError(f"profile {name!r}: {exc}") from exc
    _only(name, "profile", data, _TOP_LEVEL_KEYS)
    sampling = data.get("sampling", {})
    _only(name, "sampling", sampling, set(_SAMPLING_KEYS))
    phases_data = data.get("phases", {})
    _only(name, "phases", phases_data, set(PHASES))
    phases = dict(_DEFAULT_PHASES)
    for phase, settings in phases_data.items():
        _only(name, f"phases.{phase}", settings, {"max_tokens", "temperature"})
        phases[phase] = PhaseSettings(
            max_tokens=int(settings.get("max_tokens", phases[phase].max_tokens)),
            temperature=settings.get("temperature"),
        )
    crutches = frozenset(data.get("crutches", ()))
    unknown = crutches - KNOWN_CRUTCHES
    if unknown:
        raise ProfileError(f"profile {name!r}: unknown crutches {sorted(unknown)}")
    thinking = data.get("thinking_kwarg")
    return ModelProfile(
        name=name,
        source=source,
        sampling=MappingProxyType(
            {key: sampling[key] for key in _SAMPLING_KEYS if key in sampling}
        ),
        thinking_kwarg=thinking if thinking else None,
        phases=MappingProxyType(phases),
        crutches=crutches,
        quirks=tuple(data.get("quirks", ())),
    )


def _only(name: str, where: str, table: Any, allowed: set[str]) -> None:
    if not isinstance(table, dict):
        raise ProfileError(f"profile {name!r}: {where} must be a table")
    extra = set(table) - allowed
    if extra:
        raise ProfileError(f"profile {name!r}: unknown {where} keys {sorted(extra)}")


@cache
def load_profile(model: str) -> ModelProfile:
    """The profile for `model`, or `DEFAULT_PROFILE` with a warning when no
    file describes it. Cached: a profile is read once per process."""
    path = resources.files("chessapp") / "data" / "profiles" / f"{model}.toml"
    if not path.is_file():
        logger.warning("no_model_profile model=%s using=%s", model, DEFAULT_NAME)
        return DEFAULT_PROFILE
    return parse_profile(
        model, path.read_text(encoding="utf-8"), f"data/profiles/{model}.toml"
    )
