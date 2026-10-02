"""Model profiles (#375): loading, strictness, and what a profile sends.

The gemma-4-12b profile's *bytes* are pinned by `test_profile_bytes.py`; this
covers the profile machinery itself and the profiles that are not today's.
"""

import json
import logging
import re
from pathlib import Path

import httpx
import pytest

from chessapp.game import GameSession
from chessapp.llama_brain import create_llama_brain
from chessapp.profiles import (
    ANSWER,
    DEFAULT_NAME,
    DEFAULT_PROFILE,
    KNOWN_CRUTCHES,
    NARRATOR,
    PLANNER,
    ProfileError,
    load_profile,
    parse_profile,
)
from chessapp.provider import LlamaCppProvider
from chessapp.tools import ToolContext, build_registry

_FLEET_PROFILE = (
    Path(__file__).resolve().parents[3] / "agent-standard" / "model-profile.md"
)


# A row of the fleet profile's sampling table: | `top_k` | 64 (...) |
_FLEET_ROW = r"\|\s*`(temperature|top_p|top_k)`\s*\|\s*([\d.]+)"


def _sent(profile, **chat) -> dict:
    bodies: list[dict] = []

    def server(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]},
        )

    client = httpx.Client(transport=httpx.MockTransport(server))
    provider = LlamaCppProvider(
        "http://llm.test/v1", profile.name, client=client, profile=profile
    )
    provider.chat([{"role": "user", "content": "hi"}], **chat)
    return bodies[0]


def test_gemma_profile_holds_the_shipped_numbers():
    gemma = load_profile("gemma-4-12b")
    assert dict(gemma.sampling) == {"temperature": 1.0, "top_p": 0.95, "top_k": 64}
    assert gemma.thinking_kwarg == "enable_thinking"
    assert gemma.phase(PLANNER).temperature == 0.3
    assert [gemma.phase(p).max_tokens for p in (PLANNER, NARRATOR, ANSWER)] == [
        2048,
        4096,
        16,
    ]
    # Every crutch the 12B was measured to need (#375's scope comment).
    assert gemma.crutches == KNOWN_CRUTCHES
    assert any("minimize" in quirk for quirk in gemma.quirks)


@pytest.mark.skipif(
    not _FLEET_PROFILE.is_file(), reason="needs the ../agent-standard sibling"
)
def test_gemma_sampling_matches_the_fleet_profile():
    # Copied, never edited here: drift is fixed by re-copying.
    text = _FLEET_PROFILE.read_text()
    fleet = {key: float(value) for key, value in re.findall(_FLEET_ROW, text)}
    assert fleet == dict(load_profile("gemma-4-12b").sampling)


def test_a_model_with_no_profile_runs_on_the_default_and_says_so(caplog):
    load_profile.cache_clear()
    with caplog.at_level(logging.WARNING, logger="chessapp.profiles"):
        profile = load_profile("some-new-model")
    assert profile is DEFAULT_PROFILE
    assert profile.name == DEFAULT_NAME
    assert "some-new-model" in caplog.text
    assert not profile.crutches


def test_the_default_profile_leaves_sampling_to_the_server():
    body = _sent(DEFAULT_PROFILE)
    assert not {"temperature", "top_p", "top_k"} & set(body)
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    # A caller's own temperature still goes out.
    assert _sent(DEFAULT_PROFILE, temperature=0.5)["temperature"] == 0.5


def test_a_model_without_a_thinking_toggle_sends_no_template_kwargs():
    profile = parse_profile("plain", 'thinking_kwarg = ""\n', "test")
    assert profile.thinking_kwarg is None
    assert "chat_template_kwargs" not in _sent(profile, enable_thinking=True)


def test_a_profile_names_its_own_thinking_key():
    profile = parse_profile("other", 'thinking_kwarg = "thinking"\n', "test")
    assert _sent(profile, enable_thinking=True)["chat_template_kwargs"] == {
        "thinking": True
    }


def test_a_phase_left_out_keeps_the_default_caps():
    profile = parse_profile("p", "[phases.planner]\ntemperature = 0.2\n", "test")
    assert profile.phase(PLANNER).temperature == 0.2
    assert profile.phase(PLANNER).max_tokens == 2048
    assert profile.phase(NARRATOR) == DEFAULT_PROFILE.phase(NARRATOR)


@pytest.mark.parametrize(
    "text",
    [
        'crutches = ["undo_call_agian"]\n',  # a typo must not drop a crutch
        "[sampling]\ntemprature = 1.0\n",
        "[phases.summarizer]\nmax_tokens = 10\n",
        "[phases.planner]\nmax_token = 10\n",
        "extra = 1\n",
        "not toml = = 1\n",
    ],
)
def test_a_profile_that_says_something_unknown_does_not_load(text):
    with pytest.raises(ProfileError):
        parse_profile("bad", text, "test")


def test_the_factory_applies_the_profile_per_phase():
    profile = parse_profile(
        "tuned",
        "[phases.planner]\ntemperature = 0.1\nmax_tokens = 100\n"
        "[phases.narrator]\ntemperature = 0.9\nmax_tokens = 200\n"
        "[phases.answer]\nmax_tokens = 8\n",
        "test",
    )
    brain = create_llama_brain(
        base_url="http://llm.test/v1",
        model="tuned",
        dispatcher=build_registry(ToolContext(session=GameSession())),
        tool_definitions=[],
        provider=LlamaCppProvider("http://llm.test/v1", "tuned", profile=profile),
    )
    assert (brain.planner_temperature, brain.planner_max_tokens) == (0.1, 100)
    assert (brain.narrator_temperature, brain.narrator_max_tokens) == (0.9, 200)
    assert (brain.answer_temperature, brain.answer_max_tokens) == (None, 8)
    settings = brain.client_settings()
    assert settings["profile"]["name"] == "tuned"
    assert settings["answer_max_tokens"] == 8
    # An explicit planner temperature still wins over the profile's.
    override = create_llama_brain(
        base_url="http://llm.test/v1",
        model="tuned",
        dispatcher=build_registry(ToolContext(session=GameSession())),
        tool_definitions=[],
        planner_temperature=1.0,
        provider=LlamaCppProvider("http://llm.test/v1", "tuned", profile=profile),
    )
    assert override.planner_temperature == 1.0
