"""A model per phase (#375): the planner, the narrator and the answer reader
can each run on their own model and profile, and the default — every phase on
`LLAMACPP_MODEL` — is the single provider the app always built.
"""

import json
from typing import Any

import httpx

from chessapp.app import DEFAULT_MODEL, phase_models_from_env
from chessapp.game import GameSession
from chessapp.llama_brain import create_llama_brain
from chessapp.profiles import ANSWER, NARRATOR, PLANNER
from chessapp.provider import LlamaCppProvider, PhasedProvider, providers_for
from chessapp.serving import (
    SOURCE_PROPS,
    SOURCE_UNPROBED,
    THINKING_ABSENT,
    THINKING_OK,
    ServingManifest,
    probes_for,
    thinking_toggle,
)
from chessapp.tools import ToolContext, brain_tool_definitions, build_registry
from fakes import FakeEngine

BASE = "http://llm.test/v1"
GEMMA = "gemma-4-12b"
OTHER = "some-other-model"  # no profile file: the default profile


def _server(bodies: list[dict[str, Any]]):
    def handle(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        bodies.append(body)
        tools = "tools" in body
        message: dict[str, Any] = {"role": "assistant", "content": "Done."}
        if tools and not any(m["role"] == "tool" for m in body["messages"]):
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "c0",
                        "type": "function",
                        "function": {
                            "name": "make_move",
                            "arguments": json.dumps(
                                {"move": "e4", "source": "said_the_move"}
                            ),
                        },
                    }
                ],
            }
        elif not tools and "Their reply" in str(body["messages"]):
            message = {"role": "assistant", "content": "confirm"}
        return httpx.Response(
            200, json={"choices": [{"message": message, "finish_reason": "stop"}]}
        )

    return handle


def _split_brain(bodies: list[dict[str, Any]], phase_models: dict[str, str]):
    client = httpx.Client(transport=httpx.MockTransport(_server(bodies)))
    built = {
        model: LlamaCppProvider(BASE, model, client=client)
        for model in {GEMMA, *phase_models.values()}
    }
    provider = PhasedProvider(
        {phase: built[model] for phase, model in phase_models.items()}, built[GEMMA]
    )
    ctx = ToolContext(session=GameSession(), engine=FakeEngine())
    registry = build_registry(ctx)
    return create_llama_brain(
        base_url=BASE,
        model=GEMMA,
        dispatcher=registry,
        tool_definitions=lambda: brain_tool_definitions(registry, ctx),
        provider=provider,
        phase_models=phase_models,
    )


# --- configuration ----------------------------------------------------------


def test_every_phase_runs_on_llamacpp_model_unless_told_otherwise(monkeypatch):
    for name in (
        "LLAMACPP_MODEL",
        "CHESSAPP_PLANNER_MODEL",
        "CHESSAPP_NARRATOR_MODEL",
        "CHESSAPP_ANSWER_MODEL",
    ):
        monkeypatch.delenv(name, raising=False)
    assert phase_models_from_env() == dict.fromkeys(
        (PLANNER, NARRATOR, ANSWER), DEFAULT_MODEL
    )

    monkeypatch.setenv("LLAMACPP_MODEL", "base")
    monkeypatch.setenv("CHESSAPP_PLANNER_MODEL", "parser")
    assert phase_models_from_env() == {
        PLANNER: "parser",
        NARRATOR: "base",
        ANSWER: "base",
    }


def test_one_model_everywhere_is_the_plain_provider_it_always_was():
    one = providers_for(BASE, dict.fromkeys((PLANNER, NARRATOR), GEMMA), GEMMA)
    assert type(one) is LlamaCppProvider
    assert one.model == GEMMA


def test_a_split_builds_one_client_per_model():
    split = providers_for(BASE, {PLANNER: OTHER, NARRATOR: GEMMA, ANSWER: GEMMA}, GEMMA)
    assert isinstance(split, PhasedProvider)
    assert split.provider_for(PLANNER).model == OTHER
    assert split.provider_for(NARRATOR) is split.provider_for(ANSWER)
    assert split.phases[PLANNER] == {"model": OTHER, "profile": "default"}
    assert split.phases[NARRATOR] == {"model": GEMMA, "profile": GEMMA}


# --- each phase on its own model ---------------------------------------------


def test_each_call_goes_to_its_phases_model_with_that_models_profile():
    bodies: list[dict[str, Any]] = []
    brain = _split_brain(bodies, {PLANNER: GEMMA, NARRATOR: OTHER, ANSWER: GEMMA})
    board = {"fen": GameSession().fen(), "legal_moves": ["e4"]}

    brain.get_agent_response(board, "play e4")
    brain.read_answer("Resign?", "yes")

    planner = [b for b in bodies if "tools" in b]
    narrator = [b for b in bodies if "tools" not in b and b["max_tokens"] != 16]
    (answer,) = [b for b in bodies if b.get("max_tokens") == 16]
    assert planner and {b["model"] for b in planner} == {GEMMA}
    assert {b["temperature"] for b in planner} == {0.3}
    # The narrator's model has no profile: it sends no sampling of its own.
    assert narrator and {b["model"] for b in narrator} == {OTHER}
    assert not {"temperature", "top_p", "top_k"} & set(narrator[0])
    assert answer["model"] == GEMMA and answer["top_k"] == 64


def test_the_brain_reports_what_each_phase_runs_on():
    brain = _split_brain([], {PLANNER: OTHER, NARRATOR: GEMMA, ANSWER: GEMMA})
    settings = brain.client_settings()
    assert settings["phases"] == {
        PLANNER: {"model": OTHER, "profile": "default"},
        NARRATOR: {"model": GEMMA, "profile": GEMMA},
        ANSWER: {"model": GEMMA, "profile": GEMMA},
    }
    assert set(settings["profiles"]) == {"default", GEMMA}
    # A planner on the default profile samples at the server's temperature.
    assert brain.planner_temperature is None
    assert brain.serving_identity()["planner_model"] == OTHER
    assert "narrator_model" not in brain.serving_identity()


def test_the_default_wiring_names_no_phase_model():
    brain = _split_brain([], dict.fromkeys((PLANNER, NARRATOR, ANSWER), GEMMA))
    identity = brain.serving_identity()
    assert identity["model"] == GEMMA
    assert not [key for key in identity if key.endswith("_model")]


# --- the manifest and its probes ---------------------------------------------


_PROPS = {
    "model_path": "/m.gguf",
    "chat_template": "{% if enable_thinking %}<|think|>{% endif %}",
    "default_generation_settings": {"n_ctx": 4096, "params": {}},
}


def _swap(props_by_model: dict[str, dict[str, Any]]):
    def handle(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/running":
            return httpx.Response(
                200,
                json={
                    "running": [
                        {"model": model, "state": "ready", "cmd": f"serve {model}"}
                        for model in props_by_model
                    ]
                },
            )
        for model, props in props_by_model.items():
            if path == f"/upstream/{model}/props":
                return httpx.Response(200, json=props)
        return httpx.Response(404)

    return httpx.Client(transport=httpx.MockTransport(handle))


def _manifest(other_models=()) -> ServingManifest:
    return ServingManifest(
        model=GEMMA,
        base_url=BASE,
        client={},
        revision="abc",
        session_id="s",
        other_models=other_models,
    )


def test_a_single_model_manifest_has_no_other_servers():
    assert "servers" not in _manifest().record()
    assert "servers" not in _manifest(other_models=[GEMMA]).record()


def test_each_phase_model_gets_its_own_server_half():
    manifest = _manifest(other_models=[OTHER])
    assert manifest.record()["servers"][OTHER]["source"] == SOURCE_UNPROBED
    client = _swap({GEMMA: _PROPS, OTHER: {**_PROPS, "chat_template": "plain"}})
    probes = probes_for(
        manifest,
        base_url=BASE,
        model=GEMMA,
        phase_models={PLANNER: OTHER, NARRATOR: GEMMA},
    )
    assert len(probes) == 2
    for probe in probes:
        probe._client = client
        probe.probe()
    record = manifest.record()
    assert record["server"]["source"] == SOURCE_PROPS
    assert record["server"]["cmd"] == f"serve {GEMMA}"
    assert record["server"]["thinking_toggle"] == THINKING_OK
    assert record["servers"][OTHER]["cmd"] == f"serve {OTHER}"
    # The default profile looks for `enable_thinking`; this template ignores it.
    assert record["servers"][OTHER]["thinking_toggle"] == THINKING_ABSENT


def test_the_thinking_check_says_unknown_when_it_cannot_tell():
    assert thinking_toggle({"chat_template": "x enable_thinking"}, "enable_thinking")
    assert thinking_toggle({}, "enable_thinking") is None
    assert thinking_toggle({"chat_template": "anything"}, None) is None
