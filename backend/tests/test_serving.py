"""The serving manifest and its probe (#317).

The manifest is what tells two runs under the same model alias apart, so these
pin three things: what it records and how its id moves; that the server half is
learned from the server and left explicitly unknown otherwise; and that the
probe never asks a server for a model it has not loaded — through llama-swap
that request is a cold load, which a diagnostic must never cause.

`httpx.MockTransport` plays llama-swap and llama-server; the probe's thread is
replaced by running its work inline, so nothing here is timing-dependent.
"""

import json
import subprocess
from typing import Any

import httpx
from fastapi.testclient import TestClient

from chessapp.app import build_app
from chessapp.brain import ServerStamp
from chessapp.llama_brain import LlamaBrain
from chessapp.provider import ServerMeta
from chessapp.serving import (
    SOURCE_PROPS,
    SOURCE_UNAVAILABLE,
    SOURCE_UNPROBED,
    ServingManifest,
    ServingProbe,
    app_revision,
    read_props,
)
from chessapp.trace import JsonlTracer
from fakes import ScriptedProvider, text_turn

BASE_URL = "http://swap.test/v1"
MODEL = "gemma-4-12b"
MODEL_PATH = (
    "/root/.cache/huggingface/hub/models--unsloth--gemma-4-12B-it-qat-GGUF/"
    "snapshots/980b060c/gemma-4-12B-it-qat-UD-Q4_K_XL.gguf"
)
CMD = "/app/llama-server -hf unsloth/gemma-4-12B-it-qat-GGUF:UD-Q4_K_XL -c 131072"

PROPS = {
    "model_path": MODEL_PATH,
    "model_alias": "unsloth/gemma-4-12B-it-qat-GGUF:UD-Q4_K_XL",
    "model_ftype": "Q4_K - Medium",
    "build_info": "b9935-f2d1c2f39",
    "total_slots": 4,
    "chat_template": "{{ a very long template }}",
    "default_generation_settings": {
        "n_ctx": 131072,
        "params": {
            "temperature": 1.0,
            "top_k": 64,
            "top_p": 0.95,
            "min_p": 0.05,
            "seed": 4294967295,
            "speculative.types": ["draft-mtp"],
            "dry_base": 1.75,
        },
    },
}


def manifest(**overrides: Any) -> ServingManifest:
    fields: dict[str, Any] = {
        "model": MODEL,
        "base_url": BASE_URL,
        "client": {"planner_temperature": 0.3},
        "revision": "abc123",
        "session_id": "session-1",
    }
    fields.update(overrides)
    return ServingManifest(**fields)


class Swap:
    """A llama-swap double: `/running` says what is loaded, and the upstream
    route answers props — for a model it would otherwise have to *load*, which
    is what `requests` lets a test prove never happened."""

    def __init__(
        self,
        *,
        running: list[dict[str, Any]] | None = None,
        props: dict[str, Any] | None = None,
        running_status: int = 200,
        props_status: int = 200,
    ) -> None:
        self.running = (
            [{"model": MODEL, "state": "ready", "cmd": CMD}]
            if running is None
            else running
        )
        self.props = PROPS if props is None else props
        self.running_status = running_status
        self.props_status = props_status
        self.requests: list[str] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        self.requests.append(path)
        if path == "/running":
            if self.running_status != 200:
                return httpx.Response(self.running_status)
            return httpx.Response(200, json={"running": self.running})
        if path in (f"/upstream/{MODEL}/props", "/props"):
            return httpx.Response(self.props_status, json=self.props)
        return httpx.Response(404)

    def probe(self, target: ServingManifest, **kwargs: Any) -> ServingProbe:
        client = httpx.Client(transport=httpx.MockTransport(self.handler))
        return ServingProbe(
            target,
            base_url=BASE_URL,
            model=MODEL,
            client=client,
            spawn=lambda work: work(),
            **kwargs,
        )


FIRST = ServerStamp(fingerprint="b9935-f2d1c2f39", server_ms=200)


# --- the record --------------------------------------------------------------


def test_a_fresh_manifest_names_the_server_unknown_rather_than_the_alias():
    record = manifest().record()

    assert record["schema"] == 2
    assert record["kind"] == "serving"
    assert record["app"]["revision"] == "abc123"
    assert record["client"]["model"] == MODEL
    assert record["client"]["planner_temperature"] == 0.3
    assert record["session"] == {"id": "session-1", "experiment": ""}
    # Nothing about the server is inferred from the alias it was asked by.
    assert record["server"]["source"] == SOURCE_UNPROBED
    assert record["server"]["model_path"] is None
    assert record["server"]["build_info"] is None


def test_the_same_configuration_has_the_same_id_across_sessions():
    one = manifest(session_id="a", experiment="run-1")
    two = manifest(session_id="b", experiment="run-2")
    assert one.label()["manifest_id"] == two.label()["manifest_id"]
    assert one.label()["session"] == "a"
    assert one.label()["experiment"] == "run-1"


def test_the_same_alias_with_different_weights_has_a_different_id():
    one, two = manifest(), manifest()
    one.update_server(read_props(PROPS))
    two.update_server(read_props({**PROPS, "model_path": "/other/Q8_0.gguf"}))
    assert one.label()["manifest_id"] != two.label()["manifest_id"]


def test_a_different_client_setting_or_revision_has_a_different_id():
    base = manifest().label()["manifest_id"]
    assert manifest(client={"planner_temperature": 0.7}).label()["manifest_id"] != base
    assert manifest(revision="def456").label()["manifest_id"] != base


def test_a_change_is_announced_and_a_no_op_is_not():
    heard: list[dict[str, Any]] = []
    target = manifest(on_change=heard.append)

    assert target.update_server({"fingerprint": "b1"})
    assert not target.update_server({"fingerprint": "b1"})
    assert len(heard) == 1
    assert heard[0]["server"]["fingerprint"] == "b1"


def test_a_listener_that_raises_is_ignored():
    def broken(_record: dict[str, Any]) -> None:
        raise OSError("disk full")

    target = manifest(on_change=broken)
    assert target.update_server({"fingerprint": "b1"})


def test_props_are_read_field_by_field():
    fields = read_props(PROPS)
    assert fields["model_path"] == MODEL_PATH
    assert fields["build_info"] == "b9935-f2d1c2f39"
    assert fields["n_ctx"] == 131072
    assert fields["total_slots"] == 4
    assert fields["sampling"] == {
        "temperature": 1.0,
        "top_k": 64,
        "top_p": 0.95,
        "min_p": 0.05,
        "seed": 4294967295,
        "speculative.types": ["draft-mtp"],
    }
    assert fields["source"] == SOURCE_PROPS


def test_mis_shaped_props_leave_fields_unknown():
    fields = read_props({"model_path": 7.5, "default_generation_settings": "?"})
    assert fields["model_path"] is None
    assert fields["n_ctx"] is None
    assert fields["sampling"] is None


# --- the probe ---------------------------------------------------------------


def test_a_warm_model_is_probed_through_llama_swap():
    swap, target = Swap(), manifest()
    swap.probe(target).observe(FIRST)

    server = target.server
    assert server["source"] == SOURCE_PROPS
    assert server["model_path"] == MODEL_PATH
    assert server["cmd"] == CMD
    assert server["fingerprint"] == "b9935-f2d1c2f39"
    assert swap.requests == ["/running", f"/upstream/{MODEL}/props"]


def test_a_model_llama_swap_has_not_loaded_is_never_asked_for_its_props():
    """Through llama-swap, asking an unloaded model for its props loads it."""
    for running in ([], [{"model": MODEL, "state": "starting"}], [{"model": "other"}]):
        swap, target = Swap(running=running), manifest()
        swap.probe(target).observe(FIRST)

        assert swap.requests == ["/running"]
        assert target.server["source"] == SOURCE_UNPROBED
        assert target.server["model_path"] is None


def test_a_plain_llama_server_is_asked_directly():
    swap, target = Swap(running_status=404), manifest()
    swap.probe(target).observe(FIRST)

    assert swap.requests == ["/running", "/props"]
    assert target.server["source"] == SOURCE_PROPS
    assert target.server["cmd"] is None


def test_a_server_that_will_not_say_is_marked_unavailable():
    swap, target = Swap(props_status=500), manifest()
    swap.probe(target).observe(FIRST)
    assert target.server["source"] == SOURCE_UNAVAILABLE
    assert target.server["model_path"] is None


def test_an_unreachable_server_is_marked_unavailable():
    def dead(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    target = manifest()
    ServingProbe(
        target,
        base_url=BASE_URL,
        model=MODEL,
        client=httpx.Client(transport=httpx.MockTransport(dead)),
        spawn=lambda work: work(),
    ).observe(FIRST)
    assert target.server["source"] == SOURCE_UNAVAILABLE


def test_a_settled_probe_is_not_repeated_until_the_build_changes():
    swap, target = Swap(), manifest()
    probe = swap.probe(target)
    probe.observe(FIRST)
    probe.observe(FIRST)
    probe.observe(ServerStamp(fingerprint=None, server_ms=10))
    assert swap.requests.count("/running") == 1

    swap.props = {**PROPS, "build_info": "b10573-0000000"}
    probe.observe(ServerStamp(fingerprint="b10573-0000000"))
    assert swap.requests.count("/running") == 2
    assert target.server["build_info"] == "b10573-0000000"
    assert target.server["fingerprint"] == "b10573-0000000"


def test_an_unfinished_probe_is_retried_after_its_interval():
    now = [0.0]
    swap, target = Swap(running=[]), manifest()
    probe = swap.probe(target, clock=lambda: now[0], retry_s=60.0)

    probe.observe(FIRST)
    now[0] = 30.0
    probe.observe(FIRST)
    assert swap.requests.count("/running") == 1

    swap.running = [{"model": MODEL, "state": "ready", "cmd": CMD}]
    now[0] = 61.0
    probe.observe(FIRST)
    assert target.server["source"] == SOURCE_PROPS


def test_only_one_probe_runs_at_a_time():
    parked: list[Any] = []
    swap, target = Swap(), manifest()
    probe = ServingProbe(
        target,
        base_url=BASE_URL,
        model=MODEL,
        client=httpx.Client(transport=httpx.MockTransport(swap.handler)),
        spawn=parked.append,
    )
    probe.observe(FIRST)
    probe.observe(ServerStamp(fingerprint="b2"))
    assert len(parked) == 1
    parked[0]()
    assert target.server["source"] == SOURCE_PROPS


# --- the revision -------------------------------------------------------------


def test_a_declared_revision_wins():
    assert app_revision({"CHESSAPP_REVISION": " 1a2b3c "}) == "1a2b3c"


def test_without_one_the_checkout_names_itself_or_says_unknown(monkeypatch):
    revision = app_revision({})
    assert revision == "unknown" or len(revision.removesuffix("-dirty")) == 40

    def no_git(*_args: Any, **_kwargs: Any) -> None:
        raise FileNotFoundError("git")

    monkeypatch.setattr(subprocess, "run", no_git)
    assert app_revision({}) == "unknown"


# --- assembly -----------------------------------------------------------------


def test_the_app_writes_its_manifest_and_names_it_on_every_turn(tmp_path, monkeypatch):
    monkeypatch.setenv("CHESSAPP_EXPERIMENT", "exp-7")
    path = tmp_path / "turns.jsonl"
    app = build_app(
        provider=ScriptedProvider(text_turn("hi")),
        tracer=JsonlTracer(path),
    )
    TestClient(app).post("/api/command", json={"text": "how am I doing?"})

    serving, turn = (json.loads(line) for line in path.read_text().splitlines())
    assert serving["kind"] == "serving"
    assert serving["session"]["experiment"] == "exp-7"
    # An injected provider has no server to describe, so none is probed.
    assert serving["server"]["source"] == SOURCE_UNPROBED
    assert serving["client"]["planning_deadline_s"] == 60.0
    assert serving["client"]["reaction_budget_s"] == 10.0
    assert turn["kind"] == "turn"
    assert turn["serving"]["manifest_id"] == serving["manifest_id"]
    assert turn["serving"]["session"] == serving["session"]["id"]
    assert turn["serving"]["experiment"] == "exp-7"


def test_the_brain_tells_its_listener_what_the_server_said():
    heard: list[ServerStamp] = []
    provider = ScriptedProvider(text_turn("fine"))
    brain = LlamaBrain(
        provider=provider,
        dispatcher=None,  # type: ignore[arg-type]  # narrate dispatches nothing
        tool_definitions=[],
        system_prompt="persona",
        on_server=heard.append,
    )
    provider.rescript(
        text_turn("fine").model_copy(
            update={
                "server": ServerMeta(fingerprint="b1", server_ms=90, cached_tokens=3)
            }
        )
    )
    narration = brain.narrate({}, [])

    assert heard == [ServerStamp(fingerprint="b1", server_ms=90, cached_tokens=3)]
    assert narration.server == heard[0]
