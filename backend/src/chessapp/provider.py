"""llama.cpp chat provider: the OpenAI wire format spoken over plain httpx.

llama.cpp's server speaks the OpenAI chat API, so a provider is just a POST to
`{base_url}/chat/completions` — no SDK. This module is the request half, the
one place chess says anything to llama.cpp; the `Brain` on top
(`llama_brain.py`) sees only `ChatResult`.

The response half is the fleet's, vendored verbatim in `agentkit/chat.py`
(canonical in `../agent-standard/agentkit/`; re-copy, never edit): the wire
models every body is validated against, `ChatResult`, and the typed errors
with the `ProviderFailure` each carries, which says whether asking again
could work. Its names are re-exported here, so callers import them from
`chessapp.provider` as they always did.

What the request says about the *model* — its sampling and how its thinking
channel is switched — comes from the model's profile (`profiles.py`, #375),
never from constants here. Thinking is toggled per request via the
non-standard `chat_template_kwargs` field, under the key the profile names
(default OFF, for fast tool calls); llama-server also takes `top_k` as a plain
body field. Both ride in the JSON payload directly — no SDK `extra_body`
indirection.

Correction retries on bad tool calls live in the `Brain`, not here (that is
where the retry budget and schema validation already are), which is why
`ToolCallArgumentsError` carries the tool name for the correction message.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

import httpx

from chessapp.agentkit.chat import (
    ChatResult,
    ProviderError,
    ProviderFailure,
    ProviderRequestError,
    ProviderResponseError,
    ServerStamp,
    ToolCall,
    ToolCallArgumentsError,
    Usage,
    WireCompletion,
    bad_status,
    body_text,
    chat_result,
    parse_completion,
    request_failed,
)
from chessapp.context_capture import CallStamp, ContextCapture, current_phase
from chessapp.profiles import ModelProfile, load_profile

__all__ = [
    "ChatProvider",
    "ChatResult",
    "LlamaCppProvider",
    "PhasedProvider",
    "ProviderError",
    "ProviderFailure",
    "ProviderRequestError",
    "ProviderResponseError",
    "ServerMeta",
    "ToolCall",
    "ToolCallArgumentsError",
    "Usage",
    "providers_for",
]

# What the server reported about the call it just served (#317). The fleet
# calls it a `ServerStamp`; chess's name for it stays, and `brain.ServerStamp`
# is the narrower thing the seam carries.
ServerMeta = ServerStamp

logger = logging.getLogger(__name__)

# One generous read timeout, not a special-cased first request: a cold model
# load through llama-swap is ~100s before the first byte, warm calls never get
# near it; the connect phase stays short so a dead server fails fast.
_READ_TIMEOUT = 300.0
_CONNECT_TIMEOUT = 10.0
# How long the context capture waits for llama-server to render a call's
# template (#359). Rendering is string work with no generation behind it, so
# this is a ceiling for a server that went away, never a wait anyone meets.
_TEMPLATE_TIMEOUT = 5.0


class ChatProvider(Protocol):
    """What the brain needs from a provider — matched by `LlamaCppProvider`.

    `tools` are OpenAI-style function definitions (the registry's
    `definitions()` output) passed straight through; the brain owns schema
    validation of the parsed arguments, so the provider only guarantees they
    are a JSON object.

    `temperature` is the one sampling knob a caller may set per request, for
    the planner/narrator split's per-phase sampling; `None` means the module
    default, so a caller that does not care sends exactly what it always did.

    `timeout` is the same shape of per-request override for how long this one
    call may take: a phase whose words the app will stop waiting for
    (`deadline.NARRATION_BUDGET_S`) has no use for a round trip that outlives the
    turn it was for, and hanging up is what frees the server's slot for the
    next one. `None` is the client's configured read timeout, which is what
    every phase that is worth waiting for still sends.
    """

    def chat(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        tools: Sequence[dict[str, Any]] | None = None,
        enable_thinking: bool = False,
        max_tokens: int | None = None,
        temperature: float | None = None,
        timeout: float | None = None,
        thinking_budget: int | None = None,
        effort: str | None = None,
    ) -> ChatResult: ...


class LlamaCppProvider:
    """Chat-completions client for the shared llama-server (behind llama-swap).

    Synchronous by design, matching the sync tool/API layer; FastAPI runs sync
    callers in worker threads. `client` is injectable for tests
    (`httpx.MockTransport`); otherwise the provider owns one.

    `capture`, when given, keeps every call's exact request and response bytes
    and the prompt the server's chat template rendered from them
    (`context_capture.py`, #359). `None` — the default — sends one request per
    call, exactly as before the knob existed.

    `seed`, when set, is sent as llama-server's sampling `seed` on every call
    that does not pass its own. Only the eval harness sets it: sample *j* of
    every arm of an A/B draws from the same random stream (common random
    numbers, #363), so a small change is judged on the pairs it flips rather
    than on two independent counts. The pairing is approximate — llama.cpp is
    not bit-reproducible across batch sizes or slots (ggml-org/llama.cpp#7052)
    — but never worse than unpaired. `None`, the default and what the app
    always uses, omits the field.

    `profile` is what the requests say about the model (`profiles.py`): the
    sampling sent on every call, so a drift in the shared server config never
    changes chess's behaviour, and the thinking toggle's key. `None` — the
    default — is `model`'s own profile, which is what every caller wants
    except a test pinning one.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        timeout_seconds: float = _READ_TIMEOUT,
        client: httpx.Client | None = None,
        capture: ContextCapture | None = None,
        seed: int | None = None,
        profile: ModelProfile | None = None,
    ) -> None:
        self.seed = seed
        self.profile = load_profile(model) if profile is None else profile
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._capture = capture
        # llama-swap serves the OpenAI routes under `/v1` and proxies a model's
        # own llama-server routes under `/upstream/<model>/` from the root —
        # the same split `serving.ServingProbe` reads `/props` through.
        root = self._base_url
        self._root = root[: -len("/v1")] if root.endswith("/v1") else root
        self._client = client or httpx.Client(
            timeout=httpx.Timeout(timeout_seconds, connect=_CONNECT_TIMEOUT)
        )

    @property
    def model(self) -> str:
        """The model id every request names."""
        return self._model

    @property
    def sampling(self) -> dict[str, float | int]:
        """The sampling every request sends unless a caller overrides it — the
        client half of a serving configuration, for the manifest (#317). Only
        what the profile sets: a key it leaves out is never sent."""
        return dict(self.profile.sampling)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> LlamaCppProvider:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def chat(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        tools: Sequence[dict[str, Any]] | None = None,
        enable_thinking: bool = False,
        max_tokens: int | None = None,
        temperature: float | None = None,
        timeout: float | None = None,
        thinking_budget: int | None = None,
        effort: str | None = None,
        cache_prompt: bool | None = None,
        seed: int | None = None,
    ) -> ChatResult:
        """One completion turn, optionally offering tools.

        `cache_prompt` is llama-server's per-request switch for reusing the
        slot's cached KV prefix (its default is on). It is not part of
        `ChatProvider` yet: the only caller is the planner probe
        (`scripts/probe_planner.py`), which measures whether prefix reuse on a
        long-lived server is what moves the knight-ask rate (#286). `None`
        omits the field, so every existing caller sends the bytes it always
        did; the knob joins the protocol only if the measurement says the app
        should send it.

        `seed` is the same kind of probe-only knob: the sampling seed for this
        one request, overriding the provider's own `seed`; `None` falls back to
        it, and with neither set the field is omitted.
        """
        payload = self._payload(
            messages,
            tools=tools,
            enable_thinking=enable_thinking,
            max_tokens=max_tokens,
            temperature=temperature,
            thinking_budget=thinking_budget,
            effort=effort,
            cache_prompt=cache_prompt,
            seed=self.seed if seed is None else seed,
        )
        return chat_result(self._post(payload, timeout))

    def _payload(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        tools: Sequence[dict[str, Any]] | None,
        enable_thinking: bool,
        max_tokens: int | None,
        temperature: float | None,
        thinking_budget: int | None = None,
        effort: str | None = None,
        cache_prompt: bool | None = None,
        seed: int | None = None,
    ) -> dict[str, Any]:
        sampling = self.profile.sampling
        payload: dict[str, Any] = {"model": self._model, "messages": list(messages)}
        # `None` resolves to the profile's temperature rather than omitting the
        # field, so a caller with no opinion sends the same bytes as ever; a
        # profile with no opinion either leaves it to the server.
        if temperature is None:
            temperature = sampling.get("temperature")
        if temperature is not None:
            payload["temperature"] = temperature
        # llama-server takes `top_k` (not OpenAI's) as a plain body field.
        for key in ("top_p", "top_k"):
            if key in sampling:
                payload[key] = sampling[key]
        if self.profile.thinking_kwarg is not None:
            kwargs: dict[str, Any] = {self.profile.thinking_kwarg: enable_thinking}
            if enable_thinking and effort is not None and self.profile.effort_kwarg:
                # The template's own thinking level (Qwen3.8's
                # `reasoning_effort`); omitted, its default applies.
                kwargs[self.profile.effort_kwarg] = effort
            payload["chat_template_kwargs"] = kwargs
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if thinking_budget is not None and enable_thinking:
            # llama-server's own field: past this many thinking tokens it
            # closes the thought and the model answers. Omitted means unbounded.
            payload["thinking_budget_tokens"] = thinking_budget
        if cache_prompt is not None:
            # llama-server's own field (not OpenAI's); omitted means its default.
            payload["cache_prompt"] = cache_prompt
        if seed is not None:
            # Also llama-server's; omitted means a fresh random stream per call.
            payload["seed"] = seed
        if tools:
            # Already OpenAI-format dicts (registry `definitions()`) — verbatim.
            payload["tools"] = list(tools)
            payload["tool_choice"] = "auto"
        return payload

    def _post(
        self, payload: dict[str, Any], timeout: float | None = None
    ) -> WireCompletion:
        # A per-request read ceiling for the callers that have one; the connect
        # phase stays as short as it is on the client, so a dead server still
        # fails fast either way. No ceiling means the client's own, which is
        # what every call that is worth waiting for sends.
        deadline = (
            httpx.USE_CLIENT_DEFAULT
            if timeout is None
            else httpx.Timeout(timeout, connect=_CONNECT_TIMEOUT)
        )
        # Built once and then sent, rather than `post(json=...)`: the same
        # bytes httpx always sent, and `request.content` is then literally what
        # went out — which is what the context capture keeps (#359).
        request = self._client.build_request(
            "POST",
            f"{self._base_url}/chat/completions",
            json=payload,
            timeout=deadline,
        )
        stamp = self._begin_capture()
        if stamp is None:
            return self._completion(self._send(request))
        started = time.monotonic()
        response: httpx.Response | None = None
        error: Exception | None = None
        try:
            response = self._send(request)
            return self._completion(response)
        except Exception as exc:
            error = exc
            raise
        finally:
            self._capture_call(stamp, request, response, error, started)

    def _send(self, request: httpx.Request) -> httpx.Response:
        try:
            return self._client.send(request)
        except httpx.HTTPError as exc:
            raise request_failed(exc) from exc

    @staticmethod
    def _completion(response: httpx.Response) -> WireCompletion:
        if response.status_code != 200:
            raise bad_status(response.status_code, response.text)
        return parse_completion(response.text)

    def _begin_capture(self) -> CallStamp | None:
        if self._capture is None:
            return None
        try:
            return self._capture.begin()
        except Exception:
            logger.warning("context_capture_failed", exc_info=True)
            return None

    def _capture_call(
        self,
        stamp: CallStamp,
        request: httpx.Request,
        response: httpx.Response | None,
        error: Exception | None,
        started: float,
    ) -> None:
        """Hand one finished call to the capture. Never raises: the call's own
        outcome — a result or its typed error — is what the turn gets."""
        assert self._capture is not None
        try:
            ms = round((time.monotonic() - started) * 1000)
            self._capture.record(
                stamp,
                {
                    "url": str(request.url),
                    "ms": ms,
                    "request": body_text(request.content),
                    "status_code": None if response is None else response.status_code,
                    "response": None
                    if response is None
                    else body_text(response.content),
                    "error": ""
                    if error is None
                    else f"{type(error).__name__}: {error}",
                    # A server that never answered cannot render either; asking
                    # would only hang a turn that is already failing.
                    "template": None
                    if response is None
                    else self._render_template(request.content),
                },
            )
        except Exception:
            logger.warning("context_capture_failed", exc_info=True)

    def _render_template(self, body: bytes) -> dict[str, str]:
        """The prompt string llama-server's chat template renders from `body`.

        Asked of the server rather than re-rendered here: the text the model
        tokenized is whatever *its* `--jinja` template makes of these messages
        and tools, and a copy of the template in this repo would be a second
        truth that drifts. The same bytes go to `/apply-template` as went to
        `/chat/completions`, so the rendering is of exactly that request.
        """
        try:
            response = self._client.post(
                f"{self._root}/upstream/{self._model}/apply-template",
                content=body,
                headers={"content-type": "application/json"},
                timeout=_TEMPLATE_TIMEOUT,
            )
            if response.status_code != 200:
                return {"error": f"HTTP {response.status_code}: {response.text[:500]}"}
            prompt = response.json().get("prompt")
            if not isinstance(prompt, str):
                return {"error": "apply-template answered without a prompt string"}
            return {"prompt": prompt}
        except Exception as exc:
            return {"error": f"{type(exc).__name__}: {exc}"}


class PhasedProvider:
    """One `ChatProvider` that sends each phase to its own model (#375).

    The planner, the narrator and the answer reader are separate jobs, and the
    best model for a parse need not be the best for words. Each call goes to
    the provider for the phase the brain named around it (`model_phase`, the
    same `ContextVar` the context capture reads), and to `default`'s when no
    phase is named, so the brain and every wrapper around it (the eval
    harness's `CountingProvider`) see one provider and stay as they are.

    `sampling` and `profile` are `default`'s — the planner's, by the factory's
    convention — so a single-model reader still finds what it expects;
    `phases` says what each phase talks to.
    """

    def __init__(
        self, routes: Mapping[str, LlamaCppProvider], default: LlamaCppProvider
    ) -> None:
        self._routes = dict(routes)
        self._default = default

    def provider_for(self, phase: str) -> LlamaCppProvider:
        return self._routes.get(phase, self._default)

    @property
    def model(self) -> str:
        return self._default.model

    @property
    def sampling(self) -> dict[str, float | int]:
        return self._default.sampling

    @property
    def profile(self) -> ModelProfile:
        return self._default.profile

    @property
    def phases(self) -> dict[str, dict[str, str]]:
        return {
            phase: {"model": provider.model, "profile": provider.profile.name}
            for phase, provider in self._routes.items()
        }

    def close(self) -> None:
        for provider in {id(p): p for p in self._all()}.values():
            provider.close()

    def _all(self) -> list[LlamaCppProvider]:
        return [self._default, *self._routes.values()]

    def chat(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        tools: Sequence[dict[str, Any]] | None = None,
        enable_thinking: bool = False,
        max_tokens: int | None = None,
        temperature: float | None = None,
        timeout: float | None = None,
        thinking_budget: int | None = None,
        effort: str | None = None,
    ) -> ChatResult:
        return self.provider_for(current_phase()).chat(
            messages,
            tools=tools,
            enable_thinking=enable_thinking,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=timeout,
            thinking_budget=thinking_budget,
            effort=effort,
        )


def providers_for(
    base_url: str,
    phase_models: Mapping[str, str],
    default_model: str,
    *,
    capture: ContextCapture | None = None,
    seed: int | None = None,
) -> LlamaCppProvider | PhasedProvider:
    """The provider for a phase-to-model map: a plain `LlamaCppProvider` when
    every phase names `default_model` — exactly the provider a single-model
    app always built — and a `PhasedProvider` otherwise, with one client per
    distinct model, each carrying that model's profile."""
    models = {default_model, *phase_models.values()}
    built = {
        model: LlamaCppProvider(base_url, model, capture=capture, seed=seed)
        for model in sorted(models)
    }
    default = built[default_model]
    if models == {default_model}:
        return default
    return PhasedProvider(
        {phase: built[model] for phase, model in phase_models.items()}, default
    )
