"""The serving manifest: what actually served a run, as far as it is known (#317).

A trace record used to name its model by the alias the app asked for
(`gemma-4-12b`) and the URL it asked at. Neither proves anything about what
answered: the alias maps to whatever `../llama-swap/config.yaml` said when the
server last started, and two runs under the same alias can differ in weights,
quantization, context size, KV cache type, server build or sampling defaults —
every one of which can move an eval result. This module keeps the record that
can tell those runs apart, and says so plainly when it cannot.

A manifest has four parts:

- `app`: the code — the revision it was built from (`CHESSAPP_REVISION`, baked
  into the image at deploy; a git checkout's own HEAD otherwise) and the
  package version.
- `session`: this process — a fresh id per start, and the experiment it
  belongs to when a campaign names one (`CHESSAPP_EXPERIMENT`).
- `client`: what the app itself sends or enforces — the alias and URL, the
  per-phase sampling and generation ceilings, and every turn budget. Stated as
  fact, because the app owns it.
- `server`: what the server says it is running — the model file, its type, the
  server build, the context size, the slot count, its sampling defaults and the
  command line llama-swap started it with. Learned, never inferred: every field
  is `null` until the server has said it, and `source` says how far that got
  (`unprobed`, `props`, `unavailable`).

`manifest_id` is a short digest of everything but the session, so two runs
with one id served under one configuration, and two ids differ somewhere a
reader can diff. Turn records carry it; the manifest itself is written to the
trace as a `serving` record at startup and whenever it changes.

**How the server half is learned.** Not per move: one read of llama-server's
`/props`, taken off the request path once a call has come back — so the model
is already warm and the read cannot be what loads it — and again when a call's
build fingerprint changes. Behind llama-swap the read goes through
`/upstream/<model>/props`, and only after `/running` lists the model as ready:
through that route a request for an unloaded model *loads* it (a ~100 s cold
start on a shared card), which a diagnostic must never cause. Everything here is
best-effort and off the turn: a probe that fails leaves fields unknown, never a
turn slower or broken.
"""

import hashlib
import json
import logging
import os
import subprocess
import threading
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx

from chessapp import __version__
from chessapp.brain import ServerStamp
from chessapp.trace import TRACE_SCHEMA

logger = logging.getLogger(__name__)

KIND_SERVING = "serving"

SOURCE_UNPROBED = "unprobed"
SOURCE_PROPS = "props"
SOURCE_UNAVAILABLE = "unavailable"

# The server's sampling defaults worth a record: the ones a client that sends
# its own sampling may still inherit (min_p, the sampler chain, repetition,
# the seed) and the speculative decoder, which changes what is generated.
_SAMPLING_KEYS = (
    "temperature",
    "top_k",
    "top_p",
    "min_p",
    "typical_p",
    "repeat_penalty",
    "samplers",
    "seed",
    "speculative.types",
)

_SERVER_FIELDS = (
    "model_path",
    "model_alias",
    "model_ftype",
    "build_info",
    "n_ctx",
    "total_slots",
    "sampling",
    "cmd",
    "fingerprint",
)

# How long a probe waits on each request. Short: it runs on its own thread,
# but a server that takes longer than this to describe itself is not one whose
# description is worth holding a thread for.
_PROBE_TIMEOUT_S = 5.0
# How soon a probe that could not complete is tried again, on the next call
# that comes back — the model was not warm yet, or the server did not answer.
_RETRY_S = 60.0


def app_revision(env: Mapping[str, str] | None = None) -> str:
    """The code revision this process runs, or `"unknown"`.

    `CHESSAPP_REVISION` when set — the deployed image has it baked in at build,
    because the image carries no `.git`. Otherwise a git checkout's own HEAD,
    marked `-dirty` when tracked files differ from it: a run from an edited
    tree is not a run of that commit, and saying so is the point."""
    env = os.environ if env is None else env
    declared = env.get("CHESSAPP_REVISION", "").strip()
    if declared:
        return declared
    here = Path(__file__).resolve().parent
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=here,
            capture_output=True,
            text=True,
            timeout=2,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=here,
            capture_output=True,
            text=True,
            timeout=2,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    if not head:
        return "unknown"
    return f"{head}-dirty" if dirty else head


def _digest(value: Any) -> str:
    text = json.dumps(value, sort_keys=True, default=str)
    return hashlib.sha256(text.encode()).hexdigest()[:12]


class ServingManifest:
    """This process's serving configuration, as far as it is known.

    Thread-safe: the probe writes the server half from its own thread while
    requests read the label. `on_change` hears the whole record each time the
    manifest changes — assembly points it at the tracer — and a listener that
    raises is logged and ignored.
    """

    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        client: Mapping[str, Any],
        revision: str,
        experiment: str = "",
        session_id: str | None = None,
        on_change: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self._lock = threading.Lock()
        self._session = {
            "id": session_id or uuid4().hex[:12],
            "experiment": experiment,
        }
        self._app = {"revision": revision, "version": __version__}
        self._client = {"model": model, "base_url": base_url, **client}
        self._server: dict[str, Any] = dict.fromkeys(_SERVER_FIELDS)
        self._server["source"] = SOURCE_UNPROBED
        self.on_change = on_change

    def _id(self) -> str:
        return _digest(
            {"app": self._app, "client": self._client, "server": self._server}
        )

    def record(self) -> dict[str, Any]:
        """The manifest as the trace's `serving` record."""
        with self._lock:
            return {
                "schema": TRACE_SCHEMA,
                "kind": KIND_SERVING,
                "manifest_id": self._id(),
                "session": dict(self._session),
                "app": dict(self._app),
                "client": json.loads(json.dumps(self._client, default=str)),
                "server": dict(self._server),
            }

    def label(self) -> dict[str, str]:
        """What a turn record carries to point at this manifest."""
        with self._lock:
            return {
                "manifest_id": self._id(),
                "session": self._session["id"],
                "experiment": self._session["experiment"],
            }

    @property
    def server(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._server)

    def update_server(self, fields: Mapping[str, Any]) -> bool:
        """Merge what was learned about the server; tell `on_change` if that
        changed anything. Unknown field names are ignored rather than
        recorded — the server half says only what it was built to say."""
        with self._lock:
            before = dict(self._server)
            for key, value in fields.items():
                if key in self._server:
                    self._server[key] = value
            changed = self._server != before
        if changed:
            self.announce()
        return changed

    def announce(self) -> None:
        """Hand the current record to `on_change`, best-effort."""
        if self.on_change is None:
            return
        try:
            self.on_change(self.record())
        except Exception:
            logger.warning("serving_manifest_listener_failed", exc_info=True)


class ServingProbe:
    """Learns the manifest's server half from the server itself, off the turn.

    `observe` is the brain's `on_server` listener: it hears each completed
    call's stamp and decides whether a probe is due — the first call, a changed
    build fingerprint, or an earlier probe that could not complete and is past
    its retry interval. The probe itself runs on a daemon thread (`spawn`,
    injectable so tests run it inline) and never more than one at a time.
    """

    def __init__(
        self,
        manifest: ServingManifest,
        *,
        base_url: str,
        model: str,
        client: httpx.Client | None = None,
        clock: Callable[[], float] = time.monotonic,
        spawn: Callable[[Callable[[], None]], None] | None = None,
        retry_s: float = _RETRY_S,
    ) -> None:
        self._manifest = manifest
        root = base_url.rstrip("/")
        self._root = root[: -len("/v1")] if root.endswith("/v1") else root
        self._model = model
        self._client = client or httpx.Client(timeout=_PROBE_TIMEOUT_S)
        self._clock = clock
        self._spawn = spawn or _daemon
        self._retry_s = retry_s
        self._lock = threading.Lock()
        self._fingerprint: str | None = None
        self._inflight = False
        self._last_attempt: float | None = None

    def observe(self, stamp: ServerStamp) -> None:
        fingerprint = stamp.fingerprint
        with self._lock:
            changed = fingerprint is not None and fingerprint != self._fingerprint
            if changed:
                self._fingerprint = fingerprint
            settled = self._manifest.server["source"] == SOURCE_PROPS
            waited = (
                self._last_attempt is None
                or self._clock() - self._last_attempt >= self._retry_s
            )
            due = (changed or (not settled and waited)) and not self._inflight
            if due:
                self._inflight = True
                self._last_attempt = self._clock()
        if changed:
            self._manifest.update_server({"fingerprint": fingerprint})
        if due:
            self._spawn(self._run)

    def _run(self) -> None:
        try:
            self.probe()
        except Exception:
            logger.warning("serving_probe_failed", exc_info=True)
            self._manifest.update_server({"source": SOURCE_UNAVAILABLE})
        finally:
            with self._lock:
                self._inflight = False

    def probe(self) -> None:
        """One read of what the server is running, never one that loads it."""
        cmd: str | None = None
        running = self._get(f"{self._root}/running")
        if running is None:
            return
        if running.status_code == 404:
            # No llama-swap in front: a plain llama-server has one model and
            # serves its own props, loaded by definition.
            props_url = f"{self._root}/props"
        elif running.status_code == 200:
            entry = next(
                (
                    item
                    for item in _list(_json(running).get("running"))
                    if isinstance(item, dict) and item.get("model") == self._model
                ),
                None,
            )
            if entry is None or entry.get("state") != "ready":
                # Not warm (unloaded since the call, or still starting): asking
                # for its props would load it. Left unprobed, and tried again
                # on a later call.
                return
            cmd = entry.get("cmd") if isinstance(entry.get("cmd"), str) else None
            props_url = f"{self._root}/upstream/{self._model}/props"
        else:
            self._manifest.update_server({"source": SOURCE_UNAVAILABLE})
            return
        props = self._get(props_url)
        if props is None:
            return
        if props.status_code != 200:
            self._manifest.update_server({"source": SOURCE_UNAVAILABLE})
            return
        self._manifest.update_server({**read_props(_json(props)), "cmd": cmd})

    def _get(self, url: str) -> httpx.Response | None:
        try:
            return self._client.get(url, timeout=_PROBE_TIMEOUT_S)
        except httpx.HTTPError:
            self._manifest.update_server({"source": SOURCE_UNAVAILABLE})
            return None


def read_props(props: Mapping[str, Any]) -> dict[str, Any]:
    """The manifest's server fields out of a llama-server `/props` body. Each
    field is read on its own and left `null` when absent or mis-shaped: the
    body is the server's to change, and a field it stopped sending must read
    as unknown, not break the rest."""
    settings = props.get("default_generation_settings")
    settings = settings if isinstance(settings, dict) else {}
    params = settings.get("params")
    params = params if isinstance(params, dict) else {}
    n_ctx = settings.get("n_ctx", props.get("n_ctx"))
    return {
        "model_path": _scalar(props.get("model_path")),
        "model_alias": _scalar(props.get("model_alias")),
        "model_ftype": _scalar(props.get("model_ftype")),
        "build_info": _scalar(props.get("build_info")),
        "n_ctx": n_ctx if isinstance(n_ctx, int) else None,
        "total_slots": _scalar(props.get("total_slots")),
        "sampling": {key: params[key] for key in _SAMPLING_KEYS if key in params}
        or None,
        "source": SOURCE_PROPS,
    }


def _scalar(value: Any) -> str | int | None:
    return (
        value if isinstance(value, str | int) and not isinstance(value, bool) else None
    )


def _json(response: httpx.Response) -> dict[str, Any]:
    try:
        body = response.json()
    except ValueError:
        return {}
    return body if isinstance(body, dict) else {}


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _daemon(work: Callable[[], None]) -> None:
    threading.Thread(target=work, name="serving-probe", daemon=True).start()
