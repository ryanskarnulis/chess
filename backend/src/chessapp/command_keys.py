"""Idempotency keys for the panel's command route (#553).

The page keeps a conductor handoff in the URL until its command has run, and
retries it after a 30 s timeout or a reload. If the first request did reach
the backend, the retry would run the same words again: a move attempted twice.
So `/api/command` takes the delegate route's optional ``Idempotency-Key``
header (`agent_api`, #291) and remembers what each key's command did:

- **running**: a retry waits for the first run and answers with its result.
  Never a second run.
- **done**: a retry gets the first run's words and tool results back.
- **incomplete**: the run started and never finished (it raised, or the app
  restarted mid-run). Whatever it did is on the board, so a retry is told so
  rather than run.

A command that ran nothing (a stale 409, a dead provider before anything ran)
is forgotten, so its retry runs as a first attempt. A key is bound to the words
and origin it was first sent with; anything else under it is a caller bug.

Bounded: at most `MAX_KEYS` keys, none older than `KEY_TTL_SECONDS`, oldest
dropped first. Persisted beside the live game when there is a save dir, so a
retry after a restart is still recognised; a run cut off by the restart reads
back as incomplete. Best-effort both ways, like `conversations.json`.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from chessapp.tools import _write_json_atomic

logger = logging.getLogger(__name__)

COMMAND_KEYS_FILENAME = "command_keys.json"
_FORMAT = 1
# One key per handoff intent: a handful a day, so this is weeks of them.
MAX_KEYS = 64
# Long enough for a tab left on a failed handoff and reloaded later the same
# day; past it the key is forgotten and its words run as new.
KEY_TTL_SECONDS = 24 * 60 * 60

Status = Literal["running", "done", "incomplete"]


@dataclass
class KeyedCommand:
    """What one key's command did. `reply` is the stored part of the response
    (words, tool results, correlation id) once it is done; the board is read
    fresh on every replay."""

    text: str
    origin: str
    created_at: float
    status: Status = "running"
    reply: dict[str, Any] | None = None
    # Set when the run settles (done, incomplete or forgotten), so a retry
    # that arrived mid-run can wait for it.
    settled: asyncio.Event = field(default_factory=asyncio.Event, repr=False)


class CommandKeys:
    def __init__(
        self,
        path: Path | None = None,
        *,
        max_keys: int = MAX_KEYS,
        ttl_seconds: float = KEY_TTL_SECONDS,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._entries: OrderedDict[str, KeyedCommand] = OrderedDict()
        self._path = path
        self._max_keys = max_keys
        self._ttl = ttl_seconds
        self._clock = clock
        if path is not None:
            self._load(path)

    def get(self, key: str) -> KeyedCommand | None:
        self._prune()
        return self._entries.get(key)

    def start(self, key: str, text: str, origin: str) -> KeyedCommand:
        """Claim `key` for a run that is about to start. The caller has just
        seen `get` return None, with no await in between."""
        entry = KeyedCommand(text=text, origin=origin, created_at=self._clock())
        self._entries[key] = entry
        self._prune()
        self._save()
        return entry

    def finish(self, key: str, entry: KeyedCommand, reply: dict[str, Any]) -> None:
        entry.status = "done"
        entry.reply = reply
        entry.settled.set()
        self._save()

    def abandon(self, key: str, entry: KeyedCommand) -> None:
        """The run started and did not finish: it may have changed the board."""
        entry.status = "incomplete"
        entry.settled.set()
        self._save()

    def forget(self, key: str, entry: KeyedCommand) -> None:
        """The run did nothing, so the key is free for a retry to run it."""
        if self._entries.get(key) is entry:
            del self._entries[key]
        entry.settled.set()
        self._save()

    def _prune(self) -> None:
        """Drop expired keys, then the oldest past the cap. A running key is
        never dropped: its retry must keep waiting on it, not start a run."""
        cutoff = self._clock() - self._ttl
        settled = [k for k, e in self._entries.items() if e.status != "running"]
        for key in settled:
            if self._entries[key].created_at < cutoff:
                del self._entries[key]
        for key in [k for k in settled if k in self._entries]:
            if len(self._entries) <= self._max_keys:
                break
            del self._entries[key]

    # --- the file -------------------------------------------------------------

    def _save(self) -> None:
        if self._path is None:
            return
        document = {
            "format": _FORMAT,
            "keys": [
                {
                    "key": key,
                    "text": e.text,
                    "origin": e.origin,
                    "created_at": e.created_at,
                    "status": e.status,
                    "reply": e.reply,
                }
                for key, e in self._entries.items()
            ],
        }
        try:
            _write_json_atomic(self._path, document)
        except OSError:
            logger.warning("could not persist command keys to %s", self._path)

    def _load(self, path: Path) -> None:
        try:
            data = json.loads(path.read_text())
        except FileNotFoundError:
            return
        except (OSError, ValueError):
            logger.warning("ignoring unreadable command keys %s", path)
            return
        try:
            if not isinstance(data, dict) or data.get("format") != _FORMAT:
                raise ValueError("not a command key store")
            entries: OrderedDict[str, KeyedCommand] = OrderedDict()
            for raw in data["keys"]:
                status = raw["status"]
                if status not in ("running", "done", "incomplete"):
                    raise ValueError(f"unknown status {status!r}")
                reply = raw["reply"]
                if status == "done" and not _is_reply(reply):
                    raise ValueError("a done key without its reply")
                entry = KeyedCommand(
                    text=str(raw["text"]),
                    origin=str(raw["origin"]),
                    created_at=float(raw["created_at"]),
                    # A run the restart cut off did whatever it did: incomplete.
                    status="incomplete" if status == "running" else status,
                    reply=reply if status == "done" else None,
                )
                entry.settled.set()
                entries[str(raw["key"])] = entry
        except (AttributeError, KeyError, TypeError, ValueError):
            logger.warning("ignoring invalid command keys %s", path, exc_info=True)
            return
        self._entries = entries
        self._prune()


def _is_reply(value: Any) -> bool:
    """What a replay reads off a stored reply, each the type it is sent as."""
    return (
        isinstance(value, dict)
        and isinstance(value.get("commentary"), str)
        and isinstance(value.get("tool_results"), list)
        and isinstance(value.get("correlation_id"), str)
    )
