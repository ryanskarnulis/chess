"""Delegate REST API: conversation persistence + the messages round trip.

Implements the workspace delegate contract (`../agent-standard/delegate-api.md`)
so a conductor agent can drive chess over HTTP, mounted under `/api/agent/*`.
The wire models mirror PCC's `app/schemas/conversations.py` field-for-field so
the two apps speak one byte-compatible contract; the router mirrors PCC's
`app/api/routes_agent.py`. The one model-calling endpoint
(`POST …/messages`) reuses chess's command pipeline (`chessapp.api`) against
the single shared game session, so a conductor-played move runs the same
fast-parse → brain → dispatch → retry → react path a web command does, and
still broadcasts to the live board.

Two deliberate divergences from PCC (chess is leaner and in-memory by design):

- **A JSON file, not a DB** (:class:`ConversationStore`). With a save dir the
  store is rewritten atomically to ``conversations.json`` after every change
  and reloaded at startup (#291), so a thread survives a restart the way the
  live game does; without one it is process-lifetime, which the contract's
  404-then-recreate-once rule already allows for. Soft delete is a
  `deleted_at` mark; a soft-deleted thread 404s indistinguishably from one
  that never existed. Ids are monotonic ints, and stay so across a restart.
- **An optional ``Idempotency-Key``** on ``POST …/messages`` (#291): a retry
  of an exchange that finished returns the stored exchange instead of
  running the pipeline again, so a conductor that timed out — or talked to
  an app that restarted — can retry without acting twice.
- **Actor attribution is a log line, not an audit row.** Chess has no
  `activity_events` table (unlike PCC), so the resolved `X-Agent-Actor` is
  bound into a structured log line for the run and kept off the wire —
  `MessageRead` stays byte-compatible with PCC.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time
from collections import defaultdict, deque
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Protocol

from fastapi import APIRouter, Depends, Header, HTTPException, Request, status
from pydantic import BaseModel, ConfigDict, StringConstraints

from chessapp.conversation import DEFAULT_WINDOW_TURNS, condense
from chessapp.provider import ProviderError
from chessapp.tools import _write_json_atomic, delegate_origin

if TYPE_CHECKING:
    from chessapp.api import CommandOutcome

logger = logging.getLogger(__name__)

# Cap on one chat turn: generous for typed input while bounding what a run
# feeds the local model's context window (matches PCC's MAX_AGENT_MESSAGE_LENGTH).
MAX_AGENT_MESSAGE_LENGTH = 8_000

# An `Idempotency-Key` is the caller's name for one exchange; longer than this
# is not a key but a payload.
MAX_IDEMPOTENCY_KEY_LENGTH = 128

# Auto-titles from the first user message are cut here, on a word boundary.
MAX_DERIVED_TITLE_LENGTH = 60

# Default per-IP messages/minute cap (PCC's `agent_messages_per_min` default);
# `CHESSAPP_AGENT_MESSAGES_PER_MIN` overrides it, read live per request.
DEFAULT_AGENT_MESSAGES_PER_MIN = 10

# The audit actor a run is stamped with by default, and the delegate actors a
# trusted caller (conductor) may bind via ``X-Agent-Actor`` in its place. An
# absent or unrecognized header falls back to the default rather than erroring,
# so a caller can never stamp an arbitrary identity into the log.
LOOP_ACTOR = "agent:loop"
DELEGATE_ACTORS = frozenset({"agent:conductor"})


def resolve_actor(header_value: str | None) -> str:
    """The audit actor for a run given an ``X-Agent-Actor`` header, if any.

    Returns the header value only when it names a recognized delegate actor
    (``agent:conductor``); anything else falls back to :data:`LOOP_ACTOR`.

    A *label*, not authentication (#291): any caller that can reach the port
    can send the header, and nothing here checks who sent it. What it buys is
    an honest audit line from a trusted caller on a trusted network. Before
    anything grants access on an actor, the external gateway has to be
    verified to authenticate callers and strip a forged header on the way in.
    """
    if header_value is not None and header_value in DELEGATE_ACTORS:
        return header_value
    return LOOP_ACTOR


# --- wire models (field names mirror PCC's schemas exactly) -------------------

AgentMessageText = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True, min_length=1, max_length=MAX_AGENT_MESSAGE_LENGTH
    ),
]
NonBlankStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class ConversationCreate(BaseModel):
    # Optional: an untitled conversation is titled from its first user message.
    title: NonBlankStr | None = None


class ConversationRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    title: str | None
    created_at: datetime
    # Touched on every appended message — the conversation list's recency order.
    updated_at: datetime


class MessageCreate(BaseModel):
    content: AgentMessageText
    # The delegate half of the board-version precondition (audit item 7). A
    # conductor drives the *same* session the web board does, so it gets the
    # same opt-in guarantee at its own entry point: supply the `state.version`
    # you last saw and a turn that would land on a board somebody else has
    # already moved is refused 409 instead of played. Omitted is today's
    # behavior — the field is additive, and PCC's schema has no counterpart, so
    # a caller that never sends it stays byte-compatible.
    version: int | None = None
    # The same opt-in for *which game* (#291): the `state.game_id` the
    # conductor is playing. A turn sent after the board moved on to another
    # game — a new game, a resume — is refused 409 rather than played there.
    game_id: str | None = None


class ToolCallRead(BaseModel):
    """One dispatched tool call as persisted on an assistant message. Exactly
    one of ``result`` / ``error`` is set."""

    tool: str
    arguments: dict[str, Any]
    result: str | None = None
    error: str | None = None


class MessageRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    conversation_id: int
    role: str
    content: str | None
    tool_calls: list[ToolCallRead] | None = None
    stop_reason: str | None = None
    created_at: datetime


class ConversationDetail(ConversationRead):
    """A conversation with its full message history, oldest first."""

    messages: list[MessageRead]


class MessageExchange(BaseModel):
    """What one ``POST …/messages`` produced: the stored user turn and the
    assistant turn the pipeline answered it with."""

    user_message: MessageRead
    assistant_message: MessageRead


# --- in-memory store ----------------------------------------------------------


def _utcnow() -> datetime:
    return datetime.now(UTC)


@dataclass
class StoredMessage:
    id: int
    conversation_id: int
    role: str
    content: str | None
    tool_calls: list[dict[str, Any]] | None
    stop_reason: str | None
    created_at: datetime
    # What the loop is replayed instead of `content`, when the two differ:
    # the app substituted its own words for the model's and the caller must
    # see the correction while the model must not be taught to write it
    # (`api.CommandOutcome.memory`). `None` means "no divergence", which is
    # every turn but a guarded one. Never on the wire — `MessageRead` names
    # its fields, and this is not one of them.
    memory: str | None = None
    # The caller's `Idempotency-Key` for the exchange this user turn opened
    # (#291). On the user turn, because that is committed before the pipeline
    # runs: a key whose turn has no answer after it names an exchange that
    # never finished. Never on the wire.
    idempotency_key: str | None = None


@dataclass
class StoredConversation:
    id: int
    title: str | None
    created_at: datetime
    updated_at: datetime
    deleted_at: datetime | None = None
    messages: list[StoredMessage] = field(default_factory=list)


CONVERSATIONS_FILENAME = "conversations.json"
_STORE_FORMAT = 1


class ConversationStore:
    """The delegate conversation store (chess's documented divergence from
    PCC's SQLite). Threads and their immutable user/assistant turns live in a
    dict keyed by monotonic int id; a soft delete is a ``deleted_at`` mark, so
    :meth:`get` returns ``None`` for a missing *or* soft-deleted id alike.

    With a `path` the whole store is written there after every change — one
    atomic replace, so a crash leaves the previous whole document — and read
    back on construction (#291): threads, their turns, soft deletes, the id
    counters and every idempotency key survive a restart. Best-effort both
    ways, the settings file's contract: an unreadable file is an empty store
    and a warning, and a failed write costs a warning, never the exchange.
    Without a `path` it is process-lifetime, exactly as before.
    """

    def __init__(self, path: Path | None = None) -> None:
        self._conversations: dict[int, StoredConversation] = {}
        self._next_conversation_id = 1
        self._next_message_id = 1
        self._path = path
        if path is not None:
            self._load(path)

    def create(self, *, title: str | None = None) -> StoredConversation:
        now = _utcnow()
        conversation = StoredConversation(
            id=self._next_conversation_id, title=title, created_at=now, updated_at=now
        )
        self._next_conversation_id += 1
        self._conversations[conversation.id] = conversation
        self._save()
        return conversation

    def get(self, conversation_id: int) -> StoredConversation | None:
        """The active conversation, or ``None`` if unknown or soft-deleted."""
        conversation = self._conversations.get(conversation_id)
        if conversation is None or conversation.deleted_at is not None:
            return None
        return conversation

    def list_active(self) -> list[StoredConversation]:
        """Active conversations, most recently touched first."""
        active = [c for c in self._conversations.values() if c.deleted_at is None]
        return sorted(active, key=lambda c: (c.updated_at, c.id), reverse=True)

    def soft_delete(self, conversation: StoredConversation) -> None:
        conversation.deleted_at = _utcnow()
        self._save()

    def append_user_message(
        self,
        conversation: StoredConversation,
        content: str,
        idempotency_key: str | None = None,
    ) -> StoredMessage:
        """Store one user turn; an untitled conversation is titled from it."""
        if conversation.title is None:
            conversation.title = _derive_title(content)
        return self._append(
            conversation,
            "user",
            content,
            None,
            None,
            idempotency_key=idempotency_key,
        )

    def find_exchange(
        self, conversation: StoredConversation, idempotency_key: str
    ) -> tuple[StoredMessage, StoredMessage | None] | None:
        """The exchange a key opened in this thread: its user turn and the
        answer to it, or None for the answer when the exchange never finished.
        None when the key was never used here.

        The answer is the turn right after: exchanges in one thread are
        serialized, so nothing can land between a question and its answer.
        """
        messages = conversation.messages
        for index, message in enumerate(messages):
            if message.role == "user" and message.idempotency_key == idempotency_key:
                following = messages[index + 1] if index + 1 < len(messages) else None
                if following is not None and following.role == "assistant":
                    return message, following
                return message, None
        return None

    def append_assistant_message(
        self,
        conversation: StoredConversation,
        content: str | None,
        tool_calls: list[dict[str, Any]] | None,
        stop_reason: str | None,
        memory: str | None = None,
    ) -> StoredMessage:
        return self._append(
            conversation, "assistant", content, tool_calls, stop_reason, memory
        )

    def _append(
        self,
        conversation: StoredConversation,
        role: str,
        content: str | None,
        tool_calls: list[dict[str, Any]] | None,
        stop_reason: str | None,
        memory: str | None = None,
        *,
        idempotency_key: str | None = None,
    ) -> StoredMessage:
        message = StoredMessage(
            id=self._next_message_id,
            conversation_id=conversation.id,
            role=role,
            content=content,
            tool_calls=tool_calls,
            stop_reason=stop_reason,
            created_at=_utcnow(),
            memory=memory,
            idempotency_key=idempotency_key,
        )
        self._next_message_id += 1
        conversation.messages.append(message)
        conversation.updated_at = message.created_at
        self._save()
        return message

    # --- the file ---------------------------------------------------------------

    def _save(self) -> None:
        if self._path is None:
            return
        document = {
            "format": _STORE_FORMAT,
            "next_conversation_id": self._next_conversation_id,
            "next_message_id": self._next_message_id,
            "conversations": [
                _conversation_to_dict(c) for c in self._conversations.values()
            ],
        }
        try:
            _write_json_atomic(self._path, document)
        except OSError:
            logger.warning("could not persist conversations to %s", self._path)

    def _load(self, path: Path) -> None:
        try:
            data = json.loads(path.read_text())
        except FileNotFoundError:
            return
        except (OSError, ValueError):
            logger.warning("ignoring unreadable conversation store %s", path)
            return
        try:
            if not isinstance(data, dict) or data.get("format") != _STORE_FORMAT:
                raise ValueError("not a conversation store")
            conversations = {
                c.id: c for c in map(_conversation_from_dict, data["conversations"])
            }
            next_conversation = int(data["next_conversation_id"])
            next_message = int(data["next_message_id"])
        except (AttributeError, KeyError, TypeError, ValueError):
            logger.warning(
                "ignoring invalid conversation store %s", path, exc_info=True
            )
            return
        # The counters are taken as at least one past every id on record, so a
        # hand-edited or truncated file can never hand out an id twice.
        message_ids = [m.id for c in conversations.values() for m in c.messages]
        self._conversations = conversations
        self._next_conversation_id = max(
            [next_conversation, *(i + 1 for i in conversations)]
        )
        self._next_message_id = max([next_message, *(i + 1 for i in message_ids)])

    def history_for_loop(
        self, conversation: StoredConversation
    ) -> list[dict[str, str]]:
        """Prior turns as chat messages for the pipeline's ``transcript``.

        Text turns only — persisted tool trajectories are for display/audit and
        are deliberately never round-tripped into model context. Condensed by
        the same policy the web panel uses (`conversation.condense`) so the two
        entry points have one memory, not two: recent turns verbatim behind a
        digest of what the caller asked for earlier. The store keeps everything;
        only the model's view is reduced.

        A turn whose `memory` diverges from its `content` replays the former,
        for the same reason the panel's does: the caller is owed the correction
        the app made, and the model must not be handed a first-person apology as
        something it said.
        """
        text_turns = [
            {"role": m.role, "content": m.content if m.memory is None else m.memory}
            for m in conversation.messages
            if m.content is not None
        ]
        return condense(text_turns[-2 * DEFAULT_WINDOW_TURNS :])


def _message_to_dict(message: StoredMessage) -> dict[str, Any]:
    return {
        "id": message.id,
        "role": message.role,
        "content": message.content,
        "tool_calls": message.tool_calls,
        "stop_reason": message.stop_reason,
        "created_at": message.created_at.isoformat(),
        "memory": message.memory,
        "idempotency_key": message.idempotency_key,
    }


def _conversation_to_dict(conversation: StoredConversation) -> dict[str, Any]:
    return {
        "id": conversation.id,
        "title": conversation.title,
        "created_at": conversation.created_at.isoformat(),
        "updated_at": conversation.updated_at.isoformat(),
        "deleted_at": (
            conversation.deleted_at.isoformat()
            if conversation.deleted_at is not None
            else None
        ),
        "messages": [_message_to_dict(m) for m in conversation.messages],
    }


def _optional_str(value: Any, name: str) -> str | None:
    if value is not None and not isinstance(value, str):
        raise ValueError(f"{name} must be a string: {value!r}")
    return value


def _conversation_from_dict(data: Any) -> StoredConversation:
    """One stored thread, validated on the way in: roles are the two the loop
    replays, so a tampered file cannot inject a system turn into a model's
    context (the same rule `Transcript.from_dict` holds)."""
    conversation_id = data["id"]
    if type(conversation_id) is not int:
        raise ValueError(f"conversation id must be an int: {conversation_id!r}")
    messages = []
    for raw in data["messages"]:
        role = raw["role"]
        if role not in ("user", "assistant"):
            raise ValueError(f"stored message has invalid role: {role!r}")
        message_id = raw["id"]
        if type(message_id) is not int:
            raise ValueError(f"message id must be an int: {message_id!r}")
        tool_calls = raw.get("tool_calls")
        if tool_calls is not None and not isinstance(tool_calls, list):
            raise ValueError("stored tool_calls must be a list")
        messages.append(
            StoredMessage(
                id=message_id,
                conversation_id=conversation_id,
                role=role,
                content=_optional_str(raw.get("content"), "content"),
                tool_calls=tool_calls,
                stop_reason=_optional_str(raw.get("stop_reason"), "stop_reason"),
                created_at=datetime.fromisoformat(raw["created_at"]),
                memory=_optional_str(raw.get("memory"), "memory"),
                idempotency_key=_optional_str(
                    raw.get("idempotency_key"), "idempotency_key"
                ),
            )
        )
    deleted_at = data.get("deleted_at")
    return StoredConversation(
        id=conversation_id,
        title=_optional_str(data.get("title"), "title"),
        created_at=datetime.fromisoformat(data["created_at"]),
        updated_at=datetime.fromisoformat(data["updated_at"]),
        deleted_at=datetime.fromisoformat(deleted_at) if deleted_at else None,
        messages=messages,
    )


def _derive_title(content: str) -> str:
    """First user message, cut to a title on a word boundary."""
    first_line = content.strip().splitlines()[0]
    if len(first_line) <= MAX_DERIVED_TITLE_LENGTH:
        return first_line
    cut = first_line[: MAX_DERIVED_TITLE_LENGTH + 1]
    head, _, _ = cut.rpartition(" ")
    return (head or cut[:MAX_DERIVED_TITLE_LENGTH]).rstrip() + "…"


# --- tool-call → wire mapping -------------------------------------------------


def _tool_call_read(
    name: str, arguments: dict[str, Any], result: dict[str, Any]
) -> dict[str, Any]:
    """Map one dispatched call to a ToolCallRead-shaped dict.

    A dispatch/domain error (``ok: false``) rides on ``error``; every other
    result — including a legitimate ``legal: false`` move rejection — rides on
    ``result`` as compact JSON. Exactly one is set.
    """
    if result.get("ok") is False:
        return {
            "tool": name,
            "arguments": arguments,
            "result": None,
            "error": result.get("error"),
        }
    return {
        "tool": name,
        "arguments": arguments,
        "result": json.dumps(result, separators=(",", ":")),
        "error": None,
    }


def _tool_calls_read(
    tool_results: Sequence[dict[str, Any]], tool_args: Sequence[dict[str, Any]]
) -> list[dict[str, Any]] | None:
    """The run's tool calls as wire dicts, in order — ``None`` (not ``[]``)
    when no tools ran."""
    calls = [
        _tool_call_read(tr["name"], args, tr["result"])
        for tr, args in zip(tool_results, tool_args, strict=True)
    ]
    return calls or None


# --- per-IP rate limiter (ported from PCC, simplified for chess) --------------
#
# In-process sliding window, deliberately dependency-free: chess is a
# single-process, single-user, local-first app on a trusted LAN (loopback
# binding), so a shared/Redis limiter would be overkill.

# Maps "{bucket}:{client_ip}" -> monotonic-second timestamps of recent hits.
_HITS: dict[str, deque[float]] = defaultdict(deque)
_LOCK = threading.Lock()


def reset_rate_limit() -> None:
    """Clear all recorded hits. For tests only — the store is module-global."""
    with _LOCK:
        _HITS.clear()


def _messages_per_min() -> int:
    """The configured cap, read live so ops/tests can tune without re-import."""
    raw = os.environ.get("CHESSAPP_AGENT_MESSAGES_PER_MIN")
    if raw is None:
        return DEFAULT_AGENT_MESSAGES_PER_MIN
    try:
        return int(raw)
    except ValueError:
        return DEFAULT_AGENT_MESSAGES_PER_MIN


def rate_limit(
    bucket: str, *, window_seconds: float = 60.0
) -> Callable[[Request], None]:
    """A FastAPI dependency enforcing a per-IP cap on ``bucket``; 429 +
    ``Retry-After`` on breach."""

    def dependency(request: Request) -> None:
        limit = _messages_per_min()
        client_ip = (request.client.host if request.client else None) or "unknown"
        key = f"{bucket}:{client_ip}"
        now = time.monotonic()
        cutoff = now - window_seconds

        with _LOCK:
            hits = _HITS[key]
            while hits and hits[0] <= cutoff:
                hits.popleft()

            if len(hits) >= limit:
                retry_after = max(1, int(hits[0] + window_seconds - now) + 1)
                logger.warning(
                    "rate_limited bucket=%s client_ip=%s limit=%s retry_after=%s",
                    bucket,
                    client_ip,
                    limit,
                    retry_after,
                )
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="rate limit exceeded — slow down",
                    headers={"Retry-After": str(retry_after)},
                )

            if not hits:
                # The prune loop emptied this key; drop it before re-adding so
                # the store doesn't accumulate stale deques for idle IPs.
                del _HITS[key]
            _HITS[key].append(now)

    return dependency


# --- router -------------------------------------------------------------------


class RunCommand(Protocol):
    """The shared command pipeline, as this router calls it (`api._run_command`).

    A `Protocol` rather than a `Callable` alias because `origin` is keyword-only
    — a conversation's question may only be answered in that conversation
    (#281), so the caller has to name which one it is speaking for, and a
    `Callable[...]` cannot express a keyword at all.
    """

    def __call__(
        self,
        text: str,
        transcript: Sequence[dict[str, str]],
        version: int | None = None,
        *,
        origin: str,
        game_id: str | None = None,
    ) -> Awaitable[CommandOutcome]: ...


def build_agent_router(
    *, store: ConversationStore, run_command: RunCommand | None
) -> APIRouter:
    """The delegate router, mounted at ``/api/agent``.

    ``run_command`` is the shared command pipeline (from ``chessapp.api``);
    it is ``None`` when the app has no brain, in which case the one
    model-calling endpoint answers 503 (CRUD still works).
    """
    router = APIRouter(prefix="/api/agent", tags=["agent"])

    # One lock per conversation id. A message is an *exchange* — read the
    # history, commit the question, run the pipeline, commit the answer — and
    # unserialized that sequence interleaves: two concurrent posts to one thread
    # committed both user turns before either assistant turn, so the stored
    # thread stopped alternating and the second run reasoned from a transcript
    # ending in the first, still-unanswered question (#221). Every later
    # `history_for_loop` then replayed that order to the model.
    #
    # Keyed by id and not global, because different threads share nothing here:
    # a conductor waiting on one conversation must not stall another. (The one
    # thing they *do* share — the single game session — has its own guard
    # downstream in `api._run_command`; that is a different race, about one
    # board, and this lock neither replaces nor duplicates it.)
    #
    # `asyncio.Lock`, because the waiting has to yield the event loop rather
    # than block it, and the app is single-process. Router-scoped, so the map
    # lives exactly as long as the store it guards and a fresh app starts with
    # fresh locks. Entries are never reaped: only an id the store already knows
    # can mint one (the 404 below comes first), so the map is bounded by the
    # same conversations the store is already holding, at a fraction of the size.
    exchange_locks: defaultdict[int, asyncio.Lock] = defaultdict(asyncio.Lock)

    def _get_or_404(conversation_id: int) -> StoredConversation:
        conversation = store.get(conversation_id)
        if conversation is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="conversation not found"
            )
        return conversation

    def _detail(conversation: StoredConversation) -> ConversationDetail:
        return ConversationDetail(
            **ConversationRead.model_validate(conversation).model_dump(),
            messages=[MessageRead.model_validate(m) for m in conversation.messages],
        )

    def _replayed(
        conversation_id: int,
        content: str,
        asked: StoredMessage,
        answered: StoredMessage | None,
    ) -> MessageExchange:
        """The answer to a retried key (see `post_message`)."""
        if asked.content != content:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail="that Idempotency-Key was already used for a different message",
            )
        if answered is None:
            logger.info(
                "agent_delegate_retry_incomplete conversation_id=%s", conversation_id
            )
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=(
                    "that exchange did not complete — whatever it did is on the "
                    "board; read the state and send it again under a new key"
                ),
            )
        logger.info("agent_delegate_replayed conversation_id=%s", conversation_id)
        return MessageExchange(
            user_message=MessageRead.model_validate(asked),
            assistant_message=MessageRead.model_validate(answered),
        )

    @router.get("/conversations", response_model=list[ConversationRead])
    def list_conversations() -> list[StoredConversation]:
        return store.list_active()

    @router.post(
        "/conversations",
        response_model=ConversationRead,
        status_code=status.HTTP_201_CREATED,
    )
    def create_conversation(data: ConversationCreate) -> StoredConversation:
        conversation = store.create(title=data.title)
        logger.info("conversation_created conversation_id=%s", conversation.id)
        return conversation

    @router.get("/conversations/{conversation_id}", response_model=ConversationDetail)
    def get_conversation(conversation_id: int) -> ConversationDetail:
        return _detail(_get_or_404(conversation_id))

    @router.delete(
        "/conversations/{conversation_id}",
        status_code=status.HTTP_204_NO_CONTENT,
    )
    def delete_conversation(conversation_id: int) -> None:
        store.soft_delete(_get_or_404(conversation_id))
        logger.info("conversation_deleted conversation_id=%s", conversation_id)

    @router.post(
        "/conversations/{conversation_id}/messages",
        response_model=MessageExchange,
        dependencies=[Depends(rate_limit("agent_messages"))],
    )
    async def post_message(
        conversation_id: int,
        data: MessageCreate,
        x_agent_actor: Annotated[str | None, Header()] = None,
        idempotency_key: Annotated[
            str | None, Header(min_length=1, max_length=MAX_IDEMPOTENCY_KEY_LENGTH)
        ] = None,
    ) -> MessageExchange:
        """Store the user turn, run the command pipeline, store and return the
        assistant turn.

        **One exchange at a time per conversation.** The whole sequence — read
        the history, commit the question, run the pipeline, commit the answer —
        is serialized under that thread's lock, because it is one indivisible
        thing: concurrent posts to the same thread otherwise commit both
        questions before either answer, and the second run is handed a
        transcript ending in the first unanswered one (#221). A second caller
        waits its turn rather than being refused, so the conductor never has to
        retry; different conversations still run concurrently.

        The user turn is committed *before* the pipeline runs, so a provider
        failure — surfaced as 502 — never loses what the caller said, and a
        stale `version` (409, from the pipeline's mutation guard) is the same
        deal: the turn is on the record, the board is untouched, and the caller
        can resync and say it again. That ordering is preserved *inside* the
        serialized section. History
        replays as text turns only; ``X-Agent-Actor`` binds a trusted delegate
        caller (conductor) as the run's audit actor in the log, otherwise it
        falls back to the loop's default identity.

        The run is tagged with this conversation's own origin, which is what
        binds a destructive confirmation to the thread it was asked in (#281):
        the board version alone could not, because nothing needs to move between
        a question here and a "yes" said somewhere else.

        **A retry is not a second exchange** (#291). With an ``Idempotency-Key``
        the key is stored on the user turn, and a later post to this thread
        with the same key is answered from the store, never by the pipeline:

        - the exchange finished → the stored exchange, 200, byte for byte the
          first answer — whether the first reply was lost to a timeout or to a
          restart (the store is on disk when the app has a save dir);
        - the exchange never finished (a 502, a stale 409, a crash mid-run) →
          409: whatever it did is on the board, and running it again could do
          it twice; read the state and send a new key;
        - the key was used for different words → 422, a caller bug.

        Checked under the thread's lock, so a retry racing the original waits
        for it and then gets its answer. No key is today's behaviour.
        """
        # Fail fast before taking a lock, so an unknown id never mints one.
        _get_or_404(conversation_id)
        if run_command is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="agent unavailable: no brain",
            )
        actor = resolve_actor(x_agent_actor)
        logger.info(
            "agent_delegate_run conversation_id=%s actor=%s",
            conversation_id,
            actor,
        )

        async with exchange_locks[conversation_id]:
            # Re-read under the lock: this message may have spent the wait
            # queued behind another while the thread was deleted, and then it
            # must 404 like any other unknown thread rather than append to a
            # soft-deleted ghost.
            conversation = _get_or_404(conversation_id)
            if idempotency_key is not None:
                seen = store.find_exchange(conversation, idempotency_key)
                if seen is not None:
                    return _replayed(conversation_id, data.content, *seen)
            history = store.history_for_loop(conversation)
            user_message = store.append_user_message(
                conversation, data.content, idempotency_key
            )

            try:
                outcome = await run_command(
                    data.content,
                    history,
                    data.version,
                    # This thread, named: a destructive op the gate arms on this
                    # turn is armed for this conversation, and only a later turn
                    # *here* can answer it (#281). A "yes" posted to another
                    # thread, or typed into the web panel, used to run it.
                    origin=delegate_origin(conversation_id),
                    game_id=data.game_id,
                )
            except ProviderError as exc:
                logger.error(
                    "agent_delegate_run_failed conversation_id=%s error=%s",
                    conversation_id,
                    exc,
                )
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail=f"agent run failed: {exc}",
                ) from exc

            assistant_message = store.append_assistant_message(
                conversation,
                outcome.commentary,
                _tool_calls_read(outcome.tool_results, outcome.tool_args),
                outcome.stop_reason,
                memory=(
                    None if outcome.memory == outcome.commentary else outcome.memory
                ),
            )
        return MessageExchange(
            user_message=MessageRead.model_validate(user_message),
            assistant_message=MessageRead.model_validate(assistant_message),
        )

    return router
