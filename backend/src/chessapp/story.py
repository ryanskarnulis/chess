"""The story of the game: a running record, written by a model from code's facts (#372).

Both model phases used to read the chat history: the player's older requests
quoted back, and the last four exchanges verbatim (`conversation.condense`).
The planner read Glitch's slang as its own past turns, and a false line stayed
in history as fact. The story replaces that: a neutral, third-person account of
what has happened, rewritten after every turn from the **ledger** (`ledger.py`,
what code saw happen), the turn's tool calls and results, and the player's
words — with Glitch's words riding along labelled as what he said, never as a
source of facts. `docs/story-and-ledger.md` is the design.

The pieces:

- `TurnNote` — one turn, as the summarizer is told about it: written by code as
  the turn ends, holding the ledger events since the conversation's last note.
  The app's own lines (a stuck reply, a move confirmation) are never in it:
  only what Glitch himself said, as the transcript already remembers him.
- `StoryState` — one conversation's story: the text, which notes it covers, the
  notes still pending, the ledger cursor, and the accumulated evidence the
  story's claims are scored against offline (`facts.story_facts`). It rides
  wherever that conversation's transcript rides.
- `ChatSummarizer` — the one model call: previous story + every pending note →
  new story, thinking off, a hard token cap. A failed or truncated call leaves
  the old story standing and the notes pending; the next call catches up.
- `StoryKeeper` — one background worker for every conversation. A turn enqueues
  its note and returns; the worker summarizes off the turn's path. Nothing
  raises into a turn.

Nothing reads the story yet (PR 2 of #372). Each turn records whether its story
was caught up when it began, and each summarizer run records how long the turns
that began before it finished would have waited on it (`lag_ms`): the wait the
next PR adds when Glitch starts reading it.
"""

import json
import logging
import queue
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from chessapp.brain import (
    CALL_FAILED,
    CALL_OK,
    CALL_TRUNCATED,
    PHASE_SUMMARIZER,
    ModelCall,
    ServerStamp,
)
from chessapp.context_capture import model_phase
from chessapp.ledger import (
    DRAW_OFFER,
    GAME_END,
    MOVE,
    NEW_GAME,
    RESUMED,
    SETTING,
    TAKEBACK,
    Ledger,
)
from chessapp.personality import SUMMARIZER_PROMPT
from chessapp.progress import attributed
from chessapp.provider import ChatProvider, ProviderError

logger = logging.getLogger(__name__)

# The call's hard ceiling: the prompt's shape runs ~250 tokens; a story the
# cap cuts off is not a story, so a truncated call keeps the old one.
STORY_MAX_TOKENS = 450
# A record, not words: the planner's measured number (#286), not the
# narrator's spread.
SUMMARIZER_TEMPERATURE = 0.3
# How long one call may take before the worker hangs up. Off the turn's path,
# so generous; it only stops a stalled server holding the worker forever.
SUMMARIZER_TIMEOUT_S = 90.0
# How many turns one call may take in. A backlog past this is caught up over
# several calls rather than in one prompt that grows without bound.
NOTES_PER_CALL = 12
# A player's words and Glitch's, cut to this many characters in a note: the
# story keeps what matters in a sentence, and a pasted wall of text should not
# become one.
WORDS_CHARS = 400


@dataclass(frozen=True)
class TurnNote:
    """One turn, as the summarizer is told about it. `words` is None for a
    turn with none (a board drag). `tools` is the turn's `{name, args,
    result}` list; `events` the ledger events since the previous note, as
    dicts; `draft` Glitch's own words, or "" when he said nothing."""

    seq: int
    correlation_id: str
    turn_id: int
    route: str
    words: str | None
    tools: Sequence[Mapping[str, Any]]
    draft: str
    events: Sequence[Mapping[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "correlation_id": self.correlation_id,
            "turn_id": self.turn_id,
            "route": self.route,
            "words": self.words,
            "tools": [dict(t) for t in self.tools],
            "draft": self.draft,
            "events": [dict(e) for e in self.events],
        }

    @classmethod
    def from_dict(cls, data: Any) -> "TurnNote":
        if not isinstance(data, dict):
            raise ValueError("a turn note must be an object")
        words = data.get("words")
        if words is not None and not isinstance(words, str):
            raise ValueError("note words must be a string")
        tools, events = data.get("tools", []), data.get("events", [])
        if not isinstance(tools, list) or not isinstance(events, list):
            raise ValueError("note tools and events must be lists")
        return cls(
            seq=int(data["seq"]),
            correlation_id=str(data.get("correlation_id", "")),
            turn_id=int(data.get("turn_id", 0)),
            route=str(data.get("route", "")),
            words=words,
            tools=tools,
            draft=str(data.get("draft", "")),
            events=events,
        )


class StoryState:
    """One conversation's story and what it has yet to take in.

    `covered_through` is the last note seq the text covers (-1: none);
    `cursor` the next ledger seq this conversation has not noted. `version`
    moves on every change, so a checkpoint knows to write it. All access goes
    through `lock`: the worker updates it while the turn thread serializes it.
    """

    def __init__(self, *, cursor: int = 0) -> None:
        self.text = ""
        self.covered_through = -1
        self.next_seq = 0
        self.cursor = cursor
        self.pending: list[TurnNote] = []
        self.evidence: dict[str, Any] = {}
        self.version = 0
        self.lock = threading.Lock()

    def to_dict(self) -> dict[str, Any]:
        with self.lock:
            return {
                "text": self.text,
                "covered_through": self.covered_through,
                "next_seq": self.next_seq,
                "cursor": self.cursor,
                "pending": [n.to_dict() for n in self.pending],
                "evidence": json.loads(json.dumps(self.evidence)),
            }

    @classmethod
    def from_dict(cls, data: Any, *, cursor: int = 0) -> "StoryState":
        """A saved story, or an empty one (at `cursor`) when there is none or
        it does not parse: a story is derived memory, and losing it costs the
        conversation its past, never a failed start."""
        state = cls(cursor=cursor)
        if data is None:
            return state
        try:
            if not isinstance(data, dict):
                raise ValueError("a story must be an object")
            text = data.get("text", "")
            if not isinstance(text, str):
                raise ValueError("story text must be a string")
            pending = [TurnNote.from_dict(n) for n in data.get("pending", [])]
            evidence = data.get("evidence", {})
            if not isinstance(evidence, dict):
                raise ValueError("story evidence must be an object")
            state.text = text
            state.covered_through = int(data.get("covered_through", -1))
            state.next_seq = int(data.get("next_seq", 0))
            state.cursor = int(data.get("cursor", cursor))
            state.pending = pending
            state.evidence = evidence
        except (KeyError, TypeError, ValueError):
            logger.warning("ignoring invalid story", exc_info=True)
            return cls(cursor=cursor)
        return state

    def last_note(self) -> int:
        return self.next_seq - 1


# --- what the summarizer reads -------------------------------------------------


def render_event(event: Mapping[str, Any]) -> str:
    """One ledger event in plain words. Deterministic: the summarizer is told
    what happened in sentences code wrote, never asked to read JSON."""
    kind = event.get("kind")
    if kind == MOVE:
        number = event.get("move_number", "?")
        label = f"{number}. " if event.get("color") == "white" else f"{number}... "
        who = "the player" if event.get("by") == "player" else "Glitch"
        line = f"{label}{event.get('san')} by {who}"
        if event.get("capture"):
            line += f", taking a {event['capture']}"
            line += f" ({_material_words(event.get('material', 0))})"
        if event.get("check"):
            line += ", check"
        return line
    if kind == TAKEBACK:
        undone = ", ".join(event.get("undone", ()))
        plies = event.get("plies", 0)
        return f"taken back: {undone} ({plies} move{'s' if plies != 1 else ''})"
    if kind == SETTING:
        return (
            f"{event.get('name')} changed from {event.get('before')} "
            f"to {event.get('after')}"
        )
    if kind == DRAW_OFFER:
        if event.get("accepted"):
            return "the player offered a draw, and Glitch accepted"
        reason = str(event.get("reason") or "").replace("_", " ")
        return "the player offered a draw, and Glitch declined" + (
            f" ({reason})" if reason else ""
        )
    if kind == GAME_END:
        winner = {"player": "the player won", "opponent": "Glitch won"}.get(
            event.get("winner"), "a draw"
        )
        termination = str(event.get("termination", "")).replace("_", " ")
        return f"the game ended by {termination}: {winner} ({event.get('result')})"
    if kind in (NEW_GAME, RESUMED):
        start = (
            f"the saved game '{event.get('name')}' was resumed"
            if kind == RESUMED
            else "a new game began"
        )
        start += f"; the player has {event.get('player_color')}"
        if event.get("root_fen"):
            start += ", from a set-up position"
        return start
    return str(kind)


def _material_words(material: Any) -> str:
    if not isinstance(material, int) or material == 0:
        return "material level"
    side = "up" if material > 0 else "down"
    return f"the player {side} {abs(material)}"


def render_events(events: Sequence[Mapping[str, Any]]) -> list[str]:
    """The events, a line each — except a resumed save's replayed line, which
    is one line of moves: it is where the game stood, not what happened."""
    lines: list[str] = []
    replayed: list[str] = []
    for event in events:
        if event.get("kind") == MOVE and event.get("restored"):
            number, san = event.get("move_number"), event.get("san")
            replayed.append(
                f"{number}. {san}" if event.get("color") == "white" else str(san)
            )
            continue
        if replayed:
            lines.append("its moves so far: " + " ".join(replayed))
            replayed = []
        lines.append(render_event(event))
    if replayed:
        lines.append("its moves so far: " + " ".join(replayed))
    return lines


def render_tool(tool: Mapping[str, Any]) -> str:
    """One tool call and what it answered, in a line."""
    name = tool.get("name", "?")
    args = tool.get("args") or {}
    call = f"{name}({', '.join(f'{k}={v}' for k, v in args.items())})"
    result = tool.get("result") or {}
    if result.get("ok") is False:
        return f"{call}: refused — {_cut(str(result.get('error', '')), 160)}"
    if name == "make_move":
        if result.get("legal") is False:
            return f"{call}: not played — {result.get('reason', '')}"
        return f"{call}: played {result.get('san')}"
    if name == "ask_player":
        return f"{call}: asked the player to choose between " + ", ".join(
            result.get("candidates", ())
        )
    if name == "get_best_moves":
        sans = [m.get("san") for m in result.get("moves", ()) if m.get("san")]
        return f"{call}: the engine suggested " + ", ".join(sans)
    for key in ("summary", "description", "verdict"):
        if isinstance(result.get(key), str):
            return f"{call}: {_cut(result[key], 240)}"
    return f"{call}: done"


def render_note(note: TurnNote, number: int) -> str:
    """One turn, as the summarizer reads it."""
    lines = [f"Turn {number}"]
    events = render_events(note.events)
    if events:
        lines.append("What happened:")
        lines += [f"- {line}" for line in events]
    else:
        # Said, not left out: an absent line reads as "not mentioned", and a
        # 12B then takes Glitch's "I'm going Nf6" for the move nobody played
        # (the handoff's "Done this turn: nothing." for the same reason).
        lines.append("What happened: nothing on the board.")
    if note.words is None:
        lines.append("The player moved a piece on the board.")
    elif note.words:
        lines.append(f'The player said: "{_cut(note.words, WORDS_CHARS)}"')
    if note.tools:
        lines.append("Tools:")
        lines += [f"- {render_tool(tool)}" for tool in note.tools]
    if note.draft:
        lines.append(
            "Glitch said (his words only — a move or event he names happened "
            "only if What happened shows it): "
            f'"{_cut(note.draft, WORDS_CHARS)}"'
        )
    else:
        lines.append("Glitch said nothing.")
    return "\n".join(lines)


def render_request(previous: str, notes: Sequence[TurnNote], first: int) -> str:
    """The summarizer's user message: the story so far, then the new turns
    numbered from `first`."""
    story = previous.strip() or "(nothing yet: the story starts here)"
    turns = "\n\n".join(
        render_note(note, first + index) for index, note in enumerate(notes)
    )
    return f"The story so far:\n{story}\n\nNew turns, oldest first:\n\n{turns}"


def _cut(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


# --- the evidence the story is scored against ----------------------------------


def accumulate(evidence: dict[str, Any], note: TurnNote) -> dict[str, Any]:
    """The evidence after the story takes in `note`: every fact a story
    covering it may state, in JSON (`facts.story_facts` reads it back).
    Game-spanning like the story itself — a capture twenty turns ago is still
    a true thing for the story to say."""
    from chessapp.facts import analysis_numbers, reported_moves

    e = {key: value for key, value in evidence.items()}
    sets = (
        "moves",
        "by_player",
        "by_opponent",
        "captured_by_player",
        "captured_by_opponent",
        "settings_changed",
        "numbers",
    )
    bag = {key: set(e.get(key, ())) for key in sets}
    material = list(e.get("material", [0]))
    for event in note.events:
        kind = event.get("kind")
        if kind in (NEW_GAME, RESUMED):
            if e.get("games"):
                e["restarted"] = True
            e["games"] = int(e.get("games", 0)) + 1
            e["player_color"] = event.get("player_color")
            if kind == RESUMED:
                e["saved"] = True
        elif kind == MOVE:
            san = event.get("san")
            bag["moves"].add(san)
            side = "player" if event.get("by") == "player" else "opponent"
            bag[f"by_{side}"].add(san)
            if event.get("capture"):
                bag[f"captured_by_{side}"].add(event["capture"])
            e.setdefault("captures_by_move", {})[san] = event.get("capture") or ""
            if isinstance(event.get("material"), int):
                material.append(event["material"])
            if event.get("check"):
                e["check"] = True
        elif kind == TAKEBACK:
            e["undone"] = True
        elif kind == SETTING:
            bag["settings_changed"].add(event.get("name"))
            e.setdefault("settings", {})[event.get("name")] = event.get("after")
        elif kind == GAME_END:
            # Historical like everything here: a game that ended earlier in
            # the story stays a true ending to tell, whatever runs now.
            e["ended"] = True
            e["drawn"] = event.get("winner") is None
            e["winner"] = event.get("winner")
            e["termination"] = event.get("termination")
    tools = [
        {"name": t.get("name"), "result": t.get("result") or {}} for t in note.tools
    ]
    bag["numbers"] |= analysis_numbers(tools)
    bag["moves"] |= reported_moves(tools)
    if any(
        t["name"] in ("save_game", "resume_game") and t["result"].get("ok") is True
        for t in tools
    ):
        e["saved"] = True
    for key, values in bag.items():
        e[key] = sorted(v for v in values if isinstance(v, str))
    e["material"] = sorted(set(material))
    return e


# --- the call --------------------------------------------------------------------


@dataclass(frozen=True)
class Summary:
    """One summarizer run: the new story (None when it produced none) and
    every call it made, as the trace records them — one, or two when the first
    ran into the token cap and was asked again for less."""

    text: str | None
    calls: tuple[ModelCall, ...]

    @property
    def call(self) -> ModelCall:
        """The call whose words were kept (or the last one tried)."""
        return self.calls[-1]


# Said once, to a story that ran into the cap. The prompt's word limit is a
# request a 12B does not always keep; left alone, a story that outgrew the cap
# would be cut every turn after and never told again.
_SHORTER = (
    "\n\nYour last reply ran too long. Write it again, shorter: at most 80 "
    "words for The game so far."
)


@dataclass
class ChatSummarizer:
    """The summarizer on any `ChatProvider` — the same server the brain uses
    (#372 decision 4). Thinking off, a low temperature, a hard token cap."""

    provider: ChatProvider
    max_tokens: int = STORY_MAX_TOKENS
    temperature: float | None = SUMMARIZER_TEMPERATURE
    timeout: float | None = SUMMARIZER_TIMEOUT_S
    clock: Callable[[], float] = field(default=time.monotonic)

    def summarize(
        self, previous: str, notes: Sequence[TurnNote], first: int
    ) -> Summary:
        request = render_request(previous, notes, first)
        text, call = self._call(request)
        if call.status != CALL_TRUNCATED:
            return Summary(text, (call,))
        retry_text, retry = self._call(request + _SHORTER)
        return Summary(retry_text, (call, retry))

    def _call(self, request: str) -> tuple[str | None, ModelCall]:
        messages = [
            {"role": "system", "content": SUMMARIZER_PROMPT},
            {"role": "user", "content": request},
        ]
        started = self.clock()
        try:
            with model_phase(PHASE_SUMMARIZER):
                result = self.provider.chat(
                    messages,
                    tools=None,
                    enable_thinking=False,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    timeout=self.timeout,
                )
        except ProviderError as exc:
            logger.warning("summarizer_failed", exc_info=True)
            return None, ModelCall(
                PHASE_SUMMARIZER,
                CALL_FAILED,
                self._ms(started),
                failure=str(exc.failure),
            )
        usage = result.usage
        meta = result.server
        call = ModelCall(
            PHASE_SUMMARIZER,
            CALL_TRUNCATED if result.finish_reason == "length" else CALL_OK,
            self._ms(started),
            usage.prompt_tokens if usage is not None else None,
            usage.completion_tokens if usage is not None else None,
            server=ServerStamp(
                fingerprint=meta.fingerprint,
                server_ms=meta.server_ms,
                cached_tokens=meta.cached_tokens,
            )
            if meta is not None
            else None,
        )
        text = (result.content or "").strip()
        if call.status != CALL_OK or not text:
            return None, call
        return text, call

    def _ms(self, started: float) -> int:
        return round((self.clock() - started) * 1000)


class Summarizer(Protocol):
    """What the keeper calls: `ChatSummarizer`, or a test double. `first` is
    the number the first note is shown under ("Turn 7")."""

    def summarize(
        self, previous: str, notes: Sequence[TurnNote], first: int
    ) -> Summary: ...


# --- the keeper ------------------------------------------------------------------


@dataclass
class _Waiter:
    origin: str
    needs: int
    started: float
    correlation_id: str


class StoryKeeper:
    """Keeps every conversation's story, off the turns' path.

    `states(origin)` finds a conversation's `StoryState` (None when it has
    none); `ledger()` the app's ledger; `record(fields)` writes a `story`
    trace record. One daemon worker thread takes origins off a queue and
    summarizes whatever that origin has pending, one call per batch.
    """

    def __init__(
        self,
        summarizer: Summarizer,
        *,
        states: Callable[[str], StoryState | None],
        ledger: Callable[[], Ledger],
        record: Callable[[dict[str, Any]], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._summarizer = summarizer
        self._states = states
        self._ledger = ledger
        self._record = record
        self._clock = clock
        self._queue: queue.Queue[str] = queue.Queue()
        self._queued: set[str] = set()
        self._busy = 0
        self._lock = threading.Lock()
        self._idle = threading.Condition(self._lock)
        self._waiters: list[_Waiter] = []
        self._thread: threading.Thread | None = None

    # --- the turn's side: quick, never raises --------------------------------

    def turn_started(self, origin: str, correlation_id: str) -> dict[str, Any]:
        """Whether this conversation's story covered every earlier turn as
        this one began, for the turn's trace. When it did not, the turn is
        remembered, and the run that catches up reports how long it would have
        waited (`lag_ms` on the `story` record)."""
        try:
            state = self._states(origin)
            if state is None:
                return {"pending": 0}
            with state.lock:
                covered, needs = state.covered_through, state.last_note()
                pending = len(state.pending)
            if needs <= covered:
                return {"pending": 0, "covered_through": covered}
            with self._lock:
                self._waiters.append(
                    _Waiter(origin, needs, self._clock(), correlation_id)
                )
            return {"pending": pending, "covered_through": covered}
        except Exception:
            logger.warning("story_turn_started_failed", exc_info=True)
            return {}

    def enqueue(
        self,
        origin: str,
        *,
        correlation_id: str,
        turn_id: int,
        route: str,
        words: str | None,
        tools: Sequence[Mapping[str, Any]],
        draft: str,
    ) -> None:
        """Note a finished turn and start its story off the turn's path. The
        ledger events since this conversation's last note ride in it, so call
        it after the ledger has observed the turn."""
        try:
            state = self._states(origin)
            if state is None:
                return
            ledger = self._ledger()
            with state.lock:
                if state.cursor > ledger.next_seq:
                    # The ledger was rebuilt under a restored story: start from
                    # its current game rather than from a seq it never reached.
                    current = ledger.current()
                    state.cursor = current[0].seq if current else ledger.next_seq
                events = [e.to_dict() for e in ledger.since(state.cursor)]
                state.cursor = ledger.next_seq
                state.pending.append(
                    TurnNote(
                        seq=state.next_seq,
                        correlation_id=correlation_id,
                        turn_id=turn_id,
                        route=route,
                        words=words,
                        tools=[_compact_tool(t) for t in tools],
                        draft=draft,
                        events=events,
                    )
                )
                state.next_seq += 1
                state.version += 1
            self.kick(origin)
        except Exception:
            logger.warning("story_enqueue_failed", exc_info=True)

    def kick(self, origin: str) -> None:
        """Schedule `origin`'s pending notes (a restored backlog, say)."""
        with self._lock:
            if origin in self._queued:
                return
            self._queued.add(origin)
            self._queue.put(origin)
            if self._thread is None:
                self._thread = threading.Thread(
                    target=self._work, name="story-keeper", daemon=True
                )
                self._thread.start()

    def wait_idle(self, timeout: float | None = None) -> bool:
        """Block until nothing is queued or running; True if it got there.
        For tests and the eval harness, never the live path."""
        deadline = None if timeout is None else self._clock() + timeout
        with self._idle:
            while self._queued or self._busy:
                remaining = None if deadline is None else deadline - self._clock()
                if remaining is not None and remaining <= 0:
                    return False
                self._idle.wait(remaining)
            return True

    # --- the worker's side ----------------------------------------------------

    def _work(self) -> None:
        while True:
            origin = self._queue.get()
            with self._lock:
                self._queued.discard(origin)
                self._busy += 1
            again = False
            try:
                again = self._run(origin)
            except Exception:
                logger.warning("story_run_failed", exc_info=True)
            finally:
                with self._idle:
                    self._busy -= 1
                    self._idle.notify_all()
            if again:
                self.kick(origin)

    def _run(self, origin: str) -> bool:
        """Summarize `origin`'s pending notes; True if more are waiting."""
        state = self._states(origin)
        if state is None:
            return False
        with state.lock:
            notes = list(state.pending[:NOTES_PER_CALL])
            previous = state.text
            game_id = self._ledger().game_id
        if not notes:
            return False
        last = notes[-1]
        first = notes[0].seq + 1
        with attributed(last.correlation_id, last.turn_id):
            summary = self._summarizer.summarize(previous, notes, first)
        waited: list[dict[str, Any]] = []
        with state.lock:
            if summary.text is not None:
                covered = {note.seq for note in notes}
                state.pending = [n for n in state.pending if n.seq not in covered]
                state.text = summary.text
                state.covered_through = last.seq
                for note in notes:
                    state.evidence = accumulate(state.evidence, note)
                state.version += 1
            evidence = json.loads(json.dumps(state.evidence))
            more = bool(state.pending)
            covered_through = state.covered_through
        if summary.text is not None:
            now = self._clock()
            with self._lock:
                keep: list[_Waiter] = []
                for waiter in self._waiters:
                    if waiter.origin == origin and waiter.needs <= covered_through:
                        waited.append(
                            {
                                "correlation_id": waiter.correlation_id,
                                "lag_ms": round((now - waiter.started) * 1000),
                            }
                        )
                    else:
                        keep.append(waiter)
                self._waiters = keep
        self._write(
            {
                "origin": origin,
                "game_id": game_id,
                "status": summary.call.status,
                "notes": [n.to_dict() for n in notes],
                "covered": [n.correlation_id for n in notes],
                "covered_through": covered_through,
                "previous": previous,
                "story": summary.text,
                "call": summary.call.as_trace(len(summary.calls) - 1),
                "calls": [c.as_trace(i) for i, c in enumerate(summary.calls)],
                "waited": waited,
                "pending_after": len(state.pending),
                "evidence": evidence,
            }
        )
        # A failed call does not retry by itself: the next turn's note kicks
        # the worker again, so a dead server is not hammered in a loop.
        return more and summary.text is not None

    def _write(self, fields: dict[str, Any]) -> None:
        if self._record is None:
            return
        try:
            self._record(fields)
        except Exception:
            logger.warning("story_record_failed", exc_info=True)


def _compact_tool(tool: Mapping[str, Any]) -> dict[str, Any]:
    """A tool call as a note keeps it: name, args, and the result without the
    board dumps no story needs (the menu, the FEN)."""
    result = dict(tool.get("result") or {})
    for key in ("legal_moves", "captures", "fen", "legal_destinations"):
        result.pop(key, None)
    return {
        "name": tool.get("name"),
        "args": dict(tool.get("args") or {}),
        "result": result,
    }
