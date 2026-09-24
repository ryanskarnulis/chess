"""Seeded, composed interaction trajectories over the shipped app (#318).

Every other deterministic test here drives one feature: one command, one tool,
one gate. The three bugs #318 was filed over lived *between* features — an ask
batched with a move (#314), an undo refreshing the board while the offer stayed
on the old one (#315), a closer outliving its budget (#316) — and each of them
passed a suite in which every feature, alone, was green.

So this drives *sequences*: a seeded random walk over what a player, a
conductor and a planner can do to one game, through the real app
(`app.build_app`, the same assembly that ships, with a `ScriptedProvider`
standing in for llama-server and a deterministic engine for Stockfish), and
after every step checks invariants that must hold however the steps were
combined. The model is scripted, so a walk is fully determined by its seed: a
failure is a seed, and the seed is the reproduction.

The walk reaches every surface that can move the board or answer a question:
the web panel, two delegate conversations (with and without an
`Idempotency-Key`, and retries of a keyed exchange), the board endpoints, and
a restart over the same save directory. It also injects the failures a real
evening produces — llama-server dying mid-command, Stockfish dying
mid-exchange, a narrator that stalls past its budget — and can start from the
84-ply fixture rather than the opening.

Nothing here reads language. The scripted planner's batches are chosen off the
live `legal_moves` (the menu a real planner is handed), the narrator says a
fixed neutral line, and every check is over the board, the tool results, the
requests the provider was sent, and the save directory.
"""

from __future__ import annotations

import functools
import json
import random
import threading
import time
import zlib
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any
from unittest import mock

import chess
import chess.engine
import chess.pgn
from fastapi.testclient import TestClient

from chessapp import app as app_module
from chessapp.agent_api import reset_rate_limit
from chessapp.api import create_app
from chessapp.app import build_app
from chessapp.engine import CandidateMove, Evaluation
from chessapp.game import GameSession
from chessapp.llama_brain import (
    _ANSWER_MAX_TOKENS,
    _DEFAULT_MAX_ANALYSIS_CALLS,
    _DEFAULT_MAX_TOOL_CALLS,
    _EXPENSIVE_TOOLS,
    create_llama_brain,
)
from chessapp.provider import ChatResult, ProviderError
from chessapp.tools import LIVE_CHECKPOINT_FILENAME
from fakes import ScriptedProvider, text_turn, tool_calls_turn

# What the scripted narrator (and a planner with nothing left to do) says: no
# move, no verdict, no ending — nothing the honesty guard could hold against a
# board, so a guard firing on it would be the guard's bug, not the script's.
NEUTRAL = "Okay."

# What a stalled narrator says once it is finally released, long after its
# turn moved on. It must appear nowhere: not in the answer, and not in any
# later request's memory (#316).
LATE_WORDS = "Late words nobody should ever hear."

# The two budgets a stalled narrator runs into, cut down from the shipped 10 s
# so a stall costs the corpus a second rather than ten. Both are the knobs the
# shipped values ride (`LlamaBrain.closing_budget_s` / `closing_ceiling_s` and
# `create_app(reaction_budget=…)`); only the numbers differ.
BUDGET_S = 0.5
# How far past its budget a stalled step may run: the thread hand-off and the
# engine's reply, never the stall itself.
STALL_SLACK_S = 3.0
# How long a stalled narrator waits to be released before giving up on its own.
PATIENCE_S = 10.0

# The planner's opening block and its mid-command refresh, as the loop writes
# them (`llama_brain._messages`, `llama_brain._board_refresh_message`). The
# offer invariant reads the last board shown out of the request itself.
_OPENING = "Board state:\n"
_REFRESH = "Board state after those tool calls:\n"

# Tools whose success changes the game on the board. Everything else a batch
# can call reads, or changes a setting, or arms a question.
_BOARD_MUTATORS = frozenset(
    {"make_move", "undo", "new_game", "resign", "resume_game", "claim_draw"}
)
# Tools the confirmation gate stands in front of (`tools._gate`).
_GATED = frozenset({"new_game", "resign", "claim_draw", "resume_game", "save_game"})
# The gate's refusal when it arms an op, and when one is already armed.
_ARMED = "confirmation required"

# A move no position has: SAN-shaped, on a square that does not exist.
_NEVER_LEGAL = "Qz9"

# Where the conversation endpoints live, and the surfaces a walk speaks from:
# the web panel and two delegate threads, so a question asked in one can be
# answered — wrongly — from another (#281).
PANEL = "panel"
DELEGATES = ("d0", "d1")
_MESSAGES = "/api/agent/conversations/{id}/messages"

_LATE_GAME = Path(__file__).with_name("late_game_84_plies.pgn")


class LegalEngine:
    """Engine double that always answers with a legal move, and can die.

    `FakeEngine` replies one fixed UCI, which is illegal in almost every
    position a walk reaches. This picks among the legal replies by a hash of
    the position, so the reply varies across a game and is still a pure
    function of the board — the walk's determinism rests on it. While `dead`
    every call raises what a real Stockfish going away raises (#284).
    """

    def __init__(self) -> None:
        self.closed = False
        self.dead = False

    def _alive(self) -> None:
        if self.dead:
            raise chess.engine.EngineTerminatedError("engine process died")

    def choose_move(self, session: GameSession) -> str:
        self._alive()
        board = chess.Board(session.fen())
        moves = sorted(move.uci() for move in board.legal_moves)
        return moves[zlib.crc32(board.fen().encode()) % len(moves)]

    def play_move(self, session: GameSession):
        return session.submit_move(self.choose_move(session))

    def get_best_moves(self, session: GameSession, n: int = 3):
        self._alive()
        board = chess.Board(session.fen())
        moves = sorted(board.legal_moves, key=lambda move: move.uci())[:n]
        return [
            CandidateMove(uci=move.uci(), san=board.san(move), score_cp=0, mate_in=None)
            for move in moves
        ]

    def evaluate_position(self, session: GameSession) -> Evaluation:
        self._alive()
        return Evaluation(score_cp=0, mate_in=None)

    def close(self) -> None:
        self.closed = True

    def set_skill_level(self, level: int) -> None:
        pass

    def set_elo(self, elo: int) -> None:
        pass

    def set_tier(self, tier: str) -> None:
        pass


class WalkProvider(ScriptedProvider):
    """The scripted llama-server, which can stall its narrator.

    With `stall` set, a narrator call — the one offered no tools — is recorded
    and then sits until released, as `BlockingNarratorProvider` does; the
    planner's round trips play their script regardless. Only the words stall.
    Reading a free-form answer to a pending question is tool-free too, but it
    is the turn's own first step, like a planner call, with nothing ready held
    behind it — so it is not a narrator here, and it never stalls.
    """

    def __init__(self) -> None:
        super().__init__(text_turn(NEUTRAL))
        self.stall = False
        self.release = threading.Event()
        self.stalled: list[threading.Event] = []

    def chat(self, messages, *, tools=None, **kwargs) -> ChatResult:
        speech = tools is None and kwargs.get("max_tokens") != _ANSWER_MAX_TOKENS
        if not self.stall or not speech:
            try:
                return super().chat(messages, tools=tools, **kwargs)
            except ProviderError as exc:
                # A fresh one per call: the scripted exception is one object,
                # and re-raising it would chain every earlier traceback on.
                raise ProviderError(str(exc), exc.failure) from None
        super().chat(messages, tools=tools, **kwargs)  # recorded, not played
        finished = threading.Event()
        self.stalled.append(finished)
        self.release.wait(timeout=PATIENCE_S)
        finished.set()
        return text_turn(LATE_WORDS)

    def unstall(self) -> None:
        """Let every stalled narrator finish, and wait until each has."""
        self.stall = False
        self.release.set()
        for finished in self.stalled:
            finished.wait(timeout=PATIENCE_S)
        self.stalled.clear()
        self.release = threading.Event()


@dataclass(frozen=True)
class Step:
    """One thing that happens to the game.

    `kind` is the seam: `command` (a conversation, through the planner),
    `literal` (a conversation, a text the fast paths settle — a SAN move,
    "yes", "no"), `control` (a board endpoint), `retry` (a keyed delegate
    exchange sent again), or `restart` (a new process over the same save
    directory). `origin` is which conversation speaks: the panel or one of the
    delegate threads. `rounds` is the scripted planner: one tuple of calls per
    planner round trip, several calls in one round being one model response.

    The injected failures: `provider_dies` is the planner round at which
    llama-server stops answering for the rest of the step, `engine_dies` kills
    Stockfish for the step, and `stall` holds every narrator call past its
    budget.
    """

    kind: str
    text: str = ""
    origin: str = PANEL
    rounds: tuple[tuple[tuple[str, dict[str, Any]], ...], ...] = ()
    endpoint: str = ""
    body: dict[str, Any] = field(default_factory=dict)
    key: str | None = None
    provider_dies: int | None = None
    engine_dies: bool = False
    stall: bool = False

    def describe(self) -> str:
        flags = "".join(
            f" [{flag}]"
            for flag, on in (
                (
                    f"provider dies at round {self.provider_dies}",
                    self.provider_dies is not None,
                ),
                ("engine dies", self.engine_dies),
                ("narrator stalls", self.stall),
                (f"key={self.key}", self.key is not None),
            )
            if on
        )
        where = "" if self.origin == PANEL else f"@{self.origin} "
        if self.kind == "command":
            rounds = " | ".join(
                ", ".join(f"{name}({_args(args)})" for name, args in batch)
                for batch in self.rounds
            )
            return f"command {where}{self.text!r}: {rounds}{flags}"
        if self.kind in ("literal", "retry"):
            return f"{self.kind} {where}{self.text!r}{flags}"
        if self.kind == "control":
            return f"control {self.endpoint} {self.body}{flags}"
        return self.kind


def _args(args: dict[str, Any]) -> str:
    return ", ".join(f"{key}={value!r}" for key, value in args.items())


class InvariantBreach(AssertionError):
    """An invariant failed. Carries the whole reproduction in its message."""


# --- the app under the walk ----------------------------------------------------


def late_game_session() -> GameSession:
    """The 84-ply fixture, replayed through the session's own legality gate."""
    with _LATE_GAME.open() as source:
        game = chess.pgn.read_game(source)
    assert game is not None
    session = GameSession()
    for move in game.mainline_moves():
        assert session.submit_move(move.uci()).legal
    return session


class TurnRecords:
    """The app's `Tracer` seam, in memory: every turn record, across restarts.
    What the open-question invariant reads (#319) — the record is the only
    place the app says which question a turn found open, and what it did to
    it."""

    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    def record(self, record: dict[str, Any]) -> None:
        if record.get("kind") == "turn":
            self.records.append(record)


class Harness:
    """The shipped app over a scripted provider, restartable in place."""

    def __init__(self, save_dir: Path, *, start: GameSession | None = None) -> None:
        self.save_dir = save_dir
        save_dir.mkdir(parents=True, exist_ok=True)
        if start is not None:
            # A checkpoint, exactly as the app writes one, so the first
            # process restores it the way it restores any game on disk.
            (save_dir / LIVE_CHECKPOINT_FILENAME).write_text(
                json.dumps(
                    {
                        "checkpoint": 1,
                        "board_version": 0,
                        "session": start.to_dict(),
                        "transcript": [],
                    }
                )
            )
        self.provider = WalkProvider()
        self.engine = LegalEngine()
        self.tracer = TurnRecords()
        self.client = self._build()
        self.conversations = {
            origin: self.client.post("/api/agent/conversations", json={}).json()["id"]
            for origin in DELEGATES
        }

    def _build(self) -> TestClient:
        """`build_app`, with only the two narrator budgets cut down."""

        def brain(**kwargs: Any):
            built = create_llama_brain(**kwargs)
            built.closing_budget_s = BUDGET_S
            built.closing_ceiling_s = BUDGET_S
            return built

        with (
            mock.patch.object(app_module, "create_llama_brain", brain),
            mock.patch.object(
                app_module,
                "create_app",
                functools.partial(create_app, reaction_budget=BUDGET_S),
            ),
        ):
            app = build_app(
                provider=self.provider,
                engine=self.engine,
                save_dir=self.save_dir,
                tracer=self.tracer,
            )
        return TestClient(app)

    def restart(self) -> None:
        """A new process on the same disk: the old client is closed and the app
        is assembled again, restoring `live.json` and `conversations.json`."""
        self.client.close()
        self.client = self._build()

    def state(self) -> dict[str, Any]:
        return self.client.get("/api/state").json()

    @contextmanager
    def failures(self, step: Step) -> Iterator[None]:
        """The step's injected failures, switched on for it alone."""
        self.engine.dead = step.engine_dies
        self.provider.stall = step.stall
        try:
            yield
        finally:
            self.engine.dead = False
            self.provider.unstall()

    def close(self) -> None:
        self.provider.unstall()
        self.client.close()


# --- the generator ---------------------------------------------------------------

# How often each kind of step is drawn. Commands dominate because the planner's
# batches are where features meet; the rest are the other ways a board moves.
_KINDS: tuple[tuple[str, int], ...] = (
    ("command", 10),
    ("literal_move", 3),
    ("literal_answer", 3),
    ("control_undo", 2),
    ("control_move", 2),
    ("control_new", 1),
    ("control_confirm", 1),
    ("retry", 1),
    ("restart", 1),
)
# How often a command, literal or drag carries an injected failure.
_FAILURE_RATE = 0.12
# The names a walk saves and resumes under: two, so a save can collide.
_SAVE_NAMES = ("alpha", "beta")


def _pick(rng: random.Random, weighted: Sequence[tuple[str, int]]) -> str:
    names = [name for name, _ in weighted]
    weights = [weight for _, weight in weighted]
    return rng.choices(names, weights=weights, k=1)[0]


def _planner_call(
    rng: random.Random, legal: Sequence[str]
) -> tuple[str, dict[str, Any]]:
    """One call a planner might emit, chosen off the menu it was shown."""
    options: list[tuple[str, int]] = [
        ("move", 6),
        ("illegal_move", 1),
        ("ask", 4),
        ("ask_off_menu", 1),
        ("undo", 3),
        ("read", 3),
        ("setting", 2),
        ("save", 1),
        ("resume", 1),
        ("destructive", 1),
    ]
    choice = _pick(rng, options)
    if choice == "move" and legal:
        return "make_move", {"move": rng.choice(list(legal))}
    if choice == "ask" and len(legal) >= 2:
        return "ask_player", {"candidates": rng.sample(list(legal), 2)}
    if choice == "ask_off_menu" and legal:
        return "ask_player", {"candidates": [rng.choice(list(legal)), _NEVER_LEGAL]}
    if choice == "undo":
        return "undo", ({} if rng.random() < 0.5 else {"plies": rng.choice([1, 2])})
    if choice == "read":
        return rng.choice(
            [
                ("describe_position", {}),
                ("evaluate_position", {}),
                ("get_best_moves", {"n": 2}),
            ]
        )
    if choice == "setting":
        return rng.choice(
            [
                ("set_verbosity", {"verbosity": rng.choice(["low", "normal", "high"])}),
                ("set_voice_output", {"enabled": rng.random() < 0.5}),
            ]
        )
    if choice == "save":
        return "save_game", {"name": rng.choice(_SAVE_NAMES)}
    if choice == "resume":
        return "resume_game", {"name": rng.choice(_SAVE_NAMES)}
    if choice == "destructive":
        return rng.choice([("new_game", {}), ("resign", {})])
    return "make_move", {"move": _NEVER_LEGAL}


@dataclass
class Memory:
    """What the walk remembers between steps: the last keyed delegate
    exchange (so it can be retried) and a counter for fresh keys."""

    last_keyed: Step | None = None
    keys: int = 0

    def fresh_key(self) -> str:
        self.keys += 1
        return f"k{self.keys}"


def next_step(rng: random.Random, state: dict[str, Any], memory: Memory) -> Step:
    """The walk's next step, drawn from the live state."""
    legal: list[str] = list(state["legal_moves"]) if not state["game_over"] else []
    kind = _pick(rng, _KINDS)
    origin = PANEL if rng.random() < 0.5 else rng.choice(DELEGATES)
    key = memory.fresh_key() if origin != PANEL and rng.random() < 0.6 else None
    failure = rng.random() < _FAILURE_RATE
    if kind == "command":
        rounds = []
        for _ in range(1 if rng.random() < 0.75 else 2):
            # Mostly small batches, now and then one past the per-turn cap on
            # tool calls, so the budget stop (#288) is met as well.
            size = rng.choice([1, 1, 2, 2, 3, 4, 10])
            rounds.append(tuple(_planner_call(rng, legal) for _ in range(size)))
        dies = rng.choice(["provider", "engine", "stall"]) if failure else None
        step = Step(
            kind="command",
            text="do the thing",
            origin=origin,
            rounds=tuple(rounds),
            key=key,
            provider_dies=rng.randrange(len(rounds) + 1)
            if dies == "provider"
            else None,
            engine_dies=dies == "engine",
            stall=dies == "stall",
        )
    elif kind == "literal_move" and legal:
        dies = rng.choice(["engine", "stall"]) if failure else None
        step = Step(
            kind="literal",
            text=rng.choice(legal),
            origin=origin,
            key=key,
            engine_dies=dies == "engine",
            stall=dies == "stall",
        )
    elif kind == "literal_answer":
        step = Step(
            kind="literal", text=rng.choice(["yes", "no"]), origin=origin, key=key
        )
    elif kind == "control_undo":
        return Step(kind="control", endpoint="/api/game/undo", body={})
    elif kind == "control_move" and legal:
        board = chess.Board(state["fen"])
        move = board.parse_san(rng.choice(legal))
        return Step(
            kind="control",
            endpoint="/api/game/move",
            body={"move": move.uci()},
            engine_dies=failure,
        )
    elif kind == "control_new":
        # The endpoint's own default rolls the colour; the walk rolls it off
        # its seed instead, so a replay gets the same side.
        color = rng.choice(["white", "black"])
        return Step(kind="control", endpoint="/api/game/new", body={"color": color})
    elif kind == "control_confirm":
        return Step(
            kind="control",
            endpoint="/api/game/confirm",
            body={"confirm": rng.random() < 0.7},
        )
    elif kind == "retry" and memory.last_keyed is not None:
        return replace(
            memory.last_keyed,
            kind="retry",
            provider_dies=None,
            engine_dies=False,
            stall=False,
        )
    elif kind == "restart":
        return Step(kind="restart")
    else:
        step = Step(kind="literal", text="no", origin=origin, key=key)
    if step.key is not None:
        memory.last_keyed = step
    return step


# --- running a step --------------------------------------------------------------


@dataclass(frozen=True)
class Pending:
    """The walk's model of the one armed confirmation (`ToolContext.pending`):
    which op, asked of which surface, about which board."""

    name: str
    origin: str
    version: int


@dataclass(frozen=True)
class Question:
    """The walk's model of one origin's open question (#319): the one the
    trace said a turn asked, keyed by the trace's own origin."""

    id: str
    candidates: tuple[str, ...]
    version: int
    game_id: str


# The walk cannot see what a step that died half-way armed: a delegate turn
# whose provider failed answers 502 with no results. Until the next turn
# clears the slot, the confirmation checks stand down rather than guess.
UNKNOWN = Pending(name="?", origin="?", version=-1)


@dataclass
class Observed:
    """What one step did, as the invariants read it."""

    step: Step
    before: dict[str, Any]
    after: dict[str, Any]
    status_code: int | None = None
    response: dict[str, Any] | None = None
    # The turn's tool calls, as `{"name", "result"}`, from either surface.
    results: list[dict[str, Any]] = field(default_factory=list)
    commentary: str | None = None
    provider_calls: list[dict[str, Any]] = field(default_factory=list)
    seconds: float = 0.0
    pending_before: Pending | None = None
    # The turn records this step wrote, and the walk's model of every
    # origin's open question before it ran (#319).
    turns: list[dict[str, Any]] = field(default_factory=list)
    questions_before: dict[str, Question] = field(default_factory=dict)
    # For a retry: the step it repeats, as it was first observed.
    original: Observed | None = None


def _delegate_results(response: dict[str, Any]) -> list[dict[str, Any]]:
    """A delegate exchange's tool calls in the panel's shape."""
    calls = (response.get("assistant_message") or {}).get("tool_calls") or []
    return [
        {
            "name": call["tool"],
            "result": (
                json.loads(call["result"])
                if call["result"] is not None
                else {"ok": False, "error": call["error"]}
            ),
        }
        for call in calls
    ]


def _script(step: Step) -> list[ChatResult | Exception]:
    turns: list[ChatResult | Exception] = [
        tool_calls_turn(*batch) for batch in step.rounds
    ]
    if step.provider_dies is not None:
        # Every call from that round on fails: the server went away.
        return [*turns[: step.provider_dies], ProviderError("injected: server gone")]
    return [*turns, text_turn(NEUTRAL)]


def run_step(
    harness: Harness,
    step: Step,
    pending: Pending | None = None,
    questions: dict[str, Question] | None = None,
) -> Observed:
    reset_rate_limit()
    before = harness.state()
    observed = Observed(
        step=step,
        before=before,
        after=before,
        pending_before=pending,
        questions_before=dict(questions or {}),
    )
    traced = len(harness.tracer.records)
    started = time.monotonic()
    if step.kind == "restart":
        harness.restart()
    elif step.kind in ("command", "literal", "retry"):
        # A retry must not run anything, so its script would be visible: a
        # planner call on a retry is a breach whatever it returns.
        harness.provider.rescript(*_script(step))
        with harness.failures(step):
            if step.origin == PANEL:
                response = harness.client.post("/api/command", json={"text": step.text})
            else:
                conversation = harness.conversations[step.origin]
                headers = {"Idempotency-Key": step.key} if step.key else {}
                response = harness.client.post(
                    _MESSAGES.format(id=conversation),
                    json={"content": step.text},
                    headers=headers,
                )
            observed.seconds = time.monotonic() - started
        observed.status_code = response.status_code
        observed.response = response.json()
        observed.provider_calls = list(harness.provider.calls)
        if response.status_code == 200:
            if step.origin == PANEL:
                observed.results = list(observed.response.get("tool_results") or [])
                observed.commentary = observed.response.get("commentary")
            else:
                observed.results = _delegate_results(observed.response)
                observed.commentary = observed.response["assistant_message"]["content"]
    else:
        with harness.failures(step):
            response = harness.client.post(step.endpoint, json=step.body)
            observed.seconds = time.monotonic() - started
        observed.status_code = response.status_code
        observed.response = response.json()
        observed.commentary = observed.response.get("commentary")
        observed.provider_calls = list(harness.provider.calls)
    observed.after = harness.state()
    observed.turns = harness.tracer.records[traced:]
    return observed


def next_questions(
    observed: Observed, questions: dict[str, Question]
) -> dict[str, Question]:
    """Every origin's open question after a step, as the trace reported it:
    closed or expired drops it, asked opens one, a restart forgets them all
    (`tools.live_checkpoint`). Built off the records so that
    `check_question_is_its_askers` can hold each later record to it."""
    if observed.step.kind == "restart":
        return {}
    questions = dict(questions)
    for record in observed.turns:
        told = record.get("clarification")
        if not told:
            continue
        origin = record["origin"]
        if told["expired"] or told["closed"]:
            questions.pop(origin, None)
        if (created := told["created"]) is not None:
            questions[origin] = Question(
                id=created["id"],
                candidates=tuple(created["candidates"]),
                version=created["board_version"],
                game_id=created["game_id"],
            )
    return questions


def next_pending(observed: Observed, pending: Pending | None) -> Pending | None:
    """The armed confirmation after a step, as the app keeps it: every turn
    from any surface disarms on its way in and may arm one of its own; a
    button arms or answers only for the panel; a board that moves leaves the
    op standing but stale; a restart forgets it (`tools.live_checkpoint`)."""
    step = observed.step
    if step.kind == "restart":
        return None
    if step.kind in ("command", "literal"):
        if observed.status_code not in (200, 502):
            return pending
        if observed.status_code == 502:
            return UNKNOWN
        armed = None
        for entry in observed.results:
            error = (entry.get("result") or {}).get("error") or ""
            if entry["name"] in _GATED and error.startswith(_ARMED):
                armed = Pending(entry["name"], step.origin, observed.after["version"])
                break
        return armed
    if step.endpoint == "/api/game/new":
        body = observed.response or {}
        if observed.status_code == 409 and body.get("confirm") is True:
            return Pending("new_game", PANEL, observed.after["version"])
        return None if observed.status_code == 200 else pending
    if step.endpoint == "/api/game/confirm":
        return None if observed.status_code == 200 else pending
    return pending


# --- the invariants ----------------------------------------------------------------


def _ok(result: dict[str, Any]) -> bool:
    """Whether one tool result is a success, in either of its shapes: a move is
    `legal`, everything else is `ok`."""
    if "legal" in result:
        return result["legal"] is True
    return result.get("ok") is True


def _game(state: dict[str, Any]) -> tuple[Any, ...]:
    return (state["game_id"], state["fen"], tuple(state["history"]))


def _owed(state: dict[str, Any]) -> bool:
    """The engine's reply is owed: the game is on and it is not the player's
    move."""
    return not state["game_over"] and state["turn"] != state["player_color"]


def check_status(observed: Observed) -> None:
    """No step may be a server error. A refusal (4xx) is an answer; a 5xx is a
    crash in whatever the combination reached — except the delegate wire's
    documented 502 for a provider that died under it."""
    code = observed.status_code
    if code is None or code < 500:
        return
    step = observed.step
    if code == 502 and step.provider_dies is not None and step.origin != PANEL:
        return
    raise InvariantBreach(f"server error {code}: {observed.response}")


def check_clarification_moves_nothing(observed: Observed) -> None:
    """#314: once an ask lands, nothing after it in the turn runs."""
    asked = False
    for entry in observed.results:
        result = entry.get("result") or {}
        if asked and _ok(result):
            raise InvariantBreach(
                f"{entry['name']} succeeded after a landed ask_player: {result}"
            )
        if entry["name"] == "ask_player" and _ok(result):
            asked = True


def _latest_board(messages: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    """The last board the planner was shown in one request: the opening block,
    or a refresh the loop appended since (merged into a user message or not)."""
    latest: dict[str, Any] | None = None
    for message in messages:
        content = message.get("content")
        if message.get("role") != "user" or not isinstance(content, str):
            continue
        found: list[tuple[int, dict[str, Any]]] = []
        for label in (_OPENING, _REFRESH):
            start = 0
            while (at := content.find(label, start)) != -1:
                try:
                    board, _ = json.JSONDecoder().raw_decode(content[at + len(label) :])
                except json.JSONDecodeError:
                    board = None
                if isinstance(board, dict) and "legal_moves" in board:
                    found.append((at, board))
                start = at + len(label)
        if found:
            latest = max(found, key=lambda item: item[0])[1]
    return latest


def check_question_is_its_askers(observed: Observed) -> None:
    """#319: an open question is read only by the origin it was asked in, and
    only while its game and board are the ones it was asked about. A stale
    one is reported expired, once; a live one never is; one is never lost
    without saying so; it is answered only by one of its own candidates; and
    a turn opens one exactly when its ask landed, on the board it ended on.
    """
    before, after = observed.before, observed.after
    for record in observed.turns:
        told = record.get("clarification")
        if not told:
            continue
        origin = record["origin"]
        known = observed.questions_before.get(origin)
        standing = known is not None and (known.version, known.game_id) == (
            before["version"],
            before["game_id"],
        )
        if told["open"] is not None:
            if known is None or known.id != told["open"]:
                raise InvariantBreach(
                    f"{origin} read question {told['open']} it was never asked"
                )
            if not standing:
                raise InvariantBreach(
                    f"{origin} read a stale question as open: {known}"
                )
        if told["expired"] is not None:
            if known is None or known.id != told["expired"]["id"]:
                raise InvariantBreach(
                    f"{origin} was told {told['expired']} expired, not its own"
                )
            if standing:
                raise InvariantBreach(f"{origin}'s live question read as expired")
        if known is not None and told["open"] is None and told["expired"] is None:
            raise InvariantBreach(f"{origin}'s question {known.id} vanished unread")
        closed = told["closed"]
        if closed is not None and closed["status"] == "answered":
            if known is None or closed["move"] not in known.candidates:
                raise InvariantBreach(
                    f"{closed['move']} answered a question that did not offer it: "
                    f"{known}"
                )
        created = told["created"]
        if observed.status_code not in (None, 200):
            continue
        asks = [
            entry["result"]["candidates"]
            for entry in observed.results
            if entry["name"] == "ask_player" and _ok(entry.get("result") or {})
        ]
        if bool(asks) != (created is not None):
            raise InvariantBreach(
                f"a landed ask and an opened question disagree: {asks} / {created}"
            )
        if created is not None and (
            created["candidates"] != asks[0]
            or created["board_version"] != after["version"]
            or created["game_id"] != after["game_id"]
        ):
            raise InvariantBreach(
                f"the question opened is not the one asked, on the board it ended "
                f"on: {created} vs {asks[0]} at {after['version']}"
            )


def check_offer_follows_board(observed: Observed) -> None:
    """#315: every planner request offers `ask_player` over exactly the menu of
    the last board it shows — never the board before an undo."""
    for index, call in enumerate(observed.provider_calls):
        tools = call.get("tools")
        if not tools:
            continue  # a narrator call: offered nothing
        ask = next((t for t in tools if t["function"]["name"] == "ask_player"), None)
        if ask is None:
            continue
        board = _latest_board(call["messages"])
        if board is None:
            raise InvariantBreach(f"planner call {index} shows no board")
        enum = ask["function"]["parameters"]["properties"]["candidates"]["items"][
            "enum"
        ]
        if sorted(enum) != sorted(board["legal_moves"]):
            raise InvariantBreach(
                f"planner call {index}: ask_player offers {sorted(enum)} over a "
                f"board whose legal_moves are {sorted(board['legal_moves'])}"
            )


def check_no_unexplained_mutation(observed: Observed) -> None:
    """A planner turn in which no board-changing tool succeeded leaves the game
    exactly where it was. (A turn that opens on an owed reply settles it first;
    `check_owed_reply_settled_once` owns that case.)"""
    if observed.step.kind != "command" or observed.status_code != 200:
        return
    if _owed(observed.before):
        return
    mutated = any(
        entry["name"] in _BOARD_MUTATORS and _ok(entry.get("result") or {})
        for entry in observed.results
    )
    if not mutated and _game(observed.before) != _game(observed.after):
        raise InvariantBreach(
            "the game changed with no board-changing tool succeeding: "
            f"{observed.before['history']} -> {observed.after['history']}"
        )


def check_board_is_coherent(observed: Observed) -> None:
    """The state document agrees with itself: the menu is the FEN's, and one
    position per ply."""
    state = observed.after
    board = chess.Board(state["fen"])
    legal = sorted(board.san(move) for move in board.legal_moves)
    if sorted(state["legal_moves"]) != legal and not state["game_over"]:
        raise InvariantBreach(f"legal_moves disagree with the FEN {state['fen']}")
    if len(state["fens"]) != len(state["history"]) + 1:
        raise InvariantBreach(
            f"{len(state['fens'])} positions for {len(state['history'])} plies"
        )


def check_no_reply_left_owed(observed: Observed) -> None:
    """At rest the player is to move: no step may leave the engine's reply
    uncollected, including a restart over a checkpoint taken mid-exchange.

    Two exceptions, both the documented recovery (#284): a step Stockfish died
    under keeps the player's committed move and owes the reply; and a step
    that changed nothing leaves an already-owed reply for the next mutation.
    """
    if not _owed(observed.after) or observed.step.engine_dies:
        return
    if _owed(observed.before) and _game(observed.before) == _game(observed.after):
        return
    raise InvariantBreach(
        f"the step ended with {observed.after['turn']} to move against a "
        f"{observed.after['player_color']} player: {observed.after['history']}"
    )


def check_owed_reply_settled_once(observed: Observed) -> None:
    """A reply owed from an earlier step is collected exactly once: the history
    it was owed on stands, and exactly one engine move is added before the
    step's own work (so the growth is odd — one reply, or one reply and one
    exchange)."""
    step = observed.step
    before, after = observed.before, observed.after
    if not _owed(before) or step.engine_dies or observed.status_code not in (200, None):
        return
    if step.kind not in ("command", "literal", "restart") and step.endpoint != (
        "/api/game/move"
    ):
        return
    reset = any(
        entry["name"] in _BOARD_MUTATORS - {"make_move"}
        and _ok(entry.get("result") or {})
        for entry in observed.results
    )
    if reset or before["game_id"] != after["game_id"]:
        return
    grown = len(after["history"]) - len(before["history"])
    if (
        after["history"][: len(before["history"])] != before["history"]
        or grown % 2 != 1
    ):
        raise InvariantBreach(
            f"an owed reply was not settled exactly once: {before['history']} -> "
            f"{after['history']}"
        )


def check_response_is_the_board(observed: Observed) -> None:
    """What a mutation answered is the board that stands afterwards."""
    response = observed.response or {}
    state = response.get("state")
    if observed.status_code != 200 or not isinstance(state, dict) or "fen" not in state:
        return
    if _game(state) != _game(observed.after):
        raise InvariantBreach(
            f"the response describes {state['history']} but the board holds "
            f"{observed.after['history']}"
        )


def check_restart_restores(observed: Observed) -> None:
    """A restart brings back the game that was on the board (#291) — settled,
    when it was checkpointed between the player's move and the reply."""
    if observed.step.kind != "restart":
        return
    before, after = observed.before, observed.after
    expected = before["history"]
    if _owed(before):
        if (
            after["history"][:-1] != expected
            or len(after["history"]) != len(expected) + 1
        ):
            raise InvariantBreach(
                f"restart did not settle the owed reply once: {expected} -> "
                f"{after['history']}"
            )
        return
    same = (before["fen"], expected) == (after["fen"], after["history"])
    if expected and before["game_id"] != after["game_id"]:
        # An untouched board has nothing to lose, and a fresh process names a
        # fresh game; a game in progress is the same game, restored.
        same = False
    if not same:
        raise InvariantBreach(
            f"restart lost the game: {expected} -> {after['history']}"
        )


def check_confirmation_is_answered_by_its_asker(observed: Observed) -> None:
    """#281: a destructive op runs on a yes only from the surface it was asked
    of, about the board it was asked about — and then it does run, once. A no
    never runs one, and neither does a yes with nothing of its own to answer.
    """
    step = observed.step
    pending = observed.pending_before
    if pending is UNKNOWN:
        return
    before = observed.before

    def live(origin: str) -> bool:
        return (
            pending is not None
            and pending.origin == origin
            and pending.version == before["version"]
        )

    if step.kind == "literal" and step.text in ("yes", "no"):
        if observed.status_code != 200:
            return
        ran = [
            entry["name"]
            for entry in observed.results
            if entry["name"] in _GATED and _ok(entry.get("result") or {})
        ]
        if step.text == "yes" and live(step.origin):
            assert pending is not None
            if ran != [pending.name]:
                raise InvariantBreach(
                    f"a yes from {step.origin} to its own live {pending.name} ran {ran}"
                )
        elif ran:
            raise InvariantBreach(
                f"{step.text!r} from {step.origin} ran {ran}; armed was {pending}"
            )
    if step.endpoint == "/api/game/confirm":
        if live(PANEL) and observed.status_code != 200:
            raise InvariantBreach(
                f"the panel's own live question refused: {observed.response}"
            )
        if not live(PANEL):
            if observed.status_code == 200:
                raise InvariantBreach(f"the button answered {pending}, not the panel's")
            if _game(before) != _game(observed.after):
                raise InvariantBreach("a refused confirm moved the board")


def check_retry_never_acts_twice(observed: Observed) -> None:
    """#291: a keyed exchange sent again is answered from the store — the same
    answer for a finished one, 409 for one that never finished — and runs
    nothing either way."""
    if observed.step.kind != "retry":
        return
    original = observed.original
    if observed.provider_calls:
        raise InvariantBreach(
            f"a retry reached the model {len(observed.provider_calls)}x"
        )
    if _game(observed.before) != _game(observed.after):
        raise InvariantBreach(
            f"a retry moved the board: {observed.before['history']} -> "
            f"{observed.after['history']}"
        )
    if original is None:
        return
    if original.status_code == 200:
        if observed.status_code != 200 or observed.response != original.response:
            raise InvariantBreach(
                f"a finished exchange replayed as {observed.status_code}, not as "
                "its first answer"
            )
    elif observed.status_code != 409:
        raise InvariantBreach(
            f"an unfinished exchange ({original.status_code}) replayed as "
            f"{observed.status_code}, not 409"
        )


def check_late_words_land_nowhere(observed: Observed) -> None:
    """#316: what a stalled narrator writes after its turn moved on is never
    said, and never remembered into a later request."""
    if observed.commentary and LATE_WORDS in observed.commentary:
        raise InvariantBreach(f"the late words were said: {observed.commentary!r}")
    for index, call in enumerate(observed.provider_calls):
        for message in call["messages"]:
            if LATE_WORDS in str(message.get("content")):
                raise InvariantBreach(f"request {index} remembers the late words")


def check_stall_is_bounded(observed: Observed) -> None:
    """#316/#283: a narrator that stalls holds the turn for its budget, not for
    as long as it likes."""
    if observed.step.stall and observed.seconds > BUDGET_S + STALL_SLACK_S:
        raise InvariantBreach(
            f"a stalled narrator held the step {observed.seconds:.1f}s against a "
            f"{BUDGET_S}s budget"
        )


def check_budgets_cap_the_turn(observed: Observed) -> None:
    """#288: however long the batch, at most the turn's budget of calls runs,
    and at most its budget of Stockfish searches."""
    ran = [e for e in observed.results if _ok(e.get("result") or {})]
    searches = [e for e in ran if e["name"] in _EXPENSIVE_TOOLS]
    if (
        len(ran) > _DEFAULT_MAX_TOOL_CALLS
        or len(searches) > _DEFAULT_MAX_ANALYSIS_CALLS
    ):
        raise InvariantBreach(
            f"{len(ran)} calls and {len(searches)} searches ran in one turn"
        )


INVARIANTS: tuple[Callable[[Observed], None], ...] = (
    check_status,
    check_clarification_moves_nothing,
    check_question_is_its_askers,
    check_offer_follows_board,
    check_no_unexplained_mutation,
    check_board_is_coherent,
    check_no_reply_left_owed,
    check_owed_reply_settled_once,
    check_response_is_the_board,
    check_restart_restores,
    check_confirmation_is_answered_by_its_asker,
    check_retry_never_acts_twice,
    check_late_words_land_nowhere,
    check_stall_is_bounded,
    check_budgets_cap_the_turn,
)


# --- a whole walk --------------------------------------------------------------------


def replay_command(seed: int, length: int) -> str:
    return (
        f"CHESSAPP_TRAJ_SEED={seed} CHESSAPP_TRAJ_LENGTH={length} "
        "pytest tests/test_trajectories.py -k replay -s"
    )


class _Run:
    """One walk's bookkeeping: the harness, the confirmation model, the keyed
    exchanges already seen (for retries), and the log."""

    def __init__(self, harness: Harness) -> None:
        self.harness = harness
        self.pending: Pending | None = None
        self.questions: dict[str, Question] = {}
        self.keyed: dict[tuple[str, str], Observed] = {}
        self.log: list[Observed] = []

    def step(self, step: Step) -> Observed:
        observed = run_step(self.harness, step, self.pending, self.questions)
        if step.key is not None:
            where = (step.origin, step.key)
            if step.kind == "retry":
                observed.original = self.keyed.get(where)
            else:
                self.keyed[where] = observed
        self.log.append(observed)
        return observed

    def settle(self, observed: Observed) -> None:
        self.pending = next_pending(observed, self.pending)
        self.questions = next_questions(observed, self.questions)


def _check(run: _Run, observed: Observed, seed: int | None, length: int) -> None:
    for invariant in INVARIANTS:
        try:
            invariant(observed)
        except InvariantBreach as breach:
            raise InvariantBreach(
                _report(seed, length, run.log, invariant, breach)
            ) from breach
    run.settle(observed)


def walk(seed: int, length: int, save_dir: Path) -> list[Observed]:
    """Run one seeded walk, checking every invariant after every step.

    One walk in four starts from the 84-ply fixture instead of the opening, so
    undo, save and resume meet a real game's history. A breach is re-raised
    with the seed, the steps up to and including the failing one, and the
    command that replays it — everything needed to turn the failure into a
    named regression.
    """
    rng = random.Random(seed)
    start = late_game_session() if rng.random() < 0.25 else None
    harness = Harness(save_dir, start=start)
    run = _Run(harness)
    memory = Memory()
    try:
        for _ in range(length):
            step = next_step(rng, harness.state(), memory)
            _check(run, run.step(step), seed, length)
    finally:
        harness.close()
    return run.log


def run_steps(
    steps: Sequence[Step], save_dir: Path, *, start: GameSession | None = None
) -> list[Observed]:
    """Run a fixed sequence (a named regression) under the same invariants."""
    harness = Harness(save_dir, start=start)
    run = _Run(harness)
    try:
        for step in steps:
            _check(run, run.step(step), None, len(steps))
    finally:
        harness.close()
    return run.log


def _report(
    seed: int | None,
    length: int,
    log: Sequence[Observed],
    invariant: Callable[[Observed], None],
    breach: InvariantBreach,
) -> str:
    lines = [f"{invariant.__name__}: {breach}"]
    if seed is not None:
        lines.append(f"seed={seed} length={length}")
        lines.append(f"replay: {replay_command(seed, length)}")
    lines.append(
        f"start: {log[0].before['fen']} ({len(log[0].before['history'])} plies)"
    )
    for index, observed in enumerate(log):
        lines.append(
            f"  {index:>2}. {observed.step.describe()}  -> "
            f"{observed.status_code} {observed.after['history'][-4:]}"
        )
    return "\n".join(lines)
