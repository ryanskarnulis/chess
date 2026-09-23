"""Seeded, composed interaction trajectories over the shipped app (#318).

Every other deterministic test here drives one feature: one command, one tool,
one gate. The three bugs #318 was filed over lived *between* features — an ask
batched with a move (#314), an undo refreshing the board while the offer stayed
on the old one (#315), a closer outliving its budget (#316) — and each of them
passed a suite in which every feature, alone, was green.

So this drives *sequences*: a seeded random walk over what a player and a
planner can do to one game, through the real app (`app.build_app`, the same
assembly that ships, with a `ScriptedProvider` standing in for llama-server and
a deterministic engine for Stockfish), and after every step checks invariants
that must hold however the steps were combined. The model is scripted, so a
walk is fully determined by its seed: a failure is a seed, and the seed is the
reproduction.

Nothing here reads language. The scripted planner's batches are chosen off the
live `legal_moves` (the menu a real planner is handed), the narrator says a
fixed neutral line, and every check is over the board, the tool results, the
requests the provider was sent, and the save directory.
"""

from __future__ import annotations

import json
import random
import zlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import chess
from fastapi.testclient import TestClient

from chessapp.agent_api import reset_rate_limit
from chessapp.app import build_app
from chessapp.engine import CandidateMove, Evaluation
from chessapp.game import GameSession
from fakes import ScriptedProvider, text_turn, tool_calls_turn

# What the scripted narrator (and a planner with nothing left to do) says: no
# move, no verdict, no ending — nothing the honesty guard could hold against a
# board, so a guard firing on it would be the guard's bug, not the script's.
NEUTRAL = "Okay."

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

# A move no position has: SAN-shaped, on a square that does not exist.
_NEVER_LEGAL = "Qz9"


class LegalEngine:
    """Engine double that always answers with a legal move.

    `FakeEngine` replies one fixed UCI, which is illegal in almost every
    position a walk reaches. This picks among the legal replies by a hash of
    the position, so the reply varies across a game and is still a pure
    function of the board — the walk's determinism rests on it.
    """

    def __init__(self) -> None:
        self.closed = False

    def choose_move(self, session: GameSession) -> str:
        board = chess.Board(session.fen())
        moves = sorted(move.uci() for move in board.legal_moves)
        return moves[zlib.crc32(board.fen().encode()) % len(moves)]

    def play_move(self, session: GameSession):
        return session.submit_move(self.choose_move(session))

    def get_best_moves(self, session: GameSession, n: int = 3):
        board = chess.Board(session.fen())
        moves = sorted(board.legal_moves, key=lambda move: move.uci())[:n]
        return [
            CandidateMove(uci=move.uci(), san=board.san(move), score_cp=0, mate_in=None)
            for move in moves
        ]

    def evaluate_position(self, session: GameSession) -> Evaluation:
        return Evaluation(score_cp=0, mate_in=None)

    def close(self) -> None:
        self.closed = True

    def set_skill_level(self, level: int) -> None:
        pass

    def set_elo(self, elo: int) -> None:
        pass

    def set_tier(self, tier: str) -> None:
        pass


@dataclass(frozen=True)
class Step:
    """One thing that happens to the game.

    `kind` is the seam: `command` (the panel, through the planner), `literal`
    (the panel, a text the fast paths settle — a SAN move, "yes", "no"),
    `control` (a board endpoint), or `restart` (a new process over the same
    save directory). `rounds` is the scripted planner: one tuple of calls per
    planner round trip, several calls in one round being one model response.
    """

    kind: str
    text: str = ""
    rounds: tuple[tuple[tuple[str, dict[str, Any]], ...], ...] = ()
    endpoint: str = ""
    body: dict[str, Any] = field(default_factory=dict)

    def describe(self) -> str:
        if self.kind == "command":
            rounds = " | ".join(
                ", ".join(f"{name}({_args(args)})" for name, args in batch)
                for batch in self.rounds
            )
            return f"command {self.text!r}: {rounds}"
        if self.kind == "literal":
            return f"literal {self.text!r}"
        if self.kind == "control":
            return f"control {self.endpoint} {self.body}"
        return self.kind


def _args(args: dict[str, Any]) -> str:
    return ", ".join(f"{key}={value!r}" for key, value in args.items())


class InvariantBreach(AssertionError):
    """An invariant failed. Carries the whole reproduction in its message."""


# --- the app under the walk ----------------------------------------------------


class Harness:
    """The shipped app over a scripted provider, restartable in place."""

    def __init__(self, save_dir: Path) -> None:
        self.save_dir = save_dir
        save_dir.mkdir(parents=True, exist_ok=True)
        self.provider = ScriptedProvider(text_turn(NEUTRAL))
        self.engine = LegalEngine()
        self.client = self._build()

    def _build(self) -> TestClient:
        app = build_app(
            provider=self.provider, engine=self.engine, save_dir=self.save_dir
        )
        return TestClient(app)

    def restart(self) -> None:
        """A new process on the same disk: the old client is closed and the app
        is assembled again, restoring whatever `live.json` holds."""
        self.client.close()
        self.client = self._build()

    def state(self) -> dict[str, Any]:
        return self.client.get("/api/state").json()

    def close(self) -> None:
        self.client.close()


# --- the generator ---------------------------------------------------------------

# How often each kind of step is drawn. Commands dominate because the planner's
# batches are where features meet; the rest are the other ways a board moves.
_KINDS: tuple[tuple[str, int], ...] = (
    ("command", 10),
    ("literal_move", 3),
    ("literal_answer", 2),
    ("control_undo", 2),
    ("control_move", 2),
    ("control_new", 1),
    ("restart", 1),
)


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
    if choice == "destructive":
        return rng.choice([("new_game", {}), ("resign", {})])
    return "make_move", {"move": _NEVER_LEGAL}


def next_step(rng: random.Random, state: dict[str, Any]) -> Step:
    """The walk's next step, drawn from the live state."""
    legal: list[str] = list(state["legal_moves"]) if not state["game_over"] else []
    kind = _pick(rng, _KINDS)
    if kind == "command":
        rounds = []
        for _ in range(1 if rng.random() < 0.75 else 2):
            size = rng.choice([1, 1, 2, 2, 3, 4])
            rounds.append(tuple(_planner_call(rng, legal) for _ in range(size)))
        return Step(kind="command", text="do the thing", rounds=tuple(rounds))
    if kind == "literal_move" and legal:
        return Step(kind="literal", text=rng.choice(legal))
    if kind == "literal_answer":
        return Step(kind="literal", text=rng.choice(["yes", "no"]))
    if kind == "control_undo":
        return Step(kind="control", endpoint="/api/game/undo", body={})
    if kind == "control_move" and legal:
        board = chess.Board(state["fen"])
        move = board.parse_san(rng.choice(legal))
        return Step(
            kind="control", endpoint="/api/game/move", body={"move": move.uci()}
        )
    if kind == "control_new":
        # The endpoint's own default rolls the colour; the walk rolls it off
        # its seed instead, so a replay gets the same side.
        color = rng.choice(["white", "black"])
        return Step(kind="control", endpoint="/api/game/new", body={"color": color})
    if kind == "restart":
        return Step(kind="restart")
    return Step(kind="literal", text="no")


# --- running a step --------------------------------------------------------------


@dataclass
class Observed:
    """What one step did, as the invariants read it."""

    step: Step
    before: dict[str, Any]
    after: dict[str, Any]
    status_code: int | None = None
    response: dict[str, Any] | None = None
    provider_calls: list[dict[str, Any]] = field(default_factory=list)


def run_step(harness: Harness, step: Step) -> Observed:
    reset_rate_limit()
    before = harness.state()
    observed = Observed(step=step, before=before, after=before)
    if step.kind == "restart":
        harness.restart()
    elif step.kind in ("command", "literal"):
        turns = [tool_calls_turn(*batch) for batch in step.rounds]
        harness.provider.rescript(*turns, text_turn(NEUTRAL))
        response = harness.client.post("/api/command", json={"text": step.text})
        observed.status_code = response.status_code
        observed.response = response.json()
        observed.provider_calls = list(harness.provider.calls)
    else:
        response = harness.client.post(step.endpoint, json=step.body)
        observed.status_code = response.status_code
        observed.response = response.json()
    observed.after = harness.state()
    return observed


# --- the invariants ----------------------------------------------------------------


def _ok(result: dict[str, Any]) -> bool:
    """Whether one tool result is a success, in either of its shapes: a move is
    `legal`, everything else is `ok`."""
    if "legal" in result:
        return result["legal"] is True
    return result.get("ok") is True


def _results(observed: Observed) -> list[dict[str, Any]]:
    if not observed.response:
        return []
    return list(observed.response.get("tool_results") or [])


def _game(state: dict[str, Any]) -> tuple[Any, ...]:
    return (state["game_id"], state["fen"], tuple(state["history"]))


def check_status(observed: Observed) -> None:
    """No step may be a server error. A refusal (4xx) is an answer; a 5xx is a
    crash in whatever the combination reached."""
    if observed.status_code is not None and observed.status_code >= 500:
        raise InvariantBreach(
            f"server error {observed.status_code}: {observed.response}"
        )


def check_clarification_moves_nothing(observed: Observed) -> None:
    """#314: once an ask lands, nothing after it in the turn runs."""
    asked = False
    for entry in _results(observed):
        result = entry.get("result") or {}
        if asked and _ok(result):
            raise InvariantBreach(
                f"{entry['name']} succeeded after a landed ask_player: {result}"
            )
        if entry["name"] == "ask_player" and _ok(result):
            asked = True


def _latest_board(messages: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
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
    exactly where it was."""
    if observed.step.kind != "command" or observed.status_code != 200:
        return
    mutated = any(
        entry["name"] in _BOARD_MUTATORS and _ok(entry.get("result") or {})
        for entry in _results(observed)
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
    uncollected, including a restart over a checkpoint taken mid-exchange."""
    state = observed.after
    if state["game_over"]:
        return
    if state["turn"] != state["player_color"]:
        raise InvariantBreach(
            f"the step ended with {state['turn']} to move against a "
            f"{state['player_color']} player: {state['history']}"
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
    """A restart brings back the game that was on the board (#291)."""
    if observed.step.kind != "restart":
        return
    before, after = _game(observed.before), _game(observed.after)
    if not observed.before["history"]:
        # An untouched board has nothing to lose, and a fresh process names a
        # fresh game: only the position must come back.
        before, after = before[1:], after[1:]
    if before != after:
        raise InvariantBreach(
            f"restart lost the game: {observed.before['history']} -> "
            f"{observed.after['history']}"
        )


INVARIANTS: tuple[Callable[[Observed], None], ...] = (
    check_status,
    check_clarification_moves_nothing,
    check_offer_follows_board,
    check_no_unexplained_mutation,
    check_board_is_coherent,
    check_no_reply_left_owed,
    check_response_is_the_board,
    check_restart_restores,
)


# --- a whole walk --------------------------------------------------------------------


def replay_command(seed: int, length: int) -> str:
    return (
        f"CHESSAPP_TRAJ_SEED={seed} CHESSAPP_TRAJ_LENGTH={length} "
        "pytest tests/test_trajectories.py -k replay -s"
    )


def walk(seed: int, length: int, save_dir: Path) -> list[Observed]:
    """Run one seeded walk, checking every invariant after every step.

    A breach is re-raised with the seed, the steps up to and including the
    failing one, and the command that replays it — everything needed to turn
    the failure into a named regression.
    """
    rng = random.Random(seed)
    harness = Harness(save_dir)
    log: list[Observed] = []
    try:
        for _ in range(length):
            step = next_step(rng, harness.state())
            observed = run_step(harness, step)
            log.append(observed)
            for invariant in INVARIANTS:
                try:
                    invariant(observed)
                except InvariantBreach as breach:
                    raise InvariantBreach(
                        _report(seed, length, log, invariant, breach)
                    ) from breach
    finally:
        harness.close()
    return log


def run_steps(steps: Sequence[Step], save_dir: Path) -> list[Observed]:
    """Run a fixed sequence (a named regression) under the same invariants."""
    harness = Harness(save_dir)
    log: list[Observed] = []
    try:
        for step in steps:
            observed = run_step(harness, step)
            log.append(observed)
            for invariant in INVARIANTS:
                try:
                    invariant(observed)
                except InvariantBreach as breach:
                    raise InvariantBreach(
                        _report(None, len(steps), log, invariant, breach)
                    ) from breach
    finally:
        harness.close()
    return log


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
    lines.append(f"start: {log[0].before['fen']}")
    for index, observed in enumerate(log):
        lines.append(
            f"  {index:>2}. {observed.step.describe()}  -> "
            f"{observed.status_code} {observed.after['history'][-4:]}"
        )
    return "\n".join(lines)
