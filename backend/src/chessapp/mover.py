"""The mover: Glitch choosing his own move (#472, docs/glitch-difficulty.md).

In the `glitch` difficulty tier the model picks Glitch's move and no engine
helps it. This module is that pick: one model phase, `mover`, beside the
brain's planner, narrator and answer reader, with one tool, `play_move(move,
intent)`. It is a move *source* (`coordinator.MoveSource`), not a step in the
planner's loop, so no turn's language reaches it and the planner cannot ask
for, skip or steer it.

**What it sees is the board and nothing else** (decision 2). The game's
moves, the position as a FEN and a readable board, which side it plays, and
the legal moves. Never the conversation, never anything Stockfish computed,
never the knowledge notes or the opening's name. That is enforced by what it
is given: `Mover.choose` takes a `GameSession` and reads only its board, so
there is nothing else in reach to leak. `tests/test_mover_bytes.py` pins the
exact request bytes per position, so nothing can creep in later unnoticed.

**The prompt is append-only per game** (#362): fixed instructions and the side
first, then the moves, which only ever grow at the end, then the position.
Each call shares the previous one's prefix up to the newest moves.

**Code owns the move, the model owns the choice.** `play_move`'s `move` is
checked against the position by code. One that isn't legal is refused with
the reason, and the model gets `MAX_ATTEMPTS` replies in all to get it right.
The mover never touches the session: it returns a UCI string and the
coordinator submits it through the session's legality gate (#471). `intent` is
the model's one line about what the move is for. Code checks only that there
is one, and never cuts or rewrites its words (CLAUDE.md). It is a final
answer, not a thought, so it may reach the narrator. Thinking is never fed
back into history.

**Running out is a failure the caller sees, never a move.** Its own deadline
and token cap come from the profile's `[phases.mover]`. Passing either,
exhausting the attempts, or a provider that is down raises `MoverFailed` with
its kind, and the caller (#473) decides what plays instead.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import chess
from pydantic import BaseModel, ValidationError

from chessapp.brain import (
    CALL_BAD_ARGS,
    CALL_FAILED,
    CALL_OK,
    CALL_TRUNCATED,
    PHASE_MOVER,
    ModelCall,
    ServerStamp,
)
from chessapp.context_capture import model_phase
from chessapp.game import GameSession
from chessapp.profiles import MOVER, ModelProfile
from chessapp.provider import (
    ChatProvider,
    ChatResult,
    ProviderError,
    ToolCallArgumentsError,
)

# How many replies the model gets to play a legal move: the first, and two
# more after a refusal. A small budget on purpose: each reply is a thinking
# call on the shared GPU, and a model that misses three times is not about to
# find it on the fourth.
MAX_ATTEMPTS = 3

PLAY_MOVE = "play_move"

# An attempt's verdict.
PLAYED = "played"
ILLEGAL = "illegal"
NO_CALL = "no_call"
BAD_ARGS = "bad_args"

_STANDARD_START = chess.STARTING_FEN

_SYSTEM = """\
You are playing a game of chess as {side}. Choose your next move.

You are shown the moves so far, the position (as a FEN and as a board), and \
every legal move. Pick one of the legal moves and play it by calling \
play_move once:
- move: the move in SAN, as the legal-move list writes it.
- intent: one short line in your own words saying what the move is for, \
like "developing, eyeing f7".

The choice is yours alone."""

PLAY_MOVE_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": PLAY_MOVE,
        "description": "Play your move. Call it once, with one of the legal moves.",
        "parameters": {
            "type": "object",
            "properties": {
                "move": {
                    "type": "string",
                    "description": "One of the legal moves, in SAN (e.g. Nf3).",
                },
                "intent": {
                    "type": "string",
                    "description": "One short line: what the move is for.",
                },
            },
            "required": ["move", "intent"],
        },
    },
}


class _PlayMoveArgs(BaseModel):
    move: str
    intent: str


class MoverFailureKind(StrEnum):
    """Why the mover returned no move. Each is the caller's to handle (#473
    plays Stockfish at `beginner`); none of them is ever a move."""

    #: Every attempt was illegal, made no call, or had bad arguments.
    RETRIES = "retries"
    #: The run's `deadline_s` passed.
    DEADLINE = "deadline"
    #: A reply hit `max_tokens` without calling the tool.
    TRUNCATED = "truncated"
    #: The provider failed, inside the deadline.
    PROVIDER = "provider"


@dataclass(frozen=True)
class MoverAttempt:
    """One `play_move` call, or one reply that made none."""

    verdict: str
    move: str | None = None
    intent: str | None = None
    reason: str = ""

    def as_trace(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "move": self.move,
            "intent": self.intent,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class MoverChoice:
    """The move the mover chose, legal by code's check, with its intent."""

    uci: str
    san: str
    intent: str
    model: str | None
    attempts: tuple[MoverAttempt, ...]
    calls: tuple[ModelCall, ...]

    def as_trace(self) -> dict[str, Any]:
        """The turn record's `mover` field (`trace.turn_record`). The model
        calls go in the record's `calls`, like every phase's."""
        return {
            "model": self.model,
            "outcome": PLAYED,
            "uci": self.uci,
            "san": self.san,
            "intent": self.intent,
            "attempts": [attempt.as_trace() for attempt in self.attempts],
        }


class MoverFailed(Exception):
    """The mover returned no move. `kind` says why, and the attempts and calls
    it made are kept for the trace."""

    def __init__(
        self,
        kind: MoverFailureKind,
        detail: str,
        *,
        model: str | None,
        attempts: tuple[MoverAttempt, ...],
        calls: tuple[ModelCall, ...],
    ) -> None:
        super().__init__(f"mover {kind}: {detail}")
        self.kind = kind
        self.detail = detail
        self.model = model
        self.attempts = attempts
        self.calls = calls

    def as_trace(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "outcome": str(self.kind),
            "detail": self.detail,
            "uci": None,
            "san": None,
            "intent": None,
            "attempts": [attempt.as_trace() for attempt in self.attempts],
        }


class Mover:
    """Asks the model for the side to move's move. Holds no game state: each
    `choose` reads the session it is handed, so one mover serves every game."""

    def __init__(
        self,
        provider: ChatProvider,
        profile: ModelProfile,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.provider = provider
        self.settings = profile.phase(MOVER)
        self.temperature = profile.temperature_for(MOVER)
        self.clock = clock

    @property
    def model(self) -> str | None:
        """The model the mover's calls go to: the phase's route when the
        provider routes phases, the provider's own otherwise."""
        route = getattr(self.provider, "provider_for", None)
        routed = route(MOVER) if route is not None else self.provider
        model = getattr(routed, "model", None)
        return model if isinstance(model, str) else None

    def choose(self, session: GameSession) -> MoverChoice:
        """The move for the side to move in `session`, or `MoverFailed`.
        Never changes `session`."""
        board = chess.Board(session.fen())
        if session.is_game_over() or not any(board.legal_moves):
            raise ValueError("the mover was asked for a move in a finished game")
        run = _Run(self, session, board)
        return run.play()


class _Run:
    """One `choose`: its messages, attempts, calls and clock."""

    def __init__(self, mover: Mover, session: GameSession, board: chess.Board):
        self.mover = mover
        self.session = session
        self.board = board
        self.messages = mover_messages(session)
        self.attempts: list[MoverAttempt] = []
        self.calls: list[ModelCall] = []
        self.started = mover.clock()
        deadline_s = mover.settings.deadline_s
        self.deadline = None if deadline_s is None else self.started + deadline_s

    def play(self) -> MoverChoice:
        for _ in range(MAX_ATTEMPTS):
            result = self._call()
            if result is None:
                continue
            choice = self._read(result)
            if choice is not None:
                return choice
        raise self._failed(
            MoverFailureKind.RETRIES, f"no legal move in {MAX_ATTEMPTS} replies"
        )

    def _remaining(self) -> float | None:
        if self.deadline is None:
            return None
        return self.deadline - self.mover.clock()

    def _call(self) -> ChatResult | None:
        """One round trip. `None` after a reply whose arguments were not
        JSON, which is an attempt the model gets to correct."""
        remaining = self._remaining()
        if remaining is not None and remaining <= 0:
            raise self._failed(MoverFailureKind.DEADLINE, "deadline passed")
        settings = self.mover.settings
        started = self.mover.clock()
        try:
            with model_phase(PHASE_MOVER):
                result = self.mover.provider.chat(
                    self.messages,
                    tools=[PLAY_MOVE_TOOL],
                    enable_thinking=settings.thinking,
                    max_tokens=settings.max_tokens,
                    temperature=self.mover.temperature,
                    timeout=remaining,
                    thinking_budget=settings.thinking_budget,
                )
        except ToolCallArgumentsError as exc:
            self._count(started, CALL_BAD_ARGS, remaining, failure=exc.failure)
            self.attempts.append(MoverAttempt(BAD_ARGS, reason=str(exc)))
            self.messages.append(
                {
                    "role": "user",
                    "content": (
                        f"Your play_move call failed before it ran: {exc}. "
                        "Call play_move again with JSON arguments."
                    ),
                }
            )
            return None
        except ProviderError as exc:
            self._count(started, CALL_FAILED, remaining, failure=exc.failure)
            left = self._remaining()
            if left is not None and left <= 0:
                raise self._failed(MoverFailureKind.DEADLINE, str(exc)) from exc
            raise self._failed(MoverFailureKind.PROVIDER, str(exc)) from exc
        truncated = result.finish_reason == "length"
        self._count(
            started,
            CALL_TRUNCATED if truncated else CALL_OK,
            remaining,
            result=result,
        )
        if truncated and not result.tool_calls:
            raise self._failed(
                MoverFailureKind.TRUNCATED,
                f"reply hit max_tokens ({settings.max_tokens}) without a move",
            )
        return result

    def _read(self, result: ChatResult) -> MoverChoice | None:
        """The move in `result`, or `None` after telling the model why there
        was none: the history gains its reply (final answer only, thinking
        never round-trips) and one answer per call."""
        self.messages.append(result.to_message())
        if not result.tool_calls:
            self.attempts.append(MoverAttempt(NO_CALL, intent=result.content))
            self.messages.append(
                {
                    "role": "user",
                    "content": "Play your move by calling play_move.",
                }
            )
            return None
        chosen: MoverChoice | None = None
        for call in result.tool_calls:
            if chosen is not None:
                answer = {"ok": False, "error": "your move is already played"}
            else:
                chosen, answer = self._judge(call.name, call.arguments)
            self.messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": json.dumps(answer),
                }
            )
        return chosen

    def _judge(
        self, name: str, arguments: dict[str, Any]
    ) -> tuple[MoverChoice | None, dict[str, Any]]:
        """One call, checked by code: the choice if it is a legal move with an
        intent, else the refusal the model reads."""
        if name != PLAY_MOVE:
            reason = f"there is no tool {name!r}; call play_move"
            self.attempts.append(MoverAttempt(BAD_ARGS, reason=reason))
            return None, _refusal(reason)
        try:
            args = _PlayMoveArgs.model_validate(arguments)
        except ValidationError as exc:
            reason = "play_move needs a move and an intent, both strings: " + (
                "; ".join(
                    f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}"
                    for err in exc.errors()
                )
            )
            self.attempts.append(MoverAttempt(BAD_ARGS, reason=reason))
            return None, _refusal(reason)
        intent = args.intent.strip()
        move, reason = self._legal(args.move.strip())
        if move is None:
            self.attempts.append(
                MoverAttempt(ILLEGAL, move=args.move, intent=intent, reason=reason)
            )
            return None, _refusal(reason, "play one of the legal moves listed")
        if not intent:
            reason = "intent is empty: say in one short line what the move is for"
            self.attempts.append(MoverAttempt(BAD_ARGS, move=args.move, reason=reason))
            return None, _refusal(reason)
        san = self.board.san(move)
        self.attempts.append(MoverAttempt(PLAYED, move=args.move, intent=intent))
        choice = MoverChoice(
            uci=move.uci(),
            san=san,
            intent=intent,
            model=self.mover.model,
            attempts=tuple(self.attempts),
            calls=tuple(self.calls),
        )
        return choice, {"ok": True, "played": san}

    def _legal(self, text: str) -> tuple[chess.Move | None, str]:
        """`text` as a legal move here (SAN, then UCI, the order
        `GameSession` parses in), or why it is not one."""
        try:
            return self.board.parse_san(text), ""
        except chess.AmbiguousMoveError:
            options = self.session.move_alternatives(text)
            return None, (
                f"{text} is ambiguous: more than one piece can play it"
                + (f" (one of {', '.join(options)})" if options else "")
            )
        except ValueError:
            pass
        try:
            move = chess.Move.from_uci(text)
        except ValueError:
            move = None
        if move is not None and move in self.board.legal_moves:
            return move, ""
        options = self.session.move_alternatives(text)
        return None, (
            f"{text} is not a legal move in this position"
            + (f"; legal moves like it: {', '.join(options)}" if options else "")
        )

    def _count(
        self,
        started: float,
        status: str,
        budget: float | None,
        *,
        result: ChatResult | None = None,
        failure: str = "",
    ) -> None:
        usage = result.usage if result is not None else None
        meta = result.server if result is not None else None
        self.calls.append(
            ModelCall(
                PHASE_MOVER,
                status,
                max(0, round((self.mover.clock() - started) * 1000)),
                usage.prompt_tokens if usage is not None else None,
                usage.completion_tokens if usage is not None else None,
                failure=str(failure),
                budget_ms=None if budget is None else round(budget * 1000),
                server=None
                if meta is None
                else ServerStamp(
                    fingerprint=meta.fingerprint,
                    server_ms=meta.server_ms,
                    cached_tokens=meta.cached_tokens,
                ),
            )
        )

    def _failed(self, kind: MoverFailureKind, detail: str) -> MoverFailed:
        return MoverFailed(
            kind,
            detail,
            model=self.mover.model,
            attempts=tuple(self.attempts),
            calls=tuple(self.calls),
        )


def mover_messages(session: GameSession) -> list[dict[str, Any]]:
    """The mover's opening request for the side to move in `session`: the
    fixed instructions and side, then the moves, then the position. Built
    from the board alone (module docstring)."""
    board = chess.Board(session.fen())
    side = session.turn
    return [
        {"role": "system", "content": _SYSTEM.format(side=side.capitalize())},
        {
            "role": "user",
            "content": (
                f"{_moves_section(session)}\n\n"
                f"Position:\nFEN: {board.fen()}\n{_board_text(board)}\n"
                f"{side.capitalize()} to move.\n\n"
                f"Legal moves: {', '.join(session.legal_moves())}"
            ),
        },
    ]


def _moves_section(session: GameSession) -> str:
    """The game so far, numbered like a score sheet. A game that began from
    a set-up position says which, first, so the moves read from it."""
    root = session.position_fens()[0]
    start = "" if root == _STANDARD_START else f"Starting position: {root}\n"
    sans = session.move_history()
    if not sans:
        return f"{start}Moves so far: none yet."
    parts: list[str] = []
    for index, ((number, color), san) in enumerate(
        zip(session.ply_labels(), sans, strict=True)
    ):
        if color == "white":
            parts.append(f"{number}. {san}")
        elif index == 0:
            parts.append(f"{number}... {san}")
        else:
            parts.append(san)
    return f"{start}Moves so far: {' '.join(parts)}"


def _board_text(board: chess.Board) -> str:
    """The board as text, White at the bottom: rank labels on the left, file
    labels below, upper case for White's pieces and lower case for Black's."""
    rows = []
    for rank in range(7, -1, -1):
        cells = []
        for file in range(8):
            piece = board.piece_at(chess.square(file, rank))
            cells.append(piece.symbol() if piece else ".")
        rows.append(f"{rank + 1} {' '.join(cells)}")
    rows.append("  a b c d e f g h")
    rows.append("(upper case White, lower case Black, . empty)")
    return "\n".join(rows)


def _refusal(error: str, retry: str = "call play_move again") -> dict[str, Any]:
    return {"ok": False, "error": error, "retry": retry}
