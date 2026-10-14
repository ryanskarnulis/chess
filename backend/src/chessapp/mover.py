"""The mover: Glitch choosing his own move (#472, docs/glitch-difficulty.md).

In the `glitch` difficulty tier the model picks Glitch's move and no engine
helps it. This module is that pick: one model phase, `mover`, beside the
brain's planner, narrator and answer reader, with one tool, `play_move(move,
intent)`, and a second, `try_line(lines)`, when the profile allows it (#498).
It is a move *source* (`coordinator.MoveSource`), not a step in the
planner's loop, so no turn's language reaches it and the planner cannot ask
for, skip or steer it.

**What it sees is the board and nothing else** (decision 2). The game's
moves, the position as a FEN and a readable board, which side it plays, and
the legal moves, and when the profile turns them on, the board facts
(`board_facts.py`, #497): what attacks and defends what, read off the board
by python-chess. Never the conversation, never anything Stockfish computed,
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

**Looking ahead is his, the board is code's** (#498, docs/glitch-sight.md
decision 3). With `[phases.mover] look_calls` above 0, `try_line` lets him
name up to `MAX_LINES` lines of up to `MAX_LINE_PLIES` plies, his move first,
and code plays each on a copy of the board and shows where it lands: the
board, its facts, how the game stands and the material. It never extends a
line, picks a reply or compares lines. A look never counts against
`MAX_ATTEMPTS`; past `look_calls` the tool stays listed (the prefix still
caches) and refuses, and that reply is an attempt. The deadline covers looks
like every call. At `0` the tool is left out and the request is as before.

**Running out is a failure the caller sees, never a move.** Its own deadline
and token cap come from the profile's `[phases.mover]`. Passing either,
exhausting the attempts, or a provider that is down raises `MoverFailed` with
its kind, and the caller (#473) decides what plays instead.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import chess
from pydantic import BaseModel, ValidationError

from chessapp.board_facts import board_facts as facts_for
from chessapp.board_facts import material
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
TRY_LINE = "try_line"

# How much one look may ask for (docs/glitch-sight.md decision 3): a few
# candidate moves, each a short exchange. Longer lines are what an engine
# reads, and every line is prompt the next call carries.
MAX_LINES = 3
MAX_LINE_PLIES = 6

# An attempt's verdict.
PLAYED = "played"
ILLEGAL = "illegal"
NO_CALL = "no_call"
BAD_ARGS = "bad_args"
# A `try_line` past the profile's `look_calls`: refused, and an attempt.
NO_LOOKS = "no_looks"

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

# Added to the instructions when the profile allows looks (#498). Without them
# the request is the one `_SYSTEM` alone makes, byte for byte.
_LOOKS = """

Before you play, you may look ahead by calling try_line, up to {calls} \
times. Give it up to {lines} lines, each starting with your move and going on \
with the replies you expect, up to {plies} moves in all, like "Qxd4 cxd4". \
Each line is played on a copy of the board, and you see the position where \
it ends. Looking never plays a move; only play_move does."""

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


TRY_LINE_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": TRY_LINE,
        "description": (
            "Look ahead before you play: each line is played on a copy of the "
            "board and you see where it ends. Nothing is played."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "lines": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        f"Up to {MAX_LINES} lines, each your move then the "
                        f"replies you expect, up to {MAX_LINE_PLIES} moves in "
                        'SAN (e.g. "Qxd4 cxd4").'
                    ),
                },
            },
            "required": ["lines"],
        },
    },
}


class _PlayMoveArgs(BaseModel):
    move: str
    intent: str


class _TryLineArgs(BaseModel):
    # One line as a bare string, or a line as a list of moves, is still a
    # line: the shape of the arguments, not words to interpret.
    lines: list[str | list[str]] | str


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
class LineResult:
    """One line of a look: what he asked, the plies code played, why it
    stopped early if it did, and where it ended."""

    asked: str
    played: tuple[str, ...]
    stopped: str
    fen: str
    status: str

    def as_trace(self) -> dict[str, Any]:
        return {
            "asked": self.asked,
            "played": list(self.played),
            "stopped": self.stopped,
            "fen": self.fen,
            "status": self.status,
        }


@dataclass(frozen=True)
class Look:
    """One `try_line` call that was answered with lines: what it asked and
    what came back. A refused call is an attempt instead, not a look."""

    lines: tuple[LineResult, ...]
    dropped: tuple[str, ...] = ()

    def as_trace(self) -> dict[str, Any]:
        return {
            "lines": [line.as_trace() for line in self.lines],
            "dropped": list(self.dropped),
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
    looks: tuple[Look, ...] = ()

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
            "looks": [look.as_trace() for look in self.looks],
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
        looks: tuple[Look, ...] = (),
    ) -> None:
        super().__init__(f"mover {kind}: {detail}")
        self.kind = kind
        self.detail = detail
        self.model = model
        self.attempts = attempts
        self.calls = calls
        self.looks = looks

    def as_trace(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "outcome": str(self.kind),
            "detail": self.detail,
            "uci": None,
            "san": None,
            "intent": None,
            "attempts": [attempt.as_trace() for attempt in self.attempts],
            "looks": [look.as_trace() for look in self.looks],
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
        settings = mover.settings
        self.messages = mover_messages(
            session, board_facts=settings.board_facts, look_calls=settings.look_calls
        )
        self.tools = [PLAY_MOVE_TOOL] + ([TRY_LINE_TOOL] if settings.look_calls else [])
        self.attempts: list[MoverAttempt] = []
        self.calls: list[ModelCall] = []
        self.looks: list[Look] = []
        self.started = mover.clock()
        deadline_s = mover.settings.deadline_s
        self.deadline = None if deadline_s is None else self.started + deadline_s

    def play(self) -> MoverChoice:
        # A reply that looked is not an attempt. Looks are capped, so the run
        # still ends: at most MAX_ATTEMPTS + look_calls replies.
        spent = 0
        while spent < MAX_ATTEMPTS:
            looked = len(self.looks)
            result = self._call()
            if result is None:
                spent += 1
                continue
            choice = self._read(result)
            if choice is not None:
                return choice
            if len(self.looks) == looked:
                spent += 1
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
                    tools=self.tools,
                    enable_thinking=settings.thinking,
                    max_tokens=settings.max_tokens,
                    temperature=self.mover.temperature,
                    timeout=remaining,
                    thinking_budget=settings.thinking_budget,
                    effort=settings.effort,
                )
        except ToolCallArgumentsError as exc:
            self._count(started, CALL_BAD_ARGS, remaining, failure=exc.failure)
            self.attempts.append(MoverAttempt(BAD_ARGS, reason=str(exc)))
            self.messages.append(
                {
                    "role": "user",
                    "content": (
                        f"Your {exc.tool_name} call failed before it ran: "
                        f"{exc}. Call {exc.tool_name} again with JSON arguments."
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
        looked = len(self.looks)
        for call in result.tool_calls:
            answer: dict[str, Any] | str
            if chosen is not None:
                answer = {"ok": False, "error": "your move is already played"}
            else:
                chosen, answer = self._judge(call.name, call.arguments)
            self.messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": answer
                    if isinstance(answer, str)
                    else json.dumps(answer),
                }
            )
        if chosen is None and len(self.looks) > looked:
            # His turn ends with the look, and the next reply starts a turn of
            # its own (#499). Left to continue his turn after a tool answer,
            # Gemma 4's template opens the thought for him, and the 12B closes
            # it unread or loops on it until the token cap.
            left = self.mover.settings.look_calls - len(self.looks)
            self.messages.append(
                {
                    "role": "user",
                    "content": "Play your move with play_move"
                    + (", or look again with try_line." if left else "."),
                }
            )
        return chosen

    def _judge(
        self, name: str, arguments: dict[str, Any]
    ) -> tuple[MoverChoice | None, dict[str, Any] | str]:
        """One call, checked by code: the choice if it is a legal move with an
        intent, else the refusal the model reads."""
        if name == TRY_LINE and self.mover.settings.look_calls:
            return None, self._look(arguments)
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
            looks=tuple(self.looks),
        )
        return choice, {"ok": True, "played": san}

    def _look(self, arguments: dict[str, Any]) -> dict[str, Any] | str:
        """One `try_line` call: the lines played on a copy of the board as
        the text he reads, or the refusal. The session is never touched."""
        left = self.mover.settings.look_calls - len(self.looks)
        if left <= 0:
            reason = "no looks left; play your move"
            self.attempts.append(MoverAttempt(NO_LOOKS, reason=reason))
            return _refusal(reason, "call play_move")
        try:
            args = _TryLineArgs.model_validate(arguments)
        except ValidationError:
            reason = 'try_line needs lines: a list of lines in SAN, like ["Qxd4 cxd4"]'
            self.attempts.append(MoverAttempt(BAD_ARGS, reason=reason))
            return _refusal(reason, "call try_line again, or play_move")
        asked = [args.lines] if isinstance(args.lines, str) else args.lines
        texts = [line if isinstance(line, str) else " ".join(line) for line in asked]
        texts = [text.strip() for text in texts if text.strip()]
        if not texts:
            reason = "try_line got no lines; give at least one, your move first"
            self.attempts.append(MoverAttempt(BAD_ARGS, reason=reason))
            return _refusal(reason, "call try_line again, or play_move")
        start = _game_board(self.session)
        results = [_play_line(start, text) for text in texts[:MAX_LINES]]
        look = Look(
            lines=tuple(result for result, _ in results),
            dropped=tuple(texts[MAX_LINES:]),
        )
        self.looks.append(look)
        return _look_text(start, look, [text for _, text in results], left - 1)

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
            looks=tuple(self.looks),
        )


def mover_messages(
    session: GameSession, *, board_facts: bool = False, look_calls: int = 0
) -> list[dict[str, Any]]:
    """The mover's opening request for the side to move in `session`: the
    fixed instructions and side (and how to look ahead, when `look_calls`
    allows it, #498), then the moves, then the position, with its board facts
    when the profile asks for them (#497). Built from the board alone (module
    docstring)."""
    board = chess.Board(session.fen())
    side = session.turn
    facts = f"{facts_for(board).text()}\n\n" if board_facts else ""
    system = _SYSTEM.format(side=side.capitalize())
    if look_calls:
        system += _LOOKS.format(calls=look_calls, lines=MAX_LINES, plies=MAX_LINE_PLIES)
    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": (
                f"{_moves_section(session)}\n\n"
                f"Position:\n{_position_text(board)}\n\n"
                f"{facts}"
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


def _position_text(board: chess.Board) -> str:
    """The FEN, the board and whose move it is."""
    side = "White" if board.turn == chess.WHITE else "Black"
    return f"FEN: {board.fen()}\n{_board_text(board)}\n{side} to move."


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


# --- looking ahead (#498) ----------------------------------------------------

# Score-sheet clutter around a move: "17.", "17...", "…", and the !/? a
# player annotates with. Notation, not words: what is left is a SAN or UCI move
# for python-chess to read, or not.
_MOVE_NUMBER = re.compile(r"^\d*(?:\.+|…)")
_ANNOTATION = re.compile(r"[!?]+$")


def _plies(text: str) -> list[str]:
    tokens = []
    for raw in re.split(r"[\s,;]+", text):
        token = _ANNOTATION.sub("", _MOVE_NUMBER.sub("", raw.strip()))
        if token and token.strip(".…"):
            tokens.append(token)
    return tokens


def _game_board(session: GameSession) -> chess.Board:
    """The game's board with its moves, so a line that repeats a position
    is seen to."""
    board = chess.Board(session.position_fens()[0])
    for san in session.move_history():
        board.push_san(san)
    return board


def _parse(board: chess.Board, text: str) -> tuple[chess.Move | None, str]:
    """`text` as a legal move on `board` (SAN, then UCI), or why not, with
    the legal moves like it: `play_move`'s refusal, on the line's board."""
    alternatives = GameSession(fen=board.fen()).move_alternatives
    try:
        return board.parse_san(text), ""
    except chess.AmbiguousMoveError:
        options = alternatives(text)
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
    if move is not None and move in board.legal_moves:
        return move, ""
    options = alternatives(text)
    return None, (
        f"{text} is not a legal move there"
        + (f"; legal moves like it: {', '.join(options)}" if options else "")
    )


def _status(board: chess.Board) -> str:
    """How the game stands on `board`: over and how, or a draw that may be
    claimed, or "" while it goes on. The rules', read off the board."""
    outcome = board.outcome()
    if outcome is not None:
        if outcome.termination == chess.Termination.CHECKMATE:
            winner = "White" if outcome.winner == chess.WHITE else "Black"
            return f"checkmate, {winner} wins"
        reason = outcome.termination.name.lower().replace("_", " ")
        return f"drawn by {reason}"
    if board.can_claim_threefold_repetition():
        return "a draw may be claimed by threefold repetition"
    if board.can_claim_fifty_moves():
        return "a draw may be claimed by the fifty-move rule"
    return ""


def _play_line(start: chess.Board, asked: str) -> tuple[LineResult, str]:
    """`asked` played on a copy of `start`, as far as it is legal, and the
    text he reads about it."""
    board = start.copy()
    tokens = _plies(asked)
    played: list[str] = []
    stopped = ""
    for token in tokens[:MAX_LINE_PLIES]:
        if board.is_game_over():
            stopped = "the game is over there; the rest was not played"
            break
        move, reason = _parse(board, token)
        if move is None:
            stopped = reason
            break
        played.append(board.san(move))
        board.push(move)
    else:
        if len(tokens) > MAX_LINE_PLIES:
            stopped = (
                f"only the first {MAX_LINE_PLIES} moves are played; the rest was not"
            )
    result = LineResult(
        asked=asked,
        played=tuple(played),
        stopped=stopped,
        fen=board.fen(),
        status=_status(board),
    )
    return result, _line_text(start, board, result)


def _numbered(start: chess.Board, sans: tuple[str, ...]) -> str:
    """`sans` from `start` as a score sheet writes them: "17... Qxd4 18. cxd4"."""
    number, white = start.fullmove_number, start.turn == chess.WHITE
    parts = []
    for index, san in enumerate(sans):
        if white:
            parts.append(f"{number}. {san}")
        elif index == 0:
            parts.append(f"{number}... {san}")
        else:
            parts.append(san)
        if not white:
            number += 1
        white = not white
    return " ".join(parts)


def _line_text(start: chess.Board, end: chess.Board, result: LineResult) -> str:
    if not result.played:
        return f"not played: {result.stopped}"
    lines = [_numbered(start, result.played)]
    if result.stopped:
        lines.append(f"Stopped there: {result.stopped}.")
    lines.append(f"Position at its end:\n{_position_text(end)}")
    if result.status:
        lines.append(f"The game: {result.status}.")
    lines.append(facts_for(end).text())
    lines.append(
        "Material (1/3/3/5/9): "
        f"White {material(end, chess.WHITE)}, Black {material(end, chess.BLACK)}; "
        f"before the line White {material(start, chess.WHITE)}, "
        f"Black {material(start, chess.BLACK)}."
    )
    return "\n".join(lines)


def _look_text(start: chess.Board, look: Look, texts: list[str], left: int) -> str:
    """What one look answers: each line in turn, then what was left out and
    how many looks remain."""
    parts = ["Played on a copy of the board; nothing was played in the game."]
    for number, text in enumerate(texts, 1):
        parts.append(f"Line {number}: {text}")
    if look.dropped:
        parts.append(
            f"Only {MAX_LINES} lines a look; not played: {'; '.join(look.dropped)}"
        )
    parts.append(
        f"Looks left: {left}." if left else "No looks left; play your move next."
    )
    return "\n\n".join(parts)
