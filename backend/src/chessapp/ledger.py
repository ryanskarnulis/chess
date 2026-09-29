"""The ledger: what happened in the game, as code saw it (#372).

A board holds a position and a move stack; it forgets everything else. A
takeback pops the stack and keeps nothing, a setting overwrites the last one,
and a declined draw offer changes nothing at all. So "took back Qxf7", "went
up a difficulty level" and "offered a draw, declined" have no record anywhere
but the trace — and the UI buttons and MCP write no trace. The ledger is that
record: an append-only list of events, keyed to the move list, written by code
and never by a model.

It is what the model phases read in place of the chat history (#372, rendered
by `render_record`; `docs/game-record.md`): no model writes or rewrites it, so
it cannot hold a fact that did not happen. #373 grows it into
`lookup(this_game)`, so the queries are shaped for that: the moves, the
captures, the material by ply, the takebacks and the setting changes of one
game.

**Observed, not hooked.** `observe` diffs the session and the settings against
the last state it saw: the common prefix of the two move lines is kept, what was
popped off it is a takeback, what was pushed is moves, a new `game_id` is a new
or resumed game, a newly finished outcome is the ending, and a changed setting
is a setting event. Hooking every mutator would have to find every road onto
the board — the tools, the coordinator's engine replies, the undo and difficulty
buttons, MCP — and the first one missed would be a silent gap. A diff needs
only to be called often enough: after every tool dispatch, which keeps "undo,
then play e4" as two events in order, and wherever a request that can mutate
lets go of the board. `note_draw_offer`, `note_offer` and `expect_resume` are
the facts no board shows.

Only the current game is persisted (`live.json`). A restored ledger is kept
only if its events replay to the session's move line; otherwise it is rebuilt
from the session, moves only (`restore`).
"""

import threading
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import chess

from chessapp.analysis import captured_piece
from chessapp.game import GameSession

# The event kinds, and the only ones a persisted ledger may hold.
NEW_GAME = "new_game"
RESUMED = "resumed"
MOVE = "move"
TAKEBACK = "takeback"
SETTING = "setting"
DRAW_OFFER = "draw_offer"
GAME_END = "game_end"
OFFER = "offer"
KINDS = frozenset(
    {NEW_GAME, RESUMED, MOVE, TAKEBACK, SETTING, DRAW_OFFER, GAME_END, OFFER}
)

# Which tool results put moves in front of the player, what the ledger calls
# them, and where the moves are in the result. A read of the result, never of
# anybody's words: what was offered is what the tool answered.
OFFER_SOURCES = {
    "get_best_moves": "hint",
    "ask_player": "question",
    "make_move": "alternatives",
}

# The settings the ledger follows, named as the facts name them
# (`facts.settings_of`): what a player can change and hear about.
SETTING_NAMES = ("difficulty", "verbosity", "voice")

# How many events of earlier games are kept in memory beside the current
# game's, for a reader that asks about the game before this one (#373's
# lookup); this bounds a process that plays for days. Never persisted.
EARLIER_GAMES_KEPT = 500

_PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
}
_COLOR_NAMES = {chess.WHITE: "white", chess.BLACK: "black"}


@dataclass(frozen=True)
class LedgerEvent:
    """One thing that happened. `seq` orders every event the process has
    seen; `ply` is the length of the move line right after it, which is what
    keys an event to the move list ("after 12... e5"). `details` holds the
    kind's own fields (see `Ledger`), JSON values only."""

    seq: int
    kind: str
    game_id: str
    ply: int
    details: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "kind": self.kind,
            "game_id": self.game_id,
            "ply": self.ply,
            **dict(self.details),
        }

    @classmethod
    def from_dict(cls, data: Any) -> "LedgerEvent":
        if not isinstance(data, dict):
            raise ValueError(f"ledger event is not an object: {data!r}")
        rest = dict(data)
        seq, kind = rest.pop("seq", None), rest.pop("kind", None)
        game_id, ply = rest.pop("game_id", None), rest.pop("ply", None)
        if type(seq) is not int or seq < 0:
            raise ValueError(f"invalid ledger seq: {seq!r}")
        if kind not in KINDS:
            raise ValueError(f"invalid ledger kind: {kind!r}")
        if not isinstance(game_id, str):
            raise ValueError(f"invalid ledger game_id: {game_id!r}")
        if type(ply) is not int or ply < 0:
            raise ValueError(f"invalid ledger ply: {ply!r}")
        return cls(seq=seq, kind=kind, game_id=game_id, ply=ply, details=rest)


def difficulty_label(settings: Mapping[str, Any]) -> str:
    """The difficulty as one value: the tier's name, or the raw strength a
    session was dialed in by (`Settings` holds exactly one of the three)."""
    if settings.get("tier") is not None:
        return str(settings["tier"])
    if settings.get("skill_level") is not None:
        return f"skill {settings['skill_level']}"
    if settings.get("elo") is not None:
        return f"elo {settings['elo']}"
    return "default"


def followed_settings(snapshot: Mapping[str, Any]) -> dict[str, str]:
    """`Settings.snapshot()` reduced to the values the ledger follows."""
    return {
        "difficulty": difficulty_label(snapshot),
        "verbosity": str(snapshot.get("verbosity", "normal")),
        "voice": "on" if snapshot.get("voice_output") else "off",
    }


class Ledger:
    """The event list, and the state the next `observe` diffs against.

    Event details by kind:

    - `new_game`: `player_color`, and `root_fen` when the game did not start
      from the standard position.
    - `resumed`: `name` (the save), `player_color`, and `root_fen` as above.
      The resumed line follows as `move` events marked `restored`.
    - `move`: `san`, `uci`, `color`, `by` (`player` or `engine`, off the
      player's colour), `move_number`, `capture` (the piece taken, or None),
      `check`, `material` (the player's advantage in pawns after the move), and
      `restored` when the move was replayed from a save rather than played.
    - `takeback`: `undone` (SAN, last move first) and `plies`.
    - `setting`: `name` (`SETTING_NAMES`), `before`, `after`.
    - `draw_offer`: `accepted`, `reason` (`draw_offer.judge_draw_offer`'s).
    - `game_end`: `termination`, `result`, `winner` (`player`, `opponent`, or
      None for a draw).
    """

    def __init__(self) -> None:
        self._events: list[LedgerEvent] = []
        self._next_seq = 0
        self._game_id: str | None = None
        self._line: list[str] = []  # UCI
        self._sans: list[str] = []
        self._ended = False
        self._settings: dict[str, str] | None = None
        self._resume_name: str | None = None
        self._lock = threading.Lock()

    # --- recording -----------------------------------------------------------

    def observe(self, session: GameSession, settings: Mapping[str, Any]) -> None:
        """Record whatever changed since the last observation. Idempotent: an
        unchanged session and settings add nothing."""
        with self._lock:
            self._observe(session, followed_settings(settings))

    def note_draw_offer(
        self,
        session: GameSession,
        settings: Mapping[str, Any],
        *,
        accepted: bool,
        reason: str | None,
    ) -> None:
        """A draw offer and the engine's answer: the one event no board shows
        when declined. Called before an accepted offer ends the game, so the
        offer reads before the ending it caused."""
        with self._lock:
            self._observe(session, followed_settings(settings))
            self._append(
                DRAW_OFFER,
                session.game_id,
                {"accepted": accepted, "reason": reason},
            )

    def note_offer(
        self,
        session: GameSession,
        settings: Mapping[str, Any],
        tool: str,
        result: Mapping[str, Any],
    ) -> None:
        """The moves a tool result offered the player — a hint's candidates,
        a question's choices, a refused move's alternatives — which "the
        second one" may point back to long after the result has scrolled out
        of view. Nothing is recorded for a result that offered nothing."""
        moves = offered_moves(tool, result)
        if not moves:
            return
        with self._lock:
            self._observe(session, followed_settings(settings))
            self._append(
                OFFER,
                session.game_id,
                {"source": OFFER_SOURCES[tool], "moves": moves},
            )

    def expect_resume(self, name: str) -> None:
        """The next new game this ledger observes is the save `name` coming
        back, not a reset: a board cannot tell the two apart."""
        with self._lock:
            self._resume_name = name

    def _observe(self, session: GameSession, settings: dict[str, str]) -> None:
        game_id = session.game_id
        if game_id != self._game_id:
            self._start_game(session)
        else:
            self._follow_line(session)
        self._follow_outcome(session)
        if self._settings is not None:
            for name in SETTING_NAMES:
                before, after = self._settings[name], settings[name]
                if before != after:
                    self._append(
                        SETTING,
                        game_id,
                        {"name": name, "before": before, "after": after},
                    )
        self._settings = settings

    def _start_game(self, session: GameSession) -> None:
        resumed, self._resume_name = self._resume_name, None
        details: dict[str, Any] = {"player_color": session.player_color}
        root = session.position_fens()[0]
        if root != chess.STARTING_FEN:
            details["root_fen"] = root
        self._game_id = session.game_id
        self._line, self._sans, self._ended = [], [], False
        if resumed is not None:
            self._append(RESUMED, session.game_id, {"name": resumed, **details})
        else:
            self._append(NEW_GAME, session.game_id, details)
        self._trim_earlier_games()
        # A resumed line (or a session that arrived with moves: a restored
        # checkpoint that had to be rebuilt) is replayed as moves, so the
        # game's captures and material are there to ask about.
        self._push_moves(session, _uci_line(session), restored=True)

    def _follow_line(self, session: GameSession) -> None:
        line = _uci_line(session)
        common = 0
        while (
            common < len(line)
            and common < len(self._line)
            and line[common] == self._line[common]
        ):
            common += 1
        if common < len(self._line):
            undone = list(reversed(self._sans[common:]))
            del self._line[common:]
            del self._sans[common:]
            self._ended = False
            self._append(
                TAKEBACK,
                session.game_id,
                {"undone": undone, "plies": len(undone)},
            )
        self._push_moves(session, line[common:], restored=False)

    def _push_moves(
        self, session: GameSession, ucis: Sequence[str], *, restored: bool
    ) -> None:
        if not ucis:
            return
        board = _board_at(session, len(self._line))
        player = chess.WHITE if session.player_color == "white" else chess.BLACK
        for uci in ucis:
            move = chess.Move.from_uci(uci)
            color = board.turn
            details: dict[str, Any] = {
                "san": board.san(move),
                "uci": uci,
                "color": _COLOR_NAMES[color],
                "by": "player" if color == player else "engine",
                "move_number": board.fullmove_number,
                "capture": captured_piece(board, move)
                if board.is_capture(move)
                else None,
            }
            board.push(move)
            details["check"] = board.is_check()
            details["material"] = _material(board, player)
            if restored:
                details["restored"] = True
            self._line.append(uci)
            self._sans.append(details["san"])
            self._append(MOVE, session.game_id, details)

    def _follow_outcome(self, session: GameSession) -> None:
        outcome = session.outcome()
        if outcome is None:
            self._ended = False
            return
        if self._ended:
            return
        self._ended = True
        winner = None
        if outcome.winner is not None:
            winner = "player" if outcome.winner == session.player_color else "opponent"
        self._append(
            GAME_END,
            session.game_id,
            {
                "termination": outcome.termination,
                "result": outcome.result,
                "winner": winner,
            },
        )

    def _append(self, kind: str, game_id: str, details: dict[str, Any]) -> None:
        self._events.append(
            LedgerEvent(
                seq=self._next_seq,
                kind=kind,
                game_id=game_id,
                ply=len(self._line),
                details=details,
            )
        )
        self._next_seq += 1

    def _trim_earlier_games(self) -> None:
        current = [e for e in self._events if e.game_id == self._game_id]
        earlier = [e for e in self._events if e.game_id != self._game_id]
        self._events = earlier[-EARLIER_GAMES_KEPT:] + current

    # --- reading ---------------------------------------------------------------

    @property
    def game_id(self) -> str | None:
        return self._game_id

    @property
    def next_seq(self) -> int:
        """The `seq` the next event will get: a cursor for `since`."""
        return self._next_seq

    def events(self, game_id: str | None = None) -> list[LedgerEvent]:
        """Every event held, oldest first; one game's when `game_id` is given."""
        with self._lock:
            return [e for e in self._events if game_id is None or e.game_id == game_id]

    def since(self, seq: int) -> list[LedgerEvent]:
        """The events from `seq` on, across games — what happened since a
        reader last looked."""
        with self._lock:
            return [e for e in self._events if e.seq >= seq]

    def current(self) -> list[LedgerEvent]:
        """The current game's events."""
        return self.events(self._game_id)

    def moves(self, game_id: str | None = None) -> list[LedgerEvent]:
        """Every move played in a game (the current one by default),
        including ones later taken back, in the order they were played."""
        return [e for e in self._of(game_id) if e.kind == MOVE]

    def takebacks(self, game_id: str | None = None) -> list[LedgerEvent]:
        return [e for e in self._of(game_id) if e.kind == TAKEBACK]

    def setting_changes(self, game_id: str | None = None) -> list[LedgerEvent]:
        return [e for e in self._of(game_id) if e.kind == SETTING]

    def captures(self, game_id: str | None = None) -> list[LedgerEvent]:
        """The moves that took something, taken back or not."""
        return [e for e in self.moves(game_id) if e.details.get("capture")]

    def line(self, game_id: str | None = None) -> list[LedgerEvent]:
        """The move events of the line standing now: every move, minus the
        ones a takeback popped. What `session.move_history()` should equal."""
        standing: list[LedgerEvent] = []
        for event in self._of(game_id):
            if event.kind == MOVE:
                standing.append(event)
            elif event.kind == TAKEBACK:
                del standing[len(standing) - event.details["plies"] :]
        return standing

    def material_by_ply(self, game_id: str | None = None) -> dict[int, int]:
        """The player's material after each ply of the standing line."""
        return {e.ply: e.details["material"] for e in self.line(game_id)}

    def _of(self, game_id: str | None) -> list[LedgerEvent]:
        return self.events(self._game_id if game_id is None else game_id)

    # --- persistence -----------------------------------------------------------

    def to_dict(self) -> list[dict[str, Any]]:
        """The current game's events, for the live checkpoint."""
        return [e.to_dict() for e in self.current()]

    @classmethod
    def restore(
        cls, data: Any, session: GameSession, settings: Mapping[str, Any]
    ) -> "Ledger":
        """The ledger a checkpoint recorded for `session`, or — when there is
        none, it does not parse, or its events do not replay to the session's
        move line — one rebuilt from the session, moves only. Either way the
        result is caught up with `session` and `settings`, so the next
        `observe` records only what happens next."""
        ledger = cls()
        try:
            events = _parse_events(data)
        except ValueError:
            events = []
        if events and _replays_to(events, session):
            ledger._events = events
            ledger._next_seq = events[-1].seq + 1
            ledger._game_id = session.game_id
            standing = ledger.line()
            ledger._line = [e.details["uci"] for e in standing]
            ledger._sans = [e.details["san"] for e in standing]
            ledger._ended = any(
                e.kind == GAME_END for e in events if e.seq > _last_line_change(events)
            )
            ledger._settings = followed_settings(settings)
            ledger._follow_outcome(session)
            return ledger
        ledger.observe(session, settings)
        return ledger


def offered_moves(tool: str, result: Mapping[str, Any]) -> list[str]:
    """The moves `result` offered, by `tool`: none unless it is one of
    `OFFER_SOURCES` and the call did what it offers."""
    if tool == "get_best_moves" and result.get("ok") is True:
        return [m["san"] for m in result.get("moves", ()) if m.get("san")]
    if tool == "ask_player" and result.get("ok") is True:
        return list(result.get("candidates", ()))
    if tool == "make_move" and result.get("legal") is False:
        return list(result.get("alternatives", ()))
    return []


def from_session(session: GameSession, settings: Mapping[str, Any]) -> Ledger:
    """A ledger for a game nobody watched being played: its start and its
    standing line (moves marked `restored`), and its ending if it has one."""
    ledger = Ledger()
    ledger.observe(session, settings)
    return ledger


def _parse_events(data: Any) -> list[LedgerEvent]:
    if not isinstance(data, list):
        raise ValueError("ledger must be a list of events")
    events = [LedgerEvent.from_dict(item) for item in data]
    seqs = [e.seq for e in events]
    if seqs != sorted(set(seqs)):
        raise ValueError("ledger seqs must be strictly increasing")
    for event in events:
        if event.kind == MOVE:
            for key in ("san", "uci", "material"):
                if key not in event.details:
                    raise ValueError(f"move event missing {key}")
        elif event.kind == TAKEBACK:
            plies = event.details.get("plies")
            if type(plies) is not int or plies < 1:
                raise ValueError(f"invalid takeback plies: {plies!r}")
    return events


def _replays_to(events: Iterable[LedgerEvent], session: GameSession) -> bool:
    """Whether `events` are this game's, from its start, and their standing
    line is the session's move line."""
    events = list(events)
    if not events or events[0].kind not in (NEW_GAME, RESUMED):
        return False
    if any(e.game_id != session.game_id for e in events):
        return False
    standing: list[str] = []
    for event in events:
        if event.kind == MOVE:
            standing.append(event.details["uci"])
        elif event.kind == TAKEBACK:
            plies = event.details["plies"]
            if plies > len(standing):
                return False
            del standing[len(standing) - plies :]
    return standing == _uci_line(session)


def _last_line_change(events: Sequence[LedgerEvent]) -> int:
    changes = [e.seq for e in events if e.kind in (MOVE, TAKEBACK)]
    return changes[-1] if changes else -1


def _uci_line(session: GameSession) -> list[str]:
    return list(session.to_dict()["moves"])


def _board_at(session: GameSession, ply: int) -> chess.Board:
    board = chess.Board(session.position_fens()[0])
    for uci in _uci_line(session)[:ply]:
        board.push(chess.Move.from_uci(uci))
    return board


def _material(board: chess.Board, player: chess.Color) -> int:
    totals = {chess.WHITE: 0, chess.BLACK: 0}
    for piece_type, value in _PIECE_VALUES.items():
        for color in (chess.WHITE, chess.BLACK):
            totals[color] += value * len(board.pieces(piece_type, color))
    return totals[player] - totals[not player]


# --- the record of the game, as the model phases will read it -------------------

# How many lines the record keeps, newest last. Past it the oldest go, with a
# count: a record that quietly forgets reads like one that never heard.
RECORD_MAX_LINES = 30

_OFFER_WORDS = {
    "hint": "a hint offered {moves}",
    "question": "the player was asked to choose between {moves}",
    "alternatives": "a move could not be played; the alternatives offered were {moves}",
}


def render_record(
    events: Sequence[LedgerEvent], max_lines: int = RECORD_MAX_LINES
) -> list[str]:
    """The game's events that its move list cannot show — takebacks, setting
    changes, draw offers, what was offered, the ending, the game's start —
    one line each, keyed to the move list ("after 12... e5: ..."). Moves are
    not listed: the state block's `history` holds them, and a second copy
    would be the ageing duplicate `docs/turn-memory.md` forbids. Written by
    code from the ledger, so it cannot say what did not happen."""
    lines: list[str] = []
    standing: list[str] = []

    def at() -> str:
        return f"after {standing[-1]}" if standing else "before the first move"

    for event in events:
        d = event.details
        kind = event.kind
        if kind == MOVE:
            standing.append(_label(d))
        elif kind in (NEW_GAME, RESUMED):
            start = (
                f"The saved game '{d.get('name')}' was resumed"
                if kind == RESUMED
                else "A new game began"
            )
            start += f"; the player has {d.get('player_color')}"
            if d.get("root_fen"):
                start += ", from a set-up position"
            lines.append(start + ".")
        elif kind == TAKEBACK:
            plies = min(int(d.get("plies", 0)), len(standing))
            before = at()
            undone = list(reversed(standing[len(standing) - plies :]))
            del standing[len(standing) - plies :]
            lines.append(f"{before}: took back {', '.join(undone)}.")
        elif kind == SETTING:
            lines.append(
                f"{at()}: {d.get('name')} changed from {d.get('before')} "
                f"to {d.get('after')}."
            )
        elif kind == DRAW_OFFER:
            answer = "accepted" if d.get("accepted") else "declined"
            reason = str(d.get("reason") or "").replace("_", " ")
            lines.append(
                f"{at()}: the player offered a draw; the engine {answer}"
                + (f" ({reason})." if reason and not d.get("accepted") else ".")
            )
        elif kind == GAME_END:
            winner = {"player": "the player won", "opponent": "the engine won"}.get(
                d.get("winner"), "a draw"
            )
            termination = str(d.get("termination", "")).replace("_", " ")
            result = d.get("result")
            lines.append(
                f"{at()}: the game ended by {termination}; {winner} ({result})."
            )
        elif kind == OFFER:
            words = _OFFER_WORDS.get(str(d.get("source")), "{moves} were offered")
            lines.append(
                f"{at()}: " + words.format(moves=", ".join(d.get("moves", ()))) + "."
            )
    if len(lines) > max_lines:
        dropped = len(lines) - max_lines
        lines = [f"({dropped} earlier events not listed)", *lines[-max_lines:]]
    return lines


def _label(details: Mapping[str, Any]) -> str:
    number, san = details.get("move_number", "?"), details.get("san")
    return (
        f"{number}. {san}" if details.get("color") == "white" else f"{number}... {san}"
    )
