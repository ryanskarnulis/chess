"""What a turn can honestly say: the facts, assembled from the record of it.

Speech is scored against these (#367), and until #368 the honesty guard checks
against them too. The assembly used to read the live `ToolContext` directly,
which made a traced turn impossible to re-judge: the move history, the
player's colour, the settings and the mid-turn boards never reached the trace.
So the pipeline now writes down a `TurnEvidence` — everything the assembly
reads, in a shape JSON holds — and `assemble` turns that into `VerifiedFacts`.
The live path and the offline scorer (`speech_accuracy`) both go through
`assemble`, so a turn re-judged from its trace is judged on exactly the facts
the live turn had.

`honesty` owns the reading of the string; this module owns what is true.
"""

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import chess

from chessapp.analysis import captured_piece
from chessapp.game import GameSession
from chessapp.honesty import VerifiedFacts
from chessapp.tools import DESTRUCTIVE_TOOLS

# Board symbol → the word commentary uses for it, for the capture claim class.
_PIECE_NAMES = {
    "p": "pawn",
    "n": "knight",
    "b": "bishop",
    "r": "rook",
    "q": "queen",
    "k": "king",
}


def relative_outcome(session: GameSession) -> dict[str, Any] | None:
    """The ending from the player's side, or None on a live board: the shape the
    honesty guard checks a winner claim against (`VerifiedFacts.winner` is
    `"player"` / `"opponent"` / None) and the shape the trace records, so a
    traced turn on a finished game can be re-judged without knowing which
    color the player had (#287)."""
    outcome = session.outcome()
    if outcome is None:
        return None
    winner = None
    if outcome.winner is not None:
        winner = "player" if outcome.winner == session.player_color else "opponent"
    return {"winner": winner, "termination": outcome.termination}


def analysis_moves(tool_results: Sequence[dict[str, Any]]) -> set[str]:
    """The moves the turn's *analysis* tools named, in SAN: the engine's word.

    This is the advice guard's evidence. Since 2026-09-10 the guard fires only
    when this set is non-empty — the turn asked Stockfish and the reply names a
    playable move Stockfish did not — because that is the one shape that is
    an honesty problem rather than an opinion. A turn that ran no analysis and
    names a move is Glitch speaking from his own chess ("play Bf4, that's the
    London"), which is his to do, and which the old rule ("evidence is the
    only licence") cut as a matter of policy: live it ate a correct opening
    answer (2026-09-06) and a list of legal alternatives a refused move had
    itself reported (2026-09-04). Hint asks still reach the engine — the
    planner routes them to `get_best_moves` and the eval gate measures that —
    so what this changes is whose word an *unasked* move is.
    """
    reported: set[str] = set()
    for r in tool_results:
        result = r["result"]
        if result.get("ok") is not True:
            continue
        if r["name"] == "get_best_moves":
            reported.update(m["san"] for m in result.get("moves", ()) if m.get("san"))
        elif r["name"] == "analyze_last_move":
            reported.update(
                san for san in (result.get("played"), result.get("best")) if san
            )
    return reported


def reported_moves(tool_results: Sequence[dict[str, Any]]) -> set[str]:
    """Every move a tool result this turn named, in SAN — the analysis moves
    and every other report a narrator may repeat.

    The advice guard's licence, once its evidence exists (`analysis_moves`).
    It is scoped to what the tools said rather than switched off wholesale:
    live, the planner answered "what should I play here?" with
    `evaluate_position` + `analyze_last_move` and the old boolean test read
    that as permission, letting the narrator hand over a list of moves no tool
    had mentioned (docs/agent-evals.md, 2026-07-25).

    `describe_position` is licensed for exactly the one move it names, the last
    one played. A description that ends "Last move: O-O." is a report, and the
    narrator repeating it is a fact — but castling is the one SAN both sides
    spell the same way, so when the other side can still castle the same
    string is a currently legal move too, and the guard read the echo as advice
    and ate the whole description. A quiet move cannot collide (its destination
    is occupied once it has been played), so the licence costs nothing else.

    A refused `make_move` reports `alternatives`, the legal moves it offered in
    place of the one it could not play, and `get_legal_moves` reports the list
    itself. Both are the tool's own words, and a narrator reading them back is
    reporting, not advising: live, "That move's cooked. You gotta pick from
    these instead: Ng5, Ne5, ..." was every one of the refusal's own
    alternatives, and the guard cut it (2026-09-04).
    """
    reported = analysis_moves(tool_results)
    for r in tool_results:
        result = r["result"]
        if result.get("ok") is not True:
            continue
        if r["name"] == "describe_position" and result.get("last_move"):
            reported.add(result["last_move"])
        elif r["name"] == "make_move" and result.get("legal") is False:
            reported.update(result.get("alternatives", ()))
        elif r["name"] == "get_legal_moves":
            reported.update(result.get("moves", ()))
        elif r["name"] == "ask_player":
            # The clarification's candidates (#289): the board-validated moves
            # the question is about, which the narrator has to name to ask it.
            reported.update(result.get("candidates", ()))
    return reported


def analysis_numbers(tool_results: Sequence[dict[str, Any]]) -> set[str]:
    """Every number the turn's analysis tools reported, in the spellings a
    commentary might quote them in: raw centipawns, pawns to two places, and
    both one-place roundings, signed and unsigned. The evaluation claim class
    checks against this, so a score with no analysis behind it has nothing to
    derive from.

    Both roundings because the sign and the magnitude are the fact and the
    rounding is wording — 147 centipawns said as "1.4" is the same report as
    "1.5", and replacing good commentary over the tenths place would cost more
    than that lie is worth.
    """
    numbers: set[str] = set()

    def record(score_cp: int | None, mate_in: int | None) -> None:
        if score_cp is not None:
            numbers.add(str(score_cp))
            pawns = score_cp / 100
            tenths = (math.floor(pawns * 10) / 10, math.ceil(pawns * 10) / 10)
            for text in (f"{pawns:.2f}", *(f"{tenth:.1f}" for tenth in tenths)):
                numbers.add(text)
                numbers.add(f"+{text}" if not text.startswith("-") else text)
        if mate_in is not None:
            numbers.update({str(mate_in), str(abs(mate_in))})

    for r in tool_results:
        result = r["result"]
        if result.get("ok") is not True:
            continue
        if r["name"] == "evaluate_position":
            record(result.get("score_cp"), result.get("mate_in"))
        elif r["name"] == "offer_draw":
            # The verdict's number, from either side of the board: the narrator
            # may say "he's up half a pawn" or "you're down half a pawn" about
            # the same fact.
            evaluation = result.get("evaluation") or {}
            cp = evaluation.get("cp_engine_pov")
            record(cp, evaluation.get("mate_in"))
            record(-cp if cp is not None else None, None)
        elif r["name"] == "get_best_moves":
            for candidate in result.get("moves", ()):
                record(candidate.get("score_cp"), candidate.get("mate_in"))
        elif r["name"] == "analyze_last_move":
            record(result.get("cp_loss"), None)
        elif r["name"] == "review_game":
            for move in result.get("critical", ()):
                record(move.get("cp_loss"), None)
            numbers.update(str(value) for value in result.get("accuracy", {}).values())
            numbers.update(str(value) for value in result.get("counts", {}).values())
    return numbers


@dataclass(frozen=True)
class TurnEvidence:
    """Everything `assemble` reads, and nothing it does not: the record of a
    turn the facts are built from, in a shape JSON holds.

    `session` is `GameSession.to_dict()` as the turn left it — the history,
    the player's colour and the session-level endings, replayed through the
    legality gate on the way back in. `settings` is the claimable value of
    each setting (`settings_of`). `tool_results` is the turn's `{"name",
    "result"}` list, `engine_reply_san` the engine's reply if one was
    collected, and `fen_before`, `fens_observed` and `pending_reply_fen` the
    boards the turn held (see `assemble`).

    `as_trace` drops `tool_results`: the trace record already carries them in
    `tools`, and `from_trace` reads them back from there.
    """

    session: Mapping[str, Any]
    settings: Mapping[str, str]
    tool_results: Sequence[Mapping[str, Any]]
    engine_reply_san: str | None
    fen_before: str
    fens_observed: tuple[str, ...] = ()
    pending_reply_fen: str | None = None

    def as_trace(self) -> dict[str, Any]:
        return {
            "session": dict(self.session),
            "settings": dict(self.settings),
            "engine_reply_san": self.engine_reply_san,
            "fen_before": self.fen_before,
            "fens_observed": list(self.fens_observed),
            "pending_reply_fen": self.pending_reply_fen,
        }

    @classmethod
    def from_trace(
        cls, evidence: Mapping[str, Any], tools: Sequence[Mapping[str, Any]]
    ) -> "TurnEvidence":
        """The evidence a trace record wrote (`as_trace`), with the tool
        results read back from the record's own `tools` list."""
        return cls(
            session=evidence["session"],
            settings=evidence["settings"],
            tool_results=[
                {"name": tool["name"], "result": tool["result"]} for tool in tools
            ],
            engine_reply_san=evidence.get("engine_reply_san"),
            fen_before=evidence["fen_before"],
            fens_observed=tuple(evidence.get("fens_observed", ())),
            pending_reply_fen=evidence.get("pending_reply_fen"),
        )


def settings_of(voice_output: bool, verbosity: str, tier: str | None) -> dict[str, str]:
    """The settings a commentary may claim, as the facts name them."""
    settings = {"voice": "on" if voice_output else "off", "verbosity": verbosity}
    if tier is not None:
        # Only a named tier is a claimable difficulty. A session dialed in by
        # elo or skill level has no tier to be honest about, and mapping one
        # back would be the code inventing the fact instead of the model.
        settings["difficulty"] = tier
    return settings


def assemble(evidence: TurnEvidence) -> VerifiedFacts:
    """What this turn may honestly say, assembled from the record of it.

    Audit item 13, the pipeline's half. The ending guard's evidence — board plus
    tool results — generalized to every operational fact a turn produces, and
    assembled here for the same reason the ending check is: this is the one
    place that holds the tool results, the engine's reply and the board at once.

    `moves` deliberately spans the whole game rather than this turn's position.
    A reaction legitimately names the move the player *didn't* play ("Bb5 was
    better"), which stopped being legal the moment the turn played something
    else, and reciting the move list or a PGN is a read the tools support —
    both turn up in the 46 recorded live turns, and both are board truth.

    That width is why the *credited* move needs its own two sets: a move the
    player played derives from `moves` perfectly, so "I played Nf3" was
    unguardable while there was only the one set. Those hold played moves only
    — the history split by side, plus this turn's own (`make_move` is the
    player's, the engine's reply is Glitch's).

    `evidence.fens_observed` is every *other* board the turn held — the positions
    between `fen_before` and now. Without them the turn is checked from boards
    its own commentary never saw: a recapture flips the material count between
    one and the next, and a move playable only mid-turn appears in neither end.
    All of them are boards this turn really held, so all of them count; an
    invented fact is invented from all of them, and staleness is not
    invention. (The narrator is no longer *handed* a mid-turn move list — its
    view carries no side to play for, #193 — but the width stays: the guard
    exists to catch invention, not tense.)

    There are two of these, one per route, and the brain route needed one too
    (audit finding 7, 2026-09-05). The fast path's is its observation beat: the
    position after the player's move, while Stockfish computes the answer this
    function is standing behind. The brain route's is the whole trail of boards
    its mutating tool calls left behind, because its narrator reads the tool
    results and those are the boards the tools ran on — `make_move(exd5)` then
    `describe_position()` counts a pawn the engine's Qxd5 has taken back by the
    time the guard looks, and the count was true when it was made. Empty for a
    route that held only the two ends.

    `evidence.pending_reply_fen` is the board the narrator spoke over when it spoke
    before the engine's reply existed — the fast path's observe beat, or a
    brain-route narrator closing a turn whose move is still owed its answer —
    and `None` when no reply was pending as it spoke. Every other piece of
    evidence here is written *after* the reply is collected, so the reply is
    in the history and among the engine's legal moves, and "My turn. Nf6."
    read as true whenever Nf6 was playable or happened to be what Stockfish
    chose. From that board come `unplayed_replies` (#289): the engine's
    options there, and its actual reply, less every move the turn accounts for
    otherwise — the game's history before the reply, what a tool reported, and
    what the *player* could play at either end of the turn, because a SAN both
    sides can spell is not evidence of anything.
    """
    session = GameSession.from_dict(evidence.session)
    tool_results = evidence.tool_results
    fen_before = evidence.fen_before
    outcome = session.outcome()
    captured = session.captured_pieces()
    opponent = "black" if session.player_color == "white" else "white"
    # Every position this turn held, the player's side carried along: whose
    # advantage the count is measured from is session state, and a session
    # rebuilt from a FEN alone would default it to white and silently invert
    # the material fact for a player playing black.
    boards = [session] + [
        GameSession(fen=fen, player_color=session.player_color)
        # Deduped: a route can name the same position twice (the fast path's
        # observed board is also the one its `make_move` left on the trail),
        # and a board counted twice is evidence exactly once.
        for fen in dict.fromkeys((fen_before, *evidence.fens_observed))
    ]
    moves = set(session.move_history())
    for board in boards:
        moves |= set(board.legal_moves())
    reported = reported_moves(tool_results)
    moves |= reported
    # The moves this turn actually talked about — what an analysis named, plus
    # what was played in it. `captures_by_move` is built from these and not
    # from every legal move, because these are the ones a narration hangs a
    # capture on, and resolving a SAN costs a legal-move generation per board.
    discussed = set(reported)
    # Whose move each one was, which `moves` deliberately cannot say. The board
    # already knows: the history splits by whose turn it was, and the session's
    # color says which of those two sides the player is. Everything else in
    # `moves` — an analysis's candidates, a move that is merely playable — is
    # nobody's move and stays out of both sets.
    played_by_color = session.move_history_by_color()
    by_player = set(played_by_color[session.player_color])
    by_opponent = set(played_by_color[opponent])
    if reply := evidence.engine_reply_san:
        moves.add(reply)
        by_opponent.add(reply)
        discussed.add(reply)
    checked = session.is_check()
    for r in tool_results:
        result = r["result"]
        if result.get("ok") is not True:
            continue
        if san := result.get("san"):
            moves.add(san)
            discussed.add(san)
            if r["name"] == "make_move":
                # The one tool that plays the player's move; every other `san`
                # a result carries is a move somebody merely talked about.
                by_player.add(san)
        moves.update(result.get("undone", ()))
        if played := result.get("engine_move"):
            moves.add(played["san"])
            by_opponent.add(played["san"])
        checked = checked or result.get("check") is True
    ending = relative_outcome(session)
    succeeded = {r["name"] for r in tool_results if r["result"].get("ok") is True}
    return VerifiedFacts(
        ended=session.is_game_over() or destructive_succeeded(tool_results),
        drawn=outcome is not None and outcome.winner is None,
        winner=ending["winner"] if ending is not None else None,
        termination=ending["termination"] if ending is not None else None,
        check=checked,
        captured_by_player=frozenset(
            _PIECE_NAMES[symbol] for symbol in captured[session.player_color]
        ),
        captured_by_opponent=frozenset(
            _PIECE_NAMES[symbol] for symbol in captured[opponent]
        ),
        moves=frozenset(moves),
        moves_by_player=frozenset(by_player),
        moves_by_opponent=frozenset(by_opponent),
        saved=any(
            r["name"] in ("save_game", "resume_game") and r["result"].get("ok") is True
            for r in tool_results
        ),
        settings=dict(evidence.settings),
        settings_changed=frozenset(_settings_changed(tool_results)),
        captures_by_move=_captures_by_move(boards, discussed),
        numbers=frozenset(analysis_numbers(tool_results)),
        # Board truth, and the one fact here no tool has to have run for: who
        # is ahead is a piece count, so it is always available on every board
        # the turn held — including the one the reaction was written from.
        material=tuple(dict.fromkeys(board.material_balance() for board in boards)),
        undone="undo" in succeeded,
        restarted="new_game" in succeeded,
        unplayed_replies=_unplayed_replies(
            session,
            evidence.engine_reply_san,
            fen_before,
            reported,
            evidence.pending_reply_fen,
        ),
        placements=_placements(
            [
                *boards,
                *(
                    [GameSession(fen=evidence.pending_reply_fen)]
                    if evidence.pending_reply_fen
                    else []
                ),
            ]
        ),
    )


# The piece word `GameSession.piece_placement` names → its SAN letter.
_PIECE_LETTERS = {name: symbol.upper() for symbol, name in _PIECE_NAMES.items()}


def _placements(boards: Sequence[GameSession]) -> frozenset[str]:
    """Every non-pawn piece as SAN writes it — `"Ke1"`, `"Nf3"` — on any of
    these boards, either colour: the move class's evidence that a piece named
    on its own square is where it stands, not a move (`VerifiedFacts`)."""
    return frozenset(
        f"{_PIECE_LETTERS[name]}{square}"
        for board in boards
        for by_type in board.piece_placement().values()
        for name, squares in by_type.items()
        if name != "pawn"
        for square in squares
    )


def _unplayed_replies(
    session: GameSession,
    engine_reply_san: str | None,
    fen_before: str,
    reported: set[str],
    pending_reply_fen: str | None,
) -> frozenset[str]:
    """The engine replies a narration spoken before the reply cannot have
    known about (`assemble`'s `pending_reply_fen`)."""
    if pending_reply_fen is None:
        return frozenset()
    player = session.player_color
    spoken_over = GameSession(fen=pending_reply_fen, player_color=player)
    if spoken_over.turn == player:
        return frozenset()  # nothing was pending on this board after all
    options = set(spoken_over.legal_moves())
    history = session.move_history()
    if engine_reply_san:
        options.add(engine_reply_san)
        if history and history[-1] == engine_reply_san:
            history = history[:-1]
    accounted = set(history) | reported
    accounted.update(GameSession(fen=fen_before, player_color=player).legal_moves())
    if session.turn == player:
        accounted.update(session.legal_moves())
    bare = {san.rstrip("+#") for san in accounted}
    return frozenset(san for san in options if san.rstrip("+#") not in bare)


def _captures_by_move(
    boards: Sequence[GameSession], sans: Iterable[str]
) -> dict[str, str]:
    """What each named move takes, off a board this turn really held.

    The piece's name, or `""` for a move that takes nothing — both are facts,
    and the empty one is the fact that catches "Nf3 grabs the knight". A move
    no board can parse is simply absent: it belonged to an earlier position
    and nothing here can say what stood on that square.

    Board truth, not a tool's report, and that is the point (walkthrough #5).
    SAN says *that* a move captures and never what, so a narration describing
    `Qxe2` had a capture with no victim and supplied one — a queen, for a move
    that takes a pawn. Only the position before the move knows, and the turn is
    holding every position it had.
    """
    victims: dict[str, str] = {}
    for board in boards:
        position = chess.Board(board.fen())
        for san in sans:
            if san in victims:
                continue
            try:
                move = position.parse_san(san)
            except ValueError:  # not legal here — a different board's move
                continue
            victims[san] = captured_piece(position, move) or ""
    return victims


# Which setting each setter owns, for the fact that the live value cannot
# carry: *that it just changed*. Keyed by tool rather than by result field so
# a setter that reports nothing still counts as having moved its setting.
_SETTERS = {
    "set_verbosity": "verbosity",
    "set_voice_output": "voice",
    "set_difficulty": "difficulty",
}


def _settings_changed(tool_results: Sequence[dict[str, Any]]) -> set[str]:
    """The settings a tool really moved this turn.

    The live value answers "voice output is on"; only this answers "I'm
    talking more from here", which names no value and is true only of a turn
    that changed one. Walkthrough #3: the model narrated exactly that twice
    and the setting never moved.
    """
    return {
        _SETTERS[r["name"]]
        for r in tool_results
        if r["name"] in _SETTERS and r["result"].get("ok") is True
    }


def destructive_succeeded(tool_results: Sequence[dict[str, Any]]) -> bool:
    """Did a destructive op actually run this turn? A gate refusal is `ok:
    False`, so an armed-but-unconfirmed resign correctly counts as nothing
    happening."""
    return any(
        r["name"] in DESTRUCTIVE_TOOLS and r["result"].get("ok") is True
        for r in tool_results
    )
