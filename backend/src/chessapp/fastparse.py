"""Deterministic fast-parse for plain move commands (the seam BRIEF reserves).

Most commands in a game are just a move. `parse_move` maps an utterance that
is *entirely* one unambiguous legal move — notation ("e4", "Nf3", "e2e4") or
plain speech ("knight to f3", "takes on d5", "bishop takes", "take the h6
pawn", "castle kingside") — to that move's SAN, so the command pipeline can
skip the phase-one LLM call and go straight to make_move. It never guesses: the
utterance must match exactly one currently legal move, or it returns None and
the text falls through to the agent unchanged.

Pure function of (text, fen): legality and disambiguation come from the board
itself (python-chess), never from pattern cleverness. A phrase becomes
constraints — moving piece, target square, must-capture, captured piece,
promotion piece — and fires only when exactly one legal move satisfies them
all; a constraint the phrase omits is simply not applied, which is what lets a
capture name no square at all ("bishop takes" is exactly one move when the
bishop has exactly one capture). Zero matches
(illegal), several (ambiguous: two knights reach f3, four promotions land on
e8) and anything that isn't purely a move are all None. Illegal-move recovery
and clarifying questions stay the agent's job.

`parse_confirmation` is the same idea for the other deterministic answer the
pipeline needs: yes or no to a destructive-op confirmation question. Same rule
— it fires only on an utterance that is *entirely* a bare yes or no, and
anything carrying further intent ("yes, but undo first") falls through to the
agent. It is deliberately the narrowest thing that works: it only runs while an
op is armed, but a false positive throws a real game away.

`parse_resign` is the third: the player conceding. Same rule again — the
utterance must be *entirely* a resignation — but it can be more generous than
`parse_confirmation`, because the route it feeds sends `resign` through the
confirmation gate, so its worst mistake is a question.
"""

import re

import chess

_AFFIRMATIONS = frozenset(
    {
        "yes",
        "yeah",
        "yep",
        "yup",
        "y",
        "sure",
        "ok",
        "okay",
        "confirm",
        "confirmed",
        "do it",
        "go ahead",
        "yes please",
        "please do",
    }
)

_NEGATIONS = frozenset(
    {
        "no",
        "nope",
        "nah",
        "n",
        "cancel",
        "stop",
        "never mind",
        "nevermind",
        "no thanks",
        "no thank you",
        "forget it",
    }
)

_PUNCTUATION = re.compile(r"[.!?,]+")


def parse_confirmation(text: str) -> bool | None:
    """True for a bare yes, False for a bare no, None for anything else.

    Whole-utterance match, never a substring: "i don't know" is not a no and
    "yes, but undo my last move first" is not a yes — both carry intent only
    the agent can read, so both fall through.
    """
    normalized = _PUNCTUATION.sub("", text).strip().lower()
    normalized = " ".join(normalized.split())
    if normalized in _AFFIRMATIONS:
        return True
    if normalized in _NEGATIONS:
        return False
    return None


_RESIGNATIONS = frozenset(
    {
        "resign",
        "i resign",
        "im resigning",
        "i am resigning",
        "resign the game",
        "i resign the game",
        "i give up",
        "i concede",
        "i forfeit",
        "i quit",
        "i surrender",
    }
)

# Clauses that carry no intent of their own — a resignation is still entirely a
# resignation when it arrives behind one of these.
_FILLERS = frozenset(
    {
        "you know what",
        "ok",
        "okay",
        "alright",
        "right",
        "well",
        "fine",
        "yeah",
        "man",
        "damn",
        "thats it",
        "screw it",
    }
)

_CLAUSE_SPLIT = re.compile(r"[.!?,;]+")


def parse_resign(text: str) -> bool:
    """True when the utterance is *entirely* the player conceding the game.

    Resignation is not a judgment call — the utterance either says it or it
    doesn't — so the pipeline settles it and the model never gets a vote. Live,
    it got one and answered "Word. Game over." with no tool call on a live board.

    Whole-clause match, never a substring: every clause must be a resignation or
    a filler, so "you know what, i give up. i resign" fires and "i give up on
    this bishop" does not. Unlike `parse_confirmation` this can afford to be a
    little generous — the route it feeds dispatches `resign` through the same
    confirmation gate the agent hits, so a false positive costs a question, not
    a game.
    """
    normalized = text.lower().replace("'", "")
    clauses = [" ".join(c.split()) for c in _CLAUSE_SPLIT.split(normalized)]
    clauses = [c for c in clauses if c]
    if not clauses:
        return False
    if not all(c in _RESIGNATIONS or c in _FILLERS for c in clauses):
        return False
    return any(c in _RESIGNATIONS for c in clauses)


_PIECE_WORDS = {
    "pawn": chess.PAWN,
    "knight": chess.KNIGHT,
    "night": chess.KNIGHT,  # STT homophone
    "bishop": chess.BISHOP,
    "rook": chess.ROOK,
    "queen": chess.QUEEN,
    "king": chess.KING,
}

_PROMO_CLAUSE = (
    r"(?:\s+promot(?:e|es|ing|ion)\s+to\s+(?:a\s+)?"
    r"(?P<promo>queen|rook|bishop|knight|night))?"
)

# Anchored: the whole utterance must be the move phrase, so "can I play e4?"
# never fires. "to" implies nothing about capturing (people say "queen to d5"
# for captures too); takes/captures require one.
_PHRASE = re.compile(
    r"^(?:(?P<piece>pawn|knight|night|bishop|rook|queen|king)\s+)?"
    r"(?:(?P<verb>to|takes|take|captures|capture)\s+)?(?:on\s+)?"
    r"(?P<square>[a-h][1-8])" + _PROMO_CLAUSE + r"$"
)

# "d takes e5" — a pawn capture named by its source file, how players
# pronounce dxe5 (SAN semantics: piece captures are named by the piece, so
# the file form always means a pawn). "x"/"ex" cover STT's hearing of "dxe5".
_FILE_CAPTURE = re.compile(
    r"^(?P<file>[a-h])\s+(?:takes|take|captures|capture|x|ex)\s+(?:on\s+)?"
    r"(?P<square>[a-h][1-8])" + _PROMO_CLAUSE + r"$"
)

_PIECE_WORD = r"pawn|knight|night|bishop|rook|queen|king"
_CAPTURE_VERB = r"takes|take|captures|capture"

# A capture that never names the target square: the player names the capturing
# piece ("bishop takes"), the victim ("bishop takes pawn", "take the h6 pawn"),
# or neither ("takes"). Each is a constraint set like any other phrase, so it
# fires only when exactly one legal capture satisfies it — two bishops that can
# both take stay ambiguous and fall through to the agent.
_CAPTURE = re.compile(
    rf"^(?:(?P<piece>{_PIECE_WORD})\s+)?(?:{_CAPTURE_VERB})"
    rf"(?:\s+(?:the\s+|a\s+)?"
    rf"(?:(?P<victim>{_PIECE_WORD})(?:\s+(?:on\s+)?(?P<square>[a-h][1-8]))?"
    rf"|(?P<square_first>[a-h][1-8])\s+(?P<victim_last>{_PIECE_WORD})))?$"
)

_SIDE_WORD = r"king\s?side|queen\s?side|short|long"
_CASTLE_WORD = r"(?:castles?|castling)"
_CASTLE = re.compile(
    rf"^(?:{_CASTLE_WORD}(?:\s+(?P<after>{_SIDE_WORD}))?"
    rf"|(?P<before>{_SIDE_WORD})\s+{_CASTLE_WORD})$"
)


def parse_move(text: str, fen: str) -> str | None:
    """The SAN of the single legal move `text` unambiguously names, else None."""
    raw = " ".join(re.sub(r"[.,!?;:'\"]", " ", text).split())
    if not raw:
        return None
    lowered = raw.lower().replace("0-0-0", "o-o-o").replace("0-0", "o-o")
    board = chess.Board(fen)

    castle = _CASTLE.match(lowered)
    if castle:
        side = castle.group("after") or castle.group("before")
        return _single_san(board, _castling_moves(board, side))
    move = _notation_match(board, raw, lowered)
    if move is not None:
        return board.san(move)
    file_capture = _FILE_CAPTURE.match(lowered)
    if file_capture:
        promo = file_capture.group("promo")
        candidates = _constrained_moves(
            board,
            piece=chess.PAWN,
            from_file=chess.FILE_NAMES.index(file_capture.group("file")),
            target=chess.parse_square(file_capture.group("square")),
            capture=True,
            promotion=_PIECE_WORDS[promo] if promo else None,
        )
        return _single_san(board, candidates)
    phrase = _PHRASE.match(lowered)
    if phrase is not None:
        promo = phrase.group("promo")
        piece = phrase.group("piece")
        candidates = _constrained_moves(
            board,
            piece=_PIECE_WORDS[piece] if piece else None,
            target=chess.parse_square(phrase.group("square")),
            capture=phrase.group("verb") in ("takes", "take", "captures", "capture"),
            promotion=_PIECE_WORDS[promo] if promo else None,
        )
        return _single_san(board, candidates)
    capture = _CAPTURE.match(lowered)
    if capture is None:
        return None
    victim = capture.group("victim") or capture.group("victim_last")
    square = capture.group("square") or capture.group("square_first")
    candidates = _constrained_moves(
        board,
        piece=_PIECE_WORDS[capture.group("piece")] if capture.group("piece") else None,
        target=chess.parse_square(square) if square else None,
        capture=True,
        victim=_PIECE_WORDS[victim] if victim else None,
        promotion=None,
    )
    return _single_san(board, candidates)


def _single_san(board: chess.Board, moves: list[chess.Move]) -> str | None:
    return board.san(moves[0]) if len(moves) == 1 else None


def _notation_match(board: chess.Board, raw: str, lowered: str) -> chess.Move | None:
    """The legal move `raw` names as SAN or UCI, forgiving case and check
    suffixes. When lowercasing collides ("bxc6" the pawn vs "Bxc6" the
    bishop), the exact case as typed settles it; otherwise a collision is
    ambiguous."""
    exact: list[chess.Move] = []
    folded: list[chess.Move] = []
    bare_raw = raw.rstrip("+#")
    for move in board.legal_moves:
        san = board.san(move)
        bare = san.rstrip("+#")
        if bare_raw == bare:
            exact.append(move)
        if lowered.rstrip("+#") == bare.lower() or lowered == move.uci():
            folded.append(move)
    if len(folded) == 1:
        return folded[0]
    if len(folded) > 1 and len(exact) == 1:
        return exact[0]
    return None


def _castling_moves(board: chess.Board, side: str | None) -> list[chess.Move]:
    kingside = side is None or side.startswith(("king", "short"))
    queenside = side is None or side.startswith(("queen", "long"))
    return [
        move
        for move in board.legal_moves
        if (kingside and board.is_kingside_castling(move))
        or (queenside and board.is_queenside_castling(move))
    ]


def _victim_of(board: chess.Board, move: chess.Move) -> chess.PieceType | None:
    """The piece type `move` captures, or None if it captures nothing. En
    passant takes a pawn that isn't standing on the square the capturer lands
    on, so the board can't be read there."""
    if board.is_en_passant(move):
        return chess.PAWN
    return board.piece_type_at(move.to_square)


def _constrained_moves(
    board: chess.Board,
    *,
    piece: chess.PieceType | None,
    target: chess.Square | None,
    capture: bool,
    promotion: chess.PieceType | None,
    victim: chess.PieceType | None = None,
    from_file: int | None = None,
) -> list[chess.Move]:
    """Every legal move satisfying all the phrase's constraints. Each constraint
    left as None simply isn't applied — so "bishop takes" constrains the piece
    and the capture and nothing else, and matches every capture that bishop can
    make. No promotion constraint means promotion moves still match: "pawn to
    e8" hits all four and stays ambiguous, which is the right question for the
    agent."""
    matches = []
    for move in board.legal_moves:
        if target is not None and move.to_square != target:
            continue
        if victim is not None and _victim_of(board, move) != victim:
            continue
        if piece is not None and board.piece_at(move.from_square).piece_type != piece:
            continue
        if from_file is not None and chess.square_file(move.from_square) != from_file:
            continue
        if capture and not board.is_capture(move):
            continue
        if promotion is not None and move.promotion != promotion:
            continue
        matches.append(move)
    return matches
