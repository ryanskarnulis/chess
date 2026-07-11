"""Deterministic fast-parse for plain move commands (the seam BRIEF reserves).

Most commands in a game are just a move. `parse_move` maps an utterance that
is *entirely* one unambiguous legal move — notation ("e4", "Nf3", "e2e4") or
plain speech ("knight to f3", "takes on d5", "castle kingside") — to that
move's SAN, so the command pipeline can skip the phase-one LLM call and go
straight to make_move. It never guesses: the utterance must match exactly one
currently legal move, or it returns None and the text falls through to the
agent unchanged.

Pure function of (text, fen): legality and disambiguation come from the board
itself (python-chess), never from pattern cleverness. A phrase becomes
constraints — moving piece, target square, must-capture, promotion piece —
and fires only when exactly one legal move satisfies them all; zero matches
(illegal), several (ambiguous: two knights reach f3, four promotions land on
e8) and anything that isn't purely a move are all None. Illegal-move recovery
and clarifying questions stay the agent's job.
"""

import re

import chess

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
    if phrase is None:
        return None
    promo = phrase.group("promo")
    candidates = _constrained_moves(
        board,
        piece=_PIECE_WORDS[phrase.group("piece")] if phrase.group("piece") else None,
        target=chess.parse_square(phrase.group("square")),
        capture=phrase.group("verb") in ("takes", "take", "captures", "capture"),
        promotion=_PIECE_WORDS[promo] if promo else None,
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


def _constrained_moves(
    board: chess.Board,
    *,
    piece: chess.PieceType | None,
    target: chess.Square,
    capture: bool,
    promotion: chess.PieceType | None,
    from_file: int | None = None,
) -> list[chess.Move]:
    """Every legal move satisfying all the phrase's constraints. No promotion
    constraint means promotion moves still match — so "pawn to e8" hits all
    four and stays ambiguous, which is the right question for the agent."""
    matches = []
    for move in board.legal_moves:
        if move.to_square != target:
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
