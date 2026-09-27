"""The legal moves that fit a move described by its parts (#371).

The planner says what the player described — a piece, which one, where to —
and this works out which legal moves fit, so that "move my king's knight" is
asked with the g1 knight's two moves and never all four (#348), and "push my
e pawn" is asked rather than played (#357). The words are the model's to
read; which moves fit a piece on this board is not, and nothing here reads
language: every argument is a board term (a piece type, a square, a file, a
starting side, a square colour).

"Which one" is answered off the game's own history: the king's knight is the
knight that *started* on g1 (or g8), wherever it stands now, so the board's
move stack is replayed to follow each piece from its starting square. A board
set up from a FEN has no history, and there a starting side falls back to the
half of the board the piece stands on.
"""

from __future__ import annotations

import chess

PIECES: dict[str, chess.PieceType] = {
    "king": chess.KING,
    "queen": chess.QUEEN,
    "rook": chess.ROOK,
    "bishop": chess.BISHOP,
    "knight": chess.KNIGHT,
    "pawn": chess.PAWN,
}

# The starting side: the king's or the queen's half of the back rank.
KINGS_SIDE = "kings"
QUEENS_SIDE = "queens"
LIGHT = "light"
DARK = "dark"
WHICH_WORDS = (KINGS_SIDE, QUEENS_SIDE, LIGHT, DARK)

# A pawn is named after the piece in front of it ("the king's pawn" is the e
# pawn); every other piece after the half of the board it starts on.
_PAWN_SIDE_FILES = {KINGS_SIDE: {4}, QUEENS_SIDE: {3}}
_SIDE_FILES = {KINGS_SIDE: {4, 5, 6, 7}, QUEENS_SIDE: {0, 1, 2, 3}}


class PartsError(ValueError):
    """An argument that is not a board term this understands."""


def starting_squares(board: chess.Board) -> dict[chess.Square, chess.Square]:
    """Where each piece on `board` stood when the game began: current square →
    starting square. Built by replaying the move stack from its root; a piece
    that promoted keeps its pawn's starting square."""
    root = board.root()
    origin = {square: square for square in root.piece_map()}
    replay = root.copy(stack=False)
    for move in board.move_stack:
        if replay.is_castling(move):
            rook_from, rook_to = _castling_rook(replay, move)
            origin[rook_to] = origin.pop(rook_from, rook_from)
            origin[move.to_square] = origin.pop(move.from_square, move.from_square)
        else:
            if replay.is_en_passant(move):
                captured = chess.square(
                    chess.square_file(move.to_square),
                    chess.square_rank(move.from_square),
                )
                origin.pop(captured, None)
            origin[move.to_square] = origin.pop(move.from_square, move.from_square)
        replay.push(move)
    return origin


def _castling_rook(
    board: chess.Board, move: chess.Move
) -> tuple[chess.Square, chess.Square]:
    rank = chess.square_rank(move.from_square)
    if board.is_kingside_castling(move):
        return chess.square(7, rank), chess.square(5, rank)
    return chess.square(0, rank), chess.square(3, rank)


def _side_of(piece: chess.PieceType, start: chess.Square, which: str) -> bool:
    files = (_PAWN_SIDE_FILES if piece == chess.PAWN else _SIDE_FILES)[which]
    return chess.square_file(start) in files


def _origin_filter(
    board: chess.Board, piece: chess.PieceType, which: str
) -> set[chess.Square]:
    """The squares of the player's `piece`s that `which` picks out."""
    own = board.pieces(piece, board.turn)
    which = which.strip().lower()
    if which in (KINGS_SIDE, QUEENS_SIDE):
        started = starting_squares(board) if board.move_stack else {}
        return {
            square
            for square in own
            if _side_of(piece, started.get(square, square), which)
        }
    if which in (LIGHT, DARK):
        # chess.BB_LIGHT_SQUARES: a1 is dark.
        light = chess.BB_LIGHT_SQUARES
        return {
            square
            for square in own
            if bool(light & chess.BB_SQUARES[square]) == (which == LIGHT)
        }
    if len(which) == 1 and which in chess.FILE_NAMES:
        file = chess.FILE_NAMES.index(which)
        return {square for square in own if chess.square_file(square) == file}
    try:
        square = chess.parse_square(which)
    except ValueError:
        raise PartsError(
            f"which must be a square, a file, or one of {', '.join(WHICH_WORDS)};"
            f" got {which!r}"
        ) from None
    return {square} & set(own)


def fits(
    board: chess.Board, piece: str, which: str | None = None, to: str | None = None
) -> list[str]:
    """SAN of every legal move of the side to move that moves `piece` (a name
    from `PIECES`), from the one(s) `which` picks out, to `to` when given — in
    `board.legal_moves` order. Castling is the king's move to its square."""
    kind = PIECES.get(piece.strip().lower())
    if kind is None:
        raise PartsError(f"piece must be one of {', '.join(PIECES)}; got {piece!r}")
    origins = (
        _origin_filter(board, kind, which)
        if which
        else set(board.pieces(kind, board.turn))
    )
    target: chess.Square | None = None
    if to:
        try:
            target = chess.parse_square(to.strip().lower())
        except ValueError:
            raise PartsError(f"to must be a square; got {to!r}") from None
    fitting = []
    for move in board.legal_moves:
        if move.from_square not in origins:
            continue
        if target is not None and move.to_square != target:
            continue
        fitting.append(board.san(move))
    return fitting
