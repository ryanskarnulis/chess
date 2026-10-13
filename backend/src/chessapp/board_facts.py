"""Board facts: what a player sees at a glance at a position (#496,
docs/glitch-sight.md decisions 1 and 2).

The Glitch tier's mover reads the board as text, which leaves him playing
blindfold: he has to work out from a grid of letters what attacks and
defends what, and he gets it wrong (game `927c238e`, 17… Qxd4 for a pawn a
pawn defended). These facts are his eyes, not his judgement. The test for a
fact: could a player point at it on the board without moving a piece in his
head and without deciding anything?

So this module says where every piece stands, which pieces are attacked and
by what, what defends them, which pieces are pinned to their king, whether
the side to move is in check, and the material in 1/3/3/5/9. Attackers and
defenders are direct ones: a piece behind another on the same line (an
x-ray, a battery) is not counted, and a pinned defender still counts, with
the pin listed separately.

It never concludes. There is no "hanging", no "safe", no list of the other
side's captures or checks, no mate marked, no evaluation and no search:
weighing an attack is the mover's job, and looking ahead is `try_line`'s,
where he names the line (#498). It is pure python-chess and never imports
the engine (`tests/test_board_facts.py` pins that).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import chess

# The usual count. The king has no material value.
PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0,
}

# Pieces listed king first, pawns last, as a score sheet's diagram key reads.
_LIST_ORDER = (chess.KING, chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT)

_SIDE = {chess.WHITE: "white", chess.BLACK: "black"}


def piece_label(piece: chess.Piece, square: chess.Square) -> str:
    """ "Qd7", "Nf3", or "pawn c3": a piece's SAN letter and square, with
    pawns named, since a bare square reads as a square."""
    name = chess.square_name(square)
    if piece.piece_type == chess.PAWN:
        return f"pawn {name}"
    return f"{piece.symbol().upper()}{name}"


@dataclass(frozen=True)
class AttackedPiece:
    """A piece (not a king) the other side attacks directly, with its direct
    attackers and its own side's direct defenders, cheapest first."""

    square: str
    label: str
    attackers: tuple[str, ...]
    defenders: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "square": self.square,
            "piece": self.label,
            "attackers": list(self.attackers),
            "defenders": list(self.defenders),
        }


@dataclass(frozen=True)
class Pin:
    """A piece that may not leave its line because its king stands behind
    it, and the piece pinning it."""

    side: str
    pinned: str
    by: str

    def to_dict(self) -> dict[str, Any]:
        return {"side": self.side, "pinned": self.pinned, "by": self.by}


@dataclass(frozen=True)
class BoardFacts:
    """Everything this module says about one position. `pieces` and
    `attacked` are keyed "white" / "black"; `checkers` is empty unless the
    side to move is in check."""

    turn: str
    pieces: dict[str, tuple[str, ...]]
    material: dict[str, int]
    attacked: dict[str, tuple[AttackedPiece, ...]]
    pins: tuple[Pin, ...]
    checkers: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn": self.turn,
            "pieces": {side: list(labels) for side, labels in self.pieces.items()},
            "material": dict(self.material),
            "attacked": {
                side: [piece.to_dict() for piece in pieces]
                for side, pieces in self.attacked.items()
            },
            "pins": [pin.to_dict() for pin in self.pins],
            "checkers": list(self.checkers),
        }

    def text(self) -> str:
        """The facts as the prompt section, the side to move first."""
        order = [self.turn, _other(self.turn)]
        lines = ["Board facts (what is on the board; nothing here judges it):"]
        for side in order:
            lines.append(f"{side.capitalize()}: {_piece_list(self.pieces[side])}.")
        lines.append(
            f"Material: White {self.material['white']}, Black {self.material['black']}."
        )
        for side in order:
            attacked = self.attacked[side]
            if not attacked:
                lines.append(f"{side.capitalize()}'s pieces under attack: none.")
                continue
            lines.append(f"{side.capitalize()}'s pieces under attack:")
            for piece in attacked:
                defended = ", ".join(piece.defenders) or "nothing"
                lines.append(
                    f"  {piece.label}: attacked by {', '.join(piece.attackers)}; "
                    f"defended by {defended}"
                )
        if self.pins:
            pinned = "; ".join(
                f"{pin.side.capitalize()}'s {pin.pinned}, by {pin.by}"
                for pin in self.pins
            )
            lines.append(f"Pinned to the king: {pinned}.")
        else:
            lines.append("Pinned to the king: none.")
        if self.checkers:
            lines.append(
                f"{self.turn.capitalize()} is in check from {', '.join(self.checkers)}."
            )
        return "\n".join(lines)


def board_facts(board: chess.Board) -> BoardFacts:
    """The facts of `board`, from the board alone."""
    return BoardFacts(
        turn=_SIDE[board.turn],
        pieces={_SIDE[color]: _pieces(board, color) for color in chess.COLORS},
        material={_SIDE[color]: material(board, color) for color in chess.COLORS},
        attacked={_SIDE[color]: _attacked(board, color) for color in chess.COLORS},
        pins=tuple(pin for color in chess.COLORS for pin in _pins(board, color)),
        checkers=_labels(board, board.checkers()),
    )


def material(board: chess.Board, color: chess.Color) -> int:
    """`color`'s material in 1/3/3/5/9."""
    return sum(
        PIECE_VALUES[piece_type] * len(board.pieces(piece_type, color))
        for piece_type in chess.PIECE_TYPES
    )


def _other(side: str) -> str:
    return "black" if side == "white" else "white"


def _piece_list(labels: tuple[str, ...]) -> str:
    pieces = [label for label in labels if not label.startswith("pawn ")]
    pawns = [label.removeprefix("pawn ") for label in labels if label not in pieces]
    if pawns:
        pieces.append(f"pawns {' '.join(pawns)}")
    return ", ".join(pieces)


def _pieces(board: chess.Board, color: chess.Color) -> tuple[str, ...]:
    """`color`'s pieces, king first and pawns last, each kind by file."""
    return tuple(
        piece_label(chess.Piece(piece_type, color), square)
        for piece_type in (*_LIST_ORDER, chess.PAWN)
        for square in sorted(board.pieces(piece_type, color), key=_by_file)
    )


def _by_file(square: chess.Square) -> tuple[int, int]:
    return chess.square_file(square), chess.square_rank(square)


def _attacked(board: chess.Board, color: chess.Color) -> tuple[AttackedPiece, ...]:
    """`color`'s pieces, king aside, that the other side attacks: most
    valuable first, then by square."""
    found = []
    for square, piece in board.piece_map().items():
        if piece.color != color or piece.piece_type == chess.KING:
            continue
        attackers = board.attackers(not color, square) | _en_passant_attackers(
            board, square
        )
        if not attackers:
            continue
        found.append(
            (
                -PIECE_VALUES[piece.piece_type],
                square,
                AttackedPiece(
                    square=chess.square_name(square),
                    label=piece_label(piece, square),
                    attackers=_labels(board, attackers),
                    defenders=_labels(board, board.attackers(color, square)),
                ),
            )
        )
    return tuple(entry for _, _, entry in sorted(found, key=lambda e: e[:2]))


def _en_passant_attackers(board: chess.Board, square: chess.Square) -> chess.SquareSet:
    """The side to move's pawns that may take the pawn on `square` en
    passant: it has just moved two squares past them."""
    if board.ep_square is None:
        return chess.SquareSet()
    behind = -8 if board.turn == chess.WHITE else 8
    if board.ep_square + behind != square:
        return chess.SquareSet()
    return board.attackers(board.turn, board.ep_square) & board.pieces(
        chess.PAWN, board.turn
    )


def _pins(board: chess.Board, color: chess.Color) -> list[Pin]:
    pins = []
    king = board.king(color)
    for square, piece in sorted(board.piece_map().items()):
        if piece.color != color or piece.piece_type == chess.KING:
            continue
        if not board.is_pinned(color, square):
            continue
        pinner = _pinner(board, king, square, board.pin(color, square))
        pins.append(
            Pin(
                side=_SIDE[color],
                pinned=piece_label(piece, square),
                by=piece_label(board.piece_at(pinner), pinner),
            )
        )
    return pins


def _pinner(
    board: chess.Board, king: chess.Square, pinned: chess.Square, ray: chess.SquareSet
) -> chess.Square:
    """The other side's piece on `ray` with only `pinned` between it and
    `king`. The ray runs the whole line, so a piece further along is not it."""
    for square in ray & board.occupied_co[not board.color_at(pinned)]:
        between = chess.SquareSet(chess.between(king, square)) & board.occupied
        if between == chess.SquareSet([pinned]):
            return square
    raise AssertionError(f"no pinner for {chess.square_name(pinned)}")


def _labels(board: chess.Board, squares: chess.SquareSet | int) -> tuple[str, ...]:
    """The pieces on `squares`, cheapest first, then by square."""
    placed = [(square, board.piece_at(square)) for square in chess.SquareSet(squares)]
    placed.sort(key=lambda sp: (PIECE_VALUES[sp[1].piece_type] or 100, sp[0]))
    return tuple(piece_label(piece, square) for square, piece in placed)
