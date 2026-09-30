"""Vendor Lichess's chess-openings data as `src/chessapp/data/openings.tsv` (#373).

    git clone https://github.com/lichess-org/chess-openings /tmp/chess-openings
    python scripts/build_openings.py /tmp/chess-openings

Concatenates a.tsv..e.tsv and adds what the app would otherwise work out on
every start: each line's length in plies and the key of the position it ends
in (`openings._key`). Parsing 3,800 PGN lines cost 0.86 s cold, on the first
turn that read the state block; reading the keys costs a few milliseconds.
Never edit the output by hand: re-run this.
"""

import argparse
import io
import subprocess
import sys
from pathlib import Path

import chess.pgn

from chessapp.openings import _key

OUT = Path(__file__).resolve().parents[1] / "src/chessapp/data/openings.tsv"
SOURCES = ("a.tsv", "b.tsv", "c.tsv", "d.tsv", "e.tsv")


def rows(source: Path) -> list[str]:
    out = []
    for name in SOURCES:
        lines = (source / name).read_text().splitlines()
        if lines[0] != "eco\tname\tpgn":
            raise SystemExit(f"{name}: unexpected header {lines[0]!r}")
        for line in lines[1:]:
            eco, opening, pgn = line.split("\t")
            game = chess.pgn.read_game(io.StringIO(pgn))
            if game is None or game.errors:
                raise SystemExit(f"{name}: unreadable line {line!r}")
            board = game.board()
            for move in game.mainline_moves():
                board.push(move)
            plies = len(board.move_stack)
            out.append(f"{eco}\t{opening}\t{pgn}\t{plies}\t{_key(board.fen())}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source", type=Path, help="a chess-openings checkout")
    args = parser.parse_args()
    commit = subprocess.run(
        ["git", "-C", str(args.source), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    header = [
        "# Opening names and ECO codes: Lichess chess-openings "
        "(https://github.com/lichess-org/chess-openings),",
        f"# commit {commit}, a.tsv..e.tsv. CC0 1.0 (public domain).",
        "# Written by scripts/build_openings.py; never edit a line by hand.",
        "eco\tname\tpgn\tplies\tkey",
    ]
    OUT.write_text("\n".join(header + rows(args.source)) + "\n")
    print(f"wrote {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
