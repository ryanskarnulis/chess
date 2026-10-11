// What the board highlights for a position: the move that led to it and a
// king in check. Both are read off data the backend already sends — the FEN
// of every position (`fens`) and python-chess's SAN for every ply (`history`)
// — so nothing here decides anything about the rules: a check is the `+`/`#`
// python-chess wrote, and a move is the difference between two positions it
// produced.

type Color = 'white' | 'black'

export interface BoardHighlights {
  /** Origin and destination of the move into this position, if there was one. */
  lastMove?: [string, string]
  /** The side whose king is in check here (mated included), or false. */
  check: Color | false
}

const FILES = 'abcdefgh'

/** Square → piece letter (`P` white, `p` black) from a FEN's placement field. */
function placement(fen: string): Map<string, string> {
  const squares = new Map<string, string>()
  fen
    .split(' ')[0]
    .split('/')
    .forEach((row, i) => {
      const rank = 8 - i
      let file = 0
      for (const ch of row) {
        if (/\d/.test(ch)) file += Number(ch)
        else squares.set(`${FILES[file++]}${rank}`, ch)
      }
    })
  return squares
}

const colorOf = (piece: string): Color => (piece === piece.toUpperCase() ? 'white' : 'black')

/**
 * The origin and destination of the one legal move from `before` to `after`.
 * A plain move or a promotion changes one square each way; en passant also
 * empties the captured pawn's square (the opponent's, so it is skipped); and
 * castling moves a rook too, so where the mover touched two squares the
 * king's are the move — the same squares chessground marks when the player
 * castles by dragging the king.
 */
export function moveBetween(before: string, after: string): [string, string] | undefined {
  const from = placement(before)
  const to = placement(after)
  const emptied: string[] = []
  const filled: string[] = []
  for (const sq of from.keys()) if (!to.has(sq)) emptied.push(sq)
  for (const [sq, piece] of to) if (from.get(sq) !== piece) filled.push(sq)
  if (filled.length === 0) return undefined
  const mover = colorOf(to.get(filled[0])!)
  const pick = (squares: string[], board: Map<string, string>) => {
    const own = squares.filter((sq) => colorOf(board.get(sq)!) === mover)
    return own.length > 1 ? own.find((sq) => board.get(sq)!.toLowerCase() === 'k') : own[0]
  }
  const orig = pick(emptied, from)
  const dest = pick(filled, to)
  return orig && dest ? [orig, dest] : undefined
}

/**
 * The highlights for position `ply` of a game (0 = the root), given its
 * `fens` (root first) and SAN `history` (one per ply). Reviewing an earlier
 * position asks for that ply, so it shows the move into it and its check.
 */
export function boardHighlights(fens: string[], history: string[], ply: number): BoardHighlights {
  if (ply <= 0 || ply >= fens.length) return { check: false }
  const san = history[ply - 1] ?? ''
  const toMove: Color = fens[ply].split(' ')[1] === 'b' ? 'black' : 'white'
  return {
    lastMove: moveBetween(fens[ply - 1], fens[ply]),
    check: /[+#]$/.test(san) ? toMove : false,
  }
}
