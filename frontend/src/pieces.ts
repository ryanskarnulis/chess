// Piece glyphs and material values, shared by the captured-pieces panel.
// Piece symbols are the lowercase type letters the backend emits (`p`, `n`,
// `b`, `r`, `q`, `k`); colour is supplied by the caller.

export type PieceType = 'p' | 'n' | 'b' | 'r' | 'q' | 'k'

// Every glyph carries U+FE0E (text presentation): the black pawn ♟ is an
// emoji-default codepoint and otherwise renders as a colored emoji, oversized
// next to the other pieces.
export const PIECE_GLYPHS: Record<'white' | 'black', Record<PieceType, string>> = {
  white: { k: '♔︎', q: '♕︎', r: '♖︎', b: '♗︎', n: '♘︎', p: '♙︎' },
  black: { k: '♚︎', q: '♛︎', r: '♜︎', b: '♝︎', n: '♞︎', p: '♟︎' },
}

const PIECE_VALUES: Record<PieceType, number> = { p: 1, n: 3, b: 3, r: 5, q: 9, k: 0 }

// Spoken names for the glyphs, which a screen reader reads as nothing. The
// capture rows carry no visible label any more, so this is what the row's
// accessible name is built from.
export const PIECE_NAMES: Record<PieceType, string> = {
  k: 'king',
  q: 'queen',
  r: 'rook',
  b: 'bishop',
  n: 'knight',
  p: 'pawn',
}

/** Total material value of a list of piece-type symbols; unknowns count as 0. */
export function materialScore(symbols: string[]): number {
  return symbols.reduce((sum, s) => sum + (PIECE_VALUES[s as PieceType] ?? 0), 0)
}

/** One piece type in a capture cluster, and how many of it were taken. */
export interface CaptureGroup {
  symbol: string
  count: number
}

// The order a cluster reads in: heaviest first. Not derived from
// PIECE_VALUES, because bishop and knight tie there and the reading order
// still has to be stable.
const CAPTURE_ORDER: PieceType[] = ['q', 'r', 'b', 'n', 'p', 'k']

/**
 * Collapse a capture list into one entry per piece type, heaviest first.
 * This is what bounds a cluster's width: however long the game runs there
 * are only ever five types to show, so the row grows by a few px over a
 * whole game instead of overflowing the column. Symbols outside the known
 * set keep their place at the end rather than vanishing — the panel renders
 * them as "?", the same tolerance materialScore has in counting them zero.
 */
export function groupCaptures(symbols: string[]): CaptureGroup[] {
  const counts = new Map<string, number>()
  for (const s of symbols) counts.set(s, (counts.get(s) ?? 0) + 1)
  const rank = (s: string) => {
    const i = CAPTURE_ORDER.indexOf(s as PieceType)
    return i === -1 ? CAPTURE_ORDER.length : i
  }
  return [...counts]
    .map(([symbol, count]) => ({ symbol, count }))
    .sort((a, b) => rank(a.symbol) - rank(b.symbol) || a.symbol.localeCompare(b.symbol))
}
