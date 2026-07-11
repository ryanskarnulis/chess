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

/** Total material value of a list of piece-type symbols; unknowns count as 0. */
export function materialScore(symbols: string[]): number {
  return symbols.reduce((sum, s) => sum + (PIECE_VALUES[s as PieceType] ?? 0), 0)
}
