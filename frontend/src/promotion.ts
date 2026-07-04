// Promotion is the one move the board can't express with bare origin/dest:
// a pawn reaching the last rank needs a piece to promote to. The backend
// stays the move-truth source — this only detects that a choice is required
// and carries the chosen piece's UCI letter; it never decides legality.

/** Promotion targets as their UCI suffix letters. */
export type PromotionPiece = 'q' | 'r' | 'b' | 'n'

/** The piece-placement character on `square` (e.g. 'P', 'n'), or null if empty. */
function pieceAt(fen: string, square: string): string | null {
  const file = square.charCodeAt(0) - 'a'.charCodeAt(0) // a..h -> 0..7
  const rank = Number(square[1]) // 1..8
  const rows = fen.split(' ')[0].split('/')
  const row = rows[8 - rank] // FEN lists rank 8 first
  if (row === undefined) return null
  let f = 0
  for (const ch of row) {
    if (ch >= '1' && ch <= '8') {
      f += Number(ch)
    } else {
      if (f === file) return ch
      f += 1
    }
  }
  return null
}

/** True if moving from→to is a pawn reaching the promotion rank in `fen`. */
export function isPromotion(fen: string, from: string, to: string): boolean {
  const piece = pieceAt(fen, from)
  if (piece === null || piece.toLowerCase() !== 'p') return false
  const toRank = to[1]
  return (piece === 'P' && toRank === '8') || (piece === 'p' && toRank === '1')
}
