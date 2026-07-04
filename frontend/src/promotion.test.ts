import { describe, expect, it } from 'vitest'
import { isPromotion } from './promotion'

// A white pawn on e7, one square from promoting, with a black rook on d8 to
// capture into. Enough of a position to exercise the pawn/last-rank checks.
const WHITE_PROMO_FEN = '3r2k1/4P3/8/8/8/8/8/4K3 w - - 0 1'
// Mirror image: a black pawn on e2, white rook on d1.
const BLACK_PROMO_FEN = '4k3/8/8/8/8/8/4p3/3R2K1 b - - 0 1'
const START_FEN = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'

describe('isPromotion', () => {
  it('is true for a white pawn pushing to the last rank', () => {
    expect(isPromotion(WHITE_PROMO_FEN, 'e7', 'e8')).toBe(true)
  })

  it('is true for a white pawn capturing onto the last rank', () => {
    expect(isPromotion(WHITE_PROMO_FEN, 'e7', 'd8')).toBe(true)
  })

  it('is true for a black pawn pushing to the first rank', () => {
    expect(isPromotion(BLACK_PROMO_FEN, 'e2', 'e1')).toBe(true)
  })

  it('is false for an ordinary pawn move', () => {
    expect(isPromotion(START_FEN, 'e2', 'e4')).toBe(false)
  })

  it('is false when the moving piece is not a pawn', () => {
    // The black rook reaching rank 1 is not a promotion.
    expect(isPromotion(BLACK_PROMO_FEN, 'd1', 'd8')).toBe(false)
  })

  it('is false for an empty origin square', () => {
    expect(isPromotion(START_FEN, 'e4', 'e5')).toBe(false)
  })
})
