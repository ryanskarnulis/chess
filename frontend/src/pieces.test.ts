import { describe, expect, it } from 'vitest'
import { materialScore, PIECE_GLYPHS } from './pieces'

describe('materialScore', () => {
  it('is zero for no captures', () => {
    expect(materialScore([])).toBe(0)
  })

  it('sums standard piece values', () => {
    // pawn + knight + rook + queen = 1 + 3 + 5 + 9
    expect(materialScore(['p', 'n', 'r', 'q'])).toBe(18)
  })

  it('ignores unknown symbols', () => {
    expect(materialScore(['p', 'x'])).toBe(1)
  })
})

describe('PIECE_GLYPHS', () => {
  it('has distinct white and black glyphs for every piece type', () => {
    for (const type of ['p', 'n', 'b', 'r', 'q', 'k'] as const) {
      expect(PIECE_GLYPHS.white[type]).not.toBe(PIECE_GLYPHS.black[type])
    }
  })
})
