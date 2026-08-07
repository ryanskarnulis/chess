import { describe, expect, it } from 'vitest'
import { groupCaptures, materialScore, PIECE_GLYPHS, PIECE_NAMES } from './pieces'

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

  it('names every piece type it has a glyph for', () => {
    for (const type of ['p', 'n', 'b', 'r', 'q', 'k'] as const) {
      expect(PIECE_NAMES[type]).toBeTruthy()
    }
  })
})

// Grouping is what bounds a cluster's width: a long game captures many
// pieces but only ever five *types*, so the row grows by a few px over a
// whole game instead of overflowing the 390px column.
describe('groupCaptures', () => {
  it('is empty for no captures', () => {
    expect(groupCaptures([])).toEqual([])
  })

  it('collapses repeats of a type into one entry with a count', () => {
    expect(groupCaptures(['p', 'p', 'p'])).toEqual([{ symbol: 'p', count: 3 }])
  })

  it('orders types by descending value, not capture order', () => {
    // Taken pawn-first; read queen-first.
    expect(groupCaptures(['p', 'q', 'n', 'r', 'b'])).toEqual([
      { symbol: 'q', count: 1 },
      { symbol: 'r', count: 1 },
      { symbol: 'b', count: 1 },
      { symbol: 'n', count: 1 },
      { symbol: 'p', count: 1 },
    ])
  })

  it('keeps unknown symbols last rather than dropping them', () => {
    // The panel stays tolerant of a symbol it does not know (it renders "?"),
    // the same way materialScore counts it as zero.
    expect(groupCaptures(['x', 'p'])).toEqual([
      { symbol: 'p', count: 1 },
      { symbol: 'x', count: 1 },
    ])
  })
})
