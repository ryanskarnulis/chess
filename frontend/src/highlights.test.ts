import { describe, expect, it } from 'vitest'
import { boardHighlights, moveBetween } from './highlights'

// Positions as python-chess writes them (`Board.fen()`), with the SAN it
// writes for the move into each.
const START = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'
const E4 = 'rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1'
const E4_E5 = 'rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2'

// Fool's mate: 1. f3 e5 2. g4 Qh4#
const FOOL = [
  START,
  'rnbqkbnr/pppppppp/8/8/8/5P2/PPPPP1PP/RNBQKBNR b KQkq - 0 1',
  'rnbqkbnr/pppp1ppp/8/4p3/8/5P2/PPPPP1PP/RNBQKBNR w KQkq - 0 2',
  'rnbqkbnr/pppp1ppp/8/4p3/6P1/5P2/PPPPP2P/RNBQKBNR b KQkq - 0 2',
  'rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3',
]
const FOOL_SAN = ['f3', 'e5', 'g4', 'Qh4#']

describe('moveBetween', () => {
  it('finds a plain move, white or black', () => {
    expect(moveBetween(START, E4)).toEqual(['e2', 'e4'])
    expect(moveBetween(E4, E4_E5)).toEqual(['e7', 'e5'])
  })

  it('finds a capture', () => {
    expect(
      moveBetween(
        'rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2',
        'rnbqkbnr/ppp1pppp/8/3P4/8/8/PPPP1PPP/RNBQKBNR b KQkq - 0 2',
      ),
    ).toEqual(['e4', 'd5'])
  })

  it('marks the king for castling, both sides', () => {
    expect(
      moveBetween('4k3/8/8/8/8/8/8/4K2R w K - 0 1', '4k3/8/8/8/8/8/8/5RK1 b - - 1 1'),
    ).toEqual(['e1', 'g1'])
    expect(
      moveBetween('r3k3/8/8/8/8/8/8/4K3 b q - 0 1', '2kr4/8/8/8/8/8/8/4K3 w - - 1 2'),
    ).toEqual(['e8', 'c8'])
  })

  it('skips the pawn taken en passant', () => {
    expect(
      moveBetween('4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 2', '4k3/8/3P4/8/8/8/8/4K3 b - - 0 2'),
    ).toEqual(['e5', 'd6'])
  })

  it('finds a promotion', () => {
    expect(
      moveBetween('4k3/1P6/8/8/8/8/8/4K3 w - - 0 1', '1Q2k3/8/8/8/8/8/8/4K3 b - - 0 1'),
    ).toEqual(['b7', 'b8'])
  })
})

describe('boardHighlights', () => {
  it('has nothing to show at the root', () => {
    expect(boardHighlights([START], [], 0)).toEqual({ check: false })
  })

  it("shows the move into the position, whoever played it", () => {
    expect(boardHighlights([START, E4, E4_E5], ['e4', 'e5'], 2)).toEqual({
      lastMove: ['e7', 'e5'],
      check: false,
    })
  })

  it('flags the mated side to move', () => {
    expect(boardHighlights(FOOL, FOOL_SAN, 4)).toEqual({ lastMove: ['d8', 'h4'], check: 'white' })
  })

  it('reads an earlier ply for review', () => {
    expect(boardHighlights(FOOL, FOOL_SAN, 1)).toEqual({ lastMove: ['f2', 'f3'], check: false })
  })

  it('flags a plain check on the side to move', () => {
    // 1. e4 f5 2. Qh5+ — black in check, not mate.
    const before = 'rnbqkbnr/ppppp1pp/8/5p2/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2'
    const after = 'rnbqkbnr/ppppp1pp/8/5p1Q/4P3/8/PPPP1PPP/RNB1KBNR b KQkq - 1 2'
    expect(boardHighlights([before, after], ['Qh5+'], 1)).toEqual({
      lastMove: ['d1', 'h5'],
      check: 'black',
    })
  })
})
