import { describe, expect, it } from 'vitest'
import { pairMoves } from './moves'

describe('pairMoves', () => {
  it('is empty for no moves', () => {
    expect(pairMoves([])).toEqual([])
  })

  it('pairs white and black moves under one move number', () => {
    expect(pairMoves(['e4', 'e5', 'Nf3', 'Nc6'])).toEqual([
      { number: 1, white: 'e4', black: 'e5' },
      { number: 2, white: 'Nf3', black: 'Nc6' },
    ])
  })

  it('leaves black null when white has the last (incomplete) move', () => {
    expect(pairMoves(['e4', 'e5', 'Nf3'])).toEqual([
      { number: 1, white: 'e4', black: 'e5' },
      { number: 2, white: 'Nf3', black: null },
    ])
  })
})
