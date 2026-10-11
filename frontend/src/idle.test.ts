import { describe, expect, it } from 'vitest'
import { YOUR_MOVE, idleLine } from './idle'

describe('idleLine', () => {
  it('is "Your move." while the game is on, and before any state', () => {
    expect(idleLine(null)).toBe(YOUR_MOVE)
    expect(idleLine({ game_over: false, outcome: null, player_color: 'white' })).toBe(YOUR_MOVE)
  })

  it('gives the player’s verdict once the game is over, never "Your move."', () => {
    const over = (winner: 'white' | 'black' | null) => ({
      game_over: true,
      outcome: { termination: 'checkmate', winner, result: '' },
    })
    expect(idleLine({ ...over('black'), player_color: 'black' })).toMatch(/win is yours/i)
    expect(idleLine({ ...over('white'), player_color: 'black' })).toMatch(/glitch took it/i)
    expect(idleLine({ ...over(null), player_color: 'white' })).toMatch(/drawn/i)
    expect(idleLine({ game_over: true, outcome: null, player_color: 'white' })).not.toBe(YOUR_MOVE)
  })
})
