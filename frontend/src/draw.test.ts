import { describe, expect, it } from 'vitest'
import { drawAnswer } from './draw'

describe('drawAnswer', () => {
  it('says the game is drawn only when the result says so', () => {
    expect(drawAnswer(true, null)).toMatch(/draw agreed/i)
    for (const reason of ['engine_ahead', 'player_ahead', 'not_an_endgame', 'too_early']) {
      expect(drawAnswer(false, reason)).toMatch(/declined/i)
      expect(drawAnswer(false, reason)).not.toMatch(/agreed/i)
    }
  })

  it('names each reason in the player’s language', () => {
    expect(drawAnswer(false, 'engine_ahead')).toMatch(/glitch is ahead/i)
    expect(drawAnswer(false, 'player_ahead')).toMatch(/you.re the one ahead/i)
    expect(drawAnswer(false, 'not_an_endgame')).toMatch(/on the board/i)
    expect(drawAnswer(false, 'too_early')).toMatch(/make a move first/i)
  })

  it('still declines truthfully for a reason it has never heard of', () => {
    expect(drawAnswer(false, 'stockfish_is_tired')).toBe('Draw declined. Play on.')
    expect(drawAnswer(false, null)).toBe('Draw declined. Play on.')
  })
})
