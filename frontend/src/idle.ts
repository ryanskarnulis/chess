/**
 * What the agent bubble says with no reply to show and nothing in flight:
 * "Your move." while the game is on, the player's verdict once it is over —
 * a finished game is nobody's move (#458). App-owned wording, like the draw
 * answer, never Glitch's — and worded apart from the status row's "Game over —
 * 1-0" and the post-game screen's "You won", which say the same thing nearby.
 */

import type { GameState } from './api'

export const YOUR_MOVE = 'Your move.'

type Finish = Pick<GameState, 'game_over' | 'outcome' | 'player_color'>

export function idleLine(state: Finish | null): string {
  if (!state?.game_over) return YOUR_MOVE
  const winner = state.outcome?.winner ?? null
  if (winner === null) return 'Finished — drawn.'
  return winner === state.player_color ? 'Finished — the win is yours.' : 'Finished — Glitch took it.'
}
