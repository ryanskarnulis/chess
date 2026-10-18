import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { EvalGraph } from './EvalGraph'
import type { GameReview, ReviewedMove } from './api'

function move(overrides: Partial<ReviewedMove>): ReviewedMove {
  return {
    san: 'e4',
    uci: 'e2e4',
    color: 'white',
    cp_loss: 0,
    classification: 'good',
    best: 'e4',
    accuracy: 100,
    move_number: 1,
    player_eval_cp: 0,
    mate: null,
    ...overrides,
  }
}

// The issue's own shape: down two, up three, back to minus one.
const SWING: GameReview = {
  moves: [
    move({ san: 'e4', player_eval_cp: -200 }),
    move({ san: 'e5', color: 'black', player_eval_cp: 300 }),
    move({
      san: 'Qh5',
      move_number: 2,
      player_eval_cp: -100,
      classification: 'blunder',
      cp_loss: 400,
      best: 'Nf3',
    }),
  ],
  accuracy: { white: 60, black: 90 },
  counts: {},
  player_color: 'white',
  start: { player_eval_cp: 20, mate: null },
}

function points(): [number, number][] {
  const line = document.querySelector('.eval-line')!.getAttribute('points')!
  return line.split(' ').map((p) => p.split(',').map(Number) as [number, number])
}

describe('EvalGraph', () => {
  it('draws one point per position, from the start, on the player’s side', () => {
    render(<EvalGraph review={SWING} />)
    const plot = screen.getByRole('img', { name: /evaluation from your side/i })
    expect(plot).toHaveAccessibleName(/starts \+0\.2, peaks at \+3\.0, dips to −2\.0, ends −1\.0/)
    // Reach is ±3 pawns (the swing's own peak): y = 3 − pawns, so up is better.
    expect(points()).toEqual([
      [0, 2.8],
      [1, 5],
      [2, 0],
      [3, 4],
    ])
    expect(screen.getByText('+3')).toBeInTheDocument()
    expect(screen.getByText('−3')).toBeInTheDocument()
  })

  it('marks the blunders on the line', () => {
    render(<EvalGraph review={SWING} />)
    const dots = document.querySelectorAll('.eval-dot')
    expect(dots).toHaveLength(1)
    expect(dots[0]).toHaveClass('eval-dot-blunder')
    expect((dots[0] as HTMLElement).style.left).toBe('100%')
  })

  it('reads a move off the graph by keyboard', () => {
    render(<EvalGraph review={SWING} />)
    const plot = screen.getByRole('img')
    fireEvent.keyDown(plot, { key: 'ArrowRight' })
    expect(screen.getByText('Start · +0.2')).toBeInTheDocument()
    fireEvent.keyDown(plot, { key: 'ArrowLeft' })
    fireEvent.keyDown(plot, { key: 'ArrowRight' })
    fireEvent.keyDown(plot, { key: 'ArrowRight' })
    expect(screen.getByText('1… e5 · +3.0')).toBeInTheDocument()
    fireEvent.keyDown(plot, { key: 'ArrowRight' })
    expect(screen.getByText('2. Qh5?? · −1.0 · best was Nf3')).toBeInTheDocument()
  })

  it('holds a mate at the ceiling and labels it', () => {
    const mated: GameReview = {
      ...SWING,
      moves: [
        move({ player_eval_cp: 1000, mate: { in: 2, for: 'player' } }),
        move({ san: 'Qxf7#', player_eval_cp: 1000, mate: { in: 0, for: 'player' } }),
      ],
    }
    render(<EvalGraph review={mated} />)
    expect(screen.getByText('+10')).toBeInTheDocument()
    expect(points().map(([, y]) => y)).toEqual([9.8, 0, 0])
    expect(screen.getByRole('img')).toHaveAccessibleName(/ends checkmate/)
  })

  it('spells a mate against the player, and a level position', () => {
    const lost: GameReview = {
      ...SWING,
      start: { player_eval_cp: 0, mate: null },
      moves: [move({ player_eval_cp: -1000, mate: { in: 3, for: 'glitch' } })],
    }
    render(<EvalGraph review={lost} />)
    expect(screen.getByRole('img')).toHaveAccessibleName(/starts 0\.0, .* ends −M3/)
  })
})
