import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import { ReviewPanel } from './ReviewPanel'
import type { GameReview } from './api'

const REVIEW: GameReview = {
  moves: [
    {
      san: 'e4',
      uci: 'e2e4',
      color: 'white',
      cp_loss: 5,
      classification: 'good',
      best: 'e4',
      accuracy: 100,
      move_number: 1,
      player_eval_cp: 30,
      mate: null,
    },
    {
      san: 'g5',
      uci: 'g7g5',
      color: 'black',
      cp_loss: 350,
      classification: 'blunder',
      best: 'e5',
      accuracy: 42.3,
      move_number: 1,
      player_eval_cp: 380,
      mate: null,
    },
  ],
  accuracy: { white: 97.1, black: 42.3 },
  counts: {
    white: { good: 1, inaccuracy: 0, mistake: 0, blunder: 0 },
    black: { good: 0, inaccuracy: 0, mistake: 0, blunder: 1 },
  },
  player_color: 'white',
  start: { player_eval_cp: 20, mate: null },
}

function panel(overrides: Partial<Parameters<typeof ReviewPanel>[0]> = {}) {
  return (
    <ReviewPanel
      review={REVIEW}
      loading={false}
      unavailable={false}
      playerColor="white"
      onRetry={vi.fn()}
      onDone={vi.fn()}
      {...overrides}
    />
  )
}

describe('ReviewPanel', () => {
  it('is its own sheet, closed by Done', () => {
    const onDone = vi.fn()
    render(panel({ onDone }))
    expect(screen.getByRole('dialog', { name: /game review/i })).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: /done/i }))
    expect(onDone).toHaveBeenCalledTimes(1)
  })

  it('says it is working while the review runs', () => {
    render(panel({ review: null, loading: true }))
    expect(screen.getByText(/reviewing…/i)).toBeInTheDocument()
  })

  it('shows per-side accuracy and counts, the player first as "You"', () => {
    render(panel({ playerColor: 'black' }))
    const cards = document.querySelectorAll('.review-color')
    expect(cards[0]).toHaveTextContent(/you · black/i)
    expect(cards[1]).toHaveTextContent(/glitch · white/i)
    expect(screen.getByText('97.1%')).toBeInTheDocument()
    expect(screen.getByText('42.3%')).toBeInTheDocument()
    const black = screen.getByLabelText(/black review summary/i)
    expect(black).toHaveTextContent(/1\s*blunder/i)
  })

  it('pluralises each classification count', () => {
    render(
      panel({
        review: {
          ...REVIEW,
          counts: {
            white: { good: 1, inaccuracy: 8, mistake: 2, blunder: 3 },
            black: { good: 0, inaccuracy: 1, mistake: 0, blunder: 1 },
          },
        },
      }),
    )
    expect(screen.getByLabelText(/white review summary/i)).toHaveTextContent(
      '8 inaccuracies, 2 mistakes, 3 blunders',
    )
    expect(screen.getByLabelText(/black review summary/i)).toHaveTextContent(
      '1 inaccuracy, 1 blunder',
    )
  })

  it('pairs the moves into numbered rows and notes the better move under a flagged one', () => {
    render(panel())
    const row = screen.getByText('g5').closest('li')!
    expect(row).toHaveTextContent('1.')
    expect(row).toHaveTextContent('e4')
    expect(row).toHaveClass('review-row-flagged')
    expect(screen.getByLabelText('blunder')).toHaveTextContent('??')
    expect(row).toHaveTextContent('Glitch’s 1… g5 — best was e5')
  })

  it('leaves good moves unremarked', () => {
    render(panel({ review: { ...REVIEW, moves: [REVIEW.moves[0]] } }))
    const row = screen.getByText('e4').closest('li')!
    expect(row).not.toHaveClass('review-row-flagged')
    expect(row).not.toHaveTextContent(/best was/)
  })

  it('credits a flagged move to the player as "Your"', () => {
    render(panel({ playerColor: 'black' }))
    expect(screen.getByText('Your 1… g5 — best was e5')).toBeInTheDocument()
  })

  it('starts a Black-first game with an empty White cell', () => {
    render(panel({ review: { ...REVIEW, moves: [REVIEW.moves[1]] } }))
    const row = screen.getByText('g5').closest('li')!
    expect(row.querySelectorAll('.review-cell')[0]).toBeEmptyDOMElement()
  })

  it('reports when review is unavailable and offers a retry', () => {
    const onRetry = vi.fn()
    render(panel({ review: null, unavailable: true, onRetry }))
    expect(screen.getByText(/review unavailable/i)).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: /retry/i }))
    expect(onRetry).toHaveBeenCalledTimes(1)
  })
})
