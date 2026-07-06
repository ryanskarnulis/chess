import { fireEvent, render, screen } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { ReviewPanel } from './ReviewPanel'
import type { GameReview } from './api'

// The panel talks to the backend only through the typed client; mocking it
// keeps these tests off the network, same as the other component tests.
vi.mock('./api', () => ({ fetchReview: vi.fn() }))
import { fetchReview } from './api'

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
    },
    {
      san: 'g5',
      uci: 'g7g5',
      color: 'black',
      cp_loss: 350,
      classification: 'blunder',
      best: 'e5',
      accuracy: 42.3,
    },
  ],
  accuracy: { white: 97.1, black: 42.3 },
  counts: {
    white: { good: 1, inaccuracy: 0, mistake: 0, blunder: 0 },
    black: { good: 0, inaccuracy: 0, mistake: 0, blunder: 1 },
  },
}

beforeEach(() => {
  vi.mocked(fetchReview).mockReset()
})

describe('ReviewPanel', () => {
  it('shows only the review button until asked', () => {
    render(<ReviewPanel />)
    expect(screen.getByRole('button', { name: /review game/i })).toBeInTheDocument()
    expect(fetchReview).not.toHaveBeenCalled()
    expect(screen.queryByText(/accuracy/i)).not.toBeInTheDocument()
  })

  it('fetches on demand and shows per-color accuracy and counts', async () => {
    vi.mocked(fetchReview).mockResolvedValue(REVIEW)
    render(<ReviewPanel />)
    fireEvent.click(screen.getByRole('button', { name: /review game/i }))
    expect(fetchReview).toHaveBeenCalledTimes(1)
    expect(await screen.findByText('97.1%')).toBeInTheDocument()
    expect(screen.getByText('42.3%')).toBeInTheDocument()
    // Black's blunder shows up in the classification counts.
    const black = screen.getByLabelText(/black review summary/i)
    expect(black).toHaveTextContent(/1\s*blunder/i)
  })

  it('lists the moves and flags a bad move with the better alternative', async () => {
    vi.mocked(fetchReview).mockResolvedValue(REVIEW)
    render(<ReviewPanel />)
    fireEvent.click(screen.getByRole('button', { name: /review game/i }))
    const row = (await screen.findByText('g5')).closest('tr')!
    expect(row).toHaveTextContent(/blunder/i)
    expect(row).toHaveTextContent('e5')
    // A good move is not second-guessed with an alternative.
    const goodRow = screen.getByText('e4').closest('tr')!
    expect(goodRow).not.toHaveTextContent(/blunder|inaccuracy|mistake/i)
  })

  it('reports when review is unavailable instead of crashing', async () => {
    // Null is the client's word for a refused request (no engine / no moves).
    vi.mocked(fetchReview).mockResolvedValue(null)
    render(<ReviewPanel />)
    fireEvent.click(screen.getByRole('button', { name: /review game/i }))
    expect(await screen.findByText(/review unavailable/i)).toBeInTheDocument()
    // The button stays so the user can retry (e.g. engine came back).
    expect(screen.getByRole('button', { name: /review game/i })).toBeInTheDocument()
  })
})
