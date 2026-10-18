import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { PostGameModal } from './PostGameModal'
import type { GameReview, Outcome } from './api'

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
  ],
  accuracy: { white: 97.1, black: 50 },
  counts: { white: { good: 1 }, black: {} },
  player_color: 'white',
  start: { player_eval_cp: 20, mate: null },
}

function outcome(overrides: Partial<Outcome> = {}): Outcome {
  return { termination: 'checkmate', winner: 'white', result: '1-0', ...overrides }
}

const noop = () => {}

afterEach(() => vi.unstubAllGlobals())

describe('PostGameModal', () => {
  it('congratulates the player when they won', () => {
    render(
      <PostGameModal outcome={outcome()} playerColor="white" onNewGame={noop} onClose={noop} />,
    )
    expect(screen.getByRole('dialog', { name: /game over/i })).toBeInTheDocument()
    expect(screen.getByText(/you won/i)).toBeInTheDocument()
    expect(screen.getByText('checkmate · 1-0')).toBeInTheDocument()
  })

  it('commiserates when the player lost', () => {
    render(
      <PostGameModal
        outcome={outcome({ winner: 'black', result: '0-1' })}
        playerColor="white"
        onNewGame={noop}
        onClose={noop}
      />,
    )
    expect(screen.getByText(/you lost/i)).toBeInTheDocument()
  })

  it('calls a draw a draw, humanizing the termination', () => {
    render(
      <PostGameModal
        outcome={outcome({ termination: 'insufficient_material', winner: null, result: '1/2-1/2' })}
        playerColor="white"
        onNewGame={noop}
        onClose={noop}
      />,
    )
    expect(screen.getByText(/^draw$/i)).toBeInTheDocument()
    expect(screen.getByText('insufficient material · 1/2-1/2')).toBeInTheDocument()
  })

  it('opens the review in its own sheet, fetching it once', async () => {
    const fetchMock = vi.fn(() => Promise.resolve({ ok: true, json: () => Promise.resolve(REVIEW) }))
    vi.stubGlobal('fetch', fetchMock)
    render(
      <PostGameModal outcome={outcome()} playerColor="white" onNewGame={noop} onClose={noop} />,
    )
    expect(screen.queryByRole('dialog', { name: /game review/i })).not.toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: /review the game/i }))
    expect(await screen.findByText('97.1%')).toBeInTheDocument()
    // Done goes back to the game-over sheet; reopening keeps the review.
    fireEvent.click(screen.getByRole('button', { name: /done/i }))
    expect(screen.queryByRole('dialog', { name: /game review/i })).not.toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: /review the game/i }))
    expect(screen.getByText('97.1%')).toBeInTheDocument()
    expect(fetchMock).toHaveBeenCalledTimes(1)
  })

  it('starts a new game', () => {
    const onNewGame = vi.fn()
    render(
      <PostGameModal outcome={outcome()} playerColor="white" onNewGame={onNewGame} onClose={noop} />,
    )
    fireEvent.click(screen.getByRole('button', { name: /new game/i }))
    expect(onNewGame).toHaveBeenCalledTimes(1)
    // Called bare — never with the click event (it would leak into the
    // new-game color argument upstream).
    expect(onNewGame).toHaveBeenCalledWith()
  })

  it('closes from "Look at the board" and the backdrop', () => {
    const onClose = vi.fn()
    const { container } = render(
      <PostGameModal outcome={outcome()} playerColor="white" onNewGame={noop} onClose={onClose} />,
    )
    fireEvent.click(screen.getByRole('button', { name: /look at the board/i }))
    fireEvent.click(container.querySelector('.postgame-backdrop')!)
    expect(onClose).toHaveBeenCalledTimes(2)
  })

  it('copies the PGN to the clipboard', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(() => Promise.resolve({ ok: true, json: () => Promise.resolve({ pgn: '1. e4 e5' }) })),
    )
    const writeText = vi.fn(() => Promise.resolve())
    vi.stubGlobal('navigator', { ...window.navigator, clipboard: { writeText } })
    render(
      <PostGameModal outcome={outcome()} playerColor="white" onNewGame={noop} onClose={noop} />,
    )
    fireEvent.click(screen.getByRole('button', { name: /copy pgn/i }))
    await waitFor(() => expect(writeText).toHaveBeenCalledWith('1. e4 e5'))
    // The button confirms in place.
    expect(screen.getByRole('button', { name: /copied/i })).toBeInTheDocument()
  })
})
