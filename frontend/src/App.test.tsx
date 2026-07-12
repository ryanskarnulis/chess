import { render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import App from './App'
import type { GameState } from './api'

vi.mock('./tts', () => ({ playText: vi.fn() }))

const START_FEN = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'

function state(overrides: Partial<GameState> = {}): GameState {
  return {
    fen: START_FEN,
    turn: 'white',
    player_color: 'white',
    game_over: false,
    outcome: null,
    history: [],
    fens: [START_FEN],
    captured: { white: [], black: [] },
    legal_moves: ['e4'],
    dests: { e2: ['e3', 'e4'] },
    ...overrides,
  }
}

class FakeWebSocket {
  onmessage: ((ev: { data: string }) => void) | null = null
  close = vi.fn()
  url: string
  constructor(url: string) {
    this.url = url
  }
}

function stubMatchMedia(matches: boolean) {
  vi.stubGlobal(
    'matchMedia',
    vi.fn(() => ({
      matches,
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
    })),
  )
}

// What /api/state serves; tests reassign to render App against a position.
let served: GameState
let fetchMock: ReturnType<typeof vi.fn>

beforeEach(() => {
  served = state()
  vi.stubGlobal('WebSocket', FakeWebSocket)
  fetchMock = vi.fn((url: string) => {
    const path = String(url)
    const body = path.includes('/api/settings')
      ? {
          verbosity: 'normal',
          hints_mode: false,
          voice_output: false,
          tier: 'casual',
          skill_level: null,
          elo: null,
        }
      : path.includes('/api/game/')
        ? { state: served }
        : served
    return Promise.resolve({ ok: true, json: () => Promise.resolve(body) })
  })
  vi.stubGlobal('fetch', fetchMock)
})

afterEach(() => vi.unstubAllGlobals())

describe('App layout', () => {
  // One layout at every viewport: the stacked layout (agent bubble, board,
  // move strip, bottom bar) is the app — there is no separate desktop tree.
  it.each([
    ['narrow viewports', true],
    ['wide viewports', false],
  ])('shows the single stacked layout on %s', async (_label, matches) => {
    stubMatchMedia(matches)
    render(<App />)
    await waitFor(() =>
      expect(screen.getByRole('navigation', { name: /game controls/i })).toBeInTheDocument(),
    )
    expect(screen.getByRole('button', { name: /hint/i })).toBeInTheDocument()
    // The agent bubble stages the commentary next to the spider mascot.
    expect(document.querySelector('.spider-icon')).toBeInTheDocument()
    expect(document.querySelector('.move-strip')).toBeInTheDocument()
    // No desktop-only side panels anywhere.
    expect(screen.queryByRole('complementary')).not.toBeInTheDocument()
  })

  it('opens the options sheet from the bottom bar', async () => {
    stubMatchMedia(false)
    render(<App />)
    await waitFor(() =>
      expect(screen.getByRole('navigation', { name: /game controls/i })).toBeInTheDocument(),
    )
    screen.getByRole('button', { name: /options/i }).click()
    await waitFor(() =>
      expect(screen.getByRole('dialog', { name: /options/i })).toBeInTheDocument(),
    )
    expect(screen.getByRole('button', { name: /new game/i })).toBeInTheDocument()
  })
})

describe('player color', () => {
  it('orients the board for the player side', async () => {
    served = state({ player_color: 'black', turn: 'black', history: ['e4'] })
    const { container } = render(<App />)
    await waitFor(() =>
      expect(container.querySelector('.cg-wrap.orientation-black')).toBeInTheDocument(),
    )
  })

  it('offers a side switch until the player has moved', async () => {
    render(<App />)
    await waitFor(() =>
      expect(screen.getByRole('button', { name: /switch to black/i })).toBeInTheDocument(),
    )
    screen.getByRole('button', { name: /switch to black/i }).click()
    await waitFor(() =>
      expect(fetchMock).toHaveBeenCalledWith(
        '/api/game/new',
        expect.objectContaining({ method: 'POST', body: JSON.stringify({ color: 'black' }) }),
      ),
    )
  })

  it('keeps the switch while only the engine has moved', async () => {
    served = state({ player_color: 'black', turn: 'black', history: ['e4'] })
    render(<App />)
    await waitFor(() =>
      expect(screen.getByRole('button', { name: /switch to white/i })).toBeInTheDocument(),
    )
  })

  it('withdraws the switch once the player has moved', async () => {
    served = state({ history: ['e4', 'e5'] })
    render(<App />)
    await waitFor(() => expect(document.querySelector('.move-strip')).toBeInTheDocument())
    expect(screen.queryByRole('button', { name: /switch to/i })).not.toBeInTheDocument()
  })

  it('disables undo while only the engine has moved', async () => {
    served = state({ player_color: 'black', turn: 'black', history: ['e4'] })
    render(<App />)
    await waitFor(() =>
      expect(screen.getByRole('button', { name: /undo/i })).toBeDisabled(),
    )
  })
})
