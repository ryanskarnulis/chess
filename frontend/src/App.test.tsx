import { render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import App from './App'
import type { GameState } from './api'

vi.mock('./tts', () => ({ playText: vi.fn() }))

const START_FEN = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'

function state(): GameState {
  return {
    fen: START_FEN,
    turn: 'white',
    game_over: false,
    outcome: null,
    history: [],
    fens: [START_FEN],
    captured: { white: [], black: [] },
    legal_moves: ['e4'],
    dests: { e2: ['e3', 'e4'] },
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

beforeEach(() => {
  vi.stubGlobal('WebSocket', FakeWebSocket)
  vi.stubGlobal(
    'fetch',
    vi.fn((url: string) => {
      const path = String(url)
      const body = path.includes('/api/settings')
        ? {
            personality: 'friendly_rival',
            verbosity: 'normal',
            hints_mode: false,
            voice_output: false,
            tier: 'casual',
            skill_level: null,
            elo: null,
          }
        : state()
      return Promise.resolve({ ok: true, json: () => Promise.resolve(body) })
    }),
  )
})

afterEach(() => vi.unstubAllGlobals())

describe('App layout switch', () => {
  it('shows the stacked mobile layout on narrow viewports', async () => {
    stubMatchMedia(true)
    render(<App />)
    await waitFor(() =>
      expect(screen.getByRole('navigation', { name: /game controls/i })).toBeInTheDocument(),
    )
    // Bottom bar replaces the desktop side panels.
    expect(screen.getByRole('button', { name: /hint/i })).toBeInTheDocument()
    expect(screen.queryByRole('complementary')).not.toBeInTheDocument()
    // The agent bubble stages the commentary next to the spider mascot.
    expect(document.querySelector('.spider-icon')).toBeInTheDocument()
    expect(document.querySelector('.move-strip')).toBeInTheDocument()
  })

  it('keeps the desktop layout on wide viewports', async () => {
    stubMatchMedia(false)
    render(<App />)
    await waitFor(() => expect(screen.getByRole('complementary')).toBeInTheDocument())
    // No mobile chrome on desktop.
    expect(screen.queryByRole('navigation', { name: /game controls/i })).not.toBeInTheDocument()
    expect(document.querySelector('.spider-icon')).not.toBeInTheDocument()
    expect(screen.getByRole('button', { name: /new game/i })).toBeInTheDocument()
  })
})
