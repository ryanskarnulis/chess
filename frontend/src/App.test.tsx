import { StrictMode } from 'react'
import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import App from './App'
import type { GameState } from './api'
import type { BoardProps } from './Board'

vi.mock('./tts', () => ({ playText: vi.fn() }))

// Chessground refuses synthetic events (`isTrusted`), so what the board is
// *offered* cannot be read back out of its DOM. The real board still renders —
// this only records the props on the way through.
const boardProps = vi.hoisted(() => [] as BoardProps[])
vi.mock('./Board', async (importOriginal) => {
  const actual = await importOriginal<typeof import('./Board')>()
  return {
    Board: (props: BoardProps) => {
      boardProps.push(props)
      return <actual.Board {...props} />
    },
  }
})

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
  boardProps.length = 0
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

afterEach(() => {
  vi.unstubAllGlobals()
  window.history.replaceState({}, '', '/')
})

// The conductor hands a user off to chess by navigating here with the thing
// they actually said (`/?intent=…`); we run it through the agent so Glitch
// opens the session already acting on it.
describe('conductor handoff (?intent=)', () => {
  const commandCalls = () =>
    fetchMock.mock.calls.filter(([url]) => String(url).includes('/api/command'))

  it('runs the intent through the agent and scrubs it from the URL', async () => {
    window.history.replaceState({}, '', '/?intent=play%20e4')
    render(<App />)
    await waitFor(() => expect(commandCalls()).toHaveLength(1))
    expect(commandCalls()[0][1]).toMatchObject({
      method: 'POST',
      body: JSON.stringify({ text: 'play e4' }),
    })
    // Scrubbed, so a reload doesn't replay the command.
    expect(window.location.search).toBe('')
  })

  it('fires exactly once under StrictMode double-mount', async () => {
    window.history.replaceState({}, '', '/?intent=play%20e4')
    render(
      <StrictMode>
        <App />
      </StrictMode>,
    )
    await waitFor(() => expect(commandCalls()).toHaveLength(1))
    // Give a second effect pass room to double-fire before asserting it didn't.
    await new Promise((resolve) => setTimeout(resolve, 20))
    expect(commandCalls()).toHaveLength(1)
  })

  it('sends nothing when there is no intent', async () => {
    render(<App />)
    await waitFor(() => expect(document.querySelector('.move-strip')).toBeInTheDocument())
    expect(commandCalls()).toHaveLength(0)
  })

  it('ignores a blank intent', async () => {
    window.history.replaceState({}, '', '/?intent=%20%20')
    render(<App />)
    await waitFor(() => expect(document.querySelector('.move-strip')).toBeInTheDocument())
    expect(commandCalls()).toHaveLength(0)
    expect(window.location.search).toBe('')
  })
})

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

describe('a board mid-turn', () => {
  it('offers nothing to drag while it is the engine to move', async () => {
    // The state document now arrives mid-turn — the player's move is on the
    // board before the engine has answered — and such a frame carries the
    // engine's turn and the engine's own legal moves. Handing those to the
    // board would let the player drag the opponent's pieces.
    served = state({ turn: 'black', history: ['e4'], dests: { e7: ['e6', 'e5'] } })
    render(<App />)
    await waitFor(() => expect(boardProps.length).toBeGreaterThan(0))
    await waitFor(() => expect(boardProps.at(-1)!.turnColor).toBe('black'))
    expect(boardProps.at(-1)!.dests).toEqual({})
  })

  it('offers the player their own moves once the turn is theirs', async () => {
    render(<App />)
    await waitFor(() => expect(boardProps.length).toBeGreaterThan(0))
    expect(boardProps.at(-1)!.dests).toEqual({ e2: ['e3', 'e4'] })
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

// Direct mode: no brain configured. The game is fully playable against
// Stockfish, and that is a deliberate mode — so it is shown, and the command
// box is a designed dead state rather than a 503 the player has to trip over.
describe('direct mode', () => {
  function serveSettings(extra: Record<string, unknown>) {
    fetchMock.mockImplementation((url: string) => {
      const path = String(url)
      const body = path.includes('/api/settings')
        ? {
            verbosity: 'normal',
            hints_mode: false,
            voice_output: false,
            tier: 'casual',
            skill_level: null,
            elo: null,
            ...extra,
          }
        : path.includes('/api/game/')
          ? { state: served }
          : served
      return Promise.resolve({ ok: true, json: () => Promise.resolve(body) })
    })
  }

  it('shows a standing indicator and locks the command box', async () => {
    serveSettings({ agent_available: false })
    render(<App />)
    await waitFor(() => expect(screen.getByText(/direct mode/i)).toBeInTheDocument())
    expect(screen.getByRole('textbox', { name: /command/i })).toBeDisabled()
    expect(screen.getByRole('button', { name: /send/i })).toBeDisabled()
  })

  it('says nothing and leaves the box open when an agent is available', async () => {
    serveSettings({ agent_available: true })
    render(<App />)
    await waitFor(() => expect(document.querySelector('.move-strip')).toBeInTheDocument())
    expect(screen.queryByText(/direct mode/i)).not.toBeInTheDocument()
    expect(screen.getByRole('textbox', { name: /command/i })).not.toBeDisabled()
  })
})

describe('post-game screen', () => {
  const overState = () =>
    state({
      game_over: true,
      outcome: { termination: 'checkmate', winner: 'white', result: '1-0' },
      history: ['e4', 'f6', 'd4', 'g5', 'Qh5#'],
    })

  it('pops up when the game is over', async () => {
    served = overState()
    render(<App />)
    await waitFor(() =>
      expect(screen.getByRole('dialog', { name: /game over/i })).toBeInTheDocument(),
    )
    expect(screen.getByText(/you won/i)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /review game/i })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /new game/i })).toBeInTheDocument()
  })

  it('dismisses to the final board and reopens from the results chip', async () => {
    served = overState()
    render(<App />)
    await waitFor(() =>
      expect(screen.getByRole('dialog', { name: /game over/i })).toBeInTheDocument(),
    )
    fireEvent.click(screen.getByRole('button', { name: /^close$/i }))
    // Hidden, not unmounted (a fetched review must survive reopening) —
    // but gone from the accessibility tree either way.
    expect(screen.queryByRole('dialog', { name: /game over/i })).not.toBeInTheDocument()
    // Review lives in the modal now — nothing visible under the board.
    expect(document.querySelector('.review-panel')).not.toBeVisible()
    fireEvent.click(screen.getByRole('button', { name: /results/i }))
    expect(screen.getByRole('dialog', { name: /game over/i })).toBeInTheDocument()
  })
})
