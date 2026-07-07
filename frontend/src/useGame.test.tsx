import { act, renderHook, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { useGame } from './useGame'
import type { GameState, MoveResponse } from './api'

// Playback is a side effect owned by tts.ts (tested there); here we only
// assert the hook asks for it when — and only when — the backend says so.
vi.mock('./tts', () => ({ playText: vi.fn() }))

const START_FEN = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'
const AFTER_E4_FEN = 'rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1'
// A white pawn on e7, ready to promote.
const PROMO_FEN = '4k3/4P3/8/8/8/8/8/4K3 w - - 0 1'
const AFTER_PROMO_FEN = '4Q1k1/8/8/8/8/8/8/4K3 b - - 0 1'

function state(overrides: Partial<GameState> = {}): GameState {
  return {
    fen: START_FEN,
    turn: 'white',
    game_over: false,
    outcome: null,
    history: [],
    captured: { white: [], black: [] },
    legal_moves: ['e4'],
    dests: { e2: ['e3', 'e4'] },
    ...overrides,
  }
}

// jsdom has no WebSocket; this stand-in lets a test push a server frame.
class FakeWebSocket {
  static instances: FakeWebSocket[] = []
  onmessage: ((ev: { data: string }) => void) | null = null
  close = vi.fn()
  url: string
  constructor(url: string) {
    this.url = url
    FakeWebSocket.instances.push(this)
  }
  emit(data: unknown) {
    this.onmessage?.({ data: JSON.stringify(data) })
  }
}

let moveResponse: MoveResponse
let fetchMock: ReturnType<typeof vi.fn>

function jsonResponse(body: unknown) {
  return Promise.resolve({ ok: true, json: () => Promise.resolve(body) })
}

beforeEach(() => {
  FakeWebSocket.instances = []
  vi.stubGlobal('WebSocket', FakeWebSocket)
  moveResponse = {
    legal: true,
    san: 'e4',
    uci: 'e2e4',
    reason: null,
    engine_move: null,
    state: state({ fen: AFTER_E4_FEN, turn: 'black' }),
  }
  fetchMock = vi.fn((url: string) => {
    const path = String(url)
    if (path.includes('/api/game/move')) return jsonResponse(moveResponse)
    if (path.includes('/api/game/difficulty'))
      return jsonResponse({ skill_level: 15, elo: null })
    if (path.includes('/api/command'))
      return jsonResponse({ commentary: 'Nice move!', tool_results: [], state: state() })
    if (path.includes('/api/settings/voice')) return jsonResponse({ voice_output: true })
    if (path.includes('/api/settings'))
      return jsonResponse({
        personality: 'friendly_rival',
        verbosity: 'normal',
        hints_mode: false,
        voice_output: false,
        skill_level: 5,
        elo: null,
      })
    // Lifecycle mutations (new / undo / resign) answer with { state }.
    if (path.includes('/api/game/')) return jsonResponse({ state: state() })
    return jsonResponse(state())
  })
  vi.stubGlobal('fetch', fetchMock)
})

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('useGame', () => {
  it('loads the initial state from the backend', async () => {
    const { result } = renderHook(() => useGame())
    await waitFor(() => expect(result.current.state?.fen).toBe(START_FEN))
    expect(result.current.state?.dests.e2).toEqual(['e3', 'e4'])
  })

  it('applies state pushed over the websocket', async () => {
    const { result } = renderHook(() => useGame())
    await waitFor(() => expect(result.current.state).not.toBeNull())
    act(() => {
      FakeWebSocket.instances[0].emit({
        type: 'state',
        state: state({ fen: AFTER_E4_FEN, turn: 'black' }),
      })
    })
    expect(result.current.state?.fen).toBe(AFTER_E4_FEN)
  })

  it('applies the new state and clears feedback on a legal move', async () => {
    const { result } = renderHook(() => useGame())
    await waitFor(() => expect(result.current.state).not.toBeNull())
    await act(async () => {
      await result.current.play('e2', 'e4')
    })
    expect(result.current.state?.fen).toBe(AFTER_E4_FEN)
    expect(result.current.moveError).toBeNull()
    expect(fetchMock).toHaveBeenCalledWith(
      '/api/game/move',
      expect.objectContaining({ method: 'POST', body: JSON.stringify({ move: 'e2e4' }) }),
    )
  })

  it('surfaces feedback and snaps back to the authoritative state on an illegal move', async () => {
    const { result } = renderHook(() => useGame())
    await waitFor(() => expect(result.current.state).not.toBeNull())
    const revisionBefore = result.current.revision
    // Backend rejects; its returned state is authoritative (position unchanged).
    moveResponse = {
      legal: false,
      san: null,
      uci: null,
      reason: 'illegal move: e2e5',
      engine_move: null,
      state: state(),
    }
    await act(async () => {
      await result.current.play('e2', 'e5')
    })
    expect(result.current.moveError).toBe('illegal move: e2e5')
    expect(result.current.state?.fen).toBe(START_FEN)
    // Revision must advance even though the fen is unchanged, so the board
    // re-syncs and the illegally-moved piece snaps back.
    expect(result.current.revision).toBeGreaterThan(revisionBefore)
  })

  it('defers a promotion move to a piece choice instead of submitting bare', async () => {
    const { result } = renderHook(() => useGame())
    // Start from a position where e7->e8 is a promotion.
    await waitFor(() => expect(result.current.state).not.toBeNull())
    act(() => {
      FakeWebSocket.instances[0].emit({ type: 'state', state: state({ fen: PROMO_FEN }) })
    })
    await act(async () => {
      await result.current.play('e7', 'e8')
    })
    // No move submitted yet — the picker gates it.
    expect(result.current.pendingPromotion).toEqual({ from: 'e7', to: 'e8' })
    expect(fetchMock).not.toHaveBeenCalledWith('/api/game/move', expect.anything())
  })

  it('submits the promotion with the chosen piece and clears the pending choice', async () => {
    const { result } = renderHook(() => useGame())
    await waitFor(() => expect(result.current.state).not.toBeNull())
    act(() => {
      FakeWebSocket.instances[0].emit({ type: 'state', state: state({ fen: PROMO_FEN }) })
    })
    await act(async () => {
      await result.current.play('e7', 'e8')
    })
    moveResponse = {
      legal: true,
      san: 'e8=Q',
      uci: 'e7e8q',
      reason: null,
      engine_move: null,
      state: state({ fen: AFTER_PROMO_FEN, turn: 'black' }),
    }
    await act(async () => {
      await result.current.completePromotion('q')
    })
    expect(fetchMock).toHaveBeenCalledWith(
      '/api/game/move',
      expect.objectContaining({ body: JSON.stringify({ move: 'e7e8q' }) }),
    )
    expect(result.current.pendingPromotion).toBeNull()
    expect(result.current.state?.fen).toBe(AFTER_PROMO_FEN)
  })

  it('starts a new game and clears any move feedback', async () => {
    const { result } = renderHook(() => useGame())
    await waitFor(() => expect(result.current.state).not.toBeNull())
    // Return a fresh start position from the new-game endpoint.
    fetchMock.mockImplementation((url: string) => {
      if (String(url).includes('/api/game/new'))
        return jsonResponse({ state: state({ history: [] }) })
      return jsonResponse(state())
    })
    await act(async () => {
      await result.current.newGame()
    })
    expect(fetchMock).toHaveBeenCalledWith(
      '/api/game/new',
      expect.objectContaining({ method: 'POST' }),
    )
    expect(result.current.moveError).toBeNull()
    expect(result.current.state?.history).toEqual([])
  })

  it('undoes and applies the returned state', async () => {
    const { result } = renderHook(() => useGame())
    await waitFor(() => expect(result.current.state).not.toBeNull())
    await act(async () => {
      await result.current.undo()
    })
    expect(fetchMock).toHaveBeenCalledWith(
      '/api/game/undo',
      expect.objectContaining({ method: 'POST', body: JSON.stringify({ plies: 1 }) }),
    )
  })

  it('leaves state untouched when the backend refuses a lifecycle action', async () => {
    const { result } = renderHook(() => useGame())
    await waitFor(() => expect(result.current.state).not.toBeNull())
    const revisionBefore = result.current.revision
    // 409: nothing to undo — no state comes back, so nothing should apply.
    fetchMock.mockImplementation(() => Promise.resolve({ ok: false, json: () => Promise.resolve({}) }))
    await act(async () => {
      await result.current.undo()
    })
    expect(result.current.revision).toBe(revisionBefore)
  })

  it('resigns and applies the returned state', async () => {
    const { result } = renderHook(() => useGame())
    await waitFor(() => expect(result.current.state).not.toBeNull())
    await act(async () => {
      await result.current.resign()
    })
    expect(fetchMock).toHaveBeenCalledWith(
      '/api/game/resign',
      expect.objectContaining({ method: 'POST' }),
    )
  })

  it('loads the current difficulty from settings', async () => {
    const { result } = renderHook(() => useGame())
    await waitFor(() => expect(result.current.skillLevel).toBe(5))
  })

  it('sets difficulty by skill level without touching board state', async () => {
    const { result } = renderHook(() => useGame())
    await waitFor(() => expect(result.current.state).not.toBeNull())
    const revisionBefore = result.current.revision
    await act(async () => {
      await result.current.setDifficulty(15)
    })
    expect(fetchMock).toHaveBeenCalledWith(
      '/api/game/difficulty',
      expect.objectContaining({ method: 'POST', body: JSON.stringify({ skill_level: 15 }) }),
    )
    // The hook reflects only what the server confirmed.
    expect(result.current.skillLevel).toBe(15)
    // Difficulty is not a board mutation — no re-render churn.
    expect(result.current.revision).toBe(revisionBefore)
  })

  it('sends a command, applies the returned state, and surfaces the commentary', async () => {
    const { result } = renderHook(() => useGame())
    await waitFor(() => expect(result.current.state).not.toBeNull())
    fetchMock.mockImplementation((url: string) => {
      if (String(url).includes('/api/command'))
        return jsonResponse({
          commentary: 'A classic king-pawn opening.',
          tool_results: [{ name: 'make_move', result: {} }],
          state: state({ fen: AFTER_E4_FEN, turn: 'black' }),
        })
      return jsonResponse(state())
    })
    await act(async () => {
      await result.current.sendCommand('play e4')
    })
    expect(fetchMock).toHaveBeenCalledWith(
      '/api/command',
      expect.objectContaining({ method: 'POST', body: JSON.stringify({ text: 'play e4' }) }),
    )
    expect(result.current.commentary).toBe('A classic king-pawn opening.')
    expect(result.current.state?.fen).toBe(AFTER_E4_FEN)
    expect(result.current.agentThinking).toBe(false)
  })

  it('loads the voice-output setting on mount', async () => {
    const { result } = renderHook(() => useGame())
    await waitFor(() => expect(result.current.voiceOutput).toBe(false))
    expect(fetchMock).toHaveBeenCalledWith('/api/settings')
  })

  it('toggles voice output through the settings endpoint', async () => {
    const { result } = renderHook(() => useGame())
    await waitFor(() => expect(result.current.voiceOutput).toBe(false))
    await act(async () => {
      await result.current.setVoiceOutput(true)
    })
    expect(fetchMock).toHaveBeenCalledWith(
      '/api/settings/voice',
      expect.objectContaining({ method: 'POST', body: JSON.stringify({ enabled: true }) }),
    )
    // The hook reflects what the server confirmed, not what was requested.
    expect(result.current.voiceOutput).toBe(true)
  })

  it('syncs voice output from the command speak flag (agent-side toggle)', async () => {
    const { result } = renderHook(() => useGame())
    await waitFor(() => expect(result.current.voiceOutput).toBe(false))
    // The agent flipped voice on via set_voice_output: speak comes back true.
    fetchMock.mockImplementation((url: string) => {
      if (String(url).includes('/api/command'))
        return jsonResponse({ commentary: 'Voice on!', tool_results: [], state: state(), speak: true })
      return jsonResponse(state())
    })
    await act(async () => {
      await result.current.sendCommand('turn on voice')
    })
    expect(result.current.voiceOutput).toBe(true)
  })

  it('voices the commentary when the backend says speak', async () => {
    const { playText } = await import('./tts')
    vi.mocked(playText).mockClear()
    const { result } = renderHook(() => useGame())
    await waitFor(() => expect(result.current.state).not.toBeNull())
    fetchMock.mockImplementation((url: string) => {
      if (String(url).includes('/api/command'))
        return jsonResponse({
          commentary: 'Check!',
          tool_results: [],
          state: state(),
          speak: true,
        })
      return jsonResponse(state())
    })
    await act(async () => {
      await result.current.sendCommand('any threats?')
    })
    expect(playText).toHaveBeenCalledWith('Check!')
  })

  it('stays silent when the backend does not ask to speak', async () => {
    const { playText } = await import('./tts')
    vi.mocked(playText).mockClear()
    const { result } = renderHook(() => useGame())
    await waitFor(() => expect(result.current.state).not.toBeNull())
    fetchMock.mockImplementation((url: string) => {
      if (String(url).includes('/api/command'))
        return jsonResponse({ commentary: 'Check!', tool_results: [], state: state(), speak: false })
      return jsonResponse(state())
    })
    await act(async () => {
      await result.current.sendCommand('any threats?')
    })
    expect(playText).not.toHaveBeenCalled()
  })

  it('reports an unavailable agent without touching board state', async () => {
    const { result } = renderHook(() => useGame())
    await waitFor(() => expect(result.current.state).not.toBeNull())
    const revisionBefore = result.current.revision
    // 503: no brain configured — no commentary or state comes back.
    fetchMock.mockImplementation((url: string) => {
      if (String(url).includes('/api/command'))
        return Promise.resolve({ ok: false, json: () => Promise.resolve({}) })
      return jsonResponse(state())
    })
    await act(async () => {
      await result.current.sendCommand('play e4')
    })
    expect(result.current.commentary).toMatch(/unavailable/i)
    expect(result.current.state?.fen).toBe(START_FEN)
    expect(result.current.revision).toBe(revisionBefore)
    expect(result.current.agentThinking).toBe(false)
  })

  it('cancels a promotion, snapping the pawn back without submitting', async () => {
    const { result } = renderHook(() => useGame())
    await waitFor(() => expect(result.current.state).not.toBeNull())
    act(() => {
      FakeWebSocket.instances[0].emit({ type: 'state', state: state({ fen: PROMO_FEN }) })
    })
    await act(async () => {
      await result.current.play('e7', 'e8')
    })
    const revisionBefore = result.current.revision
    act(() => {
      result.current.cancelPromotion()
    })
    expect(result.current.pendingPromotion).toBeNull()
    expect(fetchMock).not.toHaveBeenCalledWith('/api/game/move', expect.anything())
    // Revision advances so the board re-syncs and the pawn snaps back.
    expect(result.current.revision).toBeGreaterThan(revisionBefore)
  })
})
