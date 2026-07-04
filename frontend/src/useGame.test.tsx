import { act, renderHook, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { useGame } from './useGame'
import type { GameState, MoveResponse } from './api'

const START_FEN = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'
const AFTER_E4_FEN = 'rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1'

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
    if (String(url).includes('/api/game/move')) return jsonResponse(moveResponse)
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
})
