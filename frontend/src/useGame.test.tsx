import { act, renderHook, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { useGame } from './useGame'
import type { GameState, MoveResponse } from './api'

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
    // Difficulty is not a board mutation — no re-render churn.
    expect(result.current.revision).toBe(revisionBefore)
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
