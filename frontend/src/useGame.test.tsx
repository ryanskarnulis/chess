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
    version: 1,
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

/** One live-progress frame, as the backend broadcasts it. */
function progressFrame(kind: string, name = '', correlation_id = 'abc123') {
  return { type: 'progress', progress: { correlation_id, turn_id: 1, kind, name } }
}

const AFTER_E4_E5_FEN = 'rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2'

/** State two plies in, with the per-ply fens the review arrows walk. */
function reviewableState(): GameState {
  return state({
    fen: AFTER_E4_E5_FEN,
    history: ['e4', 'e5'],
    fens: [START_FEN, AFTER_E4_FEN, AFTER_E4_E5_FEN],
  })
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
      return jsonResponse({ tier: 'advanced', skill_level: null, elo: null })
    if (path.includes('/api/command'))
      return jsonResponse({ commentary: 'Nice move!', tool_results: [], state: state() })
    if (path.includes('/api/game/hint'))
      return jsonResponse({ uci: 'e2e4', san: 'e4', from: 'e2', to: 'e4' })
    if (path.includes('/api/settings/voice')) return jsonResponse({ voice_output: true })
    if (path.includes('/api/settings'))
      return jsonResponse({
        verbosity: 'normal',
        hints_mode: false,
        voice_output: false,
        tier: 'casual',
        skill_level: null,
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

  it('submits rapid competing drags against the same rendered version', async () => {
    const { result } = renderHook(() => useGame())
    await waitFor(() => expect(result.current.state?.version).toBe(1))

    await act(async () => {
      await Promise.all([
        result.current.play('e2', 'e4'),
        result.current.play('d2', 'd4'),
      ])
    })

    const moveBodies = fetchMock.mock.calls
      .filter(([url]) => String(url).includes('/api/game/move'))
      .map(([, init]) => JSON.parse(String(init.body)))
    expect(moveBodies).toEqual([
      { move: 'e2e4', version: 1 },
      { move: 'd2d4', version: 1 },
    ])
  })

  it('does not let a delayed initial fetch overwrite a newer websocket state', async () => {
    let resolveState!: (value: Awaited<ReturnType<typeof jsonResponse>>) => void
    const delayedState = new Promise<Awaited<ReturnType<typeof jsonResponse>>>((resolve) => {
      resolveState = resolve
    })
    fetchMock.mockImplementation((url: string) => {
      if (String(url).includes('/api/settings'))
        return jsonResponse({
          verbosity: 'normal',
          hints_mode: false,
          voice_output: false,
          tier: 'casual',
          skill_level: null,
          elo: null,
        })
      if (String(url).includes('/api/state')) return delayedState
      return jsonResponse(state())
    })

    const { result } = renderHook(() => useGame())
    act(() => {
      FakeWebSocket.instances[0].emit({
        type: 'state',
        state: state({ version: 3, fen: AFTER_E4_FEN, turn: 'black' }),
      })
    })
    await waitFor(() => expect(result.current.state?.version).toBe(3))
    const revisionAfterSocket = result.current.revision

    await act(async () => {
      resolveState(await jsonResponse(state({ version: 1 })))
      await delayedState
    })

    expect(result.current.state?.version).toBe(3)
    expect(result.current.state?.fen).toBe(AFTER_E4_FEN)
    expect(result.current.revision).toBe(revisionAfterSocket)
  })

  it('does not let a delayed mutation response overwrite a newer websocket state', async () => {
    let resolveMove!: (value: Awaited<ReturnType<typeof jsonResponse>>) => void
    const delayedMove = new Promise<Awaited<ReturnType<typeof jsonResponse>>>((resolve) => {
      resolveMove = resolve
    })
    fetchMock.mockImplementation((url: string) => {
      if (String(url).includes('/api/game/move')) return delayedMove
      if (String(url).includes('/api/settings'))
        return jsonResponse({
          verbosity: 'normal',
          hints_mode: false,
          voice_output: false,
          tier: 'casual',
          skill_level: null,
          elo: null,
        })
      return jsonResponse(state())
    })
    const { result } = renderHook(() => useGame())
    await waitFor(() => expect(result.current.state?.version).toBe(1))

    let pending!: Promise<void>
    act(() => {
      pending = result.current.play('e2', 'e4')
    })
    act(() => {
      FakeWebSocket.instances[0].emit({
        type: 'state',
        state: state({ version: 3, fen: AFTER_E4_E5_FEN }),
      })
    })
    const revisionAfterSocket = result.current.revision
    await act(async () => {
      resolveMove(
        await jsonResponse({
          ...moveResponse,
          state: state({ version: 2, fen: AFTER_E4_FEN, turn: 'black' }),
        }),
      )
      await pending
    })

    expect(result.current.state?.version).toBe(3)
    expect(result.current.state?.fen).toBe(AFTER_E4_E5_FEN)
    expect(result.current.revision).toBe(revisionAfterSocket)
  })

  it('adopts the current state returned by a stale 409', async () => {
    fetchMock.mockImplementation((url: string) => {
      if (String(url).includes('/api/game/move'))
        return Promise.resolve({
          ok: false,
          status: 409,
          json: () =>
            Promise.resolve({
              detail: 'the board changed',
              stale: true,
              version: 4,
              state: state({ version: 4, fen: AFTER_E4_E5_FEN }),
            }),
        })
      if (String(url).includes('/api/settings'))
        return jsonResponse({
          verbosity: 'normal',
          hints_mode: false,
          voice_output: false,
          tier: 'casual',
          skill_level: null,
          elo: null,
        })
      return jsonResponse(state())
    })
    const { result } = renderHook(() => useGame())
    await waitFor(() => expect(result.current.state?.version).toBe(1))

    await act(async () => {
      await result.current.play('e2', 'e4')
    })

    expect(result.current.state?.version).toBe(4)
    expect(result.current.state?.fen).toBe(AFTER_E4_E5_FEN)
    expect(result.current.moveError).toBeNull()
  })

  it('adopts stale-409 state from commands and lifecycle helpers', async () => {
    fetchMock.mockImplementation((url: string) => {
      const path = String(url)
      if (path.includes('/api/command'))
        return Promise.resolve({
          ok: false,
          status: 409,
          json: () =>
            Promise.resolve({
              detail: 'the board changed',
              stale: true,
              version: 2,
              state: state({ version: 2, fen: AFTER_E4_FEN }),
            }),
        })
      if (path.includes('/api/game/undo') || path.includes('/api/game/new'))
        return Promise.resolve({
          ok: false,
          status: 409,
          json: () =>
            Promise.resolve({
              detail: 'the board changed again',
              stale: true,
              version: path.includes('/undo') ? 3 : 4,
              state: state({
                version: path.includes('/undo') ? 3 : 4,
                fen: AFTER_E4_E5_FEN,
              }),
            }),
        })
      if (path.includes('/api/settings'))
        return jsonResponse({
          verbosity: 'normal',
          hints_mode: false,
          voice_output: false,
          tier: 'casual',
          skill_level: null,
          elo: null,
        })
      return jsonResponse(state())
    })
    const { result } = renderHook(() => useGame())
    await waitFor(() => expect(result.current.state?.version).toBe(1))

    await act(async () => {
      await result.current.sendCommand('play e4')
    })
    expect(result.current.state?.version).toBe(2)
    expect(result.current.commentary).toBeNull()
    await act(async () => {
      await result.current.undo()
    })
    expect(result.current.state?.version).toBe(3)
    await act(async () => {
      await result.current.newGame()
    })
    expect(result.current.state?.version).toBe(4)
  })

  it('adopts stale-409 state while answering a destructive confirmation', async () => {
    vi.stubGlobal('confirm', vi.fn(() => true))
    fetchMock.mockImplementation((url: string) => {
      const path = String(url)
      if (path.includes('/api/game/confirm'))
        return Promise.resolve({
          ok: false,
          status: 409,
          json: () =>
            Promise.resolve({
              detail: 'the board changed',
              stale: true,
              version: 2,
              state: state({ version: 2, fen: AFTER_E4_FEN }),
            }),
        })
      if (path.includes('/api/game/new'))
        return Promise.resolve({
          ok: false,
          status: 409,
          json: () =>
            Promise.resolve({ detail: 'Start a new one?', confirm: true, op: 'new_game' }),
        })
      if (path.includes('/api/settings'))
        return jsonResponse({
          verbosity: 'normal',
          hints_mode: false,
          voice_output: false,
          tier: 'casual',
          skill_level: null,
          elo: null,
        })
      return jsonResponse(state())
    })
    const { result } = renderHook(() => useGame())
    await waitFor(() => expect(result.current.state?.version).toBe(1))

    await act(async () => {
      await result.current.newGame()
    })

    expect(result.current.state?.version).toBe(2)
    expect(result.current.state?.fen).toBe(AFTER_E4_FEN)
  })

  it('keeps working with an older backend that omits state versions', async () => {
    const unversioned = state()
    delete unversioned.version
    fetchMock.mockImplementation((url: string) => {
      if (String(url).includes('/api/game/move'))
        return jsonResponse({ ...moveResponse, state: unversioned })
      if (String(url).includes('/api/settings'))
        return jsonResponse({
          verbosity: 'normal',
          hints_mode: false,
          voice_output: false,
          tier: 'casual',
          skill_level: null,
          elo: null,
        })
      return jsonResponse(unversioned)
    })
    const { result } = renderHook(() => useGame())
    await waitFor(() => expect(result.current.state).not.toBeNull())

    await act(async () => {
      await result.current.play('e2', 'e4')
    })

    expect(fetchMock).toHaveBeenCalledWith(
      '/api/game/move',
      expect.objectContaining({ body: JSON.stringify({ move: 'e2e4' }) }),
    )
  })

  it('cites the rendered version on command and lifecycle mutations', async () => {
    const { result } = renderHook(() => useGame())
    await waitFor(() => expect(result.current.state?.version).toBe(1))

    await act(async () => {
      await result.current.sendCommand('play e4')
      await result.current.undo()
      await result.current.newGame('black')
      await result.current.resign()
    })

    const bodyFor = (path: string) => {
      const call = fetchMock.mock.calls.find(([url]) => String(url).includes(path))
      return JSON.parse(String(call?.[1]?.body))
    }
    expect(bodyFor('/api/command')).toEqual({ text: 'play e4', version: 1 })
    expect(bodyFor('/api/game/undo')).toEqual({ version: 1 })
    expect(bodyFor('/api/game/new')).toEqual({ color: 'black', version: 1 })
    expect(bodyFor('/api/game/resign')).toEqual({ version: 1 })
  })

  // --- live turn progress (audit item 19) ------------------------------------
  //
  // The events are *broadcast*, which is the point: a dragged move and a turn
  // another client started both light this up, and neither goes through
  // `sendCommand`.

  it('is quiet between turns', async () => {
    const { result } = renderHook(() => useGame())
    await waitFor(() => expect(result.current.state).not.toBeNull())
    expect(result.current.agentProgress).toBeNull()
  })

  it('shows what the turn in flight is doing', async () => {
    const { result } = renderHook(() => useGame())
    await waitFor(() => expect(result.current.state).not.toBeNull())
    act(() => {
      FakeWebSocket.instances[0].emit(progressFrame('begin'))
      FakeWebSocket.instances[0].emit(progressFrame('tool', 'make_move'))
    })
    expect(result.current.agentProgress).toBe('Validating your move')
    act(() => {
      FakeWebSocket.instances[0].emit(progressFrame('phase', 'engine_calculating'))
    })
    expect(result.current.agentProgress).toBe('Stockfish is calculating')
  })

  it('clears the line when the turn ends', async () => {
    const { result } = renderHook(() => useGame())
    await waitFor(() => expect(result.current.state).not.toBeNull())
    act(() => {
      FakeWebSocket.instances[0].emit(progressFrame('begin'))
      FakeWebSocket.instances[0].emit(progressFrame('tool', 'make_move'))
      FakeWebSocket.instances[0].emit(progressFrame('end'))
    })
    expect(result.current.agentProgress).toBeNull()
  })

  it('leaves the board alone — progress is not state', async () => {
    const { result } = renderHook(() => useGame())
    await waitFor(() => expect(result.current.state).not.toBeNull())
    const before = result.current.revision
    act(() => {
      FakeWebSocket.instances[0].emit(progressFrame('tool', 'make_move'))
    })
    expect(result.current.revision).toBe(before)
    expect(result.current.state?.fen).toBe(START_FEN)
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
      expect.objectContaining({
        method: 'POST',
        body: JSON.stringify({ move: 'e2e4', version: 1 }),
      }),
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
      expect.objectContaining({ body: JSON.stringify({ move: 'e7e8q', version: 1 }) }),
    )
    expect(result.current.pendingPromotion).toBeNull()
    expect(result.current.state?.fen).toBe(AFTER_PROMO_FEN)
  })

  it('starts a new game and clears any move feedback', async () => {
    const { result } = renderHook(() => useGame())
    await waitFor(() => expect(result.current.state).not.toBeNull())
    // Return a fresh start position from the new-game endpoint (the gate stood
    // aside: 200 with a state, no question to ask).
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
    // No plies: the backend decides the player's takeback (full exchange
    // vs the engine, one ply engine-free).
    expect(fetchMock).toHaveBeenCalledWith(
      '/api/game/undo',
      expect.objectContaining({ method: 'POST', body: JSON.stringify({ version: 1 }) }),
    )
  })

  it('starts a new game as a requested color', async () => {
    const { result } = renderHook(() => useGame())
    await waitFor(() => expect(result.current.state).not.toBeNull())
    await act(async () => {
      await result.current.newGame('black')
    })
    expect(fetchMock).toHaveBeenCalledWith(
      '/api/game/new',
      expect.objectContaining({
        method: 'POST',
        body: JSON.stringify({ color: 'black', version: 1 }),
      }),
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
    await waitFor(() => expect(result.current.tier).toBe('casual'))
  })

  it('sets difficulty by tier without touching board state', async () => {
    const { result } = renderHook(() => useGame())
    await waitFor(() => expect(result.current.state).not.toBeNull())
    const revisionBefore = result.current.revision
    await act(async () => {
      await result.current.setDifficulty('advanced')
    })
    expect(fetchMock).toHaveBeenCalledWith(
      '/api/game/difficulty',
      expect.objectContaining({ method: 'POST', body: JSON.stringify({ tier: 'advanced' }) }),
    )
    // The hook reflects only what the server confirmed.
    expect(result.current.tier).toBe('advanced')
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
      expect.objectContaining({
        method: 'POST',
        body: JSON.stringify({ text: 'play e4', version: 1 }),
      }),
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

  it('syncs the difficulty tier from the command response (agent-side change)', async () => {
    const { result } = renderHook(() => useGame())
    await waitFor(() => expect(result.current.tier).toBe('casual'))
    // The agent changed strength via set_difficulty: tier rides the response.
    fetchMock.mockImplementation((url: string) => {
      if (String(url).includes('/api/command'))
        return jsonResponse({
          commentary: 'Cranked up.',
          tool_results: [],
          state: state(),
          speak: false,
          tier: 'advanced',
        })
      return jsonResponse(state())
    })
    await act(async () => {
      await result.current.sendCommand('make it harder')
    })
    expect(result.current.tier).toBe('advanced')
  })

  it('clears the tier when the agent sets a raw strength', async () => {
    const { result } = renderHook(() => useGame())
    await waitFor(() => expect(result.current.tier).toBe('casual'))
    // A raw elo has no named tier; null must reach the selector so it stops
    // highlighting a strength the engine is no longer playing.
    fetchMock.mockImplementation((url: string) => {
      if (String(url).includes('/api/command'))
        return jsonResponse({
          commentary: 'About 1500 now.',
          tool_results: [],
          state: state(),
          speak: false,
          tier: null,
        })
      return jsonResponse(state())
    })
    await act(async () => {
      await result.current.sendCommand('play at 1500')
    })
    expect(result.current.tier).toBeNull()
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

  // --- agent mode on the board: a drag is a turn -------------------------
  //
  // In agent mode the move endpoint runs the same beats a typed move does, so
  // the response carries Glitch's reaction to the drag (and whether to voice
  // it). Direct mode sends neither key, and the bubble is left alone.

  it('stages and voices the reaction to a dragged move', async () => {
    const { playText } = await import('./tts')
    vi.mocked(playText).mockClear()
    const { result } = renderHook(() => useGame())
    await waitFor(() => expect(result.current.state).not.toBeNull())
    moveResponse = {
      ...moveResponse,
      commentary: 'Bold opener.\n\ne5.',
      speak: true,
    }
    await act(async () => {
      await result.current.play('e2', 'e4')
    })
    expect(result.current.commentary).toBe('Bold opener.\n\ne5.')
    expect(playText).toHaveBeenCalledWith('Bold opener.\n\ne5.')
  })

  it('shows a dragged move reaction without speaking it when voice is off', async () => {
    const { playText } = await import('./tts')
    vi.mocked(playText).mockClear()
    const { result } = renderHook(() => useGame())
    await waitFor(() => expect(result.current.state).not.toBeNull())
    moveResponse = { ...moveResponse, commentary: 'Quiet one.', speak: false }
    await act(async () => {
      await result.current.play('e2', 'e4')
    })
    expect(result.current.commentary).toBe('Quiet one.')
    expect(playText).not.toHaveBeenCalled()
  })

  it('leaves the bubble alone for a direct-mode drag', async () => {
    const { result } = renderHook(() => useGame())
    await waitFor(() => expect(result.current.state).not.toBeNull())
    // Direct mode: no commentary key at all on the move response.
    await act(async () => {
      await result.current.play('e2', 'e4')
    })
    expect(result.current.commentary).toBeNull()
  })

  it('reports direct mode from the settings document', async () => {
    fetchMock.mockImplementation((url: string) => {
      if (String(url).includes('/api/settings'))
        return jsonResponse({
          verbosity: 'normal',
          hints_mode: false,
          voice_output: false,
          tier: 'casual',
          skill_level: null,
          elo: null,
          agent_available: false,
        })
      return jsonResponse(state())
    })
    const { result } = renderHook(() => useGame())
    await waitFor(() => expect(result.current.agentAvailable).toBe(false))
  })

  // --- the destructive-op gate, from the buttons -------------------------
  //
  // Mid-game the backend arms the op and answers 409 with its question rather
  // than throwing the game away; the hook puts that question to the player and
  // sends the answer back to the same armed op a spoken "yes" would answer.

  function gated(op: string, detail: string) {
    return (url: string) => {
      const path = String(url)
      if (path.includes('/api/game/confirm'))
        return jsonResponse({ op, confirmed: true, state: state({ history: [] }) })
      if (path.includes(`/api/game/${op === 'new_game' ? 'new' : 'resign'}`))
        return Promise.resolve({
          ok: false,
          status: 409,
          json: () => Promise.resolve({ detail, confirm: true, op }),
        })
      return jsonResponse(state())
    }
  }

  it('asks before a gated new game, then confirms it', async () => {
    const confirmSpy = vi.fn(() => true)
    vi.stubGlobal('confirm', confirmSpy)
    const { result } = renderHook(() => useGame())
    await waitFor(() => expect(result.current.state).not.toBeNull())
    fetchMock.mockImplementation(gated('new_game', 'Start a new one?'))
    act(() => {
      FakeWebSocket.instances[0].emit({ type: 'state', state: reviewableState() })
    })

    await act(async () => {
      await result.current.newGame()
    })

    expect(confirmSpy).toHaveBeenCalledWith('Start a new one?')
    expect(fetchMock).toHaveBeenCalledWith(
      '/api/game/confirm',
      expect.objectContaining({
        method: 'POST',
        body: JSON.stringify({ confirm: true, version: 1 }),
      }),
    )
    expect(result.current.state?.history).toEqual([])
  })

  it('cancels a gated new game and leaves the board where it was', async () => {
    vi.stubGlobal('confirm', vi.fn(() => false))
    const { result } = renderHook(() => useGame())
    await waitFor(() => expect(result.current.state).not.toBeNull())
    act(() => {
      FakeWebSocket.instances[0].emit({ type: 'state', state: reviewableState() })
    })
    fetchMock.mockImplementation(gated('new_game', 'Start a new one?'))
    const revisionBefore = result.current.revision

    await act(async () => {
      await result.current.newGame()
    })

    // The cancel still goes to the server — it is what disarms the op — but
    // nothing on the board moves.
    expect(fetchMock).toHaveBeenCalledWith(
      '/api/game/confirm',
      expect.objectContaining({ body: JSON.stringify({ confirm: false, version: 1 }) }),
    )
    expect(result.current.state?.history).toEqual(['e4', 'e5'])
    expect(result.current.revision).toBe(revisionBefore)
  })

  it('asks before a gated resignation, then confirms it', async () => {
    const confirmSpy = vi.fn(() => true)
    vi.stubGlobal('confirm', confirmSpy)
    const { result } = renderHook(() => useGame())
    await waitFor(() => expect(result.current.state).not.toBeNull())
    fetchMock.mockImplementation((url: string) => {
      const path = String(url)
      if (path.includes('/api/game/confirm'))
        return jsonResponse({
          op: 'resign',
          confirmed: true,
          state: state({ game_over: true }),
        })
      if (path.includes('/api/game/resign'))
        return Promise.resolve({
          ok: false,
          status: 409,
          json: () => Promise.resolve({ detail: 'Resign?', confirm: true, op: 'resign' }),
        })
      return jsonResponse(state())
    })

    await act(async () => {
      await result.current.resign()
    })

    expect(confirmSpy).toHaveBeenCalledWith('Resign?')
    expect(result.current.state?.game_over).toBe(true)
  })

  it('does not ask when the gate stood aside', async () => {
    const confirmSpy = vi.fn(() => true)
    vi.stubGlobal('confirm', confirmSpy)
    const { result } = renderHook(() => useGame())
    await waitFor(() => expect(result.current.state).not.toBeNull())
    await act(async () => {
      await result.current.newGame('black')
    })
    expect(confirmSpy).not.toHaveBeenCalled()
  })

  // --- history review: browse past positions without undoing ------------

  async function renderReviewable() {
    const rendered = renderHook(() => useGame())
    await waitFor(() => expect(rendered.result.current.state).not.toBeNull())
    act(() => {
      FakeWebSocket.instances[0].emit({ type: 'state', state: reviewableState() })
    })
    return rendered
  }

  it('shows the live position and is not reviewing by default', async () => {
    const { result } = await renderReviewable()
    expect(result.current.reviewing).toBe(false)
    expect(result.current.viewPly).toBeNull()
    expect(result.current.displayFen).toBe(AFTER_E4_E5_FEN)
  })

  it('steps back from live to the previous position without submitting anything', async () => {
    const { result } = await renderReviewable()
    const callsBefore = fetchMock.mock.calls.length
    act(() => result.current.stepBack())
    expect(result.current.reviewing).toBe(true)
    expect(result.current.viewPly).toBe(1)
    expect(result.current.displayFen).toBe(AFTER_E4_FEN)
    // Review is client-side only — no backend mutation.
    expect(fetchMock.mock.calls.length).toBe(callsBefore)
  })

  it('clamps stepping back at the root position', async () => {
    const { result } = await renderReviewable()
    act(() => result.current.stepBack())
    act(() => result.current.stepBack())
    expect(result.current.viewPly).toBe(0)
    expect(result.current.displayFen).toBe(START_FEN)
    act(() => result.current.stepBack())
    expect(result.current.viewPly).toBe(0)
  })

  it('steps forward back to live at the last position', async () => {
    const { result } = await renderReviewable()
    act(() => result.current.stepBack())
    act(() => result.current.stepForward())
    expect(result.current.reviewing).toBe(false)
    expect(result.current.viewPly).toBeNull()
    expect(result.current.displayFen).toBe(AFTER_E4_E5_FEN)
  })

  it('ignores stepping forward while live', async () => {
    const { result } = await renderReviewable()
    act(() => result.current.stepForward())
    expect(result.current.viewPly).toBeNull()
  })

  it('snaps back to live when an authoritative state arrives during review', async () => {
    const { result } = await renderReviewable()
    act(() => result.current.stepBack())
    expect(result.current.reviewing).toBe(true)
    act(() => {
      FakeWebSocket.instances[0].emit({ type: 'state', state: reviewableState() })
    })
    expect(result.current.reviewing).toBe(false)
    expect(result.current.displayFen).toBe(AFTER_E4_E5_FEN)
  })

  it('does not submit moves while reviewing', async () => {
    const { result } = await renderReviewable()
    act(() => result.current.stepBack())
    await act(async () => {
      await result.current.play('e2', 'e4')
    })
    expect(fetchMock).not.toHaveBeenCalledWith('/api/game/move', expect.anything())
  })

  // --- hint: engine best move as a board arrow ---------------------------

  it('fetches a hint and exposes it as a board arrow shape', async () => {
    const { result } = renderHook(() => useGame())
    await waitFor(() => expect(result.current.state).not.toBeNull())
    expect(result.current.hintShapes).toEqual([])
    await act(async () => {
      await result.current.requestHint()
    })
    expect(result.current.hintShapes).toEqual([{ orig: 'e2', dest: 'e4', brush: 'green' }])
  })

  it('clears the hint arrow when a new authoritative state arrives', async () => {
    const { result } = renderHook(() => useGame())
    await waitFor(() => expect(result.current.state).not.toBeNull())
    await act(async () => {
      await result.current.requestHint()
    })
    act(() => {
      FakeWebSocket.instances[0].emit({ type: 'state', state: state({ fen: AFTER_E4_FEN }) })
    })
    expect(result.current.hintShapes).toEqual([])
  })

  it('leaves the hint empty when the backend refuses (no engine)', async () => {
    const { result } = renderHook(() => useGame())
    await waitFor(() => expect(result.current.state).not.toBeNull())
    fetchMock.mockImplementation((url: string) => {
      if (String(url).includes('/api/game/hint'))
        return Promise.resolve({ ok: false, json: () => Promise.resolve({}) })
      return jsonResponse(state())
    })
    await act(async () => {
      await result.current.requestHint()
    })
    expect(result.current.hintShapes).toEqual([])
  })

  it('ignores hint requests while reviewing', async () => {
    const { result } = await renderReviewable()
    act(() => result.current.stepBack())
    const callsBefore = fetchMock.mock.calls.length
    await act(async () => {
      await result.current.requestHint()
    })
    expect(fetchMock.mock.calls.length).toBe(callsBefore)
    expect(result.current.hintShapes).toEqual([])
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
