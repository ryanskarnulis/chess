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

/** A PGN as `export_pgn` now hands it over: real headers, then the movetext. */
const PGN = '[Event "Casual game"]\n[White "Player"]\n\n1. e4 e5 *'

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
  onopen: (() => void) | null = null
  onclose: (() => void) | null = null
  onerror: (() => void) | null = null
  close = vi.fn(() => this.emitClose())
  url: string
  constructor(url: string) {
    this.url = url
    FakeWebSocket.instances.push(this)
  }
  emit(data: unknown) {
    this.onmessage?.({ data: JSON.stringify(data) })
  }
  emitOpen() {
    this.onopen?.()
  }
  emitClose() {
    this.onclose?.()
  }
  emitError() {
    this.onerror?.()
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
      // `version` is the board the engine analyzed, as the real endpoint sends
      // it — matching `state()` here, so the default hint is for the live board.
      return jsonResponse({ uci: 'e2e4', san: 'e4', from: 'e2', to: 'e4', version: 1 })
    if (path.includes('/api/settings/voice')) return jsonResponse({ voice_output: true })
    if (path.includes('/api/settings'))
      return jsonResponse({
        verbosity: 'normal',
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
  vi.useRealTimers()
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

  it('reconnects once after an unexpected close, after a delay', async () => {
    vi.useFakeTimers()
    renderHook(() => useGame())
    const first = FakeWebSocket.instances[0]

    act(() => first.emitClose())
    expect(FakeWebSocket.instances).toHaveLength(1)

    await act(async () => vi.advanceTimersByTimeAsync(999))
    expect(FakeWebSocket.instances).toHaveLength(1)
    await act(async () => vi.advanceTimersByTimeAsync(1))
    expect(FakeWebSocket.instances).toHaveLength(2)
  })

  it('fetches and applies fresh authoritative state when a reconnect opens', async () => {
    const { result } = renderHook(() => useGame())
    await waitFor(() => expect(result.current.state?.version).toBe(1))
    fetchMock.mockImplementation((url: string) => {
      if (String(url).includes('/api/state'))
        return jsonResponse(state({ version: 4, fen: AFTER_E4_FEN, turn: 'black' }))
      if (String(url).includes('/api/settings'))
        return jsonResponse({
          verbosity: 'normal',
          voice_output: false,
          tier: 'casual',
          skill_level: null,
          elo: null,
        })
      return jsonResponse(state())
    })
    vi.useFakeTimers()

    act(() => FakeWebSocket.instances[0].emitClose())
    await act(async () => vi.advanceTimersByTimeAsync(1_000))
    act(() => FakeWebSocket.instances[1].emitOpen())
    await act(async () => Promise.resolve())

    expect(result.current.state?.version).toBe(4)
    expect(result.current.state?.fen).toBe(AFTER_E4_FEN)
    expect(
      fetchMock.mock.calls.filter(([url]) => String(url).includes('/api/state')),
    ).toHaveLength(2)
  })

  it('backs off failed reconnects exponentially and caps the delay at ten seconds', async () => {
    vi.useFakeTimers()
    renderHook(() => useGame())

    for (const delay of [1_000, 2_000, 4_000, 8_000, 10_000, 10_000]) {
      const current = FakeWebSocket.instances.at(-1)!
      act(() => current.emitClose())
      await act(async () => vi.advanceTimersByTimeAsync(delay - 1))
      expect(FakeWebSocket.instances.at(-1)).toBe(current)
      await act(async () => vi.advanceTimersByTimeAsync(1))
      expect(FakeWebSocket.instances.at(-1)).not.toBe(current)
    }
  })

  it('lets close own the retry after an error and ignores repeated close events', async () => {
    vi.useFakeTimers()
    renderHook(() => useGame())
    const first = FakeWebSocket.instances[0]
    const repeatedClose = first.onclose

    act(() => {
      first.emitError()
      repeatedClose?.()
    })
    expect(first.close).toHaveBeenCalledOnce()

    await act(async () => vi.advanceTimersByTimeAsync(1_000))
    expect(FakeWebSocket.instances).toHaveLength(2)
    await act(async () => vi.advanceTimersByTimeAsync(10_000))
    expect(FakeWebSocket.instances).toHaveLength(2)
  })

  it('cleans up socket handlers and retry timers without reconnecting on unmount', async () => {
    vi.useFakeTimers()
    const firstHook = renderHook(() => useGame())
    const unexpectedlyClosed = FakeWebSocket.instances[0]
    act(() => unexpectedlyClosed.emitClose())
    expect(vi.getTimerCount()).toBe(1)
    expect(unexpectedlyClosed.onclose).toBeNull()

    firstHook.unmount()
    expect(vi.getTimerCount()).toBe(0)
    await act(async () => vi.advanceTimersByTimeAsync(10_000))
    expect(FakeWebSocket.instances).toHaveLength(1)

    const secondHook = renderHook(() => useGame())
    const intentionallyClosed = FakeWebSocket.instances[1]
    secondHook.unmount()
    expect(intentionallyClosed.close).toHaveBeenCalledOnce()
    expect(intentionallyClosed.onopen).toBeNull()
    expect(intentionallyClosed.onmessage).toBeNull()
    expect(intentionallyClosed.onclose).toBeNull()
    expect(intentionallyClosed.onerror).toBeNull()
    await act(async () => vi.advanceTimersByTimeAsync(10_000))
    expect(FakeWebSocket.instances).toHaveLength(2)
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
      await result.current.claimDraw()
    })

    const bodyFor = (path: string) => {
      const call = fetchMock.mock.calls.find(([url]) => String(url).includes(path))
      return JSON.parse(String(call?.[1]?.body))
    }
    expect(bodyFor('/api/command')).toEqual({ text: 'play e4', version: 1 })
    expect(bodyFor('/api/game/undo')).toEqual({ version: 1 })
    expect(bodyFor('/api/game/new')).toEqual({ color: 'black', version: 1 })
    expect(bodyFor('/api/game/resign')).toEqual({ version: 1 })
    expect(bodyFor('/api/game/claim-draw')).toEqual({ version: 1 })
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

  /** Answer any request whose path contains `fragment` with `answer`, leaving
   * every other route on whatever implementation was installed before the call
   * — so successive calls compose, each wrapping the last. */
  function routeAnswers(fragment: string, answer: () => Promise<unknown>) {
    const rest = fetchMock.getMockImplementation()
    fetchMock.mockImplementation((url: string) =>
      String(url).includes(fragment) ? answer() : rest?.(url),
    )
  }

  /** Answer /api/game/move with `answer`, leaving every other route on the
   * default implementation from `beforeEach`. */
  function moveAnswers(answer: () => Promise<unknown>) {
    routeAnswers('/api/game/move', answer)
  }

  /** A move that never landed leaves the board exactly where it was, and must
   * still snap the dragged piece back: nothing else will, because a refused
   * move broadcasts no state frame (#231). */
  async function expectRefusedMove(
    result: { current: ReturnType<typeof useGame> },
    revisionBefore: number,
    detail: string,
  ) {
    await act(async () => {
      // The board discards this promise, so a rejection here would escape
      // unhandled with no feedback at all. It must always resolve.
      await expect(result.current.play('e2', 'e4')).resolves.toBeUndefined()
    })
    expect(result.current.moveError).toBe(detail)
    expect(result.current.state?.fen).toBe(START_FEN)
    expect(result.current.state?.version).toBe(1)
    expect(result.current.revision).toBeGreaterThan(revisionBefore)
  }

  it('surfaces a non-stale 409 and snaps the piece back without applying state', async () => {
    const { result } = renderHook(() => useGame())
    await waitFor(() => expect(result.current.state).not.toBeNull())
    const revisionBefore = result.current.revision
    // The turn coordinator's documented rejection: a detail and no `state` at
    // all, so there is nothing to adopt and `apply` must never see it.
    moveAnswers(() =>
      Promise.resolve({
        ok: false,
        status: 409,
        json: () =>
          Promise.resolve({ detail: 'cannot apply a player move while engine_replying' }),
      }),
    )

    await expectRefusedMove(
      result,
      revisionBefore,
      'cannot apply a player move while engine_replying',
    )
  })

  it('surfaces a transport failure the same way', async () => {
    const { result } = renderHook(() => useGame())
    await waitFor(() => expect(result.current.state).not.toBeNull())
    const revisionBefore = result.current.revision
    // The backend container restarts on every merge, so a refused connection
    // mid-drag is routine.
    moveAnswers(() => Promise.reject(new TypeError('Failed to fetch')))

    await expectRefusedMove(
      result,
      revisionBefore,
      'Could not reach the server — the move was not played.',
    )
  })

  it('surfaces a gateway error whose body is not JSON', async () => {
    const { result } = renderHook(() => useGame())
    await waitFor(() => expect(result.current.state).not.toBeNull())
    const revisionBefore = result.current.revision
    // A proxy 502 carries an HTML page, so the body has no detail to relay.
    moveAnswers(() =>
      Promise.resolve({
        ok: false,
        status: 502,
        json: () => Promise.reject(new SyntaxError('Unexpected token <')),
      }),
    )

    await expectRefusedMove(result, revisionBefore, 'The server refused the move (502).')
  })

  it('refuses a 200 that carries no state rather than applying undefined', async () => {
    const { result } = renderHook(() => useGame())
    await waitFor(() => expect(result.current.state).not.toBeNull())
    const revisionBefore = result.current.revision
    // A truncated body still parses as OK at the HTTP layer. `apply` may only
    // ever be handed a real GameState.
    moveAnswers(() =>
      Promise.resolve({
        ok: true,
        status: 200,
        json: () => Promise.reject(new SyntaxError('Unexpected end of JSON input')),
      }),
    )

    await expectRefusedMove(
      result,
      revisionBefore,
      'The server sent an unusable response — the move was not played.',
    )
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

  /** Arm the promotion picker: put a promotable board on screen at `version`,
   * then drag the pawn onto the last rank. Returns the hook handle. */
  async function armPromotion(version: number) {
    const { result } = renderHook(() => useGame())
    await waitFor(() => expect(result.current.state?.version).toBe(1))
    act(() => {
      FakeWebSocket.instances[0].emit({
        type: 'state',
        state: state({ version, fen: PROMO_FEN }),
      })
    })
    await act(async () => {
      await result.current.play('e7', 'e8')
    })
    expect(result.current.pendingPromotion).toEqual({ from: 'e7', to: 'e8' })
    return result
  }

  /** Every move body the hook has sent, oldest first. */
  function submittedMoves() {
    return fetchMock.mock.calls
      .filter(([url]) => String(url).includes('/api/game/move'))
      .map(([, init]) => JSON.parse(String(init.body)))
  }

  it('invalidates a pending promotion when a newer authoritative board arrives', async () => {
    const result = await armPromotion(2)
    const revisionBefore = result.current.revision

    // Another client moved — or a command, an undo, a new game. The board the
    // drag was made against is gone (#222).
    act(() => {
      FakeWebSocket.instances[0].emit({
        type: 'state',
        state: state({ version: 3, fen: AFTER_E4_FEN, turn: 'black' }),
      })
    })

    // The picker closing is the visible symptom: App renders it off this.
    expect(result.current.pendingPromotion).toBeNull()
    // `apply` bumps the revision, so the pawn sitting on the last rank snaps
    // back with the rest of the re-sync — no separate snap needed here.
    expect(result.current.revision).toBeGreaterThan(revisionBefore)
  })

  it('submits nothing when a promotion piece is chosen after the board moved on', async () => {
    const result = await armPromotion(2)
    act(() => {
      FakeWebSocket.instances[0].emit({
        type: 'state',
        state: state({ version: 3, fen: AFTER_E4_FEN, turn: 'black' }),
      })
    })

    // The click can still arrive — a rendered handler, a keypress in flight.
    await act(async () => {
      await result.current.completePromotion('q')
    })

    // Nothing at all goes out: on the new board `e7e8q` is either an illegal
    // move round trip or, worse, a legal move the player never asked for.
    expect(submittedMoves()).toEqual([])
    expect(result.current.state?.version).toBe(3)
    expect(result.current.state?.fen).toBe(AFTER_E4_FEN)
  })

  /**
   * Deliberate: an equal-version frame is the *same* board — a reconnect
   * re-sync or a duplicated broadcast — and the picker stays open across it.
   * `apply` accepts equal versions precisely so those redundant snapshots
   * land, and closing the picker on one would throw away input the player has
   * already given for a board that never moved. (`apply` does drop the hint
   * arrow and the review view on the same frame; those are derived displays
   * the app can recompute on demand, a half-finished promotion is not.)
   */
  it('keeps a pending promotion across an equal-version re-sync of the same board', async () => {
    const result = await armPromotion(2)

    act(() => {
      FakeWebSocket.instances[0].emit({
        type: 'state',
        state: state({ version: 2, fen: PROMO_FEN }),
      })
    })
    expect(result.current.pendingPromotion).toEqual({ from: 'e7', to: 'e8' })

    // And the choice still goes through, citing the board it was armed on.
    moveResponse = {
      legal: true,
      san: 'e8=Q',
      uci: 'e7e8q',
      reason: null,
      engine_move: null,
      state: state({ version: 3, fen: AFTER_PROMO_FEN, turn: 'black' }),
    }
    await act(async () => {
      await result.current.completePromotion('q')
    })
    expect(submittedMoves()).toEqual([{ move: 'e7e8q', version: 2 }])
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

  it('claims a draw and applies the returned state', async () => {
    const { result } = renderHook(() => useGame())
    await waitFor(() => expect(result.current.state).not.toBeNull())
    routeAnswers('/api/game/claim-draw', () =>
      jsonResponse({
        outcome: { termination: 'fifty_moves', winner: null, result: '1/2-1/2' },
        state: state({ game_over: true, version: 2 }),
      }),
    )
    await act(async () => {
      await result.current.claimDraw()
    })
    expect(fetchMock).toHaveBeenCalledWith(
      '/api/game/claim-draw',
      expect.objectContaining({ method: 'POST' }),
    )
    expect(result.current.state?.game_over).toBe(true)
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

  // --- the PGN a reply exported ------------------------------------------
  //
  // The notation is no longer in the words: `export_pgn`'s description tells
  // Glitch to say it is ready and recite nothing, so the UI takes it off the
  // tool result and renders it with a copy button.

  it('keeps the PGN a command exported', async () => {
    const { result } = renderHook(() => useGame())
    await waitFor(() => expect(result.current.state).not.toBeNull())
    fetchMock.mockImplementation((url: string) => {
      if (String(url).includes('/api/command'))
        return jsonResponse({
          commentary: "Exported. It's yours to copy.",
          tool_results: [{ name: 'export_pgn', result: { ok: true, pgn: PGN } }],
          state: state(),
        })
      return jsonResponse(state())
    })
    await act(async () => {
      await result.current.sendCommand('export the pgn')
    })
    expect(result.current.pgn).toBe(PGN)
  })

  it('ignores an export that failed', async () => {
    const { result } = renderHook(() => useGame())
    await waitFor(() => expect(result.current.state).not.toBeNull())
    // `tool_results` is unvalidated wire JSON, and a refusal is in it too. A
    // copy button over a refusal would copy nothing and say it worked.
    fetchMock.mockImplementation((url: string) => {
      if (String(url).includes('/api/command'))
        return jsonResponse({
          commentary: 'That did not work.',
          tool_results: [{ name: 'export_pgn', result: { ok: false, error: 'nope' } }],
          state: state(),
        })
      return jsonResponse(state())
    })
    await act(async () => {
      await result.current.sendCommand('export the pgn')
    })
    expect(result.current.pgn).toBeNull()
  })

  it('drops the PGN as soon as the next command is sent', async () => {
    const { result } = renderHook(() => useGame())
    await waitFor(() => expect(result.current.state).not.toBeNull())
    fetchMock.mockImplementation((url: string) => {
      if (String(url).includes('/api/command'))
        return jsonResponse({
          commentary: 'Exported.',
          tool_results: [{ name: 'export_pgn', result: { ok: true, pgn: PGN } }],
          state: state(),
        })
      return jsonResponse(state())
    })
    await act(async () => {
      await result.current.sendCommand('export the pgn')
    })
    expect(result.current.pgn).toBe(PGN)

    // Held open deliberately: the chip belongs to the reply it came with, so
    // what has to be observed is the state *during* the next turn, which an
    // instantly-resolving mock would flush past inside `act`.
    let release!: () => void
    fetchMock.mockImplementation((url: string) => {
      if (String(url).includes('/api/command'))
        return new Promise((resolve) => {
          release = () =>
            resolve({
              ok: true,
              json: () =>
                Promise.resolve({
                  commentary: 'Holding steady.',
                  tool_results: [],
                  state: state(),
                }),
            })
        })
      return jsonResponse(state())
    })
    let sent!: Promise<void>
    await act(async () => {
      sent = result.current.sendCommand('how am I doing?')
    })
    expect(result.current.agentThinking).toBe(true)
    expect(result.current.pgn).toBeNull()
    await act(async () => {
      release()
      await sent
    })
    expect(result.current.pgn).toBeNull()
  })

  it('drops the PGN when a new game starts', async () => {
    const { result } = renderHook(() => useGame())
    await waitFor(() => expect(result.current.state).not.toBeNull())
    fetchMock.mockImplementation((url: string) => {
      if (String(url).includes('/api/command'))
        return jsonResponse({
          commentary: 'Exported.',
          tool_results: [{ name: 'export_pgn', result: { ok: true, pgn: PGN } }],
          state: state(),
        })
      if (String(url).includes('/api/game/')) return jsonResponse({ state: state() })
      return jsonResponse(state())
    })
    await act(async () => {
      await result.current.sendCommand('export the pgn')
    })
    await act(async () => {
      await result.current.newGame()
    })
    // Whatever was exported was the old game.
    expect(result.current.pgn).toBeNull()
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

  it('reports an unavailable agent when the command never reaches the server', async () => {
    const { result } = renderHook(() => useGame())
    await waitFor(() => expect(result.current.state).not.toBeNull())
    const revisionBefore = result.current.revision
    // The container restarts on every merge to main, so a refused connection
    // mid-command is routine. It has to land on the same documented
    // "unavailable" line: this promise is discarded by CommandBox's
    // `void onSubmit` and awaited by the hands-free loop, so a rejection is
    // either silence or a wedged mic (#232).
    fetchMock.mockImplementation((url: string) => {
      if (String(url).includes('/api/command'))
        return Promise.reject(new TypeError('Failed to fetch'))
      return jsonResponse(state())
    })
    await act(async () => {
      await expect(result.current.sendCommand('play e4')).resolves.toBeUndefined()
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

  it('asks before a gated draw claim, then confirms it', async () => {
    const confirmSpy = vi.fn(() => true)
    vi.stubGlobal('confirm', confirmSpy)
    const { result } = renderHook(() => useGame())
    await waitFor(() => expect(result.current.state).not.toBeNull())
    fetchMock.mockImplementation((url: string) => {
      const path = String(url)
      if (path.includes('/api/game/confirm'))
        return jsonResponse({
          op: 'claim_draw',
          confirmed: true,
          state: state({
            game_over: true,
            outcome: { termination: 'threefold_repetition', winner: null, result: '1/2-1/2' },
          }),
        })
      if (path.includes('/api/game/claim-draw'))
        return Promise.resolve({
          ok: false,
          status: 409,
          json: () =>
            Promise.resolve({
              detail: 'That ends the game in a draw. Claim it?',
              confirm: true,
              op: 'claim_draw',
            }),
        })
      return jsonResponse(state())
    })

    await act(async () => {
      await result.current.claimDraw()
    })

    expect(confirmSpy).toHaveBeenCalledWith('That ends the game in a draw. Claim it?')
    expect(fetchMock).toHaveBeenCalledWith(
      '/api/game/confirm',
      expect.objectContaining({ body: JSON.stringify({ confirm: true, version: 1 }) }),
    )
    expect(result.current.state?.outcome?.termination).toBe('threefold_repetition')
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

  // --- transport failures on the remaining helpers ------------------------
  //
  // The defect class #231/#232 fixed for moves, commands and transcription,
  // applied to the rest of the client: every one of these promises ends in a
  // handler or mount effect that discards it (a button's onClick, `void
  // fetchState().then(...)`), so a rejection escapes unhandled — no feedback,
  // no snap-back, nothing. The backend container restarts on every merge to
  // main, so a refused connection under any of these is routine, not exotic.

  const REFUSED_CONNECTION = () => Promise.reject(new TypeError('Failed to fetch'))
  const TRUNCATED_OK_BODY = () =>
    Promise.resolve({
      ok: true,
      status: 200,
      json: () => Promise.reject(new SyntaxError('Unexpected end of JSON input')),
    })

  it('keeps running when the initial state fetch never reaches the server', async () => {
    routeAnswers('/api/state', REFUSED_CONNECTION)
    const { result } = renderHook(() => useGame())
    // Settings still load: the hook did not die with the state fetch.
    await waitFor(() => expect(result.current.voiceOutput).toBe(false))
    expect(result.current.state).toBeNull()
    // The WebSocket snapshot is the retry path: the next frame heals the miss.
    act(() => {
      FakeWebSocket.instances[0].emit({ type: 'state', state: state() })
    })
    expect(result.current.state?.fen).toBe(START_FEN)
  })

  it('loads the board even when the settings fetch never reaches the server', async () => {
    routeAnswers('/api/settings', REFUSED_CONNECTION)
    const { result } = renderHook(() => useGame())
    await waitFor(() => expect(result.current.state).not.toBeNull())
    // No settings arrived, so all three stay null — the hook's word for
    // "not loaded", which the UI already renders as unknown rather than
    // claiming a mode or a strength the server never confirmed.
    expect(result.current.voiceOutput).toBeNull()
    expect(result.current.tier).toBeNull()
    expect(result.current.agentAvailable).toBeNull()
  })

  it('resolves an undo that never reached the server and leaves the board untouched', async () => {
    const { result } = renderHook(() => useGame())
    await waitFor(() => expect(result.current.state).not.toBeNull())
    const revisionBefore = result.current.revision
    routeAnswers('/api/game/undo', REFUSED_CONNECTION)
    await act(async () => {
      // BottomBar discards this promise; a rejection would escape unhandled.
      await expect(result.current.undo()).resolves.toBeUndefined()
    })
    expect(result.current.state?.fen).toBe(START_FEN)
    expect(result.current.revision).toBe(revisionBefore)
  })

  it('resolves gated lifecycle calls that never reached the server without asking anything', async () => {
    const confirmSpy = vi.fn(() => true)
    vi.stubGlobal('confirm', confirmSpy)
    const { result } = renderHook(() => useGame())
    await waitFor(() => expect(result.current.state).not.toBeNull())
    const revisionBefore = result.current.revision
    routeAnswers('/api/game/new', REFUSED_CONNECTION)
    routeAnswers('/api/game/resign', REFUSED_CONNECTION)
    routeAnswers('/api/game/claim-draw', REFUSED_CONNECTION)
    await act(async () => {
      await expect(result.current.newGame()).resolves.toBeUndefined()
      await expect(result.current.resign()).resolves.toBeUndefined()
      await expect(result.current.claimDraw()).resolves.toBeUndefined()
    })
    // No state and no question came back: nothing to ask, nothing to move.
    expect(confirmSpy).not.toHaveBeenCalled()
    expect(result.current.revision).toBe(revisionBefore)
  })

  /** Arm the gate on /api/game/new (its 409 question), so the test can aim a
   * failure at the /api/game/confirm answer that follows. */
  function gateAsksFirst() {
    routeAnswers('/api/game/new', () =>
      Promise.resolve({
        ok: false,
        status: 409,
        json: () =>
          Promise.resolve({ detail: 'Start a new one?', confirm: true, op: 'new_game' }),
      }),
    )
  }

  it('resolves a confirmation answer that never reached the server and leaves the board untouched', async () => {
    vi.stubGlobal('confirm', vi.fn(() => true))
    const { result } = renderHook(() => useGame())
    await waitFor(() => expect(result.current.state).not.toBeNull())
    act(() => {
      FakeWebSocket.instances[0].emit({ type: 'state', state: reviewableState() })
    })
    const revisionBefore = result.current.revision
    // The gate asked, the player said yes, and the answer died on the way — a
    // redeploy mid-dialog. Nothing may move on a state the server never sent.
    gateAsksFirst()
    routeAnswers('/api/game/confirm', REFUSED_CONNECTION)
    await act(async () => {
      await expect(result.current.newGame()).resolves.toBeUndefined()
    })
    expect(result.current.state?.history).toEqual(['e4', 'e5'])
    expect(result.current.revision).toBe(revisionBefore)
  })

  it('drops a confirmed answer whose 200 carries no state rather than applying nothing', async () => {
    vi.stubGlobal('confirm', vi.fn(() => true))
    const { result } = renderHook(() => useGame())
    await waitFor(() => expect(result.current.state).not.toBeNull())
    act(() => {
      FakeWebSocket.instances[0].emit({ type: 'state', state: reviewableState() })
    })
    const revisionBefore = result.current.revision
    gateAsksFirst()
    // A truncated body still parses as OK at the HTTP layer; `apply` may only
    // ever be handed a real GameState, never `undefined` off a cast.
    routeAnswers('/api/game/confirm', TRUNCATED_OK_BODY)
    await act(async () => {
      await expect(result.current.newGame()).resolves.toBeUndefined()
    })
    expect(result.current.state?.history).toEqual(['e4', 'e5'])
    expect(result.current.revision).toBe(revisionBefore)
  })

  it('keeps the confirmed tier when a difficulty change fails in transit', async () => {
    const { result } = renderHook(() => useGame())
    await waitFor(() => expect(result.current.tier).toBe('casual'))
    routeAnswers('/api/game/difficulty', REFUSED_CONNECTION)
    await act(async () => {
      await expect(result.current.setDifficulty('advanced')).resolves.toBeUndefined()
    })
    expect(result.current.tier).toBe('casual')
    // A 200 whose body never parsed confirmed nothing either.
    routeAnswers('/api/game/difficulty', TRUNCATED_OK_BODY)
    await act(async () => {
      await expect(result.current.setDifficulty('advanced')).resolves.toBeUndefined()
    })
    expect(result.current.tier).toBe('casual')
  })

  it('keeps the confirmed voice setting when a toggle fails in transit', async () => {
    const { result } = renderHook(() => useGame())
    await waitFor(() => expect(result.current.voiceOutput).toBe(false))
    routeAnswers('/api/settings/voice', REFUSED_CONNECTION)
    await act(async () => {
      await expect(result.current.setVoiceOutput(true)).resolves.toBeUndefined()
    })
    expect(result.current.voiceOutput).toBe(false)
    routeAnswers('/api/settings/voice', TRUNCATED_OK_BODY)
    await act(async () => {
      await expect(result.current.setVoiceOutput(true)).resolves.toBeUndefined()
    })
    expect(result.current.voiceOutput).toBe(false)
  })

  it('leaves the hint empty when the request fails in transit', async () => {
    const { result } = renderHook(() => useGame())
    await waitFor(() => expect(result.current.state).not.toBeNull())
    routeAnswers('/api/game/hint', REFUSED_CONNECTION)
    await act(async () => {
      await expect(result.current.requestHint()).resolves.toBeUndefined()
    })
    expect(result.current.hintShapes).toEqual([])
    routeAnswers('/api/game/hint', TRUNCATED_OK_BODY)
    await act(async () => {
      await expect(result.current.requestHint()).resolves.toBeUndefined()
    })
    expect(result.current.hintShapes).toEqual([])
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
    expect(result.current.hintShapes).toEqual([{ orig: 'e2', dest: 'e4', brush: 'hint' }])
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

  /** Hold the hint response open, so the board can move while it is in flight.
   * `version` is the board the held answer is about; returns the release handle. */
  function deferHint(version: number) {
    let release: () => void = () => {}
    const held = new Promise<void>((resolve) => {
      release = resolve
    })
    fetchMock.mockImplementation((url: string) => {
      if (String(url).includes('/api/game/hint'))
        return held.then(() => ({
          ok: true,
          json: () =>
            Promise.resolve({ uci: 'e2e4', san: 'e4', from: 'e2', to: 'e4', version }),
        }))
      return jsonResponse(state())
    })
    return release
  }

  it('discards a hint that resolves after a newer board has been applied', async () => {
    const { result } = renderHook(() => useGame())
    await waitFor(() => expect(result.current.state?.version).toBe(1))
    const release = deferHint(1)

    let pending: Promise<void> | undefined
    act(() => {
      pending = result.current.requestHint()
    })
    // The board moves on while the engine is still searching — another client,
    // or this one dragging a piece. `apply` clears the arrow, but the answer
    // already in flight is still an answer about version 1.
    act(() => {
      FakeWebSocket.instances[0].emit({
        type: 'state',
        state: state({ version: 2, fen: AFTER_E4_FEN, turn: 'black' }),
      })
    })
    await act(async () => {
      release()
      await pending
    })

    // Painting it here would show the old position's recommendation on the new
    // board, where it may not even be legal (#218).
    expect(result.current.hintShapes).toEqual([])
  })

  it('discards a hint that resolves after the player enters history review', async () => {
    const { result } = await renderReviewable()
    const release = deferHint(1)

    let pending: Promise<void> | undefined
    act(() => {
      pending = result.current.requestHint()
    })
    // Review moves the view without moving the board, so the version still
    // matches — the arrow would land on a position the player is only browsing.
    act(() => result.current.stepBack())
    await act(async () => {
      release()
      await pending
    })

    expect(result.current.reviewing).toBe(true)
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
