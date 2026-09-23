import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type { GameState } from './api'
import {
  CEILING_MS,
  beginInteraction,
  correlate,
  finishInteraction,
  mark,
  observeBoard,
  resetInteractions,
  unwatchBoard,
  watchBoard,
} from './voiceTiming'

function board(version: number, turn: 'white' | 'black'): GameState {
  return {
    version,
    fen: 'x',
    turn,
    player_color: 'white',
    game_over: false,
    outcome: null,
    history: [],
    fens: [],
    captured: { white: [], black: [] },
    legal_moves: [],
    dests: {},
  }
}

let now = 0
let fetchMock: ReturnType<typeof vi.fn>

function reported(): Record<string, unknown>[] {
  return fetchMock.mock.calls
    .filter(([url]) => url === '/api/telemetry/voice')
    .map(([, init]) => JSON.parse(String((init as RequestInit).body)))
}

beforeEach(() => {
  now = 1_000
  vi.spyOn(performance, 'now').mockImplementation(() => now)
  fetchMock = vi.fn(async () => new Response(null, { status: 204 }))
  vi.stubGlobal('fetch', fetchMock)
})

afterEach(() => {
  resetInteractions()
  vi.useRealTimers()
  vi.restoreAllMocks()
  vi.unstubAllGlobals()
})

describe('voiceTiming', () => {
  it('reports each milestone as an offset from the start, in its own clock', () => {
    const id = beginInteraction('voice')
    now = 2_400
    mark(id, 'stt_done')
    now = 2_500
    mark(id, 'command_sent')
    correlate(id, 'abc123def456')
    now = 9_000
    finishInteraction(id, 'ended')

    const [report] = reported()
    expect(report).toEqual({
      interaction_id: id,
      correlation_id: 'abc123def456',
      origin: 'voice',
      clock: 'client_monotonic_ms',
      start: 'speech_end',
      marks: { stt_done: 1400, command_sent: 1500 },
      outcome: 'ended',
      censored: false,
    })
    expect(fetchMock.mock.calls[0][1]).toMatchObject({ method: 'POST', keepalive: true })
  })

  it('keeps the first reading of a milestone', () => {
    const id = beginInteraction('typed')
    now = 1_100
    mark(id, 'command_done')
    now = 5_000
    mark(id, 'command_done')
    finishInteraction(id, 'silent')
    expect(reported()[0]).toMatchObject({ start: 'submit', marks: { command_done: 100 } })
  })

  it('sends one report per interaction, and ignores unknown ids', () => {
    const id = beginInteraction('typed')
    finishInteraction(id, 'silent')
    finishInteraction(id, 'ended')
    finishInteraction(undefined, 'ended')
    mark('not-an-id', 'stt_done')
    expect(reported()).toHaveLength(1)
  })

  it('marks the first newer board, then the reply once the player moved', () => {
    const id = beginInteraction('voice')
    watchBoard(id, board(4, 'white'))
    now = 1_050
    observeBoard(board(4, 'white')) // a duplicate frame of the old board
    now = 1_200
    observeBoard(board(5, 'black')) // the player's move landed
    now = 1_900
    observeBoard(board(6, 'white')) // the engine answered
    finishInteraction(id, 'silent')

    expect(reported()[0].marks).toEqual({ first_board_update: 200, engine_reply: 900 })
  })

  it('does not call a board that never left the player to move an engine reply', () => {
    const id = beginInteraction('typed')
    watchBoard(id, board(4, 'white'))
    observeBoard(board(6, 'white')) // an undo pair: the player is to move again
    finishInteraction(id, 'silent')
    expect(reported()[0].marks).toEqual({ first_board_update: 0 })
  })

  it('stops attributing boards once the command is done', () => {
    const id = beginInteraction('typed')
    watchBoard(id, board(4, 'white'))
    unwatchBoard(id)
    observeBoard(board(5, 'black'))
    finishInteraction(id, 'silent')
    expect(reported()[0].marks).toEqual({})
  })

  it('gives up on an interaction at the ceiling and says it was censored', () => {
    vi.useFakeTimers()
    const id = beginInteraction('voice')
    mark(id, 'stt_done')
    vi.advanceTimersByTime(CEILING_MS)
    expect(reported()[0]).toMatchObject({ interaction_id: id, outcome: 'timeout', censored: true })
  })

  it('never throws into the game when the report cannot be sent', () => {
    fetchMock.mockImplementation(() => {
      throw new Error('offline')
    })
    const id = beginInteraction('typed')
    expect(() => finishInteraction(id, 'silent')).not.toThrow()
  })
})
