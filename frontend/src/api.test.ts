import { afterEach, describe, expect, it, vi } from 'vitest'
import { fetchPgn, fetchReview, fetchSettings, fetchState, sendCommand, setBrain } from './api'

// Direct coverage for the client's never-rejects contract on the helpers whose
// callers can't exercise the fetch layer: ReviewPanel and PostGameModal mock
// this module wholesale, and the state/settings loaders run in mount effects
// that discard their promise — so a rejection from any of them is an unhandled
// escape with no feedback at all. The failure modes are the ones #231/#232
// established for moves, commands and transcription: a refused connection
// during a container redeploy, a gateway error page, a truncated body.
// The other helpers' contracts are pinned at the caller in useGame.test.tsx,
// where the null-handling they feed is also under test.

afterEach(() => {
  vi.useRealTimers()
  vi.unstubAllGlobals()
})

function fetchAnswers(answer: () => Promise<unknown>) {
  vi.stubGlobal('fetch', vi.fn(answer))
}

const REFUSED_CONNECTION = () => Promise.reject(new TypeError('Failed to fetch'))
const GATEWAY_ERROR_PAGE = () =>
  Promise.resolve({
    ok: false,
    status: 502,
    json: () => Promise.reject(new SyntaxError('Unexpected token <')),
  })
const TRUNCATED_OK_BODY = () =>
  Promise.resolve({
    ok: true,
    status: 200,
    json: () => Promise.reject(new SyntaxError('Unexpected end of JSON input')),
  })

describe('api helpers resolve to their failure shape instead of rejecting', () => {
  const helpers: [string, () => Promise<unknown>][] = [
    ['fetchState', fetchState],
    ['fetchSettings', fetchSettings],
    ['fetchReview', fetchReview],
    ['fetchPgn', fetchPgn],
    ['setBrain', () => setBrain('gemma-4-12b')],
  ]

  for (const [name, call] of helpers) {
    it(`${name} resolves null when the request never reaches the server`, async () => {
      fetchAnswers(REFUSED_CONNECTION)
      await expect(call()).resolves.toBeNull()
    })

    it(`${name} resolves null on a gateway error whose body is not JSON`, async () => {
      fetchAnswers(GATEWAY_ERROR_PAGE)
      await expect(call()).resolves.toBeNull()
    })

    it(`${name} resolves null on a 200 whose body never parses`, async () => {
      fetchAnswers(TRUNCATED_OK_BODY)
      await expect(call()).resolves.toBeNull()
    })
  }

  it('fetchPgn resolves null when the body carries no pgn string', async () => {
    // "No transcript" beats handing the clipboard the string "undefined".
    fetchAnswers(() =>
      Promise.resolve({ ok: true, status: 200, json: () => Promise.resolve({}) }),
    )
    await expect(fetchPgn()).resolves.toBeNull()
  })
})

describe('setBrain', () => {
  it('posts the model and returns the brain the server confirmed', async () => {
    const fetchMock = vi.fn(() =>
      Promise.resolve({
        ok: true,
        status: 200,
        json: () =>
          Promise.resolve({
            brain: 'gemma-4-12b',
            brain_serving: 'gemma-4-26b-a4b',
            brain_cold: true,
            brain_choices: ['gemma-4-12b', 'gemma-4-26b-a4b'],
          }),
      }),
    )
    vi.stubGlobal('fetch', fetchMock)
    await expect(setBrain('gemma-4-12b')).resolves.toEqual({
      brain: 'gemma-4-12b',
      cold: true,
      choices: ['gemma-4-12b', 'gemma-4-26b-a4b'],
    })
    expect(fetchMock).toHaveBeenCalledWith(
      '/api/settings/brain',
      expect.objectContaining({ method: 'POST', body: JSON.stringify({ model: 'gemma-4-12b' }) }),
    )
  })

  for (const status of [409, 422]) {
    it(`resolves null when the server refuses (${status})`, async () => {
      fetchAnswers(() =>
        Promise.resolve({ ok: false, status, json: () => Promise.resolve({ detail: 'no' }) }),
      )
      await expect(setBrain('gemma-4-12b')).resolves.toBeNull()
    })
  }
})

// A fetch that answers only when told, and fails the way the browser's does
// when the request is aborted.
function hangingFetch() {
  return vi.fn(
    (_url: string, init?: { signal?: AbortSignal }) =>
      new Promise((_resolve, reject) => {
        init?.signal?.addEventListener('abort', () =>
          reject(new DOMException('The operation was aborted.', 'AbortError')),
        )
      }),
  )
}

describe('sendCommand (#546)', () => {
  const sentBody = (fetchMock: ReturnType<typeof vi.fn>, call = 0) =>
    JSON.parse(String(fetchMock.mock.calls[call][1].body))

  it('sends no origin unless one is named, so an ordinary command stays the panel’s', async () => {
    const fetchMock = vi.fn((_url: string, _init: RequestInit) =>
      Promise.resolve({ ok: true, status: 200, json: () => Promise.resolve({ commentary: '' }) }),
    )
    vi.stubGlobal('fetch', fetchMock)
    await sendCommand('play e4', 3, 'abc123abc123')
    expect(sentBody(fetchMock)).toEqual({ text: 'play e4', interaction_id: 'abc123abc123', version: 3 })
    await sendCommand('let us play', undefined, undefined, { origin: 'handoff' })
    expect(sentBody(fetchMock, 1)).toEqual({
      text: 'let us play',
      origin: 'handoff',
    })
  })

  it('gives up with null when the turn outlasts the timeout, and aborts the request', async () => {
    vi.useFakeTimers()
    const fetchMock = hangingFetch()
    vi.stubGlobal('fetch', fetchMock)
    const pending = sendCommand('hi', undefined, undefined, { timeoutMs: 30_000 })
    await vi.advanceTimersByTimeAsync(29_999)
    expect(fetchMock.mock.calls[0][1]?.signal?.aborted).toBe(false)
    await vi.advanceTimersByTimeAsync(1)
    await expect(pending).resolves.toBeNull()
    expect(fetchMock.mock.calls[0][1]?.signal?.aborted).toBe(true)
  })

  it('counts a reply that stalls while its body is read as a miss, not a success', async () => {
    vi.useFakeTimers()
    vi.stubGlobal(
      'fetch',
      vi.fn((_url: string, init: { signal: AbortSignal }) =>
        Promise.resolve({
          ok: true,
          status: 200,
          // Headers arrived; the body never does, and the abort ends it.
          json: () =>
            new Promise((_resolve, reject) =>
              init.signal.addEventListener('abort', () => reject(new Error('aborted'))),
            ),
        }),
      ),
    )
    const pending = sendCommand('hi', undefined, undefined, { timeoutMs: 30_000 })
    await vi.advanceTimersByTimeAsync(30_000)
    await expect(pending).resolves.toBeNull()
  })

  it('does not abort a turn that answers in time, and sets no timer without a timeout', async () => {
    vi.useFakeTimers()
    const answer = { commentary: 'ok', tool_results: [], state: {}, speak: false }
    vi.stubGlobal(
      'fetch',
      vi.fn(() => Promise.resolve({ ok: true, status: 200, json: () => Promise.resolve(answer) })),
    )
    await expect(
      sendCommand('hi', undefined, undefined, { timeoutMs: 30_000 }),
    ).resolves.toEqual(answer)
    await expect(sendCommand('hi')).resolves.toEqual(answer)
    expect(vi.getTimerCount()).toBe(0)
  })

  it('never times out a command that named no ceiling', async () => {
    vi.useFakeTimers()
    const fetchMock = hangingFetch()
    vi.stubGlobal('fetch', fetchMock)
    void sendCommand('hi')
    await vi.advanceTimersByTimeAsync(10 * 60_000)
    expect(fetchMock.mock.calls[0][1]?.signal?.aborted).toBe(false)
  })
})
