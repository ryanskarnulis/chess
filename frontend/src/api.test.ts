import { afterEach, describe, expect, it, vi } from 'vitest'
import { fetchPgn, fetchReview, fetchSettings, fetchState } from './api'

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
