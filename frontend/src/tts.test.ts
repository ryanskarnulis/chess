import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

// jsdom implements neither URL.createObjectURL nor Audio playback; these
// stubs let the tests assert the fetch → blob → play flow.
//
// tts.ts keeps module-level state (the shared, gesture-unlocked <audio>
// element), so every test re-imports a fresh copy via loadTts().

const play = vi.fn()
let audioInstances: FakeAudio[]

class FakeAudio {
  src = ''
  onended: (() => void) | null = null
  onerror: (() => void) | null = null

  constructor(src?: string) {
    if (src !== undefined) this.src = src
    audioInstances.push(this)
  }

  play = play
}

async function loadTts() {
  vi.resetModules()
  return import('./tts')
}

/** Let the fetch → blob → play microtask chain run to completion. */
async function playbackStarted() {
  await vi.waitFor(() => expect(play).toHaveBeenCalled())
}

beforeEach(() => {
  audioInstances = []
  play.mockReset().mockResolvedValue(undefined)
  vi.stubGlobal('Audio', FakeAudio)
  vi.stubGlobal('URL', {
    createObjectURL: vi.fn(() => 'blob:fake-url'),
    revokeObjectURL: vi.fn(),
  })
})

afterEach(() => {
  vi.unstubAllGlobals()
  vi.useRealTimers()
})

describe('playText', () => {
  it('fetches the audio and plays it', async () => {
    const { playText } = await loadTts()
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => new Response(new Blob([new Uint8Array([1, 2])]), { status: 200 })),
    )
    const done = playText('Check!')
    await playbackStarted()
    const [url, init] = (fetch as ReturnType<typeof vi.fn>).mock.calls[0]
    expect(url).toBe('/api/voice/speak')
    expect(JSON.parse(init.body)).toEqual({ text: 'Check!' })
    expect(audioInstances).toHaveLength(1)
    expect(audioInstances[0].src).toBe('blob:fake-url')
    audioInstances[0].onended?.()
    await done
  })

  it('resolves only once playback has finished (hands-free needs this)', async () => {
    // The conversation loop reopens the mic when the spoken reply ends —
    // resolving at playback *start* would make the agent hear itself.
    const { playText } = await loadTts()
    vi.stubGlobal('fetch', vi.fn(async () => new Response(new Blob(), { status: 200 })))
    let finished = false
    const done = playText('Check!').then(() => {
      finished = true
    })
    await playbackStarted()
    expect(finished).toBe(false)
    audioInstances[0].onended?.()
    await done
    expect(finished).toBe(true)
  })

  it('resolves when the element reports a playback error', async () => {
    const { playText } = await loadTts()
    vi.stubGlobal('fetch', vi.fn(async () => new Response(new Blob(), { status: 200 })))
    const done = playText('Check!')
    await playbackStarted()
    audioInstances[0].onerror?.()
    await done
    expect(URL.revokeObjectURL).toHaveBeenCalledWith('blob:fake-url')
  })

  it('reuses one shared audio element across plays (mobile autoplay unlock)', async () => {
    // iOS/Android only allow programmatic playback on an element that has
    // already played inside a user gesture — a fresh Audio per clip would
    // never be unlocked, so every clip must go through the same element.
    const { playText } = await loadTts()
    vi.stubGlobal('fetch', vi.fn(async () => new Response(new Blob(), { status: 200 })))
    const first = playText('Check!')
    await playbackStarted()
    audioInstances[0].onended?.()
    await first
    const second = playText('Mate!')
    await vi.waitFor(() => expect(play).toHaveBeenCalledTimes(2))
    audioInstances[0].onended?.()
    await second
    expect(audioInstances).toHaveLength(1)
  })

  it('releases the object URL once playback ends', async () => {
    const { playText } = await loadTts()
    vi.stubGlobal('fetch', vi.fn(async () => new Response(new Blob(), { status: 200 })))
    const done = playText('Check!')
    await playbackStarted()
    audioInstances[0].onended?.()
    await done
    expect(URL.revokeObjectURL).toHaveBeenCalledWith('blob:fake-url')
  })

  it('releases the previous clip and settles its promise when a new one interrupts', async () => {
    const { playText } = await loadTts()
    vi.stubGlobal('fetch', vi.fn(async () => new Response(new Blob(), { status: 200 })))
    const urls = ['blob:first', 'blob:second']
    ;(URL.createObjectURL as ReturnType<typeof vi.fn>).mockImplementation(() => urls.shift())
    const first = playText('Check!')
    await playbackStarted()
    const second = playText('Mate!')
    await vi.waitFor(() => expect(play).toHaveBeenCalledTimes(2))
    // The interrupted clip's promise must not hang forever — its onended
    // will never fire now that the element has moved on.
    await first
    expect(URL.revokeObjectURL).toHaveBeenCalledWith('blob:first')
    // The interrupted clip's stale onended must not revoke the live one.
    expect(URL.revokeObjectURL).not.toHaveBeenCalledWith('blob:second')
    audioInstances[0].onended?.()
    await second
    expect(URL.revokeObjectURL).toHaveBeenCalledWith('blob:second')
  })

  it('does nothing when voice is unavailable (non-ok response)', async () => {
    const { playText } = await loadTts()
    vi.stubGlobal('fetch', vi.fn(async () => new Response('nope', { status: 503 })))
    await playText('Check!')
    expect(audioInstances).toHaveLength(0)
  })

  it('resolves rather than rejecting when the request fails outright', async () => {
    // playText documents that it never rejects. A transport failure (offline,
    // DNS, aborted connection) must settle exactly like a 503 does.
    const { playText } = await loadTts()
    vi.stubGlobal('fetch', vi.fn(() => Promise.reject(new TypeError('Failed to fetch'))))
    await expect(playText('Check!')).resolves.toBeUndefined()
    expect(audioInstances).toHaveLength(0)
    expect(URL.createObjectURL).not.toHaveBeenCalled()
  })

  it('resolves when the response body cannot be read', async () => {
    const { playText } = await loadTts()
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => ({
        ok: true,
        blob: () => Promise.reject(new Error('body stream errored')),
      })),
    )
    await expect(playText('Check!')).resolves.toBeUndefined()
    expect(audioInstances).toHaveLength(0)
  })

  it('does not disturb the live clip when a later request fails', async () => {
    const { playText } = await loadTts()
    vi.stubGlobal('fetch', vi.fn(async () => new Response(new Blob(), { status: 200 })))
    const first = playText('Check!')
    await playbackStarted()
    vi.stubGlobal('fetch', vi.fn(() => Promise.reject(new TypeError('Failed to fetch'))))
    await playText('Mate!')
    // The failed request loaded nothing, so it must neither interrupt the
    // playing clip nor revoke the URL still attached to the element.
    expect(URL.revokeObjectURL).not.toHaveBeenCalledWith('blob:fake-url')
    audioInstances[0].onended?.()
    await first
    expect(URL.revokeObjectURL).toHaveBeenCalledWith('blob:fake-url')
  })

  it('gives up on a request that never answers, best-effort like a 503', async () => {
    // A backend or proxy that accepts the request and then never settles it
    // rejects nothing, so the catch above never runs: without a client
    // deadline the promise hangs, currentPlayback never settles and the
    // hands-free loop awaiting audioIdle() never reopens the mic.
    vi.useFakeTimers()
    const { playText, audioIdle } = await loadTts()
    const signals: (AbortSignal | undefined)[] = []
    vi.stubGlobal(
      'fetch',
      vi.fn((_url: string, init?: RequestInit) => {
        signals.push(init?.signal ?? undefined)
        return new Promise<Response>(() => {})
      }),
    )
    let settled = false
    const done = playText('Check!').then(() => {
      settled = true
    })
    await vi.advanceTimersByTimeAsync(60_000)
    expect(settled).toBe(false)
    await vi.advanceTimersByTimeAsync(30_000)
    await expect(done).resolves.toBeUndefined()
    expect(settled).toBe(true)
    // Nothing was played, exactly as when voice answers 503.
    expect(audioInstances).toHaveLength(0)
    expect(URL.createObjectURL).not.toHaveBeenCalled()
    // And the wait the hands-free loop does is released too.
    await expect(audioIdle()).resolves.toBeUndefined()
    // The request itself is cancelled, not just abandoned.
    expect(signals[0]?.aborted).toBe(true)
  })

  it('clears the deadline once a clip has been fetched', async () => {
    // A completed request must not leave a timer armed to abort nothing.
    vi.useFakeTimers()
    const { playText } = await loadTts()
    vi.stubGlobal('fetch', vi.fn(async () => new Response(new Blob(), { status: 200 })))
    const done = playText('Check!')
    await vi.advanceTimersByTimeAsync(0)
    expect(play).toHaveBeenCalled()
    expect(vi.getTimerCount()).toBe(0)
    audioInstances[0].onended?.()
    await done
  })

  it('survives a blocked autoplay without leaking the URL', async () => {
    const { playText } = await loadTts()
    vi.stubGlobal('fetch', vi.fn(async () => new Response(new Blob(), { status: 200 })))
    play.mockRejectedValue(new Error('NotAllowedError'))
    await playText('Check!')
    expect(URL.revokeObjectURL).toHaveBeenCalledWith('blob:fake-url')
  })
})

describe('audioIdle', () => {
  it('resolves immediately when nothing was ever played', async () => {
    const { audioIdle } = await loadTts()
    await expect(audioIdle()).resolves.toBeUndefined()
  })

  it('resolves after a failed request instead of rejecting', async () => {
    // A rejected currentPlayback would poison audioIdle(), leaving hands-free
    // voice stuck with the VAD paused and the mic never reopened.
    const { playText, audioIdle } = await loadTts()
    vi.stubGlobal('fetch', vi.fn(() => Promise.reject(new TypeError('Failed to fetch'))))
    await playText('Check!')
    await expect(audioIdle()).resolves.toBeUndefined()
  })

  it('waits for the in-flight clip to finish', async () => {
    const { playText, audioIdle } = await loadTts()
    vi.stubGlobal('fetch', vi.fn(async () => new Response(new Blob(), { status: 200 })))
    const done = playText('Check!')
    let idle = false
    const wait = audioIdle().then(() => {
      idle = true
    })
    await playbackStarted()
    expect(idle).toBe(false)
    audioInstances[0].onended?.()
    await done
    await wait
    expect(idle).toBe(true)
  })
})

describe('unlockAudio', () => {
  it('synchronously plays a silent clip on the shared element', async () => {
    const { unlockAudio } = await loadTts()
    unlockAudio()
    expect(audioInstances).toHaveLength(1)
    expect(audioInstances[0].src).toMatch(/^data:audio\/wav/)
    expect(play).toHaveBeenCalledTimes(1)
  })

  it('primes the same element playText uses', async () => {
    const { unlockAudio, playText } = await loadTts()
    unlockAudio()
    vi.stubGlobal('fetch', vi.fn(async () => new Response(new Blob(), { status: 200 })))
    const done = playText('Check!')
    await vi.waitFor(() => expect(play).toHaveBeenCalledTimes(2))
    audioInstances[0].onended?.()
    await done
    expect(audioInstances).toHaveLength(1)
    expect(audioInstances[0].src).toBe('blob:fake-url')
  })

  it('only primes once after a successful unlock', async () => {
    const { unlockAudio } = await loadTts()
    unlockAudio()
    // Let the successful play() settle so the unlocked flag latches.
    await Promise.resolve()
    unlockAudio()
    expect(play).toHaveBeenCalledTimes(1)
  })

  it('retries after a failed unlock attempt', async () => {
    const { unlockAudio } = await loadTts()
    play.mockRejectedValueOnce(new Error('NotAllowedError'))
    unlockAudio()
    await Promise.resolve()
    unlockAudio()
    expect(play).toHaveBeenCalledTimes(2)
  })
})
