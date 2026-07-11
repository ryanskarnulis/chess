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
})

describe('playText', () => {
  it('fetches the audio and plays it', async () => {
    const { playText } = await loadTts()
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => new Response(new Blob([new Uint8Array([1, 2])]), { status: 200 })),
    )
    await playText('Check!')
    const [url, init] = (fetch as ReturnType<typeof vi.fn>).mock.calls[0]
    expect(url).toBe('/api/voice/speak')
    expect(JSON.parse(init.body)).toEqual({ text: 'Check!' })
    expect(audioInstances).toHaveLength(1)
    expect(audioInstances[0].src).toBe('blob:fake-url')
    expect(play).toHaveBeenCalled()
  })

  it('reuses one shared audio element across plays (mobile autoplay unlock)', async () => {
    // iOS/Android only allow programmatic playback on an element that has
    // already played inside a user gesture — a fresh Audio per clip would
    // never be unlocked, so every clip must go through the same element.
    const { playText } = await loadTts()
    vi.stubGlobal('fetch', vi.fn(async () => new Response(new Blob(), { status: 200 })))
    await playText('Check!')
    await playText('Mate!')
    expect(audioInstances).toHaveLength(1)
    expect(play).toHaveBeenCalledTimes(2)
  })

  it('releases the object URL once playback ends', async () => {
    const { playText } = await loadTts()
    vi.stubGlobal('fetch', vi.fn(async () => new Response(new Blob(), { status: 200 })))
    await playText('Check!')
    audioInstances[0].onended?.()
    expect(URL.revokeObjectURL).toHaveBeenCalledWith('blob:fake-url')
  })

  it('releases the previous clip when a new one interrupts it', async () => {
    const { playText } = await loadTts()
    vi.stubGlobal('fetch', vi.fn(async () => new Response(new Blob(), { status: 200 })))
    const urls = ['blob:first', 'blob:second']
    ;(URL.createObjectURL as ReturnType<typeof vi.fn>).mockImplementation(() => urls.shift())
    await playText('Check!')
    await playText('Mate!')
    expect(URL.revokeObjectURL).toHaveBeenCalledWith('blob:first')
    // The interrupted clip's stale onended must not revoke the live one.
    expect(URL.revokeObjectURL).not.toHaveBeenCalledWith('blob:second')
  })

  it('does nothing when voice is unavailable (non-ok response)', async () => {
    const { playText } = await loadTts()
    vi.stubGlobal('fetch', vi.fn(async () => new Response('nope', { status: 503 })))
    await playText('Check!')
    expect(audioInstances).toHaveLength(0)
  })

  it('survives a blocked autoplay without leaking the URL', async () => {
    const { playText } = await loadTts()
    vi.stubGlobal('fetch', vi.fn(async () => new Response(new Blob(), { status: 200 })))
    play.mockRejectedValue(new Error('NotAllowedError'))
    await playText('Check!')
    expect(URL.revokeObjectURL).toHaveBeenCalledWith('blob:fake-url')
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
    await playText('Check!')
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
