import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { playText } from './tts'

// jsdom implements neither URL.createObjectURL nor Audio playback; these
// stubs let the tests assert the fetch → blob → play flow.

const play = vi.fn()
let audioInstances: { src: string; onended: (() => void) | null }[]

class FakeAudio {
  src: string
  onended: (() => void) | null = null

  constructor(src: string) {
    this.src = src
    audioInstances.push(this)
  }

  play = play
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

  it('releases the object URL once playback ends', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => new Response(new Blob(), { status: 200 })))
    await playText('Check!')
    audioInstances[0].onended?.()
    expect(URL.revokeObjectURL).toHaveBeenCalledWith('blob:fake-url')
  })

  it('does nothing when voice is unavailable (non-ok response)', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => new Response('nope', { status: 503 })))
    await playText('Check!')
    expect(audioInstances).toHaveLength(0)
  })

  it('survives a blocked autoplay without leaking the URL', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => new Response(new Blob(), { status: 200 })))
    play.mockRejectedValue(new Error('NotAllowedError'))
    await playText('Check!')
    expect(URL.revokeObjectURL).toHaveBeenCalledWith('blob:fake-url')
  })
})
