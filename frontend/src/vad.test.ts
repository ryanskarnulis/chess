import { beforeEach, describe, expect, it, vi } from 'vitest'
import { createVad, staleChunkReload } from './vad'
import { MicVAD } from '@ricky0123/vad-web'

// The real library spins up an AudioWorklet + WASM Silero model — none of
// which exists in jsdom. These tests pin down the wrapper's contract: local
// asset paths (local-first app, no CDN), callback wiring, and a null return
// on any failure so the caller can fall back to push-to-talk.

vi.mock('@ricky0123/vad-web', () => ({
  MicVAD: { new: vi.fn() },
}))

const micVadNew = vi.mocked(MicVAD.new)

const fakeVad = {
  start: vi.fn(async () => {}),
  pause: vi.fn(async () => {}),
  destroy: vi.fn(async () => {}),
}

beforeEach(() => {
  vi.clearAllMocks()
  micVadNew.mockResolvedValue(fakeVad as unknown as MicVAD)
})

describe('createVad', () => {
  it('configures the mic VAD with self-hosted assets and the speech callbacks', async () => {
    const onSpeechStart = vi.fn()
    const onSpeechEnd = vi.fn()
    const vad = await createVad({ onSpeechStart, onSpeechEnd })

    expect(vad).not.toBeNull()
    const options = micVadNew.mock.calls[0][0]!
    // Local-first: the model, worklet, and WASM must come from this app's
    // own origin, never a CDN.
    expect(options.baseAssetPath).toBe('/vad/')
    expect(options.onnxWASMBasePath).toBe('/vad/')
    expect(options.onSpeechStart).toBe(onSpeechStart)
    expect(options.onSpeechEnd).toBe(onSpeechEnd)
    // End-of-utterance silence window: long enough for a mid-command pause.
    expect(options.redemptionMs).toBeGreaterThanOrEqual(800)
  })

  it('exposes pause/resume/destroy on the underlying VAD', async () => {
    const vad = await createVad({ onSpeechStart: vi.fn(), onSpeechEnd: vi.fn() })
    vad!.pause()
    expect(fakeVad.pause).toHaveBeenCalled()
    vad!.resume()
    expect(fakeVad.start).toHaveBeenCalled()
    vad!.destroy()
    expect(fakeVad.destroy).toHaveBeenCalled()
  })

  it('returns null when the VAD fails to initialize (caller falls back)', async () => {
    micVadNew.mockRejectedValue(new Error('no AudioWorklet'))
    const vad = await createVad({ onSpeechStart: vi.fn(), onSpeechEnd: vi.fn() })
    expect(vad).toBeNull()
  })

  it('reports why the VAD is unavailable (phones have no console)', async () => {
    // Silent degradation made the iPhone failure undiagnosable (2026-07-11):
    // the mic quietly fell back to push-to-talk with no trace of the cause.
    micVadNew.mockRejectedValue(new Error('no AudioWorklet'))
    const onUnavailable = vi.fn()
    const vad = await createVad({
      onSpeechStart: vi.fn(),
      onSpeechEnd: vi.fn(),
      onUnavailable,
    })
    expect(vad).toBeNull()
    expect(onUnavailable).toHaveBeenCalledWith('no AudioWorklet')
  })
})

describe('staleChunkReload', () => {
  // A rebuilt image renames every hashed chunk; a tab still running the old
  // build 404s importing the lazy VAD chunk. That tab needs one reload to
  // fetch the fresh index.html — this is how the 2026-07-11 iPhone lost
  // continuous voice after a rebuild.
  const storage = () => {
    const map = new Map<string, string>()
    return {
      getItem: (k: string) => map.get(k) ?? null,
      setItem: (k: string, v: string) => void map.set(k, v),
      removeItem: (k: string) => void map.delete(k),
    } as Storage
  }

  it.each([
    'Failed to fetch dynamically imported module: https://x/assets/dist-abc.js',
    'Importing a module script failed.',
    'error loading dynamically imported module',
  ])('recognizes each browser wording: %s', (message) => {
    expect(staleChunkReload(new TypeError(message), storage())).toBe(true)
  })

  it('reloads only once, so a genuinely missing chunk cannot loop', () => {
    const s = storage()
    expect(staleChunkReload(new TypeError('Importing a module script failed.'), s)).toBe(true)
    expect(staleChunkReload(new TypeError('Importing a module script failed.'), s)).toBe(false)
  })

  it('never reloads for ordinary VAD failures', () => {
    expect(staleChunkReload(new Error('no AudioWorklet'), storage())).toBe(false)
  })
})
