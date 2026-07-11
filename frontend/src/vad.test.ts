import { beforeEach, describe, expect, it, vi } from 'vitest'
import { createVad } from './vad'
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
})
