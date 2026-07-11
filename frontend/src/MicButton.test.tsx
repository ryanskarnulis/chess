import { act, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { MicButton } from './MicButton'
import { audioIdle, unlockAudio } from './tts'
import { createVad, type SpeechEvents, type Vad } from './vad'

// Mobile browsers only allow playback primed inside a user gesture; the mic
// tap is the last gesture before the agent's spoken reply, so it must unlock.
vi.mock('./tts', () => ({ unlockAudio: vi.fn(), audioIdle: vi.fn(async () => {}) }))

// The real VAD needs an AudioWorklet + WASM model; the wrapper is mocked so
// tests can fire speech events by hand and simulate load failure.
vi.mock('./vad', () => ({ createVad: vi.fn() }))

// jsdom has no MediaRecorder/getUserMedia; these fakes stand in so the
// push-to-talk fallback tests drive record → stop → transcribe → submit
// without a browser.

class FakeMediaRecorder {
  static instances: FakeMediaRecorder[] = []
  mimeType = 'audio/webm'
  ondataavailable: ((e: { data: Blob }) => void) | null = null
  onstop: (() => void) | null = null
  started = false

  stream: unknown

  constructor(stream: unknown) {
    this.stream = stream
    FakeMediaRecorder.instances.push(this)
  }

  start() {
    this.started = true
  }

  stop() {
    this.ondataavailable?.({ data: new Blob(['opus'], { type: 'audio/webm' }) })
    this.onstop?.()
  }
}

const fakeTrack = { stop: vi.fn() }
const fakeStream = { getTracks: () => [fakeTrack] }
const getUserMedia = vi.fn()

function stubMediaSupport() {
  vi.stubGlobal('MediaRecorder', FakeMediaRecorder)
  Object.defineProperty(navigator, 'mediaDevices', {
    configurable: true,
    value: { getUserMedia },
  })
}

function stubTranscribeResponse(response: { ok: boolean; text?: string }) {
  vi.stubGlobal(
    'fetch',
    vi.fn(async () =>
      response.ok
        ? new Response(JSON.stringify({ text: response.text }), { status: 200 })
        : new Response('{"detail":"no speech service"}', { status: 503 }),
    ),
  )
}

const fakeVad: Vad = { pause: vi.fn(), resume: vi.fn(), destroy: vi.fn() }
let speechEvents: SpeechEvents | null = null

/** Tap the mic and wait for hands-free listening to engage. */
async function enterConversation() {
  fireEvent.click(screen.getByRole('button', { name: /voice conversation/i }))
  await screen.findByRole('button', { name: /stop listening/i })
}

/** Simulate the VAD reporting a finished utterance. */
function speak(samples = [0.1, -0.1]) {
  act(() => speechEvents?.onSpeechEnd(new Float32Array(samples)))
}

beforeEach(() => {
  FakeMediaRecorder.instances = []
  getUserMedia.mockReset()
  getUserMedia.mockResolvedValue(fakeStream)
  fakeTrack.stop.mockReset()
  vi.mocked(unlockAudio).mockClear()
  vi.mocked(audioIdle).mockReset()
  vi.mocked(audioIdle).mockResolvedValue(undefined)
  speechEvents = null
  vi.mocked(fakeVad.pause).mockClear()
  vi.mocked(fakeVad.resume).mockClear()
  vi.mocked(fakeVad.destroy).mockClear()
  vi.mocked(createVad).mockReset()
  vi.mocked(createVad).mockImplementation(async (events) => {
    speechEvents = events
    return fakeVad
  })
  stubMediaSupport()
})

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('MicButton (hands-free conversation mode)', () => {
  it('enters listening on a single tap and unlocks audio in the gesture', async () => {
    render(<MicButton onTranscript={vi.fn()} disabled={false} />)
    await enterConversation()
    expect(unlockAudio).toHaveBeenCalled()
    expect(createVad).toHaveBeenCalled()
  })

  it('transcribes an utterance as WAV and sends it down the command pipeline', async () => {
    stubTranscribeResponse({ ok: true, text: '  pawn to e4  ' })
    const onTranscript = vi.fn(async () => {})
    render(<MicButton onTranscript={onTranscript} disabled={false} />)
    await enterConversation()

    speak()
    // Half-duplex: the mic pauses the moment the utterance is captured.
    expect(fakeVad.pause).toHaveBeenCalled()
    await waitFor(() => expect(onTranscript).toHaveBeenCalledWith('pawn to e4'))

    const [url, init] = (fetch as ReturnType<typeof vi.fn>).mock.calls[0]
    expect(url).toBe('/api/voice/transcribe')
    const clip = (init.body as FormData).get('audio') as File
    expect(clip.name).toBe('clip.wav')
    expect(clip.type).toBe('audio/wav')

    // After the agent turn (and its spoken reply) the mic reopens.
    await waitFor(() => expect(fakeVad.resume).toHaveBeenCalled())
    expect(audioIdle).toHaveBeenCalled()
  })

  it('keeps the mic closed until the agent turn and spoken reply finish', async () => {
    stubTranscribeResponse({ ok: true, text: 'castle' })
    let finishAgent!: () => void
    const onTranscript = vi.fn(
      () =>
        new Promise<void>((resolve) => {
          finishAgent = resolve
        }),
    )
    let finishSpeech!: () => void
    vi.mocked(audioIdle).mockImplementation(
      () =>
        new Promise<void>((resolve) => {
          finishSpeech = resolve
        }),
    )
    render(<MicButton onTranscript={onTranscript} disabled={false} />)
    await enterConversation()

    speak()
    await waitFor(() => expect(onTranscript).toHaveBeenCalled())
    expect(fakeVad.resume).not.toHaveBeenCalled()

    await act(async () => finishAgent())
    // The reply may still be playing — listening now would hear the agent.
    expect(fakeVad.resume).not.toHaveBeenCalled()

    await act(async () => finishSpeech())
    await waitFor(() => expect(fakeVad.resume).toHaveBeenCalled())
  })

  it('ignores an empty transcript and keeps listening without nagging', async () => {
    stubTranscribeResponse({ ok: true, text: '   ' })
    const onTranscript = vi.fn()
    render(<MicButton onTranscript={onTranscript} disabled={false} />)
    await enterConversation()

    speak()
    await waitFor(() => expect(fakeVad.resume).toHaveBeenCalled())
    expect(onTranscript).not.toHaveBeenCalled()
    expect(screen.queryByRole('alert')).not.toBeInTheDocument()
  })

  it('exits conversation mode when transcription is unavailable', async () => {
    // Auto-resuming would hammer a dead speech service forever.
    stubTranscribeResponse({ ok: false })
    render(<MicButton onTranscript={vi.fn()} disabled={false} />)
    await enterConversation()

    speak()
    expect(await screen.findByRole('alert')).toHaveTextContent(/unavailable/i)
    expect(fakeVad.destroy).toHaveBeenCalled()
    expect(screen.getByRole('button', { name: /voice conversation/i })).toBeInTheDocument()
  })

  it('ends the conversation on a second tap', async () => {
    render(<MicButton onTranscript={vi.fn()} disabled={false} />)
    await enterConversation()

    fireEvent.click(screen.getByRole('button', { name: /stop listening/i }))
    expect(fakeVad.destroy).toHaveBeenCalled()
    expect(screen.getByRole('button', { name: /voice conversation/i })).toBeInTheDocument()
  })

  it('stays tappable (to exit) while the agent is busy', async () => {
    // CommandBox passes disabled=true while a command is in flight; that must
    // lock starting a new conversation, never escaping the current one.
    stubTranscribeResponse({ ok: true, text: 'castle' })
    const onTranscript = vi.fn(() => new Promise<void>(() => {}))
    const { rerender } = render(<MicButton onTranscript={onTranscript} disabled={false} />)
    await enterConversation()

    speak()
    await waitFor(() => expect(onTranscript).toHaveBeenCalled())
    rerender(<MicButton onTranscript={onTranscript} disabled={true} />)

    const button = screen.getByRole('button', { name: /stop listening/i })
    expect(button).toBeEnabled()
    fireEvent.click(button)
    expect(fakeVad.destroy).toHaveBeenCalled()
  })

  it('does not reopen the mic when the user exited during the agent turn', async () => {
    stubTranscribeResponse({ ok: true, text: 'castle' })
    let finishAgent!: () => void
    const onTranscript = vi.fn(
      () =>
        new Promise<void>((resolve) => {
          finishAgent = resolve
        }),
    )
    render(<MicButton onTranscript={onTranscript} disabled={false} />)
    await enterConversation()

    speak()
    await waitFor(() => expect(onTranscript).toHaveBeenCalled())
    fireEvent.click(screen.getByRole('button', { name: /stop listening/i }))
    await act(async () => finishAgent())

    expect(fakeVad.resume).not.toHaveBeenCalled()
    expect(screen.getByRole('button', { name: /voice conversation/i })).toBeInTheDocument()
  })

  it('destroys the VAD on unmount', async () => {
    const { unmount } = render(<MicButton onTranscript={vi.fn()} disabled={false} />)
    await enterConversation()
    unmount()
    expect(fakeVad.destroy).toHaveBeenCalled()
  })
})

describe('MicButton (push-to-talk fallback)', () => {
  beforeEach(() => {
    // No worklet/WASM (or mic denied inside the library) — wrapper reports null.
    vi.mocked(createVad).mockResolvedValue(null)
  })

  it('renders nothing when the browser has no recording support', () => {
    vi.unstubAllGlobals()
    // No MediaRecorder / mediaDevices stubs: plain jsdom.
    render(<MicButton onTranscript={vi.fn()} disabled={false} />)
    expect(screen.queryByRole('button')).not.toBeInTheDocument()
  })

  it('records, transcribes, and hands the text to onTranscript', async () => {
    stubTranscribeResponse({ ok: true, text: '  pawn to e4  ' })
    const onTranscript = vi.fn()
    render(<MicButton onTranscript={onTranscript} disabled={false} />)

    fireEvent.click(screen.getByRole('button', { name: /voice conversation/i }))
    await waitFor(() => expect(FakeMediaRecorder.instances).toHaveLength(1))
    expect(getUserMedia).toHaveBeenCalledWith({ audio: true })
    expect(FakeMediaRecorder.instances[0].started).toBe(true)

    fireEvent.click(screen.getByRole('button', { name: /stop recording/i }))
    await waitFor(() => expect(onTranscript).toHaveBeenCalledWith('pawn to e4'))
    // The mic is released once the clip is captured.
    expect(fakeTrack.stop).toHaveBeenCalled()
    const [url, init] = (fetch as ReturnType<typeof vi.fn>).mock.calls[0]
    expect(url).toBe('/api/voice/transcribe')
    expect(init.body).toBeInstanceOf(FormData)
  })

  it('shows an error and submits nothing when the backend refuses', async () => {
    stubTranscribeResponse({ ok: false })
    const onTranscript = vi.fn()
    render(<MicButton onTranscript={onTranscript} disabled={false} />)

    fireEvent.click(screen.getByRole('button', { name: /voice conversation/i }))
    await waitFor(() => expect(FakeMediaRecorder.instances).toHaveLength(1))
    fireEvent.click(screen.getByRole('button', { name: /stop recording/i }))

    expect(await screen.findByRole('alert')).toHaveTextContent(/unavailable/i)
    expect(onTranscript).not.toHaveBeenCalled()
  })

  it('shows an error when the microphone is blocked', async () => {
    getUserMedia.mockRejectedValue(new Error('denied'))
    render(<MicButton onTranscript={vi.fn()} disabled={false} />)

    fireEvent.click(screen.getByRole('button', { name: /voice conversation/i }))
    expect(await screen.findByRole('alert')).toHaveTextContent(/microphone/i)
    // The failed attempt must not leave the button stuck mid-start.
    expect(screen.getByRole('button', { name: /voice conversation/i })).toBeEnabled()
  })

  it('is disabled while the agent is busy', () => {
    render(<MicButton onTranscript={vi.fn()} disabled={true} />)
    expect(screen.getByRole('button', { name: /voice conversation/i })).toBeDisabled()
  })
})
