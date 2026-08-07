import { act, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { sendCommand } from './api'
import { MicButton } from './MicButton'
import { audioIdle, stopPlayback, unlockAudio } from './tts'
import { createVad, type SpeechEvents, type Vad } from './vad'

// Mobile browsers only allow playback primed inside a user gesture; the mic
// tap is the last gesture before the agent's spoken reply, so it must unlock.
vi.mock('./tts', () => ({
  unlockAudio: vi.fn(),
  audioIdle: vi.fn(async () => {}),
  stopPlayback: vi.fn(),
}))

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
  state: 'inactive' | 'recording' = 'inactive'

  stream: unknown

  constructor(stream: unknown) {
    this.stream = stream
    FakeMediaRecorder.instances.push(this)
  }

  start() {
    this.started = true
    this.state = 'recording'
  }

  stop() {
    this.state = 'inactive'
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

/**
 * Tap the mic and wait for hands-free listening to actually engage — the
 * label says "stop listening" from the first frame ('starting'), so the class
 * is the only signal that the VAD is up and speech events will land.
 */
async function enterConversation() {
  fireEvent.click(screen.getByRole('button', { name: /voice conversation/i }))
  await waitFor(() => expect(screen.getByRole('button')).toHaveClass('mic-listening'))
}

/** Simulate the VAD reporting a finished utterance. */
function speak(samples = [0.1, -0.1]) {
  act(() => speechEvents?.onSpeechEnd(new Float32Array(samples)))
}

/**
 * The wedge (#232): the "…" working icon with the VAD paused and nothing left
 * to resume it — hands-free silently deaf until someone notices and taps out.
 * No path through a turn may end here.
 */
function expectNotWedged() {
  expect(screen.getByRole('button')).not.toHaveClass('mic-working')
}

beforeEach(() => {
  FakeMediaRecorder.instances = []
  getUserMedia.mockReset()
  getUserMedia.mockResolvedValue(fakeStream)
  fakeTrack.stop.mockReset()
  vi.mocked(unlockAudio).mockClear()
  vi.mocked(stopPlayback).mockClear()
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
    render(<MicButton onTranscript={onTranscript} disabled={false} />)
    await enterConversation()

    // Installed after the session is up: starting one also waits on playback.
    let finishSpeech!: () => void
    vi.mocked(audioIdle).mockImplementation(
      () =>
        new Promise<void>((resolve) => {
          finishSpeech = resolve
        }),
    )
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

  // --- a turn that dies must still leave the loop recoverable (#232) -------
  //
  // The primary voice device is a phone on wifi and the backend container
  // restarts on every merge to main, so a request that never completes is
  // routine, not exotic. Whatever a turn's promise does, the session ends
  // visibly or resumes cleanly — never "…" with the VAD paused and no line.

  it('exits conversation mode when transcription cannot be reached', async () => {
    // A refused connection: fetch rejects rather than answering 503, so the
    // rejection used to escape past the resume path entirely.
    vi.stubGlobal('fetch', vi.fn(() => Promise.reject(new TypeError('Failed to fetch'))))
    render(<MicButton onTranscript={vi.fn()} disabled={false} />)
    await enterConversation()

    speak()
    expect(await screen.findByRole('alert')).toHaveTextContent(/unavailable/i)
    expectNotWedged()
    expect(fakeVad.destroy).toHaveBeenCalled()
    expect(screen.getByRole('button', { name: /voice conversation/i })).toBeInTheDocument()
  })

  it('exits conversation mode when a transcription response does not parse', async () => {
    // A gateway that answers 200 with an HTML error page: there is no text to
    // read, and the .json() rejection wedged the loop exactly like a dead
    // socket did.
    vi.stubGlobal('fetch', vi.fn(async () => new Response('<html>502</html>', { status: 200 })))
    render(<MicButton onTranscript={vi.fn()} disabled={false} />)
    await enterConversation()

    speak()
    expect(await screen.findByRole('alert')).toHaveTextContent(/unavailable/i)
    expectNotWedged()
    expect(screen.getByRole('button', { name: /voice conversation/i })).toBeInTheDocument()
  })

  it('keeps listening when the command the transcript went to cannot be reached', async () => {
    // The real sendCommand runs here, so this pins the two layers together: a
    // dropped command is a turn that says nothing, not a dead session. The mic
    // reopens because nothing threw — the api client resolved the failure.
    vi.stubGlobal(
      'fetch',
      vi.fn(async (url: string) => {
        if (String(url).includes('/api/voice/transcribe'))
          return new Response(JSON.stringify({ text: 'castle' }), { status: 200 })
        throw new TypeError('Failed to fetch')
      }),
    )
    render(<MicButton onTranscript={(text) => sendCommand(text).then(() => {})} disabled={false} />)
    await enterConversation()

    speak()
    await waitFor(() => expect(fakeVad.resume).toHaveBeenCalled())
    expectNotWedged()
    expect(screen.getByRole('button', { name: /stop listening/i })).toBeInTheDocument()
  })

  it('exits conversation mode when the agent turn rejects outright', async () => {
    // handleUtterance does not own onTranscript's promise, so nothing it can
    // fix upstream makes this unreachable — the belt is for the unknown.
    stubTranscribeResponse({ ok: true, text: 'castle' })
    const onTranscript = vi.fn(() => Promise.reject(new Error('turn died')))
    render(<MicButton onTranscript={onTranscript} disabled={false} />)
    await enterConversation()

    expect(await screen.findByRole('button', { name: /stop listening/i })).toBeInTheDocument()
    speak()
    expect(await screen.findByRole('alert')).toHaveTextContent(/tap to start/i)
    expectNotWedged()
    expect(fakeVad.destroy).toHaveBeenCalled()
    expect(screen.getByRole('button', { name: /voice conversation/i })).toBeInTheDocument()
  })

  it('ends the conversation on a second tap, and the spoken reply with it', async () => {
    render(<MicButton onTranscript={vi.fn()} disabled={false} />)
    await enterConversation()

    fireEvent.click(screen.getByRole('button', { name: /stop listening/i }))
    expect(fakeVad.destroy).toHaveBeenCalled()
    // Tapping out while the agent is talking must silence it: the reply would
    // otherwise still be audible over the next session's open mic.
    expect(stopPlayback).toHaveBeenCalled()
    expect(screen.getByRole('button', { name: /voice conversation/i })).toBeInTheDocument()
  })

  // --- restarting mid-reply must not poison the new session (#257) ---------

  it('accepts utterances again after a restart during the previous reply', async () => {
    // The busy latch used to be component-scoped: session 1 held it until its
    // reply finished playing, so everything said to session 2 hit the early
    // return and vanished while the button still read "listening".
    stubTranscribeResponse({ ok: true, text: 'castle' })
    const onTranscript = vi.fn(async () => {})
    render(<MicButton onTranscript={onTranscript} disabled={false} />)
    await enterConversation()

    // Only session 1's reply hangs; every other wait resolves at once.
    let finishSpeech!: () => void
    vi.mocked(audioIdle).mockImplementationOnce(
      () =>
        new Promise<void>((resolve) => {
          finishSpeech = resolve
        }),
    )
    speak()
    // Session 1 is now parked on the reply's playback, holding the latch.
    await waitFor(() => expect(onTranscript).toHaveBeenCalledTimes(1))
    fireEvent.click(screen.getByRole('button', { name: /stop listening/i }))
    await enterConversation()

    speak()
    await waitFor(() => expect(onTranscript).toHaveBeenCalledTimes(2))
    expect(onTranscript).toHaveBeenLastCalledWith('castle')

    // The abandoned turn settling afterwards touches neither the latch nor
    // the live session's mic.
    await act(async () => finishSpeech())
    expectNotWedged()
    speak()
    await waitFor(() => expect(onTranscript).toHaveBeenCalledTimes(3))
  })

  it('does not open the mic until playback is idle', async () => {
    // Starting a conversation while a typed command's reply is still speaking
    // would put the VAD in front of the speakers.
    let finishSpeech!: () => void
    vi.mocked(audioIdle).mockImplementationOnce(
      () =>
        new Promise<void>((resolve) => {
          finishSpeech = resolve
        }),
    )
    render(<MicButton onTranscript={vi.fn()} disabled={false} />)

    fireEvent.click(screen.getByRole('button', { name: /voice conversation/i }))
    await waitFor(() => expect(unlockAudio).toHaveBeenCalled())
    expect(createVad).not.toHaveBeenCalled()

    await act(async () => finishSpeech())
    await waitFor(() => expect(createVad).toHaveBeenCalled())
    expect(screen.getByRole('button')).toHaveClass('mic-listening')
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

  it('says why hands-free is unavailable while degrading', async () => {
    // The reason must reach the screen: on a phone there is no console, and
    // a silent fallback reads as "continuous voice is broken".
    vi.mocked(createVad).mockImplementation(async (events) => {
      events.onUnavailable?.('no AudioWorklet')
      return null
    })
    render(<MicButton onTranscript={vi.fn()} disabled={false} />)
    fireEvent.click(screen.getByRole('button', { name: /voice conversation/i }))
    await waitFor(() => expect(screen.getByRole('alert')).toBeInTheDocument())
    expect(screen.getByRole('alert').textContent).toMatch(/hands-free.*unavailable/i)
    expect(screen.getByRole('alert').textContent).toMatch(/no AudioWorklet/)
    // Still degrades: push-to-talk recording starts despite the notice.
    await waitFor(() => expect(FakeMediaRecorder.instances).toHaveLength(1))
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

  it('releases the microphone when unmounted mid-recording', async () => {
    // Nothing else ever stops these tracks: navigating away used to leave the
    // browser's mic indicator lit for the life of the page.
    const onTranscript = vi.fn()
    const { unmount } = render(<MicButton onTranscript={onTranscript} disabled={false} />)

    fireEvent.click(screen.getByRole('button', { name: /voice conversation/i }))
    await waitFor(() => expect(FakeMediaRecorder.instances).toHaveLength(1))

    unmount()
    expect(fakeTrack.stop).toHaveBeenCalled()
    // Teardown is not a clip worth transcribing.
    expect(onTranscript).not.toHaveBeenCalled()
  })

  it('does not start recording when the user taps stop during the permission prompt', async () => {
    let grant!: (stream: unknown) => void
    getUserMedia.mockReturnValue(
      new Promise((resolve) => {
        grant = resolve
      }),
    )
    render(<MicButton onTranscript={vi.fn()} disabled={false} />)

    fireEvent.click(screen.getByRole('button', { name: /voice conversation/i }))
    // Still awaiting the prompt — the button offers the way out.
    fireEvent.click(await screen.findByRole('button', { name: /stop listening/i }))
    await act(async () => grant(fakeStream))

    expect(FakeMediaRecorder.instances).toHaveLength(0)
    // A grant that arrives after the exit is released, not orphaned.
    expect(fakeTrack.stop).toHaveBeenCalled()
    expect(screen.getByRole('button', { name: /voice conversation/i })).toBeInTheDocument()
  })

  it('does not start recording when unmounted during the permission prompt', async () => {
    let grant!: (stream: unknown) => void
    getUserMedia.mockReturnValue(
      new Promise((resolve) => {
        grant = resolve
      }),
    )
    const { unmount } = render(<MicButton onTranscript={vi.fn()} disabled={false} />)

    fireEvent.click(screen.getByRole('button', { name: /voice conversation/i }))
    await screen.findByRole('button', { name: /stop listening/i })
    unmount()
    await act(async () => grant(fakeStream))

    // Worst case of all: recording on a dead component, no button left to stop it.
    expect(FakeMediaRecorder.instances).toHaveLength(0)
    expect(fakeTrack.stop).toHaveBeenCalled()
  })
})
