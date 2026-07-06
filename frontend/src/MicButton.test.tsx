import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { MicButton } from './MicButton'

// jsdom has no MediaRecorder/getUserMedia; these fakes stand in so the tests
// drive the full record → stop → transcribe → submit flow without a browser.

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

beforeEach(() => {
  FakeMediaRecorder.instances = []
  getUserMedia.mockReset()
  getUserMedia.mockResolvedValue(fakeStream)
  fakeTrack.stop.mockReset()
})

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('MicButton', () => {
  it('renders nothing when the browser has no recording support', () => {
    // No MediaRecorder / mediaDevices stubs: plain jsdom.
    render(<MicButton onTranscript={vi.fn()} disabled={false} />)
    expect(screen.queryByRole('button')).not.toBeInTheDocument()
  })

  it('records, transcribes, and hands the text to onTranscript', async () => {
    stubMediaSupport()
    stubTranscribeResponse({ ok: true, text: '  pawn to e4  ' })
    const onTranscript = vi.fn()
    render(<MicButton onTranscript={onTranscript} disabled={false} />)

    fireEvent.click(screen.getByRole('button', { name: /voice command/i }))
    await waitFor(() => expect(FakeMediaRecorder.instances).toHaveLength(1))
    expect(getUserMedia).toHaveBeenCalledWith({ audio: true })
    expect(FakeMediaRecorder.instances[0].started).toBe(true)

    fireEvent.click(screen.getByRole('button', { name: /stop/i }))
    await waitFor(() => expect(onTranscript).toHaveBeenCalledWith('pawn to e4'))
    // The mic is released once the clip is captured.
    expect(fakeTrack.stop).toHaveBeenCalled()
    const [url, init] = (fetch as ReturnType<typeof vi.fn>).mock.calls[0]
    expect(url).toBe('/api/voice/transcribe')
    expect(init.body).toBeInstanceOf(FormData)
  })

  it('shows an error and submits nothing when the backend refuses', async () => {
    stubMediaSupport()
    stubTranscribeResponse({ ok: false })
    const onTranscript = vi.fn()
    render(<MicButton onTranscript={onTranscript} disabled={false} />)

    fireEvent.click(screen.getByRole('button', { name: /voice command/i }))
    await waitFor(() => expect(FakeMediaRecorder.instances).toHaveLength(1))
    fireEvent.click(screen.getByRole('button', { name: /stop/i }))

    expect(await screen.findByRole('alert')).toHaveTextContent(/unavailable/i)
    expect(onTranscript).not.toHaveBeenCalled()
  })

  it('shows an error when the microphone is blocked', async () => {
    stubMediaSupport()
    getUserMedia.mockRejectedValue(new Error('denied'))
    render(<MicButton onTranscript={vi.fn()} disabled={false} />)

    fireEvent.click(screen.getByRole('button', { name: /voice command/i }))
    expect(await screen.findByRole('alert')).toHaveTextContent(/microphone/i)
  })

  it('is disabled while the agent is busy', () => {
    stubMediaSupport()
    render(<MicButton onTranscript={vi.fn()} disabled={true} />)
    expect(screen.getByRole('button', { name: /voice command/i })).toBeDisabled()
  })
})
