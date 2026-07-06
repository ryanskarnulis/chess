import { useRef, useState } from 'react'
import { transcribe } from './api'

export interface MicButtonProps {
  /** Receives the recognized text — the caller sends it down the same
   * pipeline as a typed command. */
  onTranscript: (text: string) => void
  /** Recording is pointless while the agent is busy — lock the button. */
  disabled: boolean
}

type MicState = 'idle' | 'recording' | 'transcribing'

/**
 * Push-to-talk: click to record, click again to stop; the clip goes to the
 * backend for transcription and the text comes back through `onTranscript`.
 * Renders nothing in browsers without MediaRecorder/getUserMedia — voice is
 * an enhancement, the text box always works.
 */
export function MicButton({ onTranscript, disabled }: MicButtonProps) {
  const [micState, setMicState] = useState<MicState>('idle')
  const [error, setError] = useState<string | null>(null)
  const recorderRef = useRef<MediaRecorder | null>(null)

  const supported =
    typeof MediaRecorder !== 'undefined' &&
    typeof navigator.mediaDevices?.getUserMedia === 'function'
  if (!supported) return null

  async function start() {
    setError(null)
    let stream: MediaStream
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true })
    } catch {
      setError('Microphone unavailable — check browser permissions.')
      return
    }
    const recorder = new MediaRecorder(stream)
    const chunks: Blob[] = []
    recorder.ondataavailable = (e) => {
      if (e.data.size > 0) chunks.push(e.data)
    }
    recorder.onstop = async () => {
      // Release the mic as soon as the clip is captured; transcription is
      // backend work.
      stream.getTracks().forEach((t) => t.stop())
      setMicState('transcribing')
      const clip = new Blob(chunks, { type: recorder.mimeType || 'audio/webm' })
      const text = await transcribe(clip)
      setMicState('idle')
      if (text === null) {
        setError('Voice input is unavailable.')
      } else if (text.trim()) {
        onTranscript(text.trim())
      } else {
        setError("Didn't catch that — try again.")
      }
    }
    recorderRef.current = recorder
    recorder.start()
    setMicState('recording')
  }

  function toggle() {
    if (micState === 'recording') recorderRef.current?.stop()
    else if (micState === 'idle') void start()
  }

  return (
    <>
      <button
        type="button"
        className={`mic-button mic-${micState}`}
        aria-label={micState === 'recording' ? 'Stop recording' : 'Start voice command'}
        title={micState === 'recording' ? 'Stop recording' : 'Speak a command'}
        onClick={toggle}
        disabled={disabled || micState === 'transcribing'}
      >
        {micState === 'recording' ? '■' : micState === 'transcribing' ? '…' : '🎤'}
      </button>
      {error && (
        <p className="mic-error" role="alert">
          {error}
        </p>
      )}
    </>
  )
}
