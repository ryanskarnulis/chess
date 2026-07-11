import { useState, type FormEvent } from 'react'
import { MicButton } from './MicButton'
import { unlockAudio } from './tts'

export interface CommandBoxProps {
  /** Send a (trimmed, non-empty) command to the agent. Hands-free voice
   * awaits the returned promise before listening again. */
  onSubmit: (text: string) => void | Promise<void>
  /** The agent's latest reply, shown below the input. */
  commentary: string | null
  /** A command is in flight — the input locks and a hint replaces the reply. */
  thinking: boolean
  /** Whether replies are spoken aloud; null hides the toggle until known. */
  voiceOutput: boolean | null
  /** Ask the parent to turn voice output on/off (the mute toggle). */
  onToggleVoice: (enabled: boolean) => void
  /** Render the commentary/thinking block below the input (default). The
   * mobile layout passes false and stages commentary in the agent bubble. */
  showCommentary?: boolean
}

/**
 * Free-text command box for the agent plus its commentary display. Purely
 * presentational: it owns only the ephemeral input text and hands trimmed,
 * non-empty commands to the parent, which owns the backend call.
 */
export function CommandBox({
  onSubmit,
  commentary,
  thinking,
  voiceOutput,
  onToggleVoice,
  showCommentary = true,
}: CommandBoxProps) {
  const [text, setText] = useState('')
  const trimmed = text.trim()

  function handleSubmit(event: FormEvent) {
    event.preventDefault()
    // Mobile browsers only allow audio primed inside a user gesture; this
    // submit is the gesture that precedes the agent's spoken reply.
    unlockAudio()
    if (!trimmed || thinking) return
    void onSubmit(trimmed)
    setText('')
  }

  return (
    <section className="command-box" aria-label="Agent">
      <form onSubmit={handleSubmit}>
        {/* Voice in/out controls lead the row so they hold their position
            regardless of how the input or commentary below reflows. */}
        {/* Voice in: the transcript goes down the exact same pipeline as a
            typed command. Renders nothing in unsupporting browsers. */}
        <MicButton onTranscript={onSubmit} disabled={thinking} />
        {/* Voice out: mute/unmute. Hidden until the setting has loaded so
            the toggle never shows a state it just guessed. */}
        {voiceOutput !== null && (
          <button
            type="button"
            className="voice-toggle"
            aria-label={voiceOutput ? 'Turn voice output off' : 'Turn voice output on'}
            title={voiceOutput ? 'Mute the agent' : 'Speak replies aloud'}
            onClick={() => {
              // Unmuting is a gesture too — prime playback while we have it.
              unlockAudio()
              onToggleVoice(!voiceOutput)
            }}
          >
            {voiceOutput ? '🔊' : '🔇'}
          </button>
        )}
        <input
          type="text"
          aria-label="Command"
          placeholder="Tell the agent what to do…"
          value={text}
          onChange={(e) => setText(e.target.value)}
          disabled={thinking}
        />
        <button type="submit" disabled={thinking || !trimmed}>
          Send
        </button>
      </form>
      {showCommentary &&
        (thinking ? (
          <p className="commentary commentary-thinking" role="status">
            Thinking…
          </p>
        ) : (
          commentary && (
            <p className="commentary" role="status">
              {commentary}
            </p>
          )
        ))}
    </section>
  )
}
