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
  /** What the turn in flight is doing right now, when it has said. Same
   * contract as `AgentBubble`'s: the specific line wins over the generic. */
  progress?: string | null
  /** Whether replies are spoken aloud; null hides the toggle until known. */
  voiceOutput: boolean | null
  /** Ask the parent to turn voice output on/off (the mute toggle). */
  onToggleVoice: (enabled: boolean) => void
  /** Render the commentary/thinking block below the input (default). The
   * mobile layout passes false and stages commentary in the agent bubble. */
  showCommentary?: boolean
  /** No agent is configured — direct mode. The box locks: there is nothing
   * behind it (the endpoint 503s), so a dead input is the honest state rather
   * than an error the player has to trip over. */
  disabled?: boolean
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
  progress = null,
  voiceOutput,
  onToggleVoice,
  showCommentary = true,
  disabled = false,
}: CommandBoxProps) {
  const [text, setText] = useState('')
  const trimmed = text.trim()
  const locked = thinking || disabled

  function handleSubmit(event: FormEvent) {
    event.preventDefault()
    // Mobile browsers only allow audio primed inside a user gesture; this
    // submit is the gesture that precedes the agent's spoken reply.
    unlockAudio()
    if (!trimmed || locked) return
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
        <MicButton onTranscript={onSubmit} disabled={locked} />
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
          placeholder={disabled ? 'No agent — play on the board' : 'Tell the agent what to do…'}
          value={text}
          onChange={(e) => setText(e.target.value)}
          disabled={locked}
        />
        <button type="submit" disabled={locked || !trimmed}>
          Send
        </button>
      </form>
      {showCommentary &&
        (thinking || progress !== null ? (
          <p className="commentary commentary-thinking" role="status">
            {progress ?? 'Thinking…'}
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
