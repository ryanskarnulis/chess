import { useState, type FormEvent } from 'react'
import { GatewayLink } from './GatewayLink'
import { MicButton } from './MicButton'
import { unlockAudio } from './tts'
import { beginInteraction } from './voiceTiming'

/** The backend's cap on one command (`agent_api.MAX_AGENT_MESSAGE_LENGTH`):
 * longer is refused 422, so the box never lets one be typed. */
export const MAX_COMMAND_LENGTH = 8_000

export interface CommandBoxProps {
  /** Send a (trimmed, non-empty) command to the agent. Hands-free voice
   * awaits the returned promise before listening again. */
  onSubmit: (text: string, interactionId?: string) => void | Promise<void>
  /** The agent's latest reply, shown below the input. */
  commentary: string | null
  /** A command is in flight — the input locks and a hint replaces the reply. */
  thinking: boolean
  /** What the turn in flight is doing right now, when it has said. Same
   * contract as `AgentBubble`'s: the specific line wins over the generic. */
  progress?: string | null
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

  // Voice-first: no standing Send button. The ↑ submit docks inside the
  // input's right edge only while there is text to send; the locked states
  // that used to disable Send simply leave it hidden (the input is disabled
  // in those states anyway).
  const sendable = Boolean(trimmed) && !locked

  return (
    <section className="command-box" aria-label="Agent">
      <form onSubmit={handleSubmit}>
        {/* Back-to-gateway lives in the row as its leftmost control. Renders
            nothing off the gateway domain — the row then starts at the input. */}
        <GatewayLink />
        {/* One flat, underlined field: the input, the ↑ submit docked while
            there is text to send, and the mic at its right end. Whether
            replies are spoken is a setting, so it lives in the Game menu. */}
        <div className={sendable ? 'command-input command-input-sendable' : 'command-input'}>
          <input
            type="text"
            aria-label="Command"
            maxLength={MAX_COMMAND_LENGTH}
            placeholder={disabled ? 'No agent — play on' : 'Talk to Glitch…'}
            value={text}
            onChange={(e) => setText(e.target.value)}
            disabled={locked}
          />
          {sendable && (
            <button type="submit" className="command-submit" aria-label="Send">
              ↑
            </button>
          )}
          {/* Voice in: the transcript goes down the exact same pipeline as a
              typed command. Renders nothing in unsupporting browsers. */}
          <MicButton
            onTranscript={onSubmit}
            /* The player's wait starts when they stop speaking (#317). */
            beginUtterance={() => beginInteraction('voice')}
            disabled={locked}
          />
        </div>
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
