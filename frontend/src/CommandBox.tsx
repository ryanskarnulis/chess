import { useState, type FormEvent } from 'react'

export interface CommandBoxProps {
  /** Send a (trimmed, non-empty) command to the agent. */
  onSubmit: (text: string) => void
  /** The agent's latest reply, shown below the input. */
  commentary: string | null
  /** A command is in flight — the input locks and a hint replaces the reply. */
  thinking: boolean
}

/**
 * Free-text command box for the agent plus its commentary display. Purely
 * presentational: it owns only the ephemeral input text and hands trimmed,
 * non-empty commands to the parent, which owns the backend call.
 */
export function CommandBox({ onSubmit, commentary, thinking }: CommandBoxProps) {
  const [text, setText] = useState('')
  const trimmed = text.trim()

  function handleSubmit(event: FormEvent) {
    event.preventDefault()
    if (!trimmed || thinking) return
    onSubmit(trimmed)
    setText('')
  }

  return (
    <section className="command-box" aria-label="Agent">
      <form onSubmit={handleSubmit}>
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
      {thinking ? (
        <p className="commentary commentary-thinking" role="status">
          Thinking…
        </p>
      ) : (
        commentary && (
          <p className="commentary" role="status">
            {commentary}
          </p>
        )
      )}
    </section>
  )
}
