export interface AgentBubbleProps {
  /** The agent's latest reply, or null before the first command. */
  commentary: string | null
  /** A command is in flight — a hint replaces the reply. */
  thinking: boolean
  /** What the turn in flight is doing right now, when it has said. Preferred
   * over the generic hint: a turn has intentional phases, and "Stockfish is
   * calculating" is the answer to the question a spinner only raises. */
  progress?: string | null
}

/**
 * The agent's face on mobile: the spider mascot with its commentary in a
 * speech bubble. Presentational twin of CommandBox's commentary block —
 * the same texts, just staged like a character.
 */
export function AgentBubble({ commentary, thinking, progress = null }: AgentBubbleProps) {
  // Progress outranks the generic hint, and stands alone when there is no
  // command in flight: a dragged move is a turn too, and nothing else here
  // would say so.
  const busy = thinking || progress !== null
  const text = busy ? (progress ?? 'Thinking…') : (commentary ?? 'Your move.')
  return (
    <div className="agent-bubble">
      <img className="spider-icon" src="/glitch.png" alt="" aria-hidden="true" />
      <p className={busy ? 'bubble bubble-thinking' : 'bubble'} role="status">
        {text}
      </p>
    </div>
  )
}
