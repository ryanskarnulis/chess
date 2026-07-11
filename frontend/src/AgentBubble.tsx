import { SpiderIcon } from './SpiderIcon'

export interface AgentBubbleProps {
  /** The agent's latest reply, or null before the first command. */
  commentary: string | null
  /** A command is in flight — a hint replaces the reply. */
  thinking: boolean
}

/**
 * The agent's face on mobile: the spider mascot with its commentary in a
 * speech bubble. Presentational twin of CommandBox's commentary block —
 * the same texts, just staged like a character.
 */
export function AgentBubble({ commentary, thinking }: AgentBubbleProps) {
  const text = thinking ? 'Thinking…' : (commentary ?? 'Your move.')
  return (
    <div className="agent-bubble">
      <SpiderIcon />
      <p className={thinking ? 'bubble bubble-thinking' : 'bubble'} role="status">
        {text}
      </p>
    </div>
  )
}
