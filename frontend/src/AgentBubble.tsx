import { useState } from 'react'
import { COPY_IDLE, copyText } from './clipboard'

export interface AgentBubbleProps {
  /** The agent's latest reply, or null before the first command. */
  commentary: string | null
  /** A command is in flight — a hint replaces the reply. */
  thinking: boolean
  /** What the turn in flight is doing right now, when it has said. Preferred
   * over the generic hint: a turn has intentional phases, and "Stockfish is
   * calculating" is the answer to the question a spinner only raises. */
  progress?: string | null
  /** The PGN the last reply exported, when it exported one. The notation is
   * the app's to render — Glitch says it is ready, this shows it — so a reply
   * with one gets a copy button and the text itself under the bubble. */
  pgn?: string | null
}

/**
 * The agent's face on mobile: the spider mascot with its commentary in a
 * speech bubble. Presentational twin of CommandBox's commentary block —
 * the same texts, just staged like a character.
 */
export function AgentBubble({
  commentary,
  thinking,
  progress = null,
  pgn = null,
}: AgentBubbleProps) {
  // Progress outranks the generic hint, and stands alone when there is no
  // command in flight: a dragged move is a turn too, and nothing else here
  // would say so.
  const busy = thinking || progress !== null
  const text = busy ? (progress ?? 'Thinking…') : (commentary ?? 'Your move.')
  return (
    <div className="agent-bubble">
      <img className="spider-icon" src="/glitch.png" alt="" aria-hidden="true" />
      <div className="bubble-column">
        <p className={busy ? 'bubble bubble-thinking' : 'bubble'} role="status">
          {text}
        </p>
        {/* Keyed on the notation: a second export is a different game, and a
            button still reading "Copied ✓" from the first would tell the
            player they have something they do not. */}
        {pgn !== null && <PgnChip key={pgn} pgn={pgn} />}
      </div>
    </div>
  )
}

/** The copy affordance for one exported PGN: a button that confirms in place,
 * and the notation itself folded away. */
function PgnChip({ pgn }: { pgn: string }) {
  const [copyLabel, setCopyLabel] = useState(COPY_IDLE)
  return (
    <div className="bubble-pgn">
      <button type="button" onClick={() => void copyText(pgn).then(setCopyLabel)}>
        {copyLabel}
      </button>
      {/* Collapsed, because the dump in the bubble is the thing this slice
          removed. Open, it is still the way to get the notation on a browser
          whose clipboard is unavailable — read it, select it by hand. */}
      <details>
        <summary>Show PGN</summary>
        <pre>{pgn}</pre>
      </details>
    </div>
  )
}
