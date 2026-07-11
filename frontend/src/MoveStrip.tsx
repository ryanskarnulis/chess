import { useEffect, useRef } from 'react'

export interface MoveStripProps {
  /** Moves played so far, in SAN. */
  history: string[]
  /** Plies currently shown on the board: history.length when live, fewer
   * while reviewing. The move that produced the shown position (the
   * currentPly-th) is highlighted. */
  currentPly: number
  onBack: () => void
  onForward: () => void
  canBack: boolean
  canForward: boolean
}

/**
 * One-line move history with review arrows. The arrows only change which
 * past position the board shows — they never mutate the game (that's undo).
 */
export function MoveStrip({
  history,
  currentPly,
  onBack,
  onForward,
  canBack,
  canForward,
}: MoveStripProps) {
  const currentRef = useRef<HTMLSpanElement | null>(null)

  // Keep the highlighted move in view as the game grows or review walks.
  useEffect(() => {
    // Guarded: jsdom has no scrollIntoView.
    currentRef.current?.scrollIntoView?.({ inline: 'end', block: 'nearest' })
  }, [history.length, currentPly])

  return (
    <div className="move-strip">
      <button type="button" aria-label="Previous move" onClick={onBack} disabled={!canBack}>
        ◀
      </button>
      <div className="move-strip-moves">
        {history.map((san, i) => (
          <span key={i} className="move-strip-item">
            {i % 2 === 0 && <span className="move-number">{i / 2 + 1}.</span>}
            <span
              className={i === currentPly - 1 ? 'move current' : 'move'}
              ref={i === currentPly - 1 ? currentRef : undefined}
            >
              {san}
            </span>
          </span>
        ))}
      </div>
      <button type="button" aria-label="Next move" onClick={onForward} disabled={!canForward}>
        ▶
      </button>
    </div>
  )
}
