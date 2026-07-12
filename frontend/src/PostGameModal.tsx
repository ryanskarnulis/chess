import { useState } from 'react'
import { fetchPgn, type Outcome } from './api'
import { ReviewPanel } from './ReviewPanel'

export interface PostGameModalProps {
  outcome: Outcome
  /** The human's side — the verdict ("You won") is theirs, not white's. */
  playerColor: 'white' | 'black'
  /** Hidden-not-unmounted when false, so a fetched review (a whole-game
   * engine run) survives dismissing and reopening the screen. */
  open?: boolean
  /** Called bare (no arguments), so the parent's optional new-game color
   * argument stays untouched. */
  onNewGame: () => void
  /** Dismiss to look at the final board; the parent owns reopening. */
  onClose: () => void
}

function verdict(outcome: Outcome, playerColor: 'white' | 'black'): string {
  if (outcome.winner === null) return 'Draw'
  return outcome.winner === playerColor ? 'You won' : 'You lost'
}

/**
 * The post-game screen: pops up when the game ends with the player's verdict,
 * the review flow, and what comes next (new game, PGN export). Presentational
 * over the outcome the backend declared — nothing is decided here.
 */
export function PostGameModal({
  outcome,
  playerColor,
  open = true,
  onNewGame,
  onClose,
}: PostGameModalProps) {
  const [copyLabel, setCopyLabel] = useState('Copy PGN')
  // Inline (not a class) so jsdom's computed styles see it too.
  const hidden = open ? undefined : { display: 'none' as const }

  const copyPgn = async () => {
    const pgn = await fetchPgn()
    if (pgn === null) {
      setCopyLabel('PGN unavailable')
      return
    }
    try {
      await navigator.clipboard.writeText(pgn)
      setCopyLabel('Copied ✓')
    } catch {
      setCopyLabel('Copy failed')
    }
  }

  return (
    <>
      <div className="postgame-backdrop" style={hidden} onClick={onClose} />
      <div
        className="postgame-modal"
        style={hidden}
        role="dialog"
        aria-modal="true"
        aria-label="Game over"
      >
        <h2 className="postgame-verdict">{verdict(outcome, playerColor)}</h2>
        <p className="postgame-detail">
          {`${outcome.termination.replaceAll('_', ' ')} · ${outcome.result}`}
        </p>
        <div className="postgame-actions">
          <button type="button" onClick={() => onNewGame()}>
            New game
          </button>
          <button type="button" onClick={copyPgn}>
            {copyLabel}
          </button>
          <button type="button" className="postgame-close" onClick={onClose}>
            Close
          </button>
        </div>
        <ReviewPanel />
      </div>
    </>
  )
}
