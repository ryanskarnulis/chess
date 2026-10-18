import { useState } from 'react'
import { fetchPgn, fetchReview, type GameReview, type Outcome } from './api'
import { COPY_IDLE, copyText } from './clipboard'
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
 * The post-game sheet: rises when the game ends with the player's verdict and
 * what comes next — a new game, the review, the PGN, or a look at the final
 * board. Presentational over the outcome the backend declared — nothing is
 * decided here. It owns the review's fetch, so a review run once (a
 * whole-game engine pass) is kept for the rest of the post-game.
 */
export function PostGameModal({
  outcome,
  playerColor,
  open = true,
  onNewGame,
  onClose,
}: PostGameModalProps) {
  const [copyLabel, setCopyLabel] = useState(COPY_IDLE)
  const [reviewOpen, setReviewOpen] = useState(false)
  const [review, setReview] = useState<GameReview | null>(null)
  const [loading, setLoading] = useState(false)
  const [unavailable, setUnavailable] = useState(false)
  // Inline (not a class) so jsdom's computed styles see it too.
  const hidden = open ? undefined : { display: 'none' as const }

  const copyPgn = async () => {
    // This screen has no PGN in hand — it fetches one, and the fetch is the
    // half the chat's chip does not have (that one was handed the notation
    // with the reply). The write and the label it reports are shared.
    const pgn = await fetchPgn()
    if (pgn === null) {
      setCopyLabel('PGN unavailable')
      return
    }
    setCopyLabel(await copyText(pgn))
  }

  // The review runs Stockfish over the whole game, so it runs once, when first
  // asked for, and again only on a retry after a refusal.
  const runReview = async () => {
    setLoading(true)
    setUnavailable(false)
    try {
      const result = await fetchReview()
      if (result) setReview(result)
      else setUnavailable(true)
    } finally {
      setLoading(false)
    }
  }

  const openReview = () => {
    setReviewOpen(true)
    if (review === null && !loading) void runReview()
  }

  return (
    <div style={hidden}>
      <div className="sheet-backdrop postgame-backdrop" onClick={onClose} />
      <section className="sheet postgame-modal" role="dialog" aria-modal="true" aria-label="Game over">
        <span className="sheet-handle" aria-hidden="true" />
        <h2 className="postgame-verdict">{verdict(outcome, playerColor)}</h2>
        <p className="postgame-detail">
          {`${outcome.termination.replaceAll('_', ' ')} · ${outcome.result}`}
        </p>
        <button type="button" className="postgame-primary" onClick={() => onNewGame()}>
          New game
        </button>
        <button type="button" className="postgame-secondary" onClick={openReview}>
          Review the game
        </button>
        <div className="postgame-links">
          <button type="button" className="postgame-link" onClick={copyPgn}>
            {copyLabel}
          </button>
          <button type="button" className="postgame-link postgame-close" onClick={onClose}>
            Look at the board
          </button>
        </div>
      </section>
      {reviewOpen && (
        <ReviewPanel
          review={review}
          loading={loading}
          unavailable={unavailable}
          playerColor={playerColor}
          onRetry={() => void runReview()}
          onDone={() => setReviewOpen(false)}
        />
      )}
    </div>
  )
}
