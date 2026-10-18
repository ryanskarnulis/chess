import type { GameReview, ReviewedMove } from './api'

export interface ReviewPanelProps {
  /** The finished review, or null while it runs or when it could not. */
  review: GameReview | null
  loading: boolean
  /** The backend refused (no engine, no moves): say so and offer a retry. */
  unavailable: boolean
  /** The human's side, so the cards and notes say "You" and "Glitch". */
  playerColor: 'white' | 'black'
  onRetry: () => void
  /** Back to the game-over sheet. */
  onDone: () => void
}

/**
 * Post-game review, a near-full-height sheet of its own: per-side accuracy
 * cards, a legend for the marks, and the move list in two columns where only
 * the flagged moves stand out, each with the engine's better move under it.
 * Presentational over the backend's numbers — nothing is computed here.
 */
export function ReviewPanel({
  review,
  loading,
  unavailable,
  playerColor,
  onRetry,
  onDone,
}: ReviewPanelProps) {
  return (
    <>
      <div className="sheet-backdrop review-backdrop" onClick={onDone} />
      <section
        className="sheet review-panel"
        role="dialog"
        aria-modal="true"
        aria-label="Game review"
      >
        <header className="review-header">
          <h2>Review</h2>
          <button type="button" className="review-done" onClick={onDone}>
            Done
          </button>
        </header>
        {review ? (
          <ReviewResults review={review} playerColor={playerColor} />
        ) : loading ? (
          <p className="panel-empty review-status">Reviewing…</p>
        ) : (
          unavailable && (
            <div className="review-status">
              <p className="panel-empty">Review unavailable</p>
              <button type="button" className="review-retry" onClick={onRetry}>
                Retry
              </button>
            </div>
          )
        )}
      </section>
    </>
  )
}

const MARKS = { inaccuracy: '?!', mistake: '?', blunder: '??' } as const

/** One line of the move list: a move number and each side's move there. */
interface MoveRow {
  number: number
  white: ReviewedMove | null
  black: ReviewedMove | null
}

/** Pair the moves into numbered rows by colour — not by index, so a game
 * resumed with Black to move starts with an empty White cell. */
function moveRows(moves: ReviewedMove[]): MoveRow[] {
  const rows: MoveRow[] = []
  for (const move of moves) {
    const last = rows[rows.length - 1]
    if (move.color === 'black' && last && last.black === null) last.black = move
    else
      rows.push({
        number: rows.length + 1,
        white: move.color === 'white' ? move : null,
        black: move.color === 'black' ? move : null,
      })
  }
  return rows
}

function flagged(move: ReviewedMove | null): move is ReviewedMove {
  return move !== null && move.classification !== 'good'
}

function ReviewResults({
  review,
  playerColor,
}: {
  review: GameReview
  playerColor: 'white' | 'black'
}) {
  const owner = (color: 'white' | 'black') => (color === playerColor ? 'Your' : 'Glitch’s')
  const sides = [playerColor, playerColor === 'white' ? 'black' : 'white'] as const
  return (
    <>
      <div className="review-summary">
        {sides.map((color) => (
          <ColorSummary
            key={color}
            color={color}
            who={color === playerColor ? 'You' : 'Glitch'}
            accuracy={review.accuracy[color]}
            counts={review.counts[color]}
          />
        ))}
      </div>
      {/* The evaluation graph (#528) goes here, between the cards and the
          legend. */}
      <p className="review-legend">
        {(['inaccuracy', 'mistake', 'blunder'] as const).map((c) => (
          <span key={c}>
            <span className={`review-mark review-mark-${c}`}>{MARKS[c]}</span> {c}
          </span>
        ))}
      </p>
      <ol className="review-moves">
        {moveRows(review.moves).map((row) => {
          const notes = [row.white, row.black].filter(flagged)
          return (
            <li
              key={row.number}
              className={notes.length > 0 ? 'review-row review-row-flagged' : 'review-row'}
            >
              <div className="review-row-moves">
                <span className="review-number">{row.number}.</span>
                <MoveCell move={row.white} />
                <MoveCell move={row.black} />
              </div>
              {notes.map((move) => (
                <p key={move.color} className="review-note">
                  {`${owner(move.color)} ${row.number}${move.color === 'white' ? '.' : '…'} ${move.san} — best was ${move.best}`}
                </p>
              ))}
            </li>
          )
        })}
      </ol>
    </>
  )
}

function MoveCell({ move }: { move: ReviewedMove | null }) {
  if (move === null) return <span className="review-cell" />
  return (
    <span className={`review-cell review-${move.classification}`}>
      {move.san}
      {move.classification !== 'good' && (
        <span
          className={`review-mark review-mark-${move.classification}`}
          aria-label={move.classification}
        >
          {MARKS[move.classification]}
        </span>
      )}
    </span>
  )
}

// Spelled out: "inaccuracy" does not take a bare "s" (#454).
const PLURALS = {
  inaccuracy: 'inaccuracies',
  mistake: 'mistakes',
  blunder: 'blunders',
} as const

function ColorSummary({
  color,
  who,
  accuracy,
  counts,
}: {
  color: 'white' | 'black'
  who: 'You' | 'Glitch'
  accuracy: number | undefined
  counts: Record<string, number> | undefined
}) {
  return (
    <div
      className={who === 'You' ? 'review-color review-color-you' : 'review-color'}
      aria-label={`${color} review summary`}
    >
      <span className="review-color-label">{`${who} · ${color}`}</span>
      {accuracy !== undefined && <span className="review-accuracy">{accuracy}%</span>}
      {counts && (
        <span className="review-counts">
          {(['inaccuracy', 'mistake', 'blunder'] as const)
            .filter((c) => (counts[c] ?? 0) > 0)
            .map((c) => `${counts[c]} ${counts[c] === 1 ? c : PLURALS[c]}`)
            .join(', ')}
        </span>
      )}
    </div>
  )
}
