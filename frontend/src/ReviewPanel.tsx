import { useState } from 'react'
import { fetchReview, type GameReview } from './api'

/**
 * Post-game review: on demand (analysis runs Stockfish over the whole game,
 * so it's a button, not automatic), shows per-color accuracy, classification
 * counts, and the move list with mistakes flagged alongside the engine's
 * better alternative. Presentational over the backend's numbers — nothing is
 * computed here.
 */
export function ReviewPanel() {
  const [review, setReview] = useState<GameReview | null>(null)
  const [loading, setLoading] = useState(false)
  const [unavailable, setUnavailable] = useState(false)

  const run = async () => {
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

  return (
    <section className="review-panel" aria-label="Game review">
      <h2>Review</h2>
      {review ? (
        <ReviewResults review={review} />
      ) : (
        <>
          <button onClick={run} disabled={loading}>
            {loading ? 'Reviewing…' : 'Review game'}
          </button>
          {unavailable && <p className="panel-empty">Review unavailable</p>}
        </>
      )}
    </section>
  )
}

function ReviewResults({ review }: { review: GameReview }) {
  return (
    <>
      <div className="review-summary">
        {(['white', 'black'] as const).map((color) => (
          <ColorSummary
            key={color}
            color={color}
            accuracy={review.accuracy[color]}
            counts={review.counts[color]}
          />
        ))}
      </div>
      <table className="review-moves">
        <tbody>
          {review.moves.map((move, i) => (
            <tr key={i} className={`review-${move.classification}`}>
              <th scope="row">{i % 2 === 0 ? `${i / 2 + 1}.` : ''}</th>
              <td>{move.san}</td>
              <td className="review-verdict">
                {move.classification !== 'good' && (
                  <>
                    {move.classification} — best {move.best}
                  </>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </>
  )
}

function ColorSummary({
  color,
  accuracy,
  counts,
}: {
  color: 'white' | 'black'
  accuracy: number | undefined
  counts: Record<string, number> | undefined
}) {
  return (
    <div className="review-color" aria-label={`${color} review summary`}>
      <span className="review-color-label">{color}</span>
      {accuracy !== undefined && <span className="review-accuracy">{accuracy}%</span>}
      {counts && (
        <span className="review-counts">
          {(['inaccuracy', 'mistake', 'blunder'] as const)
            .filter((c) => (counts[c] ?? 0) > 0)
            .map((c) => `${counts[c]} ${c}${counts[c] === 1 ? '' : 's'}`)
            .join(', ')}
        </span>
      )}
    </div>
  )
}
