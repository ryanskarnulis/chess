import { useState, type KeyboardEvent, type PointerEvent } from 'react'
import type { GameReview, ReviewEval, ReviewedMove } from './api'

/** The graph's vertical reach, in pawns: never less than this, so a quiet
 * game's half-pawn wobble doesn't fill the plot edge to edge… */
const MIN_PAWNS = 3
/** …and never more than the backend's ceiling (±1000 cp), where mates sit. */
const MAX_PAWNS = 10

const MARKS = { inaccuracy: '?!', mistake: '?', blunder: '??' } as const

/**
 * The review's evaluation over the whole game (#528): one point per position,
 * from the starting one, always from the player's side, so above the zero line
 * is good for you. Tap, drag or arrow-key along it to read a move. Presentational
 * over the backend's numbers — the only arithmetic here is scaling to the plot.
 */
export function EvalGraph({ review }: { review: GameReview }) {
  const points: ReviewEval[] = [review.start, ...review.moves]
  const last = points.length - 1
  const [selected, setSelected] = useState<number | null>(null)

  const pawns = points.map((p) => p.player_eval_cp / 100)
  const reach = Math.min(
    MAX_PAWNS,
    Math.max(MIN_PAWNS, Math.ceil(Math.max(...pawns.map(Math.abs)))),
  )
  // viewBox units: x is the ply, y is pawns down from the top edge (+reach).
  const y = (p: number) => reach - p
  const line = pawns.map((p, i) => `${i},${y(p)}`).join(' ')
  const area = `0,${reach} ${line} ${last},${reach}`
  const left = (i: number) => `${(i / last) * 100}%`
  const top = (p: number) => `${(y(p) / (2 * reach)) * 100}%`

  const pick = (event: PointerEvent<HTMLDivElement>) => {
    const box = event.currentTarget.getBoundingClientRect()
    if (box.width === 0) return
    const ratio = (event.clientX - box.left) / box.width
    setSelected(Math.max(0, Math.min(last, Math.round(ratio * last))))
  }
  const step = (event: KeyboardEvent<HTMLDivElement>) => {
    const by = ({ ArrowRight: 1, ArrowLeft: -1 } as Record<string, number>)[event.key]
    if (by === undefined) return
    event.preventDefault()
    setSelected((s) => Math.max(0, Math.min(last, (s ?? (by > 0 ? -1 : last + 1)) + by)))
  }

  return (
    <figure className="eval-graph">
      <div className="eval-axis" aria-hidden="true">
        <span>+{reach}</span>
        <span>−{reach}</span>
      </div>
      <div
        className="eval-plot"
        role="img"
        tabIndex={0}
        aria-label={summary(points)}
        onPointerDown={pick}
        onPointerMove={(e) => {
          if (e.pointerType === 'mouse' || e.buttons) pick(e)
        }}
        onKeyDown={step}
      >
        <svg viewBox={`0 0 ${last} ${2 * reach}`} preserveAspectRatio="none">
          <defs>
            <clipPath id="eval-above">
              <rect x="0" y="0" width={last} height={reach} />
            </clipPath>
            <clipPath id="eval-below">
              <rect x="0" y={reach} width={last} height={reach} />
            </clipPath>
          </defs>
          <polygon className="eval-area-you" points={area} clipPath="url(#eval-above)" />
          <polygon className="eval-area-glitch" points={area} clipPath="url(#eval-below)" />
          <line className="eval-zero" x1="0" y1={reach} x2={last} y2={reach} />
          <polyline className="eval-line" points={line} />
        </svg>
        {review.moves.map(
          (move, i) =>
            (move.classification === 'mistake' || move.classification === 'blunder') && (
              <span
                key={i}
                className={`eval-dot eval-dot-${move.classification}`}
                style={{ left: left(i + 1), top: top(pawns[i + 1]) }}
              />
            ),
        )}
        {selected !== null && (
          <span className="eval-cursor" style={{ left: left(selected) }}>
            <span className="eval-cursor-dot" style={{ top: top(pawns[selected]) }} />
          </span>
        )}
      </div>
      <figcaption className="eval-caption">
        {selected === null
          ? 'Above the line is you · tap to read a move'
          : caption(selected === 0 ? null : review.moves[selected - 1], points[selected])}
      </figcaption>
    </figure>
  )
}

/** "+1.2", "−0.4", "0.0"; a mate reads "+M3" / "−M3", and a mate on the
 * board "checkmate". */
function evalText(point: ReviewEval): string {
  if (point.mate) {
    if (point.mate.in === 0) return 'checkmate'
    return `${point.mate.for === 'player' ? '+' : '−'}M${point.mate.in}`
  }
  const pawns = point.player_eval_cp / 100
  if (pawns === 0) return '0.0'
  return `${pawns > 0 ? '+' : '−'}${Math.abs(pawns).toFixed(1)}`
}

function caption(move: ReviewedMove | null, point: ReviewEval): string {
  if (move === null) return `Start · ${evalText(point)}`
  const number = `${move.move_number}${move.color === 'white' ? '.' : '…'}`
  const mark = move.classification === 'good' ? null : MARKS[move.classification]
  return [`${number} ${move.san}${mark ?? ''}`, evalText(point), mark && `best was ${move.best}`]
    .filter(Boolean)
    .join(' · ')
}

function summary(points: ReviewEval[]): string {
  const cps = points.map((p) => p.player_eval_cp)
  const at = (cp: number) => evalText(points[cps.indexOf(cp)])
  return (
    `Evaluation from your side over the game: starts ${evalText(points[0])}, ` +
    `peaks at ${at(Math.max(...cps))}, dips to ${at(Math.min(...cps))}, ` +
    `ends ${evalText(points[points.length - 1])}`
  )
}
