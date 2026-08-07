import type { ReactNode } from 'react'
import { materialScore, PIECE_GLYPHS, type PieceType } from './pieces'

export interface CapturedPiecesProps {
  /**
   * Pieces each colour has captured, in capture order (from the backend).
   * `white` holds the black pieces White took, and vice versa.
   */
  captured: { white: string[]; black: string[] }
  /** Which side the player drives: their captures sit left, under YOU;
   * Glitch's mirror on the right. */
  playerColor: 'white' | 'black'
  /** The turn chip (and anything beside it), centered between the clusters. */
  children?: ReactNode
}

/** One cluster's captured pieces — glyphs that *look like* the opponent's. */
function Side({
  side,
  label,
  ariaLabel,
  mirrored = false,
  symbols,
  advantage,
}: {
  side: 'white' | 'black'
  label: string
  ariaLabel: string
  /** Glitch's cluster reads right-to-left: glyphs, then the label. */
  mirrored?: boolean
  symbols: string[]
  advantage: number
}) {
  // White captures black pieces, which must *look* black. The dark theme
  // renders text glyphs in the light ink color, so the filled ("black")
  // set reads as white pieces and the hollow ("white") set reads as black —
  // the perceived colors are inverted from the glyph names. Use the same-
  // name set so the rendered color matches the captured piece's color.
  const glyphColor = side
  // A cluster with no captures collapses entirely — no orphaned label — but
  // the element itself stays, an empty flex sibling keeping the chip put.
  const cluster = symbols.length > 0 && (
    <>
      <span className="captured-pieces-glyphs">
        {symbols.map((s, i) => (
          <span key={i} aria-hidden>
            {PIECE_GLYPHS[glyphColor][s as PieceType] ?? '?'}
          </span>
        ))}
      </span>
      {advantage > 0 && <span className="captured-advantage">{`+${advantage}`}</span>}
    </>
  )
  return (
    <div
      className={mirrored ? 'captured-side captured-side-right' : 'captured-side'}
      aria-label={ariaLabel}
    >
      {symbols.length > 0 && !mirrored && <span className="captured-label">{label}</span>}
      {cluster}
      {symbols.length > 0 && mirrored && <span className="captured-label">{label}</span>}
    </div>
  )
}

/**
 * The one-line game status row under the board: the player's captures on the
 * left, Glitch's on the right, whoever is ahead wearing the material badge,
 * and the turn chip (passed as children — the app owns the status text)
 * centered between them. Presentational — the backend derives captures from
 * the move stack.
 */
export function CapturedPieces({ captured, playerColor, children }: CapturedPiecesProps) {
  const diff = materialScore(captured.white) - materialScore(captured.black)
  const advantage = { white: diff, black: -diff }
  const opponentColor = playerColor === 'white' ? 'black' : 'white'
  return (
    <section className="status-row" aria-label="Game status">
      <Side
        side={playerColor}
        label="YOU"
        ariaLabel="Captured by you"
        symbols={captured[playerColor]}
        advantage={advantage[playerColor]}
      />
      <div className="status-center">{children}</div>
      <Side
        side={opponentColor}
        label="GLITCH"
        ariaLabel="Captured by Glitch"
        mirrored
        symbols={captured[opponentColor]}
        advantage={advantage[opponentColor]}
      />
    </section>
  )
}
