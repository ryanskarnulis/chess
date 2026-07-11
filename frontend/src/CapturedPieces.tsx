import { materialScore, PIECE_GLYPHS, type PieceType } from './pieces'

export interface CapturedPiecesProps {
  /**
   * Pieces each colour has captured, in capture order (from the backend).
   * `white` holds the black pieces White took, and vice versa.
   */
  captured: { white: string[]; black: string[] }
}

/** One side's captured pieces — glyphs that *look like* the opponent's. */
function Side({
  side,
  symbols,
  advantage,
}: {
  side: 'white' | 'black'
  symbols: string[]
  advantage: number
}) {
  // White captures black pieces, which must *look* black. The dark theme
  // renders text glyphs in the light ink color, so the filled ("black")
  // set reads as white pieces and the hollow ("white") set reads as black —
  // the perceived colors are inverted from the glyph names. Use the same-
  // name set so the rendered color matches the captured piece's color.
  const glyphColor = side
  return (
    <div className="captured-side" aria-label={`Captured by ${side}`}>
      <span className="captured-pieces-glyphs">
        {symbols.map((s, i) => (
          <span key={i} aria-hidden>
            {PIECE_GLYPHS[glyphColor][s as PieceType] ?? '?'}
          </span>
        ))}
      </span>
      {advantage > 0 && <span className="captured-advantage">{`+${advantage}`}</span>}
    </div>
  )
}

/**
 * Both players' captured pieces plus the material advantage of whoever is
 * ahead. Presentational — the backend derives captures from the move stack.
 */
export function CapturedPieces({ captured }: CapturedPiecesProps) {
  const diff = materialScore(captured.white) - materialScore(captured.black)
  return (
    <section className="captured" aria-label="Captured pieces">
      <Side side="white" symbols={captured.white} advantage={diff} />
      <Side side="black" symbols={captured.black} advantage={-diff} />
    </section>
  )
}
