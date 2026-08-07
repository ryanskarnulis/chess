import { groupCaptures, materialScore, PIECE_GLYPHS, PIECE_NAMES, type PieceType } from './pieces'

export interface CapturedPiecesProps {
  /**
   * Pieces each colour has captured, in capture order (from the backend).
   * `white` holds the black pieces White took, and vice versa.
   */
  captured: { white: string[]; black: string[] }
  /** Which side the player drives — the row shows whichever colour's
   * captures belong to `owner`. */
  playerColor: 'white' | 'black'
  /** Whose row this is. Each owner gets a full-width row of their own:
   * Glitch's under the agent bubble, the player's under the board. */
  owner: 'you' | 'glitch'
}

/** "2 pawns, 1 knight" — the glyph run, spelled out for a screen reader. */
function spell(groups: { symbol: string; count: number }[]): string {
  return groups
    .map(({ symbol, count }) => {
      const name = PIECE_NAMES[symbol as PieceType] ?? 'piece'
      return `${count} ${count === 1 ? name : `${name}s`}`
    })
    .join(', ')
}

/**
 * One owner's captured pieces: a left-aligned row of glyphs grouped by piece
 * type with a ×N count, plus the material badge when this side is the one
 * ahead. Presentational — the backend derives captures from the move stack.
 *
 * The row survives with nothing in it. Both capture rows hold their height
 * from move one so the board doesn't jump on the game's first capture.
 */
export function CapturedPieces({ captured, playerColor, owner }: CapturedPiecesProps) {
  const opponentColor = playerColor === 'white' ? 'black' : 'white'
  // The player's row shows what the player took, Glitch's what Glitch took.
  const side = owner === 'you' ? playerColor : opponentColor
  const symbols = captured[side]

  // White captures black pieces, which must *look* black. The dark theme
  // renders text glyphs in the light ink color, so the filled ("black")
  // set reads as white pieces and the hollow ("white") set reads as black —
  // the perceived colors are inverted from the glyph names. Use the
  // capturer's own name so the rendered color matches the captured piece's.
  const glyphColor = side
  const groups = groupCaptures(symbols)

  const diff = materialScore(captured.white) - materialScore(captured.black)
  const advantage = side === 'white' ? diff : -diff

  const who = owner === 'you' ? 'Captured by you' : 'Captured by Glitch'
  const label = groups.length === 0 ? `${who}: nothing` : `${who}: ${spell(groups)}`

  return (
    <div
      className="captured-row"
      aria-label={advantage > 0 ? `${label}, up ${advantage}` : label}
    >
      {groups.length > 0 && (
        <>
          {/* Hidden wholesale: read one glyph at a time a screen reader gets
              "unknown character, ×5" — the row's own label says it properly. */}
          <span className="captured-pieces-glyphs" aria-hidden>
            {groups.map(({ symbol, count }) => (
              <span key={symbol} className="captured-glyph">
                {PIECE_GLYPHS[glyphColor][symbol as PieceType] ?? '?'}
                {/* A lone capture carries no count — "×1" is noise. */}
                {count > 1 && <sup className="captured-count">{`×${count}`}</sup>}
              </span>
            ))}
          </span>
          {advantage > 0 && <span className="captured-advantage">{`+${advantage}`}</span>}
        </>
      )}
    </div>
  )
}
