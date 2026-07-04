import type { PromotionPiece } from './promotion'

export interface PromotionPickerProps {
  /** Colour of the promoting pawn — picks which piece glyphs to show. */
  color: 'white' | 'black'
  /** Fired with the chosen piece's UCI letter. */
  onSelect: (piece: PromotionPiece) => void
  /** Fired when the user dismisses the picker without choosing. */
  onCancel: () => void
}

const PIECES: { piece: PromotionPiece; label: string }[] = [
  { piece: 'q', label: 'Queen' },
  { piece: 'r', label: 'Rook' },
  { piece: 'b', label: 'Bishop' },
  { piece: 'n', label: 'Knight' },
]

const GLYPHS: Record<'white' | 'black', Record<PromotionPiece, string>> = {
  white: { q: '♕', r: '♖', b: '♗', n: '♘' },
  black: { q: '♛', r: '♜', b: '♝', n: '♞' },
}

/**
 * Modal overlay letting the user pick the piece a promoting pawn becomes.
 * Purely presentational — the parent owns the pending move and submits the
 * chosen piece as the UCI suffix. Clicking the backdrop cancels.
 */
export function PromotionPicker({ color, onSelect, onCancel }: PromotionPickerProps) {
  return (
    <div
      className="promotion-overlay"
      role="dialog"
      aria-label="Choose promotion piece"
      onClick={onCancel}
    >
      <div className="promotion-picker" onClick={(e) => e.stopPropagation()}>
        {PIECES.map(({ piece, label }) => (
          <button
            key={piece}
            type="button"
            className="promotion-piece"
            aria-label={label}
            onClick={() => onSelect(piece)}
          >
            <span aria-hidden>{GLYPHS[color][piece]}</span>
          </button>
        ))}
      </div>
    </div>
  )
}
