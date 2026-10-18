import type { PromotionPiece } from './promotion'

export interface PromotionPickerProps {
  /** Colour of the promoting pawn — picks which side's artwork to show. */
  color: 'white' | 'black'
  /** Fired with the chosen piece's UCI letter. */
  onSelect: (piece: PromotionPiece) => void
  /** Fired when the user dismisses the picker without choosing. */
  onCancel: () => void
}

const PIECES: { piece: PromotionPiece; label: string; role: string }[] = [
  { piece: 'q', label: 'Queen', role: 'queen' },
  { piece: 'r', label: 'Rook', role: 'rook' },
  { piece: 'b', label: 'Bishop', role: 'bishop' },
  { piece: 'n', label: 'Knight', role: 'knight' },
]

/**
 * Overlay that dims the board and lets the user pick the piece a promoting
 * pawn becomes, drawn in the board's own piece art (`.piece-art`, styled in
 * chessground.nightsilk.css). Purely presentational — the parent owns the
 * pending move and submits the chosen piece as the UCI suffix. Clicking
 * outside the tiles cancels.
 */
export function PromotionPicker({ color, onSelect, onCancel }: PromotionPickerProps) {
  return (
    <div
      className="promotion-overlay"
      role="dialog"
      aria-label="Choose promotion piece"
      onClick={onCancel}
    >
      <p className="promotion-caption">Promote to</p>
      <div className="promotion-picker" onClick={(e) => e.stopPropagation()}>
        {PIECES.map(({ piece, label, role }) => (
          <button
            key={piece}
            type="button"
            className="promotion-piece"
            aria-label={label}
            onClick={() => onSelect(piece)}
          >
            <span className={`piece-art ${role} ${color}`} aria-hidden="true" />
          </button>
        ))}
      </div>
      <p className="promotion-caption promotion-cancel">Tap outside to cancel</p>
    </div>
  )
}
