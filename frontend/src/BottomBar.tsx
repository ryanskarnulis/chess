import { BulbIcon, DrawIcon, FlagIcon, MenuIcon, UndoIcon } from './icons'

export interface BottomBarProps {
  onOptions: () => void
  onResign: () => void
  onDraw: () => void
  onHint: () => void
  onUndo: () => void
  resignDisabled: boolean
  /** Whether the rules allow a claim right now (`state.claimable_draws` is
   * non-empty). The button reads "Claim draw" then; otherwise it is the offer
   * the engine may accept or decline. */
  drawClaimable: boolean
  drawDisabled: boolean
  hintDisabled: boolean
  undoDisabled: boolean
}

/**
 * Fixed mobile control bar: Options / Resign / Draw / Hint / Undo. Purely
 * presentational — the parent owns every action.
 */
export function BottomBar({
  onOptions,
  onResign,
  onDraw,
  onHint,
  onUndo,
  resignDisabled,
  drawClaimable,
  drawDisabled,
  hintDisabled,
  undoDisabled,
}: BottomBarProps) {
  return (
    <nav className="bottom-bar" aria-label="Game controls">
      <button type="button" onClick={onOptions}>
        <MenuIcon />
        Options
      </button>
      <button type="button" onClick={onResign} disabled={resignDisabled}>
        <FlagIcon />
        Resign
      </button>
      <button type="button" onClick={onDraw} disabled={drawDisabled}>
        <DrawIcon />
        {drawClaimable ? 'Claim draw' : 'Offer draw'}
      </button>
      <button type="button" onClick={onHint} disabled={hintDisabled}>
        <BulbIcon />
        Hint
      </button>
      <button type="button" onClick={onUndo} disabled={undoDisabled}>
        <UndoIcon />
        Undo
      </button>
    </nav>
  )
}
