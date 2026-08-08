import { BulbIcon, FlagIcon, MenuIcon, UndoIcon } from './icons'

export interface BottomBarProps {
  onOptions: () => void
  onResign: () => void
  onHint: () => void
  onUndo: () => void
  resignDisabled: boolean
  hintDisabled: boolean
  undoDisabled: boolean
}

/**
 * Fixed mobile control bar: Options / Resign / Hint / Undo. Purely
 * presentational — the parent owns every action.
 */
export function BottomBar({
  onOptions,
  onResign,
  onHint,
  onUndo,
  resignDisabled,
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
