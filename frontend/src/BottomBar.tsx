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
        <span aria-hidden="true">☰</span>
        Options
      </button>
      <button type="button" onClick={onResign} disabled={resignDisabled}>
        <span aria-hidden="true">🏳</span>
        Resign
      </button>
      <button type="button" onClick={onHint} disabled={hintDisabled}>
        <span aria-hidden="true">💡</span>
        Hint
      </button>
      <button type="button" onClick={onUndo} disabled={undoDisabled}>
        <span aria-hidden="true">↩</span>
        Undo
      </button>
    </nav>
  )
}
