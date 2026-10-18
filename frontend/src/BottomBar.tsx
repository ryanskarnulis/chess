import { BulbIcon, MenuIcon, UndoIcon } from './icons'

export interface BottomBarProps {
  onUndo: () => void
  onHint: () => void
  /** Open (or close) the Game menu, which holds everything else. */
  onGame: () => void
  undoDisabled: boolean
  hintDisabled: boolean
  /** The Game menu is showing — the button lights and says so. */
  gameOpen: boolean
}

/**
 * Fixed mobile control bar: Undo / Hint / Game. Only the two actions a player
 * reaches for mid-game get a button; resigning, the draw and the settings live
 * in the Game menu. Purely presentational — the parent owns every action.
 */
export function BottomBar({
  onUndo,
  onHint,
  onGame,
  undoDisabled,
  hintDisabled,
  gameOpen,
}: BottomBarProps) {
  return (
    <nav className="bottom-bar" aria-label="Game controls">
      <button type="button" onClick={onUndo} disabled={undoDisabled}>
        <UndoIcon />
        Undo
      </button>
      <button type="button" onClick={onHint} disabled={hintDisabled}>
        <BulbIcon />
        Hint
      </button>
      <button
        type="button"
        className={gameOpen ? 'bottom-bar-lit' : undefined}
        aria-expanded={gameOpen}
        onClick={onGame}
      >
        <MenuIcon />
        Game
      </button>
    </nav>
  )
}
