import { DEFAULT_SKILL_LEVEL, DIFFICULTY_LEVELS } from './difficulty'

export interface GameControlsProps {
  /** No moves to take back — undo is disabled. */
  canUndo: boolean
  /** Game already finished — resign is disabled. */
  gameOver: boolean
  onNewGame: () => void
  onUndo: () => void
  onResign: () => void
  onSetDifficulty: (skillLevel: number) => void
}

/**
 * Buttons for the game lifecycle (new / undo / resign) plus a difficulty
 * selector. Purely presentational — the parent owns the backend calls. The
 * difficulty select is uncontrolled: it starts at a sensible default and only
 * emits on user change (the backend defaults to full strength until set).
 */
export function GameControls({
  canUndo,
  gameOver,
  onNewGame,
  onUndo,
  onResign,
  onSetDifficulty,
}: GameControlsProps) {
  return (
    <section className="game-controls" aria-label="Game controls">
      <div className="control-buttons">
        <button type="button" onClick={onNewGame}>
          New game
        </button>
        <button type="button" onClick={onUndo} disabled={!canUndo}>
          Undo
        </button>
        <button type="button" onClick={onResign} disabled={gameOver}>
          Resign
        </button>
      </div>
      <label className="difficulty">
        Difficulty
        <select
          defaultValue={DEFAULT_SKILL_LEVEL}
          onChange={(e) => onSetDifficulty(Number(e.target.value))}
        >
          {DIFFICULTY_LEVELS.map(({ label, skillLevel }) => (
            <option key={skillLevel} value={skillLevel}>
              {label}
            </option>
          ))}
        </select>
      </label>
    </section>
  )
}
