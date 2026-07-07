import { DIFFICULTY_LEVELS } from './difficulty'

export interface GameControlsProps {
  /** No moves to take back — undo is disabled. */
  canUndo: boolean
  /** Game already finished — resign is disabled. */
  gameOver: boolean
  /** Server-confirmed difficulty tier; null while settings load (or when the
   * strength was set outside the tiers, e.g. by raw skill/elo). */
  tier: string | null
  onNewGame: () => void
  onUndo: () => void
  onResign: () => void
  onSetDifficulty: (tier: string) => void
}

/**
 * Buttons for the game lifecycle (new / undo / resign) plus a difficulty
 * selector. Purely presentational — the parent owns the backend calls. The
 * difficulty select is controlled by the server's settings: it shows a tier
 * only when the backend's skill level matches one, so it never claims a
 * strength the engine isn't actually playing at.
 */
export function GameControls({
  canUndo,
  gameOver,
  tier,
  onNewGame,
  onUndo,
  onResign,
  onSetDifficulty,
}: GameControlsProps) {
  const isPreset = DIFFICULTY_LEVELS.some((l) => l.tier === tier)
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
          value={isPreset && tier !== null ? tier : ''}
          onChange={(e) => onSetDifficulty(e.target.value)}
        >
          <option value="" disabled hidden>
            —
          </option>
          {DIFFICULTY_LEVELS.map(({ label, tier: value }) => (
            <option key={value} value={value}>
              {label}
            </option>
          ))}
        </select>
      </label>
    </section>
  )
}
