import { DIFFICULTY_LEVELS } from './difficulty'

export interface OptionsSheetProps {
  open: boolean
  onClose: () => void
  onNewGame: () => void
  /** Server-confirmed difficulty tier; null while settings load (or when the
   * strength was set outside the tiers). */
  tier: string | null
  onSetDifficulty: (tier: string) => void
  /** Whether replies are spoken aloud; null hides the toggle until known. */
  voiceOutput: boolean | null
  onToggleVoice: (enabled: boolean) => void
}

/**
 * Bottom sheet behind the mobile Options button: the lifecycle and settings
 * controls that don't earn a spot in the bar (new game, difficulty, voice).
 * Same server-confirmed semantics as GameControls — the select never claims
 * a strength the engine isn't playing at.
 */
export function OptionsSheet({
  open,
  onClose,
  onNewGame,
  tier,
  onSetDifficulty,
  voiceOutput,
  onToggleVoice,
}: OptionsSheetProps) {
  if (!open) return null
  const isPreset = DIFFICULTY_LEVELS.some((l) => l.tier === tier)
  return (
    <>
      <div className="options-backdrop" onClick={onClose} />
      <div className="options-sheet" role="dialog" aria-label="Options">
        <button
          type="button"
          onClick={() => {
            onNewGame()
            onClose()
          }}
        >
          New game
        </button>
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
        {voiceOutput !== null && (
          <button
            type="button"
            className="voice-toggle"
            aria-label={voiceOutput ? 'Turn voice output off' : 'Turn voice output on'}
            onClick={() => onToggleVoice(!voiceOutput)}
          >
            {voiceOutput ? '🔊 Voice on' : '🔇 Voice off'}
          </button>
        )}
        <button type="button" className="options-close" onClick={onClose}>
          Close
        </button>
      </div>
    </>
  )
}
