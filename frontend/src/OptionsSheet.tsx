import type { BrainSettings } from './api'
import { brainOption } from './brains'
import { DIFFICULTY_LEVELS, isMoverTier } from './difficulty'
import { SpeakerOffIcon, SpeakerOnIcon } from './icons'

export interface OptionsSheetProps {
  open: boolean
  onClose: () => void
  onNewGame: () => void
  /** Server-confirmed difficulty tier; null while settings load (or when the
   * strength was set outside the tiers). */
  tier: string | null
  /** Whether the `glitch` tier can be chosen; false disables it and says why,
   * null (unknown) leaves it to the server to refuse. */
  glitchAvailable: boolean | null
  onSetDifficulty: (tier: string) => void
  /** Whether replies are spoken aloud; null hides the toggle until known. */
  voiceOutput: boolean | null
  onToggleVoice: (enabled: boolean) => void
  /** Glitch's brain setting; null (or no choices: a fixed brain, direct mode)
   * hides the picker. */
  brain: BrainSettings | null
  onSetBrain: (model: string) => void
}

/**
 * Bottom sheet behind the Options button: the lifecycle and settings
 * controls that don't earn a spot in the bar (new game, difficulty, brain,
 * voice). The selects are server-confirmed — they never claim a strength the
 * engine isn't playing at, or a brain that was never chosen. Brains go by
 * feel (Fast, Balanced, …), not by model name. Voice cannot pick one: this is
 * the only way (#435). In the Glitch tier his brain also picks his moves
 * (#475), so both selects say so.
 */
export function OptionsSheet({
  open,
  onClose,
  onNewGame,
  tier,
  glitchAvailable,
  onSetDifficulty,
  voiceOutput,
  onToggleVoice,
  brain,
  onSetBrain,
}: OptionsSheetProps) {
  if (!open) return null
  const isPreset = DIFFICULTY_LEVELS.some((l) => l.tier === tier)
  const brainHint = brain ? brainOption(brain.brain).hint : null
  const glitchTier = isMoverTier(tier)
  const noGlitch = glitchAvailable === false
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
            {DIFFICULTY_LEVELS.map(({ label, tier: value, mover }) => (
              <option key={value} value={value} disabled={mover && noGlitch}>
                {mover && noGlitch ? `${label} (needs a brain)` : label}
              </option>
            ))}
          </select>
        </label>
        {glitchTier && (
          <p className="difficulty-hint">Glitch picks his own moves, so his replies take a while</p>
        )}
        {brain !== null && brain.choices.length > 0 && (
          <div className="brain-picker">
            <label className="difficulty">
              Brain
              <select value={brain.brain} onChange={(e) => onSetBrain(e.target.value)}>
                {brain.choices.map((id) => (
                  <option key={id} value={id}>
                    {brainOption(id).label}
                  </option>
                ))}
              </select>
            </label>
            {brainHint !== null && <p className="brain-hint">{brainHint}</p>}
            {glitchTier && (
              <p className="brain-hint">
                It also plays his moves here: a deeper brain plays stronger, and slower
              </p>
            )}
          </div>
        )}
        {voiceOutput !== null && (
          <button
            type="button"
            className="voice-toggle voice-row"
            aria-label={voiceOutput ? 'Turn voice output off' : 'Turn voice output on'}
            onClick={() => onToggleVoice(!voiceOutput)}
          >
            {voiceOutput ? <SpeakerOnIcon /> : <SpeakerOffIcon />}
            {voiceOutput ? 'Voice on' : 'Voice off'}
          </button>
        )}
        <button type="button" className="options-close" onClick={onClose}>
          Close
        </button>
      </div>
    </>
  )
}
