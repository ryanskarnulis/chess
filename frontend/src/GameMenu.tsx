import { useEffect, useState } from 'react'
import { fetchPgn, type BrainSettings } from './api'
import { brainOption } from './brains'
import { COPY_IDLE, copyText } from './clipboard'
import { DIFFICULTY_LEVELS, isMoverTier } from './difficulty'
import { ChevronRightIcon } from './icons'
import { unlockAudio } from './tts'

export interface GameMenuProps {
  open: boolean
  onClose: () => void
  onNewGame: () => void
  /** Whether the rules allow a claim right now (`state.claimable_draws` is
   * non-empty): the draw row reads "Claim draw" then, otherwise it is the
   * offer the engine may accept or decline. */
  drawClaimable: boolean
  /** The game is over: there is nothing left to offer, claim or resign. */
  gameOver: boolean
  onDraw: () => void
  /** Ask to resign — the parent confirms before anything ends. */
  onResign: () => void
  /** Server-confirmed difficulty tier; null while settings load (or when the
   * strength was set outside the tiers). */
  tier: string | null
  /** Whether the `glitch` tier can be chosen; false disables it and says why,
   * null (unknown) leaves it to the server to refuse. */
  glitchAvailable: boolean | null
  onSetDifficulty: (tier: string) => void
  /** Whether replies are spoken aloud; null hides the switch until known. */
  voiceOutput: boolean | null
  onToggleVoice: (enabled: boolean) => void
  /** Glitch's brain setting; null (or no choices: a fixed brain, direct mode)
   * hides the picker. */
  brain: BrainSettings | null
  onSetBrain: (model: string) => void
}

/**
 * Bottom sheet behind the bar's Game button: everything that doesn't earn a
 * spot in the bar, in two groups. **Game** holds the lifecycle (new game,
 * PGN, draw, resign); **Glitch** holds his settings (difficulty, brain, voice).
 * The selects are server-confirmed — they never claim a strength the engine
 * isn't playing at, or a brain that was never chosen. Brains go by feel (Fast,
 * Balanced, …), not by model name. Voice cannot pick one: this is the only way
 * (#435). In the Glitch tier his brain also picks his moves (#475), so both
 * rows say so.
 */
export function GameMenu({
  open,
  onClose,
  onNewGame,
  drawClaimable,
  gameOver,
  onDraw,
  onResign,
  tier,
  glitchAvailable,
  onSetDifficulty,
  voiceOutput,
  onToggleVoice,
  brain,
  onSetBrain,
}: GameMenuProps) {
  const [copyLabel, setCopyLabel] = useState(COPY_IDLE)
  // A copy's result belongs to the visit it was made on.
  useEffect(() => {
    if (!open) setCopyLabel(COPY_IDLE)
  }, [open])
  if (!open) return null

  const isPreset = DIFFICULTY_LEVELS.some((l) => l.tier === tier)
  const brainHint = brain ? brainOption(brain.brain).hint : null
  const glitchTier = isMoverTier(tier)
  const noGlitch = glitchAvailable === false

  const copyPgn = async () => {
    const pgn = await fetchPgn()
    setCopyLabel(pgn === null ? 'PGN unavailable' : await copyText(pgn))
  }

  return (
    <>
      <div className="sheet-backdrop game-menu-backdrop" onClick={onClose} />
      <section className="sheet game-menu" role="dialog" aria-label="Game">
        <span className="sheet-handle" aria-hidden="true" />
        <h2 className="menu-group">Game</h2>
        <button
          type="button"
          className="menu-row"
          onClick={() => {
            onNewGame()
            onClose()
          }}
        >
          New game
        </button>
        <button type="button" className="menu-row" onClick={copyPgn}>
          {copyLabel}
        </button>
        <button
          type="button"
          className="menu-row"
          disabled={gameOver}
          onClick={() => {
            onDraw()
            onClose()
          }}
        >
          {drawClaimable ? 'Claim draw' : 'Offer draw'}
        </button>
        <button
          type="button"
          className="menu-row menu-danger"
          disabled={gameOver}
          onClick={() => {
            onClose()
            onResign()
          }}
        >
          Resign…
        </button>

        <h2 className="menu-group">Glitch</h2>
        <label className="menu-row menu-select">
          <span className="menu-row-text">
            <span>Difficulty</span>
            {glitchTier && (
              <span className="menu-row-hint difficulty-hint">
                Glitch picks his own moves, so his replies take a while
              </span>
            )}
          </span>
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
          <ChevronRightIcon />
        </label>
        {brain !== null && brain.choices.length > 0 && (
          <label className="menu-row menu-select">
            <span className="menu-row-text">
              <span>Brain</span>
              {brainHint !== null && (
                <span className="menu-row-hint brain-hint">{brainHint}</span>
              )}
              {glitchTier && (
                <span className="menu-row-hint brain-hint">
                  It also plays his moves here: a deeper brain plays stronger, and slower
                </span>
              )}
            </span>
            <select value={brain.brain} onChange={(e) => onSetBrain(e.target.value)}>
              {brain.choices.map((id) => (
                <option key={id} value={id}>
                  {brainOption(id).label}
                </option>
              ))}
            </select>
            <ChevronRightIcon />
          </label>
        )}
        {voiceOutput !== null && (
          <button
            type="button"
            className="menu-row menu-switch"
            role="switch"
            aria-checked={voiceOutput}
            onClick={() => {
              // Turning voice on is the gesture that primes playback on
              // mobile browsers — prime it while we have it.
              unlockAudio()
              onToggleVoice(!voiceOutput)
            }}
          >
            <span className="menu-row-text">Speak replies aloud</span>
            <span className="switch-track" aria-hidden="true">
              <span className="switch-thumb" />
            </span>
          </button>
        )}
      </section>
    </>
  )
}
