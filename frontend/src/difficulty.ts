/** Difficulty presets, mapped to the backend's named tiers (which own the
 * real strength mechanics — UCI options plus a weakening layer for the
 * sub-Stockfish-floor levels). The UI speaks in labels; the backend owns the
 * default — the select reflects whatever `/api/settings` reports.
 *
 * `mover` marks the tier where Glitch's brain picks his moves instead of
 * Stockfish (#336): it needs a brain, and his replies take a model call. */
export const DIFFICULTY_LEVELS = [
  { label: 'Beginner', tier: 'beginner', mover: false },
  { label: 'Casual', tier: 'casual', mover: false },
  { label: 'Intermediate', tier: 'intermediate', mover: false },
  { label: 'Advanced', tier: 'advanced', mover: false },
  { label: 'Maximum', tier: 'maximum', mover: false },
  { label: 'Glitch', tier: 'glitch', mover: true },
] as const

export type DifficultyTier = (typeof DIFFICULTY_LEVELS)[number]['tier']

/** Whether Glitch picks his own moves at `tier` (the backend's `MOVER_TIERS`). */
export function isMoverTier(tier: string | null): boolean {
  return DIFFICULTY_LEVELS.some((l) => l.tier === tier && l.mover)
}
