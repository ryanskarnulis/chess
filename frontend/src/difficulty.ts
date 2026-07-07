/** Difficulty presets, mapped to the backend's named tiers (which own the
 * real strength mechanics — UCI options plus a weakening layer for the
 * sub-Stockfish-floor levels). The UI speaks in labels; the backend owns the
 * default — the select reflects whatever `/api/settings` reports. */
export const DIFFICULTY_LEVELS = [
  { label: 'Beginner', tier: 'beginner' },
  { label: 'Casual', tier: 'casual' },
  { label: 'Intermediate', tier: 'intermediate' },
  { label: 'Advanced', tier: 'advanced' },
  { label: 'Maximum', tier: 'maximum' },
] as const

export type DifficultyTier = (typeof DIFFICULTY_LEVELS)[number]['tier']
