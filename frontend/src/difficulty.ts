/** Difficulty presets, mapped to Stockfish `Skill Level` (0–20). The UI
 * speaks in named tiers; the backend takes the raw skill level. */
export const DIFFICULTY_LEVELS = [
  { label: 'Beginner', skillLevel: 1 },
  { label: 'Casual', skillLevel: 5 },
  { label: 'Intermediate', skillLevel: 10 },
  { label: 'Advanced', skillLevel: 15 },
  { label: 'Maximum', skillLevel: 20 },
] as const

/** Where the difficulty select starts before the user picks a tier. */
export const DEFAULT_SKILL_LEVEL = 10
