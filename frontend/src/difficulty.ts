/** Difficulty presets, mapped to Stockfish `Skill Level` (0–20). The UI
 * speaks in named tiers; the backend takes the raw skill level and owns the
 * default — the select reflects whatever `/api/settings` reports. */
export const DIFFICULTY_LEVELS = [
  { label: 'Beginner', skillLevel: 1 },
  { label: 'Casual', skillLevel: 5 },
  { label: 'Intermediate', skillLevel: 10 },
  { label: 'Advanced', skillLevel: 15 },
  { label: 'Maximum', skillLevel: 20 },
] as const
