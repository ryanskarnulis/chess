/** Glitch's brains, by feel rather than by model name (#435). The backend
 * owns the list (`brain_choices` in `/api/settings`) and the choice; this only
 * says what each one is like, with an honest hint of its speed — the numbers
 * are the measured turn times in docs/model-profiles.md. An id missing from
 * here (a brain the backend added later) still shows, under its own name. */
const BRAINS: Record<string, { label: string; hint: string }> = {
  'gemma-4-12b': { label: 'Fast', hint: 'Replies in a couple of seconds' },
  'gemma-4-26b-a4b': { label: 'Balanced', hint: 'Thinks first: about 5–10 seconds a turn' },
  'qwen38-27b': { label: 'Deep', hint: 'Thinks hardest: about half a minute a turn' },
}

export interface BrainOption {
  id: string
  label: string
  /** Null for a brain this map does not know: no hint beats a made-up one. */
  hint: string | null
}

export function brainOption(id: string): BrainOption {
  const known = BRAINS[id]
  return known ? { id, ...known } : { id, label: id, hint: null }
}
