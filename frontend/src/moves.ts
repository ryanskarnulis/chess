// Pure helpers for presenting move history. No chess logic — the backend
// already hands us the moves in SAN; this only groups them for display.

export interface MovePair {
  /** 1-based move number (one per white/black pair). */
  number: number
  /** White's move in SAN. */
  white: string
  /** Black's reply in SAN, or null when white has just moved. */
  black: string | null
}

/** Group a flat ply list (`['e4','e5','Nf3']`) into numbered move pairs. */
export function pairMoves(history: string[]): MovePair[] {
  const pairs: MovePair[] = []
  for (let i = 0; i < history.length; i += 2) {
    pairs.push({
      number: i / 2 + 1,
      white: history[i],
      black: history[i + 1] ?? null,
    })
  }
  return pairs
}
