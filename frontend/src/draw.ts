/**
 * The one line the UI says for a draw offer's answer, composed from the two
 * fields the backend's rule reports (`docs/draw-offer.md`): `accepted`, and
 * the machine-readable `reason` for a decline. Deterministic text, no model —
 * the button path has none behind it, exactly like the resign and new-game
 * buttons — so this is app-owned wording, not Glitch's, and it never claims a
 * fact the result does not carry.
 */

export type DrawOfferReason = 'too_early' | 'engine_ahead' | 'player_ahead' | 'not_an_endgame'

const DECLINED: Record<DrawOfferReason, string> = {
  engine_ahead: 'Draw declined — Glitch is ahead. Play on.',
  player_ahead: 'Draw declined — you’re the one ahead. Play it out.',
  not_an_endgame: 'Draw declined — too much still on the board.',
  too_early: 'Draw declined — make a move first.',
}

export function drawAnswer(accepted: boolean, reason: string | null): string {
  if (accepted) return 'Draw agreed — half a point each.'
  // A reason this build has never heard of still gets a truthful line: the
  // backend's table is free to grow without this one going silent.
  return (reason && DECLINED[reason as DrawOfferReason]) || 'Draw declined. Play on.'
}
