// Client-side milestones for one interaction (#317): speech end → transcript →
// command → first board update → engine reply → reply audio → playback end.
//
// The server traces its half (the turn, and each speech round trip) keyed by
// the same interaction id; this is the half only the browser can see — when
// the board actually changed on screen, when the first audio actually played.
// The two are joined by id and never by clock: every mark here is an offset in
// this page's own monotonic clock (`performance.now()`) from the moment the
// interaction started, and a server timestamp is never subtracted from one.
//
// Best-effort from end to end. Nothing here can throw into the game, and the
// report goes out fire-and-forget; a page that closes early loses a record,
// never a move.

import type { GameState } from './api'

export type Mark =
  | 'stt_done'
  | 'command_sent'
  | 'first_board_update'
  | 'engine_reply'
  | 'command_done'
  | 'tts_requested'
  | 'tts_ready'
  | 'playback_started'
  | 'playback_ended'

export type Origin = 'voice' | 'typed'

/** How an interaction ended, as far as the player could hear. The playback
 * outcomes are `tts.ts`'s; `silent` is a reply that was not voiced (voice off,
 * or no words), `no_command` an utterance that never became a turn (nothing
 * transcribed, the agent unavailable, a stale board). */
export type Outcome =
  | 'ended'
  | 'error'
  | 'interrupted'
  | 'blocked'
  | 'no_audio'
  | 'timeout'
  | 'silent'
  | 'no_command'

/** An interaction still open this long is given up on and reported as
 * censored: its last mark is a lower bound on how long it really took. Above
 * the speech deadline (90 s) plus a slow turn, so only a lost one reaches it. */
export const CEILING_MS = 180_000

interface Interaction {
  id: string
  origin: Origin
  t0: number
  marks: Partial<Record<Mark, number>>
  correlationId: string | null
  /** The board as the command left it, for the two board marks. */
  fromVersion: number | null
  playerColor: GameState['player_color'] | null
  /** The player's side has moved since the command (the engine is to move). */
  playerMoved: boolean
  ceiling: ReturnType<typeof setTimeout>
}

const live = new Map<string, Interaction>()
/** The interaction whose command is in flight, which is the one a board
 * update can belong to. One at a time: the panel sends one command at once. */
let watching: string | null = null

export function newInteractionId(): string {
  const bytes = new Uint8Array(6)
  crypto.getRandomValues(bytes)
  return Array.from(bytes, (b) => b.toString(16).padStart(2, '0')).join('')
}

/** Open an interaction, starting its clock now. Voice starts at the VAD's
 * end-of-speech (or the push-to-talk stop); typed at submit. */
export function beginInteraction(origin: Origin): string {
  const id = newInteractionId()
  live.set(id, {
    id,
    origin,
    t0: performance.now(),
    marks: {},
    correlationId: null,
    fromVersion: null,
    playerColor: null,
    playerMoved: false,
    ceiling: setTimeout(() => finishInteraction(id, 'timeout', true), CEILING_MS),
  })
  return id
}

/** Record `name` for `id` — the first time only, since a milestone is when
 * something *first* happened. Unknown or finished ids are ignored. */
export function mark(id: string | undefined, name: Mark): void {
  const it = id === undefined ? undefined : live.get(id)
  if (!it || it.marks[name] !== undefined) return
  it.marks[name] = Math.max(0, Math.round(performance.now() - it.t0))
}

export function correlate(id: string | undefined, correlationId: string | undefined): void {
  const it = id === undefined ? undefined : live.get(id)
  if (it && correlationId) it.correlationId = correlationId
}

/** The command for `id` is going out against `state`: board updates from now
 * until `unwatchBoard` belong to it. */
export function watchBoard(id: string, state: GameState | null): void {
  const it = live.get(id)
  if (!it) return
  it.fromVersion = state?.version ?? null
  it.playerColor = state?.player_color ?? null
  it.playerMoved = false
  watching = id
}

export function unwatchBoard(id: string): void {
  if (watching === id) watching = null
}

/** Every accepted board state passes through here (`useGame.apply`). The
 * first newer board after the command is `first_board_update`; the engine's
 * reply is the first board where it is the player's move again *after* a
 * board where it was not — the player's move landed, then the answer did. A
 * turn whose intermediate board never reached this page (no socket frame)
 * records no engine mark rather than a guessed one. */
export function observeBoard(state: GameState): void {
  const it = watching === null ? undefined : live.get(watching)
  if (!it || it.fromVersion === null || state.version === undefined) return
  if (state.version <= it.fromVersion) return
  mark(it.id, 'first_board_update')
  const playerToMove = state.turn === it.playerColor
  if (!playerToMove) it.playerMoved = true
  else if (it.playerMoved) mark(it.id, 'engine_reply')
}

/** Close `id` and send its report. `censored` says it was given up on rather
 * than settled. Safe to call twice; the second is ignored. */
export function finishInteraction(id: string | undefined, outcome: Outcome, censored = false): void {
  const it = id === undefined ? undefined : live.get(id)
  if (!it) return
  live.delete(it.id)
  clearTimeout(it.ceiling)
  unwatchBoard(it.id)
  const report = {
    interaction_id: it.id,
    correlation_id: it.correlationId,
    origin: it.origin,
    clock: 'client_monotonic_ms',
    start: it.origin === 'voice' ? 'speech_end' : 'submit',
    marks: it.marks,
    outcome,
    censored,
  }
  try {
    void fetch('/api/telemetry/voice', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(report),
      // Outlives a page that is closing, which is when the last one is sent.
      keepalive: true,
    }).catch(() => {})
  } catch {
    // Diagnostics only.
  }
}

/** Test seam: forget every open interaction. */
export function resetInteractions(): void {
  for (const it of live.values()) clearTimeout(it.ceiling)
  live.clear()
  watching = null
}
