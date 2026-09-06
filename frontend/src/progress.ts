// Live turn progress, as words. The backend reports *what moved* — a
// coordinator phase, a tool about to run, the brain changing phase — and never
// what to say about it: phase names are machine truth, and the copy a player
// reads is the UI's business, same as every other string in here.
//
// The rule that keeps the line calm: only an event with something to say
// replaces the line. A turn passes through boundaries that are real states but
// not activities (`player_move_applied`, `completed`), and blanking the line
// for each of them would read as a flicker rather than as information.

import type { ProgressEvent } from './api'

/** What the player is told while a turn runs, and which turn it is about. */
export interface TurnProgress {
  /** The interaction being shown, or null when no turn is in flight. */
  correlationId: string | null
  /** The line to show, or null when the turn has not said anything specific. */
  label: string | null
}

export const NO_PROGRESS: TurnProgress = { correlationId: null, label: null }

// The coordinator's phases that are *waits* rather than boundaries. The ones
// left out are real states the machine passes through with nothing happening
// in them, so they deliberately map to nothing.
const PHASE_LABELS: Record<string, string> = {
  agent_observing: 'Glitch is reacting',
  engine_calculating: 'Stockfish is calculating',
}

const BRAIN_LABELS: Record<string, string> = {
  planning: 'Glitch is thinking',
  narrating: 'Glitch is reacting',
}

// Every tool the registry holds, in the player's language. A tool missing from
// here still shows something readable (see `toolLabel`) rather than nothing —
// the registry is free to grow without this map going stale and silent.
const TOOL_LABELS: Record<string, string> = {
  make_move: 'Validating your move',
  undo: 'Taking the move back',
  new_game: 'Starting a new game',
  resign: 'Resigning',
  claim_draw: 'Claiming the draw',
  save_game: 'Saving the game',
  resume_game: 'Loading the saved game',
  export_pgn: 'Exporting the PGN',
  get_best_moves: 'Asking the engine for candidates',
  evaluate_position: 'Evaluating the position',
  analyze_last_move: 'Analyzing your last move',
  review_game: 'Reviewing the game',
  get_board_state: 'Reading the board',
  describe_position: 'Describing the position',
  get_legal_moves: 'Reading your legal moves',
  get_move_history: 'Reading the move history',
  get_captured_pieces: 'Counting the captures',
  set_difficulty: 'Changing the difficulty',
  set_verbosity: 'Changing the verbosity',
  set_voice_output: 'Changing voice output',
}

function toolLabel(name: string): string {
  return TOOL_LABELS[name] ?? `Running ${name.replace(/_/g, ' ')}`
}

/** The line this event asks for, or null when it is a boundary rather than an
 * activity — `begin` and `end` included, since they bracket the turn rather
 * than describe it. */
export function progressLabel(event: ProgressEvent): string | null {
  switch (event.kind) {
    case 'tool':
      return toolLabel(event.name)
    case 'phase':
      return PHASE_LABELS[event.name] ?? null
    case 'brain':
      return BRAIN_LABELS[event.name] ?? null
    default:
      return null
  }
}

/**
 * Fold one event into what is on screen.
 *
 * `end` clears, but only its own turn's line: a late `end` from a turn that
 * has already been superseded must not blank the one running now. Everything
 * else adopts the turn it belongs to — a client that connected mid-turn never
 * saw the `begin` and should still show what it can — and keeps the previous
 * line when the event has nothing of its own to say.
 */
export function applyProgress(current: TurnProgress, event: ProgressEvent): TurnProgress {
  if (event.kind === 'end') {
    return event.correlation_id === current.correlationId ? NO_PROGRESS : current
  }
  const label = progressLabel(event)
  if (event.correlation_id !== current.correlationId) {
    return { correlationId: event.correlation_id, label }
  }
  return { correlationId: current.correlationId, label: label ?? current.label }
}
