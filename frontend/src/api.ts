// Thin typed client over the backend's game API. The backend is the single
// source of truth: this module fetches state, submits moves, and hands back
// whatever the server says — it never validates or generates moves itself.

export interface Outcome {
  termination: string
  winner: 'white' | 'black' | null
  result: string
}

export interface GameState {
  fen: string
  turn: 'white' | 'black'
  game_over: boolean
  outcome: Outcome | null
  history: string[]
  captured: { white: string[]; black: string[] }
  legal_moves: string[]
  /** Legal destinations by origin square, e.g. `{ e2: ['e3', 'e4'] }`. */
  dests: Record<string, string[]>
}

export interface MoveResponse {
  legal: boolean
  san: string | null
  uci: string | null
  reason: string | null
  engine_move: { legal: boolean; san: string | null; uci: string | null } | null
  /** The authoritative state after the attempt — unchanged if the move was illegal. */
  state: GameState
}

export async function fetchState(): Promise<GameState> {
  const res = await fetch('/api/state')
  return (await res.json()) as GameState
}

export async function submitMove(uci: string): Promise<MoveResponse> {
  const res = await fetch('/api/game/move', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ move: uci }),
  })
  return (await res.json()) as MoveResponse
}

const JSON_POST = { method: 'POST', headers: { 'Content-Type': 'application/json' } }

/**
 * POST a lifecycle mutation and return the authoritative new state, or null
 * if the backend refused it (e.g. 409: nothing to undo, resigning a finished
 * game). The board only advances on a state the server actually produced.
 */
async function postLifecycle(path: string, body: unknown = {}): Promise<GameState | null> {
  const res = await fetch(path, { ...JSON_POST, body: JSON.stringify(body) })
  if (!res.ok) return null
  const data = (await res.json()) as { state: GameState }
  return data.state
}

export function newGame(): Promise<GameState | null> {
  return postLifecycle('/api/game/new')
}

export function undo(plies = 1): Promise<GameState | null> {
  return postLifecycle('/api/game/undo', { plies })
}

export function resign(color?: 'white' | 'black'): Promise<GameState | null> {
  return postLifecycle('/api/game/resign', color ? { color } : {})
}

export interface DifficultyResponse {
  skill_level: number | null
  elo: number | null
}

/** Set engine strength by Stockfish skill level (0–20). Returns the applied
 * setting, or null if the backend rejected it. Board state is untouched. */
export async function setDifficulty(skillLevel: number): Promise<DifficultyResponse | null> {
  const res = await fetch('/api/game/difficulty', {
    ...JSON_POST,
    body: JSON.stringify({ skill_level: skillLevel }),
  })
  if (!res.ok) return null
  return (await res.json()) as DifficultyResponse
}

export interface CommandResponse {
  /** The agent's free-form reply to show the user. */
  commentary: string
  /** What each dispatched tool returned (opaque to the UI for now). */
  tool_results: { name: string; result: unknown }[]
  /** The authoritative state after any tools ran. */
  state: GameState
}

/**
 * Send a free-form command to the agent. Returns the agent's commentary plus
 * the authoritative new state, or null if the agent is unavailable (e.g. no
 * brain is configured → 503). The backend is still the sole move-truth source:
 * the agent acts only through tools, so a command can never corrupt state.
 */
export async function sendCommand(text: string): Promise<CommandResponse | null> {
  const res = await fetch('/api/command', { ...JSON_POST, body: JSON.stringify({ text }) })
  if (!res.ok) return null
  return (await res.json()) as CommandResponse
}

/** URL of the backend's one-way state broadcast channel. */
export function stateSocketUrl(): string {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws'
  return `${proto}://${location.host}/ws`
}
