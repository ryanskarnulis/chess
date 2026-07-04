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

/** URL of the backend's one-way state broadcast channel. */
export function stateSocketUrl(): string {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws'
  return `${proto}://${location.host}/ws`
}
