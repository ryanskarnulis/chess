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
  /** Which side the human plays; the engine owns the other. Drives board
   * orientation and what a takeback means. */
  player_color: 'white' | 'black'
  game_over: boolean
  outcome: Outcome | null
  history: string[]
  /** FEN of every position reached, root first, current last — one entry
   * per ply plus the root. Drives client-side history review. */
  fens: string[]
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
  /** Glitch's reaction to the dragged move — agent mode only. Absent in direct
   * mode (no brain configured), where a drag is a purely deterministic move. */
  commentary?: string
  /** Whether the client should voice that commentary (the user's voice_output
   * setting; the server decides, the client plays). Agent mode only. */
  speak?: boolean
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

/** A destructive op the backend's confirmation gate armed instead of running:
 * `detail` is the question to put to the player, `op` names what is armed
 * (`new_game` / `resign`). The same gate a spoken "new game" hits — and the
 * same one armed op, so either surface can answer it. */
export interface ConfirmQuestion {
  detail: string
  op: string
}

/** One gated lifecycle call: the new state when the op ran, or the gate's
 * question when it wants an answer first. Both null means the backend refused
 * for some other reason and nothing should move. */
export interface LifecycleOutcome {
  state: GameState | null
  question: ConfirmQuestion | null
}

/**
 * POST a lifecycle mutation that the destructive-op gate guards. A 409 carrying
 * `confirm: true` is a question, not a failure — the board is untouched and the
 * op is armed until `confirmDestructive` answers it.
 */
async function postGated(path: string, body: unknown = {}): Promise<LifecycleOutcome> {
  const res = await fetch(path, { ...JSON_POST, body: JSON.stringify(body) })
  if (!res.ok) {
    const data = (await res.json().catch(() => ({}))) as {
      detail?: string
      confirm?: boolean
      op?: string
    }
    if (res.status === 409 && data.confirm) {
      return { state: null, question: { detail: data.detail ?? '', op: data.op ?? '' } }
    }
    return { state: null, question: null }
  }
  const data = (await res.json()) as { state: GameState }
  return { state: data.state, question: null }
}

/** Start a fresh game. `color` is the side the player takes; omitted, the
 * backend rolls one at random. When the player takes black the engine's
 * opening move is already in the returned state. Mid-game the gate asks first:
 * the outcome carries its question instead of a state. */
export function newGame(color?: 'white' | 'black' | 'random'): Promise<LifecycleOutcome> {
  return postGated('/api/game/new', color ? { color } : {})
}

/**
 * Answer the armed destructive op: true runs it, false drops it. Returns the
 * authoritative state either way (unchanged on a cancel), or null if there was
 * nothing armed. Whichever way it is answered, nothing stays armed.
 */
export async function confirmDestructive(confirm: boolean): Promise<GameState | null> {
  const res = await fetch('/api/game/confirm', {
    ...JSON_POST,
    body: JSON.stringify({ confirm }),
  })
  if (!res.ok) return null
  const data = (await res.json()) as { state: GameState }
  return data.state
}

/** Take back moves. Without `plies` the backend applies the player's
 * takeback: the full exchange vs the engine, one ply engine-free. */
export function undo(plies?: number): Promise<GameState | null> {
  return postLifecycle('/api/game/undo', plies === undefined ? {} : { plies })
}

/** Resign. Gated like `newGame`: mid-game the outcome carries the gate's
 * question and the game only ends once it is answered. */
export function resign(color?: 'white' | 'black'): Promise<LifecycleOutcome> {
  return postGated('/api/game/resign', color ? { color } : {})
}

export interface DifficultyResponse {
  tier: string | null
  skill_level: number | null
  elo: number | null
}

/** Set engine strength by named tier (beginner … maximum). Returns the
 * applied setting, or null if the backend rejected it. Board state is
 * untouched. */
export async function setDifficulty(tier: string): Promise<DifficultyResponse | null> {
  const res = await fetch('/api/game/difficulty', {
    ...JSON_POST,
    body: JSON.stringify({ tier }),
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
  /** Whether the user wants the commentary voiced (their voice_output
   * setting — the server decides, the client plays). */
  speak: boolean
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

export interface Settings {
  verbosity: string
  hints_mode: boolean
  voice_output: boolean
  tier: string | null
  skill_level: number | null
  elo: number | null
  /** Whether a brain is configured at all. False is *direct mode* — Stockfish
   * only — which the UI shows as a state rather than letting the player find
   * out from a 503 on the command box. */
  agent_available?: boolean
}

/** Fetch the agent-adjustable settings (the same truth the tools mutate). */
export async function fetchSettings(): Promise<Settings> {
  const res = await fetch('/api/settings')
  return (await res.json()) as Settings
}

/** Turn voice output on/off directly (trusted UI path — the mute button
 * shouldn't need the LLM). Returns the confirmed setting, or null if the
 * backend refused. */
export async function setVoiceOutput(enabled: boolean): Promise<boolean | null> {
  const res = await fetch('/api/settings/voice', {
    ...JSON_POST,
    body: JSON.stringify({ enabled }),
  })
  if (!res.ok) return null
  const data = (await res.json()) as { voice_output: boolean }
  return data.voice_output
}

/**
 * Send recorded audio to the backend for transcription. Returns the
 * recognized text, or null when voice is unavailable (no speech service →
 * 503, speech backend down → 502). The caller feeds the text into the same
 * command pipeline as typed input — voice never gets its own path.
 */
export async function transcribe(audio: Blob, filename = 'clip.webm'): Promise<string | null> {
  const form = new FormData()
  // The filename extension tells the speech backend the container format
  // (webm from MediaRecorder push-to-talk, wav from the hands-free VAD).
  form.append('audio', audio, filename)
  const res = await fetch('/api/voice/transcribe', { method: 'POST', body: form })
  if (!res.ok) return null
  const data = (await res.json()) as { text: string }
  return data.text
}

export interface HintResponse {
  uci: string
  san: string
  from: string
  to: string
}

/** Fetch the engine's best move for the side to move, or null if the backend
 * refused (no engine → 503, game over / no moves → 409). Read-only — never
 * touches board state. */
export async function fetchHint(): Promise<HintResponse | null> {
  const res = await fetch('/api/game/hint')
  if (!res.ok) return null
  return (await res.json()) as HintResponse
}

export interface ReviewedMove {
  san: string
  uci: string
  color: 'white' | 'black'
  cp_loss: number
  classification: 'good' | 'inaccuracy' | 'mistake' | 'blunder'
  /** The engine's preferred move from the same position, in SAN. */
  best: string
  accuracy: number
}

export interface GameReview {
  moves: ReviewedMove[]
  /** Per-color accuracy percentage (0–100). */
  accuracy: Record<string, number>
  /** Per-color classification counts. */
  counts: Record<string, Record<string, number>>
}

/** Fetch the whole-game review, or null if the backend refused (no engine →
 * 503, no moves yet → 409). Read-only — never touches board state. */
export async function fetchReview(): Promise<GameReview | null> {
  const res = await fetch('/api/game/review')
  if (!res.ok) return null
  return (await res.json()) as GameReview
}

/** Fetch the game so far as PGN, or null if the backend refused. Read-only —
 * never touches board state. */
export async function fetchPgn(): Promise<string | null> {
  const res = await fetch('/api/game/pgn')
  if (!res.ok) return null
  const data = (await res.json()) as { pgn: string }
  return data.pgn
}

/**
 * One thing that happened inside one turn, live (`backend/.../progress.py`).
 * `name` is read according to `kind`: a tool name, a coordinator phase, or one
 * of the brain's two phases. `begin`/`end` bracket the turn and carry none.
 *
 * `correlation_id` identifies the interaction — the same id the backend's turn
 * trace records — which is what lets a client tell one turn's events from the
 * next one's, and ignore a late `end` for a turn it is no longer showing.
 */
export interface ProgressEvent {
  correlation_id: string
  turn_id: number
  kind: 'begin' | 'tool' | 'phase' | 'brain' | 'end'
  name: string
}

/** What arrives on the broadcast channel: the authoritative board document, or
 * an ephemeral note about the turn currently changing it. Discriminated by
 * `type` — a client that only wants the board ignores the rest. */
export type SocketMessage =
  | { type: 'state'; state: GameState }
  | { type: 'progress'; progress: ProgressEvent }

/** URL of the backend's one-way broadcast channel (state + live progress). */
export function stateSocketUrl(): string {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws'
  return `${proto}://${location.host}/ws`
}
