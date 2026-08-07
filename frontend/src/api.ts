// Thin typed client over the backend's game API. The backend is the single
// source of truth: this module fetches state, submits moves, and hands back
// whatever the server says — it never validates or generates moves itself.

export interface Outcome {
  termination: string
  winner: 'white' | 'black' | null
  result: string
}

export interface GameState {
  /** Monotonic board revision. Optional only for compatibility with an older
   * backend; once observed, it prevents older responses replacing newer state. */
  version?: number
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

/** A mutation rejected because another client already advanced the board.
 * The backend includes its current state so this client can catch up without
 * another fetch. */
export interface StaleStateResponse {
  stale: true
  version: number
  detail: string
  state: GameState
}

/** A move the backend never applied and sent no state for: the turn
 * coordinator's non-stale 409, a gateway error, or a transport failure that
 * produced no response at all. `detail` is the backend's own explanation when
 * it sent one, otherwise the app's — either way it is a line to show the
 * player, because a refused move broadcasts no state frame and nothing else
 * will mention it. */
export interface MoveFailure {
  failed: true
  detail: string
}

export type MoveOutcome = MoveResponse | StaleStateResponse | MoveFailure

function versioned(body: Record<string, unknown>, version?: number): Record<string, unknown> {
  return version === undefined ? body : { ...body, version }
}

export function isStaleStateResponse(data: unknown): data is StaleStateResponse {
  if (typeof data !== 'object' || data === null) return false
  const candidate = data as Partial<StaleStateResponse>
  return candidate.stale === true && typeof candidate.state === 'object' && candidate.state !== null
}

export function isMoveFailure(data: unknown): data is MoveFailure {
  if (typeof data !== 'object' || data === null) return false
  return (data as Partial<MoveFailure>).failed === true
}

/** Whether a parsed body actually carries the state a caller would apply. A
 * body that never parsed leaves `{}` behind, and its missing `state` must be
 * caught here rather than reaching the board as `undefined`. */
function carriesState(data: unknown): data is { state: GameState } {
  if (typeof data !== 'object' || data === null) return false
  const candidate = data as { state?: unknown }
  return typeof candidate.state === 'object' && candidate.state !== null
}

/** The backend's own explanation of a refusal, or null when the body has none
 * to relay. Nothing about the shape is assumed: a proxy error page never
 * parsed at all, and FastAPI's own `detail` is a list, not a string, when the
 * refusal is a validation error. */
function refusalDetail(data: unknown): string | null {
  if (typeof data !== 'object' || data === null) return null
  const detail = (data as { detail?: unknown }).detail
  return typeof detail === 'string' && detail !== '' ? detail : null
}

export async function fetchState(): Promise<GameState> {
  const res = await fetch('/api/state')
  return (await res.json()) as GameState
}

const MOVE_UNREACHABLE = 'Could not reach the server — the move was not played.'
const MOVE_UNUSABLE = 'The server sent an unusable response — the move was not played.'

/**
 * Submit a move. Never throws: this is the one mutation the board plays
 * optimistically — chessground has already moved the piece — so every way this
 * can fail has to come back as a value the caller can re-sync from (#231). A
 * `MoveFailure` means the move was not applied and no state came with it: the
 * coordinator's non-stale 409, a gateway error, a dead socket during a
 * redeploy. Only a `MoveResponse` carries a board to adopt.
 */
export async function submitMove(uci: string, version?: number): Promise<MoveOutcome> {
  let res: Response
  try {
    res = await fetch('/api/game/move', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(versioned({ move: uci }, version)),
    })
  } catch {
    return { failed: true, detail: MOVE_UNREACHABLE }
  }
  const data = (await res.json().catch(() => ({}))) as unknown
  if (!res.ok) {
    // A stale 409 is the one refusal that carries a board: it is the catch-up
    // state, and the caller adopts it rather than reporting anything.
    if (isStaleStateResponse(data)) return data
    return {
      failed: true,
      detail: refusalDetail(data) ?? `The server refused the move (${res.status}).`,
    }
  }
  return carriesState(data) ? (data as MoveResponse) : { failed: true, detail: MOVE_UNUSABLE }
}

const JSON_POST = { method: 'POST', headers: { 'Content-Type': 'application/json' } }

/**
 * POST a lifecycle mutation and return the authoritative resulting state (also
 * the catch-up state from a stale 409), or null if the backend refused it for
 * another reason. The board only advances on a state the server produced.
 */
async function postLifecycle(
  path: string,
  body: Record<string, unknown> = {},
  version?: number,
): Promise<GameState | null> {
  const res = await fetch(path, { ...JSON_POST, body: JSON.stringify(versioned(body, version)) })
  const data = (await res.json().catch(() => ({}))) as unknown
  if (!res.ok) return isStaleStateResponse(data) ? data.state : null
  return (data as { state: GameState }).state
}

/** A destructive op the backend's confirmation gate armed instead of running:
 * `detail` is the question to put to the player, `op` names what is armed
 * (`new_game` / `resign`). The same gate a spoken "new game" hits — and the
 * same one armed op, so either surface can answer it. */
export interface ConfirmQuestion {
  detail: string
  op: string
}

/** One gated lifecycle call: the new/catch-up state, or the gate's question
 * when it wants an answer first. Both null means the backend refused for some
 * other reason and nothing should move. */
export interface LifecycleOutcome {
  state: GameState | null
  question: ConfirmQuestion | null
}

/**
 * POST a lifecycle mutation that the destructive-op gate guards. A 409 carrying
 * `confirm: true` is a question, not a failure — the board is untouched and the
 * op is armed until `confirmDestructive` answers it.
 */
async function postGated(
  path: string,
  body: Record<string, unknown> = {},
  version?: number,
): Promise<LifecycleOutcome> {
  const res = await fetch(path, { ...JSON_POST, body: JSON.stringify(versioned(body, version)) })
  const data = (await res.json().catch(() => ({}))) as unknown
  if (!res.ok) {
    if (isStaleStateResponse(data)) return { state: data.state, question: null }
    const refused = data as {
      detail?: string
      confirm?: boolean
      op?: string
    }
    if (res.status === 409 && refused.confirm) {
      return {
        state: null,
        question: { detail: refused.detail ?? '', op: refused.op ?? '' },
      }
    }
    return { state: null, question: null }
  }
  return { state: (data as { state: GameState }).state, question: null }
}

/** Start a fresh game. `color` is the side the player takes; omitted, the
 * backend rolls one at random. When the player takes black the engine's
 * opening move is already in the returned state. Mid-game the gate asks first:
 * the outcome carries its question instead of a state. */
export function newGame(
  color?: 'white' | 'black' | 'random',
  version?: number,
): Promise<LifecycleOutcome> {
  return postGated('/api/game/new', color ? { color } : {}, version)
}

export interface ConfirmOutcome {
  state: GameState
  /** True when this is the catch-up state from a stale 409, rather than the
   * ordinary response to the player's answer. */
  stale: boolean
}

/**
 * Answer the armed destructive op: true runs it, false drops it. Returns the
 * authoritative state either way (unchanged on an ordinary cancel), flags the
 * catch-up state from a stale 409, or returns null for another refusal.
 * Whichever way a live question is answered, nothing stays armed.
 */
export async function confirmDestructive(
  confirm: boolean,
  version?: number,
): Promise<ConfirmOutcome | null> {
  const res = await fetch('/api/game/confirm', {
    ...JSON_POST,
    body: JSON.stringify(versioned({ confirm }, version)),
  })
  const data = (await res.json().catch(() => ({}))) as unknown
  if (!res.ok) {
    return isStaleStateResponse(data) ? { state: data.state, stale: true } : null
  }
  return { state: (data as { state: GameState }).state, stale: false }
}

/** Take back moves. Without `plies` the backend applies the player's
 * takeback: the full exchange vs the engine, one ply engine-free. */
export function undo(plies?: number, version?: number): Promise<GameState | null> {
  return postLifecycle('/api/game/undo', plies === undefined ? {} : { plies }, version)
}

/** Resign. Gated like `newGame`: mid-game the outcome carries the gate's
 * question and the game only ends once it is answered. */
export function resign(color?: 'white' | 'black', version?: number): Promise<LifecycleOutcome> {
  return postGated('/api/game/resign', color ? { color } : {}, version)
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
  /** The difficulty tier after the turn (null when strength was set outside
   * the tiers), so an agent-side set_difficulty reaches the selector.
   * Optional only for older backends that don't send it. */
  tier?: string | null
}

/**
 * Send a free-form command to the agent. Returns its commentary and state, the
 * catch-up state from a stale 409, or null if the agent is unavailable (e.g. no
 * brain is configured → 503). The backend is still the sole move-truth source:
 * the agent acts only through tools, so a command can never corrupt state.
 */
export async function sendCommand(
  text: string,
  version?: number,
): Promise<CommandResponse | StaleStateResponse | null> {
  const res = await fetch('/api/command', {
    ...JSON_POST,
    body: JSON.stringify(versioned({ text }, version)),
  })
  const data = (await res.json().catch(() => ({}))) as unknown
  if (!res.ok) return isStaleStateResponse(data) ? data : null
  return data as CommandResponse
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
  /** The board this hint was computed for — the same counter `GameState.version`
   * carries. A search takes real time, so the caller must check this against the
   * live board before drawing anything (#218). */
  version: number
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
