import { useCallback, useEffect, useRef, useState } from 'react'
import {
  confirmDestructive as apiConfirmDestructive,
  fetchHint,
  fetchSettings,
  fetchState,
  isMoveFailure,
  isStaleStateResponse,
  newGame as apiNewGame,
  resign as apiResign,
  sendCommand as apiSendCommand,
  setDifficulty as apiSetDifficulty,
  setVoiceOutput as apiSetVoiceOutput,
  stateSocketUrl,
  submitMove,
  undo as apiUndo,
  type ConfirmQuestion,
  type GameState,
  type LifecycleOutcome,
  type SocketMessage,
} from './api'
import { NO_PROGRESS, applyProgress, type TurnProgress } from './progress'
import { isPromotion, type PromotionPiece } from './promotion'
import { playText } from './tts'

// A brief outage should heal quickly without hammering a restarting backend.
// Successful opens reset the sequence; repeated connection failures climb to
// a fixed ceiling: 1s, 2s, 4s, 8s, 10s, 10s …
const SOCKET_RETRY_BASE_MS = 1_000
const SOCKET_RETRY_MAX_MS = 10_000

/**
 * A promotion the player has proposed but not yet completed: the squares they
 * dragged, plus the board that drag was made against. The board is the half
 * the picker used to leave out — squares alone cannot say which position they
 * were a legal, intended move on (#222).
 */
interface ArmedPromotion {
  from: string
  to: string
  /** Version of the board dragged on; undefined on a pre-versioning backend. */
  version: number | undefined
  fen: string
}

/**
 * Whether an armed promotion still belongs to `board`. The version is the
 * app's identity for "which board" everywhere else (the hint binding, the
 * backend's armed destructive ops), and the fen is what `isPromotion` actually
 * read — redundant under versioning, but the only evidence there is on an
 * older backend that sends none, where two undefined versions would otherwise
 * compare equal to every position forever.
 */
function armedForBoard(armed: ArmedPromotion, board: GameState): boolean {
  return armed.version === board.version && armed.fen === board.fen
}

export interface UseGame {
  /** Latest authoritative game state, or null until the first load. */
  state: GameState | null
  /** Set when the backend rejects a move; cleared on the next legal move. */
  moveError: string | null
  /**
   * Bumped every time state is (re)applied. A rejected move leaves the
   * position unchanged, so `fen` alone wouldn't tell the board to snap the
   * piece back — the revision does.
   */
  revision: number
  /** Submit a user move (origin/dest squares) to the backend. */
  play: (from: string, to: string) => Promise<void>
  /**
   * Set when the last move was a pawn reaching the last rank: the move is
   * held until the user picks a piece. The board has already moved the pawn
   * visually, so a cancel must re-sync it back. Back to null — closing the
   * picker — as soon as authoritative state for a different board arrives: the
   * held move was a proposal about the position that is now gone.
   */
  pendingPromotion: { from: string; to: string } | null
  /** Finish the held promotion with the chosen piece. */
  completePromotion: (piece: PromotionPiece) => Promise<void>
  /** Abandon the held promotion and snap the pawn back. */
  cancelPromotion: () => void
  /** Start a fresh game. `color` is the side the player takes; omitted,
   * the backend rolls one at random. */
  newGame: (color?: 'white' | 'black' | 'random') => Promise<void>
  /** Take back the last ply. No-op if there is nothing to undo. */
  undo: () => Promise<void>
  /** Resign the game (the side to move, unless the backend decides otherwise). */
  resign: () => Promise<void>
  /** Set engine strength by named tier (beginner … maximum). */
  setDifficulty: (tier: string) => Promise<void>
  /** Server-confirmed difficulty tier; null until settings load (or when the
   * strength was last set outside the tiers, e.g. by raw skill/elo). */
  tier: string | null
  /** The agent's latest commentary, or null before the first command — a
   * dragged move in agent mode sets it too, from the reaction the backend
   * returned with the move. */
  commentary: string | null
  /** Whether a brain is configured; null until settings load. False is direct
   * mode: Stockfish only, and the command box is a designed dead state rather
   * than a 503 waiting to happen. */
  agentAvailable: boolean | null
  /** True while a command is in flight with the agent. */
  agentThinking: boolean
  /**
   * What the current turn is doing right now ("Stockfish is calculating"), or
   * null between turns. Fed by the backend's live progress events, which are
   * broadcast — so it fills in for a dragged move and for a turn another
   * client started, neither of which `agentThinking` knows about.
   */
  agentProgress: string | null
  /** Send a free-form command to the agent. */
  sendCommand: (text: string) => Promise<void>
  /** Whether agent replies are spoken aloud; null until settings load. */
  voiceOutput: boolean | null
  /** Turn voice output on/off (the UI mute toggle). */
  setVoiceOutput: (enabled: boolean) => Promise<void>
  /**
   * Index into `state.fens` of the position being reviewed, or null when
   * showing the live game. Review is client-side only — it never mutates
   * the game, and any authoritative state (a move, an agent action, another
   * tab) snaps back to live.
   */
  viewPly: number | null
  /** True while browsing history (viewPly !== null). */
  reviewing: boolean
  /** The FEN the board should render: the reviewed position, or live. */
  displayFen: string | null
  /** Step one position back through history (clamped at the root). */
  stepBack: () => void
  /** Step one position forward; reaching the latest resumes live view. */
  stepForward: () => void
  /** Hint arrow(s) for the board, from the last requestHint call. Cleared
   * whenever a new authoritative state applies. */
  hintShapes: { orig: string; dest: string; brush: string }[]
  /** Ask the engine for its best move and show it as an arrow. No-op while
   * reviewing; silently does nothing when the backend refuses. */
  requestHint: () => Promise<void>
}

/**
 * Owns the connection to the backend game: loads the initial position, keeps
 * it live over the WebSocket broadcast, and submits moves. All truth lives on
 * the server — this hook only ferries it in and out.
 */
export function useGame(): UseGame {
  const [state, setState] = useState<GameState | null>(null)
  const [moveError, setMoveError] = useState<string | null>(null)
  const [revision, setRevision] = useState(0)
  const [pendingPromotion, setPendingPromotion] = useState<{ from: string; to: string } | null>(
    null,
  )
  const [commentary, setCommentary] = useState<string | null>(null)
  const [agentThinking, setAgentThinking] = useState(false)
  const [progress, setProgress] = useState<TurnProgress>(NO_PROGRESS)
  const [agentAvailable, setAgentAvailable] = useState<boolean | null>(null)
  const [voiceOutput, setVoiceOutputState] = useState<boolean | null>(null)
  const [tier, setTierState] = useState<string | null>(null)
  const [viewPly, setViewPly] = useState<number | null>(null)
  const [hintShapes, setHintShapes] = useState<
    { orig: string; dest: string; brush: string }[]
  >([])
  // Latest state without making the move callbacks depend on it — the board
  // holds `play` in a ref, but promotion detection still needs the live fen.
  const stateRef = useRef<GameState | null>(null)
  // Once a versioned backend has been observed, state may only move forward.
  // This is a ref because fetch, mutation, and WebSocket callbacks race outside
  // React's render cycle. It stays null for an older backend, whose unversioned
  // documents retain the pre-versioning last-arrival-wins behavior.
  const highestVersionRef = useRef<number | null>(null)
  // Mirror of viewPly for the same reason: `play` must see the live value.
  const viewPlyRef = useRef<number | null>(null)
  // The armed promotion with its board stamp. A ref because `apply` has to see
  // it: `apply` is the socket effect's one dependency and must stay
  // identity-stable, so it cannot close over the rendered value.
  const pendingPromotionRef = useRef<ArmedPromotion | null>(null)

  const setView = useCallback((ply: number | null) => {
    viewPlyRef.current = ply
    setViewPly(ply)
  }, [])

  /** Arm (or disarm) the promotion picker. The ref carries the stamp the logic
   * checks; the rendered value stays the bare squares the UI needs. */
  const armPromotion = useCallback((armed: ArmedPromotion | null) => {
    pendingPromotionRef.current = armed
    setPendingPromotion(armed === null ? null : { from: armed.from, to: armed.to })
  }, [])

  const apply = useCallback((next: GameState) => {
    const highest = highestVersionRef.current
    if (next.version === undefined) {
      if (highest !== null) return false
    } else {
      if (highest !== null && next.version < highest) return false
      highestVersionRef.current = next.version
    }
    stateRef.current = next
    setState(next)
    setRevision((r) => r + 1)
    // Any authoritative state ends a history review and invalidates the
    // hint arrow — both were drawn for a position that may no longer exist.
    viewPlyRef.current = null
    setViewPly(null)
    setHintShapes([])
    // A held promotion is a proposal about one board (#222). State for a
    // *different* board voids it: completing it would either bounce back as an
    // illegal move or — where that pawn can still promote — play a move on a
    // position the player never dragged on. Closing the picker is the visible
    // half, and the revision bump above snaps the parked pawn back for free,
    // exactly as `cancelPromotion` does it.
    //
    // An equal-version frame is deliberately *not* a different board: `apply`
    // accepts equal versions so a reconnect re-sync and a duplicated broadcast
    // land, and neither moved anything. The hint and the review view above are
    // dropped on those anyway because they are derived displays the app can
    // recompute; input the player has already given is not.
    const armed = pendingPromotionRef.current
    if (armed !== null && !armedForBoard(armed, next)) armPromotion(null)
    return true
  }, [armPromotion])

  useEffect(() => {
    let live = true
    let socket: WebSocket | null = null
    let retryTimer: ReturnType<typeof setTimeout> | null = null
    let retryDelay = SOCKET_RETRY_BASE_MS

    const syncState = () => {
      void fetchState().then((s) => {
        // Null: the fetch never made it (backend mid-redeploy, gateway drop).
        // Nothing to do here — the socket's own snapshot on (re)connect is
        // the retry.
        if (live && s) apply(s)
      })
    }
    const detach = (target: WebSocket) => {
      target.onopen = null
      target.onmessage = null
      target.onclose = null
      target.onerror = null
    }
    const scheduleReconnect = () => {
      // `socket` and `retryTimer` are the duplicate-connection guards. A
      // repeated close/error from an old socket cannot add another attempt.
      if (!live || socket !== null || retryTimer !== null) return
      const delay = retryDelay
      retryDelay = Math.min(retryDelay * 2, SOCKET_RETRY_MAX_MS)
      retryTimer = setTimeout(() => {
        retryTimer = null
        connect(true)
      }, delay)
    }
    const connect = (reconnecting: boolean) => {
      if (!live || socket !== null) return
      const next = new WebSocket(stateSocketUrl())
      socket = next
      next.onopen = () => {
        if (!live || socket !== next) return
        retryDelay = SOCKET_RETRY_BASE_MS
        // The WebSocket will also send its snapshot, but an explicit fetch
        // closes the gap even if frames were lost around the outage. `apply`
        // keeps the highest-version result whichever arrives first.
        if (reconnecting) syncState()
      }
      next.onmessage = (ev) => {
        if (!live || socket !== next) return
        const message = JSON.parse(ev.data) as SocketMessage
        // Two kinds of message on the one channel: the authoritative board,
        // and what the turn changing it is doing at this moment.
        if (message.type === 'state') apply(message.state)
        else if (message.type === 'progress') {
          setProgress((current) => applyProgress(current, message.progress))
        }
      }
      next.onclose = () => {
        if (socket !== next) return
        socket = null
        detach(next)
        scheduleReconnect()
      }
      next.onerror = () => {
        if (!live || socket !== next) return
        // `close` is the one place that schedules retries. Browsers normally
        // follow an error with close; actively closing also covers those that
        // do not, without creating a second retry path.
        next.close()
      }
    }

    syncState()
    // Null: settings never arrived. All three stay null — the hook's word for
    // "not loaded" — rather than adopting values the server never confirmed.
    fetchSettings().then((s) => {
      if (live && s) {
        setVoiceOutputState(s.voice_output)
        setTierState(s.tier)
        // Undefined (an older backend) stays null: unknown is not direct mode,
        // and the indicator only claims what the server actually said.
        setAgentAvailable(s.agent_available ?? null)
      }
    })
    connect(false)
    return () => {
      live = false
      if (retryTimer !== null) {
        clearTimeout(retryTimer)
        retryTimer = null
      }
      if (socket !== null) {
        const current = socket
        socket = null
        // Detach before the intentional close so its close event cannot
        // schedule a reconnect after unmount.
        detach(current)
        current.close()
      }
    }
  }, [apply])

  const submit = useCallback(
    async (uci: string) => {
      // Capture the rendered board's version before yielding. Two drags made
      // before either response arrives therefore cite the same version and the
      // backend deterministically accepts at most one of them.
      const result = await submitMove(uci, stateRef.current?.version)
      if (isStaleStateResponse(result)) {
        apply(result.state)
        return
      }
      // The move never landed and no state came back with the refusal, so
      // there is nothing to apply — and nothing else will re-sync the board
      // either, because a rejected move broadcasts no frame (#231). The
      // revision bump *is* the snap-back on its own: the position is unchanged,
      // so `fen` alone would leave the piece sitting where chessground dropped
      // it, exactly as `cancelPromotion` has to bump it for a pawn it parked.
      if (isMoveFailure(result)) {
        setMoveError(result.detail)
        setRevision((r) => r + 1)
        return
      }
      // A WebSocket frame may have advanced the board while this request was
      // in flight. Ignore every part of an older response, not just its FEN.
      if (!apply(result.state)) return
      setMoveError(result.legal ? null : (result.reason ?? 'Illegal move'))
      // In agent mode the drag went through the same beats a typed move does,
      // so it comes back with Glitch's reaction to it — staged and voiced
      // exactly like a command's, because it is the same kind of turn. Direct
      // mode sends no commentary at all, which leaves the bubble alone.
      if (result.commentary) {
        setCommentary(result.commentary)
        // Fire-and-forget: the words are already on screen and a playback
        // failure must never touch the game.
        if (result.speak) void playText(result.commentary)
      }
    },
    [apply],
  )

  const play = useCallback(
    async (from: string, to: string) => {
      // Belt-and-braces: the board is viewOnly during review, but never let
      // a stray drag submit a move against a position that isn't live.
      if (viewPlyRef.current !== null) return
      // A promotion can't be expressed as bare from+to; hold it for a piece
      // choice rather than submitting a move the backend would reject. Stamped
      // with the board it was dragged on, because the choice arrives later and
      // the squares alone can't say which position they were meant for.
      if (stateRef.current && isPromotion(stateRef.current.fen, from, to)) {
        armPromotion({
          from,
          to,
          version: stateRef.current.version,
          fen: stateRef.current.fen,
        })
        return
      }
      await submit(from + to)
    },
    [armPromotion, submit],
  )

  const completePromotion = useCallback(
    async (piece: PromotionPiece) => {
      const armed = pendingPromotionRef.current
      if (armed === null) return
      armPromotion(null)
      // Braces to `apply`'s belt: a promotion armed against a superseded board
      // may not be submitted even if the clear above it was somehow missed.
      // `submit` cites whatever version is live *now*, so what would go out is
      // not the move the player made — it is their squares against someone
      // else's position.
      const live = stateRef.current
      if (live === null || !armedForBoard(armed, live)) {
        // Nothing was sent, so nothing else will re-sync the board: snap the
        // pawn back off the last rank the way a cancel does.
        setRevision((r) => r + 1)
        return
      }
      await submit(armed.from + armed.to + piece)
    },
    [armPromotion, submit],
  )

  const cancelPromotion = useCallback(() => {
    armPromotion(null)
    // The board already moved the pawn to the last rank; re-sync it back.
    setRevision((r) => r + 1)
  }, [armPromotion])

  /**
   * Settle a gated destructive op: put the gate's question to the player and
   * send their answer back to the *same* armed op the spoken path uses. A
   * cancel disarms it and leaves the board where it is; only a confirmed op
   * moves anything. `window.confirm` because the app has no dialog primitive
   * for a plain yes/no (the modals it does have are screens, not questions).
   */
  const answerGate = useCallback(
    async (question: ConfirmQuestion) => {
      const yes = window.confirm(question.detail)
      const outcome = await apiConfirmDestructive(yes, stateRef.current?.version)
      if (outcome && (yes || outcome.stale)) {
        setMoveError(null)
        apply(outcome.state)
      }
    },
    [apply],
  )

  /** Apply a gated lifecycle outcome: the new state if the op ran, otherwise
   * the question it is waiting on. */
  const settle = useCallback(
    async (outcome: LifecycleOutcome) => {
      if (outcome.state) {
        setMoveError(null)
        apply(outcome.state)
      } else if (outcome.question) {
        await answerGate(outcome.question)
      }
    },
    [apply, answerGate],
  )

  const newGame = useCallback(
    async (color?: 'white' | 'black' | 'random') => {
      await settle(await apiNewGame(color, stateRef.current?.version))
    },
    [settle],
  )

  const undo = useCallback(async () => {
    const next = await apiUndo(undefined, stateRef.current?.version)
    // Null means the backend refused (nothing to undo) — leave state as is.
    if (next) {
      setMoveError(null)
      apply(next)
    }
  }, [apply])

  const resign = useCallback(async () => {
    await settle(await apiResign(undefined, stateRef.current?.version))
  }, [settle])

  const setDifficulty = useCallback(async (nextTier: string) => {
    // Difficulty is a settings change, not a board mutation — no state to
    // apply. The hook reflects only what the server confirmed, so the
    // selector can never drift from the strength the engine actually plays.
    const confirmed = await apiSetDifficulty(nextTier)
    if (confirmed !== null) setTierState(confirmed.tier)
  }, [])

  const sendCommand = useCallback(
    async (text: string) => {
      setAgentThinking(true)
      try {
        const response = await apiSendCommand(text, stateRef.current?.version)
        if (response) {
          if (isStaleStateResponse(response)) {
            apply(response.state)
            return
          }
          // A state frame may have won the race while the agent was thinking.
          // If so, none of this older response (including commentary/settings)
          // belongs to the board now being rendered.
          if (!apply(response.state)) return
          setCommentary(response.commentary)
          // Voice out is fire-and-forget: the reply is already on screen,
          // and a playback failure must never block the game.
          if (response.speak && response.commentary) void playText(response.commentary)
          // speak mirrors the server-side voice_output setting, so an
          // agent-side toggle ("turn on voice") keeps the UI switch in sync.
          if (typeof response.speak === 'boolean') setVoiceOutputState(response.speak)
          // tier mirrors the server-side difficulty the same way, so "make
          // it harder" moves the options-sheet selector too (null: strength
          // was set outside the named tiers).
          if (response.tier !== undefined) setTierState(response.tier)
        } else {
          // Null means the backend refused (e.g. 503: no brain) — leave the
          // board untouched and tell the user the agent isn't available.
          setCommentary('The agent is unavailable.')
        }
      } finally {
        setAgentThinking(false)
      }
    },
    [apply],
  )

  const stepBack = useCallback(() => {
    const fens = stateRef.current?.fens
    if (!fens || fens.length < 2) return
    // From live, start at the position before the last ply; while
    // reviewing, keep walking back, clamped at the root.
    setView(viewPlyRef.current === null ? fens.length - 2 : Math.max(0, viewPlyRef.current - 1))
  }, [setView])

  const stepForward = useCallback(() => {
    const fens = stateRef.current?.fens
    if (!fens || viewPlyRef.current === null) return
    const next = viewPlyRef.current + 1
    // Reaching the latest position resumes the live view.
    setView(next >= fens.length - 1 ? null : next)
  }, [setView])

  const requestHint = useCallback(async () => {
    // A hint only makes sense for the live position.
    if (viewPlyRef.current !== null || stateRef.current?.game_over) return
    const hint = await fetchHint()
    // Null means the backend refused (no engine, game over) — no arrow.
    if (hint === null) {
      setHintShapes([])
      return
    }
    // Both checks above are asked again here, because the search took real time
    // and the board it analyzed may no longer be the board on screen. A result
    // for a position that is gone neither paints nor erases: it has no say over
    // an arrow that legitimately belongs to the live board.
    //
    // Review moves the view without moving the version, so it is its own check.
    if (viewPlyRef.current !== null) return
    // The version binds the answer to the position it analyzed (#218): a hint
    // that resolves after `apply` accepted a newer board is a recommendation for
    // a position that is gone, and may not even be legal on this one. Plain
    // equality, so it also fails safe either way round if the two halves ever
    // disagree about carrying a version at all.
    if (hint.version !== stateRef.current?.version) return
    setHintShapes([{ orig: hint.from, dest: hint.to, brush: 'green' }])
  }, [])

  const setVoiceOutput = useCallback(async (enabled: boolean) => {
    // Settings change, not a board mutation. The hook reflects what the
    // server confirmed, so the toggle can never drift from the truth.
    const confirmed = await apiSetVoiceOutput(enabled)
    if (confirmed !== null) setVoiceOutputState(confirmed)
  }, [])

  return {
    state,
    moveError,
    revision,
    play,
    pendingPromotion,
    completePromotion,
    cancelPromotion,
    newGame,
    undo,
    resign,
    setDifficulty,
    tier,
    commentary,
    agentAvailable,
    agentThinking,
    agentProgress: progress.label,
    sendCommand,
    voiceOutput,
    setVoiceOutput,
    viewPly,
    reviewing: viewPly !== null,
    displayFen:
      viewPly !== null && state ? (state.fens[viewPly] ?? state.fen) : (state?.fen ?? null),
    stepBack,
    stepForward,
    hintShapes,
    requestHint,
  }
}
