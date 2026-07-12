import { useCallback, useEffect, useRef, useState } from 'react'
import {
  fetchHint,
  fetchSettings,
  fetchState,
  newGame as apiNewGame,
  resign as apiResign,
  sendCommand as apiSendCommand,
  setDifficulty as apiSetDifficulty,
  setVoiceOutput as apiSetVoiceOutput,
  stateSocketUrl,
  submitMove,
  undo as apiUndo,
  type GameState,
} from './api'
import { isPromotion, type PromotionPiece } from './promotion'
import { playText } from './tts'

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
   * visually, so a cancel must re-sync it back.
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
  /** The agent's latest commentary, or null before the first command. */
  commentary: string | null
  /** True while a command is in flight with the agent. */
  agentThinking: boolean
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
  const [voiceOutput, setVoiceOutputState] = useState<boolean | null>(null)
  const [tier, setTierState] = useState<string | null>(null)
  const [viewPly, setViewPly] = useState<number | null>(null)
  const [hintShapes, setHintShapes] = useState<
    { orig: string; dest: string; brush: string }[]
  >([])
  // Latest state without making the move callbacks depend on it — the board
  // holds `play` in a ref, but promotion detection still needs the live fen.
  const stateRef = useRef<GameState | null>(null)
  // Mirror of viewPly for the same reason: `play` must see the live value.
  const viewPlyRef = useRef<number | null>(null)

  const setView = useCallback((ply: number | null) => {
    viewPlyRef.current = ply
    setViewPly(ply)
  }, [])

  const apply = useCallback((next: GameState) => {
    stateRef.current = next
    setState(next)
    setRevision((r) => r + 1)
    // Any authoritative state ends a history review and invalidates the
    // hint arrow — both were drawn for a position that may no longer exist.
    viewPlyRef.current = null
    setViewPly(null)
    setHintShapes([])
  }, [])

  useEffect(() => {
    let live = true
    fetchState().then((s) => {
      if (live) apply(s)
    })
    fetchSettings().then((s) => {
      if (live) {
        setVoiceOutputState(s.voice_output)
        setTierState(s.tier)
      }
    })
    const socket = new WebSocket(stateSocketUrl())
    socket.onmessage = (ev) => {
      const message = JSON.parse(ev.data) as { type: string; state: GameState }
      if (message.type === 'state') apply(message.state)
    }
    return () => {
      live = false
      socket.close()
    }
  }, [apply])

  const submit = useCallback(
    async (uci: string) => {
      const result = await submitMove(uci)
      setMoveError(result.legal ? null : (result.reason ?? 'Illegal move'))
      // Authoritative in both cases: the new position on success, or the
      // unchanged one on rejection (which snaps the illegal move back).
      apply(result.state)
    },
    [apply],
  )

  const play = useCallback(
    async (from: string, to: string) => {
      // Belt-and-braces: the board is viewOnly during review, but never let
      // a stray drag submit a move against a position that isn't live.
      if (viewPlyRef.current !== null) return
      // A promotion can't be expressed as bare from+to; hold it for a piece
      // choice rather than submitting a move the backend would reject.
      if (stateRef.current && isPromotion(stateRef.current.fen, from, to)) {
        setPendingPromotion({ from, to })
        return
      }
      await submit(from + to)
    },
    [submit],
  )

  const completePromotion = useCallback(
    async (piece: PromotionPiece) => {
      if (!pendingPromotion) return
      const { from, to } = pendingPromotion
      setPendingPromotion(null)
      await submit(from + to + piece)
    },
    [pendingPromotion, submit],
  )

  const cancelPromotion = useCallback(() => {
    setPendingPromotion(null)
    // The board already moved the pawn to the last rank; re-sync it back.
    setRevision((r) => r + 1)
  }, [])

  const newGame = useCallback(
    async (color?: 'white' | 'black' | 'random') => {
      const next = await apiNewGame(color)
      if (next) {
        setMoveError(null)
        apply(next)
      }
    },
    [apply],
  )

  const undo = useCallback(async () => {
    const next = await apiUndo()
    // Null means the backend refused (nothing to undo) — leave state as is.
    if (next) {
      setMoveError(null)
      apply(next)
    }
  }, [apply])

  const resign = useCallback(async () => {
    const next = await apiResign()
    if (next) apply(next)
  }, [apply])

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
        const response = await apiSendCommand(text)
        if (response) {
          setCommentary(response.commentary)
          // The agent acts only through tools; the returned state is
          // authoritative (unchanged for a question or read-only command).
          apply(response.state)
          // Voice out is fire-and-forget: the reply is already on screen,
          // and a playback failure must never block the game.
          if (response.speak && response.commentary) void playText(response.commentary)
          // speak mirrors the server-side voice_output setting, so an
          // agent-side toggle ("turn on voice") keeps the UI switch in sync.
          if (typeof response.speak === 'boolean') setVoiceOutputState(response.speak)
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
    setHintShapes(hint ? [{ orig: hint.from, dest: hint.to, brush: 'green' }] : [])
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
    agentThinking,
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
