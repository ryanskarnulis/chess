import { useCallback, useEffect, useRef, useState } from 'react'
import {
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
  /** Start a fresh game from the initial position. */
  newGame: () => Promise<void>
  /** Take back the last ply. No-op if there is nothing to undo. */
  undo: () => Promise<void>
  /** Resign the game (the side to move, unless the backend decides otherwise). */
  resign: () => Promise<void>
  /** Set engine strength by Stockfish skill level (0–20). */
  setDifficulty: (skillLevel: number) => Promise<void>
  /** Server-confirmed skill level; null until settings load (or when the
   * strength was last set outside the skill scale, e.g. by elo). */
  skillLevel: number | null
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
  const [skillLevel, setSkillLevelState] = useState<number | null>(null)
  // Latest state without making the move callbacks depend on it — the board
  // holds `play` in a ref, but promotion detection still needs the live fen.
  const stateRef = useRef<GameState | null>(null)

  const apply = useCallback((next: GameState) => {
    stateRef.current = next
    setState(next)
    setRevision((r) => r + 1)
  }, [])

  useEffect(() => {
    let live = true
    fetchState().then((s) => {
      if (live) apply(s)
    })
    fetchSettings().then((s) => {
      if (live) {
        setVoiceOutputState(s.voice_output)
        setSkillLevelState(s.skill_level)
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

  const newGame = useCallback(async () => {
    const next = await apiNewGame()
    if (next) {
      setMoveError(null)
      apply(next)
    }
  }, [apply])

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

  const setDifficulty = useCallback(async (level: number) => {
    // Difficulty is a settings change, not a board mutation — no state to
    // apply. The hook reflects only what the server confirmed, so the
    // selector can never drift from the strength the engine actually plays.
    const confirmed = await apiSetDifficulty(level)
    if (confirmed !== null) setSkillLevelState(confirmed.skill_level)
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
    skillLevel,
    commentary,
    agentThinking,
    sendCommand,
    voiceOutput,
    setVoiceOutput,
  }
}
