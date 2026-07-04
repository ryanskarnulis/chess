import { useCallback, useEffect, useState } from 'react'
import { fetchState, stateSocketUrl, submitMove, type GameState } from './api'

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

  const apply = useCallback((next: GameState) => {
    setState(next)
    setRevision((r) => r + 1)
  }, [])

  useEffect(() => {
    let live = true
    fetchState().then((s) => {
      if (live) apply(s)
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

  const play = useCallback(
    async (from: string, to: string) => {
      const result = await submitMove(from + to)
      setMoveError(result.legal ? null : (result.reason ?? 'Illegal move'))
      // Authoritative in both cases: the new position on success, or the
      // unchanged one on rejection (which snaps the illegal move back).
      apply(result.state)
    },
    [apply],
  )

  return { state, moveError, revision, play }
}
