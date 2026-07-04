import { useEffect, useRef } from 'react'
import { Chessground } from 'chessground'
import type { Api } from 'chessground/api'
import type { Key } from 'chessground/types'

import 'chessground/assets/chessground.base.css'
import 'chessground/assets/chessground.brown.css'
import 'chessground/assets/chessground.cburnett.css'

export interface BoardProps {
  /** Board position as a FEN (piece-placement field is enough). */
  fen: string
  /** Side to move — only that colour's pieces are draggable. */
  turnColor?: 'white' | 'black'
  /** Legal destinations by origin square, from the backend (`{ e2: ['e3','e4'] }`). */
  dests?: Record<string, string[]>
  /** Fired when the user completes a move; args are origin/dest squares. */
  onMove?: (from: string, to: string) => void
  /** Render without interaction (e.g. once the game is over). */
  viewOnly?: boolean
  /**
   * Bump to force a re-sync even when `fen` is unchanged — used to snap an
   * illegal move back to the authoritative position after the backend
   * rejects it (the position, and thus `fen`, is unchanged in that case).
   */
  revision?: number
}

function toDests(dests?: Record<string, string[]>): Map<Key, Key[]> {
  const map = new Map<Key, Key[]>()
  if (dests) {
    for (const [from, tos] of Object.entries(dests)) map.set(from as Key, tos as Key[])
  }
  return map
}

/**
 * Thin React wrapper around Lichess's Chessground. Chessground owns its own
 * DOM subtree; React only owns the mount point and the instance lifecycle.
 * The board never validates or generates moves — legal destinations come from
 * the backend and completed moves are handed back to `onMove` for submission.
 */
export function Board({ fen, turnColor, dests, onMove, viewOnly = false, revision }: BoardProps) {
  const mountRef = useRef<HTMLDivElement>(null)
  const apiRef = useRef<Api | null>(null)
  // Hold the latest handler without making it a sync-effect dependency, so a
  // new `onMove` identity each render doesn't churn the board.
  const onMoveRef = useRef(onMove)
  onMoveRef.current = onMove

  // Create the Chessground instance once, mounted into our div. Position and
  // interactivity are applied by the sync effect below (which runs immediately
  // after mount), keeping this effect free of prop dependencies.
  useEffect(() => {
    if (!mountRef.current) return
    apiRef.current = Chessground(mountRef.current, { coordinates: true })
    return () => {
      apiRef.current?.destroy()
      apiRef.current = null
    }
  }, [])

  // Push prop changes to the live instance without re-creating it.
  useEffect(() => {
    apiRef.current?.set({
      fen,
      viewOnly,
      turnColor,
      movable: {
        free: false,
        color: viewOnly ? undefined : turnColor,
        dests: toDests(dests),
        events: {
          after: (orig, dest) => onMoveRef.current?.(orig, dest),
        },
      },
    })
  }, [fen, viewOnly, turnColor, dests, revision])

  return <div ref={mountRef} className="board" />
}
