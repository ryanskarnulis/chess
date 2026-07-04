import { useEffect, useRef } from 'react'
import { Chessground } from 'chessground'
import type { Api } from 'chessground/api'
import type { Config } from 'chessground/config'

import 'chessground/assets/chessground.base.css'
import 'chessground/assets/chessground.brown.css'
import 'chessground/assets/chessground.cburnett.css'

export interface BoardProps {
  /** Board position as a FEN (piece-placement field is enough). */
  fen: string
  /** Render without interaction (display-only). Defaults to true for this
   *  scaffold slice; move handling arrives in the next slice. */
  viewOnly?: boolean
}

/**
 * Thin React wrapper around Lichess's Chessground. Chessground owns its own
 * DOM subtree; React only owns the mount point and the instance lifecycle.
 * The board is display-only for now — the backend remains the source of truth,
 * so this component never validates or generates moves.
 */
export function Board({ fen, viewOnly = true }: BoardProps) {
  const mountRef = useRef<HTMLDivElement>(null)
  const apiRef = useRef<Api | null>(null)

  // Create the Chessground instance once, mounted into our div. Position and
  // interactivity are applied by the sync effect below (which runs immediately
  // after mount), keeping this effect free of prop dependencies.
  useEffect(() => {
    if (!mountRef.current) return
    const config: Config = { coordinates: true }
    apiRef.current = Chessground(mountRef.current, config)
    return () => {
      apiRef.current?.destroy()
      apiRef.current = null
    }
  }, [])

  // Push prop changes to the live instance without re-creating it.
  useEffect(() => {
    apiRef.current?.set({ fen, viewOnly })
  }, [fen, viewOnly])

  return <div ref={mountRef} className="board" />
}
