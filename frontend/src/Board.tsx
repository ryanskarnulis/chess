import { useEffect, useRef } from 'react'
import { Chessground } from 'chessground'
import type { Api } from 'chessground/api'
import type { DrawBrush, DrawBrushes, DrawShape } from 'chessground/draw'
import type { Key } from 'chessground/types'

import 'chessground/assets/chessground.base.css'
import './chessground.nightsilk.css'

/**
 * Glitch's voice on the board. Chessground resolves shape colours from a JS
 * brush table rather than from CSS, so the hint arrow is the one part of the
 * night-silk theme that can't live in the stylesheet. Deliberately `--warn`
 * and not `--glow`: glow is the player's own hue everywhere else on the board
 * (last move, selection, legal destinations), so advice must not read as
 * something the player selected.
 *
 * Typed as a plain record and cast at the call site because `DrawBrushes`
 * demands its four built-in keys. That's a gap between chessground's types
 * and its behaviour, not a real requirement: `configure` deep-merges this
 * table into the defaults, so naming one brush adds it and leaves
 * green/red/blue/yellow available.
 */
const BOARD_BRUSHES: Record<string, DrawBrush> = {
  hint: { key: 'hint', color: '#eec98a', opacity: 0.9, lineWidth: 10 },
}

export interface BoardProps {
  /** Board position as a FEN (piece-placement field is enough). */
  fen: string
  /** Side to move — only that colour's pieces are draggable. */
  turnColor?: 'white' | 'black'
  /** Which side sits at the bottom of the board (the player's side). */
  orientation?: 'white' | 'black'
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
  /** Program-drawn arrows/highlights (e.g. the hint arrow); replaces the
   * previous set on every change, so `[]` clears them. */
  autoShapes?: { orig: string; dest?: string; brush: string }[]
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
export function Board({
  fen,
  turnColor,
  orientation = 'white',
  dests,
  onMove,
  viewOnly = false,
  revision,
  autoShapes,
}: BoardProps) {
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
    apiRef.current = Chessground(mountRef.current, {
      coordinates: true,
      drawable: { brushes: BOARD_BRUSHES as DrawBrushes },
    })
    // Chessground caches its screen bounds and only re-measures on window
    // resize/scroll. Content-driven layout shifts (commentary appearing,
    // panels growing, a scrollbar showing up) move the board without either
    // event, so clicks would map through stale coordinates. Any such shift
    // changes the page's size, so watch it and fire chessground's own
    // re-measure event. (Guarded: jsdom has no ResizeObserver.)
    let observer: ResizeObserver | null = null
    if (typeof ResizeObserver !== 'undefined') {
      observer = new ResizeObserver(() =>
        document.body.dispatchEvent(new Event('chessground.resize')),
      )
      observer.observe(document.body)
    }
    return () => {
      observer?.disconnect()
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
      orientation,
      movable: {
        free: false,
        color: viewOnly ? undefined : turnColor,
        dests: toDests(dests),
        events: {
          after: (orig, dest) => onMoveRef.current?.(orig, dest),
        },
      },
    })
  }, [fen, viewOnly, turnColor, orientation, dests, revision])

  // Shapes are pushed through their own API call: `set` merges config, but
  // setAutoShapes replaces the drawn set, which is what a hint needs.
  useEffect(() => {
    apiRef.current?.setAutoShapes((autoShapes ?? []) as DrawShape[])
  }, [autoShapes])

  return <div ref={mountRef} className="board" />
}
