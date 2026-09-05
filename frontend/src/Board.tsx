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
    // Chessground measures the board's screen box once and caches it. It
    // drops that cache on a window resize, on a scroll, and when the mount
    // element's own *size* changes (its ResizeObserver) — and on nothing at
    // all when the board merely *moves*. It used to move constantly: Glitch's
    // bubble sits above it in the column and grew with the reply, so a
    // multi-line one slid the whole board down the page and left the
    // remembered box behind. Measured here, that slide was 112px for a
    // six-line reply — a square and a half — after which clicking a piece
    // picks up the one below it and clicking its middle picks up nothing
    // (walkthrough #2).
    //
    // The bubble is clamped to a fixed-height row now and scrolls inside
    // itself, so that particular slide is gone and this listener is
    // belt-and-braces rather than the fix. It stays: the board can still move
    // under a resting cursor — the Copy PGN chip appearing under a reply, the
    // direct-mode notice arriving, a layout change nobody has made yet — and
    // the failure is silent and looks like a broken board, not like a moved
    // one.
    //
    // The rendered geometry was never the problem: Chromium and Firefox both
    // draw exactly-square 67px squares inside a square frame, agreeing with
    // the measured rect to within 0.06 of a square. Only the *remembered*
    // rect goes stale. So the fix is to stop carrying one across a layout
    // change — and rather than guess which ancestor's resize implies a move
    // (a column growing inside a taller viewport never reaches `body`, and a
    // swap above the board need not change any height at all), re-measure at
    // the one moment the number is used: the start of an interaction.
    //
    // `pointerdown` runs ahead of both `mousedown` and `touchstart`, which
    // are what chessground starts a click or a drag from, and a `scroll` at
    // the document is chessground's own "your box may have moved" signal —
    // it drops the cached rect and does nothing else. One `getBoundingClientRect`
    // per interaction, and no layout shift can outrun it.
    const remeasure = () => document.dispatchEvent(new Event('scroll'))
    const mount = mountRef.current
    mount.addEventListener('pointerdown', remeasure)
    return () => {
      mount.removeEventListener('pointerdown', remeasure)
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
