import { render, waitFor } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import { Board } from './Board'

const START_FEN = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR'
const LONE_KING_FEN = '8/8/8/8/8/8/8/4K3'

describe('Board', () => {
  it('mounts a Chessground board into the DOM', () => {
    const { container } = render(<Board fen={START_FEN} />)
    // Chessground renders its board inside a .cg-wrap element.
    expect(container.querySelector('.cg-wrap')).toBeInTheDocument()
  })

  it('renders the 32 pieces of the starting position', () => {
    const { container } = render(<Board fen={START_FEN} />)
    // Chessground keeps one extra `.ghost` <piece> for dragging; exclude it.
    expect(container.querySelectorAll('piece:not(.ghost)')).toHaveLength(32)
  })

  it('destroys the Chessground instance on unmount', () => {
    const { container, unmount } = render(<Board fen={START_FEN} />)
    unmount()
    expect(container.querySelector('.cg-wrap')).not.toBeInTheDocument()
  })

  it('orients the board for the given side', () => {
    const { container, rerender } = render(<Board fen={START_FEN} orientation="black" />)
    // Chessground stamps the orientation on its wrap element.
    expect(container.querySelector('.cg-wrap.orientation-black')).toBeInTheDocument()
    rerender(<Board fen={START_FEN} orientation="white" />)
    expect(container.querySelector('.cg-wrap.orientation-white')).toBeInTheDocument()
  })

  it('re-renders the position when the fen prop changes', () => {
    const { container, rerender } = render(<Board fen={START_FEN} />)
    expect(container.querySelectorAll('piece:not(.ghost)')).toHaveLength(32)
    rerender(<Board fen={LONE_KING_FEN} />)
    expect(container.querySelectorAll('piece:not(.ghost)')).toHaveLength(1)
  })

  it('tells chessground to re-measure when the page layout shifts', () => {
    // Chessground caches its screen bounds and only refreshes them on window
    // resize/scroll. Content growth (commentary, panels, a scrollbar) moves
    // the board without either event, so clicks land on stale coordinates.
    // The board must watch the page size and fire chessground's re-measure
    // event ('chessground.resize' on document.body) when it changes.
    interface StubObserver {
      cb: () => void
      targets: Element[]
    }
    const observers: StubObserver[] = []
    vi.stubGlobal(
      'ResizeObserver',
      class {
        cb: () => void
        targets: Element[] = []
        constructor(cb: () => void) {
          this.cb = cb
          observers.push(this)
        }
        observe(target: Element) {
          this.targets.push(target)
        }
        disconnect() {
          this.targets = []
        }
      },
    )
    const remeasured = vi.fn()
    document.body.addEventListener('chessground.resize', remeasured)
    try {
      const { unmount } = render(<Board fen={START_FEN} />)
      const bodyObserver = observers.find((o) => o.targets.includes(document.body))
      expect(bodyObserver).toBeDefined()
      bodyObserver!.cb()
      expect(remeasured).toHaveBeenCalled()
      // Unmount must stop watching so a dead board can't keep firing.
      unmount()
      expect(bodyObserver!.targets).toHaveLength(0)
    } finally {
      document.body.removeEventListener('chessground.resize', remeasured)
      vi.unstubAllGlobals()
    }
  })

  it('draws an auto-shape arrow for a hint and clears it when removed', async () => {
    // Chessground skips shape drawing when its container has zero bounds,
    // and jsdom sizes everything at 0 — give the board a real footprint.
    const rect = { x: 0, y: 0, top: 0, left: 0, bottom: 512, right: 512, width: 512, height: 512, toJSON: () => ({}) } as DOMRect
    const spy = vi
      .spyOn(HTMLElement.prototype, 'getBoundingClientRect')
      .mockReturnValue(rect)
    try {
      const { container, rerender } = render(
        <Board fen={START_FEN} autoShapes={[{ orig: 'e2', dest: 'e4', brush: 'hint' }]} />,
      )
      // Chessground renders each auto-shape as a group in its shapes svg
      // (on its next animation frame, hence the waits).
      await waitFor(() =>
        expect(container.querySelectorAll('svg.cg-shapes g *').length).toBeGreaterThan(0),
      )
      rerender(<Board fen={START_FEN} autoShapes={[]} />)
      await waitFor(() =>
        expect(container.querySelectorAll('svg.cg-shapes g *')).toHaveLength(0),
      )
    } finally {
      spy.mockRestore()
    }
  })

  it('mounts as an interactive board when given moves and a handler', () => {
    const onMove = vi.fn()
    const { container } = render(
      <Board fen={START_FEN} turnColor="white" dests={{ e2: ['e3', 'e4'] }} onMove={onMove} />,
    )
    expect(container.querySelector('.cg-wrap')).toBeInTheDocument()
    expect(container.querySelectorAll('piece:not(.ghost)')).toHaveLength(32)
  })
})
