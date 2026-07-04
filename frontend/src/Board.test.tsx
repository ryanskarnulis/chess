import { render } from '@testing-library/react'
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

  it('re-renders the position when the fen prop changes', () => {
    const { container, rerender } = render(<Board fen={START_FEN} />)
    expect(container.querySelectorAll('piece:not(.ghost)')).toHaveLength(32)
    rerender(<Board fen={LONE_KING_FEN} />)
    expect(container.querySelectorAll('piece:not(.ghost)')).toHaveLength(1)
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
