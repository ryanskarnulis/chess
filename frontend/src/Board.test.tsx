import { render } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { Board } from './Board'

const START_FEN = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR'

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
})
