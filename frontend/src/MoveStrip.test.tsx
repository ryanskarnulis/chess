import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import { MoveStrip } from './MoveStrip'

function strip(overrides: Partial<Parameters<typeof MoveStrip>[0]> = {}) {
  return (
    <MoveStrip
      history={['e4', 'e5', 'Nf3']}
      currentPly={3}
      onBack={vi.fn()}
      onForward={vi.fn()}
      canBack={true}
      canForward={false}
      {...overrides}
    />
  )
}

describe('MoveStrip', () => {
  it('renders the moves with move numbers', () => {
    render(strip())
    expect(screen.getByText('e4')).toBeInTheDocument()
    expect(screen.getByText('e5')).toBeInTheDocument()
    expect(screen.getByText('Nf3')).toBeInTheDocument()
    expect(screen.getByText('1.')).toBeInTheDocument()
    expect(screen.getByText('2.')).toBeInTheDocument()
  })

  it('draws the review arrows as monochrome inline icons, never color emoji', () => {
    const { container } = render(strip())
    // U+25C0/U+25B6 are emoji-presentation capable and render as blue tiles
    // on iOS, the same defect the bottom bar had.
    expect(container.textContent).not.toMatch(/\p{Extended_Pictographic}/u)
    const icons = container.querySelectorAll('button svg')
    expect(icons).toHaveLength(2)
    for (const svg of icons) expect(svg).toHaveAttribute('aria-hidden', 'true')
  })

  it('highlights the move at the current ply', () => {
    render(strip({ currentPly: 2 }))
    expect(screen.getByText('e5')).toHaveClass('current')
    expect(screen.getByText('Nf3')).not.toHaveClass('current')
  })

  it('highlights the latest move when live', () => {
    render(strip())
    expect(screen.getByText('Nf3')).toHaveClass('current')
  })

  it('fires the step callbacks', () => {
    const onBack = vi.fn()
    const onForward = vi.fn()
    render(strip({ onBack, onForward, canBack: true, canForward: true }))
    fireEvent.click(screen.getByRole('button', { name: /previous move/i }))
    expect(onBack).toHaveBeenCalled()
    fireEvent.click(screen.getByRole('button', { name: /next move/i }))
    expect(onForward).toHaveBeenCalled()
  })

  it('disables the arrows at the ends of history', () => {
    render(strip({ canBack: false, canForward: false }))
    expect(screen.getByRole('button', { name: /previous move/i })).toBeDisabled()
    expect(screen.getByRole('button', { name: /next move/i })).toBeDisabled()
  })

  it('renders empty without moves', () => {
    render(strip({ history: [], currentPly: 0, canBack: false }))
    expect(screen.queryByText('1.')).not.toBeInTheDocument()
  })
})
