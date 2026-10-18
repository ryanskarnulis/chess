import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import { BottomBar } from './BottomBar'

function bar(overrides: Partial<Parameters<typeof BottomBar>[0]> = {}) {
  return (
    <BottomBar
      onUndo={vi.fn()}
      onHint={vi.fn()}
      onGame={vi.fn()}
      undoDisabled={false}
      hintDisabled={false}
      gameOpen={false}
      {...overrides}
    />
  )
}

describe('BottomBar', () => {
  it('renders exactly Undo, Hint and Game', () => {
    render(bar())
    const names = screen.getAllByRole('button').map((b) => b.textContent)
    expect(names).toEqual(['Undo', 'Hint', 'Game'])
  })

  it('fires each callback', () => {
    const onUndo = vi.fn()
    const onHint = vi.fn()
    const onGame = vi.fn()
    render(bar({ onUndo, onHint, onGame }))
    fireEvent.click(screen.getByRole('button', { name: /undo/i }))
    fireEvent.click(screen.getByRole('button', { name: /hint/i }))
    fireEvent.click(screen.getByRole('button', { name: /game/i }))
    expect(onUndo).toHaveBeenCalled()
    expect(onHint).toHaveBeenCalled()
    expect(onGame).toHaveBeenCalled()
  })

  // Reported from an iPhone: U+21A9 rendered as the blue color ↩️ tile and
  // the flag/bulb as color emoji, against an otherwise monochrome UI. The
  // guard is the character class, not the specific glyphs, so putting any
  // emoji-presentation character back here fails.
  it('marks each button with a monochrome inline icon, never a color emoji', () => {
    const { container } = render(bar())
    expect(container.textContent).not.toMatch(/\p{Extended_Pictographic}/u)
    const icons = container.querySelectorAll('button svg')
    expect(icons).toHaveLength(3)
    for (const svg of icons) expect(svg).toHaveAttribute('aria-hidden', 'true')
  })

  it('says whether the Game menu is open', () => {
    const { rerender } = render(bar({ gameOpen: false }))
    expect(screen.getByRole('button', { name: /game/i })).toHaveAttribute('aria-expanded', 'false')
    rerender(bar({ gameOpen: true }))
    expect(screen.getByRole('button', { name: /game/i })).toHaveAttribute('aria-expanded', 'true')
  })

  it('honours the disabled flags, and never disables Game', () => {
    render(bar({ hintDisabled: true, undoDisabled: true }))
    expect(screen.getByRole('button', { name: /hint/i })).toBeDisabled()
    expect(screen.getByRole('button', { name: /undo/i })).toBeDisabled()
    expect(screen.getByRole('button', { name: /game/i })).toBeEnabled()
  })
})
