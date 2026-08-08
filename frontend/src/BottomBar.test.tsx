import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import { BottomBar } from './BottomBar'

function bar(overrides: Partial<Parameters<typeof BottomBar>[0]> = {}) {
  return (
    <BottomBar
      onOptions={vi.fn()}
      onResign={vi.fn()}
      onHint={vi.fn()}
      onUndo={vi.fn()}
      resignDisabled={false}
      hintDisabled={false}
      undoDisabled={false}
      {...overrides}
    />
  )
}

describe('BottomBar', () => {
  it('renders the four control buttons', () => {
    render(bar())
    for (const name of [/options/i, /resign/i, /hint/i, /undo/i]) {
      expect(screen.getByRole('button', { name })).toBeInTheDocument()
    }
  })

  it('fires each callback', () => {
    const onOptions = vi.fn()
    const onResign = vi.fn()
    const onHint = vi.fn()
    const onUndo = vi.fn()
    render(bar({ onOptions, onResign, onHint, onUndo }))
    fireEvent.click(screen.getByRole('button', { name: /options/i }))
    fireEvent.click(screen.getByRole('button', { name: /resign/i }))
    fireEvent.click(screen.getByRole('button', { name: /hint/i }))
    fireEvent.click(screen.getByRole('button', { name: /undo/i }))
    expect(onOptions).toHaveBeenCalled()
    expect(onResign).toHaveBeenCalled()
    expect(onHint).toHaveBeenCalled()
    expect(onUndo).toHaveBeenCalled()
  })

  // Reported from an iPhone: U+21A9 rendered as the blue color ↩️ tile and
  // the flag/bulb as color emoji, against an otherwise monochrome UI. The
  // guard is the character class, not the specific glyphs, so putting any
  // emoji-presentation character back here fails.
  it('marks each button with a monochrome inline icon, never a color emoji', () => {
    const { container } = render(bar())
    expect(container.textContent).not.toMatch(/\p{Extended_Pictographic}/u)
    const icons = container.querySelectorAll('button svg')
    expect(icons).toHaveLength(4)
    for (const svg of icons) expect(svg).toHaveAttribute('aria-hidden', 'true')
  })

  it('honours the disabled flags', () => {
    render(bar({ resignDisabled: true, hintDisabled: true, undoDisabled: true }))
    expect(screen.getByRole('button', { name: /resign/i })).toBeDisabled()
    expect(screen.getByRole('button', { name: /hint/i })).toBeDisabled()
    expect(screen.getByRole('button', { name: /undo/i })).toBeDisabled()
    expect(screen.getByRole('button', { name: /options/i })).toBeEnabled()
  })
})
