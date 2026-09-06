import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import { BottomBar } from './BottomBar'

function bar(overrides: Partial<Parameters<typeof BottomBar>[0]> = {}) {
  return (
    <BottomBar
      onOptions={vi.fn()}
      onResign={vi.fn()}
      onDraw={vi.fn()}
      onHint={vi.fn()}
      onUndo={vi.fn()}
      resignDisabled={false}
      drawClaimable={false}
      drawDisabled={false}
      hintDisabled={false}
      undoDisabled={false}
      {...overrides}
    />
  )
}

describe('BottomBar', () => {
  it('renders the five control buttons', () => {
    render(bar())
    for (const name of [/options/i, /resign/i, /draw/i, /hint/i, /undo/i]) {
      expect(screen.getByRole('button', { name })).toBeInTheDocument()
    }
  })

  it('fires each callback', () => {
    const onOptions = vi.fn()
    const onResign = vi.fn()
    const onDraw = vi.fn()
    const onHint = vi.fn()
    const onUndo = vi.fn()
    render(bar({ onOptions, onResign, onDraw, onHint, onUndo }))
    fireEvent.click(screen.getByRole('button', { name: /options/i }))
    fireEvent.click(screen.getByRole('button', { name: /resign/i }))
    fireEvent.click(screen.getByRole('button', { name: /draw/i }))
    fireEvent.click(screen.getByRole('button', { name: /hint/i }))
    fireEvent.click(screen.getByRole('button', { name: /undo/i }))
    expect(onOptions).toHaveBeenCalled()
    expect(onResign).toHaveBeenCalled()
    expect(onDraw).toHaveBeenCalled()
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
    expect(icons).toHaveLength(5)
    for (const svg of icons) expect(svg).toHaveAttribute('aria-hidden', 'true')
  })

  // The draw button follows what the backend says the rules allow: a claim it
  // names, otherwise the offer. Which one is never read off the board here.
  it('reads "Claim draw" while a claim exists and "Offer draw" otherwise', () => {
    const { rerender } = render(bar({ drawClaimable: false }))
    expect(screen.getByRole('button', { name: /offer draw/i })).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: /claim draw/i })).not.toBeInTheDocument()
    rerender(bar({ drawClaimable: true }))
    expect(screen.getByRole('button', { name: /claim draw/i })).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: /offer draw/i })).not.toBeInTheDocument()
  })

  it('honours the disabled flags', () => {
    render(
      bar({ resignDisabled: true, drawDisabled: true, hintDisabled: true, undoDisabled: true }),
    )
    expect(screen.getByRole('button', { name: /resign/i })).toBeDisabled()
    expect(screen.getByRole('button', { name: /draw/i })).toBeDisabled()
    expect(screen.getByRole('button', { name: /hint/i })).toBeDisabled()
    expect(screen.getByRole('button', { name: /undo/i })).toBeDisabled()
    expect(screen.getByRole('button', { name: /options/i })).toBeEnabled()
  })
})
