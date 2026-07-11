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

  it('honours the disabled flags', () => {
    render(bar({ resignDisabled: true, hintDisabled: true, undoDisabled: true }))
    expect(screen.getByRole('button', { name: /resign/i })).toBeDisabled()
    expect(screen.getByRole('button', { name: /hint/i })).toBeDisabled()
    expect(screen.getByRole('button', { name: /undo/i })).toBeDisabled()
    expect(screen.getByRole('button', { name: /options/i })).toBeEnabled()
  })
})
