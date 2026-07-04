import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import { DIFFICULTY_LEVELS } from './difficulty'
import { GameControls } from './GameControls'

function setup(overrides: Partial<React.ComponentProps<typeof GameControls>> = {}) {
  const props = {
    gameOver: false,
    canUndo: true,
    onNewGame: vi.fn(),
    onUndo: vi.fn(),
    onResign: vi.fn(),
    onSetDifficulty: vi.fn(),
    ...overrides,
  }
  render(<GameControls {...props} />)
  return props
}

describe('GameControls', () => {
  it('fires the matching handler for each button', () => {
    const props = setup()
    fireEvent.click(screen.getByRole('button', { name: /new game/i }))
    fireEvent.click(screen.getByRole('button', { name: /undo/i }))
    fireEvent.click(screen.getByRole('button', { name: /resign/i }))
    expect(props.onNewGame).toHaveBeenCalledOnce()
    expect(props.onUndo).toHaveBeenCalledOnce()
    expect(props.onResign).toHaveBeenCalledOnce()
  })

  it('disables undo when there is nothing to take back', () => {
    setup({ canUndo: false })
    expect(screen.getByRole('button', { name: /undo/i })).toBeDisabled()
  })

  it('disables resign once the game is over', () => {
    setup({ gameOver: true })
    expect(screen.getByRole('button', { name: /resign/i })).toBeDisabled()
    // A new game is always allowed.
    expect(screen.getByRole('button', { name: /new game/i })).not.toBeDisabled()
  })

  it('reports the chosen difficulty as a skill level', () => {
    const props = setup()
    const select = screen.getByLabelText(/difficulty/i)
    const advanced = DIFFICULTY_LEVELS.find((l) => l.label === 'Advanced')!
    fireEvent.change(select, { target: { value: String(advanced.skillLevel) } })
    expect(props.onSetDifficulty).toHaveBeenCalledWith(advanced.skillLevel)
  })
})
