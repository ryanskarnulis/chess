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
    tier: 'casual' as string | null,
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

  it('reports the chosen difficulty as a tier name', () => {
    const props = setup()
    const select = screen.getByLabelText(/difficulty/i)
    const advanced = DIFFICULTY_LEVELS.find((l) => l.label === 'Advanced')!
    fireEvent.change(select, { target: { value: advanced.tier } })
    expect(props.onSetDifficulty).toHaveBeenCalledWith(advanced.tier)
  })

  it('shows the server-confirmed tier', () => {
    setup({ tier: 'advanced' })
    expect(screen.getByLabelText(/difficulty/i)).toHaveValue('advanced')
  })

  it('shows no tier before settings load', () => {
    setup({ tier: null })
    expect(screen.getByLabelText(/difficulty/i)).toHaveValue('')
  })

  it('shows no tier for a strength outside the presets', () => {
    // e.g. the agent set a raw skill/elo — don't lie by snapping to a tier.
    setup({ tier: 'custom' })
    expect(screen.getByLabelText(/difficulty/i)).toHaveValue('')
  })
})
