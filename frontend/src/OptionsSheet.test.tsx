import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import { OptionsSheet } from './OptionsSheet'

function sheet(overrides: Partial<Parameters<typeof OptionsSheet>[0]> = {}) {
  return (
    <OptionsSheet
      open={true}
      onClose={vi.fn()}
      onNewGame={vi.fn()}
      tier="casual"
      onSetDifficulty={vi.fn()}
      voiceOutput={false}
      onToggleVoice={vi.fn()}
      brain={null}
      onSetBrain={vi.fn()}
      {...overrides}
    />
  )
}

describe('OptionsSheet', () => {
  it('renders nothing when closed', () => {
    render(sheet({ open: false }))
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument()
  })

  it('renders the voice row with an inline icon, never a color emoji', () => {
    const { container } = render(sheet())
    expect(container.textContent).not.toMatch(/\p{Extended_Pictographic}/u)
    expect(container.querySelector('button svg')).toHaveAttribute('aria-hidden', 'true')
  })

  it('shows a dialog with the game options when open', () => {
    render(sheet())
    expect(screen.getByRole('dialog', { name: /options/i })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /new game/i })).toBeInTheDocument()
    expect(screen.getByRole('combobox')).toHaveValue('casual')
  })

  it('starts a new game and closes', () => {
    const onNewGame = vi.fn()
    const onClose = vi.fn()
    render(sheet({ onNewGame, onClose }))
    fireEvent.click(screen.getByRole('button', { name: /new game/i }))
    expect(onNewGame).toHaveBeenCalled()
    expect(onClose).toHaveBeenCalled()
  })

  it('changes difficulty', () => {
    const onSetDifficulty = vi.fn()
    render(sheet({ onSetDifficulty }))
    fireEvent.change(screen.getByRole('combobox'), { target: { value: 'advanced' } })
    expect(onSetDifficulty).toHaveBeenCalledWith('advanced')
  })

  it('toggles voice output', () => {
    const onToggleVoice = vi.fn()
    render(sheet({ onToggleVoice }))
    fireEvent.click(screen.getByRole('button', { name: /voice output on/i }))
    expect(onToggleVoice).toHaveBeenCalledWith(true)
  })

  it('hides the voice toggle until the setting is known', () => {
    render(sheet({ voiceOutput: null }))
    expect(screen.queryByRole('button', { name: /voice output/i })).not.toBeInTheDocument()
  })

  it('closes from the close button and the backdrop', () => {
    const onClose = vi.fn()
    const { container } = render(sheet({ onClose }))
    fireEvent.click(screen.getByRole('button', { name: /close/i }))
    expect(onClose).toHaveBeenCalledTimes(1)
    const backdrop = container.querySelector('.options-backdrop')
    expect(backdrop).not.toBeNull()
    fireEvent.click(backdrop as Element)
    expect(onClose).toHaveBeenCalledTimes(2)
  })

  describe('brain picker', () => {
    const brain = {
      brain: 'gemma-4-26b-a4b',
      cold: false,
      choices: ['gemma-4-12b', 'gemma-4-26b-a4b'],
    }

    it('names the brains by feel, with the chosen one selected and its hint', () => {
      render(sheet({ brain }))
      const picker = screen.getByRole('combobox', { name: /brain/i })
      expect(picker).toHaveValue('gemma-4-26b-a4b')
      expect(screen.getByRole('option', { name: 'Fast' })).toHaveValue('gemma-4-12b')
      expect(screen.getByRole('option', { name: 'Balanced' })).toHaveValue('gemma-4-26b-a4b')
      // Model names stay out of the player's way.
      expect(screen.queryByText(/gemma/i)).not.toBeInTheDocument()
      expect(screen.getByText(/5–10 seconds a turn/)).toBeInTheDocument()
    })

    it('asks for the picked brain', () => {
      const onSetBrain = vi.fn()
      render(sheet({ brain, onSetBrain }))
      fireEvent.change(screen.getByRole('combobox', { name: /brain/i }), {
        target: { value: 'gemma-4-12b' },
      })
      expect(onSetBrain).toHaveBeenCalledWith('gemma-4-12b')
    })

    it('is hidden when the brain cannot be chosen', () => {
      render(sheet({ brain: { ...brain, choices: [] } }))
      expect(screen.queryByRole('combobox', { name: /brain/i })).not.toBeInTheDocument()
    })

    it('shows a brain it has no name for under its id, with no made-up hint', () => {
      render(sheet({ brain: { brain: 'mystery-9b', cold: false, choices: ['mystery-9b'] } }))
      expect(screen.getByRole('option', { name: 'mystery-9b' })).toBeInTheDocument()
      expect(document.querySelector('.brain-hint')).toBeNull()
    })
  })
})
