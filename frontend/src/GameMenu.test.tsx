import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { GameMenu } from './GameMenu'
import { unlockAudio } from './tts'

// Turning voice on primes mobile playback, so the switch must call it.
vi.mock('./tts', () => ({ unlockAudio: vi.fn(), playText: vi.fn() }))

afterEach(() => {
  vi.unstubAllGlobals()
  vi.mocked(unlockAudio).mockClear()
})

function menu(overrides: Partial<Parameters<typeof GameMenu>[0]> = {}) {
  return (
    <GameMenu
      open={true}
      onClose={vi.fn()}
      onNewGame={vi.fn()}
      drawClaimable={false}
      gameOver={false}
      onDraw={vi.fn()}
      onResign={vi.fn()}
      tier="casual"
      glitchAvailable={true}
      onSetDifficulty={vi.fn()}
      voiceOutput={false}
      onToggleVoice={vi.fn()}
      brain={null}
      onSetBrain={vi.fn()}
      {...overrides}
    />
  )
}

describe('GameMenu', () => {
  it('renders nothing when closed', () => {
    render(menu({ open: false }))
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument()
  })

  it('shows the Game and Glitch groups when open', () => {
    render(menu())
    expect(screen.getByRole('dialog', { name: 'Game' })).toBeInTheDocument()
    const groups = screen.getAllByRole('heading').map((h) => h.textContent)
    expect(groups).toEqual(['Game', 'Glitch'])
    expect(screen.getByRole('combobox', { name: /difficulty/i })).toHaveValue('casual')
    expect(document.body.textContent).not.toMatch(/\p{Extended_Pictographic}/u)
  })

  it('starts a new game and closes', () => {
    const onNewGame = vi.fn()
    const onClose = vi.fn()
    render(menu({ onNewGame, onClose }))
    fireEvent.click(screen.getByRole('button', { name: /new game/i }))
    expect(onNewGame).toHaveBeenCalledWith()
    expect(onClose).toHaveBeenCalled()
  })

  it('copies the PGN and says so in place', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(() => Promise.resolve({ ok: true, json: () => Promise.resolve({ pgn: '1. e4' }) })),
    )
    const writeText = vi.fn(() => Promise.resolve())
    vi.stubGlobal('navigator', { ...window.navigator, clipboard: { writeText } })
    render(menu())
    fireEvent.click(screen.getByRole('button', { name: /copy pgn/i }))
    await waitFor(() => expect(writeText).toHaveBeenCalledWith('1. e4'))
    expect(screen.getByRole('button', { name: /copied/i })).toBeInTheDocument()
  })

  // The draw row follows what the backend says the rules allow: a claim it
  // names, otherwise the offer. Which one is never read off the board here.
  it('reads "Claim draw" while a claim exists and "Offer draw" otherwise', () => {
    const onDraw = vi.fn()
    const onClose = vi.fn()
    const { rerender } = render(menu({ onDraw, onClose }))
    expect(screen.queryByRole('button', { name: /claim draw/i })).not.toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: /offer draw/i }))
    expect(onDraw).toHaveBeenCalledTimes(1)
    expect(onClose).toHaveBeenCalledTimes(1)
    rerender(menu({ onDraw, onClose, drawClaimable: true }))
    expect(screen.getByRole('button', { name: /claim draw/i })).toBeInTheDocument()
  })

  it('closes and asks the parent to confirm a resignation', () => {
    const onResign = vi.fn()
    const onClose = vi.fn()
    render(menu({ onResign, onClose }))
    fireEvent.click(screen.getByRole('button', { name: /resign/i }))
    expect(onClose).toHaveBeenCalled()
    expect(onResign).toHaveBeenCalledTimes(1)
  })

  it('offers no draw or resignation once the game is over', () => {
    render(menu({ gameOver: true }))
    expect(screen.getByRole('button', { name: /offer draw/i })).toBeDisabled()
    expect(screen.getByRole('button', { name: /resign/i })).toBeDisabled()
    expect(screen.getByRole('button', { name: /new game/i })).toBeEnabled()
  })

  it('offers the Glitch tier, and says when it is the one picked', () => {
    const { rerender } = render(menu({ glitchAvailable: true }))
    expect(screen.getByRole('option', { name: 'Glitch' })).toBeEnabled()
    expect(screen.queryByText(/picks his own moves/)).not.toBeInTheDocument()
    rerender(menu({ tier: 'glitch', glitchAvailable: true }))
    expect(screen.getByRole('combobox', { name: /difficulty/i })).toHaveValue('glitch')
    expect(screen.getByText(/picks his own moves/)).toBeInTheDocument()
  })

  it('disables the Glitch tier with the reason when there is no brain', () => {
    render(menu({ glitchAvailable: false }))
    expect(screen.getByRole('option', { name: 'Glitch (needs a brain)' })).toBeDisabled()
  })

  it('leaves the Glitch tier to the server when it has not said', () => {
    render(menu({ glitchAvailable: null }))
    expect(screen.getByRole('option', { name: 'Glitch' })).toBeEnabled()
  })

  it('says the brain also plays his moves, only in the Glitch tier', () => {
    const brain = { brain: 'gemma-4-26b-a4b', cold: false, choices: ['gemma-4-26b-a4b'] }
    const { rerender } = render(menu({ brain }))
    expect(screen.queryByText(/also plays his moves/)).not.toBeInTheDocument()
    rerender(menu({ brain, tier: 'glitch' }))
    expect(screen.getByText(/also plays his moves/)).toBeInTheDocument()
  })

  it('changes difficulty', () => {
    const onSetDifficulty = vi.fn()
    render(menu({ onSetDifficulty }))
    fireEvent.change(screen.getByRole('combobox', { name: /difficulty/i }), {
      target: { value: 'advanced' },
    })
    expect(onSetDifficulty).toHaveBeenCalledWith('advanced')
  })

  it('switches speaking replies aloud, priming audio in the gesture', () => {
    const onToggleVoice = vi.fn()
    const { rerender } = render(menu({ onToggleVoice, voiceOutput: false }))
    const toggle = screen.getByRole('switch', { name: /speak replies aloud/i })
    expect(toggle).toHaveAttribute('aria-checked', 'false')
    fireEvent.click(toggle)
    expect(unlockAudio).toHaveBeenCalled()
    expect(onToggleVoice).toHaveBeenCalledWith(true)
    rerender(menu({ onToggleVoice, voiceOutput: true }))
    expect(screen.getByRole('switch')).toHaveAttribute('aria-checked', 'true')
    fireEvent.click(screen.getByRole('switch'))
    expect(onToggleVoice).toHaveBeenLastCalledWith(false)
  })

  it('hides the voice switch until the setting is known', () => {
    render(menu({ voiceOutput: null }))
    expect(screen.queryByRole('switch')).not.toBeInTheDocument()
  })

  it('closes from the backdrop', () => {
    const onClose = vi.fn()
    const { container } = render(menu({ onClose }))
    fireEvent.click(container.querySelector('.game-menu-backdrop')!)
    expect(onClose).toHaveBeenCalledTimes(1)
  })

  describe('brain picker', () => {
    const brain = {
      brain: 'gemma-4-26b-a4b',
      cold: false,
      choices: ['gemma-4-12b', 'gemma-4-26b-a4b'],
    }

    it('names the brains by feel, with the chosen one selected and its hint', () => {
      render(menu({ brain }))
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
      render(menu({ brain, onSetBrain }))
      fireEvent.change(screen.getByRole('combobox', { name: /brain/i }), {
        target: { value: 'gemma-4-12b' },
      })
      expect(onSetBrain).toHaveBeenCalledWith('gemma-4-12b')
    })

    it('is hidden when the brain cannot be chosen', () => {
      render(menu({ brain: { ...brain, choices: [] } }))
      expect(screen.queryByRole('combobox', { name: /brain/i })).not.toBeInTheDocument()
    })

    it('shows a brain it has no name for under its id, with no made-up hint', () => {
      render(menu({ brain: { brain: 'mystery-9b', cold: false, choices: ['mystery-9b'] } }))
      expect(screen.getByRole('option', { name: 'mystery-9b' })).toBeInTheDocument()
      expect(document.querySelector('.brain-hint')).toBeNull()
    })
  })
})
