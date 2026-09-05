import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { AgentBubble } from './AgentBubble'

const PGN = '[Event "Casual game"]\n[White "Player"]\n\n1. e4 e5 2. Nf3 Nc6 *'

afterEach(() => vi.unstubAllGlobals())

describe('AgentBubble', () => {
  it('shows the agent commentary in the bubble', () => {
    render(<AgentBubble commentary="Nice move!" thinking={false} />)
    expect(screen.getByRole('status')).toHaveTextContent('Nice move!')
  })


  it('draws a paragraph break as a gap, not an empty line', () => {
    // A move reply is "reaction\n\nNf6.", and at pre-wrap the blank line was a
    // whole one of the three lines the clamped bubble shows — on a phone it
    // scrolled the engine's move out of sight. The break becomes a block
    // boundary (App.css spaces them); a single newline stays inside its
    // paragraph for pre-wrap to draw.
    render(<AgentBubble commentary={'Bold.\n\nNf6.'} thinking={false} />)
    const paragraphs = screen.getByRole('status').querySelectorAll('.bubble-paragraph')
    expect(Array.from(paragraphs, (p) => p.textContent)).toEqual(['Bold.', 'Nf6.'])
    render(<AgentBubble commentary={'one\ntwo'} thinking={false} />)
    const single = screen.getAllByRole('status')[1].querySelectorAll('.bubble-paragraph')
    expect(Array.from(single, (p) => p.textContent)).toEqual(['one\ntwo'])
  })
  it('shows a thinking hint while a command is in flight', () => {
    render(<AgentBubble commentary="Old reply" thinking={true} />)
    expect(screen.getByRole('status')).toHaveTextContent(/thinking/i)
  })

  it('greets when there is no commentary yet', () => {
    render(<AgentBubble commentary={null} thinking={false} />)
    expect(screen.getByRole('status')).toHaveTextContent(/your move/i)
  })

  it('prefers a live progress line over the generic hint', () => {
    render(
      <AgentBubble
        commentary="Old reply"
        thinking={true}
        progress="Stockfish is calculating"
      />,
    )
    expect(screen.getByRole('status')).toHaveTextContent('Stockfish is calculating')
  })

  it('shows progress even with no command in flight', () => {
    // A dragged move is a turn too, and nothing else in the UI would say so.
    render(
      <AgentBubble commentary="Old reply" thinking={false} progress="Validating your move" />,
    )
    expect(screen.getByRole('status')).toHaveTextContent('Validating your move')
  })

  it('falls back to the last reply once the turn is over', () => {
    render(<AgentBubble commentary="Nice move!" thinking={false} progress={null} />)
    expect(screen.getByRole('status')).toHaveTextContent('Nice move!')
  })

  it('renders the spider mascot as decoration', () => {
    const { container } = render(<AgentBubble commentary={null} thinking={false} />)
    const spider = container.querySelector('.spider-icon')
    expect(spider).toBeInTheDocument()
    expect(spider).toHaveAttribute('aria-hidden', 'true')
  })

  // --- the PGN a reply exported ------------------------------------------

  it('offers nothing to copy when the reply exported no PGN', () => {
    render(<AgentBubble commentary="Nice move!" thinking={false} />)
    expect(screen.queryByRole('button', { name: /copy pgn/i })).not.toBeInTheDocument()
  })

  it('copies an exported PGN and confirms in place', async () => {
    const writeText = vi.fn(() => Promise.resolve())
    vi.stubGlobal('navigator', { ...window.navigator, clipboard: { writeText } })
    render(<AgentBubble commentary="Exported. Grab it below." thinking={false} pgn={PGN} />)
    fireEvent.click(screen.getByRole('button', { name: /copy pgn/i }))
    await waitFor(() => expect(writeText).toHaveBeenCalledWith(PGN))
    expect(screen.getByRole('button', { name: /copied/i })).toBeInTheDocument()
  })

  it('says so when the clipboard refuses', async () => {
    // A denied permission or an insecure origin: the player has to know the
    // copy did not happen, or they paste whatever was there before.
    const writeText = vi.fn(() => Promise.reject(new Error('denied')))
    vi.stubGlobal('navigator', { ...window.navigator, clipboard: { writeText } })
    render(<AgentBubble commentary="Exported." thinking={false} pgn={PGN} />)
    fireEvent.click(screen.getByRole('button', { name: /copy pgn/i }))
    expect(await screen.findByRole('button', { name: /copy failed/i })).toBeInTheDocument()
  })

  it('keeps the notation itself readable, folded away', () => {
    // The dump in the bubble is what this replaced — but a browser with no
    // clipboard still needs some way to get at the moves.
    const { container } = render(
      <AgentBubble commentary="Exported." thinking={false} pgn={PGN} />,
    )
    const details = container.querySelector('details')
    expect(details).not.toHaveAttribute('open')
    expect(screen.getByText(/show pgn/i)).toBeInTheDocument()
    expect(container.querySelector('pre')?.textContent).toBe(PGN)
  })

  it('offers a fresh copy for a second export', async () => {
    // Same bubble, a different game: a button still reading "Copied ✓" would
    // tell the player they have something they do not.
    const writeText = vi.fn(() => Promise.resolve())
    vi.stubGlobal('navigator', { ...window.navigator, clipboard: { writeText } })
    const { rerender } = render(
      <AgentBubble commentary="Exported." thinking={false} pgn={PGN} />,
    )
    fireEvent.click(screen.getByRole('button', { name: /copy pgn/i }))
    expect(await screen.findByRole('button', { name: /copied/i })).toBeInTheDocument()
    rerender(<AgentBubble commentary="Exported." thinking={false} pgn={`${PGN} 1-0`} />)
    expect(screen.getByRole('button', { name: /copy pgn/i })).toBeInTheDocument()
  })

  it('leaves the reply itself alone', () => {
    // The bubble says the export is ready; the notation lives beside it, not
    // in the words.
    render(<AgentBubble commentary="Exported." thinking={false} pgn={PGN} />)
    expect(screen.getByRole('status')).toHaveTextContent('Exported.')
    expect(screen.getByRole('status')).not.toHaveTextContent('[Event')
  })
})
