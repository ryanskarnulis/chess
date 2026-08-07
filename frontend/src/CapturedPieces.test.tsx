import { render, screen, within } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { CapturedPieces } from './CapturedPieces'

describe('CapturedPieces', () => {
  it('renders each side captures under its role cluster', () => {
    // The player is white and has captured a black pawn and knight; Glitch
    // (black) has captured a white pawn.
    render(
      <CapturedPieces captured={{ white: ['p', 'n'], black: ['p'] }} playerColor="white" />,
    )
    const you = screen.getByLabelText(/captured by you/i)
    // The dark theme renders glyphs in the light ink color, so the *filled*
    // ("black") glyphs read as white pieces. White's captures — black
    // pieces — must therefore use the hollow set: ♙ pawn, ♘ knight. Every
    // glyph carries U+FE0E so the black pawn (an emoji-default codepoint)
    // renders as text at the same size as the other pieces.
    expect(within(you).getByText('♙︎')).toBeInTheDocument()
    expect(within(you).getByText('♘︎')).toBeInTheDocument()
    const glitch = screen.getByLabelText(/captured by glitch/i)
    // Glitch's capture (a white pawn) gets the filled glyph, which renders
    // light: ♟.
    expect(within(glitch).getByText('♟︎')).toBeInTheDocument()
  })

  it('follows the player to the black side', () => {
    // Same position, player playing black: the YOU cluster now holds black's
    // captures and Glitch's holds white's.
    render(
      <CapturedPieces captured={{ white: ['p', 'n'], black: ['p'] }} playerColor="black" />,
    )
    const you = screen.getByLabelText(/captured by you/i)
    expect(within(you).getByText('♟︎')).toBeInTheDocument()
    const glitch = screen.getByLabelText(/captured by glitch/i)
    expect(within(glitch).getByText('♙︎')).toBeInTheDocument()
    expect(within(glitch).getByText('♘︎')).toBeInTheDocument()
  })

  it('labels the clusters YOU and GLITCH, never by chess color', () => {
    // The old two-cluster row was anonymous; the merged row names the
    // owners. The micro-labels are the only visible text — the color words
    // stay out (they are what the labels replace).
    render(
      <CapturedPieces captured={{ white: ['p'], black: ['n'] }} playerColor="white" />,
    )
    expect(screen.getByText('YOU')).toBeInTheDocument()
    expect(screen.getByText('GLITCH')).toBeInTheDocument()
    expect(screen.queryByRole('heading')).not.toBeInTheDocument()
    expect(screen.queryByText(/^white$/i)).not.toBeInTheDocument()
    expect(screen.queryByText(/^black$/i)).not.toBeInTheDocument()
  })

  it('collapses a cluster with no captures, label included', () => {
    render(<CapturedPieces captured={{ white: ['p'], black: [] }} playerColor="white" />)
    expect(screen.getByText('YOU')).toBeInTheDocument()
    // Glitch has taken nothing: no stray GLITCH label sitting on an empty
    // cluster (the empty flex sibling still keeps the chip centered).
    expect(screen.queryByText('GLITCH')).not.toBeInTheDocument()
  })

  it('centers the children between the two clusters', () => {
    // The turn chip arrives as children so the row owns the layout while the
    // app owns the status text.
    render(
      <CapturedPieces captured={{ white: ['p'], black: ['p'] }} playerColor="white">
        <p>White to move</p>
      </CapturedPieces>,
    )
    const row = screen.getByLabelText(/game status/i)
    const chip = screen.getByText('White to move')
    const you = screen.getByLabelText(/captured by you/i)
    const glitch = screen.getByLabelText(/captured by glitch/i)
    expect(row).toContainElement(chip)
    // DOM order: YOU cluster, then the chip, then Glitch's cluster.
    expect(you.compareDocumentPosition(chip) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy()
    expect(chip.compareDocumentPosition(glitch) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy()
  })

  it('shows the material advantage for the side that is ahead', () => {
    // Pawns traded; the player is additionally up a knight, so +3.
    render(
      <CapturedPieces captured={{ white: ['p', 'n'], black: ['p'] }} playerColor="white" />,
    )
    const you = screen.getByLabelText(/captured by you/i)
    expect(within(you).getByText('+3')).toBeInTheDocument()
    // Even material shows no advantage badge.
    const glitch = screen.getByLabelText(/captured by glitch/i)
    expect(within(glitch).queryByText(/^\+/)).not.toBeInTheDocument()
  })

  it('renders empty sides without crashing', () => {
    render(<CapturedPieces captured={{ white: [], black: [] }} playerColor="white" />)
    expect(screen.getByLabelText(/captured by you/i)).toBeInTheDocument()
  })
})
