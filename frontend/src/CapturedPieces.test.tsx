import { render, screen, within } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { CapturedPieces } from './CapturedPieces'

describe('CapturedPieces', () => {
  it('renders the captures of the owner it was asked for', () => {
    // The player is white and has captured a black pawn and knight; Glitch
    // (black) has captured a white pawn. Each owner now gets its own row —
    // Glitch's under the agent bubble, the player's under the board — so the
    // component renders one cluster, not both.
    const captured = { white: ['p', 'n'], black: ['p'] }
    render(<CapturedPieces captured={captured} playerColor="white" owner="you" />)
    const you = screen.getByLabelText(/captured by you/i)
    // The dark theme renders glyphs in the light ink color, so the *filled*
    // ("black") glyphs read as white pieces. White's captures — black
    // pieces — must therefore use the hollow set: ♙ pawn, ♘ knight. Every
    // glyph carries U+FE0E so the black pawn (an emoji-default codepoint)
    // renders as text at the same size as the other pieces.
    expect(within(you).getByText('♙︎')).toBeInTheDocument()
    expect(within(you).getByText('♘︎')).toBeInTheDocument()
    // Glitch's pawn belongs to the other row and must not leak into this one.
    expect(within(you).queryByText('♟︎')).not.toBeInTheDocument()
  })

  it("renders Glitch's captures — the player's own color — on its own row", () => {
    const captured = { white: ['p', 'n'], black: ['p'] }
    render(<CapturedPieces captured={captured} playerColor="white" owner="glitch" />)
    const glitch = screen.getByLabelText(/captured by glitch/i)
    // Glitch's capture (a white pawn) gets the filled glyph, which renders
    // light: ♟.
    expect(within(glitch).getByText('♟︎')).toBeInTheDocument()
    expect(within(glitch).queryByText('♙︎')).not.toBeInTheDocument()
  })

  it('follows the player to the black side', () => {
    // Same position, player playing black: the YOU row now holds black's
    // captures and Glitch's holds white's.
    const captured = { white: ['p', 'n'], black: ['p'] }
    render(<CapturedPieces captured={captured} playerColor="black" owner="you" />)
    expect(within(screen.getByLabelText(/captured by you/i)).getByText('♟︎')).toBeInTheDocument()
    render(<CapturedPieces captured={captured} playerColor="black" owner="glitch" />)
    const glitch = screen.getByLabelText(/captured by glitch/i)
    expect(within(glitch).getByText('♙︎')).toBeInTheDocument()
    expect(within(glitch).getByText('♘︎')).toBeInTheDocument()
  })

  // The overflow fix: one glyph per piece *type* with a ×N count, not one
  // glyph per capture. Five types is the ceiling, so the row cannot outgrow
  // the column however long the game runs.
  it('groups repeated captures into one glyph with a count', () => {
    render(
      <CapturedPieces
        captured={{ white: ['p', 'p', 'p', 'n', 'n', 'b'], black: [] }}
        playerColor="white"
        owner="you"
      />,
    )
    const you = screen.getByLabelText(/captured by you/i)
    // Three pawns, one glyph.
    expect(within(you).getAllByText('♙︎')).toHaveLength(1)
    expect(within(you).getByText('×3')).toBeInTheDocument()
    expect(within(you).getByText('×2')).toBeInTheDocument()
    // A lone capture carries no count — "×1" is noise.
    expect(within(you).queryByText('×1')).not.toBeInTheDocument()
  })

  it('reads heaviest piece first, whatever order they were taken in', () => {
    render(
      <CapturedPieces
        captured={{ white: ['p', 'q', 'n'], black: [] }}
        playerColor="white"
        owner="you"
      />,
    )
    const glyphs = screen
      .getByLabelText(/captured by you/i)
      .querySelectorAll('.captured-glyph')
    expect([...glyphs].map((g) => g.textContent)).toEqual(['♕︎', '♘︎', '♙︎'])
  })

  it('drops the YOU and GLITCH micro-labels — position identifies the owner', () => {
    render(
      <CapturedPieces captured={{ white: ['p'], black: ['n'] }} playerColor="white" owner="you" />,
    )
    expect(screen.queryByText('YOU')).not.toBeInTheDocument()
    expect(screen.queryByText('GLITCH')).not.toBeInTheDocument()
    expect(screen.queryByText(/^white$/i)).not.toBeInTheDocument()
    expect(screen.queryByText(/^black$/i)).not.toBeInTheDocument()
  })

  it('shows the material advantage only on the leading side', () => {
    // Pawns traded; the player is additionally up a knight, so +3.
    const captured = { white: ['p', 'n'], black: ['p'] }
    render(<CapturedPieces captured={captured} playerColor="white" owner="you" />)
    expect(within(screen.getByLabelText(/captured by you/i)).getByText('+3')).toBeInTheDocument()
    render(<CapturedPieces captured={captured} playerColor="white" owner="glitch" />)
    const glitch = screen.getByLabelText(/captured by glitch/i)
    expect(within(glitch).queryByText(/^\+/)).not.toBeInTheDocument()
  })

  it('renders nothing inside an empty side but keeps the row', () => {
    // Nothing to show, yet the row survives: it holds its min-height so the
    // board does not jump on the first capture of the game.
    render(<CapturedPieces captured={{ white: [], black: [] }} playerColor="white" owner="you" />)
    const you = screen.getByLabelText(/captured by you/i)
    expect(you).toBeInTheDocument()
    expect(you).toBeEmptyDOMElement()
  })

  // The visible YOU/GLITCH labels are gone, so the accessible name is now the
  // only thing that says whose captures these are — and it spells out the
  // glyphs, which a screen reader cannot read.
  it('names the owner and the pieces in the accessible label', () => {
    render(
      <CapturedPieces
        captured={{ white: ['p', 'p', 'n'], black: [] }}
        playerColor="white"
        owner="you"
      />,
    )
    // Spelled in the order the glyphs read: heaviest piece first.
    expect(
      screen.getByLabelText('Captured by you: 1 knight, 2 pawns, up 5'),
    ).toBeInTheDocument()
  })

  it('says so when a side has captured nothing', () => {
    render(
      <CapturedPieces captured={{ white: [], black: ['p'] }} playerColor="white" owner="you" />,
    )
    expect(screen.getByLabelText('Captured by you: nothing')).toBeInTheDocument()
  })
})
