import { render, screen, within } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { CapturedPieces } from './CapturedPieces'

describe('CapturedPieces', () => {
  it('renders each side captures under its label', () => {
    // White has captured a black pawn and knight; black a white pawn.
    render(<CapturedPieces captured={{ white: ['p', 'n'], black: ['p'] }} />)
    const white = screen.getByLabelText(/captured by white/i)
    // The dark theme renders glyphs in the light ink color, so the *filled*
    // ("black") glyphs read as white pieces. White's captures — black
    // pieces — must therefore use the hollow set: ♙ pawn, ♘ knight.
    expect(within(white).getByText('♙')).toBeInTheDocument()
    expect(within(white).getByText('♘')).toBeInTheDocument()
    const black = screen.getByLabelText(/captured by black/i)
    // Black's capture (a white pawn) gets the filled glyph, which renders
    // light: ♟.
    expect(within(black).getByText('♟')).toBeInTheDocument()
  })

  it('shows the material advantage for the side that is ahead', () => {
    // Pawns traded; white is additionally up a knight, so +3.
    render(<CapturedPieces captured={{ white: ['p', 'n'], black: ['p'] }} />)
    const white = screen.getByLabelText(/captured by white/i)
    expect(within(white).getByText('+3')).toBeInTheDocument()
    // Even material shows no advantage badge.
    const black = screen.getByLabelText(/captured by black/i)
    expect(within(black).queryByText(/^\+/)).not.toBeInTheDocument()
  })

  it('renders empty sides without crashing', () => {
    render(<CapturedPieces captured={{ white: [], black: [] }} />)
    expect(screen.getByLabelText(/captured by white/i)).toBeInTheDocument()
  })
})
