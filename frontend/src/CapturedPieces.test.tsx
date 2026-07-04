import { render, screen, within } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { CapturedPieces } from './CapturedPieces'

describe('CapturedPieces', () => {
  it('renders each side captures under its label', () => {
    // White has captured a black pawn and knight; black a white pawn.
    render(<CapturedPieces captured={{ white: ['p', 'n'], black: ['p'] }} />)
    const white = screen.getByLabelText(/captured by white/i)
    // Black glyphs (the pieces white took): ♟ pawn, ♞ knight.
    expect(within(white).getByText('♟')).toBeInTheDocument()
    expect(within(white).getByText('♞')).toBeInTheDocument()
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
