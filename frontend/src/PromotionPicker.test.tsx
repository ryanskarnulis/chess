import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import { PromotionPicker } from './PromotionPicker'

describe('PromotionPicker', () => {
  it('offers all four promotion pieces', () => {
    render(<PromotionPicker color="white" onSelect={vi.fn()} onCancel={vi.fn()} />)
    for (const label of ['Queen', 'Rook', 'Bishop', 'Knight']) {
      expect(screen.getByRole('button', { name: label })).toBeInTheDocument()
    }
  })

  it('reports the chosen piece by its UCI letter', () => {
    const onSelect = vi.fn()
    render(<PromotionPicker color="white" onSelect={onSelect} onCancel={vi.fn()} />)
    fireEvent.click(screen.getByRole('button', { name: 'Knight' }))
    expect(onSelect).toHaveBeenCalledWith('n')
  })

  it('cancels when the backdrop is clicked', () => {
    const onCancel = vi.fn()
    render(<PromotionPicker color="black" onSelect={vi.fn()} onCancel={onCancel} />)
    fireEvent.click(screen.getByRole('dialog', { name: /promotion/i }))
    expect(onCancel).toHaveBeenCalled()
  })

  it('does not cancel when a piece inside the picker is clicked', () => {
    const onCancel = vi.fn()
    render(<PromotionPicker color="white" onSelect={vi.fn()} onCancel={onCancel} />)
    fireEvent.click(screen.getByRole('button', { name: 'Queen' }))
    expect(onCancel).not.toHaveBeenCalled()
  })

  // The tiles wear the board's own artwork, so the piece picked is the piece
  // that lands — not a font glyph that looks nothing like it.
  it("draws each tile in the board's piece art, in the pawn's colour", () => {
    render(<PromotionPicker color="black" onSelect={vi.fn()} onCancel={vi.fn()} />)
    const art = screen.getByRole('button', { name: 'Rook' }).querySelector('.piece-art')
    expect(art).toHaveClass('rook', 'black')
    expect(document.body.textContent).not.toMatch(/[♔-♟]/u)
  })
})
