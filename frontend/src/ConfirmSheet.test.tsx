import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import { ConfirmSheet } from './ConfirmSheet'

describe('ConfirmSheet', () => {
  it('asks before a resignation, in the sheet’s own words', () => {
    render(<ConfirmSheet question={{ op: 'resign' }} onAnswer={vi.fn()} />)
    expect(screen.getByRole('alertdialog', { name: 'Resign this game?' })).toBeInTheDocument()
    expect(screen.getByText(/glitch takes the win/i)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Resign' })).toHaveClass('confirm-danger')
    expect(screen.getByRole('button', { name: 'Keep playing' })).toBeInTheDocument()
  })

  it.each([
    ['new_game', 'Start a new game?', 'New game'],
    ['claim_draw', 'Claim the draw?', 'Claim draw'],
  ])('has copy for %s', (op, heading, yes) => {
    render(<ConfirmSheet question={{ op, detail: 'ignored' }} onAnswer={vi.fn()} />)
    expect(screen.getByRole('alertdialog', { name: heading })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: yes })).toBeInTheDocument()
  })

  it('asks an op it has no copy for in the gate’s own words', () => {
    render(
      <ConfirmSheet
        question={{ op: 'save_game', detail: 'Replace it?' }}
        onAnswer={vi.fn()}
      />,
    )
    expect(screen.getByRole('alertdialog', { name: 'Replace it?' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Yes' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Cancel' })).toBeInTheDocument()
  })

  it('answers yes from the primary button and no from Keep playing or the backdrop', () => {
    const onAnswer = vi.fn()
    const { container } = render(<ConfirmSheet question={{ op: 'resign' }} onAnswer={onAnswer} />)
    fireEvent.click(screen.getByRole('button', { name: 'Resign' }))
    fireEvent.click(screen.getByRole('button', { name: 'Keep playing' }))
    fireEvent.click(container.querySelector('.confirm-backdrop')!)
    expect(onAnswer.mock.calls).toEqual([[true], [false], [false]])
  })
})
