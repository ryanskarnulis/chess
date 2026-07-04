import { render, screen, within } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { MoveHistory } from './MoveHistory'

describe('MoveHistory', () => {
  it('shows a placeholder before any moves are played', () => {
    render(<MoveHistory history={[]} />)
    expect(screen.getByText(/no moves yet/i)).toBeInTheDocument()
  })

  it('lists moves paired and numbered', () => {
    render(<MoveHistory history={['e4', 'e5', 'Nf3', 'Nc6', 'Bb5']} />)
    const rows = screen.getAllByRole('row')
    expect(rows).toHaveLength(3)
    // Third row: an incomplete pair (white move only).
    expect(within(rows[2]).getByText('3')).toBeInTheDocument()
    expect(within(rows[2]).getByText('Bb5')).toBeInTheDocument()
    expect(within(rows[0]).getByText('e4')).toBeInTheDocument()
    expect(within(rows[0]).getByText('e5')).toBeInTheDocument()
  })
})
