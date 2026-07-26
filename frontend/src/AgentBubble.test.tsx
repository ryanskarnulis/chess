import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { AgentBubble } from './AgentBubble'

describe('AgentBubble', () => {
  it('shows the agent commentary in the bubble', () => {
    render(<AgentBubble commentary="Nice move!" thinking={false} />)
    expect(screen.getByRole('status')).toHaveTextContent('Nice move!')
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
})
