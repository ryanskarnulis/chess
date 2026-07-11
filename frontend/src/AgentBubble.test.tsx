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

  it('renders the spider mascot as decoration', () => {
    const { container } = render(<AgentBubble commentary={null} thinking={false} />)
    const spider = container.querySelector('.spider-icon')
    expect(spider).toBeInTheDocument()
    expect(spider).toHaveAttribute('aria-hidden', 'true')
  })
})
