import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import { CommandBox } from './CommandBox'

function setup(overrides: Partial<React.ComponentProps<typeof CommandBox>> = {}) {
  const props = {
    onSubmit: vi.fn(),
    commentary: null as string | null,
    thinking: false,
    ...overrides,
  }
  render(<CommandBox {...props} />)
  return props
}

describe('CommandBox', () => {
  it('submits the typed command and clears the input', () => {
    const props = setup()
    const input = screen.getByLabelText(/command/i) as HTMLInputElement
    fireEvent.change(input, { target: { value: 'play e4' } })
    fireEvent.click(screen.getByRole('button', { name: /send/i }))
    expect(props.onSubmit).toHaveBeenCalledWith('play e4')
    expect(input.value).toBe('')
  })

  it('trims whitespace and never submits a blank command', () => {
    const props = setup()
    const input = screen.getByLabelText(/command/i)
    // Whitespace-only: the send button stays disabled and nothing is sent.
    fireEvent.change(input, { target: { value: '   ' } })
    expect(screen.getByRole('button', { name: /send/i })).toBeDisabled()
    fireEvent.submit(input.closest('form')!)
    expect(props.onSubmit).not.toHaveBeenCalled()
    // A padded command is trimmed before it is sent.
    fireEvent.change(input, { target: { value: '  castle  ' } })
    fireEvent.click(screen.getByRole('button', { name: /send/i }))
    expect(props.onSubmit).toHaveBeenCalledWith('castle')
  })

  it('displays the agent commentary', () => {
    setup({ commentary: 'A bold opening!' })
    expect(screen.getByText('A bold opening!')).toBeInTheDocument()
  })

  it('shows a thinking indicator and disables input while the agent works', () => {
    setup({ thinking: true, commentary: 'stale reply' })
    expect(screen.getByLabelText(/command/i)).toBeDisabled()
    expect(screen.getByRole('button', { name: /send/i })).toBeDisabled()
    expect(screen.getByText(/thinking/i)).toBeInTheDocument()
    // The prior commentary is hidden while a new reply is pending.
    expect(screen.queryByText('stale reply')).not.toBeInTheDocument()
  })
})
