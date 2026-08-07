import { fireEvent, render, screen } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { CommandBox } from './CommandBox'
import { unlockAudio } from './tts'

// Mobile browsers only allow playback primed inside a user gesture, so the
// gesture handlers must call unlockAudio synchronously.
vi.mock('./tts', () => ({ unlockAudio: vi.fn(), playText: vi.fn() }))

// The real GatewayLink renders nothing off the gateway domain (jsdom is
// localhost), so its placement inside the row is pinned through a stub.
vi.mock('./GatewayLink', () => ({
  GatewayLink: () => <a data-testid="gateway-link" href="/" aria-label="Back to The Web" />,
}))

beforeEach(() => {
  vi.mocked(unlockAudio).mockClear()
})

function setup(overrides: Partial<React.ComponentProps<typeof CommandBox>> = {}) {
  const props = {
    onSubmit: vi.fn(),
    commentary: null as string | null,
    thinking: false,
    voiceOutput: false as boolean | null,
    onToggleVoice: vi.fn(),
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

  it('shows the ↑ submit only while the input holds text', () => {
    // Voice-first row: no standing Send button. The circular ↑ docks inside
    // the input's right edge the moment there is something to send.
    setup()
    expect(screen.queryByRole('button', { name: /send/i })).not.toBeInTheDocument()
    const input = screen.getByLabelText(/command/i)
    fireEvent.change(input, { target: { value: 'knight takes e5' } })
    expect(screen.getByRole('button', { name: /send/i })).toBeInTheDocument()
    fireEvent.change(input, { target: { value: '' } })
    expect(screen.queryByRole('button', { name: /send/i })).not.toBeInTheDocument()
  })

  it('trims whitespace and never submits a blank command', () => {
    const props = setup()
    const input = screen.getByLabelText(/command/i)
    // Whitespace-only: no ↑ to press, and a forced submit sends nothing.
    fireEvent.change(input, { target: { value: '   ' } })
    expect(screen.queryByRole('button', { name: /send/i })).not.toBeInTheDocument()
    fireEvent.submit(input.closest('form')!)
    expect(props.onSubmit).not.toHaveBeenCalled()
    // A padded command is trimmed before it is sent.
    fireEvent.change(input, { target: { value: '  castle  ' } })
    fireEvent.click(screen.getByRole('button', { name: /send/i }))
    expect(props.onSubmit).toHaveBeenCalledWith('castle')
  })

  it('submits on Enter without the ↑ rendered', () => {
    // The ↑ is the form's submit button, but Enter-to-send must not depend
    // on it being in the DOM (it is hidden while the input is empty, and the
    // guard inside handleSubmit owns the empty case).
    const props = setup()
    const input = screen.getByLabelText(/command/i)
    fireEvent.change(input, { target: { value: 'play e4' } })
    fireEvent.submit(input.closest('form')!)
    expect(props.onSubmit).toHaveBeenCalledWith('play e4')
  })

  it('hides the ↑ while the agent is thinking', () => {
    // The old Send button disabled itself in the locked states; the ↑ simply
    // leaves (the input is disabled in those states anyway).
    const props = {
      onSubmit: vi.fn(),
      commentary: null as string | null,
      thinking: false,
      voiceOutput: false as boolean | null,
      onToggleVoice: vi.fn(),
    }
    const { rerender } = render(<CommandBox {...props} />)
    fireEvent.change(screen.getByLabelText(/command/i), { target: { value: 'play e4' } })
    expect(screen.getByRole('button', { name: /send/i })).toBeInTheDocument()
    rerender(<CommandBox {...props} thinking={true} />)
    expect(screen.queryByRole('button', { name: /send/i })).not.toBeInTheDocument()
  })

  it('hides the ↑ in direct mode', () => {
    const props = {
      onSubmit: vi.fn(),
      commentary: null as string | null,
      thinking: false,
      voiceOutput: false as boolean | null,
      onToggleVoice: vi.fn(),
    }
    const { rerender } = render(<CommandBox {...props} />)
    fireEvent.change(screen.getByLabelText(/command/i), { target: { value: 'play e4' } })
    rerender(<CommandBox {...props} disabled={true} />)
    expect(screen.queryByRole('button', { name: /send/i })).not.toBeInTheDocument()
    expect(screen.getByLabelText(/command/i)).toBeDisabled()
  })

  it('renders the gateway link as the leftmost control in the row', () => {
    setup()
    const form = screen.getByLabelText(/command/i).closest('form')!
    expect(form.firstElementChild).toBe(screen.getByTestId('gateway-link'))
  })

  it('unlocks audio playback inside the submit gesture', () => {
    setup()
    const input = screen.getByLabelText(/command/i)
    fireEvent.change(input, { target: { value: 'play e4' } })
    fireEvent.click(screen.getByRole('button', { name: /send/i }))
    expect(unlockAudio).toHaveBeenCalled()
  })

  it('unlocks audio playback when voice output is toggled on', () => {
    setup({ voiceOutput: false })
    fireEvent.click(screen.getByRole('button', { name: /turn voice output on/i }))
    expect(unlockAudio).toHaveBeenCalled()
  })

  it('displays the agent commentary', () => {
    setup({ commentary: 'A bold opening!' })
    expect(screen.getByText('A bold opening!')).toBeInTheDocument()
  })

  it('offers a voice-output toggle that reports the opposite of the current state', () => {
    const props = setup({ voiceOutput: false })
    const toggle = screen.getByRole('button', { name: /turn voice output on/i })
    fireEvent.click(toggle)
    expect(props.onToggleVoice).toHaveBeenCalledWith(true)
  })

  it('lets the user mute when voice output is on', () => {
    const props = setup({ voiceOutput: true })
    fireEvent.click(screen.getByRole('button', { name: /turn voice output off/i }))
    expect(props.onToggleVoice).toHaveBeenCalledWith(false)
  })

  it('hides the voice toggle until the setting has loaded', () => {
    setup({ voiceOutput: null })
    expect(screen.queryByRole('button', { name: /voice output/i })).not.toBeInTheDocument()
  })

  it('shows a thinking indicator and disables input while the agent works', () => {
    setup({ thinking: true, commentary: 'stale reply' })
    expect(screen.getByLabelText(/command/i)).toBeDisabled()
    expect(screen.getByText(/thinking/i)).toBeInTheDocument()
    // The prior commentary is hidden while a new reply is pending.
    expect(screen.queryByText('stale reply')).not.toBeInTheDocument()
  })

  it('hides the commentary block when showCommentary is false', () => {
    // Mobile shows commentary in the agent bubble instead; the input row
    // must keep working on its own.
    const props = setup({ showCommentary: false, commentary: 'elsewhere', thinking: false })
    expect(screen.queryByText('elsewhere')).not.toBeInTheDocument()
    expect(screen.queryByRole('status')).not.toBeInTheDocument()
    const input = screen.getByLabelText(/command/i)
    fireEvent.change(input, { target: { value: 'play e4' } })
    fireEvent.click(screen.getByRole('button', { name: /send/i }))
    expect(props.onSubmit).toHaveBeenCalledWith('play e4')
  })
})
