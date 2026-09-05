import { describe, expect, it } from 'vitest'
import type { ProgressEvent } from './api'
import { NO_PROGRESS, applyProgress, progressLabel, type TurnProgress } from './progress'

function event(
  kind: ProgressEvent['kind'],
  name = '',
  correlation_id = 'abc123',
): ProgressEvent {
  return { correlation_id, turn_id: 1, kind, name }
}

/** Fold a whole turn's worth of events, as the hook does. */
function fold(...events: ProgressEvent[]): TurnProgress {
  return events.reduce(applyProgress, NO_PROGRESS)
}

describe('progressLabel', () => {
  it('names the waits, not the boundaries', () => {
    expect(progressLabel(event('phase', 'engine_calculating'))).toBe(
      'Stockfish is calculating',
    )
    expect(progressLabel(event('phase', 'agent_observing'))).toBe('Glitch is reacting')
    // Real states, but nothing is happening in them — a line for each would
    // read as a flicker, not as information.
    expect(progressLabel(event('phase', 'player_move_applied'))).toBeNull()
    expect(progressLabel(event('phase', 'completed'))).toBeNull()
  })

  it('tells the brain’s two phases apart', () => {
    expect(progressLabel(event('brain', 'planning'))).toBe('Glitch is thinking')
    expect(progressLabel(event('brain', 'narrating'))).toBe('Glitch is reacting')
  })

  it('puts tools in the player’s language', () => {
    expect(progressLabel(event('tool', 'make_move'))).toBe('Validating your move')
    expect(progressLabel(event('tool', 'evaluate_position'))).toBe(
      'Evaluating the position',
    )
    expect(progressLabel(event('tool', 'describe_position'))).toBe(
      'Describing the position',
    )
  })

  it('still says something readable for a tool it has never heard of', () => {
    // The registry may grow; this map must not go silent when it does.
    expect(progressLabel(event('tool', 'control_physical_board'))).toBe(
      'Running control physical board',
    )
  })

  it('says nothing for the brackets', () => {
    expect(progressLabel(event('begin'))).toBeNull()
    expect(progressLabel(event('end'))).toBeNull()
  })
})

describe('applyProgress', () => {
  it('walks a move turn from begin to end', () => {
    expect(fold(event('begin'))).toEqual({ correlationId: 'abc123', label: null })
    expect(fold(event('begin'), event('tool', 'make_move')).label).toBe(
      'Validating your move',
    )
    expect(
      fold(
        event('begin'),
        event('tool', 'make_move'),
        event('phase', 'engine_calculating'),
      ).label,
    ).toBe('Stockfish is calculating')
    expect(
      fold(
        event('begin'),
        event('tool', 'make_move'),
        event('phase', 'engine_calculating'),
        event('end'),
      ),
    ).toEqual(NO_PROGRESS)
  })

  it('keeps the last line through an event with nothing to say', () => {
    const after = fold(
      event('begin'),
      event('tool', 'make_move'),
      event('phase', 'player_move_applied'),
    )
    expect(after.label).toBe('Validating your move')
  })

  it('adopts a turn it never saw begin', () => {
    // A client that connects mid-turn should still show what it can.
    expect(fold(event('phase', 'engine_calculating', 'later'))).toEqual({
      correlationId: 'later',
      label: 'Stockfish is calculating',
    })
  })

  it('switches cleanly to the next turn', () => {
    const after = fold(
      event('begin'),
      event('tool', 'make_move'),
      event('begin', '', 'next'),
    )
    expect(after).toEqual({ correlationId: 'next', label: null })
  })

  it('ignores a late end from a turn that is no longer on screen', () => {
    const showing = fold(event('begin', '', 'next'), event('tool', 'make_move', 'next'))
    expect(applyProgress(showing, event('end', '', 'stale'))).toEqual(showing)
  })
})
