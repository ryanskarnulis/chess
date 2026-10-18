import type { ConfirmQuestion } from './api'

export interface ConfirmSheetProps {
  /** The question to put to the player: the op it confirms, and the gate's
   * own wording for an op the sheet has no copy for. */
  question: Pick<ConfirmQuestion, 'op'> & { detail?: string }
  /** Called with the player's answer: true runs the op, false keeps playing. */
  onAnswer: (yes: boolean) => void
}

interface Copy {
  heading: string
  body: string
  yes: string
  no: string
  danger?: boolean
}

// The sheet's own words for each gated op. The gate decides *whether* to ask;
// how a phone dialog phrases it is the UI's, so the copy lives here. An op
// this map doesn't know still asks — in the gate's own words.
const COPY: Record<string, Copy> = {
  resign: {
    heading: 'Resign this game?',
    body: "Glitch takes the win. You can't take a resignation back.",
    yes: 'Resign',
    no: 'Keep playing',
    danger: true,
  },
  new_game: {
    heading: 'Start a new game?',
    body: 'That ends the game in progress.',
    yes: 'New game',
    no: 'Keep playing',
  },
  claim_draw: {
    heading: 'Claim the draw?',
    body: 'That ends the game in a draw.',
    yes: 'Claim draw',
    no: 'Keep playing',
  },
}

/**
 * Bottom sheet asking the player to confirm a destructive op — the app's one
 * yes/no dialog, in place of the browser's `window.confirm`. Tapping outside
 * is a no: nothing ends by accident.
 */
export function ConfirmSheet({ question, onAnswer }: ConfirmSheetProps) {
  const copy: Copy = COPY[question.op] ?? {
    heading: question.detail ?? `Confirm ${question.op.replaceAll('_', ' ')}?`,
    body: '',
    yes: 'Yes',
    no: 'Cancel',
  }
  return (
    <>
      <div className="sheet-backdrop confirm-backdrop" onClick={() => onAnswer(false)} />
      <section className="sheet confirm-sheet" role="alertdialog" aria-label={copy.heading}>
        <span className="sheet-handle" aria-hidden="true" />
        <h2 className="confirm-heading">{copy.heading}</h2>
        {copy.body && <p className="confirm-body">{copy.body}</p>}
        <button
          type="button"
          className={copy.danger ? 'confirm-yes confirm-danger' : 'confirm-yes'}
          onClick={() => onAnswer(true)}
        >
          {copy.yes}
        </button>
        <button type="button" className="confirm-no" onClick={() => onAnswer(false)}>
          {copy.no}
        </button>
      </section>
    </>
  )
}
