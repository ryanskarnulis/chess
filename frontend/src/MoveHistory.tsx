import { pairMoves } from './moves'

export interface MoveHistoryProps {
  /** Moves played so far, in SAN and ply order (from the backend). */
  history: string[]
}

/**
 * Scrolling table of the moves played, numbered and paired white/black.
 * Presentational only — the backend owns the move list.
 */
export function MoveHistory({ history }: MoveHistoryProps) {
  const pairs = pairMoves(history)

  return (
    <section className="move-history" aria-label="Move history">
      <h2>Moves</h2>
      {pairs.length === 0 ? (
        <p className="panel-empty">No moves yet</p>
      ) : (
        <table>
          <tbody>
            {pairs.map((pair) => (
              <tr key={pair.number}>
                <th scope="row">{pair.number}</th>
                <td>{pair.white}</td>
                <td>{pair.black ?? ''}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </section>
  )
}
