import { Board } from './Board'
import { CapturedPieces } from './CapturedPieces'
import { MoveHistory } from './MoveHistory'
import { PromotionPicker } from './PromotionPicker'
import { useGame } from './useGame'
import './App.css'

function App() {
  const { state, moveError, revision, play, pendingPromotion, completePromotion, cancelPromotion } =
    useGame()

  return (
    <main className="app">
      <h1>Chess</h1>
      <div className="game-layout">
        <div className="board-column">
          <div className="board-wrap">
            {state ? (
              <Board
                fen={state.fen}
                turnColor={state.turn}
                dests={state.dests}
                onMove={play}
                viewOnly={state.game_over}
                revision={revision}
              />
            ) : (
              <p className="status">Connecting…</p>
            )}
            {pendingPromotion && state && (
              <PromotionPicker
                color={state.turn}
                onSelect={completePromotion}
                onCancel={cancelPromotion}
              />
            )}
          </div>
          {moveError && (
            <p className="move-error" role="alert">
              {moveError}
            </p>
          )}
          {state && (
            <p className="status">
              {state.game_over
                ? `Game over — ${state.outcome?.result ?? ''}`
                : `${state.turn} to move`}
            </p>
          )}
        </div>
        {state && (
          <aside className="panels">
            <CapturedPieces captured={state.captured} />
            <MoveHistory history={state.history} />
          </aside>
        )}
      </div>
    </main>
  )
}

export default App
