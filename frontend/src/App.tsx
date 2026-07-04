import { Board } from './Board'
import { useGame } from './useGame'
import './App.css'

function App() {
  const { state, moveError, revision, play } = useGame()

  return (
    <main className="app">
      <h1>Chess</h1>
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
      </div>
      {moveError && (
        <p className="move-error" role="alert">
          {moveError}
        </p>
      )}
      {state && (
        <p className="status">
          {state.game_over ? `Game over — ${state.outcome?.result ?? ''}` : `${state.turn} to move`}
        </p>
      )}
    </main>
  )
}

export default App
