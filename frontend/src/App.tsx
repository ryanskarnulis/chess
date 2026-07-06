import { Board } from './Board'
import { CapturedPieces } from './CapturedPieces'
import { CommandBox } from './CommandBox'
import { GameControls } from './GameControls'
import { MoveHistory } from './MoveHistory'
import { PromotionPicker } from './PromotionPicker'
import { ReviewPanel } from './ReviewPanel'
import { useGame } from './useGame'
import './App.css'

function App() {
  const {
    state,
    moveError,
    revision,
    play,
    pendingPromotion,
    completePromotion,
    cancelPromotion,
    newGame,
    undo,
    resign,
    setDifficulty,
    commentary,
    agentThinking,
    sendCommand,
    voiceOutput,
    setVoiceOutput,
  } = useGame()

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
          <CommandBox
            onSubmit={sendCommand}
            commentary={commentary}
            thinking={agentThinking}
            voiceOutput={voiceOutput}
            onToggleVoice={setVoiceOutput}
          />
        </div>
        {state && (
          <aside className="panels">
            <GameControls
              canUndo={state.history.length > 0}
              gameOver={state.game_over}
              onNewGame={newGame}
              onUndo={undo}
              onResign={resign}
              onSetDifficulty={setDifficulty}
            />
            <CapturedPieces captured={state.captured} />
            <MoveHistory history={state.history} />
            {/* Only after game over; unmounting on a new game resets it. */}
            {state.game_over && <ReviewPanel />}
          </aside>
        )}
      </div>
    </main>
  )
}

export default App
