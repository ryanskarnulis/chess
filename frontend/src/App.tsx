import { useState } from 'react'
import { AgentBubble } from './AgentBubble'
import { Board } from './Board'
import { BottomBar } from './BottomBar'
import { CapturedPieces } from './CapturedPieces'
import { CommandBox } from './CommandBox'
import { GameControls } from './GameControls'
import { MoveHistory } from './MoveHistory'
import { MoveStrip } from './MoveStrip'
import { OptionsSheet } from './OptionsSheet'
import { PromotionPicker } from './PromotionPicker'
import { ReviewPanel } from './ReviewPanel'
import { useGame } from './useGame'
import { useIsMobile } from './useIsMobile'
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
    tier,
    commentary,
    agentThinking,
    sendCommand,
    voiceOutput,
    setVoiceOutput,
    viewPly,
    reviewing,
    displayFen,
    stepBack,
    stepForward,
    hintShapes,
    requestHint,
  } = useGame()
  const isMobile = useIsMobile()
  const [sheetOpen, setSheetOpen] = useState(false)

  const board = state && (
    <div className="board-wrap">
      <Board
        fen={displayFen ?? state.fen}
        turnColor={state.turn}
        dests={reviewing ? {} : state.dests}
        onMove={play}
        viewOnly={state.game_over || reviewing}
        revision={revision}
        autoShapes={hintShapes}
      />
      {pendingPromotion && (
        <PromotionPicker
          color={state.turn}
          onSelect={completePromotion}
          onCancel={cancelPromotion}
        />
      )}
    </div>
  )

  if (isMobile) {
    return (
      <main className="app mobile-layout">
        <CommandBox
          onSubmit={sendCommand}
          commentary={commentary}
          thinking={agentThinking}
          voiceOutput={voiceOutput}
          onToggleVoice={setVoiceOutput}
          showCommentary={false}
        />
        <AgentBubble commentary={commentary} thinking={agentThinking} />
        {board ?? <p className="status">Connecting…</p>}
        {moveError && (
          <p className="move-error" role="alert">
            {moveError}
          </p>
        )}
        {state && (
          <p className="status">
            {state.game_over
              ? `Game over — ${state.outcome?.result ?? ''}`
              : reviewing
                ? 'Reviewing — press ▶ to return'
                : `${state.turn} to move`}
          </p>
        )}
        {state && (
          <>
            <CapturedPieces captured={state.captured} />
            <MoveStrip
              history={state.history}
              currentPly={viewPly ?? state.history.length}
              onBack={stepBack}
              onForward={stepForward}
              canBack={state.history.length > 0 && viewPly !== 0}
              canForward={reviewing}
            />
            {state.game_over && <ReviewPanel />}
            <BottomBar
              onOptions={() => setSheetOpen(true)}
              onResign={resign}
              onHint={requestHint}
              onUndo={undo}
              resignDisabled={state.game_over}
              hintDisabled={state.game_over || reviewing}
              undoDisabled={state.history.length === 0 || reviewing}
            />
            <OptionsSheet
              open={sheetOpen}
              onClose={() => setSheetOpen(false)}
              onNewGame={newGame}
              tier={tier}
              onSetDifficulty={setDifficulty}
              voiceOutput={voiceOutput}
              onToggleVoice={setVoiceOutput}
            />
          </>
        )}
      </main>
    )
  }

  return (
    <main className="app">
      <h1>Chess</h1>
      <div className="game-layout">
        <div className="board-column">
          {board ?? (
            <div className="board-wrap">
              <p className="status">Connecting…</p>
            </div>
          )}
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
              tier={tier}
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
