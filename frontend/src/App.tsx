import { useState } from 'react'
import { AgentBubble } from './AgentBubble'
import { Board } from './Board'
import { BottomBar } from './BottomBar'
import { CapturedPieces } from './CapturedPieces'
import { CommandBox } from './CommandBox'
import { MoveStrip } from './MoveStrip'
import { OptionsSheet } from './OptionsSheet'
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
  const [sheetOpen, setSheetOpen] = useState(false)

  // The player "has moved" once history holds more than the engine's own
  // opening (one ply when the player has black, none otherwise). Until then
  // the side switch is offered and there is nothing of theirs to undo.
  const playerMoved = state
    ? state.history.length > (state.player_color === 'black' ? 1 : 0)
    : false
  const otherColor = state?.player_color === 'white' ? 'black' : 'white'

  const board = state && (
    <div className="board-wrap">
      <Board
        fen={displayFen ?? state.fen}
        turnColor={state.turn}
        orientation={state.player_color}
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

  return (
    <main className="app">
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
      {state && !state.game_over && !playerMoved && (
        <div className="side-picker">
          <span>Playing as {state.player_color}</span>
          <button type="button" onClick={() => newGame(otherColor)}>
            Switch to {otherColor}
          </button>
        </div>
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
          {/* Only after game over; unmounting on a new game resets it. */}
          {state.game_over && <ReviewPanel />}
          <BottomBar
            onOptions={() => setSheetOpen(true)}
            onResign={resign}
            onHint={requestHint}
            onUndo={undo}
            resignDisabled={state.game_over}
            hintDisabled={state.game_over || reviewing}
            undoDisabled={!playerMoved || reviewing}
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

export default App
