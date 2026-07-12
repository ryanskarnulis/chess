import { useEffect, useState } from 'react'
import { AgentBubble } from './AgentBubble'
import { Board } from './Board'
import { BottomBar } from './BottomBar'
import { CapturedPieces } from './CapturedPieces'
import { CommandBox } from './CommandBox'
import { MoveStrip } from './MoveStrip'
import { OptionsSheet } from './OptionsSheet'
import { PostGameModal } from './PostGameModal'
import { PromotionPicker } from './PromotionPicker'
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
  // The post-game screen shows whenever the game is over and the player
  // hasn't waved it away; a new game (game_over back to false) re-arms it.
  const [resultsDismissed, setResultsDismissed] = useState(false)
  const gameOver = state?.game_over ?? false
  useEffect(() => {
    if (!gameOver) setResultsDismissed(false)
  }, [gameOver])

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
          {state.game_over && (
            <button
              type="button"
              className="status-results"
              onClick={() => setResultsDismissed(false)}
            >
              Results
            </button>
          )}
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
          {/* Mounted for the whole post-game (hidden while dismissed) so a
              fetched review survives reopening; a new game unmounts it and
              resets the review. */}
          {state.game_over && state.outcome && (
            <PostGameModal
              outcome={state.outcome}
              playerColor={state.player_color}
              open={!resultsDismissed}
              onNewGame={newGame}
              onClose={() => setResultsDismissed(true)}
            />
          )}
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
