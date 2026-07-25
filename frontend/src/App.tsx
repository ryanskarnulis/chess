import { useEffect, useRef, useState } from 'react'
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

// A handed-off intent is one utterance, not an essay — anything longer is
// junk (or someone poking at the URL), so it gets cut rather than sent.
const MAX_INTENT_LENGTH = 500

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
    agentAvailable,
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
  // The conductor hands off to chess by navigating here with what the user
  // actually said (`/?intent=let's play chess as black`). Run it through the
  // agent so the session opens already acting on it. Scrub the param first —
  // a reload must not replay the command — and latch so StrictMode's second
  // mount doesn't send it twice.
  const intentSent = useRef(false)
  useEffect(() => {
    if (intentSent.current) return
    const intent = new URLSearchParams(window.location.search).get('intent')?.trim() ?? ''
    if (window.location.search) window.history.replaceState({}, '', window.location.pathname)
    if (!intent) return
    intentSent.current = true
    void sendCommand(intent.slice(0, MAX_INTENT_LENGTH))
  }, [sendCommand])
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
        disabled={agentAvailable === false}
      />
      {/* Direct mode, said out loud (audit item 1): with no brain configured
          the game is fully playable against Stockfish and the agent simply
          isn't there — a deliberate mode, not a silent bypass the player
          discovers when their words go nowhere. */}
      {agentAvailable === false && (
        <p className="direct-mode" role="status">
          Direct mode — Stockfish only, no agent
        </p>
      )}
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
