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

// What the status row shows before the first state document arrives.
const NO_CAPTURES = { white: [], black: [] }

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
    agentProgress,
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
      {/* The state document arrives mid-turn now — the player's move is on the
          board while the engine is still thinking — and such a frame carries
          the engine's turn and the engine's legal moves. Those are never the
          player's to drag, so the board is offered nothing until the turn is
          theirs again (the same silence a review position gets). */}
      <Board
        fen={displayFen ?? state.fen}
        turnColor={state.turn}
        orientation={state.player_color}
        dests={reviewing || state.turn !== state.player_color ? {} : state.dests}
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

  // Everything the slot under the board can say, one state at a time: a
  // rejected move outranks the rest while it lasts (it clears on the next
  // accepted one), and before the first state document there is only the
  // connection. Whose turn it is is *not* on this list — board orientation
  // already tells the player which side they are, and the agent bubble
  // carries the turn context — so an ordinary game leaves the slot to the
  // player's captures.
  const statusText = moveError
    ? moveError
    : !state
      ? 'Connecting…'
      : state.game_over
        ? `Game over — ${state.outcome?.result ?? ''}`
        : reviewing
          ? 'Reviewing — press ▶ to return'
          : null

  return (
    <main className="app">
      <CommandBox
        onSubmit={sendCommand}
        commentary={commentary}
        thinking={agentThinking}
        progress={agentProgress}
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
      {/* Each side's captures sit with their owner rather than sharing one
          row: Glitch's ride under the agent bubble, the player's under the
          board. Grouped by type with a ×N count, so neither can outgrow the
          column however long the game runs. */}
      <div className="agent-block">
        <AgentBubble
          commentary={commentary}
          thinking={agentThinking}
          progress={agentProgress}
        />
        <CapturedPieces
          captured={state?.captured ?? NO_CAPTURES}
          playerColor={state?.player_color ?? 'white'}
          owner="glitch"
        />
      </div>
      {board}
      {/* The player's captures, and the app's one status slot — the same
          slot, because only one of them ever has something to say. A status
          takes the row rather than pushing it aside, so nothing below the
          board shifts. The key remounts the text when an error takes the
          slot, so the alert role lands on a fresh element and is announced,
          exactly as the old standalone error <p> was. */}
      {statusText !== null ? (
        <section className="captured-row board-status" aria-label="Game status">
          <p
            key={moveError ? 'error' : 'status'}
            className={moveError ? 'board-status-text board-status-error' : 'board-status-text'}
            role={moveError ? 'alert' : undefined}
          >
            {statusText}
          </p>
          {state?.game_over && (
            <button
              type="button"
              className="status-results"
              onClick={() => setResultsDismissed(false)}
            >
              Results
            </button>
          )}
        </section>
      ) : (
        <CapturedPieces
          captured={state?.captured ?? NO_CAPTURES}
          playerColor={state?.player_color ?? 'white'}
          owner="you"
        />
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
