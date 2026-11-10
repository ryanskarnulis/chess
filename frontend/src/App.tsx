import { useCallback, useEffect, useRef, useState } from 'react'
import { AgentBubble } from './AgentBubble'
import { Board } from './Board'
import { BottomBar } from './BottomBar'
import { CapturedPieces } from './CapturedPieces'
import { CommandBox } from './CommandBox'
import { ConfirmSheet } from './ConfirmSheet'
import { GameMenu } from './GameMenu'
import { MoveStrip } from './MoveStrip'
import { PostGameModal } from './PostGameModal'
import { PromotionPicker } from './PromotionPicker'
import { boardHighlights } from './highlights'
import { idleLine } from './idle'
import { useGame } from './useGame'
import './App.css'

// A handed-off intent is one utterance, not an essay — anything longer is
// junk (or someone poking at the URL), so it gets cut rather than sent.
const MAX_INTENT_LENGTH = 500

// The request is answered (or given up on), so the URL stops carrying it.
function dropIntentParam() {
  const url = new URL(window.location.href)
  url.searchParams.delete('intent')
  window.history.replaceState({}, '', url.pathname + url.search + url.hash)
}

// What the status row shows before the first state document arrives.
const NO_CAPTURES = { white: [], black: [] }

// The endings `GameSession.undo` refuses to take back, by the termination the
// state's outcome names. A claimed draw is the only way a game ends on the
// threefold or fifty-move rule; the board's own draws are fivefold and
// seventy-five.
const FINAL_ENDINGS = new Set(['resignation', 'agreement', 'threefold_repetition', 'fifty_moves'])

// The side switch names a colour, and the name has to be one string: what is
// painted and what a screen reader announces. `text-transform: capitalize` in
// the stylesheet gave two — "Switch To White" on screen against an accessible
// name of "Switch to white" — and title-cased the "To" while it was there. So
// the colour is capitalized in the text itself.
const capitalize = (word: string) => word.charAt(0).toUpperCase() + word.slice(1)

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
    claimDraw,
    offerDraw,
    question,
    answerQuestion,
    setDifficulty,
    tier,
    commentary,
    agentAvailable,
    glitchAvailable,
    agentThinking,
    agentProgress,
    sendCommand,
    sendHandoff,
    pgn,
    voiceOutput,
    setVoiceOutput,
    brain,
    setBrain,
    viewPly,
    reviewing,
    displayFen,
    stepBack,
    stepForward,
    hintShapes,
    requestHint,
  } = useGame()
  const [menuOpen, setMenuOpen] = useState(false)
  // The Game menu's Resign… asks here first, in the app's own sheet, before
  // anything is sent — resigning is the one op that always asks, even where
  // the backend's gate would stand aside (before the player has moved).
  const [resignAsk, setResignAsk] = useState(false)
  // The conductor hands off to chess by navigating here with what the user
  // actually said (`/?intent=let's play chess as black`). Run it through the
  // agent so the session opens already acting on it (#546).
  //
  // The words stay in the URL until the agent has run them: if the model is
  // down or slow the request must survive, and a reload is then the retry. On
  // failure the page says so and offers one. Three latches keep it from being
  // sent twice — `intentStarted` (StrictMode's second mount, and any re-render),
  // `intentBusy` (a retry tapped while one is in flight) and `intentOwed`,
  // cleared the moment it succeeds (nothing left to send, nothing to re-send).
  const intentStarted = useRef(false)
  const intentBusy = useRef(false)
  const intentOwed = useRef<string | null>(null)
  const [intentFailed, setIntentFailed] = useState(false)
  const runIntent = useCallback(async () => {
    const text = intentOwed.current
    if (text === null || intentBusy.current) return
    intentBusy.current = true
    setIntentFailed(false)
    let ran = false
    try {
      ran = await sendHandoff(text)
    } finally {
      intentBusy.current = false
    }
    if (ran) {
      intentOwed.current = null
      dropIntentParam()
    } else {
      setIntentFailed(true)
    }
  }, [sendHandoff])
  const dismissIntent = () => {
    intentOwed.current = null
    dropIntentParam()
    setIntentFailed(false)
  }
  useEffect(() => {
    if (intentStarted.current) return
    intentStarted.current = true
    const intent = new URLSearchParams(window.location.search).get('intent')?.trim() ?? ''
    if (!intent) {
      // Nothing to send: scrub whatever else rode in on the query.
      if (window.location.search) window.history.replaceState({}, '', window.location.pathname)
      return
    }
    intentOwed.current = intent.slice(0, MAX_INTENT_LENGTH)
    void runIntent()
  }, [runIntent])
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
  // A claim exists only when the backend says so — the rules are its truth,
  // never read off the board here. Without one the button is the draw *offer*,
  // whose answer is the backend's rule too (docs/draw-offer.md): the UI only
  // relays it.
  const drawClaimable = (state?.claimable_draws?.length ?? 0) > 0
  // A finished game can still be taken back when the board ended it (a mate,
  // a stalemate): the server reopens it. One the players ended by declaration
  // — a resignation, an agreed or a claimed draw — it refuses, so the button
  // is not offered there (#459).
  const undoable =
    playerMoved && !(state?.game_over && FINAL_ENDINGS.has(state.outcome?.termination ?? ''))

  // The move into the shown position and any check in it — the live one, or
  // the one being reviewed — so Glitch's reply is highlighted like the
  // player's, and a reload or an undo still shows what was last played.
  const highlights = state
    ? boardHighlights(state.fens, state.history, viewPly ?? state.fens.length - 1)
    : null

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
        lastMove={highlights?.lastMove}
        check={highlights?.check}
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
  // player's captures. The game-over line is the one status that sits beside
  // the captures instead of taking their row: it lasts for the rest of the
  // game, and the captures are part of how it ended (#459).
  const statusText = moveError
    ? moveError
    : !state
      ? 'Connecting…'
      : state.game_over
        ? `Game over — ${state.outcome?.result ?? ''}`
        : viewPly !== null
          ? viewPly === 0
            ? 'Viewing the starting position — tap › to return'
            : `Viewing move ${Math.ceil(viewPly / 2)} — tap › to return`
          : null
  const playerCaptures = (
    <CapturedPieces
      captured={state?.captured ?? NO_CAPTURES}
      playerColor={state?.player_color ?? 'white'}
      owner="you"
    />
  )
  const statusRow =
    statusText === null ? null : (
      <section className="captured-row board-status" aria-label="Game status">
        <p
          key={moveError ? 'error' : 'status'}
          className={moveError ? 'board-status-text board-status-error' : 'board-status-text'}
          role={moveError ? 'alert' : undefined}
        >
          {statusText}
        </p>
        {gameOver && (
          <button type="button" className="status-results" onClick={() => setResultsDismissed(false)}>
            Results
          </button>
        )}
      </section>
    )

  return (
    <main className="app">
      <CommandBox
        onSubmit={sendCommand}
        commentary={commentary}
        thinking={agentThinking}
        progress={agentProgress}
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
      {/* A handed-off request that did not land (#546): said out loud, with the
          one way forward. It stays in the URL meanwhile, so a reload tries it
          again too. */}
      {intentFailed && (
        <div className="handoff-failed" role="alert">
          <p>Your request to Glitch didn’t go through.</p>
          <button type="button" className="review-retry" onClick={() => void runIntent()}>
            Retry
          </button>
          <button type="button" className="review-retry" onClick={dismissIntent}>
            Dismiss
          </button>
        </div>
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
          pgn={pgn}
          idle={idleLine(state)}
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
          exactly as the old standalone error <p> was. A finished game is the
          exception: its line shares the row with the captures (#459). */}
      {gameOver ? (
        <div className="board-status-row">
          {playerCaptures}
          {statusRow}
        </div>
      ) : (
        (statusRow ?? playerCaptures)
      )}
      {state && !state.game_over && !playerMoved && (
        <div className="side-picker">
          <span>Playing as {capitalize(state.player_color)} ·</span>
          <button type="button" onClick={() => newGame(otherColor)}>
            Switch to {capitalize(otherColor)}
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
            onUndo={undo}
            onHint={requestHint}
            onGame={() => setMenuOpen((o) => !o)}
            undoDisabled={!undoable || reviewing}
            hintDisabled={state.game_over || reviewing}
            gameOpen={menuOpen}
          />
          <GameMenu
            open={menuOpen}
            onClose={() => setMenuOpen(false)}
            onNewGame={newGame}
            drawClaimable={drawClaimable}
            gameOver={state.game_over}
            onDraw={drawClaimable ? claimDraw : offerDraw}
            onResign={() => setResignAsk(true)}
            tier={tier}
            glitchAvailable={glitchAvailable}
            onSetDifficulty={setDifficulty}
            voiceOutput={voiceOutput}
            onToggleVoice={setVoiceOutput}
            brain={brain}
            onSetBrain={setBrain}
          />
        </>
      )}
      {/* The app's one yes/no question: the resign it asks itself, or the one
          a gated op (new game, claim draw, …) came back with. A resign
          confirmed here answers the gate's identical question too. */}
      {resignAsk ? (
        <ConfirmSheet
          question={{ op: 'resign' }}
          onAnswer={(yes) => {
            setResignAsk(false)
            if (yes) void resign({ confirmed: true })
          }}
        />
      ) : (
        question && <ConfirmSheet question={question} onAnswer={answerQuestion} />
      )}
    </main>
  )
}

export default App
