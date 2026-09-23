# The turn coordinator

`backend/src/chessapp/coordinator.py` (2026-07-25, from the agent-control
audit; full design narrative in this file's git history).

**Why:** before this, "a turn" had no name in the code — the player's move and
the engine's reply were one atomic step written out twice, there was no beat
between the two moves for Glitch to react in, and nothing could be rejected as
out-of-order. The audit proposed handing the sequence to the model as tools;
that fails the house rules (the app must play with the LLM off), so the
sequence is deterministic code and the model gets the observation *slot*,
never the wheel.

## The machine

```
awaiting_player → player_move_applied → (agent_observing) →
engine_calculating → engine_move_applied → completed → awaiting_player

abandon_turn: from anywhere back to awaiting_player (turn_id + 1)
settle_engine_turn: awaiting_player → engine_calculating → awaiting_player
                    (same turn_id: no turn was open, and none is consumed)
                    engine raises → player_move_applied (the reply is owed)
```

- `turn_id` counts turns; a healthy move turn is exactly 2 mutations, so a
  third under one id is a duplicate the phases exist to refuse.
- An illegal player move is a *result*, not a transition.
- A game-ending player move completes the turn immediately.
- `collect_engine_reply` returns None and advances when no reply is owed —
  derived from the session at collect time, never remembered. An engine that
  *raises* leaves the phase back at `player_move_applied`: the player's move
  stands and the reply is still owed, which is the state the next command can
  heal from.
- `abandon_turn` is the only other exit: undo, new game, resign, and resume
  each run it on the path where their mutation really happens, dropping any
  pending computation. Only where it happens — a *refused* mutation replaces no
  position, so it leaves the open turn (and the reply that turn is owed) alone.
  `undo` used to abandon before finding out whether it could take anything
  back, and a refused undo beside a move in one batch discarded that move's
  engine reply.
- `settle_engine_turn` answers a *restored* position: a board with the engine
  to move and no turn open over it. Three ways in — a new game the player takes
  as black, a save written between the player's move and the reply, an explicit
  odd-ply takeback that pops the reply alone — and one condition, read off the
  session at call time: an engine, a live game, and the side to move is not the
  player's. `new_game`, `undo` and `resume_game` call it after abandoning, and
  so does `/api/game/undo`, whose client may send its own `plies`. It is not a
  reply, so it consumes no turn and there is no observation beat around it.
  An engine that raises here leaves `player_move_applied`, exactly as
  `collect_engine_reply` does (#329): back at `awaiting_player` the board would
  sit with the engine to move and nothing would ever collect it.
- `apply_player_move` refuses when an engine is attached and the side to move
  is not the player's, whatever the phase says (#329). The legality gate alone
  cannot tell whose piece a legal move belongs to, and a board left with the
  engine to move under an awaiting phase once let the player move the engine's
  pieces.

## Ownership rules

- **The engine's reply belongs to the coordinator and is never a
  model-callable tool** (`test_engine_reply_is_not_a_callable_tool`). A model
  that could ask for the reply could also fail to.
- **A restored engine-to-move board is settled by the coordinator, and the app
  announces the move.** The restoring tools report it under `engine_move` — the
  shape `make_move`'s atomic result already uses — and the command pipeline
  appends the same deterministic `_reply_announcement` an ordinary reply gets
  (the last one, if a command restored twice). Voice-first, a board that moved
  twice in silence is a board the player cannot follow; and like every other
  app-composed line it is shown to the player, never remembered as Glitch's.
- Two ways to run a turn, same boundary: `play_exchange(move)` (atomic — used
  by direct mode and MCP, which have no pipeline behind them) and the beats
  (`apply_player_move` → reaction → `collect_engine_reply` → `complete_turn`),
  which the command pipeline and dragged moves run via the shared
  `api._play_move`. The `atomic_exchange` registry flag names the sequencing
  owner, never the validation.
- `TurnStateError` subclasses `ValueError`: the registry converts it to
  `{"ok": False, ...}` result data for the model, trusted endpoints answer
  409. Every caller converges on `dispatch`, which converges here.

## The observe beat

`begin_observation()` marks where Glitch reacts to the *verified player move*.
The reply is computed in the background from the moment the move lands (a
thread over a board copy; only the collecting thread ever submits a move), so
the reaction costs no wall clock. The beat is optional by construction —
verbosity=low, no brain, or a provider failure skips only the words.

The narrator's mid-turn view deliberately carries **no side to play for** — no
`turn`, no `legal_moves`, no FEN (#188/#193): a narrator that can see whose
move it is announces one. The pipeline appends a deterministic reply
announcement instead of paying for a second narration.

A background answer is discarded (and recomputed synchronously) if the board
moved under it or the computation failed.

**The beat is bounded as well as optional** (#283). "Optional" used to mean only
that the reaction could be *skipped*; a narrator that was merely slow still held
a reply that was already computed, kept the turn in `agent_observing`, and —
the command runs under the mutation lock — parked every other road onto the
board behind it. Every `Brain.narrate` call now runs under
`api._REACTION_BUDGET_S` (10 s, through `api._narrate` → `_within_budget`): when
it expires the ready reply is applied, the turn closes on the deterministic
announcement, the lock is released, and the words that arrive afterwards are
dropped rather than spoken a beat behind the board they were about. The late
call runs on its own thread and touches nothing — the same shape as an abandoned
`_PendingReply`, safe for the same reason (a narrator is handed a board view
snapshotted before the call and answers with words). The number is measured, not
derived from the token cap: 58 observe beats in the deployed trace took 0.7–2.1 s
(median ~1.5 s, one 7.5 s outlier), so the budget clears every healthy reaction
with room for a busy GPU. Underneath it `llama_brain._NARRATE_TIMEOUT` (15 s)
hangs up on the abandoned round trip, so it stops holding a llama-server slot the
next turn needs. The trace's `reaction_late` is where a cut beat shows — the
commentary of one is indistinguishable from verbosity=low, from a dead provider,
and from a beat that never opened.

**The brain route's closer is bounded too** (#316). "Push the king pawn" never
reaches `api._narrate`: the planner plays the move and the loop's own closing
narrator writes the words, inside `get_agent_response`, so until then the reply
Stockfish already computed — and the mutation lock — waited on a phase the
pipeline's budget could not see. The bound lives in the brain
(`LlamaBrain._close`), around the tool-free speech call alone, so the thread
that can act always returns on time with the plan's complete record and only a
thread that can produce nothing but words is ever left behind. Two numbers,
chosen by what the closer is doing: `_CLOSING_BUDGET_S` (10 s) when the reply
is owed and the closer is only reacting — the 31 such closers in the deployed
trace took 0.8–2.0 s, so it only fires on a stuck model — and
`_CLOSING_CEILING_S` (60 s) otherwise. A closer that thinks is putting an
evaluation into words, the answer the player asked for, so it gets the ceiling
even with a reply owed: on the gate's move-plus-analysis scenarios it took
6–10 s and more, and a first cut at 10 s sent both below their floor. A
question's thoughtful answer runs 15–30 s and is never cut either; the ceiling
is only a stall backstop. The socket hangs up 5 s after either
(`_HANG_UP_MARGIN_S`). A cut closer comes back as `narration_late`, the trace
records it as `reaction_late`, and the player hears the fast path's late line
(`_late_close_words`: the moves and the reply — never `STUCK_REPLY` over a move
that landed). What stays unbounded, knowingly: the planner is bounded between
round trips (`planning_deadline_s`, #288) and never during one, because its
calls act and no thread holding them may outlive the turn; and the guard's
rewrite runs after the reply has been collected.

## Board controls

- **Dragged moves**: in agent mode `/api/game/move` runs the same beats as the
  fast path (`api._play_move`, trace route `board`), so drag-played games get
  reactions and memory. In direct mode it answers exactly what it always did —
  bar the one turn an engine dies on, where it gains the app's line as
  `commentary` (see the dead-engine edge below) and its next drag settles the
  reply that turn was left owing.
- **New game / resign / claim-draw buttons** dispatch through the registry, so
  the same gate (`tools._gate`) that answers a spoken "new game" arms and asks
  here: 409 + `{"detail", "confirm": true, "op"}`, answered at
  `/api/game/confirm` from either surface. The draw button reads "Claim draw"
  while the state document's `claimable_draws` is non-empty — the rules are
  board truth, and the client is told rather than left to work them out — and
  `/api/game/claim-draw` runs the same `claim_draw` tool the brain is offered
  then; nothing to claim is a plain 409 with nothing armed (the tool's own
  check, ahead of the gate). Without a claim the button is the draw *offer*:
  `/api/game/offer-draw` dispatches the same `offer_draw` tool a spoken "call
  it a draw?" reaches, and the answer is code's — Stockfish's number and the
  material, never the model's (`docs/draw-offer.md`). Not gated, because a
  decline changes nothing and an acceptance ends a position the rule has
  already judged drawn; the UI shows a deterministic line from the result's
  `accepted` and `reason`. Undo is not destructive and stays direct.
- **MCP calls** (the standalone `chessapp.mcp_server`, a game of its own)
  dispatch through the same registry and gate, and the gate's question is
  answered on a third surface: the call wrapper puts
  `tools.CONFIRM_QUESTIONS[op]` to the client's *human* by MCP form-mode
  elicitation and calls `confirm_pending` on an accepted, ticked form. The
  client's model neither emits the request nor answers it, and the advertised
  schema does not change. A no, a dismissed form, or a client that declared
  no elicitation capability leaves nothing armed and says so (`declined`,
  `confirmation_unavailable`); a board that moved, or a later gated call that
  asked its own question, while the form was open runs nothing (`stale`).
  The lock is released across the human's wait. Design and acceptance
  criteria: `docs/mcp-confirmation-surface.md`.

## Limits and preconditions (all code-owned)

- **One player move + one engine reply per turn** — structural; no transition
  admits a second.
- **What the gate guards** (`tools.GATED_TOOLS`, #291): the three ops that end
  or reset a game (`DESTRUCTIVE_TOOLS`), plus `resume_game` — which throws the
  game on the board away as surely as a reset — and `save_game` over an
  existing name, which throws an earlier save away. The first four ask only
  when a game is at stake (the player has moved and it is not over); an
  overwrite asks whenever the name exists, except `autosave` and a name the
  same command already wrote. Save and resume have no button, so they are
  answered on the spoken, delegate and MCP roads. Semantics and rationale:
  `docs/persistence-and-identity.md`.
- **One destructive op per command** — `begin_command`/`end_command` bracket
  an interaction; the budget is command-scoped because destructive ops
  `abandon_turn` themselves. `resume_game` spends it too; an overwriting
  `save_game` does not, since it leaves the board alone. `offer_draw` holds the budget too without being
  gated: it checks it before evaluating and spends it only on acceptance
  (which abandons the turn — the owed reply is dropped with the game), so a
  declined offer costs nothing and leaves an open turn, and its owed reply,
  exactly where they were. Only `/api/command` opens a window (the brain
  loop is the only surface that can chain dispatches); buttons/MCP dispatch
  once by construction. The window also owns the command's **board trail** —
  the position each mutating dispatch left behind — because chaining is
  exactly what puts boards between the command's two ends, and the honesty
  guard checks its commentary against every one of them (`api._verified_facts`,
  audit finding 7).
- **One question per command** (decided 2026-09-05): the first gated call in
  a command arms its op and its question; every later gated call in the same
  window — the same op again or a different one — is refused with a result
  naming the pending question (`pending: <op>`, `retry: never`) and arms
  nothing. So the question the narrator relays, the question the reader is
  handed (`api._confirm_question`) and the op `confirm_pending` runs are one op
  by construction; two refusals used to arm two ops with the last replacing
  the first. Across interactions the newest question is the one a yes answers
  (a new command disarms on its way in; a button press is its own
  interaction), which is the same rule seen from outside the window.
- **Board versions**: `GameSession.revision` bumps inside every mutating
  session method → `ToolContext.board_version` → `state.version`. Mutating
  requests may carry `version`; stale → 409 `{"stale": true, ...}`. The check
  is welded to the mutation (`_mutation(expected)` holds `ctx.mutation_lock`,
  acquired off the event loop). MCP serializes on the same lock instead of a
  schema param (the tool schema is frozen by the eval floor). The brain never
  sees a version.
- **An armed destructive op is a question about a position, asked in a
  conversation**: `PendingOp` carries both, and all three answering surfaces
  read it through `ctx.live_pending(origin)`, which returns it only if both
  still hold.
  - *The position*: the `board_version` it was armed against (re-stamped where
    the command closes, since the gate arms mid-turn). A stale one is dropped
    where the answer is read — so a "yes" can never answer a question about a
    board that has since moved.
  - *The conversation* (#281): the `origin` the gate stamped off
    `ToolContext.origin`, which each surface declares on its way in, under the
    mutation lock. Three of them. **`panel`** is the player's own screen —
    `/api/command`, the three board buttons and `/api/game/confirm` are one
    origin on purpose, so a question asked by a button is still answered by a
    typed "yes" and the reverse. **`delegate:<conversation id>`** is one thread
    on the delegate API, never the wire as a whole. **`mcp`** is the standalone
    server's own context, which asks and answers inside a single call
    (elicitation), so the rule costs that surface nothing.
  - *An answer from another origin is not an answer at all*: it is a new
    command, and every command from any origin disarms what is pending on its
    way in, exactly as it always has. So the words are never shown to the
    free-text reader and travel on as an ordinary utterance — and the origin
    that *was* asked no longer has a question to answer either, because the
    intervening command dropped it. The alternative, leaving a foreign op armed
    across other origins' commands, was rejected: the gate would then refuse a
    conversation that never asked ("a confirmation is already pending"), and
    `restamp_pending` at the end of the other origin's command would re-point
    the question at a board it was never about. A *button* click is not a
    command, so it neither answers nor disarms a delegate's question — it is
    the same 409 a click with nothing armed gets.
  - The interaction's origin is on the turn record (`trace.turn_record`), which
    is the only thing that tells two identical-looking "yes" turns apart.

## Live progress

Phases broadcast as they happen on the state websocket (`progress.py`),
`type: "progress"`, each event emitted by the chokepoint that owns the fact
(the coordinator's phase setter, `ToolRegistry.dispatch`, the brain's
`planning`/`narrating`) and stamped with the interaction's `correlation_id`
(the trace record's id). Mid-turn board changes publish the state document on
the same beat, deduped by version. The pipeline's blocking steps run off the
event loop (`api._offloop`) so events arrive live, not in a burst. Reporting
is wrapped and swallowed — a lost label is never a lost turn. Direct mode and
the control buttons report nothing (no interaction window).


Latency across a whole voice interaction (speech end, transcript, board,
reply audio, playback end) is measured by joining the turn record to the
browser's own milestones by `interaction_id`: `docs/latency-measurement.md`.
## Known edges, deliberately left

- A route that raises after the player's move landed leaves the turn open, and
  so does an engine that raises mid-calculation — the phase comes back to
  `player_move_applied`, so the next command heals it: refused move, owed reply
  played, at the cost of one utterance. Until 2026-09-05 the engine case did
  not: the phase stayed at `engine_calculating`, where `_require` refuses every
  ordinary player move, and only an undo, a reset or a resume could dig the game
  out. This paragraph claimed otherwise; the code now matches it.
- **A dead engine is a structured outcome, not a failed request** (#284, 2026-09-16).
  The player's move is committed and broadcast before Stockfish is asked
  anything, so every surface that collects a reply catches the failure instead
  of raising through it (`api._play_move`'s close beat, the command
  convergence, and direct mode's atomic exchange): 200 with the committed
  results and the current board, the turn deliberately left open with the reply
  owed, and the app's own line saying so composed where the reply announcement
  would have been (`ENGINE_LOST_REPLY_OWED` — the app's words, never Glitch's,
  so the turn is remembered by the facts). The loop's `stop_reason` is
  untouched (it is the delegate wire's word for how the *run* ended); what the
  turn carries instead is `engine_failure`, class and message, on the outcome
  and in the trace. The trace itself is written from a `finally`-owned envelope
  now, so any exception escaping a turn still leaves exactly one record, with
  an `error` field naming it.
- **Stockfish is relaunched once, and a death inside a tool is a result**
  (#329, 2026-09-23). `EnginePlayer` routes every call through `_run`: on
  `EngineTerminatedError` it starts the process again, re-applies the strength
  options it had set, and repeats the call once — the player never hears about
  a crash the app recovered from, and nothing above it retries. A second death
  (or a process that will not start) reaches the callers: `ToolRegistry.dispatch`
  answers it as `ok: false, retry: never`, saying whether the call changed the
  board first, so Glitch voices it from the result; a settle leaves the reply
  owed, and `undo`/`new_game`/`resume_game` report `engine_reply_owed` beside
  the work that did happen; the hint, review and difficulty endpoints answer 503.
  Found by the #318 trajectory walks, whose engine dies for one step anywhere.
- `/api/game/confirm` returns only state (the dialog already asked) but is
  traced (route `control`). Undo and direct-mode drags stay untraced — neither
  can be an agent failure.
- Windowless surfaces (MCP, buttons) are unbudgeted across calls by design;
  what they cannot do is chain destructive ops inside one command.
- Standalone MCP answers the confirmation gate through the client's human
  (audit finding 3; `docs/mcp-confirmation-surface.md`, #272): the server
  elicits, the client's user accepts or declines, and only that yes turns
  `confirm_pending`. A client whose handshake declared no form elicitation is
  refused truthfully (`confirmation_unavailable`) with nothing armed. Run end
  to end against Claude Code 2.1.261 on 2026-09-05 (DONE.md).
