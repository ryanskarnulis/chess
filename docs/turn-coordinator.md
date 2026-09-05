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

## Board controls

- **Dragged moves**: in agent mode `/api/game/move` runs the same beats as the
  fast path (`api._play_move`, trace route `board`), so drag-played games get
  reactions and memory. In direct mode it answers exactly what it always did.
- **New game / resign buttons** dispatch through the registry, so the same
  gate (`tools._gate`) that answers a spoken "new game" arms and asks here:
  409 + `{"detail", "confirm": true, "op"}`, answered at `/api/game/confirm`
  from either surface. Undo is not destructive and stays direct.

## Limits and preconditions (all code-owned)

- **One player move + one engine reply per turn** — structural; no transition
  admits a second.
- **One destructive op per command** — `begin_command`/`end_command` bracket
  an interaction; the budget is command-scoped because destructive ops
  `abandon_turn` themselves. Only `/api/command` opens a window (the brain
  loop is the only surface that can chain dispatches); buttons/MCP dispatch
  once by construction. The window also owns the command's **board trail** —
  the position each mutating dispatch left behind — because chaining is
  exactly what puts boards between the command's two ends, and the honesty
  guard checks its commentary against every one of them (`api._verified_facts`,
  audit finding 7).
- **Board versions**: `GameSession.revision` bumps inside every mutating
  session method → `ToolContext.board_version` → `state.version`. Mutating
  requests may carry `version`; stale → 409 `{"stale": true, ...}`. The check
  is welded to the mutation (`_mutation(expected)` holds `ctx.mutation_lock`,
  acquired off the event loop). MCP serializes on the same lock instead of a
  schema param (the tool schema is frozen by the eval floor). The brain never
  sees a version.
- **An armed destructive op is a question about a position**: `PendingOp`
  carries the `board_version` it was armed against (re-stamped where the
  command closes, since the gate arms mid-turn), and both answering surfaces
  read it through `ctx.live_pending()`, which drops a stale one — so a "yes"
  can never answer a question about a board that has since moved.

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

## Known edges, deliberately left

- A route that raises after the player's move landed leaves the turn open, and
  so does an engine that raises mid-calculation — the failure is loud (the
  command ends in the error) but the phase comes back to `player_move_applied`,
  so the next command heals it: refused move, owed reply played, at the cost of
  one utterance. Until 2026-09-05 the engine case did not: the phase stayed at
  `engine_calculating`, where `_require` refuses every ordinary player move, and
  only an undo, a reset or a resume could dig the game out. This paragraph
  claimed otherwise; the code now matches it.
- `/api/game/confirm` returns only state (the dialog already asked) but is
  traced (route `control`). Undo and direct-mode drags stay untraced — neither
  can be an agent failure.
- Windowless surfaces (MCP, buttons) are unbudgeted across calls by design;
  what they cannot do is chain destructive ops inside one command.
