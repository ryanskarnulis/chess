# The turn coordinator — design note (2026-07-25)

`backend/src/chessapp/coordinator.py`. Sprint 1, slice 1 of the agent-control
replan (`docs/agent-control-audit-2026-07-25.md`, items 3 and 10), extended by
slice 3 — the move flow split (items 2 and 5), which filled the observation slot
and is folded into this note where it changed something.

## Why

Before this, "a turn" was not a thing the code had a name for. The player's
move and the engine's reply were one atomic step, written out twice — once in
the `make_move` tool handler, once in `/api/game/move` — and reachable only by
*making a move*. Three consequences the audit is right about:

- **There was no point between the two moves.** Glitch's commentary could only
  ever react to a completed exchange, because by the time any tool result came
  back, Stockfish had already answered. There was nowhere to stand.
- **Two copies of one rule.** "Reply if there's an engine and the game is still
  live" existed in the tool layer and in the API layer, and the twin rule for
  the engine's opening move existed in `new_game` twice more. Four places, one
  policy — the shape that drifts.
- **Nothing could be rejected.** Without phases there is no such thing as an
  action that doesn't belong right now, so a duplicated move was
  indistinguishable from an intended one.

The report's fix was to hand the sequence to the model as tools
(`request_engine_reply`, `apply_engine_move`). That fails the house rules: it
makes a 12B responsible for whether the engine replies — something the session
already knows — and a stalled or absent model would stall a game the app must
be able to play with the LLM off. So the sequence became deterministic code
instead, and the model got the *slots* rather than the wheel.

## The machine

```
                    ┌──────────────────────────────────────────────┐
                    │                                              │
                    ▼                                              │
            awaiting_player                                        │
                    │                                              │
   apply_player_move│ (illegal move → stays here, as a result)     │
                    │ …and starts the engine thinking              │
                    ▼                                              │
          player_move_applied ──────────────────────┐              │
                    │                               │              │
  begin_observation │                               │              │
                    ▼                               │              │
            agent_observing ────────────────────────┤              │
                                                    │              │
                                 collect_engine_reply              │
                                                    ▼              │
                                          engine_calculating       │
                                                    │              │
                                                    ▼              │
                                          engine_move_applied      │
                                                    │              │
                                       complete_turn│ turn_id + 1  │
                                                    ▼              │
                                                completed ─────────┘

            abandon_turn: from anywhere back to awaiting_player
                          (turn_id + 1, pending computation dropped)
```

- `turn_id` starts at 1 and counts boundaries, so two mutations under one id is
  a readable symptom (Sprint 5 threads it into the trace record).
- `completed` is a boundary, not a resting state: `complete_turn` passes through
  it and comes back out awaiting the player. There is no idle phase between two
  turns.
- A player move that **ends the game** completes the turn immediately — there is
  nothing left to wait for.
- An **illegal** player move is a result, not a transition: the phase and the
  turn id are untouched, exactly as if nothing had been said.
- `collect_engine_reply` returns `None` and advances anyway when there is no
  engine or the game is over. Whether the engine owes a reply is derivable from
  the session, so no caller has to work it out — and it is derived *at collect
  time*, not remembered from when the computation started.
- **`abandon_turn` is the only other way out.** Undo, a new game, a resignation
  and a resumed save all replace the position the open turn is about, so each
  runs it before mutating (the tool handlers and their trusted-API twins both).
  It drops any pending computation, bumps the turn id, and comes back out
  awaiting the player; between turns it is a no-op, so the id keeps counting real
  turns rather than every button press. Without it, an undo mid-turn would leave
  the machine waiting to apply a reply decided for a board that no longer exists.

## Who owns the engine's reply

The coordinator, and only the coordinator. Two callers ask for it two ways, and
both are the same sequence:

- `play_exchange(move)` — the whole thing in one call, no beats. `/api/game/move`
  runs it (direct mode: a dragged move, no agent in the path) and so does the
  `make_move` tool when the registry was built `atomic_exchange=True`, which is
  the default and what the MCP server gets: an MCP call has nothing behind it to
  collect a reply, and its game must not stall half-way through a turn.
- **the beats** — `apply_player_move` → (the agent's reaction) →
  `collect_engine_reply` → `complete_turn`, which is what the command pipeline
  runs (`api._run_command`). App assembly builds the registry with
  `atomic_exchange=False` for exactly this: `make_move` applies the player's move
  and stops, and the pipeline owns what happens next.

The flag names the *sequencing owner*, never the boundary — the tool surface, the
validation, and the legality gate are identical in both modes. The one
caller-visible difference is that the atomic result carries `engine_move` and the
split one does not, because at that moment there is no reply to report.

`engine_opening_move()` is the same story for the engine's first move when the
player takes black, replacing the color-check-plus-`play_move` that `new_game`
had in both layers.

**The engine's reply is not a model-callable tool and must not become one.**
`test_engine_reply_is_not_a_callable_tool` pins that. A model that could ask for
the reply could also fail to ask, and the LLM-off invariant says the game keeps
playing regardless.

`complete_turn` refuses to close a turn while the engine still owes a reply, for
the same reason: closing early would be a back door to skipping the engine's
move, which is precisely what this module exists to prevent.

## The observe slot, and what fills it

`begin_observation()` marks the beat where Glitch reacts to the *verified* player
move — the audit's item 5 — and slice 2's narrator is the voice that speaks in it
(`docs/planner-narrator.md`: a persona-prompted, tool-free model call over
verified results, which is exactly what an observation beat needs and why it
cannot reorder or skip anything).

What fills it depends on the route, and neither route needed a new call:

- **the fast path** — `Brain.narrate` on the post-player-move board plus the
  `make_move` result, which is what it always did; the only change is that the
  result it reads no longer has the engine's answer in it.
- **the brain route** — the planner/narrator loop's own closing narration, which
  now naturally reacts to a player move alone for the same reason.

Then the pipeline collects the reply and appends a **deterministic** line
announcing it (`api._reply_announcement`). No second narration: the reaction has
already been paid for, and a model turn per engine move is precisely the latency
this beat is not allowed to add.

One honest gap: nothing calls `begin_observation()` yet, so `agent_observing` is
still a phase the app never enters. Neither route can bracket its own narration
from the pipeline — on the fast path it would be a one-line half-measure, and on
the brain route the reaction happens *inside* `get_agent_response`, which holds no
coordinator by design. Making the phase real is the live-progress slice's job
(audit item 19), and it needs a seam for the brain to report through; the collect
already accepts either phase, so nothing else has to change when it lands.

It is a phase rather than a callback because the reaction has to be **optional by
construction**. `collect_engine_reply` is legal from `player_move_applied` *and*
from `agent_observing`, so skipping the model — verbosity low, no brain
configured, a `ProviderError` — skips only the words, and at verbosity=low the
whole turn is one canned line assembled from the two results, byte-for-byte what
a plain move said before.

### Why the reply is computed in the background

Latency is the acceptance criterion: *if the observation beat makes a plain move
feel slower, it failed*. A reaction that runs before the engine is even asked
would add a model round trip to every move, so it doesn't. `apply_player_move`
starts the engine thinking the moment a legal move lands — a thread computing
`engine.choose_move` against a `GameSession` copy of the position — and the
narration runs while that happens. `collect_engine_reply` joins it, and the two
costs overlap instead of queueing.

The rule that makes it safe is that the background **touches nothing**: it reads
its own copy, and only the collecting thread ever submits a move, through the
session's legality gate as always. `EnginePlayer.choose_move` already worked from
`chess.Board(session.fen())`, and python-chess's `SimpleEngine` serializes
concurrent UCI commands internally, so no new locking was needed.

Two things could make a background answer wrong, and both are handled the same
way — discard it and ask the engine here and now:

- **the board moved under it** (an undo mid-turn). The position the computation
  started from is recorded, and a mismatch at collect time means the answer is
  about a position that no longer exists. Applying it blind would be a move from
  the wrong board.
- **it failed.** The exception is swallowed in the thread and surfaces from the
  synchronous ask instead, which keeps failure semantics exactly where they were
  (see the known edge below — audit item 20 still owns improving them).

The phase deliberately does not move when the computation starts:
`engine_calculating` marks where the *turn* is blocked on Stockfish, which is
what a UI progress line wants to show, and during the reaction the turn is
blocked on nothing.

## One validation layer

`TurnStateError` subclasses `ValueError`. That is the whole trick behind the
audit's item 10:

- **The agent** never sees an exception. `ToolRegistry.dispatch` already converts
  `ValueError` into `{"ok": False, "error": ...}`, so a turn-state rejection
  arrives as ordinary result data on the same road as a schema failure, an
  illegal move, and the confirmation gate's refusal — four kinds of "no", one
  shape, and the loop reads all of them and corrects.
- **Trusted callers** (`/api/game/move`, `/api/game/new`) catch it and answer
  409, next to the existing domain 409s for an impossible undo or resigning a
  finished game.

So the API path, the brain path, and the delegate/MCP paths stop drifting: they
converge on `dispatch`, and `dispatch` converges on the coordinator.

## Wiring

One coordinator per app. `app.build_app` creates it and passes it to both
`build_registry(ctx, coordinator, atomic_exchange=False)` and
`create_app(..., coordinator=...)`; `tests/fakes.scripted_app` and the eval
harness mirror that, flag included — a harness that measured the atomic tool
would be measuring a different sequencing owner than the one that ships. Both
parameters default to `None` and build a matched pair, which is what the MCP
server (its own process, owning nothing else) and the older single-purpose tests
use.

The coordinator holds the shared `ToolContext`, not a session or an engine, and
reads `ctx.session` / `ctx.engine` live on every call — `resume_game` swaps the
session object on the context mid-game.

## Known edges, deliberately left

- **An engine that raises mid-calculation** leaves the phase at
  `engine_calculating`. Recovery semantics are audit item 20 (Sprint 2), which
  owns defining what a failure between the player's move and the reply means;
  guessing at it here would be a policy invented ahead of its test. (A *failed*
  background computation is not this case: it is discarded and re-asked, so the
  failure arrives from the synchronous call exactly as it always did.)
- **A route that raises after the player's move landed** leaves the turn open, so
  the next command's `make_move` is refused as turn-state error data and *that*
  turn's close beat plays the owed reply. Self-healing rather than wedged, but the
  player pays an utterance for it — the resumability half of audit item 20.
- **No board version / expected-FEN precondition** yet, so two clients can still
  interleave turns on the one shared session (audit item 7, Sprint 2). The turn
  id is the hook that work will hang off.
- **Mutation counting is not enforced** (audit item 6, Sprint 2). The phases make
  "one player move, one engine move per turn" *expressible* — a second
  `apply_player_move` mid-turn is already refused — but the destructive-op budget
  and the per-turn tool withdrawal are that slice's.
