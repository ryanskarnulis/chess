# The turn coordinator — design note (2026-07-25)

`backend/src/chessapp/coordinator.py`. Sprint 1, slice 1 of the agent-control
replan (`docs/agent-control-audit-2026-07-25.md`, items 3 and 10), extended by
slice 3 — the move flow split (items 2 and 5), which filled the observation slot
— and by slice 4, which brought the board controls in (items 1 and 4). Sprint 2's
first slice then made the per-turn mutation limits real (item 6) and its second
gave every mutation a board-version precondition (item 7). All of them are
folded into this note where they changed something.

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
  a readable symptom — and it is now readable *from one line*: every turn record
  carries the id it opened under next to the turn's mutation count (`trace.py`,
  audit item 18). Two is what a healthy move turn spends; a third under the same
  id is the duplicate the phases exist to refuse.
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
  runs it **in direct mode** (no brain configured, so there is no reaction to
  stand between the two moves) and so does the `make_move` tool when the registry
  was built `atomic_exchange=True`, which is the default and what the MCP server
  gets: an MCP call has nothing behind it to collect a reply, and its game must
  not stall half-way through a turn.
- **the beats** — `apply_player_move` → (the agent's reaction) →
  `collect_engine_reply` → `complete_turn`, which is what the command pipeline
  runs and, since slice 4, what a dragged move runs too (`api._play_move`, shared
  by both — see below). App assembly builds the registry with
  `atomic_exchange=False` for exactly this: `make_move` applies the player's move
  and stops, and the caller owns what happens next.

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

**The phase is real as of the live-progress slice** (audit item 19), and it took
the seam that slice's entry predicted. Two routes, two ways in, one conditional
method — `mark_observation()`, which opens the beat only when a verified player
move is actually waiting on one and is otherwise a no-op:

- **the fast path and the board drag** — `api._play_move` calls it directly,
  just before it narrates. The pipeline can see this beat, so it marks it, and
  it marks it whatever brain is behind the narration.
- **the brain route** — the reaction happens *inside* `get_agent_response`,
  which holds no coordinator by design, so the brain reports its own phase
  change instead (`LlamaBrain.on_phase` → `narrating`, in the brain's own
  vocabulary) and app assembly wires that report to `mark_observation`. The
  brain still knows nothing about turns; the wiring knows that a narrator turn
  landing mid-move *is* the observation.

`begin_observation()` stays strict — you cannot observe a move that has not been
made — because that is the rule the machine exists to hold. `mark_observation`
is the caller-facing form, and it has to be conditional: both narration sites
are reached on turns where nothing moved (a question, a settings change) and on
turns where the beat is already open.

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

## The board controls enter the machine (slice 4)

Two ways in were left outside it, and the audit's items 1 and 4 are both about
that.

**A dragged move.** `/api/game/move` used to play the whole exchange itself, so
in a game played by dragging pieces Glitch was never asked anything — the
"personality in the path" story only ever covered typed and spoken input. The
endpoint is now **mode-aware**:

- **agent mode** (a brain is configured) runs the beats — and runs them through
  the *same function* the command pipeline's fast path uses, `api._play_move`:
  dispatch `make_move` through the registry, narrate the verified player move
  while Stockfish computes, collect the reply, close the turn. One road in is
  worth nothing if two roads sequence a turn differently, so there is exactly one
  copy of that sequence. What reaches it is always a structured move string
  (`e2e4`) — the fast path parsed an utterance into one, a drag never had words —
  which is the audit's own wording for item 4. The response keeps every field it
  had and gains `commentary` and `speak`; the turn joins `ctx.transcript` under
  the move's SAN, so Glitch remembers games the player dragged, and it is traced
  as its own route (`board`) with the structured move where the utterance goes.
- **direct mode** (no brain) runs `play_exchange` and answers byte-for-byte what
  it always answered, new keys included: none. This is the LLM-off invariant, not
  a fallback, which is also why the mode is now *visible* —
  `/api/settings.agent_available` drives a standing indicator and locks the
  command box (which 503s in that mode anyway).

One asymmetry worth knowing: a drag that arrives while a turn is open is refused
409 in both modes, but agent mode *also* settles the turn that was left open (the
beats' close runs regardless), so the machine heals rather than staying wedged.
Direct mode's `play_exchange` raises before reaching it. That is the fast path's
existing behavior, inherited by sharing its code.

**Undo, new game, resign.** These bypassed the *confirmation* gate rather than
the turn machine: the buttons acted immediately while a spoken "new game" armed
`ctx.pending` and asked. Now `/api/game/new` and `/api/game/resign` dispatch
through the registry, so `tools._gate` — one implementation — decides for both
surfaces. On a fresh or finished board it stands aside and the op just runs (a
question about nothing is not worth asking). Mid-game it arms the op and the
endpoint answers **409 with the gate's question** (`{"detail", "confirm": true,
"op"}`; `confirm: true` is what separates a question from a failure), and
`/api/game/confirm` answers it — the *same* armed op, so a question raised by a
button can be settled by a typed "yes" and the reverse. Whichever way it is
answered, nothing stays armed, and a destructive op that *runs* clears anything
that was: a pending op is about a game that no longer exists. `random` is still
resolved before dispatch, so the op the player confirms is the game they were
asked about rather than a fresh roll. Undo is not destructive — it is gated by
nothing, and keeps its direct endpoint (which still `abandon_turn`s).

Two direct-mode behaviors moved with this, both toward the rule the tools
already derived: a mid-game reset asks first, and the resign button concedes for
the **player** rather than for the side to move.

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

## Mutation limits (Sprint 2, slice 1 — audit item 6)

Three mutations per turn, and each limit is code's rather than the prompt's:

- **One accepted player move and one engine reply** were already structural.
  There is no transition that admits a second of either — a second
  `apply_player_move` mid-turn is refused, and so is a second
  `collect_engine_reply` — so nothing has to be *counted* for this half. It is
  the shape of the machine, and `make_move`'s description no longer repeats it in
  prose: a rule code owns does not also live in the prompt.
- **One destructive op per command.** `new_game` and `resign` were gated but not
  budgeted, and the gate guards the *player's investment*, so it stands aside
  when there is none (a finished game, a board nobody has moved). That left a
  hole exactly one command wide: reset a finished game, then resign the fresh
  board that reset just made — both past the gate, both inside one user
  interaction.

The budget is **command-scoped, not turn-scoped**, which is the load-bearing
detail: `new_game` and `resign` each `abandon_turn` as part of doing their work,
so a flag the phase machine owned would reset itself on the way out and enforce
nothing. `begin_command`/`end_command` bracket one interaction,
`require_destructive_budget()` refuses a second op inside it, and
`record_destructive_op()` spends the budget only once the session has actually
changed. Check then record: a gate refusal, or a `resign` that raised on an
already-finished game, leaves the budget where it was, because nothing was thrown
away.

Only `api._run_command` opens a window, in a `try/finally` around the whole run —
a command that raises half-way must not leak an open window into the next button
press. It is the one surface that can chain several dispatches inside a single
interaction (the brain loop), and therefore the only one that can spend a budget
twice. The board buttons (`_run_destructive`), `/api/game/confirm`,
`/api/game/move` and the MCP server dispatch once per interaction by
construction and stay **windowless on purpose**: none of their behavior changes,
and an MCP client may legitimately start game after game across a session.

The refusal is a `TurnStateError` like the phase rejections, so it adds no new
failure shape — the agent reads `{"ok": False, "error": ...}` and the message
tells it to report what happened instead of retrying, and a trusted caller would
answer 409. What this deliberately does *not* do is withdraw the tool from the
offer mid-loop: a tool list that changes shape between iterations is a second
mechanism to keep in step with this one, and the refusal already arrives where
the model reads every other kind of "no".

## The board-version precondition (Sprint 2, slice 2 — audit item 7)

The turn machine says what may happen *next*; it never said which **board** a
request was about. One session is shared by the web UI, the delegate API a
conductor drives, and MCP, so two clients could both read the position and both
play into it, and the second move would land on a board its author had never
seen. Two pieces close that, and neither is the model's business:

- **`ToolContext.board_version`** — a monotonic int, published as `state.version`
  in every state document (so it rides back on `/api/state`, the websocket push,
  and every mutation response). It is *derived*, not counted by hand:
  `GameSession.revision` bumps inside each mutating session method, which is the
  one chokepoint a board mutation cannot avoid, so a player move, the engine's
  reply, an undo, a new game and a resignation all move it with no `bump()` calls
  sprinkled through the handlers. A read, an illegal move, a takeback of more
  plies than were played, an op the confirmation gate merely *armed* — none of
  them move it, because none of them changed anything.

  It lives on the **context**, not the session, because the session is the thing
  that gets replaced: `resume_game` swaps it, and resuming is itself a mutation
  clients must be able to notice. `ToolContext.replace_session` is that one
  bump, and it absorbs the incoming session's own replay revisions so the swap
  is worth exactly one.

- **An optional `version` on every mutating request** (`VersionedRequest` in
  `api.py`, covering move / undo / new / resign / confirm / command, and
  `MessageCreate.version` on the delegate wire). Supplied and superseded → 409
  with `{"detail", "stale": true, "version", "state"}`: the same shape family as
  the gate's `confirm: true` question, carrying what a client needs to resync and
  retry in one round trip. Supplied and current, or omitted → exactly today's
  behavior. Optional on purpose — the precondition is the *client's* opt-in, and
  the frontend can adopt it in its own slice.

**Atomicity is the load-bearing half.** A check that is not welded to the
mutation is theatre: FastAPI runs sync endpoints in a threadpool and async ones
on the loop, so requests genuinely interleave, and a whole turn can land between
a version read and the move it authorized. Every mutating endpoint therefore runs
inside `_mutation(expected)`, which takes `ctx.mutation_lock`, checks, and holds
it for the duration — including across `abandon_turn`, so a stale undo does not
cost an open turn on behalf of a request that was never going to be allowed. The
lock is acquired **off the event loop** (`anyio.to_thread`), because a waiter that
blocked the loop would stop the holder from finishing its own awaits and the two
would deadlock; it is a plain non-reentrant `Lock` for the matching reason, since
an owner-bound `RLock` would refuse a release from the thread that resumes the
request.

**MCP gets the lock but no `version` parameter.** Its tools are advertised from
the very schema objects the brain is offered (`registry.definitions()`), and that
shape is frozen by the eval floor on gemma-4-12b — TODO.md's standing warning.
So `mcp_server._mcp_tool` takes `ctx.mutation_lock` around each dispatch instead:
combined with its atomic `make_move`, that means two concurrent MCP calls take
turns and neither can leave a turn half-played, which is serialization rather
than staleness detection. A second MCP client can still play a move the first did
not expect; what it cannot do is interleave with one.

**The brain never sees a version.** Not in `_agent_state_dict`, not in a tool
schema. It is transport bookkeeping between the app and its clients, and the
house rule cuts both ways: code owns what code knows, and the model is not handed
a number it would only be tempted to reason with.

## A question is about a position (Sprint 5 — the E2E gap sweep, audit item 22)

The gap sweep over the audit's eight risks found one hole, and it was in the
confirmation gate: **an armed destructive op outlived the board it was a question
about.** `/api/command` disarms a pending op on its way past — an unrelated
utterance is a new intent, not an answer — but nothing else did, and the other
surfaces can move the board too. So this was reachable in three keystrokes:

    "I resign"        → the gate arms it and asks
    drag a move       → the board advances; the question is still armed
    "yes"             → a game two plies further on ends

Same shape with an undo, a resume, or another client's move in the middle. The
fix is the board-version machinery above, one layer down: `PendingOp` carries the
`board_version` it was armed against, and both answering surfaces read the op
through **`ToolContext.live_pending()`**, which drops one whose version no longer
matches. `confirm_pending` reads through it too — it is the last gate before a
game is thrown away, so it makes the check itself rather than trusting each
caller to have made it. Dropped, not queued: the "yes" then falls through as an
ordinary utterance, and the brain can ask what they meant.

Deriving it beats clearing it. The alternative was `ctx.pending = None` at every
surface that mutates — the drag, the undo, the resume, the next one somebody adds
— which is the "documented in the docstring and hoped for" pattern the house
rules exist to refuse. One stamp read at one place cannot be forgotten by a
route that did not exist when the rule was written.

The one wrinkle is *when* the stamp is taken. The gate arms **mid**-turn, and the
turn can still mutate after it — "play Bc4 and then start over" arms the reset
before the engine's reply lands — so an arm-time stamp would be stale before the
player could speak. What they are being asked about is the board at the *end* of
that interaction, so `ToolContext.restamp_pending()` re-points it there, once,
where the command closes. The button surface needs no such call: it dispatches
once and returns.

The sweep's other two additions are coverage rather than code — agent mode had no
concurrency test of its own (its turn is longer than direct mode's, so the window
for a second client is wider), and the delegate wire was never asserted to share
the pipeline's observe split, advice guard and provider recovery. `tests/
test_risk_sweep.py` holds both, and its docstring maps all eight risks to the
files that pin them.

## Saying it out loud: live progress

Once a turn has intentional phases, a spinner is the wrong shape for it — it
says "waiting" when the question the player has is *what for*. So the phases are
broadcast as they happen (`progress.py`, audit item 19), on the same websocket
the state document uses, discriminated by `type`.

Nothing narrates. Each event comes from the chokepoint that already owns the
thing it reports, which is the same rule that made `mutations` a `board_version`
delta rather than a tally:

| event | emitted by | reports |
| --- | --- | --- |
| `phase` | `TurnCoordinator._enter` — the one writer of `_phase` | `awaiting_player` … `completed` |
| `tool` | `ToolRegistry.dispatch`, after validation | the tool about to run |
| `brain` | `LlamaBrain`, per phase | `planning` / `narrating` |
| `begin` / `end` | the pipeline, bracketing the interaction | — |

Every event carries the interaction's `correlation_id` — the trace record's id —
so a line the player saw and the record of the turn behind it are one search
apart. That is what the trace slice minted the id for before anything on the
wire carried it. Which interaction an event belongs to travels in a
`ContextVar`, not on the reporter, because of the next paragraph.

**The pipeline's blocking steps run off the event loop, and that is a
requirement rather than a tuning choice** (`api._offloop`). A loop parked inside
a model call or a Stockfish search delivers nothing, so every event would arrive
in a burst after the turn it described had already finished. The model calls,
the move beats and the reply collection therefore run in worker threads; a
report crosses back with `call_soon_threadsafe` onto a queue that one pump
drains, which is also what keeps `begin` in front of `end`. The mutation lock is
held around all of it either way, so no ordering changes.

Two things deliberately report nothing. **Direct mode**: no brain, no
multi-phase turn — the one deterministic exchange answers, and the state
broadcast is the whole story. **The board's control buttons**: each dispatches
once and answers immediately, so a line that appeared and vanished inside one
round trip would be noise. Neither is a special case anybody wrote: neither
opens an interaction, and the reporter's rule is that an event outside one does
not exist. The same rule is why MCP is silent.

Reporting never costs a turn. Every observer call — the phase setter's, the
registry's, the brain's — is wrapped and swallowed, and a sink that cannot send
drops the event, exactly as a failing tracer is swallowed. A lost label is not a
lost turn.

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
- **A button-confirmed destructive op says nothing, but it is now traced.**
  `/api/game/confirm` still returns only the new state — the dialog already told
  the player what was about to happen, so no narration and no model call stand
  between the yes and the reset — but the interaction writes a record on route
  `control` (armed-and-asked, confirmed, or declined), which closes the gap the
  trace slice inherited. What is still untraced is the control surface that never
  passes the gate: `/api/game/undo`, and a dragged move in direct mode. Neither
  can be an *agent* failure — undo is not a destructive op and reaches the
  session directly, and direct mode has no agent at all — and an undo is still
  visible in the next record, whose `fen_before` will not match the previous
  record's `fen_after`.
- **A destructive op that runs on a windowless surface is unbudgeted** by design
  (see the mutation limits above). Two MCP calls in a row can still reset a game
  and resign it; what they cannot do is happen inside one command.
