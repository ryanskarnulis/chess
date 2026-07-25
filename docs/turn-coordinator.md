# The turn coordinator — design note (2026-07-25)

`backend/src/chessapp/coordinator.py`. Sprint 1, slice 1 of the agent-control
replan (`docs/agent-control-audit-2026-07-25.md`, items 3 and 10).

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
                    ▼                                              │
          player_move_applied ──────────────────────┐              │
                    │                               │              │
  begin_observation │                               │              │
                    ▼                               │              │
            agent_observing ────────────────────────┤              │
                                                    │              │
                                        engine_reply│              │
                                                    ▼              │
                                          engine_calculating       │
                                                    │              │
                                                    ▼              │
                                          engine_move_applied      │
                                                    │              │
                                       complete_turn│ turn_id + 1  │
                                                    ▼              │
                                                completed ─────────┘
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
- `engine_reply` returns `None` and advances anyway when there is no engine or
  the game is over. Whether the engine owes a reply is derivable from the
  session, so no caller has to work it out.

## Who owns the engine's reply

The coordinator, and only the coordinator. `play_exchange(move)` is the whole
sequence in one call, and it is what the `make_move` tool, `/api/game/move`, and
(through the tool) the fast path all run — so a move dragged on the board and a
move typed at the agent go through the same machine. `engine_opening_move()` is
the same story for the engine's first move when the player takes black, replacing
the color-check-plus-`play_move` that `new_game` had in both layers.

**The engine's reply is not a model-callable tool and must not become one.**
`test_engine_reply_is_not_a_callable_tool` pins that. A model that could ask for
the reply could also fail to ask, and the LLM-off invariant says the game keeps
playing regardless.

`complete_turn` refuses to close a turn while the engine still owes a reply, for
the same reason: closing early would be a back door to skipping the engine's
move, which is precisely what this module exists to prevent.

## The observe slot, and what fills it later

`begin_observation()` marks the beat where Glitch reacts to the *verified* player
move — the audit's item 5. In this slice the slot exists and nothing fills it:
`play_exchange` skips straight to the reply, so a plain move costs exactly what
it cost before. Slice 3 fills it, and slice 2's narrator is the voice that speaks
in it.

It is a phase rather than a callback because the reaction has to be **optional by
construction**. `engine_reply` is legal from `player_move_applied` *and* from
`agent_observing`, so skipping the model — verbosity low, no brain configured, a
provider timeout — skips only the words. Latency is an acceptance criterion for
that slice: if the observation beat makes a plain move feel slower, it failed.

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
`build_registry(ctx, coordinator)` and `create_app(..., coordinator=...)`;
`tests/fakes.scripted_app` mirrors that. Both parameters default to `None` and
build a matched pair, which is what the MCP server (its own process, owning
nothing else) and the older single-purpose tests use.

The coordinator holds the shared `ToolContext`, not a session or an engine, and
reads `ctx.session` / `ctx.engine` live on every call — `resume_game` swaps the
session object on the context mid-game.

## Known edges, deliberately left

- **An engine that raises mid-calculation** leaves the phase at
  `engine_calculating`. Recovery semantics are audit item 20 (Sprint 2), which
  owns defining what a failure between the player's move and the reply means;
  guessing at it here would be a policy invented ahead of its test.
- **No board version / expected-FEN precondition** yet, so two clients can still
  interleave turns on the one shared session (audit item 7, Sprint 2). The turn
  id is the hook that work will hang off.
- **Mutation counting is not enforced** (audit item 6, Sprint 2). The phases make
  "one player move, one engine move per turn" *expressible* — a second
  `apply_player_move` mid-turn is already refused — but the destructive-op budget
  and the per-turn tool withdrawal are that slice's.
