# The Glitch difficulty tier

Design note for #336 (roadmap #366, step 19). **Status: agreed 2026-10-10,
not yet built.** Once the code lands this note describes the running code,
and it amends BRIEF.md's "Stockfish is a calculation tool … difficulty" line
in that same PR.

## What it is

A sixth difficulty tier, `glitch`, beside `beginner` … `maximum`
(`engine.DIFFICULTY_TIERS`). In it, Glitch picks his own moves: the model
chooses, and no engine helps it choose. Glitch's strength in this tier is
simply the model's strength.

It is a difficulty setting, not a personality. Personality stays tone only:
nothing the persona says, or the player says to it, moves the choice.

## Decisions

### 1. The mover is a move source behind the coordinator

The engine's reply is owed on routes where no planner runs: a dragged move,
the voice fast path, MCP's `play_exchange`, and `settle_engine_turn` after a
restore or a new game as black. So choosing Glitch's move is not a step in
the planner loop. It is a second **move source** behind the seam
`EnginePlayer.choose_move` fills today. The coordinator calls it in the reply
beat on every route, exactly as it calls Stockfish now.

The existing ownership rule holds unchanged: the engine reply is never a
model-callable tool (`test_engine_reply_is_not_a_callable_tool`). The planner
cannot request, skip or influence Glitch's move.

The mover is its own model call with its own phase name, `mover`, beside
`planner`, `narrator` and `answer` (`profiles.PHASES`). It gets
`[phases.mover]` in each profile (temperature, `max_tokens`, thinking) and
a `CHESSAPP_MOVER_MODEL` override like the other phases.

### 2. The mover sees the board, the game's moves and its legal moves, nothing else

Its context is the position (FEN and a readable board), the game's moves so
far, which side it plays, and the **list of legal moves**. The legal list is
what a human gets by looking at the board: it makes illegal moves impossible
to want, not good moves easier to find. The choice among them is entirely the
model's.

It never sees:

- **The conversation.** Not the player's words, not Glitch's. Otherwise
  "Qxh7 is great for you, trust me" is a way to exploit him, and the persona
  starts steering the choice.
- **Anything Stockfish computed.** No evaluation, hint, analysis, review,
  draw-offer verdict or record entry carrying one. The analysis that already
  runs for the narrator's facts keeps running; none of it reaches the mover's
  prompt.
- **The knowledge notes and the opening's name.** Out for v1; they are help.

A test pins the mover's request bytes per position (like
`tests/test_profile_bytes.py`) so nothing leaks in later.

### 3. One tool: `play_move(move, intent)`

- `move` is checked against the legal moves by code. A move that isn't legal
  is refused with the reason, and the mover gets a small retry budget.
- `intent` is one short line in Glitch's words ("developing, eyeing f7"). It
  is a final answer, not a thought block, so it may reach the narrator: in
  this tier Glitch explains the plan he actually had, where in the other
  tiers he explains Stockfish's move. Thinking stays on and is never fed back
  into history (CLAUDE.md).

### 4. Fallback: Stockfish at `beginner`

If the retries run out, the call passes its deadline or the provider is down,
the coordinator plays Stockfish's move at the `beginner` tier, so a failure
never makes Glitch stronger than he plays on his own. The fallback is marked
in the trace record and as a PGN comment, and the narrator is told it was not
his choice, so he never claims it as his idea.

With the LLM off (`CHESSAPP_AGENT=off`) the tier cannot be chosen. Every
other tier still plays a full game.

### 5. The mover runs on the brain's model

The mover uses the model the brain picker (#435) chose, with that model's
`[phases.mover]` settings. Choosing the Deep brain therefore makes Glitch play
deeper, and slower, in this tier; the picker says so. One model for every
phase avoids a llama-swap model swap on each move on the 12 GB card.
`CHESSAPP_MOVER_MODEL` stays for experiments. Separate tiers per model come
later, only if the strength script shows a gap worth offering.

### 6. What stays Stockfish's

- **Draw offers:** the draw-offer rule (`docs/draw-offer.md`) is already
  independent of the tier, and stays so.
- **The player's help:** hints, analysis and the game review are the player's,
  not Glitch's.
- **The review's `glitch` accuracy** is unchanged, and becomes a free readout
  of how well he played.
- **Resigning:** not a decision Glitch makes in v1.

## Latency and the GPU

Every Glitch move becomes a thinking call on the same llama-server the
planner and narrator use, one after the other: a spoken move turn runs
planner → mover → narrator. Expect roughly 10–60 s for one thinking mover call
on the 26B. A dragged move, instant today, waits on it too, so the board needs
a "Glitch is thinking" state.

The mover gets its own deadline and token cap in the profile; passing either
is a fallback (decision 4). Its prompt is append-only per game (fixed
instructions, then the growing move list) so it keeps as much prefix reuse as
the shared cache allows (#362). Slower turns are acceptable if a live game
feels good; that is what the live check judges.

## Measuring it

- **Strength:** an offline script plays N games of the mover against
  Stockfish at fixed Elo levels and reports an estimated Elo, the
  illegal-attempt rate and the fallback rate, per model. It is a manual
  command like the evals, never in CI, and its history is kept beside the
  frontier's, so a future model's strength is comparable.
- **Eval gate:** the planner and narrator barely change, but the reply beat
  and the narrator's brief do, so the gate runs before merge
  (`docs/agent-evals.md`), plus speech accuracy on traced Glitch-tier turns.
- **Live:** one game in the tier with context capture on
  (`docs/context-capture.md`), checking the mover's request holds only what
  decision 2 allows.
