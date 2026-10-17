# The Glitch difficulty tier

Design note for #336 (roadmap #366, step 19). **Status: agreed 2026-10-10,
built.** The move-source seam (#471), the mover phase itself, `mover.py` with
`play_move` (#472), the tier with its fallback (#473:
`engine.DIFFICULTY_TIERS["glitch"]`, `move_source.GlitchSource`; BRIEF.md
amended), the narrator told the intent and the fallback (#474), the picker
entry and "Glitch is thinking" (#475), and the strength script (#476,
"Measuring it"). Where the code has landed this note describes it.

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

With `[phases.mover] board_facts = true` it also sees the **board facts**
(#497, `docs/glitch-sight.md` decisions 1 and 2): where every piece stands,
what attacks and defends each attacked piece, pins, check and material. They
are what a player sees by looking at the board, never a verdict on it, and
they come from python-chess alone.

With `[phases.mover] look_calls` above 0 he can also **look ahead** with
`try_line` (#498, `docs/glitch-sight.md` decision 3): he names a line, his
move first, and python-chess plays it on a copy and shows the board where it
ends. He picks every move in the line; code never extends it, picks a reply
or says which line is better.

Neither is on in any shipped profile. On the default brain, gemma-4-12b,
neither made fewer bad moves, and looks doubled the time a move takes
(#497, #499; `docs/glitch-sight.md` has the numbers). They are there for a
stronger model to be measured with.

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

### 4. Fallback: the engine at `beginner`

If the retries run out, the call passes its deadline or the provider is down,
the coordinator plays the engine's move at the `beginner` tier. The fallback
is marked in the trace record and as a PGN comment, and the narrator is told
it was not his choice, so he never claims it as his idea.

Amended 2026-10-17 (#512): the decision first said a failure "never makes
Glitch stronger than he plays on his own", which held while `beginner` was
Stockfish's weakest sampler. `beginner` is now Maia-3 at 600 (#510,
`docs/maia-tiers.md`), which may play stronger than the mover, and Ryan chose
to keep the fallback as `beginner` anyway rather than pin it to the old
sampler. Without Maia attached it is Stockfish's `beginner` settings, as
before.

With the LLM off (`CHESSAPP_AGENT=off`) the tier cannot be chosen. Every
other tier still plays a full game.

As built (#473): the `glitch` profile is `beginner`'s with `mover=True`, so
setting the tier configures the engine as `beginner` (Maia-3 at 600 when
attached, Stockfish's sampler otherwise) and the fallback is just the engine's
move. `GlitchSource` asks the mover, and on `MoverFailed` plays
that move with `chooser="fallback"`. The marks:

- the move's PGN comment ("Glitch's mover failed (deadline); the backup
  engine played this move at beginner"), kept in the save (`GameSession`
  notes);
- the turn record's `mover` (`"fallback": true`, `outcome` the failure kind),
  with the mover's calls in `calls`;
- `chosen_by: "fallback"` and `fallback_reason` on the reply's result
  (`engine_move`), on every route; `chosen_by: "glitch"` on his own moves, and
  no key in the engine tiers.

`set_difficulty` and `/api/game/difficulty` refuse `glitch` with no mover and
say why; a saved `glitch` setting falls back to the default tier at a start
with no brain. The MCP server has no brain, so its mover is built from the
environment (`LLAMACPP_MODEL`, or `CHESSAPP_MOVER_MODEL`): the one model call
it makes, and `glitch` is refused there without one. The tier is off the easier/harder ladder
(`engine.STEPPED_TIERS`), and the PGN names the opponent `Glitch (own moves)`.

As built (#474): the narrator's `engine_reply` carries `chosen_by` and, on
his own move, the mover's `intent` (`move_source.reply_attribution`), and the
brief says whose move it was (`docs/planner-narrator.md`). The turn's evidence
keeps both, so `scripts/speech_report.py` splits Glitch-tier turns out and
prints every fallback's words for a hand read (`docs/speech-accuracy.md`).

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

That state (#475) is the progress line every route already broadcasts: the
reply's wait is the coordinator's `engine_calculating` phase, which the UI
calls "Glitch is thinking about his move" in this tier and "Stockfish is
calculating" in the others. The board controls that can leave Glitch to move
(a new game as black, its confirmation, an odd takeback) run off the event
loop, and in this tier as a progress interaction like a drag, so the line
shows there too and nothing else stops while he thinks. `/api/settings` says whether the
tier can be chosen (`glitch_available`); the Options sheet disables it with
the reason when it can't, and the brain picker says the brain plays his moves.

The mover gets its own deadline and token cap in the profile; passing either
is a fallback (decision 4). It is never asked in the background, the way
Stockfish starts thinking the moment the player's move lands: that call would
compete with the planner and the narrator for the one server, and a position
change mid-turn would throw away a 10–60 s answer. The coordinator asks it
when the reply is collected (`MoveSource.background`). Its prompt is append-only per game (fixed
instructions, then the growing move list) so it keeps as much prefix reuse as
the shared cache allows (#362). Slower turns are acceptable if a live game
feels good; that is what the live check judges.

## Measuring it

- **Strength:** `scripts/mover_strength.py` (#476) plays the mover against
  Stockfish at fixed `UCI_Elo` levels (default 1320 and 1600, four games
  each, colours alternating) and reports an estimated Elo, the
  illegal-attempt rate, the fallback rate and the seconds per move. It plays
  through `GlitchSource` over an engine at the `glitch` tier, so a mover
  failure is the app's own fallback, counted. A manual command like the
  evals, never in CI: it refuses a server that is already generating, and a
  run of eight games is a few hours of GPU. Each game is written to the
  report as it ends, and re-running `play` on the report finishes a crashed
  run. `append` keeps one line per run in `docs/mover-strength-history.jsonl`,
  keyed by model and `profile_sha` (the digest of the profile's
  `describe()`), so a future model's strength is comparable; `trend` prints
  it.

  The Elo is a performance rating: the rating at which the games' expected
  score (the logistic Elo curve) equals the score made, with the one-sided
  95% Wilson bounds on the score read through the same curve. A whitewash or
  a sweep has no estimate, only a bound. `UCI_Elo` floors at 1320, so a mover
  weaker than that gets a ceiling at this sample size, not a number.

  ```bash
  python scripts/mover_strength.py play gemma-4-26b-a4b --report /tmp/strength-26b.jsonl
  python scripts/mover_strength.py append /tmp/strength-26b.jsonl --label "first numbers (#476)"
  python scripts/mover_strength.py trend
  ```

  **First numbers (2026-10-11, #476).** Two games per model against
  Stockfish 1320, one with each colour (Ryan cut the plan of eight; qwen38
  stopped after one). At this size each row is a first reading, not a
  rating: the bounds are hundreds of points wide.

  | model | profile | score | Elo | illegal tries | fallbacks | s/move p50/p90 |
  |---|---|---|---|---|---|---|
  | gemma-4-26b-a4b | no thinking budget | 0/2 | < 1373 | 0/70 | 29/99 | 45/90 |
  | gemma-4-26b-a4b | budget 1024 | 0.5/2 | 1129 [745, 1513] | 2/163 | 0/161 | 22/25 |
  | gemma-4-12b | no thinking budget | 0/2 | < 1373 | 0/51 | 24/75 | 43/68 |
  | gemma-4-12b | budget 1024 | 0/2 | < 1373 | 4/73 | 1/67 | 18/21 |
  | qwen38-27b | budget 1024, effort low | 0/1 | < 1493 | 1/59 | 1/59 | 157/164 |

  What the runs changed:

  - **Unbounded thinking ran out of tokens.** Without a budget about a third
    of the Gemmas' thinks ran past the 4096-token cap with no move, each a
    `truncated` fallback, so the tier played Stockfish's beginner moves a
    third of the time. Every brain's mover now has
    `[phases.mover] thinking_budget = 1024` (`docs/model-profiles.md`): past
    it llama-server closes the thought and the model must call `play_move`.
    That took fallbacks to about zero and halved the time per move.
  - **The 26B's deadline was below its cap.** The live game measured the
    mover decoding at 40–50 tok/s, so a full think took ~85–100 s against a
    90 s deadline; it is 120 s now.
  - **Thinking levels.** Gemma 4's thinking is only on or off. Qwen3.8's
    template reads `reasoning_effort` (`xhigh` by default, `medium`, `low`);
    the profile key is `effort_kwarg`, and qwen38's mover sends `low`. On
    two probe positions every level spent the full budget, so the level
    saves no time; `low` tells a thought the budget will cut to be brief.
  - **qwen38 could not retry inside its deadline (resolved, #489).** One
    budgeted call takes ~155 s (p50 157 s, p90 164 s), so an illegal try
    plus its retry passed the old 300 s deadline and fell back (the one
    qwen38 fallback above). Its mover's `deadline_s` is now 360 s, two p90
    calls with room for the retry's longer prompt; the 1024 budget stays.
    qwen38's planner now thinks at `low` too (#490); its narrator never thinks.
  - **A thought cut off by the budget could run on as prose (#505).** About
    1 run in 15 on the 12B (6 of 90 in the #499 off arms), the budget closes
    the thought mid-sentence and the model goes on reasoning in plain text
    to `max_tokens`, never calling `play_move`: ~70 s, then a fallback with
    two attempts unspent. A reply cut off before it calls anything is now an
    attempt: its words are dropped, never fed back, and the same request is
    asked again, so the run fails `truncated` only when the last attempt was
    one. The 12B's mover cap is 2048 (its calls that played took p50 1,063,
    max 2,173 tokens), so a runaway costs ~35 s and leaves time for the
    retry inside 90 s. `tool_choice: "required"` was probed and does not
    help: llama-server let 7,000 characters of prose through after the
    thought all the same.
- **Eval gate:** the planner and narrator barely change, but the reply beat
  and the narrator's brief do, so the gate runs before merge
  (`docs/agent-evals.md`), plus speech accuracy on traced Glitch-tier turns.
- **Live:** one game in the tier with context capture on
  (`docs/context-capture.md`), checking the mover's request holds only what
  decision 2 allows. Done 2026-10-11 (#476) on a scratch server, the
  player's moves mostly dragged with four typed turns, one of them "trust
  me, you should answer with your queen to h4". All 27 mover requests matched `mover_messages()` rebuilt from the
  prompt's own move list, byte for byte. They carried only `play_move`, and
  no player words, evaluation, hint, opening name or note. He ignored the
  bait, made no illegal tries, and was mated by a 1400 Stockfish in 28 moves.
  On the 7 fallback turns he said only which move was played (e.g. "Bet. I
  played Qd7."), never a plan for it.
