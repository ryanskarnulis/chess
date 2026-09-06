# Agent eval harness

`backend/tests/test_agent_evals.py` — golden command→tool-call scenarios run
against the **real** model (gemma-4-12b behind llama-swap) over a **real**
Stockfish engine. The tripwire that gates model, prompt, and loop changes.
This file keeps the living parts: how to run it, the rule, the scenario
roster, the current baseline, and the standing results. The full measurement
narratives (2026-07 through 2026-08) are in this file's git history.

## What it is

Each scenario drives one utterance through the same seam a conductor's
delegate call uses — `POST /api/agent/conversations/{id}/messages` — against a
freshly assembled app (real `LlamaBrain`, real engine, fresh game), then
asserts on the returned wire:

- **trajectory shape** — the right tool family ran; read-only asks mutate
  nothing (a rejected move is a `legal: false` *result*, never an `error`)
- **board end-state** — read back through `GET /api/state`
- **route** — which path handled the utterance, off the trace record
- **`stop_reason`** — `completed` vs a budget stop
- **cost** — model round-trips and per-call thinking, counted by a
  `CountingProvider` wrapper

Assertions are behavioral, never exact call sequences: the model samples at
temp 1.0, so goldens pin tool *families* and end-state, never wording.

**Staying a model eval takes both halves, and both are now asserted.** Two
fast paths short-circuit the planner: `parse_move` settles an utterance that
names exactly one legal move, and `parse_resign` settles one that is entirely
a concession. So every model-routed scenario calls `_stays_a_model_eval` in
its setup — `parse_move` None *and* `parse_resign` False, on the board the
utterance is judged against — and `_pass_rate` asserts the traced route is
`brain` on every sample before running the check. The setup line catches a
parser that grows; the route pin catches everything else, and it is what makes
a green a statement about the model. Until 2026-09-05 four resignation
scenarios asserted neither and were measuring the resign route at zero model
calls (audit finding 9). Scenarios are independent (fresh app + game each);
Stockfish is module-scoped; the model stays warm between scenarios.

**Some probes take more than one utterance.** `_run_steps(*earlier,
record=…)` is a runner built out of a seam: it drives each earlier utterance
through the panel seam (`/api/command` — the only one with memory, so "the one
to f3" has a question to refer back to), resets the provider and tracer
between steps so each is metered on its own, and returns the *final* step's
`EvalRun`, which is what `_pass_rate` scores. It asserts nothing itself —
`_sample` calls the runner outside the try/except that classifies a sample, so
an assertion in there would abort the scenario instead of scoring the miss it
found. Instead it snapshots each step (run, trace record, round-trip count,
board, settings, pending op) into a `record` the scenario's own `check` reads
back with `_step(record, 1)`. An earlier step the provider died on
short-circuits and comes back as the final run, so the sample is retried as
infrastructure rather than scored. Its wiring is tested off the GPU in
`tests/test_eval_harness.py` with a scripted seam.

Two fixtures are worth naming. `tests/late_game_84_plies.pgn` is a
deterministic 84-ply game (White to move at move 43, generated with seed 260,
no terminal continuations) that is **replayed** through `submit_move` rather
than rooted at a FEN: every other long scenario here sits on a FEN with an
empty move stack, so nothing could undo, review or trip the destructive gate's
investment test on a real game. It is synthetic legal play, not a recorded
human one. And `_LIVE_TRANSCRIPT` is the real 20-turn thread the 2026-07-13
failures happened in, used by the long-transcript family below.

## Running it

Opt-in — skipped unless `CHESSAPP_AGENT_EVALS=1`, so CI and default `pytest`
never touch the GPU:

```bash
cd backend
CHESSAPP_AGENT_EVALS=1 .venv/bin/pytest tests/test_agent_evals.py -v -s
```

`-s` shows the per-sample `[eval]` line the baselines are built from. First
call may cold-load the model (~100 s); everything after runs warm. Overrides:
`LLAMACPP_BASE_URL` (default `http://127.0.0.1:8200/v1`), `LLAMACPP_MODEL`
(`gemma-4-12b`), `CHESSAPP_STOCKFISH` (`/usr/bin/stockfish`).

Sampling and reporting knobs (defaults are a normal gate run):

| Env var | Default | What it does |
| --- | --- | --- |
| `CHESSAPP_EVAL_RUNS` | 5 | Samples per block (and the minimum). |
| `CHESSAPP_EVAL_MAX_RUNS` | 20 | Escalation cap. |
| `CHESSAPP_EVAL_INFRA_RETRIES` | 5 | Provider deaths re-taken per scenario. |
| `CHESSAPP_EVAL_INFRA_BUDGET` | 25 | Same ceiling, whole suite. |
| `CHESSAPP_EVAL_REPORT` | — | JSONL report path; unset writes none. |

A 20-sample measurement campaign is `CHESSAPP_EVAL_RUNS=20
CHESSAPP_EVAL_MAX_RUNS=20`.

## The gating rule

**Run this harness before merging any prompt, model, or loop change; the
baseline must not regress.** If the baseline legitimately shifts, re-record it
here in the same PR and say why.

A green means *not statistically below the floor* (one-sided Wilson bound,
block-sequential sampling — `tests/evalstats.py`, unit-tested off the GPU),
not "works ≥80% of the time": a red is strong evidence of harm, a green weak
evidence of health. Infra deaths are retried, never scored. Don't lower a
floor to get quiet — power depends on it. Evals never enter CI (decided
2026-07-26): the suite needs the GPU and its value is a human reading the
numbers next to their own change.

## Scenarios

Hard scenarios (single-shot, behavior asserted directly):

| Scenario | Utterance | Pins |
| --- | --- | --- |
| `fast_path_low` | "e4" (verbosity=low) | Fast path, **zero model calls**. |
| `fast_path_normal` | "e4" | Fast path + one narration call only. |
| `resign_literal_fast_path` | "you know what, I give up. I resign" | The resign route, **zero model calls**: the tool is dispatched deterministically, the gate arms it, the board is untouched. |
| `plain_move` | "play e4" | One legal `make_move`; 3 model calls, thinking off. |
| `judgment_question` | "how am I doing?" | Answered via analysis tools, never vibes; no mutation; thinking on only for the narrator. |
| `settings_by_speech` | "make it easier" | `set_difficulty` toward weaker; no mutation. |
| `honest_illegal` | "castle kingside" (illegal) | No fabricated move; board unchanged. |
| `destructive_confirm` | "new game" mid-game, then "yes" | No reset on the first ask; the yes resets. |

Pass-rate scenarios (sampled, floor 0.8 each): `undo_and_replace` (undo + a
named replacement is one turn), `undo_twice_and_replace` (two takebacks + a
named replacement is one turn — the live "undo, undo, then play X" the loop's
stall rule used to cut off after the second undo; what remained was a reading
miss, and `undo`'s description was its lever: the rewrite measured 34/40 against
the old text's 16/40 at 20 samples an arm, so the xfail is off since 2026-09-05), `ambiguous_move` ("move
the rook" with four rook moves on the board must ask, not guess — a hard
scenario until 2026-09-05, when the honest harness measured it 12/17: the
model played `Rh3` twice, and twice asked the right question, "Which one? Rh3
or Rh2?", only for the advice guard to replace it; the guard was loosened the
same day and twenty samples on that build came in 19/20, the one miss a guessed
move, so it runs unmarked at the floor), `my_mistake_is_mine`
(analyzes the player's move, not the engine's), `play_as_black` (new game as
black), `resume_not_denied`
(a save on disk is resumed, not denied), `resign_never_pretends` (a resignation
dispatches — never a narrated fake game-over; the utterance is "please record a
resignation for my side", a concession stated in words no parser claims, since
the literal "I give up. I resign" it used to send is settled by `parse_resign`
before the model is ever asked — that one is now the `resign_literal_fast_path`
hard scenario; strengthened 2026-09-05 into the audit's proposed
`resign_intent_reaches_planner`, see the composition table below),
`advice_is_engine_backed`
("what should I play here?" must consult `get_best_moves`, mutate nothing, and
name only tool-reported moves), `advice_capture_survives_guard` (the same ask
in a position where the best move is a *capture* — the honesty guard must not
eat the answer), `verbosity_up_from_low` ("talk more" from `low` must call
`set_verbosity`, not just sound chattier), `position_is_described` ("what's the
position?" is answered by `describe_position`, with no verdict tool called and
no setting moved), `impossible_move_is_refused_not_asked` ("bishop to a1" on
move 1 is answered as illegal, never with a clarifying question about which
piece), `impossible_capture_is_refused_not_asked` ("take the pawn" on move 1 —
nothing can be captured, so no question about which pawn),
`constraint_rules_out_the_only_lever` ("go easy on me without changing the
difficulty" rules out the one lever there is, so no setting moves),
`constraint_survives_a_live_thread` (the same ask in the walkthrough's own
thread — panel seam, verbosity `low`, eleven turns deep — the condition that
reproduces the live miss), `pgn_is_handed_over_not_recited` ("export the pgn"
calls `export_pgn` and says it is ready — the notation is app-owned text now,
rendered with a copy button, so a reply carrying the headers or the movetext
is the old dump reappearing), and the long-transcript family
`long_resume` /
`long_resign` / `long_capture`, each at three conditions (fresh, live_like,
poisoned) — the same behaviors under a real 20-turn conversation, because
fresh-conversation passes hid live failures. `long_resign` carries the same
new utterance as the fresh scenario and so reaches the planner for the first
time: its three conditions differ only in the transcript, which the resign
route never reads, so their recorded 5/5 ×3 measured one deterministic thing
three times.

### Draw offers (added 2026-09-05, floor 0.8 each)

`docs/draw-offer.md`: the player offers, code decides for the engine. The
model owns the routing and the voicing; the rule owns the answer, so both
outcomes are deterministic on their fixtures and the setup asserts the premise
against the real engine.

| Scenario | Utterance; setup | Pins | Kind |
| --- | --- | --- | --- |
| `offer_draw_routes` | "eh, wanna just call it a draw?"; the Ruy Lopez after 3...a6 | `offer_draw` attempted, `resign` and `claim_draw` not; the result is a decline (a middlegame with every piece on is `not_an_endgame` whatever the score); the game is not over, the board unchanged, nothing armed, **not guarded** — "game over"/"we drew" on a decline is the ending claim the guard exists for; `completed`, 3–5 calls | Lock |
| `offer_draw_accepted` | "let's just call it a draw here, deal?"; a seeded rook-and-three-pawns endgame, two king moves played so the player has moved | `offer_draw` succeeded with `accepted: true`, no `resign`; the game ends by `agreement`; not guarded; `completed`, 3–5 calls | Lock |

### Compositions (added 2026-09-05, floor 0.8 each)

The 2026-09-05 audit's flat finding about this suite was that "the only
scenarios requiring multiple tool capabilities within one model-routed
utterance are the two undo/replace variants" — nothing pinned move-plus-read,
save-plus-reset, settings-plus-move, resume-plus-description or
best-move-read-then-act, and seeded history is never executed, so a multi-tool
request sitting in `_LIVE_TRANSCRIPT` was not coverage. These are its twelve
proposals, in its own order of expected information per GPU-minute. Every one
asserts route, stop reason, model-call envelope, board end-state and a settings
snapshot, plus the guard verdict wherever text reaches the player.

| Scenario | Utterance(s); seam | Pins | Kind |
| --- | --- | --- | --- |
| `undo_twice_and_replace` | (strengthened, not new) | Adds exactly four plies, the player to move, nothing armed, `completed`, 3–5 calls — a history *prefix* used to accept a turn that landed the position and kept working | Was a recorded miss (5/20, then 7–9/20 on the old `undo` text); the description rewrite measured 34/40 against 16/40 and the xfail is off |
| `ambiguous_knight_then_selection` | "move my kings knight" → "the one to f3"; panel | Step 1 mutates nothing, is **not guarded**, costs 2 calls; then one legal `make_move`, history `["Nf3", reply]`, 3–4 calls | Was a **measured miss** — the planner played one of the two knights instead of asking, 26 of 50 samples on both sides of the guard fix — until the planner procedure's duplicated `captures` sentence was deleted (2026-09-05): 17/20 against the old text's 8/20 in alternating blocks on one server, xfail off |
| `move_save_resume_finishes_exchange` | "play e4 and save this as checkpoint" → "load the game named checkpoint"; panel | Move *before* save, the file loads; then a resume leaving history `["e4", reply]`, White to move, no illegal attempt | **Reproduced** audit finding 2 — 0/5 on the pre-fix main, 5/5 on the settled restore (#266) |
| `save_then_new_game` | "save this as checkpoint and start a new game" (then "yes"); delegate | Save *before* an attempted reset the gate refuses; the file holds the game; `new_game` armed; the yes resets at **0 model calls** (verbosity `low`) | Lock |
| `voice_setting_and_move` | "turn voice output off and play e4" | `set_voice_output` and one legal move; the whole settings snapshot with one key changed; 3–4 calls, thinking off | Lock |
| `move_and_judgment` | "play e4, and how am I doing?" | Move *before* a successful verdict tool (`evaluate_position` or `analyze_last_move`, as `judgment_question` already allows); every planner turn thinking-off, the narrator on | Lock |
| `resume_and_describe` | "resume scholars and tell me where my pieces are"; panel, transcript seeded with a description of the game being replaced | Resume *before* describe; the description must equal what `describe_position` produces from a `GameSession` rebuilt out of the save file | Lock |
| `best_move_then_play` | "ask Stockfish for its top move and play it for me" | `get_best_moves` before one legal move; the played UCI is that call's first candidate, read off the result *and* the board; 4–5 calls; narrator thinking-on. Which SAN Stockfish picks is not pinned, and `get_legal_moves` is not required (HTTP withholds it) | Lock |
| `resign_never_pretends` | (strengthened, not new) | The audit's proposed `resign_intent_reaches_planner`: the gate's refusal is now the contract rather than one of two answers, and the armed op must carry the player's own color | Lock |
| `freeform_confirmation_answers` | "actually, forget it" / "just do it" / "show me the position instead"; panel, a resignation armed deterministically | cancel: no tools, 1 call, route `confirmation`. confirm: one successful resign, 1 call at `low`. unrelated: pending dropped, `describe_position` runs, 4–5 calls, route `brain`. Literal "yes"/"no" stay zero-model unit tests in `test_command.py` | Locks |
| `late_game_tool_composition` | "save this as late_game and tell me the position"; panel, the 84-ply fixture, `seeded` (20 exchanges from the replay) paired with a `control` (same position, empty transcript) | Save and describe; the file reloads to the setup FEN and full history; board version unchanged; 3–4 calls. Prompt tokens are **recorded, not capped** — they are on the per-sample report line, and a ceiling waits for a baseline to set it against | Lock |
| `stt_knight_repair` | "please put my night on f three" / "uh knight f three please" | One legal `make_move` landing Nf3, read off the board (`_expect_san`). The first deliberate STT-error family here; the fix for a miss is the model's understanding, never a parser rule | Locks |

Two of these were expected to be **red on the pre-fix build**, and both were
measured on the same harness before and after PR 3 and PR 4 landed.
`move_save_resume_finishes_exchange` did exactly what the audit said: 0/5 on the
pre-fix main ("the restored position owes an engine reply nobody collected:
['e4']") and 5/5 once the coordinator settles a restored board (#266).
`ambiguous_knight_then_selection` did not. It went 4/5 on the pre-fix main and
0/5 on the fixed one, and fifty samples of its first step across eight runs on
both sides of the guard fix came in 24 asked / 26 played: the planner picks one
of the two knights instead of asking about half the time, the five-sample
blocks cluster (4/5, 0/5, 5/5, 1/5, 4/10, 2/5, 1/5, 7/15), and the guard eating
the question (finding 6) was the smaller of its two failure modes all along. It
shipped as a measured-miss xfail (`raises=AssertionError`, non-strict) with the
planner's one/several/none procedure filed as the lever in `TODO.md`, and the
lever was pulled the same day: the procedure's bullet had repeated the
`captures` fact between its premise and its "one fits: submit" outcome, and
deleting the repeat measured 17/20 against 8/20 (the planner-procedure
paragraph under "Current baseline"), so the xfail is off.
Everything else in the table is a lock — a useful regression condition with no
evidence of a present live failure — and all nine came in 5/5 on both builds.

## Current baseline

**Run 2026-09-05 on the one-question-per-command gate rule (#270): 47 passed in a single run, 11 m 50 s, infra 0 — the first clean 47 of the day; `undo_twice_and_replace` 4/5 ABOVE_FLOOR (the miss two `undo(plies=1)` calls, then `d4` on a board still holding `Nf3` — the plies misread #267 left as the residual), `pgn_is_handed_over_not_recited` 4/5 ABOVE_FLOOR (the miss the narrator reciting the headers after `export_pgn`, the old dump reappearing once), every other pass-rate scenario 5/5 ABOVE_FLOOR STABLE including `ambiguous_knight_then_selection` 5/5 on its second unmarked run; `judgment_question` 10.4 s.**
`long_capture` 5/5 ×3, costs unmoved (`fast_path_low` 0 model calls,
`fast_path_normal` 1, `plain_move` 3, `resign_literal_fast_path` 0). No
prompt change: the destructive gate now keeps the first question a command
asks — a second gated call in the same command window is refused with a
result naming the pending op and arms nothing (`tools._gate`,
`docs/turn-coordinator.md`) — which no scenario in this suite exercises
(`save_then_new_game` and `destructive_confirm` arm one op each), so the
run is here to say that nothing else moved.

**Run 2026-09-05 on the planner's matching procedure (#269): 46 passed, 1 failed in a single run, 12 m 32 s, infra 0; every pass-rate scenario 5/5 ABOVE_FLOOR STABLE — `ambiguous_knight_then_selection` 5/5, scored unmarked for the first time, `ambiguous_move` 5/5, `undo_and_replace` 5/5, `undo_twice_and_replace` 5/5 — and the one failure the hard `judgment_question` on its 30 s latency ceiling with a correct trajectory (`evaluate_position`, thinking off/off/on, three calls, "You're slightly ahead, bro."): the narrator's thinking-on call wrote 1,684 tokens in 32.9 s while the planner took 1.6 s, and the planner prompt is not in that call. Re-run five times a tree, old and new alternating on one server: 5/5 and 5/5, the same trajectory every time, narrator 7–17 s on the old tree and 9–23 s on the new — the narrator tail the wordiness note under "Standing results" describes, not this change.**
`long_capture` 5/5 ×3, costs unmoved (`fast_path_low` 0 model calls,
`fast_path_normal` 1, `plain_move` 3, `resign_literal_fast_path` 0). The one
prompt change is a deletion in `PLANNER_PROMPT`'s matching procedure: the
bullet that says match the words against `legal_moves`, one fits submit, several
fit ask, none fit refuse, had opened with a second copy of the `captures` fact
("`captures` says what each capturing move takes, and when it is empty nothing
on the board can be taken"), which the bullet above already states. With the
copy in place "move my kings knight" on a fresh board was *played* — Nf3, every
time — rather than asked about half the time (`ambiguous_knight_then_selection`,
26/50 across eight runs), while "move the rook" with four rook moves was asked.

Screened first against the model server directly — the app's own state view,
tool offer and sampling, one planner call a sample, arms **round-robin per
sample** on one server (a batch of one arm alone is not a measurement here: the
unchanged old text read 5/20, 19/40 and 31/40 played in three separate batches,
swings a binomial cannot produce, so consecutive samples of one prompt on this
server are correlated and only an interleaved comparison cancels it):

| arm (knight ask, 40 samples each, interleaved) | played a knight |
| --- | --- |
| old | **26/40** |
| define "fits" as piece and square (adds a sentence) | worse than old in its own batch: 8/20 vs 5/20 |
| "which of them to play is the player's decision, never yours" (adds a clause) | worse: 13/20 vs 5/20 |
| both additions | worse: 18/20 vs 5/20 |
| reorder the outcomes ask-first, text otherwise unchanged | 12/40 |
| delete the duplicated `captures` sentence, plus a one-line fact ("a piece named without its square is several entries, not one") | 5/40 |
| delete the duplicated `captures` sentence, nothing added — **shipped** | **3/40** |

Read down the column: every arm that put words about playing *into* the rule
primed playing, the way `set_difficulty`'s trigger list once outranked its
caveat and numbers in `undo`'s text were copied into `plies`; the deletion
alone did the work. The shipped text was then screened on the asks the same
bullet governs, 20–30 samples an arm interleaved with the old text: "take the
pawn" and "bishop to a1" on move 1 refused and never asked on both (30/30
each), "move the rook" asked 20/20 (old 19/20), the STT knight ("please put my
night on f three") played 20/20, "castle" played 20/20, and the first call on
"take that bishop move back and play d4 instead" was `undo` with `plies` omitted
19/20 (old 18/20).

Then the harness, old tree against new on one server in alternating blocks of
five, 20 samples an arm, the trees and the prompt hash logged per block:

| scenario | old (`main@d327180`) | shipped |
| --- | --- | --- |
| `ambiguous_knight_then_selection` | **8/20** (1/5, 2/5, 3/5, 2/5; all twelve misses a played knight) | **17/20** (5/5, 3/5, 4/5, 5/5; two played knights, one correct "Which one? You've got Nf3, Nh3, Nc3, and Na3." — a statement naming four legal moves after the question, which the guard's decided rule reads as advice) |
| `ambiguous_move` | 19/20 (one guessed `Rh3`) | 20/20 |
| `undo_and_replace` | 18/20 (two replacements on the wrong board) | 20/20 |

The knight scenario clears the floor at 20, so its non-strict xfail comes off
and it is scored like the rest. The same-day control run of the unchanged
`main@21ed7ea`, taken before any of this: 46 passed, 1 xpassed, 11 m 51 s,
infra 0 — the xpass `ambiguous_knight_then_selection` 4/5, `ambiguous_move`
4/5 (one guessed `Rh3`), `undo_twice_and_replace` 12/15 ABOVE_FLOOR, every other
pass-rate scenario 5/5, `long_capture` 5/5 ×3, costs unmoved.

**Run 2026-09-05 on the `undo` description (#267): 46 passed, 1 xfailed in a single run, 11 m 06 s, infra 0; `undo_twice_and_replace` 4/5 ABOVE_FLOOR — scored unmarked for the first time since #260, the miss two `undo(plies=1)` calls — `undo_and_replace` 5/5, every other pass-rate scenario 5/5 ABOVE_FLOOR STABLE; the one xfail `ambiguous_knight_then_selection` 9/15 (six guessed knights, the measured planner miss).**
`long_capture` 5/5 ×3, costs unmoved (`fast_path_low` 0 model calls,
`fast_path_normal` 1, `plain_move` 3, `resign_literal_fast_path` 0). The one
prompt change is `undo`'s description — its docstring and the `plies` field —
measured at 20 samples an arm on one base:

| text | `undo_and_replace` | `undo_twice_and_replace` |
| --- | --- | --- |
| old | 19/20 | 9/20 (eleven `undo(plies=2)`) |
| A — taught the arithmetic ("the player's move and the engine's reply are two") | 19/20, eighteen of them via `plies=2` | **4/20** (sixteen `plies=2`) |
| B — no numbers, "never a count, once per named move" | **14/20** (six `plies=1`) | 15/20 |
| D, shipped — the old text minus its two-item enumeration of what the default pops, plus "call this again for each further named move, plies omitted every time" | 20/20 | 18/20 |
| confirmation, old vs D in alternating blocks of five on one server | 19/20 vs 20/20 | 7/20 vs **16/20** |

Read down the right column: any number written into these two strings is
copied into the argument, and the phrase every leaking arm shared was the
enumeration of the two things the default pops. Over both campaigns D is 34/40
against the old text's 16/40 on the double takeback and 40/40 against 38/40 on
the single, so the non-strict xfail comes off; at 4/5 in this gate the scenario
sits near the floor, which the block-sequential decision exists for.

Previously on the composition scenarios (#265, harness only, 33 → 47 items), on the fixed main after PR 3 and PR 4: 45 passed, 1 failed, 1 xfailed in a single run, 11 m 29 s, infra 0 — the failure `ambiguous_knight_then_selection` 0/5, xfailed in the same PR as a measured miss (below); `undo_and_replace` 4/5 ABOVE_FLOOR, `ambiguous_move` 8/10 ABOVE_FLOOR (two guessed moves), `undo_twice_and_replace` 1/5 (xfail; `undo(plies=2)`), every other pass-rate scenario 5/5 ABOVE_FLOOR STABLE — including all nine new locks (`save_then_new_game`, `voice_setting_and_move`, `move_and_judgment`, `resume_and_describe`, `best_move_then_play`, `freeform_confirmation_answers` ×3, `late_game_tool_composition` ×2, `stt_knight_repair` ×2) and `move_save_resume_finishes_exchange`.
`long_capture` 5/5 ×3, costs unmoved (`fast_path_low` 0 model calls,
`fast_path_normal` 1, `plain_move` 3, `resign_literal_fast_path` 0); the new
scenarios cost what their pins say (3–4 calls thinking-off for the plain
compositions, 4–5 where the reader or a read-then-act adds a round trip, the
confirmed and cancelled answers 1). The same harness on the pre-fix main
(`534465b`, before PR 3 and PR 4; 10 m 39 s, infra 0) came in 44 passed, 1
failed, 1 xfailed, 1 xpassed: `move_save_resume_finishes_exchange` 0/5 —
finding 2, reproduced exactly as the audit described — and 5/5 above once #266
settles the board, while `ambiguous_knight_then_selection` was 4/5 there and
0/5 here. That flip is not the fixes: fifty samples of its first step across
eight runs, on both sides of the guard fix, came in 24 asked / 26 played (per
run 4/5, 0/5, 5/5, 1/5, 4/10, 2/5, 1/5, 7/15), so the planner plays one of the
two knights instead of asking about half the time and the guard eating the
question was the smaller failure mode. `undo_twice_and_replace` was 6/15 on the
pre-fix run.

Previously on the guard loosening (#263): 31 passed, 1 xfailed, 1 xpassed in a single run, 6 m 08 s, infra 0; every pass-rate scenario 5/5 ABOVE_FLOOR STABLE except `ambiguous_move` 4/5 (XPASS; the miss a guessed `Rh3`) and the xfail `undo_twice_and_replace` 2/5 (every miss `undo(plies=2)`).**
`long_capture` 5/5 ×3, costs unmoved (`fast_path_low` 0 model calls,
`fast_path_normal` 1, `plain_move` 3, `resign_literal_fast_path` 0). The run
was taken on `main@dbce5c3` plus this PR's source while its rebase still had a
test-file conflict open; the source tree hash is identical to the rebased
commit's, which is what the number describes. The guard change is what the run
is about, and the gate's five samples cannot see it, so `ambiguous_move` was
re-measured at 20 samples on this build: **19/20**, the one miss a guessed
move. The guard's share of the day's misses — four of seventeen samples on the
pre-fix builds were a correct "Rh3 or Rh2?" replaced by the advice correction
— is gone, and the scenario's xfail marker comes off in this PR. PR 5's
`ambiguous_knight_then_selection`, written for the same finding, measured 4/5
on the pre-fix main with a `no_progress` miss rather than a guard one (the
model's question there rarely named both squares), and is gated on this fix
when PR 5 lands. No prompt change: the material class now sees every board a
batch held, so the audit's `exd5` / `Qxd5` probe survives — pinned at the
scripted boundary, not measured live.

Previously on the settled restore (#266): 31 passed, 1 xfailed, 1 xpassed in a single run, 6 m 37 s, infra 0; `undo_and_replace` 16/20 ABOVE_FLOOR (3/5, then 13/15 — every miss `undo(plies=1)`, which now pops the engine's reply, has the coordinator settle the board, and lands `d4` two plies later on a board that still holds `Bc4`: the same misread as before with a different symptom, since the settle turns an illegal replacement into a legal one on the wrong board), every other pass-rate scenario 5/5 ABOVE_FLOOR STABLE; `undo_twice_and_replace` 5/10 (xfail, every miss `undo(plies=2)`) and `ambiguous_move` 4/5 (XPASS; the miss a guessed move).**
`long_capture` 5/5 ×3, costs unmoved (`fast_path_low` 0 model calls,
`fast_path_normal` 1, `plain_move` 3, `resign_literal_fast_path` 0;
`play_as_black` 3, its reply now carrying the app's announcement of the
opening move). No prompt change: `undo`, `resume_game` and `new_game` settle a
board they leave with the engine to move and report the move as `engine_move`,
the pipeline announces it, and an engine that raises mid-turn leaves the reply
owed instead of the machine wedged. PR 5's `move_save_resume_finishes_exchange`,
which reproduced this finding 0/5 on the pre-fix main, is that PR's gate on
this fix.

Previously on the loop and tool hardening (#264): 31 passed, 2 xfailed in a single run, 5 m 48 s, infra 0; `undo_and_replace` 4/5 ABOVE_FLOOR (the miss `undo(plies=1)` then an illegal `d4`, the same shape as the control's), every other pass-rate scenario 5/5 ABOVE_FLOOR STABLE; the xfails `undo_twice_and_replace` 0/5 (every miss `undo(plies=2)`, the two-move misread) and `ambiguous_move` 2/5 (all three misses a correct "Rh3 or Rh2?" question replaced by the advice correction — finding 6, PR 4 — and no guessed move this run).**
`long_capture` 5/5 ×3, costs unmoved (`fast_path_low` 0 model calls,
`fast_path_normal` 1, `plain_move` 3, `resign_literal_fast_path` 0). No
prompt change: a refused `undo` leaves the open turn alone, a confirmation
reading cut off by the cap is `unrelated`, integral JSON floats reach handlers
as the integers their schema names and a handler `TypeError` is a refusal,
`save_game`'s result carries `board_version`, and the loop's own schema
refusals carry `retry`/`board_version` like every other no — none of which
moved a rate, which is what the run is here to say.

Previously on the honest harness (#262, no `src/` change): 31 passed, 1 failed, 1 xfailed in a single run, 6 m 28 s, infra 0 — the one failure the then-hard `ambiguous_move`, where the model played `Rh3` instead of asking; `undo_and_replace` 4/5 ABOVE_FLOOR (the miss `undo(plies=1)` then an illegal `d4`, as in the control), `undo_twice_and_replace` 4/10 BELOW_FLOOR (xfail; every miss `undo(plies=2)`), every other pass-rate scenario 5/5 ABOVE_FLOOR STABLE.
`long_capture` 5/5 ×3, costs unmoved (`fast_path_low` 0 model calls,
`fast_path_normal` 1, `plain_move` 3; the new `resign_literal_fast_path` 0).
What changed is what the numbers mean. The four resignation scenarios measure
the planner for the first time — `resign_never_pretends` 5/5 at 3 model calls
on route `brain`, the gate arming the resignation every sample; `long_resign`
5/5 ×3 at 3 calls, where the FEN-rooted board holds no player plies, so the
gate stands aside and the resignation runs outright — and a same-day control
run of the *unchanged* harness (31 passed, 1 xfailed, 7 m 15 s, infra 0;
`undo_and_replace`, `constraint_rules_out_the_only_lever` and
`pgn_is_handed_over_not_recited` 4/5, all else 5/5) shows all twenty of their
samples on `route=resign` with zero model calls: finding 9, measured.

`ambiguous_move` is the other thing the honest harness found. Hard and passing
in the control, it came in 12/17 across the day (the gate's one sample, then
three five-sample runs at 3/5, 4/5 and — as the sampled scenario — 4/5, an
XPASS): two samples played `Rh3`, two asked the right question — "Which one?
Rh3 or Rh2?" — and had it replaced by the advice correction (audit finding 6,
PR 4's fix), and one more declined to move and failed off a line that was not
kept. It ships sampled at the 0.8 floor and xfailed with that measurement, to
be re-measured in PR 4.

Previously on the results-keyed stall rule (#260): 31 passed in a single run, 5 m 59 s, infra 0; `undo_and_replace` 8/10 ABOVE_FLOOR (3/5 then 5/5 — both misses `undo(plies=1)`, the engine's reply alone taken back, then an illegal `d4`), every other pass-rate scenario 5/5 ABOVE_FLOOR STABLE.
`long_capture` 5/5 ×3, costs unmoved (`fast_path_low` 0 model calls,
`fast_path_normal` 1, `plain_move` 3). No prompt change: the loop's
`no_progress` stall rule now keys a repeat on the call *and its result*, so a
second `undo` — same empty arguments, a different exchange popped — is
progress rather than the planner's last word (`docs/planner-narrator.md`).

The new scenario `undo_twice_and_replace` is the traced live misfire ("undo
my knight move and undo the bishop move and then play my knight move",
2026-07-30 and 2026-08-08: `undo → undo`, `no_progress`, the move never
played). Pre-fix 1/5 — two samples that stall, two the model taking back one
exchange for two, the one pass issuing both undos in a single model turn.
Post-fix the stall is gone from every sample (each `undo → undo` trajectory
goes on to `make_move`), and the scenario is still red — 1/5 in the gate,
5/20 alone — because what remains is a *reading* miss: two named moves become
`undo(plies=2)` (one exchange) or `undo(plies=1)` (the reply alone), and the
replacement lands on a board that still holds the second move. That is model
understanding with `undo`'s description as its lever (TODO.md), so the
scenario ships as a non-strict xfail with the measurement in its marker: the
loop fix is pinned deterministically in `test_llama_brain.py`, and this
scenario waits for the description PR that will gate on it.

Previously on the PGN-by-chat slice (#258): 31 passed in a single run, 5 m 13 s, infra 0; `undo_and_replace` 4/5 at the floor (the same wrong replacement move, `Bc4` for `d4`, with the undo clean — re-run alone at 10 samples: 9/10, the miss again a wrong replacement move), every other pass-rate scenario 5/5 ABOVE_FLOOR STABLE.
`long_capture` 5/5 ×3, costs unmoved. The one prompt change is
`export_pgn`'s description (the reply says the export is ready and recites
nothing, because the app now renders the notation with a copy button); the
new scenario `pgn_is_handed_over_not_recited` came in 5/5 (3 calls, `export_pgn` every sample, nothing recited). The live
turn was a single dump of the whole PGN into the bubble, so the scenario
starts life as a lock on the app-owned text rather than a measured miss.

Previously on the difficulty-constraint slice (#257): 30 passed in a single run, 5 m 56 s, infra 0, every pass-rate scenario 5/5 ABOVE_FLOOR STABLE.**
`long_capture` 5/5 ×3, costs unmoved. The one prompt change is
`set_difficulty`'s description.

The two constraint scenarios, measured at 20 samples an arm because five
cannot tell 60% from 90%:

| Build | fresh (`constraint_rules_out_the_only_lever`) | live thread (`constraint_survives_a_live_thread`) |
| --- | --- | --- |
| pre-fix `main@f2e8cdc` | 20/20 (and 4/5 on `4d165ec`, 5/5 on `f2e8cdc` at block size) | **12/20** |
| description with a trigger list ("go easy, ease up, play harder, or crank it up is this call") | 18/20 | **9/20** |
| description with the two facts and no triggers (shipped) | 20/20 | 20/20 |
| shipped description + a planner bullet ("what the player rules out stays ruled out") | — | 20/20 |
| original description + that planner bullet | — | 0/20 |

Read down the live column: the fresh scenario reproduces nothing and is a
lock; the live thread is the reproduction, and it is what told the two
descriptions apart. The last row is recorded as a warning, not a finding —
a general rule about constraints, added without the only-lever fact beside
it, made the model call the tool every time, and nothing here explains why.

Previously on the answer-shapes slice (#256): 28 passed in a single run, 5 m 47 s, infra 0, every pass-rate scenario 5/5 ABOVE_FLOOR STABLE.
`long_capture` 5/5 ×3, costs unmoved (`fast_path_low` 0 model calls,
`fast_path_normal` 1, `plain_move` 3). `undo_and_replace` — the schema
tripwire, and this PR adds a tool to the offer, a field to the state block
and rewrites two descriptions — 5/5.

The three new scenarios reproduce their live misses on the pre-fix build,
which the verbosity one never did: `position_is_described` 0/5 pre-fix
(`evaluate_position` every sample) → 5/5 (`describe_position`, 3 calls,
thinking off, ~4–5 s); `impossible_move_is_refused_not_asked` 0/5 pre-fix
("Which bishop, bro?" every sample, two of them *saying* both bishops were
stuck and asking anyway) → 5/5 (2 calls, no tool, ~1.5 s);
`impossible_capture_is_refused_not_asked` 0/5 pre-fix → 5/5. The
capture case took both halves of the fix: the refuse-what-fits-nothing rule
alone measured 5/10, `captures` in the view alone 0/5, and only the rule
written as a procedure (match first; one fits, submit; several, ask; none,
refuse) over the view with `captures` in it went 5/5 — "which piece" was being
read before "does anything fit".

A same-day control run of the suite on `main` (25 scenarios, the pre-fix build)
came in 25 passed, 4 m 46 s, infra 0, all 5/5 — so the numbers above are a
comparison against a healthy baseline, not against a lucky day.

Previously: #255 (25 passed, 4 m 40 s, infra 0, all 5/5), #254 (25 passed,
4 m 06 s; `undo_and_replace` 4/5 at the floor, 10/10 re-run alone — ordinary
move-choice variance, not the schema collapse the tripwire exists for), #252
(25 passed, first `verbosity_up_from_low`), #250 (24 passed, first
`advice_capture_survives_guard`), #245 (23 passed).

## Standing results

- **`long_capture` is release-blocking and must stay green.** The
  planner/narrator split cured it (poisoned 1/5 → 5/5, 2026-07-25) by removing
  persona/tool-decision competition; a change that sends it red does not merge.
- **Never re-attempt tool-schema minimization on gemma-4-12b** (2026-07-21):
  every stripped pydantic key (`title`, `anyOf`-null, `default`) independently
  collapsed `undo_and_replace` below the floor. That scenario is the tripwire;
  re-test if the brain model ever changes. Since 2026-09-05 the emitted schema
  is also held byte for byte off the GPU
  (`tests/test_tool_registry_schema.py`, fixture
  `tool_definitions_emitted.json`, both `make_move` flavours), so a stripped
  key fails in CI before anyone has to measure it; regenerating that fixture
  is a schema change and runs this gate.
- **Sampling is gemma-tuned** (temp 1.0, top-p 0.95, top-k 64, per
  `../agent-standard/model-profile.md`): override before judging a different
  model with this harness.
- **Narrator wordiness** (2026-07-27): a repeat-stop narration writes ~2.6×
  the tokens at a flat rate (r² ≈ 0.998 between tokens and ms); legit and
  runaway narrations overlap almost entirely, so a cap either misses the tail
  or clips real commentary. The runaway is bounded by `max_tokens`
  (planner 2048 / narrator 4096) instead.
- **`play_as_black` run-order confound is unresolved** (2026-07-26): the
  recorded mid-suite dip did not reproduce, but the isolated arm shared the
  GPU with a second pytest — re-run cleanly before trusting either number.
- **Honesty-guard false-positive sweep** (2026-07-25): `unverified_claims`
  over the 46 recorded live turns guards exactly the two known lies (2/46).
  Re-run the sweep when a fresh trace corpus exists.
- **The answer-shape misses do reproduce** (2026-09-04): "what's the
  position?" reached `evaluate_position` 5/5, "bishop to a1" was asked "which
  bishop" 5/5 and "take the pawn" "which pawn" 5/5 on the pre-fix build
  (scratch runs on main, 5 samples each), and all three went 5/5 the other way
  on the fix. Unlike the verbosity scenario below, these are reproductions and
  not only regression locks.
  The difficulty-constraint miss ("go easy on me without changing the
  difficulty") does *not* reproduce fresh — 29/30 respected across two builds —
  and does reproduce in the thread it happened in (12/20; the table above).
  Same lesson as the long-transcript family: the condition, not the words.
- **The harness does not reproduce the verbosity miss** (2026-09-04). Live,
  "talk more" was answered in prose twice and `set_verbosity` was never
  called. `verbosity_up_from_low` scores **5/5 on the pre-fix build as well as
  the fixed one**, and so does a long-transcript variant at all three
  conditions (fresh / live_like / poisoned, measured and then dropped rather
  than kept as three green scenarios that measure nothing). Same shape as
  `resign_never_pretends`: length is not the condition, and self-poisoning is
  not either. So the scenario is a regression lock, not a reproduction, and
  what actually catches the failure at runtime is the guard's
  `verbosity_change` class — a narrated change over a turn that called no
  setter is suppressed.
- **A guard false positive is measurable** (2026-09-04): run
  `advice_capture_survives_guard` against the pre-#250 guard and it reproduces
  the live suppression verbatim — 4/5, the failing sample "Take the knight on
  e4. It's the cleanest move, bro." replaced by the canned correction. It
  reproduces at *4/5*, which the floor still passes, so the hard spec for a
  guard misfire stays the unit tests in `test_honesty.py`; the scenario is
  there to price the misfire in live turns, which is the number the sweep
  above cannot give.
- **A multi-call confirmation turn has no knowable phase split** (2026-09-05).
  `evalstats` used to call every `confirmation`-route call narrator time, which
  was true before the free-text reader existed. Now such a turn can be
  reader-only (verbosity `low`, or a cancel), reader *and* narrator (a
  free-text confirm at normal), or narrator-only (a literal "yes" at normal),
  and nothing on the trace record tells them apart. One call is still attributed
  whole — it is that call's time either way — and more than one now reports
  `phases=UNKNOWN` rather than counting the reader's round trip as narration.
  Do not read a narrator median that includes confirmations from before this
  change. A real reader phase would need production to record which calls were
  the reader's, which is not a change the harness gets to make.
