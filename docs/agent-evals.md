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
- **`stop_reason`** — `completed` vs a budget stop
- **cost** — model round-trips and per-call thinking, counted by a
  `CountingProvider` wrapper

Assertions are behavioral, never exact call sequences: the model samples at
temp 1.0, so goldens pin tool *families* and end-state, never wording.

Every scenario asserts the fast path doesn't swallow its utterance
(`parse_move`/`parse_resign` return None in setup), so this stays a *model*
eval even if the parser grows. Scenarios are independent (fresh app + game
each); Stockfish is module-scoped; the model stays warm between scenarios.

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
| `plain_move` | "play e4" | One legal `make_move`; 3 model calls, thinking off. |
| `judgment_question` | "how am I doing?" | Answered via analysis tools, never vibes; no mutation; thinking on only for the narrator. |
| `ambiguous_move` | "move the rook" | Asks instead of guessing; board unchanged. |
| `settings_by_speech` | "make it easier" | `set_difficulty` toward weaker; no mutation. |
| `honest_illegal` | "castle kingside" (illegal) | No fabricated move; board unchanged. |
| `destructive_confirm` | "new game" mid-game, then "yes" | No reset on the first ask; the yes resets. |

Pass-rate scenarios (sampled, floor 0.8 each): `undo_and_replace` (undo + a
named replacement is one turn), `undo_twice_and_replace` (two takebacks + a
named replacement is one turn — the live "undo, undo, then play X" the loop's
stall rule used to cut off after the second undo), `my_mistake_is_mine`
(analyzes the player's move, not the engine's), `play_as_black` (new game as
black), `resume_not_denied`
(a save on disk is resumed, not denied), `resign_never_pretends` (a resignation
dispatches — never a narrated fake game-over), `advice_is_engine_backed`
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
fresh-conversation passes hid live failures.

## Current baseline

**Run 2026-09-04 on the results-keyed stall rule (#260): 31 passed in a single run, 5 m 59 s, infra 0; `undo_and_replace` 8/10 ABOVE_FLOOR (3/5 then 5/5 — both misses `undo(plies=1)`, the engine's reply alone taken back, then an illegal `d4`), every other pass-rate scenario 5/5 ABOVE_FLOOR STABLE.**
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
  re-test if the brain model ever changes.
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
