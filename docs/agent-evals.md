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
named replacement is one turn), `my_mistake_is_mine` (analyzes the player's
move, not the engine's), `play_as_black` (new game as black), `resume_not_denied`
(a save on disk is resumed, not denied), `resign_never_pretends` (a resignation
dispatches — never a narrated fake game-over), `advice_is_engine_backed`
("what should I play here?" must consult `get_best_moves`, mutate nothing, and
name only tool-reported moves), and the long-transcript family `long_resume` /
`long_resign` / `long_capture`, each at three conditions (fresh, live_like,
poisoned) — the same behaviors under a real 20-turn conversation, because
fresh-conversation passes hid live failures.

## Current baseline

**Run 2026-09-01 on the hints-retirement build (#245): 23 passed in a single
run, 3 m 18 s, every pass-rate scenario 5/5 ABOVE_FLOOR STABLE, infra 0.**
`long_capture` 5/5 ×3 (release blocker), `undo_and_replace` 5/5 (schema
tripwire), costs unmoved (`fast_path_low` 0 model calls, `fast_path_normal` 1,
`plain_move` 3). First baseline for `advice_is_engine_backed` (successor to
`hints_off_no_advice`, retired with hints mode): 5/5, all samples the textbook
`trajectory=[get_best_moves(n=3)]` turn.

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
