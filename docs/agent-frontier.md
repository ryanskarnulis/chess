# The frontier tier

Live-model scenarios that are **meant to be hard**, measured over time and
never gated (#318). The gate (`docs/agent-evals.md`) holds floors that are
"regression tripwires, not aspirations"; this is where the aspirations go.

| Layer | Where | Must pass? | What it is for |
| --- | --- | --- | --- |
| Trajectory walks (`tests/test_trajectories.py`) | CI | yes | the app's own safety when features combine |
| The gate (`tests/test_agent_evals.py`) | by hand, before a prompt/model/loop merge | yes, floor 0.8 | regressions |
| **The frontier** (`tests/test_agent_frontier.py`) | by hand, on demand | **no — a score** | progress on hard, composed tasks |

## Running it

```bash
cd backend
CHESSAPP_AGENT_FRONTIER=1 CHESSAPP_EVAL_REPORT=/tmp/frontier.jsonl \
    .venv/bin/pytest tests/test_agent_frontier.py -v -s
```

| Env var | Default | What it does |
| --- | --- | --- |
| `CHESSAPP_FRONTIER_RUNS` | 10 | Samples per scenario (fixed; no escalation). |
| `CHESSAPP_FRONTIER_SPLIT` | `dev` | `dev` or `heldout` variants. |
| `CHESSAPP_EVAL_REPORT` | — | JSONL report path (shared with the gate). |

The gate's `LLAMACPP_*` and `CHESSAPP_STOCKFISH` apply. Check the shared card
is idle first, as for any eval run.

## What a run reports

Per scenario: whole-task passes out of N with a one-sided 95% Wilson
interval, the **rubric score** (the mean fraction of checkpoints met — the
number that moves while whole-task passes are still near zero), each
checkpoint's hit count, the normalised failure modes, infra deaths, and every
sample's turns (route, stop reason, tools, history). The report opens with a
`frontier_header` naming the git sha, model, planner temperature, split, and
the shas of the planner prompt, the narrator prompt and the tool offer, so a
number is always tied to what produced it.

An item **fails** only when the number cannot be trusted:

- a deterministic **trajectory invariant** broke on a live sample
  (`trajectory.INVARIANTS`, run step by step) — a bug in the app, however low
  the model's score may be;
- a **broken scenario** — a checkpoint that raised, or a step marked `model`
  that the pipeline answered without the planner (a parser grew to swallow it,
  and the scenario would otherwise measure the parser forever);
- **no sample measured** — every one an infrastructure death.

A low score is never a failure.

## Writing a scenario

`tests/frontier_corpus.py`. A `Scenario` has a tier, a `why`, `dev` and
`heldout` variants (each a sequence of `Say`s plus an optional setup), and
`Checkpoint`s — named predicates over the `Episode` (the board before and
after each turn, the tool results, the route and stop reason). The rules:

- **Grade on the board and the tools, never on wording.** No checkpoint reads
  the commentary.
- **A checkpoint must be unambiguous about what the player asked.** The first
  live run graded "so how do things stand now?" as a judgment question; the
  model answered it with `describe_position` every time, which is a fair
  reading, so the miss was the scenario's. The wording was fixed, not the
  model.
- **Held-out wordings are never tuned against.** Iterate on `dev`; believe an
  improvement only when `heldout` moves too. `test_frontier.py` refuses a
  wording shared between the splits, and any `model` step a parser would
  settle on the opening board or after 1.e4 e5.
- **Tiers:** 1 stretch (several intents in one utterance), 2 multi-turn (later
  turns lean on earlier ones), 3 frontier (long sessions and long games,
  expected near zero).

## Status

The harness shipped with two starter scenarios, one in each of tiers 1 and 2,
to exercise it end to end. Both are too easy to be frontier — a first live
smoke run (3 samples each, dev, `ade22ed`) read `undo_replace_and_judge` 2/3
(rubric 0.93, the miss a verdict skipped) and `knight_ask_then_change_of_mind`
3/3 after the wording fix. The hard corpus, with its first recorded baseline,
is #318's PR 4; the run history and graduation rules are PR 5.
