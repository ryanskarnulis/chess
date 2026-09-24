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

### Recording a run

```bash
CHESSAPP_AGENT_FRONTIER=1 CHESSAPP_FRONTIER_SPLIT=dev \
    CHESSAPP_EVAL_REPORT=/tmp/frontier.jsonl .venv/bin/pytest tests/test_agent_frontier.py -s
CHESSAPP_AGENT_FRONTIER=1 CHESSAPP_FRONTIER_SPLIT=heldout \
    CHESSAPP_EVAL_REPORT=/tmp/frontier.jsonl .venv/bin/pytest tests/test_agent_frontier.py -s
python scripts/frontier_report.py append /tmp/frontier.jsonl --label "after #340"
python scripts/frontier_report.py trend            # both splits, last six runs
```

`append` adds one line per split to `docs/frontier-history.jsonl` (commit it
with the change it measured); `trend` prints a scenario × run table per split.
Run both splits every time: a dev number alone is not believed.

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

## Reading the history

- **A mark means the intervals separated.** `trend` puts ▲/▼ on a cell only
  when its one-sided 95% Wilson interval does not overlap the previous run's.
  At ten samples that takes a big move (5/10 → 8/10 is no mark), and that is
  deliberate: runs on different days sit on differently-warmed servers, and
  consecutive samples of one prompt are correlated (`docs/agent-evals.md`).
  The history shows *trend*; it is not evidence a particular change worked.
- **A claim that a change moved a score is an A/B**, alternating blocks on
  one server:

  ```bash
  scripts/eval_campaign.sh --suite frontier --split heldout \
      --a /path/to/main --b /path/to/branch --k 'noisy_thread' --blocks 4 --runs 5
  ```

  `campaign_report.py` joins the blocks into one arm table, as for the gate.

## Graduation and retirement

- **Graduate** a scenario when its held-out whole-task rate is at least 0.8
  in each of the two most recent held-out runs (`trend` marks it `yes` in the
  `gate?` column; `frontier_report.GRADUATION_RATE` / `GRADUATION_RUNS`).
  Graduating means writing it as a gate scenario in `test_agent_evals.py`
  from its held-out wording, at the family floor (0.8) in `_FLOORS`, running
  the gate, and removing it from the corpus in the same PR — from then on it
  blocks merges.
- **Harden or retire** a scenario that sits at 10/10 on both splits without
  graduating yet: add a harder variant, or let it graduate. The corpus is
  only worth running while most of it is not solved; baseline v1 had six such
  scenarios, and corpus v2 should replace them with harder ones.
- **Never demote.** A gate scenario that regresses is a regression and fails
  the gate; it does not move here to get quiet.
- **Never lower the bar to graduate.** A scenario whose checkpoint is found
  unfair is fixed and re-measured from scratch, with the fix recorded (as in
  "Scenario fixes made before the baseline" below).

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

## Baseline v1 (2026-09-24)

Corpus v1: 17 scenarios (5 tier-1, 8 tier-2, 4 tier-3), 10 samples each on
each split, gemma-4-12b at planner temperature 0.3 on an idle card, on
`2cdf63d` plus this corpus (planner prompt `d5212278576b`, narrator prompt
`718632b62443`, offer `629dacddf1b4`). 340 samples: **0 invariant breaches,
0 infra deaths**. Whole-task passes, rubric in brackets. The raw per-split
records are the first two lines of `docs/frontier-history.jsonl`.

| Tier | Scenario | dev | held-out | Where it misses |
| --- | --- | --- | --- | --- |
| 1 | `undo_replace_and_judge` | 5/10 (0.90) | 10/10 (1.00) | judged_after_the_move |
| 1 | `settings_move_and_verdict` | 9/10 (0.98) | 9/10 (0.98) | completed_1 |
| 1 | `top_move_play_and_save` | 10/10 (1.00) | 10/10 (1.00) | — |
| 1 | `noisy_takeback_and_replace` | 10/10 (1.00) | 10/10 (1.00) | — |
| 1 | `constraint_keeps_difficulty` | 10/10 (1.00) | 10/10 (1.00) | — |
| 2 | `knight_ask_then_change_of_mind` | 7/10 (0.94) | 10/10 (1.00) | played_d4 |
| 2 | `suggested_move_later` | 10/10 (1.00) | 10/10 (1.00) | — |
| 2 | `save_reset_resume` | 10/10 (1.00) | 10/10 (1.00) | — |
| 2 | `draw_declined_then_advice` | 10/10 (1.00) | 10/10 (1.00) | — |
| 2 | `undo_chain_across_turns` | 7/10 (0.78) | 10/10 (1.00) | exactly_one_more_exchange, played_d4, second_takeback |
| 2 | `difficulty_up_and_back` | 0/10 (0.40) | 0/10 (0.40) | back_where_it_started, harder_still, one_step_not_the_top |
| 2 | `undo_then_ambiguous_bishop` | 10/10 (1.00) | 0/10 (0.75) | played_Bc4 |
| 2 | `noisy_thread` | 0/10 (0.75) | 8/10 (0.95) | played_Nc3 |
| 3 | `long_session` | 7/10 (0.97) | 10/10 (1.00) | played_the_suggestion_6 |
| 3 | `late_game_review_undo_replay` | 0/10 (0.60) | 0/10 (0.58) | back_before_the_worst_move, played_the_best_there, reviewed |
| 3 | `late_game_save_undo_resume` | 1/10 (0.78) | 9/10 (0.97) | undid_three |
| 3 | `second_choice_chain` | 10/10 (1.00) | 9/10 (0.98) | played_the_second |

`difficulty_up_and_back` was re-run after its checkpoints were made explicit
(below); every other row is the first run.

### What it says

- **Wording moves the number more than tier does.** The same task swings up
  to ten samples between its dev and held-out wording, in both directions:
  "the one to c4" 10/10 against "c4 one" 0/10 (the pawn move played), "take
  back my last three moves" 1/10 against "undo my three most recent moves"
  9/10 (three *plies* undone, the known ply misread), "see three" 0/10
  against "sea three" 8/10 (`Ne2`, not `Nc3`). One wording per split measures
  that wording; corpus v2 should carry several per split before a rate is
  read as the task's.
- **Two tasks are consistently beyond the model today, 0/20 each.** A
  relative difficulty ask from the bottom tier ("make the engine harder")
  goes straight to `maximum` every time, which leaves "harder still" nothing
  to do and "back where it started" unremembered. And rewinding a 150-ply game
  to before the player's worst move (138 plies, over the 100-ply cap on one
  undo) is one ordinary takeback, every time.
- **Six scenarios are 10/10 on both splits** (`top_move_play_and_save`,
  `noisy_takeback_and_replace`, `constraint_keeps_difficulty`,
  `suggested_move_later`, `save_reset_resume`, `draw_declined_then_advice`).
  They are gate candidates under the graduation rule, or need harder
  variants to stay frontier.

### The open-question scenarios (#319, 2026-09-24)

Four tier-2 scenarios added with #319, measured as interleaved blocks of five
against main (the record kept but not shown to the planner) on the dev split
only — a comparison, not yet a baseline, and not in the history file:

| Scenario | main | #319 | Where it misses |
| --- | --- | --- | --- |
| `knight_ask_aside_then_pick` | 6/10 | 6/10 | asked_1 (the king-side-knight wording is played, not asked) |
| `knight_ask_long_chat_then_pick` | 10/10 | 10/10 | — |
| `knight_ask_then_board_changes` | 0/10 | 0/10 | moved_nothing_3 |
| `two_threads_similar_asks` | 0/10 | 0/10 | asked_2, played_first_offered |

Only the stale-question and two-thread scenarios measure anything today, as
canaries for moves played on a question that does not stand. The other two
cannot tell the two arms apart: the planner reads an ordinal ("the first
one") as the first entry of `legal_moves` whether or not a question is open,
and the king's-knight ask lists its candidates in that same order (#348), so
the pick lands whatever the planner is shown. A discriminating version needs
an ask whose first candidate is not `legal_moves[0]` (#352); the ordinal
reading itself is #351.

### Scenario fixes made before the baseline

A 1-sample pilot and the first baseline run each found a checkpoint that
scored the scenario, not the model — the rule above, applied:

- `long_session` turn 6, "play that one", followed a reply naming several
  candidates, and the model asked which — correctly. Now "play your top pick".
- `difficulty_up_and_back` scored zero without saying why. It now starts at
  `beginner` (room to climb) and has an explicit `one_step_not_the_top`
  checkpoint, so the rubric names the miss.
