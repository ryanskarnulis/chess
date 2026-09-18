# Knight-ask clarification campaign (#286)

Plan written 2026-09-17 on `feat/286-knight-ask-campaign` from `main@51861ef`.
The measurement record goes into `docs/agent-evals.md` as it lands; this file
is the plan and the decision log for the campaign.

## What is being measured

"move my kings knight" on a fresh board must be asked about (Nf3 or Nh3),
never played. The scenario `ambiguous_knight_then_selection` has read:

| date | tree | count | note |
| --- | --- | --- | --- |
| 2026-09-05 | old text / trimmed text | 8/20 / 17/20 | alternating blocks, one server |
| 2026-09-05, -06 | gates | 5/5, 5/5, 8/10 | |
| 2026-09-10 | guard-rewrite tree / unchanged main | 13/20 / 11/20 | identical planner inputs |
| 2026-09-17 | #283 tree / #282 tree | 4/5 / 5/5 | |

Both 2026-09-10 trees had the same planner prompt, schema and state view, so
the ~60% is attributed to the serving session. Recorded, not yet proven: the
unchanged old text alone read 5/20, 19/40 and 31/40 in three separate batches
(`personality.py` comment), swings a binomial cannot produce. The suspected
mechanism is slot-KV prefix reuse under q8_0 KV quantization (the eval suite
opens a fresh conversation per sample but every planner call shares the same
long prefix, so the server restores it from the live slot rather than
prefilling), which would make the rate depend on how long the server has been
up and what it served last. Nothing has tested that directly.

## Phase 0: tooling (PR 1, no behavior change, no gate)

The 2026-09-05 campaign ran on scratchpad scripts that were never committed
(`probe_planner.py`, `campaign.sh` in the memory notes). This campaign commits
them so every later measurement is repeatable and carries its own identity.

1. `backend/scripts/probe_planner.py`, the direct planner probe.
   - Builds the app's own planner input (`api._agent_state_dict` on a
     `GameSession`, the real tool offer via `registry.definitions(exclude)`,
     the planner prompt) and calls `LlamaCppProvider.chat` once per sample,
     thinking off, the app's `max_tokens` cap.
   - Arms are named variants over five knobs: prompt text, planner
     temperature, `cache_prompt`, tool offer, model id. Arms run
     **round-robin per sample**, never in separate batches.
   - Each sample is classified deterministically from the wire result:
     `tool:<name>` per call made, or `no_tool`. For the knight ask `no_tool`
     is the pass. No language is parsed.
   - Each sample records: git HEAD, sha of `PLANNER_PROMPT`, sha of the tool
     definitions offered, model id, temperature, `cache_prompt`, llama-swap
     `/running` (model + cmd hash), the session load timestamp, and the
     request ordinal since that load. `--fresh` hits llama-swap `/unload`
     before the first sample and records the reload.
   - Pre-flight: refuses to start while `/upstream/<model>/slots` shows a slot
     processing or `nvidia-smi` shows another job, and says so.
   - Output: JSONL per sample plus a summary table per arm with raw counts
     and a lag-1 agreement statistic per arm (the number that says whether
     consecutive samples cluster).
   - Corpus: a small list of utterances with the position each is asked on.
     Ambiguous: knight ask, "move the rook" (four rook moves), a two-bishop
     ask, "castle" when both sides are legal. Must-not-regress neighbours:
     "take the pawn" (nothing to take), "bishop to a1" on move 1, the STT
     knight ("please put my night on f three"), "castle" with one side
     legal, the first call of "take that bishop move back and play d4
     instead". Two of the ambiguous items are held out of arm design and
     used only for the final confirmation.
2. `backend/scripts/eval_campaign.sh`, the harness-level confirm, and
   `backend/scripts/campaign_report.py`, which joins its block reports.
   - Alternating blocks of five between two trees on one server
     (`PYTHONPATH` switches the `chessapp` under test; the campaign
     worktree's `tests/` serves both arms), `-k` for the scenario list,
     `CHESSAPP_EVAL_RUNS=5 CHESSAPP_EVAL_MAX_RUNS=5`, `CHESSAPP_EVAL_REPORT`
     per block. Logs HEAD, `personality.py` hash, `chessapp.__file__` and the
     server identity per block. `--fresh-per-block` unloads before each
     block. Aggregates the block reports into one table.
3. Off-GPU tests for the classifier, the rules, the schedule, the statistics,
   the pre-flight decision, the corpus premises and the request shape
   (`tests/test_probe_planner.py`), in the style of `test_evalstats.py`; the
   provider gains an optional per-request `cache_prompt` (omitted → the same
   bytes as ever) for arm A, pinned in `test_provider.py`.
4. `docs/agent-evals.md` gains a short "Measuring a planner change" section
   pointing at the two scripts and restating the interleaving rule.

## Phase 1: re-baseline on fresh and warm servers (measurement only)

No code change. Unchanged `main`, current prompt.

- Probe, knight ask only, four blocks of 20 on one day: F1 (unload, then 20),
  W1 (the same session continues, 20 more), F2 (unload, 20), W2 (20). Session
  identity and request ordinal on every sample. Repeat on a second day if
  the GPU allows.
- Harness confirm: `ambiguous_knight_then_selection` at 20 samples fresh and
  20 warm, blocks of five, the existing floor.
- Report raw counts per block, per session, in `docs/agent-evals.md` under
  "Current baseline", beside the recorded 26/40, 3/40, 13/20 and 11/20.

Decision after Phase 1:

- Fresh high (≥17/20) and warm low: serving state is the cause. Phase 2
  starts with arm A.
- Fresh and warm both near 60%: the prompt or the model. Phase 2 starts with
  arms B and C, then D.
- Fresh and warm both ≥17/20: 2026-09-10 was an off day of the server. The
  re-baseline is recorded, the `TODO.md` item moves to `DONE.md`, and
  whether arm D still ships as hardening is a decision for Ryan.

## Phase 2: one variable at a time

Probe first (40 per arm, interleaved with the unchanged control), then the
harness confirm (alternating blocks, 20 an arm) on the scenarios the change
can touch, then the full gate. Arms in cost order, each against the control
alone, never stacked until one has won on its own.

- **A. `cache_prompt: false` on planner calls.** A per-request llama-server
  field the provider does not send today (`provider._payload`), so the slot
  prefix is prefilled fresh each call. Planner phase only. Tests the
  mechanism directly. Cost is the extra prefill (~3k tokens a call); the
  latency delta is read off `plain_move` and `judgment_question`.
- **B. Planner temperature 0.3.** The knob exists
  (`CHESSAPP_PLANNER_TEMPERATURE`) and has no recorded measurement. Risk: a
  cooler planner makes a wrong choice consistently rather than half the
  time, so `stt_knight_repair` and `undo_twice_and_replace` are read beside
  it.
- **C. Narrow prompt trims.** The 2026-09-05 rule holds: every arm that added
  words about playing made the ask worse, and the deletion did the work. At
  most three arms a screen, all trims:
  - `make_move`'s description ("Map loose phrasing to the matching entry
    ... never invent a string") teaches mapping a vague phrase onto one
    move with no ambiguity branch, and it sits in the tool offer on every
    call. Arm: drop the worked example. Tool descriptions are prompt text,
    not pydantic keys, but `undo_and_replace` gates any tool-text change
    (`TODO.md` standing constraint).
  - The planner bullet's own example clause ("grab that pawn").
  - Nothing added. An arm that needs a new sentence is out.
- **D. Structured clarification handoff.** A planner tool
  `ask_which_move(candidates: list[str])`. The model decides that the words
  fit several moves and names them; code checks the candidates against the
  board: every candidate must be in `legal_moves` and there must be at least
  two distinct ones. One candidate refuses with `retry: different_args`
  ("one move fits: submit it with make_move"); an illegal candidate refuses
  the same way. The result carries the candidates and mutates nothing. It
  reaches the narrator as an ordinary tool result, so the question is
  Glitch's to phrase, and the candidates are the tool's own report, which
  the advice guard licenses (the 2026-09-05 miss where a correct
  clarification phrased as a statement was cut). Offered to the planner
  only; excluded from the MCP server, which has elicitation for this.
  Consequences to carry: the schema golden changes
  (`tool_definitions_golden.json`), the scenario's step-one assertion of two
  model calls becomes three (planner call, note, narrator), and a tool-list
  change has collapsed `undo_and_replace` before, so the confirm includes it.
  Risk to measure: over-asking on single-fit asks (`plain_move`,
  `stt_knight_repair`, `long_capture`).
- **E. Another model behind the seams — split out to #298 (2026-09-17).**
  The most expensive arm and the only one that re-gates the whole suite; it
  runs only if A through D do not reach the target, under its own issue. The
  probe already takes it (`--arm qwen:model=<id>`, interleaved per block since
  the card is exclusive).

Target before anything ships: the knight ask at ≥17/20 on both a fresh and a
warm session, the held-out ambiguous items asked, the neighbours where the
2026-09-05 screen left them (refusals 30/30, rook ask 20/20, STT knight 20/20,
castle 20/20, undo's first call `undo` with `plies` omitted), then the harness
confirm on `ambiguous_knight_then_selection`, `ambiguous_move`,
`stt_knight_repair`, `undo_and_replace`, `undo_twice_and_replace`,
`long_capture` ×3, both `impossible_*` scenarios and `plain_move`, then the
full gate with `long_capture` 5/5 ×3.

## Phase 3: ship

One PR per shipped variable. Each PR re-records "Current baseline" in
`docs/agent-evals.md` with the fresh and warm counts and the session
identities, appends its arm table, moves the `TODO.md` item to `DONE.md`, and
the last one closes #286. `CLAUDE.md` rules that bind here: no phrase lists
and no regex for language; a guard that fires on a correct answer is
loosened, not scripted around; evals stay a manual local command.

## Decisions (taken 2026-09-17)

1. The llama-swap flags (`--cache-type-k/v q8_0`, `--cache-ram 0`) live in the
   shared `../llama-swap/config.yaml`. Arm A is the client-side test of the
   same hypothesis and needs no change there; a server-flag arm (f16 KV) is
   not in this plan unless asked for.
2. **Arm E is its own issue, #298.** This campaign is A through D on
   gemma-4-12b.
3. **No subagents.** The tooling PR and every measurement run in the main
   session (the live model is never a subagent's to call).
4. **The scripts may unload llama-swap without asking** when the pre-flight
   finds no slot processing and no foreign job on the card; otherwise they
   refuse and say why. An unload drops another client's in-flight request,
   which is what the pre-flight is for.

## Estimated GPU time

| step | time |
| --- | --- |
| probe, 40 samples × 2 arms | ~5 min |
| harness confirm, one scenario, 20 an arm | ~10 min |
| full gate | ~13 min |
| Phase 1 in full | ~40 min plus two reloads |
