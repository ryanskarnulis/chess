# TODO

The backlog, in priority order. One task = one branch = one PR. When a task is
merged, move its line to `DONE.md` with the date. Re-plan freely.

## Standing constraints

- `long_capture` stays green (release-blocking eval). Any prompt or schema
  change that sends it red does not merge.
- Don't re-attempt tool-schema minimization on gemma-4-12b — every stripped
  pydantic key collapsed `undo_and_replace` (full record: DONE.md 2026-07-21).
  Gate on that scenario if the brain model ever changes.

## Agent audit follow-ups (2026-09-05)

Record: `docs/agent-audit-2026-09-05.md` (external audit of the loop, tools
and eval coverage; all ten numbered findings reproduced on `main@3038c4e`
with the probe scripts in its appendices). Decisions taken 2026-09-05: a
restored engine-to-move position is settled by the coordinator; a question
that names two or more legal moves is a clarification, not advice; eval
coverage completeness outranks GPU budget, so every proposed scenario lands.
Work in this order — one PR each, tool-boundary tests with every change, a
gate run after every loop or prompt change, and the baseline re-recorded
whenever the harness itself changes.

- [ ] **PR 5 — Eval coverage: compositions and loop paths** (all twelve
      proposed scenarios, audit §"Proposed live scenarios"). Strengthen
      `undo_twice_and_replace` (length 4, player to move, completed, 3–5
      calls); `ambiguous_knight_then_selection`;
      `move_save_resume_finishes_exchange`; `save_then_new_game`;
      `voice_setting_and_move`; `move_and_judgment`; `resume_and_describe`;
      `best_move_then_play`; `resign_intent_reaches_planner`;
      `freeform_confirmation_answers` (cancel / confirm / unrelated);
      `late_game_tool_composition` on `tests/late_game_84_plies.pgn` (84
      plies, move 43, transcript seeded from the replay, paired with a
      small-history control); `stt_knight_repair` ("put my night on f three").
      Every one asserts route, stop reason, call count, board end-state and
      settings snapshot. Record which reproduce a miss on the pre-fix build
      and which are locks. Claimed description effects get 20 samples an arm.
- [ ] **PR 6 — `undo` plies description** — the item under Next; run it
      after PR 5 so its scenario is the strengthened one.
- [ ] **Later — MCP confirmation surface** (finding 3): standalone MCP
      exposes the gate but nothing can confirm it, so `new_game`/`resign`/
      `claim_draw` are unreachable there. Needs a trusted human confirmation
      path, never a model-callable bypass; define it before adding anything.
- [ ] **Later — pending-op policy**: two refusals in one batch arm two ops
      and the last replaces the first, while the narrated question may be
      about the first. Decide (first stays / last wins / second refused) and
      bind the question to the op the next yes runs. Also `_BASE` still tells
      the tool-free narrator it changes the game through tools — stale role
      text, measure before cutting.
- [ ] **Later — schema snapshot**: the golden normalizes away `title`,
      `default` and nullable unions, the very keys the standing prohibition
      protects; add an exact emitted-schema snapshot beside it.
- [ ] **Later — "move the rook" is sometimes played, not asked** (measured
      2026-09-05 by the honest harness, PR 1): with four rook moves on the
      board the planner submits one (`Rh3`) instead of asking — 3 samples in
      37 across the day (2/17 on the pre-fix builds, 1/20 on the guard fix,
      where the guard's own share of the misses is gone and the scenario runs
      unmarked at 19/20). Model understanding — the planner's one/several/none
      matching procedure is the lever, and a change to it is a prompt change
      gated on `ambiguous_move` at 20 samples an arm.

## Next

- [ ] **Glitch difficulty** (PCC #332): a tier where the LLM picks its own
      moves instead of Stockfish. Design note first — it amends BRIEF's
      "Stockfish is the move source" stance; the coordinator still owns the
      reply beat and validates against legal moves, with a deterministic
      fallback for an invalid pick, and the reply becomes a model call
      (latency + GPU serialization to think through).
- [ ] **Claim-draw button** (#220 follow-up): `/api/game/claim-draw` through
      the registry (same gate as resign) and the frontend button. The
      `_CONFIRM_QUESTIONS` entry landed with #255. Until then claims are
      unreachable in direct mode.
- [ ] **A named move is a whole takeback, not a plies count** (found
      measuring #260): "undo the bishop move and undo the knight move, then
      play d4" gets `undo(plies=2)` — one exchange, not two — or
      `undo(plies=1)` — the engine's reply alone — three times in four, and
      the replacement then lands on a board that still holds the second move;
      `undo_and_replace`'s residual misses are the same `plies=1` for one
      named move. Model understanding, so the lever is `undo`'s `plies`
      description (half-moves, which a player's move plus the reply is two
      of; a move named by piece is a full takeback; several named moves are
      several calls). Eval-gated on `undo_twice_and_replace`, a non-strict
      xfail until then — remove the marker in the same PR.
- [ ] **Re-vendor the voice module downstream** (#232 follow-up): PCC's and
      conductor's `MicButton.tsx` copies predate the #232 wedge fix and the
      #242 inline icons. Their host api clients also need the
      transport-to-null guard (app code the vendored file can't carry).

## Evals / observability

- [ ] Re-run `play_as_black` alone (20 samples, idle GPU, then again placed
      after the long-transcript block) — the 2026-07-26 order-confound arm
      wasn't clean; compare the report lines.
- [ ] Narrator wordiness cap: probably drop. The runaway is already bounded
      by `max_tokens`; the token study showed legit and runaway narrations
      overlap almost entirely, so any cap either misses the tail or clips
      real commentary.

## Someday / blocked

- [ ] Merge `set_verbosity`/`set_voice_output` into one `set_option` — only
      if token pressure ever shows; decide with evals.
- [ ] GBNF grammar-constrained decoding — only if tool-call reliability
      degrades; the live spike showed native tool calls work.
- [ ] Branch protection + native auto-merge — needs a public repo or Pro.
- [ ] Physical board (Chessnut Move) — blocked on hardware. Verify motorized
      actuation is programmatically controllable before any design work;
      until then `control_physical_board` stays a tool seam only.

## Walkthrough leftovers (2026-09-04)

Record: `docs/qa-walkthrough-2026-09-04.md`. Small items, unprioritized:

- [ ] Desktop stacked column: too much dead space — proposed answer: a
      two-column layout on wide screens (board left, agent column right);
      decided 2026-09-04 to take separately from the bubble clamp
- [ ] Not exercised: 40+ move game late-tool reliability; exported PGN in an
      external viewer; Android Chrome hands-free
