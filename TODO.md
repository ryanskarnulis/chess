# TODO

The backlog, in priority order. One task = one branch = one PR. When a task is
merged, move its line to `DONE.md` with the date. Re-plan freely.

## Standing constraints

- `long_capture` stays green (release-blocking eval). Any prompt or schema
  change that sends it red does not merge.
- Don't re-attempt tool-schema minimization on gemma-4-12b — every stripped
  pydantic key collapsed `undo_and_replace` (full record: DONE.md 2026-07-21).
  Gate on that scenario if the brain model ever changes.

## Open

Nothing open (2026-09-17). Findings recorded, not filed: a bare "castle" with both
castlings legal is played as O-O (`docs/knight-ask-campaign.md`, "Measurement
record") — a design decision for whoever picks it up.

- [ ] The move guard reads a piece-placement list as moves (found 2026-09-22 in
  #289's gate, pre-existing): asked to show the position, the narrator
  sometimes lists placement in piece-letter notation ("White: Ke1, Qd1, Ra1 …")
  and the move class cuts `Ke1` as an unplayable move — 1/20 on both `main`
  and the #289 tree in `freeform_confirmation_answers[unrelated]`, interleaved.
  A correct answer the guard fires on, so the guard loosens (a token naming a
  piece on the square it occupies is placement, not a move), measured on a
  corpus first.
- [ ] #289 PR 2: typed clarifications (`ask_player`), measured separately.

## Agent audit follow-ups (2026-09-05)

Record: `docs/agent-audit-2026-09-05.md` (external audit of the loop, tools
and eval coverage; all ten numbered findings reproduced on `main@3038c4e`
with the probe scripts in its appendices). Decisions taken 2026-09-05: a
restored engine-to-move position is settled by the coordinator; a question
that names two or more legal moves is a clarification, not advice; eval
coverage completeness outranks GPU budget, so every proposed scenario lands.
All ten findings' follow-ups landed 2026-09-05 (#262–#272, DONE.md): one PR
each, tool-boundary tests with every change, a gate run after every loop or
prompt change, and the baseline re-recorded whenever the harness changed.
Finding 3 closed as a design note (#271, agreed the same day), its
implementation (#272), and the live run the note asked for (2026-09-05,
DONE.md): from a fresh Claude Code 2.1.261 session the gated `new_game` put
the app's own question on screen as an Accept/Decline form, the yes ran the
op and the no left the game untouched. Nothing is open here.

## Next

- [ ] **Glitch difficulty** (PCC #332): a tier where the LLM picks its own
      moves instead of Stockfish. Design note first — it amends BRIEF's
      "Stockfish is the move source" stance; the coordinator still owns the
      reply beat and validates against legal moves, with a deterministic
      fallback for an invalid pick, and the reply becomes a model call
      (latency + GPU serialization to think through).
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
