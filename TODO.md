# TODO

The backlog, in priority order. One task = one branch = one PR. When a task is
merged, move its line to `DONE.md` with the date. Re-plan freely.

## Standing constraints

- `long_capture` stays green (release-blocking eval). Any prompt or schema
  change that sends it red does not merge.
- Don't re-attempt tool-schema minimization on gemma-4-12b — every stripped
  pydantic key collapsed `undo_and_replace` (full record: DONE.md 2026-07-21).
  Gate on that scenario if the brain model ever changes.

## Next

- [ ] **Guard: advice captures read as done captures** (walkthrough #1):
      "Take the rook. Rxd1 is the move." is suppressed because bare `take`
      is a done-verb in `honesty.py` and `_CAPTURE_HEDGES` has no imperative
      hedge. Every hint whose best move is a capture dies. Hedge advice
      phrasing and/or verify against the suggested move when the turn
      called `get_best_moves`; regression eval on the traced strings.
- [ ] **Board click hit-testing offset half a square up** (walkthrough #2,
      Firefox desktop 100%): top half of a square selects the piece below.
      Reproduce in Playwright with `elementFromPoint`; check Chromium; likely
      the measured rect vs rendered squares after #240/#241/#243.
- [ ] **"talk more" never sets verbosity** (walkthrough #3): model claims the
      change, `set_verbosity` not called, guard misses the claim. Fast path
      for verbosity phrases; guard pattern for "more of the breakdown".
- [ ] **Container ignores SIGTERM** (walkthrough #4): `compose stop` runs the
      full 10 s grace then SIGKILL; process outlives uvicorn shutdown (engine
      child?); health stays green while the API is down.
- [ ] **Mistake narration names the wrong captured piece** (walkthrough #5):
      "taken that queen with Qxe2" for a pawn capture. Put the captured piece
      in `analyze_last_move`'s result; guard piece names in capture claims.
- [ ] **Free-text answers to a pending confirmation** (walkthrough #6): "just
      do it" after a resign question does nothing. Unparsed answers go to the
      model to decide confirm/cancel through the same confirm path.
      Supersedes the "pending-proposal state" someday item.
- [ ] **Glitch difficulty** (PCC #332): a tier where the LLM picks its own
      moves instead of Stockfish. Design note first — it amends BRIEF's
      "Stockfish is the move source" stance; the coordinator still owns the
      reply beat and validates against legal moves, with a deterministic
      fallback for an invalid pick, and the reply becomes a model call
      (latency + GPU serialization to think through).
- [ ] **Claim-draw button** (#220 follow-up): `/api/game/claim-draw` through
      the registry (same gate as resign), a `_CONFIRM_QUESTIONS` entry, and
      the frontend button. Until then claims are unreachable in direct mode.
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
- [ ] A question turn shouldn't mutate settings the player owns — fold that
      assertion into the next eval-touching slice (residue of the retired
      flips-hints-unasked item; the flip itself died with hints mode,
      2026-09-01).

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

- [ ] "what's the position?" answers with an eval, not a description
- [ ] PGN by chat: fill the `?` headers, add a copy affordance
- [ ] "without changing the difficulty" was overridden by a difficulty change
- [ ] Illegal-for-every-piece requests answered as ambiguous, not illegal
- [ ] Illegal drags snap back silently — decide if that wants a message
- [ ] Desktop stacked column: too much dead space
- [ ] "Switch To White" Title Case vs sentence-case aria label
- [ ] Not exercised: 40+ move game late-tool reliability; exported PGN in an
      external viewer; Android Chrome hands-free
