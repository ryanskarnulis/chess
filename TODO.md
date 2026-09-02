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

- [ ] **Phase-4 manual walkthrough** (checklist below) — never done
      systematically; the UI and turn architecture have both been rebuilt
      since anything was last exercised by hand. Trace it
      (`CHESSAPP_TRACE_PATH`) to double as the fresh trace baseline.
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

- [ ] Pending-proposal state for confirmations — only if the bare-"yes"
      dead-end recurs in live games.
- [ ] Merge `set_verbosity`/`set_voice_output` into one `set_option` — only
      if token pressure ever shows; decide with evals.
- [ ] GBNF grammar-constrained decoding — only if tool-call reliability
      degrades; the live spike showed native tool calls work.
- [ ] Branch protection + native auto-merge — needs a public repo or Pro.
- [ ] Physical board (Chessnut Move) — blocked on hardware. Verify motorized
      actuation is programmatically controllable before any design work;
      until then `control_physical_board` stays a tool seam only.

## Phase-4 manual walkthrough

Go through this at the desk, note anything that feels wrong, turn notes into
backlog items. Most items have only ever been exercised by automated tests.

### Setup / stack
- [ ] `docker compose up` from cold works against the voice (`../speech/`)
      and brain (`../llama-swap/`) stacks
- [ ] Reachable from another device on the home network
- [ ] Basic gameplay fully offline (network unplugged)
- [ ] Full game vs Stockfish with the agent off (`CHESSAPP_AGENT=off`)

### Core gameplay (board UI)
- [ ] Complete game to checkmate; board renders correctly throughout
- [ ] Illegal drags rejected without corrupting state
- [ ] Castling (both sides), en passant, promotion (incl. underpromotion)
- [ ] Check / checkmate / stalemate / draws detected and surfaced
- [ ] Captured pieces update correctly
- [ ] New game: random side, board flips as black, "Switch to …" until the
      first move, undo takes back the full exchange
- [ ] The stacked layout reads well on a desktop monitor too

### Text agent
- [ ] Free-form moves land as intended ("knight to f3", "Nf3", "castle")
- [ ] Ambiguous input gets a clarifying question, not a guess
- [ ] Illegal requests get a clear rejection, board unchanged
- [ ] `undo` / `resign` / `new_game` by natural language
- [ ] Save mid-game, restart app, resume
- [ ] Exported PGN loads in an external viewer
- [ ] Reads: "what's the position?", "legal moves?", "history?", "captures?"
- [ ] Reaction latency feels acceptable
- [ ] Long game: tool reliability holds late

### Analysis & hints
- [ ] "What was my mistake?" gives a sensible answer
- [ ] A hint arrives when asked ("what should I play?") and is the engine's
      suggestion; nothing volunteered unasked (hints mode retired 2026-09-01)
- [ ] "Who's winning?" gives a coherent eval
- [ ] Post-game review: classifications and accuracy plausible; survives
      dismiss/reopen; "Review unavailable" (not a crash) with engine off;
      Copy PGN works
- [ ] Analysis and whole-game review latency tolerable

### Personality & settings
- [ ] Glitch's tone comes through (tone only — never strength or settings)
- [ ] Difficulty changes are real and survive restart; UI matches the engine
- [ ] Verbosity and voice-output switchable mid-game by speech and persist
- [ ] Verbosity levels actually differ

### Voice
- [ ] Spoken moves transcribe and execute (noisy vs quiet room)
- [ ] Hands-free mode on a real phone — iOS verified 2026-07-11; Android
      Chrome still unverified; feel checks on both (VAD endpointing,
      half-duplex, autoplay unlock, exit tap)
- [ ] STT latency acceptable in the play loop
- [ ] TTS intelligible, right voice, reasonable latency
- [ ] Voice output on/off actually mutes/unmutes
- [ ] Misrecognized speech fails safe (question, not a wrong move)
