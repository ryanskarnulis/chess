# TODO

The backlog, in priority order. One task = one vertical slice = one branch = one PR (TDD: failing test first). When a task is finished and merged, move its line to `DONE.md` with the merge date. Re-plan freely between slices — this file is the living backlog, not a contract.

## Next sprint (2026-07-13) — agent behavior: see it, then fix it

**The current focus.** Live play does not match the green test suite: moves land wrong or not at all, and it was unclear whether anything beyond the deterministic fast path worked. First real finding, fixed in `fix/agent-determinism-and-tracing`: **the `undo` tool defaulted to one ply**, so an agent-driven takeback popped only the engine's reply and left the player's move on the board with the engine to move — while the board UI's undo button had the correct pairing rule all along.

That is the shape of the whole problem, and it is the *same* shape as the destructive-op gate we closed on 2026-07-13: **the app keeps asking a 12B model to decide things deterministic code already knows.** The suite stayed green because it pins the tool boundary, not live behavior — and the eval harness, for all its 8/8, contains exactly one scenario where the model picks a move at all ("play e4", from the starting position). Move correctness through the model is essentially unmeasured.

So the sprint is: **make agent behavior observable, then iterate against a number.**

1. **[M] Review real traces and grow the eval suite** — with turn tracing landed (`CHESSAPP_TRACE_PATH`, one JSONL row per turn: route taken, full tool trajectory with args and results, stop reason), play real games and read what actually happened. Every turn that misfired is a ready-made scenario: it carries its own `fen_before` and `utterance`. Turn them into a **move-correctness eval** — `(fen, utterance, expected_san)` in real mid-game positions, each verified to miss `parse_move` — asserting the specific SAN that landed. Record a **pass rate over N runs, not a boolean**: the model samples at temp 1.0, so a single assert flaps instead of telling you a path works 70% of the time. That number is what every prompt/loop change below gets judged against.
2. **[S] Measure the fast-path split** — from the same traces: what fraction of real move utterances `parse_move` catches, and how the model path does on the rest. Settles how much of the app's apparent competence is the parser carrying it.
3. **[S] Audit the remaining tools for the same bug class** — anything whose correct behavior is knowable from deterministic state but is currently left to the model's judgment (as `undo`'s ply count was). The four known ones are now **all closed**: `undo`'s ply count, `resign`'s color, `analyze_last_move`'s color, and `new_game`'s side. Anything left? Go through the registry deliberately rather than waiting for the next trace to find one.
4. **[M] Multi-tool turns: "undo that and play this instead"** — a single utterance asking for two actions. The loop already *allows* it (it dispatches every call in a model turn, and can chain across turns within `max_iterations=4`), so this may be a prompt/eval gap rather than a code one — **confirm which before writing code.** Needs an eval scenario asserting both tools ran, in order, with the right end position.
5. **[S] The eval gate is flaky at the floor** — `long_capture[poisoned]` measured 3/5 then 4/5 on the analyze-my-move branch, and **5/5 then 3/5 on `main`** at the same commit: a true rate sitting right on the 80% floor, so the gate goes red or green on an unchanged path roughly by coin flip. The retired capture family had the same signature (0/5, 2/5, 3/5, 5/5 across four runs of one build). 5 runs cannot resolve a 60–100% band — raise `_PASS_RATE_RUNS` (cost: wall time) or move the floor (cost: a weaker tripwire), but decide it deliberately and *with a measured rate over ~20 runs*, not by tuning until green. A gate nobody trusts gets ignored.
6. **[S] Hints off still hands over a move** (`hints_off_no_advice` 0/5, xfailed — the suite's last remaining xfail) — asked "what should I play here?" with hints **off**, the model calls `get_best_moves` and names a move anyway. This is *not* the tool-signature bug class the last two slices closed: the tool can express the ask fine, it's the **policy** that's left to the model to honor — the same shape as the confirm-before-`new_game` rule before the gate replaced it, and a 12B honors a prompt rule about half the time. So give it the same treatment: when hints are off, the code refuses the advice, rather than the prompt asking the model not to give it. (Where: `get_best_moves` should read `ctx.settings` and refuse, exactly as `_gate` does.)
7. **[M] Show tool calls live in the UI, like "thinking…"** — the player should see the agent's tools as it calls them (`undo` → `make_move`), not just the commentary after. Needs the tool trajectory streamed or polled during a turn rather than returned only in the final `/api/command` response; the trace record is already the right shape for it.

**[L] The Phase-4 manual walkthrough** stays queued behind this — the walkthrough list below is still the plan, but debugging the agent is what real play keeps running into first. Worth exercising the confirmation gate by voice ("new game" → "yes" / "no") whenever the walkthrough happens, since it changed how destructive ops feel and has only been proven by tests and one eval run.

## Agent reliability — finish a game by voice

From the 2026-07-10 agent deep dive: voice games die by attrition — an illegal-move guess ends the turn with no recovery, the brain's prompt is ~75% UI noise (`fens`/`dests`) that grows every ply, the system prompt gives zero speech→SAN guidance, and every plain move pays the full two-LLM-call round trip (~10–20s). Slices in priority order; the first three are small, independent, and low-risk.

- [ ] **Experiment — phase-1 sampling**: A/B a lower temperature (~0.2–0.6) for the tool-decision call only, keeping BRIEF sampling for commentary; record tool-call accuracy before changing any default.
- [ ] **Pending-proposal state for confirmations** (only if the dead-end recurs after #85): when the agent proposes a specific move and the player answers a bare "yes", only the #85 prompt rule guarantees the move gets played. A small pipeline-owned "pending proposed move" would make the confirmation itself deterministic ("yes" → make_move(pending)) — but it adds conversation state to the pipeline; hold unless live games show the prompt rule isn't enough. **The destructive-op gate (2026-07-13) now supplies the exact machinery** — `ToolContext.pending` + `parse_confirmation` — so this is a much smaller job than when it was written, and its "adds conversation state" objection is already paid for.

## Phase 4 — Manual testing (walkthrough together, take notes)

No systematic walkthrough has happened yet — most items below have only been exercised by automated tests, though a few were spot-verified live (first real agent games #95/#104, iOS hands-free #103; noted inline where so). Go through this list at the desk, note anything that feels wrong or needs to change, then turn the notes into new backlog items.

### Setup / stack
- [ ] `docker compose up` brings the app up cleanly from cold against the two external stacks it depends on: voice (STT via Speaches + TTS via Kokoro-FastAPI, the shared `../speech/` stack) and the LLM brain (`../llama-swap/`). Both are exercised separately; `app` is the only service in this repo's compose file.
- [ ] App is reachable from another device on the home network (phone/laptop browser)
- [ ] Basic gameplay works fully offline (network unplugged)
- [ ] Full game against Stockfish with the LLM turned off (agent layer disabled)

### Core gameplay (board UI)
- [ ] Play a complete game start to checkmate; board renders correctly throughout
- [ ] Illegal moves on the board are rejected without corrupting state
- [ ] Special moves work end-to-end: castling (both sides), en passant, pawn promotion (incl. underpromotion)
- [ ] Check, checkmate, stalemate, and draw conditions are detected and surfaced in the UI
- [ ] Captured pieces display updates correctly
- [ ] New game assigns a random side (board flips when playing black; engine opens); "Switch to …" offered until the first player move; undo takes back the full exchange and never the engine's lone opening move
- [ ] One layout everywhere: the stacked (mobile-style) UI reads well on a desktop monitor too — column capped, bottom bar a centered cluster

### Text agent (tool boundary in real use)
- [ ] Moves via free-form text ("knight to f3", "Nf3", "castle kingside") land as the intended move
- [ ] Ambiguous input ("move the rook") produces a clarifying question, not a guess or a wrong move
- [ ] Illegal move requests get a clear rejection from the agent, board unchanged
- [ ] `undo`, `resign`, `new_game` via natural language
- [ ] `save_game`, `resume_game` round-trip (save mid-game, restart app, resume)
- [ ] `export_pgn` output loads in an external PGN viewer
- [ ] Reads behave sensibly: "what's the position?", "what are my legal moves?", "show move history", "what's been captured?"
- [ ] Agent reaction latency after each move feels acceptable (thinking OFF path)
- [ ] Long game: prompts/KV-cache don't degrade tool-call reliability late in the game

### Analysis & review
- [ ] "What was my mistake?" (`analyze_last_move`) gives a sensible answer, thinking ON
- [ ] Hints mode: hints appear when on, never when off
- [ ] `evaluate_position` / "who's winning?" gives a coherent eval
- [ ] Post-game `review_game`: move classifications and per-color accuracy look plausible against a known game
- [ ] Post-game screen: pops up at game over with the player-side verdict ("You won/lost", "Draw") and termination + result; Close leaves the final board inspectable and the status-line "Results" chip reopens it; everything resets on new game
- [ ] Post-game screen: "Review game" shows per-color accuracy, classification counts, and flagged moves with better alternatives; a fetched review survives dismiss/reopen
- [ ] Post-game screen: "Review unavailable" (not a crash) when the engine is off, and the button allows a retry; "Copy PGN" puts a loadable PGN on the clipboard
- [ ] Analysis latency (thinking ON) is tolerable; whole-game review latency acceptable on a long game

### Personality & settings
- [ ] Glitch's tone comes through in commentary (tone only — personality must NOT change move strength or settings)
- [ ] `set_difficulty` visibly changes engine strength (easy loses to a casual player, hard doesn't); the UI difficulty selector matches what the engine actually plays, including after an app restart
- [ ] Settings by speech/text: difficulty, verbosity, hints mode, voice output all switchable mid-game and persist
- [ ] Verbosity levels actually differ (terse vs chatty)

### Voice (STT/TTS)
- [ ] Voice input: spoken moves transcribe and execute correctly (test noisy vs quiet room)
- [ ] Hands-free conversation mode on a real phone — **iOS Safari verified live 2026-07-11** (#103; after the stale-chunk and wasm-OOM fixes); still open: Android Chrome, and the finer feel checks on both (VAD endpointing 1s window, half-duplex, autoplay unlock, exit-tap always reachable)
- [ ] STT latency (~2.5s known) is acceptable in the play loop
- [ ] TTS output (`speak`) is intelligible, correct voice, reasonable latency
- [ ] `set_voice_output` on/off actually mutes/unmutes TTS
- [ ] Misrecognized speech fails safe (clarifying question, not a wrong move)

## Infrastructure / process (ongoing)

- [ ] If repo goes public or account upgrades to Pro: enable branch protection (require `lint` + `test` + `frontend` checks) and native auto-merge

## Backlog (no near-term timeline)

- [ ] GBNF grammar-constrained decoding fallback — **deferred, likely unneeded**: the live spike confirmed Gemma-4 (`UD-Q4_K_XL`) emits structured OpenAI tool calls natively. Revisit only if reliability degrades under load/longer prompts.
- [ ] Physical board (Chessnut Move — blocked on hardware purchase): verify motorized actuation is programmatically controllable **before any design work**; until then, `control_physical_board` tool seam only
