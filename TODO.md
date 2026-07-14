# TODO

The backlog, in priority order. One task = one vertical slice = one branch = one PR (TDD: failing test first). When a task is finished and merged, move its line to `DONE.md` with the merge date. Re-plan freely between slices — this file is the living backlog, not a contract.

## Next sprint (2026-07-12)

The next three slices, in order. This is a framing block over the sections below, not a rewrite — every item stays tracked in its own section:

1. **[S] Ship Glitch's voice** — pick the final TTS voice by ear from the #97 audition artifact and resolve the swearing watch-item; closes the current-focus trio. Done when the winner is set as `TTS_VOICE`.
2. **[M] Structural fix for the destructive-op confirm gate** — a deterministic pipeline-owned "pending destructive op" state (bare "new game" → confirmation question → "yes" → `new_game`). Pre-step: measure `resign`'s adherence rate. Done when the `destructive_confirm` xfail flips to a hard assert. The agent-loop rework it was blocked on has **landed** (see DONE.md, 2026-07-13), so design this against the single-loop pipeline in `_run_command`. Note the adherence rate is genuinely a coin flip: across the measurement slice's runs `new_game` asked first 3/3 times and then fired immediately on the next run.

Stretch: start the Phase-4 manual walkthrough, prioritizing the #114–#116 UI items (single layout, random color + side switch, post-game screen) and Android Chrome hands-free.

## Personality & voice — Glitch (current focus)

The 2026-07-11 direction: one hand-dialed personality — **Glitch**, a gen-z Jarvis (fully casual, low-key troll, help stays real, swearing allowed) — instead of the selectable roster, plus a designed custom TTS voice ("laid-back young American, audible smirk") instead of stock Kokoro `af_heart`. Voice-option research (Kokoro blending / Chatterbox Turbo / Qwen3-TTS / NeuTTS Air, with the 12 GB VRAM constraint) is in Claude's project memory.

- [ ] **Watch — does Glitch actually swear?** Zero swears in the first two live games despite escalated authorization (#95). If real games stay PG, add a few-shot exchange to the tone block (one blunder → one sweary roast) — the next and probably last lever short of accepting PG-13.
- [ ] **Pick Glitch's final voice by ear**: listen through the audition artifact (13 samples + voice strings; link in the #97 PR conversation), set the winner as `TTS_VOICE` in docker-compose.yml, then `docker compose up -d --build app` (rebuild also picks up the Glitch prompt). Default meanwhile: `am_fenrir(2)+am_michael(1)`.
- [ ] **Spike — Qwen3-TTS voice design** (only if no Kokoro blend has enough smirk): trim shared llama-swap context to free ~2 GB VRAM, serve via vLLM-Omni, design the voice from the prose spec

## Agent reliability — finish a game by voice

From the 2026-07-10 agent deep dive: voice games die by attrition — an illegal-move guess ends the turn with no recovery, the brain's prompt is ~75% UI noise (`fens`/`dests`) that grows every ply, the system prompt gives zero speech→SAN guidance, and every plain move pays the full two-LLM-call round trip (~10–20s). Slices in priority order; the first three are small, independent, and low-risk.

- [ ] **Experiment — phase-1 sampling**: A/B a lower temperature (~0.2–0.6) for the tool-decision call only, keeping BRIEF sampling for commentary; record tool-call accuracy before changing any default.
- [ ] **Pending-proposal state for confirmations** (only if the dead-end recurs after #85): when the agent proposes a specific move and the player answers a bare "yes", only the #85 prompt rule guarantees the move gets played. A small pipeline-owned "pending proposed move" would make the confirmation itself deterministic ("yes" → make_move(pending)) — but it adds conversation state to the pipeline; hold unless live games show the prompt rule isn't enough.
- [ ] **Destructive-op confirmation is unreliable at temp 1.0** (found by the eval harness, 2026-07-11): gemma-4-12b honors the "confirm before `new_game`/`resign`" prompt rule only ~50% of the time (measured across trivial and developed positions; `docs/agent-evals.md` finding, `destructive_confirm` xfail). The prompt carries the rule and `test_personality` pins it, but the model doesn't follow it reliably. Likely fix is structural, not prompt-only — a pipeline-owned "pending destructive op" (same shape as the pending-proposed-move item above): bare "new game" → confirmation question → "yes" → `new_game`. Confirm the same rate for `resign` before designing. Prompt was deliberately not changed in the eval slice (evals gate prompt changes). Flip the xfail to a hard assert once fixed.

## Phase 4 — Manual testing (walkthrough together, take notes)

No systematic walkthrough has happened yet — most items below have only been exercised by automated tests, though a few were spot-verified live (first real agent games #95/#104, iOS hands-free #103; noted inline where so). Go through this list at the desk, note anything that feels wrong or needs to change, then turn the notes into new backlog items.

### Setup / stack
- [ ] `docker compose up` brings up all three containers (app, Speaches, Kokoro) cleanly from cold — the LLM brain is the external `../llama-swap/` stack, exercised separately
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
