# TODO

The backlog, in priority order. One task = one vertical slice = one branch = one PR (TDD: failing test first). When a task is finished and merged, move its line to `DONE.md` with the merge date. Re-plan freely between slices — this file is the living backlog, not a contract.

## Next sprint (2026-07-14) — the agent's per-turn context: measure it, then cut it

**The current focus, and it absorbs the old "agent reliability" work.** The "see it, then fix it" sprint is closed (2026-07-13, in DONE.md): traces made behavior readable and the four tool-signature bugs, the honesty guard, and the confirm gate all shipped. The reliability problem that's *left* isn't a specific bug — it's that ~98% of every brain turn is fixed harness overhead. System prompt + tool definitions dwarf the player's command, and on a 12B quant that overhead is instruction competition, not just latency: every token of dead persona and schema noise argues with the tool contract for the model's attention. Full accounting in `docs/agent-harness-audit-2026-07-14.md`; this sprint works that document.

**The through-line for the middle slices: the tools should hand the agent what it needs, and the agent should carry a *summary* of each turn rather than a raw window.** A rule the prompt hopes the model follows becomes tool behavior it can't skip; a transcript window sized by guess becomes per-turn summaries sized by need. Audit findings 2/4/7 collapse into one question — what does one brain turn actually need to contain — not three separate trims.

- [ ] **[M] Tools return what the agent needs; turn-summaries replace the raw window** (audit findings 2/4/7 — the sprint's real theme). Make each tool result carry the fact the agent would otherwise have to remember or re-fetch (an illegal `make_move` returns the legal moves; the confirm-gate refusal already carries the exact line to relay — extend that pattern across the surface). Then reconsider the 20-turn transcript window: a per-turn summary the pipeline writes may hold banter continuity for a fraction of the tokens. **Design before code** — this reshapes what one brain turn even contains. **Also owns curing the base-prompt-shrink regression** (finding 2, merged eval-RED in #136): `long_capture` `poisoned` 4→0/5, `live_like` 4→3/5 — the model stops recognizing a descriptive move ("grab the pawn on e6") as `make_move` in noisy context. The bet is that cleaning the context the recognition drowns in fixes it; re-run those scenarios as the acceptance gate for this slice.
- [ ] **[S] Code-own hints gating** (audit finding 9; the suite's one remaining xfail, `hints_off_no_advice` 0/5) — hints-off "keep the engine's secrets" is a prompt rule a 12B honors ~half the time. When hints are off, don't *offer* `get_best_moves` (the registry already supports per-caller `exclude`), so the prompt does only tone. Same shape as the confirm-gate fix. (The tool-name mentions in `_HINTS_INSTRUCTION` were already dropped, 2026-07-14.)
- [ ] **[S] Merge the three settings setters? — genuinely 50/50** (audit finding 6) — `set_verbosity`/`set_hints_mode`/`set_voice_output` are one shape; `set_option(name, value)` saves ~150 tok and shrinks the model's choice space, at the cost of a stringly-typed call that's easier to mis-invoke. Decide with the evals, not in the abstract.
- [ ] **[S] Fresh traced session on current `main`** (audit finding 10) — the 46 traced turns predate the capture family, the resign route, the honesty guard, and this sprint's prompt changes, so the cost model that shaped the audit is already stale. Re-baseline, ideally after finding 1 lands so it arrives with token counts attached.

Carried from the closed sprint, not yet folded above:

- [ ] **[S] The eval gate is flaky at the floor** — `_PASS_RATE_RUNS` is still 5, and a true 60–100% rate flips the gate by coin toss (`long_capture[poisoned]` went 5/5 then 3/5 on the same `main` commit). Raise the runs (cost: wall time) or move the floor (cost: a weaker tripwire), decided with a measured rate over ~20 runs — not tuned until green. A gate nobody trusts gets ignored.
- [ ] **[M] Show tool calls live in the UI, like "thinking…"** — the player should see the agent's tools as it calls them (`undo` → `make_move`), not just the commentary after. Stream or poll the tool trajectory during a turn; the trace record is already the right shape.
- [ ] **[S] Phase-1 sampling experiment** — A/B a lower temperature (~0.2–0.6) for the tool-decision call only, keeping commentary sampling as-is; record tool-call accuracy before changing any default.

**[L] The Phase-4 manual walkthrough** (below) stays queued behind this. Worth exercising the confirmation gate by voice ("new game" → "yes"/"no") whenever it happens, since it changed how destructive ops feel and has only been proven by tests and one eval run.

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

- [ ] **Pending-proposal state for confirmations** (conditional — only if the dead-end recurs): when the agent proposes a specific move and the player answers a bare "yes", only the prompt rule guarantees the move gets played. A pipeline-owned "pending proposed move" would make it deterministic ("yes" → `make_move(pending)`). **The destructive-op gate already supplies the machinery** (`ToolContext.pending` + `parse_confirmation`), so this is a small job now — but it adds conversation state, so hold unless live games show the prompt rule isn't enough.
- [ ] GBNF grammar-constrained decoding fallback — **deferred, likely unneeded**: the live spike confirmed Gemma-4 (`UD-Q4_K_XL`) emits structured OpenAI tool calls natively. Revisit only if reliability degrades under load/longer prompts.
- [ ] Physical board (Chessnut Move — blocked on hardware purchase): verify motorized actuation is programmatically controllable **before any design work**; until then, `control_physical_board` tool seam only
