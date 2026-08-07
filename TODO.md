# TODO

The backlog, in priority order. One task = one vertical slice = one branch = one PR (TDD: failing test first). When a task is finished and merged, move its line to `DONE.md` with the merge date. Re-plan freely between slices — this file is the living backlog, not a contract.

## Replanned 2026-07-25 — the agent-control audit

The backlog is rebuilt around the external agent-control audit, reviewed and annotated in `docs/agent-control-audit-2026-07-25.md` (read that first: it marks what the report got right, what already exists, and where its prescriptions were reshaped to fit the house rules). Its P0 finding is the real one — **Glitch does not genuinely control agent-enabled turns**: a board drag never touches the agent (`/api/game/move` is a silent bypass), `make_move` bundles the player's move and the engine's reply into one atomic call so Glitch has no beat between them, the loop's closing "commentary" pass still has tools on offer, and the once-per-turn mutation rule is docstring prose. The target architecture: **player intent → validated player move → Glitch observes → Stockfish replies → Glitch closes the turn — with the deterministic coordinator, never the model, owning the sequence.**

This replaces the 2026-07-14 context-sprint framing. Surviving items from it are folded in below where the audit re-derived them (hints gating, structured tool errors, turn summaries, live tool UI, eval statistics, sampling experiment). Two standing constraints carry over unconditionally:

- **The `long_capture` regression stays release-blocking** — and as of 2026-07-25 it is **green**: the planner/narrator split cured it (`poisoned` 1/5 → 5/5, all conditions, both measured planner temperatures — `docs/agent-evals.md`), confirming the instruction-competition diagnosis. The constraint now reads: it must *stay* green; any prompt or schema reduction that sends it red again does not merge. Sprint 3's structured-errors slice inherits keeping it green, not curing it.
- **The shelved schema-cut warning: do not re-attempt schema minimization on gemma-4-12b.** Every stripped pydantic key (`title`/`anyOf`-null/`default`) independently collapsed `undo_and_replace` below the floor. Full record moved to `DONE.md` (2026-07-21); gate on `test_eval_undo_and_replace_is_one_turn` if the brain model ever changes.

Parked work: branch `chore/clean-tool-schema-noise` holds the uncommitted `make_move` docstring cut (tripwires green). Sprint 1 rewrites the move tools anyway — salvage its docstring lessons there (especially: result-restating prose like "the engine decides legality" is a plausible anti-self-poisoning anchor; don't cut it blindly from `resign`/`new_game`).

### Sprint 5 — P4/P5: observability and eval hardening (audit 18, 19, 21–23)

- [ ] **[S] Re-run the `play_as_black` order experiment cleanly** (the one arm the 2026-07-26 campaign owes). The campaign measured it 20/20 at the cap and 5/5 mid-suite — the recorded 0/5-mid-suite effect did not reproduce — but for part of the 20-sample run a second pytest was hitting the same llama-server, so the isolated arm is not clean. Re-run `play_as_black` alone at `CHESSAPP_EVAL_RUNS=20 CHESSAPP_EVAL_MAX_RUNS=20` with nothing else on the GPU, and again placed after the long-transcript block, then compare the two report lines (per-block rates and per-sample `model_ms` are recorded for exactly this). Until then the run-order confound is **inconclusive, not settled** — and note the whole 2026-07-26 session saw zero provider deaths across 135 samples, so it was a healthy-server session and says little about the bad ones.
- [ ] **[S] Fresh traced session on the new architecture** (old audit finding 10, still right). The 46 traced turns predate everything above; re-baseline with token counts once Sprint 1–2 land.
- [ ] **[M] Decide whether a *tighter* narrator wordiness cap is still worth it, now that the safety ceiling exists.** The runaway itself is bounded as of the 2026-07-28 `max_tokens` slice (planner 2048 / narrator 4096, truncation handled — `docs/planner-narrator.md`), so what remains is only the wordiness question the 2026-07-27 token study raised (`docs/agent-evals.md`): a repeat-stop narration writes **2.58× the tokens** at a **flat rate** (0.98×, p = 0.30), `narrator_ms` tracks `narrator_out` at **r² = 0.998**, but legitimate `completed`-and-passing narrations reach **1,259** tokens and the repeat tail is **521–2,633** — the distributions overlap almost entirely, so a cap safe for ordinary narration (~1,300) clips only 3/20 samples and a cap that catches the median tail (~600) truncates real commentary. The overlap argues for dropping this; if attempted anyway, it is a `src/` change and owes a full gate, with `long_capture` staying green.
- [ ] **[M] The planner flips hints on unasked — confirmed; decide the capability question.** Verified 2026-07-27 across two clean 20-sample arms: **3/40 called `set_hints_mode`, every one `enabled=true`** (12/105 ≈ 11% counting the 9/65 that raised it), on a turn where the player asked only "what should I play here?". The scenario now asserts `app.ctx.settings.hints_mode is False` after the turn, and the post-assertion arm confirms it fires on the real thing (a sample that would have passed silently — no board mutation, no SAN leak) without false-positiving, gate still green at 19/20. That assertion is the *measurement*, not the remedy. **What is open is the capability question:** should the planner be offered `set_hints_mode` (and the sibling setters) at all on a turn that was a question? The house rule points at yes — a prompt line is followed ~half the time and a withheld tool is never called — and the offer is already resolved per command off live settings (`get_best_moves` is withheld with hints off), so the seam exists. Against it: the player *can* legitimately say "turn hints on", and a blanket withholding breaks settings-by-voice, so the condition has to be about the utterance, which is exactly the kind of judgment the offer resolver does not currently make. `src/` change, owes a full gate. **Related but separate:** the arm's one failure also *played a move* after flipping (`set_hints_mode(enabled=true) → make_move(move="Bc4")`); 1 failure / 2 flips / 20 samples is suggestive, not established, and needs its own arm before anyone calls it a causal chain.

**Decision (2026-07-26): evals never enter CI, and audit 21's scheduled-run bit is dropped.** The eval suite stays a manual local command. It needs the GPU, it costs minutes, and its whole value is a human reading the numbers next to a change they just made — a scheduled run would produce unread reds against a shared card and train everyone to ignore the one gate the project has. Nothing eval-related goes in `.github/`. The CI/eval split the audit asked for already exists (`CHESSAPP_AGENT_EVALS=1`, skipped by default).

**[L] The Phase-4 manual walkthrough** (below — the audit's item 24) stays queued behind the sprints; the coordinator work changes how turns *feel*, so walking through before Sprint 1 would mostly measure code about to be replaced. Worth exercising the confirmation gate by voice whenever it happens.

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

- [ ] **[S] Transport-guard the remaining `api.ts` helpers** (follow-up to #231/#232, which brought `submitMove`/`sendCommand`/`transcribe` in line): the other helpers (`fetchState`, `fetchSettings`, `postLifecycle`, `postGated`, `confirmDestructive`, `setDifficulty`, `setVoiceOutput`, `fetchHint`, `fetchReview`, `fetchPgn`) still reject on a dropped connection despite documenting a null/state return, and several `await res.json()` unguarded — the same defect class with a lower blast radius (their callers `void` the promise into an unhandled rejection or run in mount effects, rather than wedging the mic or desyncing a piece). One sweep, one convention: every helper resolves to its documented failure shape and never throws.
- [ ] **[S] Re-vendor the hands-free voice module downstream** (follow-up to #232): `MicButton.tsx` is the canonical copy other apps vendor verbatim (see its header / `agent-standard`'s voice module doc), so every downstream copy inherits the pre-#232 wedge until refreshed — their check-sync CI will flag the drift, this line is the reminder to act on it. The fix has two halves there: the vendored file (a clean copy — the belt-catch was deliberately kept self-contained), and the *host app's own* api client, whose transcribe/command helpers need the same transport-to-null guard, which is app code the vendored file can't carry.
- [ ] **[S] UI claim-draw affordance** (follow-up to #220, which shipped claimable draws backend-complete): the state document already names the claimable rules (`claimable_draws`) and the post-game modal humanizes the new terminations, so what's missing is the control — a `/api/game/claim-draw` endpoint through the registry (the same deterministic gate the resign button uses), a `_CONFIRM_QUESTIONS` entry for its confirmation wording, and the frontend button. Until then a claim is reachable by agent/MCP/delegate only — i.e. unreachable in direct mode (`CHESSAPP_AGENT=off`).
- [ ] **Pending-proposal state for confirmations** (conditional — only if the dead-end recurs): when the agent proposes a specific move and the player answers a bare "yes", only the prompt rule guarantees the move gets played. The Sprint-1 turn state machine is the natural home if this ever ships — it adds exactly the "pending" conversation state this needs; hold unless live games show the prompt rule isn't enough.
- [ ] **Merge the three settings setters? — genuinely 50/50** (old audit finding 6) — `set_verbosity`/`set_hints_mode`/`set_voice_output` are one shape; `set_option(name, value)` saves ~150 tok and shrinks the model's choice space, at the cost of a stringly-typed call that's easier to mis-invoke. Decide with the evals, not in the abstract.
- [ ] GBNF grammar-constrained decoding fallback — **deferred, likely unneeded**: the live spike confirmed Gemma-4 (`UD-Q4_K_XL`) emits structured OpenAI tool calls natively. Revisit only if reliability degrades under load/longer prompts.
- [ ] Physical board (Chessnut Move — blocked on hardware purchase): verify motorized actuation is programmatically controllable **before any design work**; until then, `control_physical_board` tool seam only
