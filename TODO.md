# TODO

The backlog, in priority order. One task = one vertical slice = one branch = one PR (TDD: failing test first). When a task is finished and merged, move its line to `DONE.md` with the merge date. Re-plan freely between slices — this file is the living backlog, not a contract.

## Replanned 2026-07-25 — the agent-control audit

The backlog is rebuilt around the external agent-control audit, reviewed and annotated in `docs/agent-control-audit-2026-07-25.md` (read that first: it marks what the report got right, what already exists, and where its prescriptions were reshaped to fit the house rules). Its P0 finding is the real one — **Glitch does not genuinely control agent-enabled turns**: a board drag never touches the agent (`/api/game/move` is a silent bypass), `make_move` bundles the player's move and the engine's reply into one atomic call so Glitch has no beat between them, the loop's closing "commentary" pass still has tools on offer, and the once-per-turn mutation rule is docstring prose. The target architecture: **player intent → validated player move → Glitch observes → Stockfish replies → Glitch closes the turn — with the deterministic coordinator, never the model, owning the sequence.**

This replaces the 2026-07-14 context-sprint framing. Surviving items from it are folded in below where the audit re-derived them (hints gating, structured tool errors, turn summaries, live tool UI, eval statistics, sampling experiment). Two standing constraints carry over unconditionally:

- **The `long_capture` regression stays release-blocking** (`live_like`/`poisoned` RED on `main` since #136 — "grab the pawn on e6" stops resolving to `make_move` in noisy context). Owned by the structured-errors slice in Sprint 3; no further prompt or schema reduction merges while it's red.
- **The shelved schema-cut warning: do not re-attempt schema minimization on gemma-4-12b.** Every stripped pydantic key (`title`/`anyOf`-null/`default`) independently collapsed `undo_and_replace` below the floor. Full record moved to `DONE.md` (2026-07-21); gate on `test_eval_undo_and_replace_is_one_turn` if the brain model ever changes.

Parked work: branch `chore/clean-tool-schema-noise` holds the uncommitted `make_move` docstring cut (tripwires green). Sprint 1 rewrites the move tools anyway — salvage its docstring lessons there (especially: result-restating prose like "the engine decides legality" is a plausible anti-self-poisoning anchor; don't cut it blindly from `resign`/`new_game`).

### Sprint 1 — P0: the turn coordinator (audit items 1–5, 10)

Design before code for the first slice — this reshapes the game loop. Non-negotiables from the review: the coordinator is deterministic code that owns `player_move → observe → engine_reply → close`; Glitch fills the observation slots and can never block, skip, or reorder the engine reply (model timeout/absence degrades that beat to direct mode); LLM-off full games keep working untouched; verbosity=low keeps today's zero-LLM plain move. **Latency is an acceptance criterion:** if the observation beat makes a plain move feel slower, it failed.

- [ ] **[L] Turn state machine + coordinator** (audit 3, 10). Explicit per-turn states (`awaiting_player → player_move_applied → agent_observing → engine_calculating → engine_move_applied → completed`), a turn ID, and rejection of actions that don't belong in the current state. One validation layer for tool schema + turn state + confirmation status, so the API path, the brain path, and the delegate/MCP paths stop drifting. The engine reply is the coordinator's job, never a model-callable tool.
- [ ] **[M] Split the move flow: Glitch observes between the player's move and the engine's reply** (audit 2, 5). `make_move` (the tool) and `/api/game/move` both become "apply player move" only; the coordinator then hands Glitch structured facts (move, capture, check/mate/draw status, new FEN) for a short reaction — streamed or run while Stockfish calculates — then applies the engine reply and offers Glitch the close. Observation beats are skippable and bounded; low verbosity skips them entirely.
- [ ] **[M] Board controls route through the coordinator in agent mode** (audit 1, 4). A drag sends the structured move (`e2e4`) — never natural language — into the same sequence, so Glitch sees drag-played games too. UI undo/resign/new-game go through the same confirmation gate as spoken commands. Direct mode (LLM off/unavailable) stays fully playable and becomes *visible* in the UI rather than a silent per-input bypass.

### Sprint 2 — P1: mutation safety in code (audit 6, 7, 9, 20)

The house rule, applied to the loop itself: a rule the prompt asks for holds ~half the time; a policy the code owns holds always.

- [ ] **[S] Per-turn mutation limits, code-enforced** (audit 6). Max one accepted player move, one engine move, one destructive op per turn; after a successful mutation the coordinator refuses (or stops offering) that tool for the rest of the turn. Deletes the "at most once per player turn" docstring prose it replaces.
- [ ] **[S] Tool-free closing pass** (audit 9). Once the turn's work is done (or the mutation limit is hit), the final model call is made with no tools on offer — the commentary CLAUDE.md already describes, now enforced structurally instead of by the model declining.
- [ ] **[S] Board-version precondition on mutations** (audit 7). Every state-changing request carries the board version / expected FEN and is rejected stale — closes the web + delegate + MCP concurrent-client race on the one shared session. E2E test: concurrent clients cannot advance the same turn twice.
- [ ] **[S] Recovery semantics for provider failure mid-turn** (audit 20). Define and test what happens when the model times out after the player's move: position stays valid and resumable, the move is never silently replayed, and a fast-path `narrate` failure can't 500 a turn whose move already landed (today it escapes unhandled after the board changed, before the broadcast).

### Sprint 3 — P2: settings and honesty as enforceable policy (audit 11–14, 16)

- [ ] **[S] Code-own hints gating** (audit 11; the suite's one remaining xfail, `hints_off_no_advice` 0/5). Hints off ⇒ don't *offer* `get_best_moves` (registry already supports per-caller exclude), don't feed evals/candidates into context; prompt does tone only. Same shape as the confirm-gate fix.
- [ ] **[S] Inject live settings into the agent's per-turn state** (audit 12, the missing half). Verbosity/hints already re-resolve into the system prompt per command; add difficulty and voice-output to the injected state block so the model never infers a setting from stale conversation text.
- [ ] **[M] Structured tool errors: tools return what the agent needs** (audit 14; the old context-sprint [M], first half). An illegal `make_move` returns legal alternatives; an ambiguous request returns the candidates; errors say whether retrying is safe. Extend the confirm-gate pattern (the refusal carries the exact line to relay) across the tool surface. **Owns curing the `long_capture` regression** (audit 16): the bet is that richer, fresher results fix recognition drowning in noisy context — re-running those scenarios is this slice's acceptance gate.
- [ ] **[M] Expand the honesty guard to verified reply facts** (audit 13). Today's guard catches false game-over/reset claims only. Build the verified-facts set from the turn's tool results + board (moves, captures, check/mate/draw, saves, settings changes, analysis numbers) and check operational claims in commentary against it — personality varies wording, never facts. Grow it claim-class by claim-class with `test_honesty.py` as the spec; trash talk and hypotheticals must keep surviving.

### Sprint 4 — P3: reasoning quality (audit 15, 17)

- [ ] **[S] Per-phase sampling experiment** (audit 15, reshaped — a full planner/narrator re-split would reverse the #124 one-loop rework; see the review doc). A/B lower temperature (~0.2–0.6) on tool-selection calls only, commentary sampling unchanged; record tool-call accuracy before changing any default. The P0 observe/close beats already give narration its own call — re-evaluate whether more separation is needed after Sprint 1 lands.
- [ ] **[M] Turn summaries replace the raw transcript window** (audit 17; the old context-sprint [M], second half). Latest few turns verbatim, older turns summarized into game events / preferences / unresolved requests; board truth always from current state, never history. Design before code — this reshapes what one brain turn contains.

### Sprint 5 — P4/P5: observability and eval hardening (audit 18, 19, 21–23)

- [ ] **[S] Close the trace gaps** (audit 18 — most of it already exists: route, FENs, trajectory, stop_reason, guard decision, tokens). Add turn/correlation IDs, per-model-call latency, and an explicit mutation count; thread the turn ID through the coordinator states so a duplicated move is diagnosable from one record.
- [ ] **[M] Live phase progress in the UI** (audit 19; absorbs the old "show tool calls live" item). Surface "validating your move" / "Glitch is reacting" / "Stockfish is calculating" as the coordinator moves through states, plus the tool calls as they happen — essential once a turn has multiple intentional phases.
- [ ] **[S] Eval statistics worth trusting** (audit 23; the old flaky-floor item). 5-run pass rates flip on a coin at the floor — raise repetitions or use confidence intervals, decided from a measured ~20-run rate; report deterministic failures separately from model variance. Add a scheduled eval run (audit 21's one new bit — the CI/eval split already exists).
- [ ] **[S] E2E gap sweep against the audit's risk list** (audit 22). Most of its eight scenarios land as acceptance tests inside the slices above; this closes whatever's left (drag-through-Glitch, observe-before-reply, no duplicate moves, hints-off, pending-op safety, provider-failure resumability, stale-version rejection, concurrent clients) as deterministic CI tests at the tool boundary.
- [ ] **[S] Fresh traced session on the new architecture** (old audit finding 10, still right). The 46 traced turns predate everything above; re-baseline with token counts once Sprint 1–2 land.

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

- [ ] **Pending-proposal state for confirmations** (conditional — only if the dead-end recurs): when the agent proposes a specific move and the player answers a bare "yes", only the prompt rule guarantees the move gets played. The Sprint-1 turn state machine is the natural home if this ever ships — it adds exactly the "pending" conversation state this needs; hold unless live games show the prompt rule isn't enough.
- [ ] **Merge the three settings setters? — genuinely 50/50** (old audit finding 6) — `set_verbosity`/`set_hints_mode`/`set_voice_output` are one shape; `set_option(name, value)` saves ~150 tok and shrinks the model's choice space, at the cost of a stringly-typed call that's easier to mis-invoke. Decide with the evals, not in the abstract.
- [ ] GBNF grammar-constrained decoding fallback — **deferred, likely unneeded**: the live spike confirmed Gemma-4 (`UD-Q4_K_XL`) emits structured OpenAI tool calls natively. Revisit only if reliability degrades under load/longer prompts.
- [ ] Physical board (Chessnut Move — blocked on hardware purchase): verify motorized actuation is programmatically controllable **before any design work**; until then, `control_physical_board` tool seam only
