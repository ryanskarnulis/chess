# TODO

The backlog, in priority order. One task = one vertical slice = one branch = one PR (TDD: failing test first). When a task is finished and merged, move its line to `DONE.md` with the merge date. Re-plan freely between slices — this file is the living backlog, not a contract.

## Agent loop rework — one real tool loop (2026-07-13, top priority)

From the 2026-07-13 loop review. Today's `_run_command` is **two phases, not a loop**: the brain is called once, its tool calls are dispatched, and `react()` comments on the results *with no tools offered*. The only path back to the model is failure (`_failed_calls`). Success is terminal — so the agent never sees a successful tool result while it still holds tools, and every multi-step intent is structurally impossible ("play the best move" needs `get_best_moves` → read → `make_move`; it can only guess blind). The four read/analysis tools exist to inform an action the agent cannot take.

Two supporting defects, both fixed by the same change:

- **Tool results never re-enter the conversation.** `ChatResult.to_message()` (`provider.py:123`) is dead code outside tests; there is no `role: "tool"` message anywhere. The retry path discards the assistant's own tool-call turn and re-prompts from scratch with a synthetic *user* message describing the failure (`_retry_command`), so the model never sees the `assistant(tool_calls)` → `tool(result)` structure it was trained on, and the prompt is rebuilt each round instead of grown (KV cache thrown away).
- **Two nested retry loops.** The brain retries schema violations (`llama_brain.py:90`, `max_retries=2`) *inside* the pipeline's domain-failure retry (`api.py:446`, `MAX_TOOL_RETRY_ROUNDS=2`), composing multiplicatively: up to 9 completions plus `react` = **10 model round-trips worst case** on a local 12B.

Target shape — one bounded loop in the brain, results fed back as real `tool` messages, the first assistant turn with no tool calls *being* the commentary:

```
messages = [system, *transcript, user(board + command)]
for _ in range(MAX_ITERATIONS):        # ~4
    result = provider.chat(messages, tools=...)
    if not result.tool_calls:
        return result.content          # the commentary — react() disappears
    messages.append(result.to_message())
    for call in result.tool_calls:
        messages.append({"role": "tool", "tool_call_id": call.id,
                         "content": json.dumps(registry.dispatch(call.name, call.args))})
```

Invariants hold: the registry still dispatches, so the agent still cannot touch the board directly; the fast path (`parse_move` → `make_move`, zero-LLM at verbosity=low) is untouched. Schema failures and illegal moves stop being special cases — both become just another tool-result message the model reads and corrects from, collapsing the two nested retry loops into the one iteration bound (plus the small wire-level correction budget noted in slice 1, per STANDARD.md §3).

Slices, in order (TDD; the eval harness gates the merge — this is a loop change, so `docs/agent-evals.md` must not regress):

1. ~~**[M] Brain owns a bounded tool loop** + **[M] pipeline: drop phase two and the domain-retry loop.**~~ **Landed together** (`feat/brain-tool-loop`) — moving dispatch into the brain made the pipeline's dispatch and domain-retry code dead on arrival, so the two slices were one change. `LlamaBrain.get_agent_response` is now the STANDARD.md §3 loop (`max_iterations=4`, separate `max_corrections=2`, `completed | max_iterations | correction_limit`); tool results go back as real `role: "tool"` messages on a growing prompt; domain rejections are results, schema failures are corrections. `react()`, `_failed_calls`, `_retry_command`, `MAX_TOOL_RETRY_ROUNDS` are gone.
   - **The `react` question is settled:** the loop's closing tool-less turn *is* the commentary and cannot act (no tools on offer), which enforces the CLAUDE.md separation structurally. CLAUDE.md's Game Loop section is updated.
   - **One thing survived:** the fast path never enters the loop, so it has no closing turn to comment with — `Brain.narrate` (react's body, scoped to that one caller) supplies it. At verbosity=low the canned confirmation still stands in, so a plain move stays zero-LLM.
2. **[S] Thinking policy for the loop — measure.** The rule is already implemented (thinking OFF until an analysis tool's result lands in context, then ON for the remaining turns — it had to be, to keep `judgment_question` from regressing). What's left is measurement: confirm a plain move stays zero-LLM on the fast path and 2 calls otherwise, and that analysis latency stays in the recorded 12–22 s band.
3. **[S] Use the authoritative player color.** `player_color = ctx.session.turn` re-derives what `session.player_color` already holds; just read the field.

Supersedes/absorbs, now landed: "Experiment — skip react() for agent-parsed moves at verbosity=low" (react is gone), and the retry half of "Agent reliability — finish a game by voice" (an illegal-move guess now self-corrects inside the loop instead of ending the turn).

## Next sprint (2026-07-12)

The next three slices, in order. This is a framing block over the sections below, not a rewrite — every item stays tracked in its own section:

1. **[S] Ship Glitch's voice** — pick the final TTS voice by ear from the #97 audition artifact and resolve the swearing watch-item; closes the current-focus trio. Done when the winner is set as `TTS_VOICE`.
2. **[M] Structural fix for the destructive-op confirm gate** — a deterministic pipeline-owned "pending destructive op" state (bare "new game" → confirmation question → "yes" → `new_game`). Pre-step: measure `resign`'s adherence rate. Done when the `destructive_confirm` xfail flips to a hard assert. **Sequence after the agent-loop rework above** — this state lives in `_run_command`, which the rework rewrites; design it against the new single-loop pipeline, not the two-phase one.

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
