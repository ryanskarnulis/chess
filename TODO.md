# TODO

The backlog, in priority order. One task = one vertical slice = one branch = one PR (TDD: failing test first). When a task is finished and merged, move its line to `DONE.md` with the merge date. Re-plan freely between slices — this file is the living backlog, not a contract.

## Personality & voice — Glitch (current focus)

The 2026-07-11 direction: one hand-dialed personality — **Glitch**, a gen-z Jarvis (fully casual, low-key troll, help stays real, swearing allowed) — instead of the selectable roster, plus a designed custom TTS voice ("laid-back young American, audible smirk") instead of stock Kokoro `af_heart`. Voice-option research (Kokoro blending / Chatterbox Turbo / Qwen3-TTS / NeuTTS Air, with the 12 GB VRAM constraint) is in Claude's project memory.

- [ ] **Watch — does Glitch actually swear?** Zero swears in the first two live games despite escalated authorization (#95). If real games stay PG, add a few-shot exchange to the tone block (one blunder → one sweary roast) — the next and probably last lever short of accepting PG-13.
- [ ] **Pick Glitch's final voice by ear**: listen through the audition artifact (13 samples + voice strings; link in the #97 PR conversation), set the winner as `CHESSAPP_TTS_VOICE` in docker-compose.yml, then `docker compose up -d --build app` (rebuild also picks up the Glitch prompt). Default meanwhile: `am_fenrir(2)+am_michael(1)`.
- [ ] **Spike — Qwen3-TTS voice design** (only if no Kokoro blend has enough smirk): trim shared llama-swap context to free ~2 GB VRAM, serve via vLLM-Omni, design the voice from the prose spec

## Agent reliability — finish a game by voice

From the 2026-07-10 agent deep dive: voice games die by attrition — an illegal-move guess ends the turn with no recovery, the brain's prompt is ~75% UI noise (`fens`/`dests`) that grows every ply, the system prompt gives zero speech→SAN guidance, and every plain move pays the full two-LLM-call round trip (~10–20s). Slices in priority order; the first three are small, independent, and low-risk.

- [ ] **Experiment — phase-1 sampling**: A/B a lower temperature (~0.2–0.6) for the tool-decision call only, keeping BRIEF sampling for commentary; record tool-call accuracy before changing any default.
- [ ] **Experiment — skip react() for agent-parsed moves at verbosity=low**: the fast path already makes parser-hit moves zero-LLM at low verbosity (#83); this would extend the canned confirmation to plain moves only the agent could parse. Check real-game frequency of agent-path plain moves first — may no longer be worth it.
- [ ] **Pending-proposal state for confirmations** (only if the dead-end recurs after #85): when the agent proposes a specific move and the player answers a bare "yes", only the #85 prompt rule guarantees the move gets played. A small pipeline-owned "pending proposed move" would make the confirmation itself deterministic ("yes" → make_move(pending)) — but it adds conversation state to the pipeline; hold unless live games show the prompt rule isn't enough.

## Phase 4 — Manual testing (walkthrough together, take notes)

No manual testing has happened yet — everything below has only been exercised by automated tests. Go through this list at the desk, note anything that feels wrong or needs to change, then turn the notes into new backlog items.

### Setup / stack
- [ ] `docker compose up` brings up all three containers (llama-server, Speaches, app) cleanly from cold
- [ ] App is reachable from another device on the home network (phone/laptop browser)
- [ ] Basic gameplay works fully offline (network unplugged)
- [ ] Full game against Stockfish with the LLM turned off (agent layer disabled)

### Core gameplay (board UI)
- [ ] Play a complete game start to checkmate; board renders correctly throughout
- [ ] Illegal moves on the board are rejected without corrupting state
- [ ] Special moves work end-to-end: castling (both sides), en passant, pawn promotion (incl. underpromotion)
- [ ] Check, checkmate, stalemate, and draw conditions are detected and surfaced in the UI
- [ ] Captured pieces display updates correctly

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
- [ ] Review panel: after game over, "Review game" shows per-color accuracy, classification counts, and flagged moves with better alternatives; panel resets on new game
- [ ] Review panel: "Review unavailable" (not a crash) when the engine is off, and the button allows a retry
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

- [ ] If repo goes public or account upgrades to Pro: enable branch protection (require `lint` + `test` checks) and native auto-merge

## Agent-standard migration (Phase 2 epic)

Tracked in `../agent-standard/AGENTS-MASTER-PLAN.md` §Phase 2 (chess migration epic). Slices 1-2 (env-var rename, layered personality) are done; remaining slices, low-risk first:

- [ ] **Tool registry decorator migration**: migrate the hand-written JSON-schema `Tool` dataclasses to the decorator/FastMCP pattern; `definitions()` output stays schema-equivalent before/after.
- [ ] **MCP server + `.mcp.json`**: stand up an MCP server off the tool registry, giving chess Claude Code tooling parity with PCC.
- [ ] **httpx `ChatProvider`**: replace the OpenAI SDK client with the httpx `ChatProvider`-style provider, keeping the `Brain` two-phase interface on top; fake-client tests move to the wire shape.
- [ ] **Conversation persistence + delegate REST endpoint**: implement the §5 contract — a conversation wraps the single active game session (transcript + tool trajectory persisted); fast-path move parsing keeps working inside delegate calls.
- [ ] **`agent:` block in `app.yaml`**.
- [ ] **Eval harness**: golden command→tool-call tasks (mirroring PCC's), gating future prompt/model changes.

## Backlog (no near-term timeline)

- [ ] GBNF grammar-constrained decoding fallback — **deferred, likely unneeded**: the live spike confirmed Gemma-4 (`UD-Q4_K_XL`) emits structured OpenAI tool calls natively. Revisit only if reliability degrades under load/longer prompts.
- [ ] Physical board (Chessnut Move — blocked on hardware purchase): verify motorized actuation is programmatically controllable **before any design work**; until then, `control_physical_board` tool seam only
