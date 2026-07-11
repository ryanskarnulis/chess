# TODO

The backlog, in priority order. One task = one vertical slice = one branch = one PR (TDD: failing test first). When a task is finished and merged, move its line to `DONE.md` with the merge date. Re-plan freely between slices — this file is the living backlog, not a contract.

## Agent reliability — finish a game by voice (top priority)

From the 2026-07-10 agent deep dive: voice games die by attrition — an illegal-move guess ends the turn with no recovery, the brain's prompt is ~75% UI noise (`fens`/`dests`) that grows every ply, the system prompt gives zero speech→SAN guidance, and every plain move pays the full two-LLM-call round trip (~10–20s). Slices in priority order; the first three are small, independent, and low-risk.

- [ ] **Agent-facing board state view**: stop passing the UI `_state_dict` to the brain. Build a purpose-made view — fen, turn, player color, in-check, SAN history, captured pieces, legal moves, game_over/outcome — with no `fens` and no `dests`, and use it in both `get_agent_response` and `react`. (Measured ~80% prompt cut at ply 40; bounds late-game growth, the BRIEF's KV-cache watch-item.)
- [ ] **Domain-retry loop for failed tool calls**: when a dispatched call comes back `ok: false` — or `make_move` returns `legal: false` — feed the error (plus the current legal-move list for moves) back to the brain and let it retry, bounded (~2 rounds), before falling through to react. Generalizes the existing schema-retry loop in `LlamaBrain`; TDD with a fake client, never live LLM output.
- [ ] **System prompt hot-path rewrite**: add speech→tool few-shot examples ("pawn to e4" → make_move("e4"), "castle kingside" → "O-O", promotion syntax), tell the agent to pick its move string from the provided legal moves, one `make_move` per player turn (the engine replies automatically — never move for it), state which color the player is, require a confirmation question before `resign`/`new_game`, and warn that voice transcripts may be mangled ("e 4", "night to f3").
- [ ] **STT hardening**: pass a chess-vocabulary `prompt` to the whisper transcription call (`SpeechClient.transcribe`) to bias recognition; add a small deterministic transcript normalizer ("e 4" → "e4", "night" → "knight") before the text enters the command pipeline.
- [ ] **Deterministic fast-parse path for plain moves** (the seam BRIEF reserves): parse unambiguous move utterances ("e4", "knight to f3", "castle kingside", "takes on d5") straight to `make_move` with no phase-1 LLM call; the agent still reacts from the new state (or a canned confirmation at verbosity=low). Anything ambiguous or non-move falls through to the agent unchanged — still one road in, one pipeline.
- [ ] **Experiment — phase-1 sampling**: A/B a lower temperature (~0.2–0.6) for the tool-decision call only, keeping BRIEF sampling for commentary; record tool-call accuracy before changing any default.
- [ ] **Experiment — skip react() for plain confirmed moves at verbosity=low**: halves LLM calls on the hot path; measure per-turn latency before/after and keep only if commentary quality holds.

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

### Personalities & settings
- [ ] Each personality is distinguishable in commentary (tone only — personalities must NOT change move strength or settings)
- [ ] `set_difficulty` visibly changes engine strength (easy loses to a casual player, hard doesn't); the UI difficulty selector matches what the engine actually plays, including after an app restart
- [ ] Settings by speech/text: difficulty, personality, verbosity, hints mode, voice output all switchable mid-game and persist
- [ ] Verbosity levels actually differ (terse vs chatty)

### Voice (STT/TTS)
- [ ] Voice input: spoken moves transcribe and execute correctly (test noisy vs quiet room)
- [ ] STT latency (~2.5s known) is acceptable in the play loop
- [ ] TTS output (`speak`) is intelligible, correct voice, reasonable latency
- [ ] `set_voice_output` on/off actually mutes/unmutes TTS
- [ ] Misrecognized speech fails safe (clarifying question, not a wrong move)

## Infrastructure / process (ongoing)

- [ ] If repo goes public or account upgrades to Pro: enable branch protection (require `lint` + `test` checks) and native auto-merge

## Backlog (no near-term timeline)

- [ ] GBNF grammar-constrained decoding fallback — **deferred, likely unneeded**: the live spike confirmed Gemma-4 (`UD-Q4_K_XL`) emits structured OpenAI tool calls natively. Revisit only if reliability degrades under load/longer prompts.
- [ ] Physical board (Chessnut Move — blocked on hardware purchase): verify motorized actuation is programmatically controllable **before any design work**; until then, `control_physical_board` tool seam only
