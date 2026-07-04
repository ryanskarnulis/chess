# TODO

The backlog, in priority order. One task = one vertical slice = one branch = one PR (TDD: failing test first). When a task is finished and merged, move its line to `DONE.md` with the merge date. Re-plan freely between slices — this file is the living backlog, not a contract.

## Phase 1 — MVP (web board + text agent)

### Epic: Deterministic core (python-chess owns truth)

- [ ] Save game / resume game (serialize session to disk)
- [ ] Export PGN
- [ ] Stockfish bridge via python-chess UCI: engine move at configurable difficulty (`Skill Level` / `UCI_Elo`)
- [ ] Stockfish analysis tools: `evaluate_position`, `get_best_moves` (MultiPV)
- [ ] Full offline game vs Stockfish with no LLM in the loop (acceptance test for the brief's core requirement)

### Epic: Tool layer (agent-facing capability surface)

- [ ] Tool registry + JSON schemas for all read tools: `get_board_state`, `get_legal_moves`, `get_move_history`, `get_captured_pieces`, `evaluate_position`, `get_best_moves`
- [ ] Write tools: `make_move` (returns legal/illegal), `undo`, `new_game`, `resign`, `save_game`, `resume_game`, `export_pgn`
- [ ] Settings tools: `set_difficulty`, `set_personality`, `set_verbosity`, `set_hints_mode`, `set_voice_output`
- [ ] Tool dispatch: name + args in → validated, deterministic result out (exhaustive unit tests; this is the boundary the LLM cannot corrupt)

### Epic: API layer

- [ ] FastAPI app: game session lifecycle endpoints + state fetch
- [ ] WebSocket (or SSE) channel for state updates to the board UI
- [ ] Text command endpoint: user string in → agent pipeline → tool calls → new state + commentary out

### Epic: Agent brain (swappable module)

- [ ] Brain interface: `get_agent_response(board_state, command) → {text, tool_calls}` — nothing outside knows the backend
- [ ] llama-server client via OpenAI SDK (`tools` param), pointed at localhost
- [ ] Verify llama.cpp parses Gemma 4 tool calls into a structured array; if not, GBNF grammar-constrained decoding fallback
- [ ] Defensive tool-call parser + retry loop, validated against tool schemas
- [ ] Agent game loop: input → agent → tool call(s) → engine executes → agent reacts from *new state* (not from the raw utterance)
- [ ] Clarifying-question path for ambiguous input
- [ ] 1–2 personalities (system-prompt level): friendly rival + one more
- [ ] Fake/scripted brain implementation for tests (no live LLM in the test suite)

### Epic: Frontend (web board)

- [ ] Scaffold React app in `frontend/` (Vite); decide Chessground vs react-chessboard (licensing note in BRIEF.md)
- [ ] Interactive board wired to backend state: render position, submit moves by drag/click, illegal-move feedback
- [ ] Move history + captured pieces panels
- [ ] Game controls: new game, undo, resign, difficulty
- [ ] Text command box → agent endpoint, commentary display
- [ ] Frontend CI job (lint + build + tests)

### Epic: Deployment

- [ ] Backend Dockerfile
- [ ] docker-compose.yml: app + llama-server (+ Speaches placeholder), home-network config
- [ ] Verify fully-offline basic gameplay in containers

## Phase 2 — Voice

- [ ] Speaches container (OpenAI-compatible STT+TTS) in compose
- [ ] STT path: mic in browser → transcription → same text pipeline
- [ ] TTS path: `speak` tool output → audio in browser
- [ ] Voice options: output on/off, mute, talk more/less
- [ ] Evaluate Web Speech API fast path vs local-only privacy mode

## Phase 3 — Full personality & settings

- [ ] Remaining personalities (calm coach, trash-talker, grandmaster, villain, silent assassin, beginner bot, streamer)
- [ ] Personality-biased move selection among Stockfish MultiPV candidates (legality still engine-guaranteed)
- [ ] Settings by natural speech (difficulty, personality, verbosity, hints mode)
- [ ] Hints, "what was my mistake" explanations (thinking mode ON for analysis)
- [ ] Game review: fork/integrate an existing Stockfish-based review engine (move classification, accuracy)

## Phase 4 — Physical board (walled off; separate project)

- [ ] Verify Chessnut Move motorized actuation is programmatically controllable **before any design work**
- [ ] `control_physical_board` tool seam only until then

## Infrastructure / process (ongoing)

- [ ] If repo goes public or account upgrades to Pro: enable branch protection (require `lint` + `test` checks) and native auto-merge
