# Project Brief: Local Agent-First Chess App

## Vision
A local-first, self-hosted, containerized chess app running on a home network, usable from any browser (phone/laptop/desktop). The core experience is playing chess against a **tool-using AI agent** that acts as opponent, interface, and game controller. Voice is the primary input.

## Core Architecture Principle (most important)
The agent is the **orchestrator and personality, NOT the referee.**
- **Deterministic code owns truth.** Board state, legal-move validation, and move history live in code (use `python-chess`). Its answers are authoritative and final.
- **Stockfish is a calculation tool**, not the app's personality. Used for evaluation and candidate moves only.
- **The agent routes natural language to tools.** It never decides legality "in its head" — it submits a move and the engine accepts/rejects. It never knows the board — it calls a read tool. This makes it impossible for the LLM to corrupt the game state.

The app must support a full game against Stockfish with the LLM turned off — the agent layer sits on top as an optional enhancement.

## Game Loop
- **Agent-in-the-path** (single pipeline): all input (voice/text/board) becomes a string → agent → tool call(s) → deterministic engine executes → state updates → agent reacts.
- One road in, one brain, no dual-path race conditions.
- The agent's reaction step reads from **current game state** ("here's the new board + what changed"), not from "the user said e4" — this keeps a future fast-parse optimization free to add later.

## Runtime Model (the "brain")
- **Model:** Gemma 4 12B (dense, native tool-calling, hybrid-thinking, multimodal/vision + experimental audio encoder). Fully GPU-offloadable on a 12GB card; faster than the 26B MoE (~94 tok/s on tool calls with MTP) at the cost of some capability.
- **Serving:** llama.cpp with the **Unsloth QAT GGUF**, quant `UD-Q4_K_XL` (`unsloth/gemma-4-12B-it-qat-GGUF`). ~6.7 GB. Do NOT hand-roll a Q4_0 conversion (scale mismatch tanks accuracy); use Unsloth's dynamic quant.
- **Speed:** enable MTP (Multi-Token Prediction) speculative decoding — 1.4–2.2x faster, no accuracy loss, drafter auto-discovered from `-hf`.
- **Flags:** `--jinja` (required for tool-calling/chat template); sampling `--temp 1.0 --top-p 0.95 --top-k 64`.
- **Thinking mode as a lever:** OFF for fast move parsing/quick reactions; ON for analysis ("what was my mistake"). Toggle via `enable_thinking`. In multi-turn history, keep only final answers — never feed thought blocks back.
- **Swappable brain module:** isolate all model-specific logic behind one interface — `get_agent_response(board_state, command, transcript) → {text, tool_calls}` (the transcript param arrived with conversation memory, 2026-07-07). Nothing else in the app should know which model/backend is behind it. Verify at build time that the serving stack parses Gemma 4 tool calls into a structured array; fall back to prompt-engineered parsing only if runner support lags.

## Agent Tools (capabilities, not hardcoded behaviors)
- **Reads:** get_board_state, get_legal_moves, get_move_history, get_captured_pieces, evaluate_position (Stockfish), get_best_moves (Stockfish MultiPV)
- **Writes/actions:** make_move (returns legal/illegal), undo, new_game, resign, save_game, resume_game, export_pgn
- **Settings:** set_difficulty, set_verbosity, set_hints_mode, set_voice_output
- **Output:** speak (TTS) + returned commentary text
- **Future:** control_physical_board

The agent maps free-form commands to tools itself (no per-phrase branching). Ambiguous input → it asks a clarifying question (also a tool response).

## Chess Features
Interactive board, legal-move validation, play vs. agent, Stockfish-powered moves, adjustable difficulty (Stockfish `Skill Level` / `UCI_Elo`), move history, captured pieces, new game, undo, resign, save/resume, export PGN, game review, basic analysis, hints, explanations.

## Personality
One personality, dialed in by hand: **Glitch** — a gen-z Jarvis; effortlessly competent, fully casual, low-key troll-y (understated needles, dry roasts, fake sympathy, long-game callbacks; help stays real, swearing allowed). Decided 2026-07-11: the original selectable eight-personality roster was collapsed into this one character and `set_personality` removed — one voice done exactly right beat eight approximations. Personality affects tone/teaching/reactions **only** — never move selection, difficulty, or any other setting (decided 2026-07: a move-bias layer was tried and removed because it bypassed the difficulty setting). The prompt lives in `backend/src/chessapp/personality.py`.

## Voice
- STT + TTS. Speak moves ("pawn to e4", "castle kingside") and natural commands ("make it easier", "talk less", "give me a hint", "what was my mistake").
- Options: voice output on/off, mute, talk more/less, hints freely vs. on-request.
- Privacy note: browser Web Speech API is the fast path but some browsers send audio to the cloud; prefer local Whisper if strict privacy is required.

## Deployment
- Runs in containers on the home network, multi-device via browser.
- Basic chess gameplay works fully offline / no public internet. Fast and private.

## Phasing
1. **MVP:** web board + python-chess + Stockfish + **text** commands to the agent, deterministic engine underneath, 1–2 personalities. Get the tool boundaries right with text first.
2. Add voice (STT/TTS).
3. Settings-by-voice + dial in the single personality and a custom voice (originally "remaining personalities"; the roster was collapsed to Glitch, 2026-07).
4. **Physical board (own project, Phase 3+):** Chessnut Move. Wall this off — robotic board control depends on whatever API/BLE the vendor exposes, which is often limited/undocumented. Verify actual programmatic control before designing around it. Leave a clean `control_physical_board` tool seam only. Future: detect board moves, drive opponent pieces, keep digital/physical synced, handle desync, agent narrates. Gemma 4 vision could later read the board from a camera.

## Reuse / OSS Components (assemble first, build glue only)
Guiding rule: use existing open source wherever possible. Roughly ~80% of the stack exists; what's left to write is the glue (agent brain module, tool definitions, game-loop wiring).

- **Chess truth + Stockfish bridge:** `python-chess` (niklasf). One library = move generation, legal-move validation, PGN read/write, AND UCI engine communication (drives Stockfish directly). This IS the deterministic core + the Stockfish tool. License: GPL-3.0. NOTE: ignore the unrelated PyPI package named "Chessnut" (a ~200-line toy model, no engine, unrelated to the hardware).
- **Stockfish:** the engine binary, GPL. Driven via python-chess UCI. Use `Skill Level` / `UCI_Elo` for difficulty, MultiPV for candidate moves (analysis/hints only).
- **Web board UI (pick one):**
  - `Chessground` (Lichess) — most featureful (fast DOM-diff, SVG arrows/shapes, no deps). GPL-3.0 → copyleft: fine for personal self-hosted use, but distributing the combined work forces GPL + source release. React wrapper exists.
  - `react-chessboard` (Clariity) — **MIT**, pairs with `chess.js`. Prefer this if permissive licensing matters.
  - `@mdwebb/react-chess` — chessground + chess.js bundled with game scaffolding worth reusing: PGN w/ annotations, move history, promotion dialog, keyboard nav, callbacks (onCheck, onGameOver, onIllegalMove, onPromotion).
  - `chess.js` — frontend move logic/validation (MIT) if using a headless board.
- **Voice (biggest win — ~one container):** `Speaches` — MIT, self-hosted, **OpenAI-API-compatible** STT+TTS server (faster-whisper for STT, Kokoro/Piper for TTS), with a realtime WebSocket API for two-way voice. Collapses the whole voice layer into one Docker service speaking the same API dialect as the LLM. Lighter à-la-carte alternative: `whisper.cpp` (STT) + `Piper` or `Kokoro-82M` (TTS, runs on CPU / ~2–3GB).
- **Agent orchestration (may need no framework):** `llama-server` exposes a fully OpenAI-compatible REST API → the agent loop can just be the OpenAI SDK pointed at localhost with a `tools` parameter (model requests tool → execute → return result → respond). (Superseded 2026-07-11: the brain speaks the OpenAI wire format over plain `httpx`, no SDK — see `chessapp/provider.py`; only `voice.py`'s Speaches client still rides the SDK, tracked in TODO.) For structural tool-call reliability, use **GBNF grammar-constrained decoding** so tool calls are valid by construction (technique demoed by `llama-cpp-agent`; note that repo is now unmaintained → adopt the technique, don't marry the dep). `LangGraph` available if a formal state-machine loop is wanted (likely overkill for one game loop).
- **Game review / analysis:** fork an existing Stockfish-based review engine (FastAPI + React, classifies moves Brilliant→Blunder with accuracy, top moves, centipawn analysis) rather than building from scratch.
- **Physical board (Phase 3):** `ChessnutPy` (staubsauger) + reverse-engineered BLE protocol (`rmarabini/chessnutair`) — already connects Chessnut eboards over Bluetooth, plays vs Stockfish/UCI, validates board state, and implements desync recovery ("fix-board"). CAVEAT: targets the *sensing* boards (Air/Pro/Air+). The Chessnut **Move** adds motorized actuation — piece *detection* is largely solved in OSS; driving the pieces is the remaining unverified piece.
- **Containerization:** each layer is already its own container (llama-server, Speaches, the app) → tie together with Docker Compose.

### License note (cross-cutting)
Key pieces cluster around **GPL** (python-chess, Chessground, Stockfish, ChessnutPy). For a personal, non-distributed home-network app this imposes essentially nothing. It only bites if the combined work is ever distributed publicly (then all must be GPL + open-sourced). If avoiding that matters, choose MIT pieces early (react-chessboard + chess.js) — easier now than later.

## Key Risks / Watch-Items
- Model tool-call reliability under quantization → prefer GBNF grammar-constrained decoding (valid-by-construction); otherwise defensive parser + retry loop, validate against tool schema.
- KV cache / context memory growth → keep prompts small (board state + short command); budget headroom.
- Physical board: piece *detection* is de-risked by existing OSS (Chessnut sensing boards); motorized *actuation* on the Chessnut Move is unverified → do not let it shape core architecture; verify before designing around it.
- Licensing: GPL cluster is fine for personal use; revisit before any public distribution.
