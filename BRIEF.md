# Project Brief: Local Agent-First Chess App

## Vision

A local-first, self-hosted, containerized chess app on the home network,
usable from any browser. The core experience is playing chess against a
tool-using AI agent that acts as opponent, interface, and game controller.
Voice is the primary input.

## Core architecture principle (most important)

The agent is the **orchestrator and personality, NOT the referee**:

- **Deterministic code owns truth.** Board state, legality, and history live
  in `python-chess`. Its answers are final.
- **Stockfish is a calculation tool** — evaluation, candidate moves,
  difficulty (`Skill Level` / `UCI_Elo`) — not the app's personality.
- **The agent routes natural language to tools.** It never decides legality
  in its head and never knows the board directly; it submits moves and reads
  state through tools, so the LLM cannot corrupt the game.
- The app plays a full game against Stockfish with the LLM off.

## Game loop

Agent-in-the-path, single pipeline: all input (voice/text/board) becomes a
string → agent → tool call(s) → deterministic engine executes → state
updates → agent reacts from the new state. One road in, one brain. An
utterance that is exactly one unambiguous legal move skips the LLM entirely
(the fast path). Details: `docs/turn-coordinator.md`, `docs/planner-narrator.md`.

## Runtime model (the "brain")

- **Model:** Gemma 4 12B — dense, native tool-calling, hybrid-thinking,
  fully GPU-offloadable on a 12GB card.
- **Serving:** llama.cpp with Unsloth's QAT GGUF, quant `UD-Q4_K_XL`
  (~6.7 GB). Never hand-roll a Q4_0 conversion. MTP speculative decoding on.
  Flags live in `../llama-swap/config.yaml` (shared GPU server).
- **Thinking as a lever:** OFF for move parsing and quick reactions, ON for
  analysis. Never feed thought blocks back into history.
- **Swappable brain module:** all model-specific logic sits behind
  `get_agent_response(board_state, command, transcript)`; nothing else knows
  which model or backend is behind it.

## Agent tools

- **Reads:** get_board_state, get_legal_moves, get_move_history,
  get_captured_pieces, describe_position (the board in words, for the phase
  that speaks and is handed no FEN), evaluate_position, get_best_moves
  (Stockfish MultiPV; also answers hint asks — hints are on-request, there is
  no hints mode)
- **Writes:** make_move, undo, new_game, resign, claim_draw (offered only
  while a claim is available), offer_draw (the engine's answer is a code-owned
  rule over Stockfish's number and the material, `docs/draw-offer.md`),
  save_game, resume_game, export_pgn
- **Settings:** set_difficulty, set_verbosity, set_voice_output
- **Output:** speak (TTS) + commentary text
- **Future seam:** control_physical_board

Tools are capabilities, not hardcoded behaviors — the agent maps free-form
commands to tools itself; ambiguous input earns a clarifying question.

## Features

Interactive board, legal-move validation, play vs. agent, Stockfish moves,
adjustable difficulty, move history, captured pieces, new game, undo, resign,
claimable draws, draw offers, save/resume, PGN export, whole-game review, analysis, hints
on request, explanations.

## Personality

One hand-dialed character: **Glitch** — a gen-z Jarvis; effortlessly
competent, casual, low-key troll-y; help stays real, swearing allowed. The
global layer is vendored from `../agent-standard/personality-global.md`;
chess adds a flavor layer. Personality affects tone only — never move
selection, difficulty, or any setting. Prompt lives in
`backend/src/chessapp/personality.py`.

## Voice

Self-hosted STT + TTS (Speaches + Kokoro via the shared `../speech/` stack).
Spoken moves and natural commands; voice output on/off, mute, talk
more/less. Local-only by decision — no browser Web Speech API
(`docs/voice-fast-path-evaluation.md`).

## Deployment

Containers on the home network (Docker Compose), multi-device via browser,
basic gameplay fully offline. CD deploys `main` to the home server.

## Phasing

1. **MVP** — board + engine + text agent — done 2026-07-06
2. **Voice** — done 2026-07-06
3. **Settings by speech + Glitch + custom voice** — done 2026-07-11
4. **Physical board (Chessnut Move)** — separate walled-off project, blocked
   on hardware. Verify motorized actuation is programmatically controllable
   before any design work; until then only the tool seam exists. Existing
   OSS (ChessnutPy, chessnutair) covers *sensing* boards; actuation is the
   unverified piece.

## Stack (chosen)

`python-chess` (truth + UCI bridge, GPL) · Stockfish (GPL) · Chessground
board UI (GPL) + React/Vite · Speaches + Kokoro (MIT) for voice · llama.cpp
via the shared llama-swap server. Glue is ours; everything else is assembled.

**License note:** the stack clusters around GPL — fine for a personal,
non-distributed home-network app. A licensing pass is required before any
public distribution.

## Watch-items

- Model tool-call reliability under quantization — native tool calls work
  today; GBNF constrained decoding is the fallback if that degrades.
- Prompt/KV growth in long games — keep per-turn prompts small.
- Physical-board actuation is unverified — don't let it shape architecture.
