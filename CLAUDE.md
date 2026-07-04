# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Status

Early scaffold. `BRIEF.md` is the full project brief — read it before making design decisions; this file summarizes its binding decisions. Monorepo layout: Python backend lives in `backend/` (src-layout package `chessapp`); `frontend/` will be added in Phase 1.

## Task Tracking

`TODO.md` is the living backlog (prioritized, one task = one branch = one PR). `DONE.md` is the completion log. When picking up work, take the next relevant task from TODO.md; when it's merged, move the line to DONE.md under the current date. Re-plan TODO.md freely between slices.

## Commands

All Python commands run from `backend/`:

```bash
cd backend
source .venv/bin/activate && pip install -e .[dev]   # setup
pytest                                               # run tests
pytest tests/test_smoke.py -k <name>                 # single file / test
ruff check . && ruff format --check .                # lint (what CI runs)
ruff format .                                        # auto-format
```

## Development Process (required)

- **TDD, strictly:** write the failing test first (red), then minimal code to pass (green), then refactor. No production code without a test that demanded it. The deterministic core (board truth, tools) gets exhaustive unit tests; agent behavior is tested at the tool boundary — never write tests that depend on live LLM output.
- **Agile:** the phases in BRIEF.md are the epics; work in small vertical slices tracked as GitHub issues, one slice = one branch = one PR.

## Git Workflow

- **Never commit or push directly to `main`.** For every change: create a branch (`feat/<slug>`, `fix/<slug>`, `chore/<slug>`, `docs/<slug>`), commit, push, and open a PR with `gh pr create`.
- PRs are **squash-merged once CI is green**. GitHub's native auto-merge and branch protection are unavailable (private repo on the Free plan), so after opening a PR run `gh pr checks --watch` and, when green, `gh pr merge --squash`. Never merge with failing or pending checks. (If the repo ever goes public or Pro, switch to `gh pr merge --auto --squash` + branch protection requiring the `lint` and `test` checks.)
- CI (`.github/workflows/ci.yml`) runs ruff lint/format-check and pytest on every PR and push to main. Both jobs must pass; run them locally before pushing.

## What This Is

A local-first, self-hosted, containerized chess app for a home network, played from any browser. The core experience is playing against a tool-using AI agent (voice-first input) that acts as opponent, interface, and game controller.

## Core Architecture Principle (non-negotiable)

The agent is the **orchestrator and personality, NOT the referee**:

- **Deterministic code owns truth.** Board state, legal-move validation, and move history live in `python-chess`. Its answers are authoritative and final.
- **Stockfish is a calculation tool** (evaluation, MultiPV candidate moves), not the app's personality. Driven via python-chess's UCI support.
- **The agent never decides legality "in its head" and never knows the board directly** — it submits moves through tools and the engine accepts/rejects; it reads state through tools. The LLM must be unable to corrupt game state.
- The app must support a full game against Stockfish **with the LLM turned off** — the agent layer is an optional enhancement on top.

## Game Loop

Agent-in-the-path, single pipeline: all input (voice/text/board) becomes a string → agent → tool call(s) → deterministic engine executes → state updates → agent reacts. One road in, one brain, no dual-path race conditions. The agent's reaction step reads from **current game state** ("new board + what changed"), not from the raw user utterance — this keeps a future fast-parse optimization free to add.

## The Brain (LLM layer)

- Model: Gemma 4 26B A4B (MoE), served by llama.cpp using the **Unsloth QAT GGUF** quant `UD-Q4_K_XL` (`unsloth/gemma-4-26B-A4B-it-qat-GGUF`). Do NOT hand-roll a Q4_0 conversion.
- llama-server flags: `--jinja` (required for tool-calling), sampling `--temp 1.0 --top-p 0.95 --top-k 64`; enable MTP speculative decoding.
- Thinking mode: OFF for fast move parsing/reactions, ON for analysis. Never feed thought blocks back into multi-turn history — final answers only.
- **Swappable brain module:** all model-specific logic lives behind one interface — `get_agent_response(board_state, command) → {text, tool_calls}`. Nothing else in the app may know which model/backend is behind it.
- Tool-call reliability under quantization: prefer GBNF grammar-constrained decoding (valid-by-construction tool calls); otherwise a defensive parser + retry loop validated against the tool schema.
- Keep prompts small (board state + short command) to control KV-cache growth.

## Agent Tools

Capabilities, not hardcoded behaviors — the agent maps free-form commands to tools itself (no per-phrase branching); ambiguous input → clarifying question.

- Reads: `get_board_state`, `get_legal_moves`, `get_move_history`, `get_captured_pieces`, `evaluate_position`, `get_best_moves` (Stockfish MultiPV)
- Writes: `make_move` (returns legal/illegal), `undo`, `new_game`, `resign`, `save_game`, `resume_game`, `export_pgn`
- Settings: `set_difficulty`, `set_personality`, `set_verbosity`, `set_hints_mode`, `set_voice_output`
- Output: `speak` (TTS) + returned commentary text
- Future seam only: `control_physical_board`

## Phasing

1. **MVP:** web board + python-chess + Stockfish + **text** commands to the agent, 1–2 personalities. Get the tool boundaries right with text first.
2. Voice (STT/TTS).
3. Remaining personalities + settings-by-voice.
4. Physical board (Chessnut Move) — walled-off separate project. Do not let it shape core architecture; motorized actuation is unverified. Leave only the `control_physical_board` tool seam.

## Chosen Stack (assemble first, build glue only)

- Chess truth + Stockfish bridge: `python-chess` (GPL-3.0). Ignore the unrelated PyPI package named "Chessnut" (a toy, not the hardware).
- Difficulty: Stockfish `Skill Level` / `UCI_Elo`; personality may bias move choice among MultiPV candidates but Stockfish guarantees legal, reasonable moves.
- Web board: `Chessground` (GPL) or `react-chessboard` + `chess.js` (MIT — prefer if permissive licensing matters); `@mdwebb/react-chess` has reusable game scaffolding.
- Voice: `Speaches` (MIT) — one self-hosted container, OpenAI-API-compatible STT+TTS. Lighter alternative: whisper.cpp + Piper/Kokoro.
- Agent orchestration: likely no framework — llama-server is OpenAI-compatible, so the loop is just the OpenAI SDK pointed at localhost with `tools`. Adopt the GBNF technique from `llama-cpp-agent` but don't depend on that unmaintained repo. LangGraph is likely overkill.
- Game review: fork an existing Stockfish-based review engine rather than building from scratch.
- Deployment: one container per layer (llama-server, Speaches, app) tied together with Docker Compose. Basic gameplay must work fully offline.

## Licensing

The key pieces cluster around GPL (python-chess, Chessground, Stockfish, ChessnutPy) — fine for a personal, non-distributed home-network app. It only bites on public distribution. If avoiding GPL matters, choose the MIT pieces (react-chessboard + chess.js) early.
