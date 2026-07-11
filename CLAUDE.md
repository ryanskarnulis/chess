# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Status

Early scaffold. `BRIEF.md` is the full project brief — the chosen stack, licensing analysis, LLM serving details (model + quant), difficulty-tier design, and agent tool list all live there and are authoritative. The llama-server *flags* are owned by the shared workspace `../llama-swap/config.yaml` (one GPU server serves chess and project-command-center); change them there, not here. Read it before making design decisions; this file holds only the always-needed rules. Monorepo layout: Python backend lives in `backend/` (src-layout package `chessapp`); `frontend/` will be added in Phase 1.

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
- PRs are **squash-merged once CI is green**. GitHub's native auto-merge and branch protection are unavailable (private repo on the Free plan), so after opening a PR run `gh pr checks --watch` and, when green, `gh pr merge --squash`. Never merge with failing or pending checks.
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

Agent-in-the-path, single pipeline: all input (voice/text/board) becomes a string → agent → tool call(s) → deterministic engine executes → state updates → agent reacts. One road in, one brain, no dual-path race conditions. The agent's reaction step reads from **current game state** ("new board + what changed"), not from the raw user utterance.

## Binding Invariants (details in BRIEF.md)

- **Swappable brain module:** all model-specific logic lives behind one interface — `get_agent_response(board_state, command) → {text, tool_calls}`. Nothing else in the app may know which model/backend is behind it.
- **Agent tools are capabilities, not hardcoded behaviors** — the agent maps free-form commands to tools itself (no per-phrase branching); ambiguous input → clarifying question. The full tool list is in BRIEF.md; `control_physical_board` stays a future seam only.
- **Personality is tone only:** it shapes commentary, never move choice, difficulty, or any other setting.
- **Never leave an engine unconfigured:** a real default difficulty tier is applied at app assembly (Stockfish's own default is full strength).
- **Local-only voice by decision** — no browser Web Speech API path (see `docs/voice-fast-path-evaluation.md`).
- **Never feed LLM thought blocks back into multi-turn history** — final answers only.

## Phasing

1. **MVP:** web board + python-chess + Stockfish + **text** commands to the agent, 1–2 personalities. Get the tool boundaries right with text first.
2. Voice (STT/TTS).
3. Remaining personalities + settings-by-voice.
4. Physical board (Chessnut Move) — walled-off separate project; do not let it shape core architecture.
