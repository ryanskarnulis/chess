# Chess — Local Agent-First Chess App

A local-first, self-hosted chess app for the home network, playable from any browser. The core experience is a game against a **tool-using AI agent** that acts as opponent, interface, and game controller — voice-first, with selectable personalities.

## How it works

The agent is the orchestrator and personality, **not** the referee:

- **Deterministic code owns truth.** `python-chess` holds board state, validates every move, and keeps history. Its answers are final.
- **Stockfish is a calculation tool** — evaluation, candidate moves, and difficulty (`Skill Level` / `UCI_Elo`) — not the app's personality.
- **The LLM only routes natural language to tools.** It submits moves and the engine accepts or rejects; it reads state through tools. It cannot corrupt the game.
- The app plays a full game against Stockfish with the LLM switched off — the agent layer is an optional enhancement on top.

Everything runs in containers on your own hardware (llama.cpp serving Gemma, Speaches for voice, the app itself) — basic gameplay works fully offline.

## Status

Early scaffold — pre-MVP. See [`TODO.md`](TODO.md) for the backlog and [`DONE.md`](DONE.md) for progress. The full design rationale lives in [`BRIEF.md`](BRIEF.md).

Roadmap:

1. **MVP** — web board + python-chess + Stockfish + text commands to the agent
2. Voice (self-hosted STT/TTS)
3. Full personality roster + settings by natural speech
4. Physical board (Chessnut Move) — separate, walled-off project

## Repo layout

```
backend/     Python backend (chessapp): game engine core, tools, API, agent brain
frontend/    Web board UI (arrives in Phase 1)
BRIEF.md     Project brief — architecture, stack, phasing, risks
CLAUDE.md    Development process: TDD, git workflow, commands
TODO.md      Living backlog
DONE.md      Completion log
```

## Development

```bash
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -e .[dev]

pytest                                  # tests
ruff check . && ruff format --check .   # lint (what CI runs)
```

Development is test-driven (red → green → refactor) with an agile, phase-driven backlog. All changes land via feature-branch PRs that auto-merge once CI is green — see [`CLAUDE.md`](CLAUDE.md) for the full process.

## License

Personal, non-distributed home-network project (private repo). Key dependencies (python-chess, Stockfish) are GPL — a licensing pass is required before any public distribution. See the license note in `BRIEF.md`.
