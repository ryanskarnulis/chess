# CLAUDE.md

Guidance for Claude Code in this repo.

## What this is

A local-first, self-hosted chess app for the home network, played from any
browser. The core experience is a game against a tool-using AI agent (Glitch,
voice-first) that acts as opponent, interface, and game controller.

Layout: Python backend in `backend/` (src-layout package `chessapp`), React
web UI in `frontend/`, decision records in `docs/`. `BRIEF.md` is the design
reference. `TODO.md` is the backlog; `DONE.md` is the completion log (move a
finished item there with the date). The llama-server flags live in the shared
`../llama-swap/config.yaml`, not here.

## Commands

All Python commands run from `backend/`:

```bash
cd backend
source .venv/bin/activate && pip install -e .[dev]   # setup
pytest                                               # tests
ruff check . && ruff format --check .                # lint (what CI runs)
ruff format .                                        # auto-format

CHESSAPP_TRACE_PATH=/tmp/turns.jsonl chessapp        # trace every agent turn
CHESSAPP_AGENT_EVALS=1 pytest tests/test_agent_evals.py -v -s   # live-model evals (needs GPU)
```

Frontend: `npm run lint`, `npm test`, `npm run build` from `frontend/`.

## Rules

- **Deterministic code owns game truth.** `python-chess` decides legality and
  holds state; Stockfish calculates; the model only routes language to tools
  and must never be able to corrupt the game. The app plays a full game with
  the LLM off (`CHESSAPP_AGENT=off`).
- **Prefer code over prompts.** When correct behavior is derivable from state,
  derive it — a prompt rule holds ~half the time, a code rule always.
- **Personality is tone only** — never move choice, difficulty, or settings.
  The global Glitch text is vendored from `../agent-standard/`; fix drift by
  re-copying, never by editing the copy.
- **Never feed model thought blocks back into history** — final answers only.
- **Git:** never commit to `main`. Branch → PR → squash-merge on green CI
  (`gh pr checks --watch`, then `gh pr merge --squash`).
- **Tests ship with changes.** The deterministic core stays thoroughly
  tested; agent behavior is tested at the tool boundary, never against live
  LLM output.
- **Eval gate:** a change to prompts, the model, or the agent loop runs the
  eval suite before merge and must not regress the recorded baseline
  (`docs/agent-evals.md`). Evals stay a manual local command — never in CI.

## Debugging agent behavior

Set `CHESSAPP_TRACE_PATH` and every interaction appends one JSONL record:
utterance, route, tool trajectory, stop reason, FENs, guard decision, and
cost. It is the first thing to reach for when a turn misbehaves, and a traced
misfire is a ready-made eval scenario. `docs/turn-coordinator.md` and
`docs/planner-narrator.md` explain the turn architecture.
