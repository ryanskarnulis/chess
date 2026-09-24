# CLAUDE.md

Guidance for Claude Code in this repo.

## What this is

A local-first, self-hosted chess app for the home network, played from any
browser. The core experience is a game against a tool-using AI agent (Glitch,
voice-first) that acts as opponent, interface, and game controller.

Layout: Python backend in `backend/` (src-layout package `chessapp`), React
web UI in `frontend/`, decision records in `docs/`. `BRIEF.md` is the design
reference. The llama-server flags live in the shared
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
CHESSAPP_AGENT_FRONTIER=1 pytest tests/test_agent_frontier.py -v -s   # frontier: scored, never gated (docs/agent-frontier.md)
python scripts/frontier_report.py trend                  # frontier history, per split
CHESSAPP_TRAJ_SEED=7 pytest tests/test_trajectories.py -k replay -s   # replay one composed walk
```

Frontend: `npm run lint`, `npm test`, `npm run build` from `frontend/`.

## Rules

- **Deterministic code owns game truth.** `python-chess` decides legality and
  holds state; Stockfish calculates; the model only routes language to tools
  and must never be able to corrupt the game. The app plays a full game with
  the LLM off (`CHESSAPP_AGENT=off`).
- **Code owns truth and safety; the model owns understanding.**
  Deterministic code decides legality, what the settings actually are, when a
  destructive op may run, and whether a claim in the commentary is backed by
  the board. Working out what the player *meant* is the model's job, and so
  is *saying* it: when the honesty guard cuts a claim, the narrator is asked
  to say it again with the true facts, never handed a canned line
  (`docs/planner-narrator.md`). So no regex fast paths or literal parsers
  for language, and when a guard fires on a correct answer, loosen the guard
  rather than script the answer. Glitch should feel alive, not canned.
- **Personality is tone only** — never move choice, difficulty, or settings.
  The global Glitch text is vendored from `../agent-standard/`; fix drift by
  re-copying, never by editing the copy.
- **Never feed model thought blocks back into history** — final answers only.
- **Backlog is GitHub issues** (`gh issue list`). File follow-up work as an
  issue, never in a markdown file; PRs close issues with `Closes #N`.
- **Git:** never commit to `main`. Branch → PR → squash-merge on green CI
  (`gh pr checks --watch`, then `gh pr merge --squash`).
- **Tests ship with changes.** The deterministic core stays thoroughly
  tested; agent behavior is tested at the tool boundary, never against live
  LLM output.
- **Eval gate:** a change to prompts, the model, or the agent loop runs the
  eval suite before merge and must not regress the recorded baseline
  (`docs/agent-evals.md`). Evals stay a manual local command — never in CI.
  `long_capture` is release-blocking: a change that sends it red does not
  merge. Never re-attempt tool-schema minimization on gemma-4-12b (stripping
  pydantic keys collapses `undo_and_replace`); re-test only if the brain model
  changes. Details: `docs/agent-evals.md` "Standing results".

## Debugging agent behavior

Set `CHESSAPP_TRACE_PATH` and every interaction appends one JSONL record:
utterance, route, tool trajectory, stop reason, FENs, guard decision, and
cost. It is the first thing to reach for when a turn misbehaves, and a traced
misfire is a ready-made eval scenario. The same file also holds `serving`
manifests (the weights, build and settings actually serving the app) and the
`speech`/`voice` records that follow one voice interaction end to end.
`docs/latency-measurement.md` has the record kinds, the clock rules and
`scripts/latency_report.py`. `docs/turn-coordinator.md` and
`docs/planner-narrator.md` explain the turn architecture;
`docs/persistence-and-identity.md` says what survives a restart
(`live.json`, `conversations.json`), which tools ask before they run, and what
a delegate can bind to (`version`, `game_id`, `Idempotency-Key`).
