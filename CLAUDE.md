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
CHESSAPP_CONTEXT_PATH=/tmp/ctx.jsonl chessapp         # exact bytes of every model call
python scripts/watch_context.py /tmp/ctx.jsonl --trace /tmp/turns.jsonl   # watch them live
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
- **Code owns actions; the model owns understanding and speech.**
  Deterministic code decides legality, what the settings actually are, and
  when a destructive op may run, and enforces it inside the tools, which
  refuse a bad call and say how to fix it. Working out what the player
  *meant* is the model's job, and so is *saying* what happened: code never
  cuts, rewrites or scripts Glitch's words. When he gets something wrong, fix
  what he was shown (context, history, tool results, prompts) and measure it
  as speech accuracy, offline. So no regex fast paths or literal parsers for
  language, and no new speech guards. The honesty guard still runs until
  #368 retires it, so don't extend it. Glitch should feel alive, not canned.
- **Personality is tone only** — never move choice, difficulty, or settings.
  The global Glitch text is vendored from `../agent-standard/`; fix drift by
  re-copying, never by editing the copy.
- **Never feed model thought blocks back into history** — final answers only.
- **Backlog is GitHub issues** (`gh issue list`). File follow-up work as an
  issue, never in a markdown file; PRs close issues with `Closes #N`. The
  pinned roadmap #366 orders the work: its sub-issues are the steps in order,
  so take the first open one unless told otherwise. `P1`–`P3` labels are
  rough priority, not the order.
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
misfire is a ready-made eval scenario. To see exactly what a model call was
shown and what it said back (request, rendered prompt, raw response), set
`CHESSAPP_CONTEXT_PATH` and run `scripts/watch_context.py`
(`docs/context-capture.md`, which also covers the deployed container); it is
heavy, so leave it off otherwise. The trace
file also holds `serving` manifests (the weights, build and settings actually serving the app) and the
`speech`/`voice` records that follow one voice interaction end to end.
`docs/latency-measurement.md` has the record kinds, the clock rules and
`scripts/latency_report.py`. `docs/turn-coordinator.md` and
`docs/planner-narrator.md` explain the turn architecture;
`docs/persistence-and-identity.md` says what survives a restart
(`live.json`, `conversations.json`), which tools ask before they run, and what
a delegate can bind to (`version`, `game_id`, `Idempotency-Key`).
