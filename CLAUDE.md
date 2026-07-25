# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Status

Feature-complete core (as of 2026-07-11): board + deterministic engine, voice (STT/TTS), the Glitch agent, analysis/whole-game review, and the workspace delegate API + eval harness. `BRIEF.md` is the full project brief — the chosen stack, licensing analysis, LLM serving details (model + quant), difficulty-tier design, and agent tool list all live there and are authoritative. The llama-server *flags* are owned by the shared workspace `../llama-swap/config.yaml` (one GPU server serves chess and project-command-center); change them there, not here. Read it before making design decisions; this file holds only the always-needed rules. Monorepo layout: Python backend lives in `backend/` (src-layout package `chessapp`); the React/Vite web UI lives in `frontend/`.

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

CHESSAPP_TRACE_PATH=/tmp/turns.jsonl chessapp        # trace every agent turn
```

**Debugging agent behavior:** set `CHESSAPP_TRACE_PATH` and each command appends one JSONL record — the utterance, which of the three routes took it (`confirmation` / `fast_path` / `brain`), the full tool trajectory (name, args, result), the loop's `stop_reason`, and the FEN before/after. Off by default. It is the first thing to reach for when a turn misbehaves: the suite pins the tool boundary, not live behavior, so a trace is the only record of what actually happened. A traced misfire is also a ready-made eval scenario (it carries its own `fen_before` + `utterance`).

## Development Process (required)

- **TDD, strictly:** write the failing test first (red), then minimal code to pass (green), then refactor. No production code without a test that demanded it. The deterministic core (board truth, tools) gets exhaustive unit tests; agent behavior is tested at the tool boundary — never write tests that depend on live LLM output.
- **Eval gate:** before merging any prompt, model, or loop change, run the opt-in agent eval harness (`cd backend && CHESSAPP_AGENT_EVALS=1 .venv/bin/pytest tests/test_agent_evals.py -v -s`); the recorded baseline in `docs/agent-evals.md` must not regress (it's the only live-model test — default `pytest` skips it).
- **Agile:** the phases in BRIEF.md are the epics; work in small vertical slices tracked as GitHub issues, one slice = one branch = one PR.

## Git Workflow

- **Never commit or push directly to `main`.** For every change: create a branch (`feat/<slug>`, `fix/<slug>`, `chore/<slug>`, `docs/<slug>`), commit, push, and open a PR with `gh pr create`.
- PRs are **squash-merged once CI is green**. GitHub's native auto-merge and branch protection are unavailable (private repo on the Free plan), so after opening a PR run `gh pr checks --watch` and, when green, `gh pr merge --squash`. Never merge with failing or pending checks.
- CI (`.github/workflows/ci.yml`) runs three jobs on every PR and push to main: `lint` (ruff check + format-check), `test` (pytest), and `frontend` (npm lint → build → test). All three must pass; run them locally before pushing.

## What This Is

A local-first, self-hosted, containerized chess app for a home network, played from any browser. The core experience is playing against a tool-using AI agent (voice-first input) that acts as opponent, interface, and game controller.

## Core Architecture Principle (non-negotiable)

The agent is the **orchestrator and personality, NOT the referee**:

- **Deterministic code owns truth.** Board state, legal-move validation, and move history live in `python-chess`. Its answers are authoritative and final.
- **Stockfish is a calculation tool** (evaluation, MultiPV candidate moves), not the app's personality. Driven via python-chess's UCI support.
- **The agent never decides legality "in its head" and never knows the board directly** — it submits moves through tools and the engine accepts/rejects; it reads state through tools. The LLM must be unable to corrupt game state.
- The app must support a full game against Stockfish **with the LLM turned off** — the agent layer is an optional enhancement on top.

## Game Loop

Agent-in-the-path, single pipeline: all input (voice/text/board) becomes a string → agent → tool call(s) → deterministic engine executes → results go **back to the agent** → it acts again or answers. One road in, one brain, no dual-path race conditions.

**The deterministic turn coordinator owns the turn sequence** (`coordinator.py`, `docs/turn-coordinator.md`): `player_move → (observe) → engine_reply → (close)`, with explicit phases, a turn id, and rejection of any action that doesn't belong in the current phase (`TurnStateError`, a `ValueError` so the agent reads it as a tool result and the API answers 409). **The engine's reply is the coordinator's job and never a model-callable tool** — a model that could ask for the reply could also fail to, and the LLM-off game must play on regardless; the model gets the observation slots, never the wheel.

**The move flow is split, and the observe beat is free.** `make_move` applies the player's move only (`atomic_exchange=False`, how the app assembles it) and reports what it did — the piece taken, whether it checks — so the narration a turn produces is a reaction to a *verified player move*, never to a finished exchange. It costs nothing extra in wall clock because the engine starts computing the moment the move lands and the reaction runs while it does; the pipeline then collects the reply and appends a **deterministic** announcement (`api._reply_announcement`) rather than paying for a second narration to react to a reply Glitch hasn't seen. The beat is skippable by construction: at verbosity=low, or when the provider fails, one canned line covers the whole turn and a plain move stays zero-LLM. Any non-move mutation (undo, new game, resign, resume) abandons the open turn first — its position is the thing being replaced. MCP keeps the atomic exchange: same boundary, different sequencing owner, because a caller with no pipeline must not leave a game mid-turn.

The loop lives in the brain (`get_agent_response`), in the fleet's standard shape (`../agent-standard/STANDARD.md` §3): call the model with tools, append its turn, dispatch each call through the registry, append each result as a `role: "tool"` message, repeat. **One turn is two model phases** (`docs/planner-narrator.md`): that loop is the **planner**, and it runs on a compact, persona-free tool contract, because on a 12B a page of tone competes with the tool decision for attention. Its first turn that asks for no tools ends the loop and is an internal handoff note — the planner never speaks to the player. The **narrator** is one further call on the personality prompt, offered **no tools**, given the utterance, the turn's tool results and that note; its text is the commentary. Being structurally unable to act is what the old "react from new state, never the raw utterance" rule was protecting, and the phase boundary enforces it rather than a withheld utterance. A budget stop reaches no narrator: nothing verified came back to speak from. This is enforced, not just described: `tests/test_closing_pass.py` pins it route-by-route — on every route (brain loop, fast path, board drag) the model call that produces player-facing text carries no tools, and once the turn's mutation budget is spent (the phase machine's one player move, the command window's one destructive op) the tools the planner still holds cannot mutate.

Bounded by `max_iterations` (4) plus a separate, smaller correction budget for schema-level failures; `stop_reason` is `completed | max_iterations | correction_limit`. **Domain rejections are results, not corrections** — an illegal move comes back as a tool result the agent reads and corrects from, inside the same turn.

**The board controls are in the machine too** (`docs/turn-coordinator.md`). In agent mode a dragged move runs the fast path's beats through the same helper — the structured move (`e2e4`), never prose — so a drag-played game gets Glitch's reactions and its own trace route (`board`); the UI's new-game and resign buttons dispatch through the registry, so the *same* deterministic gate arms `ctx.pending` and the endpoint relays its question (409 + `confirm: true`, answered at `/api/game/confirm`) — one gate, one armed op, either surface. **Direct mode is deliberate and visible:** with no brain, `/api/game/move` answers exactly what it always did, and `/api/settings.agent_available` tells the UI to say so rather than let the player discover it.

The one commentary path outside the loop is the fast path: an utterance that is exactly one unambiguous legal move (`parse_move`) goes straight to `make_move` with no model call, so there is no planner turn for the narrator to close — `Brain.narrate` is that same narrator phase on its own, and on that route it *is* the observe beat (at verbosity=low a canned confirmation stands in for that too, making a plain move zero-LLM). An explicit resignation (`parse_resign`) is settled the same way — whether the player conceded is not a judgment call — and dispatches `resign` into the same confirmation gate, so a mis-parse costs a question, never a game.

**Commentary may not announce what didn't happen.** Whatever route produced it, the pipeline checks the closing text against the board before the player sees it (`honesty.claims_destructive_outcome`): commentary asserting the game ended or restarted, when no destructive tool succeeded and the board is still live, is replaced with the truth and the turn is traced as `guarded`. The model may neither *do* a destructive op unasked (the gate) nor *say* it did (the guard); live, it did the latter — *"Word. Game over."* on a live board.

## Entry Points

The board UI (`/`), the delegate API (`/api/agent`, still served but no longer advertised in `app.yaml`), and the **conductor handoff deep link**: the fleet's conductor sends a user here with what they said (`/?intent=let's+play+chess+as+black`). `App.tsx` scrubs the param on mount and feeds the intent to `sendCommand` — the same pipeline as the command box, so a handoff opens the session already acting on it. It is one utterance in, nothing more: the intent is capped, and scrubbing the URL first means a reload never replays it. The `open:` block in `app.yaml` is what tells conductor this link exists (`../agent-standard/app-yaml-open-block.md`).

## Binding Invariants (details in BRIEF.md)

- **Swappable brain module:** all model-specific logic lives behind one interface — `get_agent_response(board_state, command, transcript) → {text, tool_calls}`. Nothing else in the app may know which model/backend is behind it.
- **Agent tools are capabilities, not hardcoded behaviors** — the agent maps free-form commands to tools itself (no per-phrase branching); ambiguous input → clarifying question. The full tool list is in BRIEF.md; `control_physical_board` stays a future seam only.
- **Never make the model decide what deterministic state already knows.** A rule the prompt asks for is a rule the model follows ~half the time (the destructive-op gate's finding); a policy the code owns holds always. Undo's ply count and the confirm-before-`new_game` rule were both this bug. When a tool's correct behavior is derivable from the session, derive it — don't document it in the docstring and hope.
- **The brain is offered fewer tools than the registry holds.** It gets the board state in its prompt every turn, so `BOARD_STATE_TOOLS` (the pure reads) are registered and dispatchable but not *offered* to it — a call it already has the answer to only burns a round trip out of `max_iterations`. Callers with no such injection (MCP, the delegate wire) still see the full list.
- **Personality is tone only:** it shapes commentary, never move choice, difficulty, or any other setting.
- **Never leave an engine unconfigured:** a real default difficulty tier is applied at app assembly (Stockfish's own default is full strength).
- **Local-only voice by decision** — no browser Web Speech API path (see `docs/voice-fast-path-evaluation.md`).
- **Never feed LLM thought blocks back into multi-turn history** — final answers only.

## Phasing

1. **MVP:** web board + python-chess + Stockfish + **text** commands to the agent. Get the tool boundaries right with text first.
2. Voice (STT/TTS).
3. Settings-by-voice + the single dialed-in personality (Glitch) and a custom voice.
4. Physical board (Chessnut Move) — walled-off separate project; do not let it shape core architecture.
