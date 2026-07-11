# Chess project guidance

## Start here

- Read `BRIEF.md` before making design decisions.
- Treat `TODO.md` as the prioritized backlog and `DONE.md` as the completion
  log. Work in small vertical slices.
- This is a monorepo: the Python backend is in `backend/` and the React/Vite
  frontend is in `frontend/`.

## Environment setup

```bash
# Backend
cd backend
python3 -m venv .venv
. .venv/bin/activate
pip install -e '.[dev]'

# Frontend
cd ../frontend
npm install
```

The LLM brain is served by the workspace-level `../llama-swap/` stack (shared
GPU llama.cpp, outside this repo); the compose stack here includes optional
voice services. Do not require the brain or voice for normal development,
testing, or Cloud tasks: engine-only gameplay and the automated test suites
must work without them.

## Verification

Run the relevant checks before handing off work:

```bash
# Backend (from backend/)
. .venv/bin/activate
pytest
ruff check .
ruff format --check .

# Frontend (from frontend/)
npm run build
npm test
```

Use focused tests while iterating, then run the applicable full suite. To
format Python code intentionally, run `ruff format .`.

## Development process

- Follow TDD strictly: write a failing test, implement the smallest passing
  change, then refactor.
- Never add production behavior without a test that requires it.
- Test deterministic chess behavior exhaustively. Test agent behavior at the
  tool boundary; never depend on live LLM output in tests.
- Keep changes narrowly scoped to the requested backlog item or task.

## Architecture invariants

- Deterministic code is the referee. `python-chess` owns board state, legal
  move validation, and history; its answers are final.
- Stockfish is a calculation tool, not the app personality.
- The LLM is an orchestrator/personality: it must submit moves through tools
  and read state through tools. It must not determine legality itself or be
  able to corrupt game state.
- A complete game against Stockfish must work with the LLM disabled.
- Keep all model-specific behavior behind the brain interface. No other layer
  should depend on a specific LLM provider or model.
- Preserve the single game pipeline: input becomes a command, then tool calls,
  then deterministic engine updates, then an agent reaction based on current
  game state.
- Personality affects commentary only—not move selection, difficulty, or other
  game settings.

## Git and task hygiene

- Never commit or push directly to `main`.
- Use one branch and PR per slice: `feat/<slug>`, `fix/<slug>`,
  `chore/<slug>`, or `docs/<slug>`.
- Run the relevant local checks before pushing. Do not merge a PR with failing
  or pending CI.
- When a task is merged, move its entry from `TODO.md` to `DONE.md` under the
  current date.

## Deployment boundaries

- Docker Compose is for the self-hosted home-network deployment. Avoid changing
  `docker-compose.yml`, model serving, or voice infrastructure unless the task
  explicitly requires it.
- Basic gameplay must remain offline-capable after local model assets are
  available.
- Physical-board support is a future, separate seam; do not let it shape core
  application architecture.
