# DONE

Completed tasks, newest first. Moved here from `TODO.md` with the completion date.

## 2026-07-04

- [x] Defensive tool-call parser + retry loop in `LlamaBrain`: each tool call is validated against its schema (name known, arguments parse as JSON, args satisfy the JSON schema); an invalid call feeds the specific error back to the model and retries (bounded by `max_retries`, default 2). On exhaustion the valid calls are kept and invalid ones dropped — never a crash, complementing the registry's dispatch-time guard.
- [x] llama-server brain (`chessapp.llama_brain.LlamaBrain`): the OpenAI-SDK `Brain` behind the seam, pointed at llama.cpp on localhost. Injected client (tests mock it — no live LLM); tool schemas sourced from the registry; reads `content` only (drops Gemma's separate `reasoning_content` so thought blocks never leak); thinking OFF by default (per-request `chat_template_kwargs.enable_thinking`), ON for analysis. **Verified end-to-end against a live Gemma-4 `UD-Q4_K_XL` run: structured tool calls work natively, so the GBNF fallback is deferred as unneeded.**
- [x] Brain interface (`chessapp.brain`): `Brain.get_agent_response(board_state, command) → AgentResponse{text, tool_calls}` — landed with the command endpoint; also closes the first Agent-brain-epic item (#17)
- [x] Text command endpoint `/api/command`: string in → brain seam (`Brain.get_agent_response`) → tool calls through the validated registry → commentary + tool results + new state out; broadcasts on change; scripted fake brain in tests (#17). **Epic "API layer" complete.**
- [x] WebSocket `/ws` state channel: snapshot on connect, broadcast to all clients after every successful mutation, dead sockets dropped silently (#16)
- [x] FastAPI app: game lifecycle endpoints (move w/ optional engine reply, new, undo, resign, PGN) + full state fetch; illegal moves are data, domain failures are 409s (#15)
- [x] Settings tools: `set_difficulty` (exactly one of skill_level/elo, applied to live engine), `set_personality`, `set_verbosity`, `set_hints_mode`, `set_voice_output` (#14). **Epic "Tool layer" complete.**
- [x] Write tools through the legality gate: `make_move` (legal/illegal as data), `undo`, `new_game`, `resign`, `save_game`/`resume_game` (traversal-safe names), `export_pgn` (#13)
- [x] Tool registry + JSON schemas + validated dispatch (un-crashable LLM boundary) + read tools: `get_board_state`, `get_legal_moves`, `get_move_history`, `get_captured_pieces`, `evaluate_position`, `get_best_moves` (#12)
- [x] Full offline game vs Stockfish with no LLM in the loop — acceptance tests: engine-vs-engine to a result, scripted-user-vs-engine, save/resume mid-game (#11). **Epic "Deterministic core" complete.**
- [x] Stockfish analysis tools: `evaluate_position` (white-POV cp/mate), `get_best_moves` (MultiPV candidates with SAN + scores) (#10)
- [x] Stockfish bridge via python-chess UCI: `EnginePlayer` with `Skill Level` / `UCI_Elo` difficulty; CI installs stockfish, local tests skip without it (#9)
- [x] Export PGN (SAN movetext, Result incl. resignation, SetUp/FEN headers for custom starts) (#8)
- [x] Save game / resume game (JSON: root FEN + UCI moves + resignation; resume replays through the legality gate) (#7)
- [x] Resign / result recording (session-level termination folded into outcome/is_game_over) (#6)
- [x] Undo (ply takeback; plies=2 gives vs-engine pair semantics, pairing policy stays in the caller) (#5)
- [x] Move history + captured-pieces derivation from board state (#4)
- [x] `GameSession` class wrapping `python-chess`: new game, submit move (SAN/UCI → accept/reject), turn tracking, game-over detection (mate/stalemate/draw rules) (#2)
- [x] Pushed initial commit to github.com/ryanskarnulis/chess; first CI run green (lint + test)
- [x] Repo settings: delete-branch-on-merge enabled; auto-merge/branch protection confirmed unavailable on Free-plan private repos → merge-on-green handled via `gh pr checks --watch` + squash-merge
- [x] README.md: project overview, architecture summary, status/roadmap, dev setup
- [x] Restructured to monorepo layout: Python project moved into `backend/` (frontend/ arrives in Phase 1)
- [x] Removed docs/DEVELOPMENT.md — CLAUDE.md is the single source of truth for process
- [x] Created TODO.md (living backlog) and DONE.md (this file)
- [x] CI pipeline: GitHub Actions running ruff lint/format-check + pytest with coverage on every PR and push to main
- [x] Python scaffold: src-layout package `chessapp`, pytest + ruff dev tooling, smoke test green locally
- [x] Created private GitHub repo `ryanskarnulis/chess` (initial push pending `workflow` token scope)
- [x] CLAUDE.md: architecture principles, TDD/agile process, git/PR workflow, commands
- [x] Project brief (BRIEF.md): architecture, stack selection, phasing, risks
