# DONE

Completed tasks, newest first. Moved here from `TODO.md` with the completion date.

## 2026-07-04

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
