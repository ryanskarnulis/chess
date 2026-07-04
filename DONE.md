# DONE

Completed tasks, newest first. Moved here from `TODO.md` with the completion date.

## 2026-07-04

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
