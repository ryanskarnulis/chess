# TODO

The backlog, in priority order. One task = one vertical slice = one branch = one PR (TDD: failing test first). When a task is finished and merged, move its line to `DONE.md` with the merge date. Re-plan freely between slices — this file is the living backlog, not a contract.

## Phase 1 — MVP (web board + text agent)

### Epic: Agent brain (swappable module)

- [ ] GBNF grammar-constrained decoding fallback — **deferred, likely unneeded**: the live spike confirmed Gemma-4 (`UD-Q4_K_XL`) emits structured OpenAI tool calls natively. Revisit only if reliability degrades under load/longer prompts.

## Phase 4 — Physical board (walled off; separate project)

- [ ] Verify Chessnut Move motorized actuation is programmatically controllable **before any design work**
- [ ] `control_physical_board` tool seam only until then

## Infrastructure / process (ongoing)

- [ ] If repo goes public or account upgrades to Pro: enable branch protection (require `lint` + `test` checks) and native auto-merge
