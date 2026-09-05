# The planner/narrator split

`llama_brain.py`, `personality.py` (2026-07-25; full design narrative in git
history).

**Why:** one turn used to run under one prompt — ~2,000 characters of Glitch
wrapped around a short tool paragraph — and on a 12B, tone and tool selection
compete for the same attention. The fix is not trimming the personality; it is
not asking one call to do both jobs. The split cured the release-blocking
`long_capture[poisoned]` regression (1/5 → 5/5).

## The two phases

- **The planner** is the bounded tool loop (`max_iterations` 4, separate
  correction budget, results fed back as `role: "tool"`, schema validation
  before dispatch, domain rejections as results). It runs on a compact,
  persona-free contract: the acting rules (board/engine own truth, map
  phrasing onto `legal_moves`, ask between the legal moves that fit and refuse
  a move no entry fits, omit optional args, read `retry` semantics, advice asks
  route to `get_best_moves`). Its first
  tool-free turn ends the loop and is an internal handoff note — the planner
  never speaks to the player. Thinking stays off: picking a tool is a parse.
- **The narrator** is one further call on the full Glitch prompt
  (+ verbosity layer), offered **no tools**, given the utterance, the turn's
  tool results, and the handoff note. Its text is the commentary. It is the
  one phase that may think (when analysis landed). Being structurally unable
  to act is the enforcement of "react from results, never the raw utterance";
  `tests/test_closing_pass.py` pins it route by route. `Brain.narrate` (the
  fast path's commentary call) is the same code path with a different brief.

A budget stop reaches no narrator — nothing verified came back to speak from.
A `no_progress` stop (a planner turn whose every call repeats one this turn
already made *and is answered as it was then*) *does* reach the narrator: real
results came back, and the loop just refuses iterations that can only repeat.
The repeated call is still dispatched — whether a repeat may run is the tool
layer's judgment — and a repeat that comes back different is progress, not a
stall: a second `undo` carries the same empty arguments as the first and pops
a different exchange, and keying the stall on the call alone once ended
"undo, undo, then play X" with X never played.

## What the narrator is not given

- **The board.** Tool results are the record of what changed; the fast path
  hands a freshly read post-move board because it has one.
- **A side to play for** (#188/#193): the observe beat runs while the reply is
  still computing, so `turn`, `legal_moves`, and the FEN are withheld
  (`api._narrator_state_dict`), the split `make_move` result carries no
  mid-exchange `fen`/`turn`, and a move turn is remembered by the reaction
  alone (`docs/turn-memory.md`) — every leak of "it is your move" produced
  narrators announcing moves of their own.

## Cost

The fast path is unchanged (0 calls at verbosity=low, 1 otherwise); brain
turns pay one extra short tool-free completion (plain move 2 → 3 calls).
Ceilings: planner 2048 / narrator 4096 `max_tokens`; a truncated call is a
failed turn, never a truncated reply that travels.
`CHESSAPP_PLANNER_TEMPERATURE` samples the planner apart from the narrator;
the eval harness reads the same variable.

Every prompt change here is eval-gated (`docs/agent-evals.md`).
