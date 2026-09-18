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

## The planner's board, mid-command (#282, 2026-09-17)

The loop is handed the board once, in its opening user message, and from there
it only *appends* — the assistant turn and one `role: "tool"` message per call,
so the KV prefix holds. Which meant that inside one command the planner's
second decision was made against the first decision's `legal_moves`: "undo that
and play e4 instead" asked it to submit a move its own list could not contain,
while its contract says a move no entry fits is illegal and the answer is to
say so. No tool result could close the gap — a mutation reports
`fen`/`turn`/`engine_move` and never the menu, and it must not report the menu,
because the same results are what the narrator speaks from (the reason
`save_game` answers with a bare `board_version`).

So the loop now asks, once per iteration, whether the board is a different one,
and appends the planner's own state block again when it is
(`api.planner_board_refresh` → `LlamaBrain.board_refresh`, labelled "Board
state after those tool calls:"). Three properties are load-bearing:

- **The planner's alone.** It is a message, not a result — `run.tool_results`
  is untouched, so the narrator's brief cannot see it and `_exchange_key`'s
  stall rule still keys on what the tools actually answered.
- **Keyed on the board, not the view.** `board_version` decides whether to
  send; a setting or a save that moved without the position moving is already
  reported by the tool that moved it, and a second copy would be the ageing
  duplicate `docs/turn-memory.md` forbids.
- **The menu, and not the turn's own history.** The first cut sent the whole
  opening view back, and it cost `undo_twice_and_replace` 19/20 → 4/20
  (interleaved blocks of five against unchanged main on one server,
  2026-09-17), every miss one takeback short: "undo the bishop move and undo
  the knight move, then play d4" took back one exchange and played d4 on a
  board still holding the knight move. A `history` the bishop move has just
  left reads to a 12B as *the takebacks are done* — a block meant to say what
  may be played was answering a question about what had been finished. Trimmed
  to the menu and what qualifies it (`api._REFRESH_KEYS`), the same screen read
  19/20 against 16/20. What a tool undid is that tool's result to report.
- **Withheld mid-exchange.** `make_move` applies the player's move and stops,
  so the board it leaves has the engine to move and the engine's legal moves.
  Handing a move-choosing phase that menu is #193 one layer up, so while a
  reply is owed nothing is sent at all — the invariant is that the planner sees
  a board the player is to move on, or no board at all. Nothing is lost: a
  second player move under one turn is refused by the phase machine.

It is per-iteration, not per-call: a model that emits `[undo, make_move e4]` in
one batch still decides blind, and a refresh between two tool messages would
break the one-answer-per-call shape the wire keeps. `PLANNER_PROMPT` is
unchanged — the label dates the block and nothing more, because every measured
arm that added a *fact* to that contract made it worse. The trace records the
board versions the planner was re-shown (`state_refreshes`), which beside
`mutations` is what says whether a turn that moved the board went on to decide
against it.

## The narrator's second draft (the honesty guard, 2026-09-10)

Every operational claim in the narrator's text — an ending, a draw, who won
and how, a check, a capture, a move, who played it, a save, a setting, an
engine number, the material count — is checked against the board and the
turn's tool results before it is spoken (`honesty.unverified`,
`api._verified_facts`). A claim the
facts don't back used to be answered with one of three canned "Scratch that"
lines in Glitch's place. It is now answered with **one more narrator call**
(`Brain.rewrite`): the same persona prompt and conversation, no tools, and a
brief holding the first draft plus one plain sentence per unbacked claim
saying what is actually so (`honesty.corrections` — "You wrote: 'Snagged your
bishop.' Pieces you have taken: nothing."). The rewrite is checked the same
way. If it passes it is the commentary and is remembered as Glitch's; if it
still asserts something unbacked it is cut and the turn says only what the app
already says with no usable model text — the deterministic move line on a move
turn, the stuck line otherwise. Never a third try, never an apology for a
sentence the player did not hear.

The winner and the termination became facts on 2026-09-18 (astra audit F7,
#287): the ending class still checks one boolean and owns the lie on a live
board ("Game over." with the game running), and an `outcome` class reads the
same words — "checkmate", "you win", "I resigned", "stalemate" — against the
session's outcome only once the game is over, so "I win" over the mate the
player just delivered goes back with "the player won, by checkmate; you
lost". No new words were added to do it: every alternative it matches was
already the ending or draw class's, and its correction is the ending as it
stands. The trace's `outcome` field (winner from the player's side, plus the
termination) is what lets a finished game's commentary be re-judged later.

The split is the house rule one step later: code decides what is true, the
model decides how to say it, and a guard false positive costs one round trip
rather than the reply. The trace records both drafts and both verdicts
(`guarded`, `suppressed`, `rewrite`, `rewrite_claims`, `rewrite_suppressed`);
`guarded` still reads the *first* draft, because the eval floor measures the
model's own discipline and the rewrite is what spares the player the miss.

The advice check rides the same call and was inverted the same day: it fires
only when the turn consulted the engine (`get_best_moves` /
`analyze_last_move` reported moves) and the reply names a playable move the
engine did not — an honesty problem. With no analysis in the turn a move
Glitch names is his own opinion and is his to give; the planner still routes
hint asks to the engine (`advice_is_engine_backed` measures it). Before the
inversion the guard ate a correct London answer and a refused move's own
list of alternatives (live, 2026-09-04 and 2026-09-06).

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
turns pay one extra short tool-free completion (plain move 2 → 3 calls). A
guarded turn pays one more for the rewrite (6 of 150 deployed turns were
guarded when the rewrite landed, four of them false positives since fixed).
Ceilings: planner 2048 / narrator 4096 `max_tokens`; a truncated call is a
failed turn, never a truncated reply that travels — and the reader in front of
the destructive gate fails the same way, to the `unrelated` that changes
nothing. A truncated planner response that nevertheless *carries tool calls*
runs them and the loop goes on (decided 2026-09-05): the provider parses each
call's arguments before the response exists, so a call that arrived is a whole
call, and only the prose is lost — the fragment beside them is never the
handoff note.
The planner samples at 0.3 (`llama_brain._PLANNER_TEMPERATURE`) and the
narrator at the model profile's 1.0: a parse wants the mode, words want the
spread. The number is measured, not chosen — at 1.0 the planner played one of
two knights on "move my kings knight" 6/40, at 0.6 3/40, at 0.3 0/40, with no
single-fit ask over-asked (2026-09-17, #286, `docs/knight-ask-campaign.md`).
`CHESSAPP_PLANNER_TEMPERATURE` overrides it; the eval harness resolves the
number through the same function the app does, and pins that it did.

`narrate` is also the one phase with a wall-clock ceiling. A token cap bounds
generation, not queueing or a stalled server, and the observe beat is the one
call whose caller has already decided it will not wait: the pipeline drops the
reaction at `api._REACTION_BUDGET_S` and plays the reply Stockfish computed
during it (#283, `docs/turn-coordinator.md`), and `_NARRATE_TIMEOUT` hangs up
just above that so the abandoned generation stops holding a server slot. The
planner and the closing narrator send no ceiling — they legitimately run 30 s
and more with thinking on, and nothing is being held while they do.

Every prompt change here is eval-gated (`docs/agent-evals.md`).
