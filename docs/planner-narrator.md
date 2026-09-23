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
  (+ verbosity layer), offered **no tools**, given the utterance and the
  typed handoff below — the turn's results sorted by the harness, the fresh
  facts it may state, and the planner's note labelled as a reading. Its text
  is the commentary. It is the
  one phase that may think (when analysis landed). Being structurally unable
  to act is the enforcement of "react from results, never the raw utterance";
  `tests/test_closing_pass.py` pins it route by route. `Brain.narrate` (the
  fast path's commentary call) is the same code path with a different brief.

A budget stop speaks from what ran (#288). The planning phase is bounded by
model turns (`max_iterations` 4), malformed calls (`max_corrections` 2),
dispatched tool calls per turn (`max_tool_calls` 8), Stockfish-backed calls per
turn (`max_analysis_calls` 3, `review_game` included) and a 60 s planning
deadline checked between round trips; the per-turn caps were sized from the
deployed trace (at most 2 calls and 2 analysis calls in any of 236 turns,
planning p99 8.7 s). A call past a cap is answered `"not run this turn"`, never
dispatched, and the phase ends there under `stop_reason="budget"` (or the
standard's `max_iterations` / `correction_limit`), with the response's `budget`
and the trace's naming which. When a tool did real work first, the narrator
closes the turn from a `partial` handoff under the loop's `_BUDGET_NOTE` —
whatever the record does not show as done was not done — so the player hears
what happened and that the rest did not, in Glitch's words. When nothing ran
there is nothing verified to speak from: the turn ends silent and the pipeline
says its stuck reply.

Because the deadline is checked between round trips, the last planner call can
start inside it and finish past it. The trace's `planning` field (#317) records
this: the phase's `elapsed_ms`, the configured `deadline_ms`, and `overrun_ms`,
which is how far past the deadline it finished. The trace's `calls` list has
one entry per model round trip. Each entry records which phase made the call
(`planner`, `closer`, `reaction`, `rewrite`, `answer`), how it ended (`ok`,
`truncated`, `bad_args`, `failed`, `late`), how long it took, and its tokens,
which are `null` when unknown. A `late` call's `ms` is the time the turn waited
before giving up, with the limit it was held to in `budget_ms`. The call may
have run longer than that, so report it as a censored wait rather than a
duration.

Prompt size has a budget too (`input_budget_tokens`, 32k estimated at three
characters a token against llama-server's 131k window — about ten times the
heaviest measured prompt, so it is a safety net, never a knob). Before the
planner's opening call and every narrator call, an over-budget prompt drops the
conversation's oldest exchanges a user/assistant pair at a time; the system
prompt, the state block and the brief are never trimmed, and neither is the
latest exchange (what "do the second one" points at, and where an unanswered
`ask_player` question lives). The loop never trims mid-run — it only appends,
so the KV prefix holds — and a run whose own results outgrow the budget ends
under `budget: input`. A narrator prompt that still cannot fit is not sent; its
empty reply is the one the pipeline already stands in for. The trace's
`input_trimmed` counts the exchanges dropped.
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

### The offer follows the board (#315, 2026-09-23)

The block alone was half the fix. `ask_player`'s candidates are an enum of the
live `legal_moves` (#289), and the loop resolved its tool offer — and the
schemas it validates calls against — once per command. So "undo and move my
king pawn" re-showed the planner the starting board, and its
`ask_player(["e3", "e4"])` came back `'e4' is not one of [...]` against the
pre-undo enum: two sources of truth inside one iteration.

The offer is now re-resolved at the refresh and nowhere else. When a new board
is appended, `brain_tool_definitions` runs again, and if the result differs
from what is offered, the tools sent and the schemas checked are both swapped;
the trace records the swap as `offer_refreshes` (a subset of
`state_refreshes`). Three consequences are deliberate:

- **Mid-exchange, nothing moves.** No refresh means no re-resolve, because an
  enum narrowed then would be the engine's menu. The offer stays on the
  player's last board — the one the planner was last shown — and the
  `ask_player` handler refuses with `retry: never` while the engine is to move
  (or the game is over), rather than calling the player's moves illegal.
- **Availability rides along.** Under two legal moves `ask_player` is withheld,
  so a takeback can bring it back or take it away mid-command, and a withheld
  tool is an unknown to the loop, the way `claim_draw` is.
- **Swapped only when different.** The tools render ahead of the conversation,
  so a new list costs the planner a re-read of its whole prompt; an identical
  one is left as it was. Command-entry schemas are unchanged (the fixed-board
  golden in `test_tool_registry_schema.py` holds), and no schema was minimized.

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

Two more classes were narrowed on 2026-09-22, after #289's gates caught them
cutting correct answers. The move class knows where the pieces stand
(`VerifiedFacts.placements`): a piece named on its own square — "White: Ke1,
Qd1, …", asked to show the position — is placement, not a move nobody could
play. The draw class reads only the shapes that report a result ("that's a
draw", "ended in a draw", "draw agreed"), not the noun somebody offered,
declined or called too early for. Every live guard firing in the deployed
trace had been on a correct reply (6/6, all fixed earlier), so the guard errs
toward letting a reply through when it cannot tell a report from talk.

## The handoff (#289, 2026-09-22)

The narrator used to close from the raw results and the planner's free-form
note, told to reply "based only on those results and that note". The note is
not a record: on a turn with no tool calls it was the only thing the narrator
had, nothing said that nothing had been done, and a note reading "undid your
last move" could come back as "Done, taken back." (astra audit F9).

So the harness now says what happened (`handoff.py`). `build` sorts the
results into **done** (an `ok` result from any tool that is not a read),
**refused** (`ok` not true, or a move the board rejected) and **looked up**
(`READ_TOOLS`; a test pins every registered tool to one side), and derives the
turn's **kind** from those and the stop reason alone — `reply` (no tool
called), `declined` (only refusals), `partial` (something done, and something
refused or the loop ended the phase itself), `completed`. The brief lists the
results with ids (`#1`, `#2`) that the sorted lines point at, says **"Done this
turn: nothing."** when nothing was, and demotes the note to "The planner's
reading of what the player wants (not a record of what happened)". It is kept
because a `reply` turn has nothing else to answer from. Telling an answer from
a clarification is language, so the harness does not try: the planner
declares it.

**Typed clarifications** (PR 2, 2026-09-22). When two or more `legal_moves`
entries fit the player's words, the planner calls `ask_player` with them. Its
`candidates` are an enum of the live legal moves (`tools.brain_tool_definitions`
builds the offer for assembly, the eval harness and the planner probe alike;
the handler re-checks against the board, and the tool is withheld with fewer
than two legal moves), so a candidate the board does not allow is a schema
correction inside the loop. A landed ask ends the planning phase — only the
player can answer it — and the handoff is `clarify`, with a brief line asking
the narrator to name each candidate. It ends it at the call, not after the
batch (#314, 2026-09-23): every call after a landed ask is answered "not run
this turn" and never dispatched, so a planner that asks and then plays one of
the candidates in the same batch leaves the board untouched. Calls before the
ask ran and stand, reported as done — nothing is rolled back. An ask that is
refused or off the enum stops nothing. The candidates count as reported moves,
so the advice licence covers them. Before this, the question lived in the
note and was paraphrased away: on `main` both ambiguity scenarios asked
"which rook and which square?" 20/20, naming nothing. The tool is registered
on the app's split registry only; the MCP server's caller asks its own user.

**Fresh facts through a seam.** `LlamaBrain.narrator_facts`, wired from
`api.narrator_facts` by `build_app` and the eval harness alike (a test pins
the two), is read once as the planner hands off: `player_color`, `in_check`,
`game_over`, the player-relative `outcome` the guard certifies (#287),
`captured` — and `reply_owed`, which the brain lifts into the handoff as "the
engine has not played its reply to the player's move yet; the app announces it
after you speak." No `history` (the refresh block's measured reason, one phase
on) and no side to move.

**One projection.** `narrator_result_view` drops `fen`, `turn`, `legal_moves`
and `captures` from every result a narrator reads, on both briefs — `undo`,
`new_game` and `resume_game` answer with `fen`/`turn`, and they reached the
narrator through the closing brief and through the fast path's confirmed-op
and resign beats while the state view beside them withheld the same keys. The
trace keeps the full results; a test pins that the projection and
`_narrator_state_dict` delete the same four keys.

**Backstops in the guard** (`honesty.py`), measured before they shipped (the
#287 rule; corpus in `test_honesty.py`, deployed sweep in
`docs/agent-evals.md`):

- `takeback` and `restart` — an action no board fact backs, checked against
  the turn's own `undo` / `new_game`. Past forms with a move or piece as the
  object, plus Glitch's own register from the deployed trace ("back to where
  we were"); never "new game" itself, which stays the ending class's.
- `unplayed_reply` — a SAN the engine could play on the board the narrator
  spoke over, while its reply was still being computed, that the turn accounts
  for no other way. Every route narrates before the reply exists (the observe
  beat by construction; the brain route because its narrator closes inside
  `get_agent_response` and the pipeline collects afterwards), so "My turn.
  Nf6." read as true whenever Nf6 was playable or happened to be the reply.
  The future hedges still exempt a threat ("I'll hit you with Nf6").

The narrator's `_BASE` is a speaking contract now: not the referee, a move
made at the player's request is the player's, say only what the record shows
done, and ask with the options named when the player has to choose. "You
change the game only through your tools" and "never claim to have done
something you did not do with a tool" were one-prompt text from before the
split, read by a phase that holds no tools.

`AgentResponse.handoff` and the trace's `handoff` field record what the
narrator was told (kind, the tools done / refused / looked up, `reply_owed`),
so a narration is re-judged against that and not against the note.

## What the narrator is not given

- **A stale board.** The brain route's facts are read as the planner hands
  off (above), after every tool of the turn has run; the fast path hands a
  freshly read post-move board because it has one. Neither carries history.
- **A side to play for** (#188/#193): the observe beat runs while the reply is
  still computing, so `turn`, `legal_moves`, and the FEN are withheld
  (`api._narrator_state_dict`, and `handoff.narrator_result_view` on every
  result), the split `make_move` result carries no mid-exchange `fen`/`turn`,
  and a move turn is remembered by the reaction alone (`docs/turn-memory.md`)
  — every leak of "it is your move" produced narrators announcing moves of
  their own.
- **The planner's note as a record.** It arrives labelled as the planner's
  reading of the ask; what was done is the harness's line above it.

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

Both narrator phases have a wall clock; the planner does not. A token cap
bounds generation, not queueing or a stalled server. The pipeline drops the
observe beat's reaction at `api._REACTION_BUDGET_S` and plays the reply
Stockfish computed during it (#283, `docs/turn-coordinator.md`), and
`_NARRATE_TIMEOUT` hangs up just above that so the abandoned generation stops
holding a server slot. The loop's closer is bounded inside the brain (#316):
`_CLOSING_BUDGET_S` when the engine's reply is owed and held behind words that
are only a reaction, `_CLOSING_CEILING_S` — a stall backstop, never a cut on a
thinking answer — when nothing is held or the closer thinks; either way only the tool-free speech thread is left behind,
and the plan's record comes back as `narration_late` with no text. The planner
sends no ceiling: it legitimately runs 30 s and more with thinking on, its
calls act, and it is bounded between round trips (`planning_deadline_s`), never
during one.

Every prompt change here is eval-gated (`docs/agent-evals.md`).
