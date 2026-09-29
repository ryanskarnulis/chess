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
  before dispatch, domain rejections as results). It runs on a short,
  persona-free contract (#370): turn the player's words into tool calls; the
  tools enforce the rules, and a failed result says how to fix it (`retry`);
  never judge legality, submit the move asked for; ask between the legal moves
  that fit (`ask_player`, which takes the parts of the move the player named
  and works the moves out itself, #371) or, when the intent is unclear, say
  what to ask; omit
  optional args. What each tool does and which requests trigger it are in the
  tool's own description, and `make_move`'s refusal says whether a corrected
  call can still be the player's move (`fix`, `retry`). Its first
  tool-free turn ends the loop and is an internal handoff note — the planner
  never speaks to the player. Thinking stays off: picking a tool is a parse.
- **The narrator** is one further call on the full Glitch prompt
  (+ verbosity layer), offered **no tools**, given the utterance and the
  typed handoff below — the turn's results sorted by the harness, the fresh
  facts it may state, and the planner's note labelled as a reading. What came
  before the turn leads the brief as data, not as chat turns (#372, below).
  Its text is the commentary. It is the
  one phase that may think (when analysis landed). Being structurally unable
  to act is the enforcement of "react from results, never the raw utterance";
  `tests/test_closing_pass.py` pins it route by route. `Brain.narrate` (the
  fast path, a board drag, a confirmed op and a resignation) is the same
  narrator with the same brief and budget policy (#369, below).

**One narrator** (#369, 2026-09-27). There used to be two: the loop's closer
spoke from the typed handoff, and the reaction (`Brain.narrate`) from a brief
of its own that opened "The player just made their own move" whatever had
happened — a confirmed `new_game` included. Now every narration is the handoff
(`handoff.render`): the player's words (a board drag has none, and the brief
says the player acted on the board), the turn's record, and `narrator_facts`.
A reaction has no planner note. One phase name, `narrator`, in the trace and
the context capture (records before it say `closer` and `reaction`, and the
readers keep both); one budget policy, `deadline.NARRATION_BUDGET_S` when
something waits on the words and the narrator is not thinking,
`NARRATION_CEILING_S` otherwise.

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
(`planner`, `narrator`, `answer`; `closer`/`reaction` on records from before
#369, `rewrite` from before #368), how it ended (`ok`,
`truncated`, `bad_args`, `failed`, `late`), how long it took, and its tokens,
which are `null` when unknown. A `late` call's `ms` is the time the turn waited
before giving up, with the limit it was held to in `budget_ms`. The call may
have run longer than that, so report it as a censored wait rather than a
duration.

Prompt size has a budget too (`input_budget_tokens`, 32k estimated at three
characters a token against llama-server's 131k window — about ten times the
heaviest measured prompt, so it is a safety net, never a knob). Before the
planner's opening call, an over-budget prompt drops the
conversation's oldest exchanges a user/assistant pair at a time; the system
prompt, the state block and the brief are never trimmed, and neither is the
latest exchange (what "do the second one" points at, and where an unanswered
`ask_player` question lives). The loop never trims mid-run — it only appends,
so the KV prefix holds — and a run whose own results outgrow the budget ends
under `budget: input`. The narrator has no conversation to trim since #372 (its
past is a capped section of the brief); a narrator prompt that still cannot
fit is not sent, and its empty reply is the one the pipeline already stands
in for. The trace's
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
while its contract (before #370) said a move no entry fits is illegal and the
answer is to say so. No tool result could close the gap — a mutation reports
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

*Since #371 `ask_player` asks by parts and carries no enum, so its schema is
the same on every board and the whole offer is byte-stable. What still follows
the board is whether it is offered at all (the availability rule below). The
history here explains the refresh machinery, which stays.*

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
  tool is an unknown to the loop. It is the last tool offered, pinned there
  by `test_tool_registry_schema.py`, so it is the only schema that follows the
  board. Everything before it is byte-stable across boards; `claim_draw` is
  always offered and refused by its handler when nothing can be claimed
  ("mask, don't remove", #364 folded into #370).
- **Swapped only when different.** The tools render ahead of the conversation,
  so a new list costs the planner a re-read of its whole prompt; an identical
  one is left as it was. Command-entry schemas are unchanged (the fixed-board
  golden in `test_tool_registry_schema.py` holds), and no schema was minimized.

## Speech is the model's, measured offline (#368, 2026-09-27)

The narrator's words are what the player hears. Nothing checks, cuts or
rewrites them on the live path: code owns actions (inside the tools), and the
model owns speech. When Glitch says something the turn does not back, the fix
is what he was shown — the handoff, the facts, the history, the prompts — and
the measure is speech accuracy (`docs/speech-accuracy.md`), scored offline
from the trace. Every model-spoken turn records `draft` (his words, before
the app composes its reply line around them) and `evidence` (the
`facts.TurnEvidence` its facts are assembled from), and the eval gate fails a
sample whose draft makes an unbacked claim (`docs/agent-evals.md`).

What that reading covers — every operational claim: an ending, a draw, who
won and how, a check, a capture, a move, who played it, a save, a setting, an
engine number, the material count, an action nothing did, a reply that does
not exist yet (`honesty.claims`, `facts.assemble`).

**History.** From 2026-07 to #368 a live honesty guard ran that reading on
every reply. A claim it could not back was first replaced with one of three
canned "Scratch that" lines in Glitch's place, then (2026-09-10) sent back for
one more narrator call (`Brain.rewrite`) with the true facts in plain words,
and cut to the app's deterministic line if the rewrite still lied. An advice
check rode along: once the turn had asked the engine, a playable move the
engine did not name was cut. Of the guard's 7 live firings in the #340
snapshot, 1 caught a real misstatement (the #365 case) and 6 were false
positives on correct replies — each one a reading error, fixed at the reading,
which is where those fixes still live. The outcome class (astra audit F7,
#287: the winner and the termination, read only once the game is over) and
the placement and draw-shape narrowing (2026-09-22) date from that period too.

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

**Asked by parts** (#371, 2026-09-27). The candidate list was the planner's to
fill, and the 12B filled it too wide or not at all: "move my king's knight"
asked all four knight moves (#348), "castle" with both sides open played O-O,
and "move my king's pawn forward" played e4. Now `ask_player` takes the parts
of the move the player described (`piece`, `which` — a square, a file, the
side it started on, or a square colour — `to`, `takes`, `castle`), and
`GameSession.moves_fitting` (`move_parts.fits`) works out the legal moves that
fit. "Which one" is answered off the game's history: the king's knight is the
one that started on g1, wherever it stands. The question offers exactly those
moves, sorted. No parts, a part that is not a board term, or a single fit
(which it says to submit) is `different_args`; nothing fitting is `never`.
Above `ASK_WIDE` (4) moves the result also names the pieces they belong to,
and the brief tells the narrator to ask which piece rather than read every
move out; the record keeps every move, so any of them answers. The schema is
the probe's screened arm verbatim (`parts_ask5`; numbers in
docs/agent-evals.md). Push-verb pawn asks ("push my e pawn") are still played
on this model, as they were before.

**Fresh facts through a seam.** `LlamaBrain.narrator_facts`, wired from
`api.narrator_facts` by `build_app` and the eval harness alike (a test pins
the two), is read once as the planner hands off: `player_color`, `in_check`,
`game_over`, the player-relative `outcome` the outcome class reads (#287),
`captured` — and two about the engine's answer, both lifted into the handoff.
`engine_reply` (`{san, capture, check}`) is the reply the coordinator just
played, settled through the brain's `settle_reply` seam right before the facts
are read (#365); the brief renders it in the record as "Your reply, already
on the board: Nf6, taking their knight, check. The player learns your move
only from what you say; say it however you like.", and the closing sentence
names it again: "Reply to the player in character, and tell them your move,
Nf6, your way." The second one is what does it. With the move only in the
record he said it on 9 of 20 brain-route turns (0 of 16 before the record
line was joined by any instruction); named in the closing sentence, 20 of 20,
none unbacked. A paragraph of its own said it as often but sent a thinking
narrator past the 60 s ceiling on 9 of 20 samples (a probe of interleaved
arms, 2026-09-27; the numbers are in `docs/agent-evals.md`). `reply_owed` is left for the reply the
engine died on: "Your reply to the player's move never came: the engine
failed." No `history` (the refresh block's measured reason, one phase on) and
no side to move. `board_version` (#320) is lifted out too, and never shown as
a number. It dates the turn's analysis: `evaluate_position` and
`get_best_moves` results carry `position.board_version`, and one computed on
an earlier board gets a record line such as "#2 evaluate_position was worked
out after the player played e4; your reply e5 came after it." That is
`move_and_judgment`'s case: the evaluation runs between the player's move and
the reply, and the narrator speaks after the reply. `analyze_last_move` judges
one past move and is not dated. Scores are the player's side
(`player_advantage_cp`, `mate.for`), not White's. Each verdict also carries a
code-written `summary` in `describe_position`'s voice ("Stockfish has the
player ahead by about 6.5 pawns.", "…has you ahead…", "…calls it about
level."). With the number alone, playing Black a queen up, the planner's note
read +654 as "you are down by about 654 centipawns" on 5 of 5 samples, and
Glitch told the player White was winning. With the summary, 5 of 5 were right
(`judgment_as_black`, 2026-09-28).

**One projection.** `narrator_result_view` drops `fen`, `turn`, `legal_moves`
and `captures` from every result a narrator reads, on both briefs — `undo`,
`new_game` and `resume_game` answer with `fen`/`turn`, and they reached the
narrator through the closing brief and through the fast path's confirmed-op
and resign beats while the state view beside them withheld the same keys. The
trace keeps the full results; a test pins that the projection deletes the
four keys and `narrator_facts` carries none of them.

**Claim classes for the handoff** (`honesty.py`), measured before they shipped (the
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
narrator was told (kind, the tools done / refused / looked up, `reply_owed`,
the `engine_reply` SAN),
so a narration is re-judged against that and not against the note.

## The open question (#319, 2026-09-23)

A landed `ask_player` used to end with the narrator's question: the next turn
had to recover *which* moves were on offer from Glitch's wording in the
transcript, which paraphrases them, drops out of the verbatim window four
turns later, and is bound to no board and no conversation. So the harness
keeps the question (`clarification.py`), the way it keeps an armed op: the
origin it was asked in, the `game_id` and `board_version` it was asked about
(the board at the end of the asking turn, which is the one the player hears
it over), what the player said, and the handoff's validated candidates.

| Event | When | Result |
|---|---|---|
| asked | a turn ends `clarify` | a record for that origin, replacing any before it (`superseded`, `asked_again`) |
| open | the game and board are unchanged | kept, however many asides come between: no turn cap |
| answered | the origin's next board change is a move among the candidates — a drag, the fast path or the planner | closed with the move; gone, so it answers once |
| superseded | the origin changes the board some other way (another move, an undo, a reset) | closed |
| invalidated | the board or game changed from anywhere else | found at the next read (`ToolContext.live_clarification`), reported once as expired |

Code decides only whether a question still stands and what closed it, from
the tool results; nothing reads the player's words and nothing ever plays a
candidate for them — an answer is the planner's ordinary `make_move` against
the live board. "Never mind" is the model's to understand: the record stays
open but inert until the next move or ask closes it. It is never read by the
confirmation gate, so it cannot license a destructive op. Not persisted
(`docs/persistence-and-identity.md`).

**What the planner sees** (`api.planner_state`). While the question stands,
the opening board state carries `open_question: {player_asked,
choose_between}`; the turn after it goes stale carries `closed_question:
{player_asked, why}` once, then neither. Planner only — the narrator's views
are derived from `_agent_state_dict`, which never holds it — and not in the
mid-command refresh, since a turn that has moved the board has made the
question stale anyway. Turns with no question carry neither key, so the gated
scenarios that ask nothing read exactly the prompt they always did. No
`PLANNER_PROMPT` change: the keys say what they are, and every arm that added
a sentence to the matching rule has made the knight ask worse (`personality.py`).
Measured on gemma-4-12b (`docs/agent-evals.md`, 2026-09-24): neutral. The
gate holds, and the frontier's #319 scenarios score the same with and
without the keys — the planner resolves "the first one" against
`legal_moves` order rather than against any question, and a stale question's
first candidate is played with `closed_question` in view. Three wordings
meant to make it heed the closed question were screened and dropped.

The trace's `clarification` field records each turn's view of it — `open`,
`expired`, `closed` (`answered` with the move, or `superseded`) and `created`
— and the seeded trajectories hold every record to the walk's own model of it
(`check_question_is_its_askers`).

## What the narrator is not given

- **A stale board.** The brain route's facts are read as the planner hands
  off (above), after every tool of the turn has run; a reaction reads the same
  facts after its change landed. Neither carries history (#369).
- **A side to play for** (#188/#193): `turn`, `legal_moves`, and the FEN are
  withheld (`api.narrator_facts`, and `handoff.narrator_result_view` on every
  result). While the narrator spoke before the reply existed, every leak of
  "it is your move" produced narrators announcing moves of their own. Since
  #365 the one move he may announce is handed to him instead, so nothing he
  says needs whose move is next.
- **The planner's note as a record.** It arrives labelled as the planner's
  reading of the ask; what was done is the harness's line above it.

## Cost

The fast path is unchanged (0 calls at verbosity=low, 1 otherwise); brain
turns pay one extra short tool-free completion (plain move 2 → 3 calls).
Until #368 a guarded turn paid one more for the rewrite (6 of 150 deployed
turns were guarded when the rewrite landed, four of them false positives).
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

The narrator has a wall clock; the planner does not. A token cap
bounds generation, not queueing or a stalled server. The pipeline drops a
reaction at `deadline.NARRATION_BUDGET_S` and closes on the app's line for
both moves (#283, `docs/turn-coordinator.md`), and
`_NARRATE_TIMEOUT` hangs up just above that so the abandoned generation stops
holding a server slot. The loop's closer is bounded inside the brain (#316):
the same budget when the turn moved a piece and the words are only a
reaction, `deadline.NARRATION_CEILING_S` — a stall backstop, never a cut on a
thinking answer — when nothing is held or the closer thinks; either way only the tool-free speech thread is left behind,
and the plan's record comes back as `narration_late` with no text. The planner
sends no ceiling: it legitimately runs 30 s and more with thinking on, its
calls act, and it is bounded between round trips (`planning_deadline_s`), never
during one.

Every prompt change here is eval-gated (`docs/agent-evals.md`).

## What the narrator remembers (#372, 2026-09-28)

The narrator no longer reads chat turns. Until #372 it got the same condensed
conversation as the planner, as user/assistant messages, so his own older
lines sat in front of him as things he had said and a false one ("black played
c5" over a real e5) stayed there as fact. Now its messages are the persona and
the brief alone, and the brief opens with **"Before this turn:"**
(`conversation.recall`, built by `api._earlier` as the turn opens):

- the game's record, kept by the app — the ledger's events the move list
  cannot show, keyed to it (`ledger.render_record`: takebacks, setting
  changes, draw offers, what was offered, the ending);
- what the player asked for earlier, in their own words (every turn but the
  latest, bare moves left out, newest kept under a cap);
- the last exchange, whole: the player's words and "What you said then (your
  words, not a record)" (`handoff.NARRATOR_REPLY_LABEL`).

Taken as the turn opens, so this turn's own events are the handoff's to tell
and never appear twice. No model writes any of it (`docs/game-record.md`
says why the model-written story was dropped). The planner still reads the
condensed conversation until the milestone PR moves it onto the same data.
