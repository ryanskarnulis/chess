# The record of the game

#372 (roadmap step 12). What both model phases will read in place of the quoted
chat history (`docs/turn-memory.md`), in four parts, all written by code:

1. **The record** — the ledger's events the move list cannot show, keyed to it
   (`ledger.render_record`).
2. **What the player has asked for** — their own words from every turn but the
   latest, bare moves left out, newest kept under a size cap
   (`conversation.player_requests`). This is where a standing ask ("only
   knights from now on") survives; which requests still stand is the model's
   to work out, and code reads none of them.
3. **The last exchange**, whole (`conversation.last_exchange`): what "the other
   one" and "undo that" point at.
4. **The state block**, as today, `open_question` included.

The narrator reads them in place of its chat turns (PR 4, `conversation.recall`
at the head of its brief; `docs/planner-narrator.md`); the planner will next,
with the chat history removed from both (PR 5, the milestone gate).

## Why no model writes it

The first plan was a *story*: a model-written, third-person summary rewritten
after every turn from the ledger (PR 2, #405, measured on gemma-4-12b and then
removed). It measured cheap on the planner's cache (no #362 eviction), but:

- it planted a false fact: Glitch said "i'm going nf6" on a turn that played
  nothing, and the story wrote it down as a move — in the one record meant to
  be ground truth;
- it cost 3–5 s of GPU a turn, slowed the planner 0.3–0.5 s when the two
  overlapped, and would have made a quick player's next turn wait 5.4 s
  median (7.2 s p90);
- its judgement was noisy: "Open" missed a hint's offered moves, "Standing
  requests" took in "let's play", and it wrote moves out of order, which no
  scorer checked;
- a free-prose version transcribed every turn and quoted Glitch in each until
  it hit its token cap and stopped.

Everything it did well, code does exactly: the events are the ledger's, the
offers are tool results, and the player's words are copied rather than
paraphrased (decided 2026-09-28, #372).

## The ledger (`ledger.py`)

A board holds a position and a move stack and forgets everything else. A
takeback pops the stack and keeps nothing, a setting overwrites the last one,
and a declined draw offer changes no state at all. Before the ledger, the only
record of those was the turn trace, and the undo, difficulty and voice buttons
and MCP write none. The ledger is that record: an append-only list of events,
keyed to the move list, written by code and never by a model. It is what the
record above is rendered from, and what #373 grows into `lookup(this_game)`.

| kind | when | details |
|---|---|---|
| `new_game` | a game starts (a reset, or the first game the process sees) | `player_color`, `root_fen` when not the standard start |
| `resumed` | a named save comes back | `name`, `player_color`, `root_fen`; its line follows as `move` events marked `restored` |
| `move` | a move lands, whoever played it | `san`, `uci`, `color`, `by` (`player`/`engine`), `move_number`, `capture` (the piece taken, en passant included), `check`, `material` (the player's advantage in pawns after it) |
| `takeback` | moves come off the board | `undone` (SAN, last first), `plies` |
| `setting` | difficulty, verbosity or voice changes | `name`, `before`, `after` |
| `draw_offer` | the player offers a draw | `accepted`, `reason` (`draw_offer`'s) |
| `game_end` | the game finishes, however | `termination`, `result`, `winner` (`player`/`opponent`/None) |

Every event carries `seq` (process-wide order), `game_id`, and `ply` — the
length of the move line right after it, which is what keys it to the move list.
Moves taken back stay in the record; `line()` is the standing line (every move
minus what a takeback popped), and a trajectory invariant holds it to the
board's move list after every step of every seeded walk.

**Observed, not hooked.** `Ledger.observe(session, settings)` diffs against
the last state it saw: the common prefix of the two move lines is kept, what
was popped is a takeback, what was pushed is moves, a new `game_id` is a new
or resumed game, a newly finished outcome is the ending, a changed setting is
a setting event. Hooking every mutator would have to find every road onto the
board, and the first one missed would be a silent gap; a diff only has to be
called often enough. It is called after every tool dispatch
(`ToolRegistry.dispatch`, so one command's "undo, then play e4" is two events
in order), on the mutation guard's way out and before every checkpoint write
(engine replies, the undo button), by the two settings endpoints that take no
guard, and after a restart settles an owed reply. The facts no board shows
are told to it directly: `offer_draw` notes the offer and its answer (before
an accepted one ends the game, so the offer reads first), every dispatch
notes what its result offered (below), and `resume_game` tells it the next
new game is a save coming back. An observation that fails is
logged and never costs the call (`ToolContext.observe_ledger`).

One diff between two observations cannot see a move taken back and played
again identically; since the ledger observes after every dispatch, that only
happens when a single call does both, and none does.

**Scope and persistence.** One ledger per app context, spanning games: each
game's events carry its `game_id`, and earlier games' are kept in memory up to
`EARLIER_GAMES_KEPT`.
Only the current game is persisted, in `live.json`
(`docs/persistence-and-identity.md`).

## What was offered

The ledger records the moves a tool result put in front of the player
(`OFFER_SOURCES`): a hint's candidates (`get_best_moves`, `hint`), a
question's choices (`ask_player`, `question`), and a refused move's
alternatives (`make_move`, `alternatives`). Recorded at the dispatch
chokepoint from the result itself (`ToolContext.note_offer`), never from
anybody's words. A question also stands in the state block as `open_question`
while it is open (#319); the offer stays in the record after it closes, which
is what "the second one you suggested" reaches for three turns later.

## The record (`render_record`)

One line per event, prefixed by where the move list stood: "after 2. Qh5: took
back 2. Qh5.", "after 11... Nc6: a hint offered d4, Ne2, f4.", "after 30.
Kg2: the player offered a draw; the engine declined (engine ahead).", "the
game ended by checkmate; the engine won (0-1).", and a game's start ("A new
game began; the player has white.", "The saved game 'keep' was resumed; ...").
Moves themselves are not listed: the state block's `history` holds them, and
a second copy would be the ageing duplicate `docs/turn-memory.md` forbids.
Past `RECORD_MAX_LINES` (30) the oldest lines go, and the first line says how
many. The current game only: an earlier game in the conversation lives on in
the player's own words.
