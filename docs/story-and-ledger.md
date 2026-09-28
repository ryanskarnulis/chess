# The story of the game, and the ledger under it

#372 (roadmap step 12). The story replaces the quoted chat history both model
phases read (`docs/turn-memory.md`): a neutral, third-person running account of
the game, written by a model from ground truth that code computes. This file
grows with the PRs that build it; so far it holds the ground truth.

## The ledger (`ledger.py`)

A board holds a position and a move stack and forgets everything else. A
takeback pops the stack and keeps nothing, a setting overwrites the last one,
and a declined draw offer changes no state at all. Before the ledger, the only
record of those was the turn trace, and the undo, difficulty and voice buttons
and MCP write none. The ledger is that record: an append-only list of events,
keyed to the move list, written by code and never by a model. It is the input
the story is written from, the oracle speech accuracy scores the story against
(#367), and what #373 grows into `lookup(this_game)`.

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
guard, and after a restart settles an owed reply. Two facts no board shows are
told to it directly: `offer_draw` notes the offer and its answer (before an
accepted one ends the game, so the offer reads first), and `resume_game` tells
it the next new game is a save coming back. An observation that fails is
logged and never costs the call (`ToolContext.observe_ledger`).

One diff between two observations cannot see a move taken back and played
again identically; since the ledger observes after every dispatch, that only
happens when a single call does both, and none does.

**Scope and persistence.** One ledger per app context, spanning games: each
game's events carry its `game_id`, and earlier games' are kept in memory up to
`EARLIER_GAMES_KEPT` so a reader that looks once per turn never misses one.
Only the current game is persisted, in `live.json`
(`docs/persistence-and-identity.md`).
