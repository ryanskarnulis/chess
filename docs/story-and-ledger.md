# The story of the game, and the ledger under it

#372 (roadmap step 12). The story replaces the quoted chat history both model
phases read (`docs/turn-memory.md`): a neutral, third-person running account of
the game, written by a model from ground truth that code computes. This file
grows with the PRs that build it: the ledger (PR 1), and the summarizer that
writes the story (PR 2), which nothing reads yet.

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

## The story (`story.py`, PR 2: trace-only)

**Off by default in this PR** (`CHESSAPP_STORY=1` turns it on, in `build_app`
and the eval harness alike, and a test pins that the two agree). Nothing reads
the story yet; it is written and recorded so its cost and its accuracy can be
measured before a phase depends on it.

**What the summarizer is told.** After every turn, code writes a `TurnNote`:
the ledger events since that conversation's last note, rendered as sentences
("2. exd5 by the player, taking a pawn (the player up 1)", "taken back: exd5
(1 move)", "verbosity changed from normal to low"), the player's words (or
"The player moved a piece on the board." for a drag), the turn's tool calls
with what each answered, and Glitch's own words (`draft`) labelled "his words
only — a move or event he names happened only if What happened shows it". The
app's lines (a move confirmation, the stuck reply) are never in it. A turn
with no events says **"What happened: nothing on the board."** — left out, a
12B read Glitch's "i'm going nf6" on a turn that played nothing as the move
2... Nf6.

**The shape.** `personality.SUMMARIZER_PROMPT` asks for exactly three parts:
*Standing requests* (in the player's words), *Open* (a question, an offer or a
choice a next remark could point back to), and *The game so far* (at most 120
words, past tense, old stretches folded into a sentence). Neutral third person,
Glitch quoted only when the player may refer back to it. The first cut asked
for free prose under 250 words: the 12B transcribed every turn and quoted
Glitch in each, the story grew by ~40 tokens a turn, and from turn 14 every
call hit the cap and the story stopped (7 of 20 runs truncated).

**The call** (`ChatSummarizer`): the same provider as the brain (decision 4),
thinking off, temperature 0.3, `STORY_MAX_TOKENS` 450, phase `summarizer` in
the context capture (attributed to the last turn it covers). A story that runs
into the cap is asked once more, shorter (`_SHORTER`); one that still does not
fit, and any failed call, leaves the old story and the notes pending, and the
next turn's note starts the catch-up. Up to `NOTES_PER_CALL` notes go in one
call.

**The keeper** (`StoryKeeper`): one daemon worker for every conversation. A
turn enqueues its note at the end of the trace `finally` (after
`observe_ledger`, so the engine's reply is in it) and returns; nothing the
worker does can raise into a turn. One story per conversation: the panel's on
`ToolContext.story` (in `live.json` and named saves, like the transcript; a
resumed save brings its story back), a delegate thread's on its
`StoredConversation` (`conversations.json`). A new game does not start a new
story (decided 2026-09-28): it is an event in the one the conversation has.
Pending notes are persisted too, so a restart catches up.

**What is recorded.** Every turn carries `story: {pending, covered_through}`,
whether its conversation's story covered every earlier turn as it began. Every
run writes a `story` trace record: the notes it took in, the story before and
after, its calls, `waited` — for each turn that began while the story was
behind, how long after it began the story caught up (`lag_ms`, the wait PR 3
adds) — and `evidence`, what a story covering those notes may state.

**Scored offline** (`speech_accuracy.score_story`, the `story` split of
`scripts/speech_report.py`). `facts.story_facts` turns the keeper's
accumulated evidence into the reading's `VerifiedFacts`, game-spanning and
historical (a capture or an ending earlier in the conversation stays true to
tell). The reading is the narrator's, with the story's persons mapped onto its
own ("the player" → you, "Glitch"/"the engine" → I), a narrative "then"
dropped (the reading takes it for a conditional and skips the sentence), and
move numbers dropped (the sentence splitter parted "6... exf4" from its
sentence). The three setting-value classes are unscored on a story, since it
may name a value a setting held before; `verbosity_change` stays scored. As
for Glitch, a pawn move named bare ("played c5") is not read at all.

### Measured (2026-09-28, gemma-4-12b, idle GPU)

One scripted 20-turn panel game per arm through a scratch server (every turn
on the brain route: moves in words, a standing ask, a takeback, verbosity down
and up, a hint, "who's winning?", "pick one for me"):

| arm | planner opening call: cached/prompt | its server ms (median) | story call (median / p90) | turns begun behind | would-be wait |
|---|---|---|---|---|---|
| story off | 0.83 | 1,378 | — | — | — |
| story on, 6 s think | 0.83 (min 0.77) | 1,639 | 3.3 s / 4.8 s | 0 / 20 | none |
| story on, 0 s think | 0.83 (min 0.78) | 1,918 | 4.8 s / 6.4 s | 19 / 20 | 5.4 s median, 7.2 s p90 |

- **The planner keeps its cached prefix** (#362): the summarizer on the same
  server does not evict it, at either pace.
- **The cost is contention, not eviction**: when a story run overlaps a turn,
  the turn's calls share the GPU, and the planner's opening call ran ~0.3–0.5 s
  slower. At a human pace (6 s between turns) the story was always done first.
- **The wait** PR 3 adds is bounded by one story call when the player is
  quick; at 0 s think it would have been 5.4 s median. Measured while the
  story and the turn competed, so a turn that waits (and so does not compete)
  should wait less.
- **Story accuracy**: 166/166 and 45/45 claims backed across the two arms'
  40 stories. Replaying the first arm's 20 notes one at a time through the
  final prompt: 19 of 20 versions fully backed, the 20th (turn 2) carrying the
  one Glitch-said move above, corrected on the next turn.
