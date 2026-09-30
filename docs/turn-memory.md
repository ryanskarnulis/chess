# Turn memory: what the model remembers between turns

`conversation.Recall` (#372, 2026-09-28; before it `conversation.condense`,
2026-07-25 — full design narrative in git history). Read
`docs/planner-narrator.md` first — this is what goes *into* the phases, and
`docs/game-record.md` for the ledger underneath.

## The shape

Neither phase is handed the conversation as chat turns. Each gets one
`Recall`, rendered as data inside its own prompt under **"Before this
turn:"** — at the head of the planner's opening user message (ahead of the
board and the command) and at the head of the narrator's brief:

1. **The game's record, kept by the app** — the ledger's events the move list
   cannot show, keyed to it: takebacks, setting changes, draw offers, what a
   tool offered (a hint's moves, a question's choices, a refused move's
   alternatives), the ending, a resume (`ledger.render_record`).
2. **What the player asked for earlier, in their own words** — every turn but
   the latest, bare moves left out, newest kept under a size cap with a count
   of what went (`conversation.player_requests`). This is where a standing
   ask ("only knights from now on") lives; which requests still stand is the
   model's to work out.
3. **The last exchange, whole** (`conversation.last_exchange`) — what "the
   other one" and "undo that" point at. The reply is labelled as words, not a
   record: "Glitch said then (his words, not a record)" to the planner, "What
   you said then (your words, not a record)" to the narrator.

Then the state block, fresh every turn, `open_question` included. Since
#373 it also names the opening (`opening`: the deepest book position the line
has reached, `openings.opening_of`), which the narrator's facts carry too.

**Why not chat turns.** The planner read Glitch's slang replies as its own past
turns — in-context examples of answering in prose and calling nothing (#361) —
and a false line stayed in history as fact ("black played c5" over a real e5,
2026-09-26). Personality competing with the tool decision is this project's
measured failure mode (self-poisoning; `long_capture[poisoned]`).

## Rules that keep it honest

- **No model writes any of it.** The record is the ledger's, the requests and
  the last exchange are copied words. A model-written story was built and
  measured (#405) and dropped: it put a line Glitch said on a turn that played
  nothing into the record as a move, and cost 3–5 s of GPU a turn
  (`docs/game-record.md`).
- **No board facts, settings, or saves copied in.** Those are injected fresh
  into the state block every turn; a copy would be a second, ageing one — the
  exact self-poisoning shape. The record leaves the moves themselves to
  `history`, and bare-move turns are left out of the requests.
- **Glitch's older words are never shown.** Only his last reply, labelled as
  words; everything before it is the player's side and code's record.
- **What the app said is never remembered as Glitch's.** App lines
  (lost-brain lines, the stuck line, a late close, a fallback's reply line)
  are for the player, not the model's memory — remembered as such, the
  narrator imitates the register (live, back when the retired honesty guard's
  fallback was a canned first-person apology: "I almost said something that
  didn't happen") or completes the format (announcing a move before the reply
  exists, #193). A turn is remembered by what Glitch himself said, exactly as
  he said it: nothing checks or cuts his words any more (#368), so a
  misstatement is remembered too, and fixed in what he is shown next rather
  than edited out. A turn he said nothing on (a budget stop, a dead provider,
  a silent low-verbosity turn) is remembered by what it *did* (the
  deterministic move confirmation) or nothing, which the last exchange shows
  as "said nothing". Carriers: `api.CommandOutcome.memory`,
  `StoredMessage.memory`.

## An open question is not memory's job

A question Glitch asked ("Nf3 or Nh3?") is a harness record (#319,
`clarification.py`), bound to the conversation, game and board it was asked
about, and shown in the planner's state block as `open_question` (the player's
ask and the candidates) while it stands (`api.planner_state`). Once the board
moves it is withdrawn, and the next turn is told once that it closed
(`closed_question`). Its candidates also stay in the record as an offer ("the
player was asked to choose between Nf3, Nh3"), which is what "the second one
you offered" reaches for after the question has closed.

## Call sites

The command pipeline (`api._command_turn`) builds the `Recall` as the turn
opens (`api._earlier`), from the conversation its caller hands it whole:
`ctx.transcript.to_dict()` for the panel, `agent_api.history_for_loop` for a
delegate thread (every text turn, `memory` over `content`). A board drag
builds the same for its narrator. One memory policy, every entry point, in one
place. `Transcript.window()` stays the raw record. The refresh block the loop
appends mid-command stays the menu alone (`api._REFRESH_KEYS`): no `history`,
no record (`docs/planner-narrator.md`).
