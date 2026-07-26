# Turn memory: what the model remembers between turns

*Design note for the Sprint 4 slice "turn summaries replace the raw transcript
window" (audit item 17). Read `docs/planner-narrator.md` first — this describes
what goes **into** those two phases, not how they run.*

## What it replaced

`Transcript.window(20)` handed both model phases up to **40 raw messages** —
every prior command and every line of Glitch's commentary, verbatim, oldest
first. That was the agent's entire memory, and it had three faults:

1. **It grew.** Twenty turns of prose is a large, ever-shifting prompt prefix on
   a 12B, and every turn of a long game made it larger.
2. **It was mostly the wrong content.** Older assistant turns are personality —
   trash talk, hedges, threats. That prose competing with the tool decision is
   the measured failure mode this project already has a name for
   (self-poisoning, trace review 2026-07-13; the `long_capture[poisoned]`
   regression). The planner needs to know *what was asked*, not how Glitch said
   it.
3. **It was a second copy of facts the app already holds.** A stale sentence
   about saving beat a save sitting on disk, until `saved_games` and `settings`
   were injected fresh into the state block (#154, #155).

## The shape

One deterministic transform, `conversation.condense`, applied to the message
list before it reaches a brain:

```
[ digest(user) , "Noted."(assistant) , …the last RECENT_TURNS turns verbatim… ]
```

- **The recent turns stay verbatim.** Reference-following ("do the second one",
  "no, the other rook") only reaches back a turn or two, and that is exactly
  what stays untouched.
- **Everything older collapses to the player's own requests** — their words,
  whitespace-collapsed, truncated on a word boundary, capped oldest-dropped-
  first with an explicit `(+N earlier requests not listed)` line. Glitch's side
  of those turns is dropped entirely.

## What is deliberately *not* in the digest

The audit's list was "game events / user preferences / active settings /
unresolved requests / lightweight conversational context". Two of those
categories are already solved better elsewhere, and putting them in a summary
would recreate the bug the injection cured:

- **Game events** — the moves, the captures, the result — are board truth, and
  board truth comes from the current state (`api._agent_state_dict`), never from
  history. The digest contains no moves. It goes further: an older turn whose
  whole command *was* a move (`e4`, `Bxe6`, `e2e4` — how a board drag records
  itself) is dropped from the request list, because that turn's content is
  already in `history` and repeating it is noise wearing a fact's clothes.
- **Active settings and saved games** are injected fresh every turn. A digest
  line saying "the player asked for hard mode" would be a second, ageing copy of
  `settings.difficulty` — the exact self-poisoning shape.

What survives is what genuinely only exists in the conversation: the standing
asks and preferences the player expressed in their own words. Unresolved
requests are covered by the same list — an ask that was never settled is an ask
that appears there and nowhere in the state block.

## Why no model writes it

An LLM-written rollup was the obvious reading of "summarize older turns", and it
is rejected here for the house reasons:

- It is a third model phase per turn (or per rollup) on a local 12B whose budget
  is already four planner iterations plus a narrator.
- It would be the model deciding what happened — the thing `honesty.py` exists
  to stop it doing out loud. A summary is not guarded, and a hallucinated one
  would be *harder* to catch than a hallucinated reply, because nobody reads it.
- Everything worth summarizing that code can verify is already injected. The
  residue is the player's literal words, and copying words needs no model.

So `condense` is pure, deterministic, and unit-tested — no provider, no
sampling, no eval dependency for its own correctness.

## Delivery, and the two synthetic messages

The digest rides as a `user` message followed by a one-word `assistant`
acknowledgement. The pair, rather than a lone message, because the transcript
feeds a chat template that expects user and assistant turns to alternate; a bare
digest message would put two user turns back to back. `Transcript`'s role
whitelist is unchanged — the pair is built at read time and never recorded, so a
save file still cannot smuggle in a role.

## Call sites

- `Transcript.memory()` — what `/api/command` and the board-drag beat pass to the
  brain. `window()` is unchanged and still the raw record (serialization, tests,
  anything that wants what was actually said).
- `agent_api.history_for_loop` — the delegate wire condenses too, off its own
  store, so both entry points have one memory policy.

The `Brain` seam is untouched: a brain still receives `transcript` as a sequence
of chat messages and neither knows nor cares that some of them were condensed.
