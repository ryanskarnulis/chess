# Turn memory: what the model remembers between turns

`conversation.condense` (2026-07-25; full design narrative in git history).
Read `docs/planner-narrator.md` first — this is what goes *into* the phases.

## The shape

One deterministic transform applied to the message list before it reaches a
brain:

```
[ digest(user) , "Noted."(assistant) , …last RECENT_TURNS turns verbatim… ]
```

- **Recent turns stay verbatim** — reference-following ("no, the other rook")
  only reaches back a turn or two.
- **Everything older collapses to the player's own requests**, their words
  only, capped with an explicit `(+N earlier requests not listed)` line.
  Glitch's older prose is dropped entirely: personality competing with the
  tool decision is this project's measured failure mode (self-poisoning;
  `long_capture[poisoned]`).

## Rules that keep it honest

- **No board facts, settings, or saves in the digest.** Those are injected
  fresh into the state block every turn; a digest line restating them would be
  a second, ageing copy — the exact self-poisoning shape. Older turns that
  were just a move (`e4`) are dropped too: they already live in `history`.
- **No model writes the summary.** Code copies the player's words; a
  model-written rollup would be an unguarded place to hallucinate, plus a
  third model phase per turn.
- **What the app said is never remembered as Glitch's.** App lines
  (lost-brain lines, the stuck line, a late close, a fallback's reply line)
  are for the player, not the model's memory — remembered as such, the
  narrator imitates the register (live, back when the retired honesty guard's
  fallback was a canned first-person apology: "I almost said something that
  didn't happen") or completes the format (announcing a move before the reply
  exists, #193). A turn is remembered by what Glitch himself said, exactly as
  he said it: nothing checks or cuts his words any more (#368), so a
  misstatement is remembered too, and fixed in what he is shown next rather
  than edited out. A move turn is remembered by his reaction, which since
  #365 names the engine's reply itself — he spoke after it. A turn he
  said nothing on (a budget stop, a dead provider, a silent low-verbosity
  turn) is remembered by what it *did* (the deterministic move confirmation)
  or an empty message, which `condense` renders as the inert ack (chat
  templates must alternate roles). Carriers: `api.CommandOutcome.memory`,
  `StoredMessage.memory`.

## What the digest cannot carry: an open question

The digest drops Glitch's words, so a question he asked ("Nf3 or Nh3?") is
gone from the planner's view four turns later, and the input budget can trim
it sooner (`LlamaBrain._admit`). It is not memory's job to keep it: the
question is a harness record (#319, `clarification.py`), bound to the
conversation, game and board it was asked about, and shown in the planner's
state block as `open_question` (the player's ask and the candidates) while it
stands (`api.planner_state`). The state block is never trimmed, so the
question survives however long the conversation grows. The record holds no
board fact, so it is not the second, ageing copy the rules above forbid; once
the board moves it is withdrawn, and the next turn is told once that it closed
(`closed_question`), because the transcript may still be quoting it.

## Call sites

`Transcript.memory()` (command pipeline + board drag) and
`agent_api.history_for_loop` (delegate wire) — one memory policy, both entry
points. `window()` stays the raw record. The `Brain` seam is untouched.
