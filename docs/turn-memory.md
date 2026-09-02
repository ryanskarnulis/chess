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
- **What the app said is never remembered as Glitch's.** Canned substitutions
  (guard corrections, lost-brain lines) and appended announcements are for the
  player, not the model's memory — remembered as such, the narrator imitates
  the register (live: "I almost said something that didn't happen") or
  completes the format (announcing a move before the reply exists, #193). A
  move turn is remembered by the reaction alone; a substituted turn by what it
  *did* (the deterministic move confirmation) or an empty message, which
  `condense` renders as the inert ack (chat templates must alternate roles).
  Carriers: `api.CommandOutcome.memory`, `StoredMessage.memory`.

## Call sites

`Transcript.memory()` (command pipeline + board drag) and
`agent_api.history_for_loop` (delegate wire) — one memory policy, both entry
points. `window()` stays the raw record. The `Brain` seam is untouched.
