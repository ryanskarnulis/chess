# The planner/narrator split — design note (2026-07-25)

`backend/src/chessapp/llama_brain.py`, `backend/src/chessapp/personality.py`.
Sprint 1, slice 2 of the agent-control replan
(`docs/agent-control-audit-2026-07-25.md`, item 15 — elevated from Sprint 4 to
P0).

## Why

One turn used to be made under one prompt: ~2,000 characters of Glitch — the
lexicon, the trolling contract, the swearing permission, the reaction-length
cap — wrapped around a short paragraph about tools, and then the model was asked
which tool to call. That is instruction competition, and it is the harness
audit's central diagnosis: on a 12B, tone and tool selection are drawing from
the same attention budget, and the tone is the bigger, louder half.

The fix is not to trim the personality. It is to stop asking one call to do both
jobs.

## What stays in the loop

Everything structural. The audit's first pass worried that re-splitting the turn
would undo #124, which replaced a single no-loop "phase one" call with a real
bounded loop and fixed multi-step failures with it. That worry doesn't apply:
what #124 won was the *loop*, and the loop is untouched.

- the iteration budget (`max_iterations`, 4) and the separate correction budget
- tool results fed back as `role: "tool"` messages on a growing prompt
- schema validation before dispatch; domain rejections as results, not errors
- `get_best_moves` → `make_move` in one turn, and undo-then-replace with it

One thing deliberately did *not* stay: the loop's turns no longer flip thinking
on after an analysis result. Picking (or declining) a tool is a parse whatever
is in context; the phase that reasons about an evaluation in words is the
narrator, so it alone inherits the flip. The old rule ran the reasoning twice —
live it showed up as a 26 s judgment turn, two thinking completions
back-to-back — and the split makes the second one the only one.

Only the *voice* left. The planner's first turn with no tool calls used to be
the commentary; now it is a one-line internal note, and the player never sees
it.

## When the loop ends the planning phase itself (`no_progress`, 2026-07-26)

One thing was added to the loop after the split, from a measurement rather than
a design: **a planner turn whose every tool call repeats one the turn already
made is its last.** Asked "what should I play?" with hints off — an ask whose
right answer, under the since-retired hints mode, was "no, hints are off" —
the planner re-ran the reads it had
already run (`evaluate_position → analyze_last_move → analyze_last_move →
evaluate_position`) and spent the whole iteration budget doing it: 2 of 20
samples in the recorded run, and a budget stop reaches no narrator, so the
player got the pipeline's canned stuck line about an ask nothing was wrong with.

The rule is deterministic and general, which is why it lives here rather than in
the prompt: a call identical to one already made this turn cannot bring anything
new back, so no further iteration can either. Three things it deliberately is
not:

- **Not a refusal.** The repeated call is still dispatched. Whether a repeat may
  *run* belongs to the tool layer, which already answers it (the phase machine
  refuses a second player move; a destructive op needs its confirmation), and
  the trace then shows the repeat that ended the phase instead of hiding it.
- **Not a budget stop.** Results came back, so there is something verified to
  speak from: this stop reaches the narrator exactly as `completed` does, and
  the player gets an answer. What it does not carry is a handoff note — the
  planner never wrote one — and the closing brief leaves that section out rather
  than presenting an empty heading as a note that said nothing.
- **Not the correction path.** A turn whose calls failed *schema* validation
  never dispatched anything, and repeating a malformed call is what the
  (smaller) correction budget is for; that budget keeps precedence.

## Every call is capped, and a cut-off turn is a failed one (2026-07-27)

Neither phase used to pass `max_tokens`, so llama-server ran `n_predict -1` and
a degenerate thought loop generated until the provider's 300 s read timeout —
observed live twice on 2026-07-27, 20k+ tokens on *ordinary planner calls*, the
player watching "thinking" for five minutes and the GPU still grinding after
the disconnect (cancellation does not reliably propagate through llama-swap).
Now each phase carries its own ceiling (`llama_brain._PLANNER_MAX_TOKENS` 2048,
`_NARRATOR_MAX_TOKENS` 4096), and `narrate` — the same phase on the fast path —
carries the narrator's.

The numbers are sized from measured output, generous side up, because clipping
a legitimate turn is the worse failure: the planner never thinks and its real
output is tool calls or a one-line note (tens of tokens), so 2048 is ~20×
headroom and bounds a runaway to ~30 s; the narrator's one thinking turn
legitimately reaches ~2.6k completion tokens (the repeat-stop token study in
`docs/agent-evals.md`; a live 2,408 in the 2026-07-27 trace), so 4096 keeps
every observed real narration intact and bounds a runaway to ~60 s. This is a
*safety ceiling* against the runaway, deliberately not the tighter wordiness
cap the token study weighed (the distributions overlap too much for one).

Truncation is handled, never forwarded — a `finish_reason == "length"` call may
come back with empty `content` and only `reasoning_content` behind it, or with
a mid-sentence fragment, and neither may travel:

- A **planner** turn the cap cut off ends the planning phase under the existing
  `no_progress` stop with the loop's own note — the fragment is never the
  handoff note. Results that landed before it are real and reach the narrator
  like any no-progress stop's.
- A **narrator** call (loop closer or `narrate`) the cap cut off returns empty
  text with its cost intact; the pipeline already composes its deterministic
  lines around an empty reply (the stuck line on the brain route, the move
  announcement on the fast path).

## What the planner runs on

`personality.planner_prompt_for()` — a compact, persona-free contract. It keeps
every load-bearing rule the old base layer carried about **acting**:

- the board and engine own truth and legality; never adjudicate in your head
- act only through tools
- every submitted move must be an entry in the injected `legal_moves`; map loose
  phrasing onto one of them and never invent a move
- ambiguous or missing information ⇒ no tool call, say what to ask instead
- describe only what the tools reported

Two bullets were added by the eval gate rather than the design (see the
measurement record in `docs/agent-evals.md`): *omit optional tool arguments —
the app derives the defaults* (the persona-free planner read "take that bishop
move back" as `undo(plies=1)`, something the persona prompt never did), and *a
failure result is yours to fix with another call* (without Glitch's
conversational instincts the bare contract gave up on the first illegal-move
rejection). Both are contract lines, not tone — the compactness tripwire in
`test_personality.py` still holds with them in.

…and drops everything about **speaking**, including the verbosity layer: the
planner has no words to lengthen or shorten. The advice line — an ask for a
hint routes to `get_best_moves` — is part of the standing contract since hints
mode retired (2026-09-01; it arrived as a hints-on-only layer, and the Sprint 3
era gated the tool out of the offer with hints off — the offer still resolves
per command off live state, `claim_draw` being the remaining case, and a call
outside it is a schema-level unknown that never dispatches). The prompt line is
orientation; what keeps advice honest now is the pipeline's evidence guard.

Its closing instruction is the handoff: one short factual line about what
happened or what needs answering, *never* addressed to the player.

## What the narrator runs on

The existing `system_prompt_for(verbosity)` — the full Glitch prompt,
unchanged, layers and all (the hints tone layer left with the mode,
2026-09-01). One model call, given:

- the transcript window
- the player's utterance
- the turn's tool calls and results
- the planner's handoff note

and **no tools**. That last point is the structural one: the phase that talks to
the player is physically unable to act, so audit item 9's "tool-free closing
pass" is not a rule the model is asked to follow — it is the shape of the call.
Sprint 2's slice for that item reduces to asserting it.

`Brain.narrate` — the fast path's commentary turn, which has always been exactly
this call — is now the same code path with a different brief
(`llama_brain._speak`). The split is the generalization of something the app was
already doing on its cheapest route.

### What the narrator is *not* given

The board. The brain has no session by design (it reads state through tools like
anything else), and the `board_state` it was handed at the top of the turn is
stale the moment a tool mutates anything — narrating from it would be worse than
narrating from nothing. So the tool results are the record of what changed,
exactly as they are for the fast path's `narrate`, where the caller passes a
freshly read board because it *has* one. When the coordinator's observe/close
beats land (slice 3), those call sites have a verified post-move board to hand
over, and this is the phase that will speak in them.

A side to play for. The observe beat runs while the engine's reply is still
being computed, from a board where it is the engine's move — so `turn`, the
`legal_moves` menu, and the FEN (whose string names the side to move) are
withheld from the narrator's view (`api._narrator_state_dict`, #193), and the
split `make_move` result stopped reporting the mid-exchange `fen`/`turn` it used
to carry. #188 established the rule by cutting "you are playing black" from the
brief; the next live game announced replies all the same ("My turn. ...Be6."),
because the data said what the prose no longer did. The memory rule is the same
fix's other half: a move turn is remembered by the reaction alone, never the
composed commentary whose trailing `\n\ne5.` is the app's announcement — given
those back verbatim, the narrator completed the format at the beat where the
reply does not exist yet (`docs/turn-memory.md`).

## The cost, and why it is accepted

A brain-routed turn is one model call more expensive than it was:

| turn | before | after |
| --- | --- | --- |
| fast-path move, verbosity=low | 0 | 0 |
| fast-path move, chatty | 1 | 1 |
| plain move through the agent | 2 | 3 |
| settings change by voice | 2 | 3 |
| clarifying question | 1 | 2 |

The fast path — the one an ordinary move takes — does not change at all, which
is what makes this affordable: the calls that got more expensive are the ones
that were already going to the model twice. The narrator's round trip is counted
in the turn's `model_calls` and tokens, so the trace and the eval baseline pay
for it honestly rather than hiding it.

Latency-wise the extra call is a short, tool-free completion with thinking off
(unless analysis landed — then the narrator is the *one* turn that thinks,
where the old loop thought twice), against a warm local model.
The bet the slice makes is that a cooler, smaller planner prompt buys more in
tool-call accuracy than one more round trip costs in seconds — which is
measured, not assumed: `CHESSAPP_PLANNER_TEMPERATURE` sets the planner phase's
temperature (unset = the provider's default, so nothing moved by default), and
the eval harness reads the same variable so a baseline recorded at a temperature
is the app's own behavior at that temperature.

## Eval gate

Every prompt change is eval-gated, and this one moves the prompt the tool
decisions are made under — precisely the kind of reshuffle that has moved
`long_capture` before. The recorded baseline in `docs/agent-evals.md` is
re-measured after this lands, including the two standing constraints: the
`long_capture` conditions, and no schema structure touched anywhere in the
slice.
