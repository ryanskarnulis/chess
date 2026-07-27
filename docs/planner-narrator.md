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
right answer is "no, hints are off" — the planner re-ran the reads it had
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
planner has no words to lengthen or shorten. Hints mode still reaches it, as one
line about `get_best_moves` when hints are on — but since the Sprint 3 gating
slice the *permission* itself is not the prompt's: with hints off the tool is
withheld from the offer entirely (the offer, like the prompts, resolves per
command off live settings), so a call at it is a schema-level unknown that never
dispatches. The prompt line is orientation; the code is the gate. The same
setting reaches the narrator too, as tone.

Its closing instruction is the handoff: one short factual line about what
happened or what needs answering, *never* addressed to the player.

## What the narrator runs on

The existing `system_prompt_for(verbosity, hints_mode)` — the full Glitch
prompt, unchanged, layers and all. One model call, given:

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
