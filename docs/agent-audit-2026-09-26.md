# Agent audit: harness, context, prompt and loop engineering — 2026-09-26

Audited `main` at `aa02bd1` (#360). No application code changed. No live-model
evals were run: GPU numbers below come from the repository's own records
(`docs/agent-evals.md`, `docs/knight-ask-campaign.md`, open issues), and each is
cited. Token figures are chars/4 estimates from the strings the app actually
builds (`brain_tool_definitions`, `planner_state`, `PLANNER_PROMPT`,
`system_prompt_for`), rendered in a scratch script against this checkout.

> **Where the findings went (added later on 2026-09-26).** They are now steps
> of roadmap #366:
>
> - Finding 1 (#361) is covered by #372, the story of the game.
> - Finding 2 is #362 (step 2).
> - Finding 3 (#298) became the model bake-off (step 17).
> - Finding 4 is #363 (step 7).
> - Findings 5 (#364) and 6 are part of #370, the lean planner.
>
> The same day's redesign also decided to retire the runtime honesty guard in
> favour of an offline speech-accuracy score (#367, #368). This audit
> predates that decision.

Scope: *what we are doing wrong compared with current agent-engineering
practice*, not a re-run of the 2026-09-05 audit's bug hunt. That audit's loop,
tool and eval findings are not repeated here. Where current practice matches
what this repo already does, the audit says so and moves on.

## Summary

The architecture is sound and ahead of most published practice. Deterministic
code owns truth. The planner and narrator run as separate calls with a typed
handoff. The loop only appends. Budgets are structural, and every agent turn is
traced and can become an eval. The frameworks-are-not-the-answer conclusion of
`agent-framework-research-2026-07-14.md` still holds.

What is wrong is narrower, and most of it lives at the edges the prompt work
never touched:

| # | Finding | Area | Cost to try | Expected lever |
|---|---|---|---|---|
| 1 (#361) | The planner reads Glitch's prose as its *own* past turns: a few-shot demo of "answer without calling a tool" | context | small | reliability in long threads |
| 2 (#362) | The planner and narrator prompts evict each other's KV prefix every turn (likely; measurable today) | harness / serving | small | latency |
| 3 (#298) | Months of prompt arms against a model ceiling; the model swap is still gated behind them | loop / model | medium | reliability, the biggest unknown |
| 4 (#363) | A/B arms are independent samples; seeded pairing (common random numbers) would resolve smaller effects per GPU-minute | evals | small | measurement power |
| 5 (#364) | Keep the tool offer byte-stable; pin the one board-varying tool last | context | tiny | latency (protects #2) |
| 6 | The "done" round trip: a second planner call writes an ~11-token note the harness demotes | loop | measure first | latency |

The order is information per effort. #1 and #2 are cheap and neither needs a
prompt change. #3 is the one most likely to move the open comprehension issues
(#338, #348, #357), which prompt and schema arms have repeatedly failed to move.

---

## 1. The planner is shown Glitch's words as its own history

**Read.** The brain route passes `ctx.transcript.memory()` straight into
`LlamaBrain._messages` (`api.py` near the `brain.get_agent_response` call site).
That memory is `conversation.condense`: the last four turns verbatim as
`user`/`assistant` messages, with the assistant side holding the *narrator's*
commentary. The planner then reads that as its own prior turns. Rendered with
three ordinary turns, the planner's input is:

```
system     "You are the tool-calling layer of a chess app. …"
user       'play e4'
assistant  'bet. clean opener, bro.'
user       'how am I doing'
assistant  "You're fine fr, dead even."
user       'take the knight'
assistant  'Sheesh, snagged it. Nasty.'
user       'Board state:\n{…}\n\nCommand: grab the pawn on e6'
```

Tool calls are never replayed (by design, `docs/turn-memory.md`). So in the
only context the planner ever sees, the model role has answered `play e4` and
`take the knight` **in slang and without calling a tool**, over and over. For a
12B, that is a set of in-context demonstrations that contradict its system
prompt. The planner/narrator split fixed tone competition in the *system*
prompt (`long_capture[poisoned]` 1/5 → 5/5). It left the same competition in
the *history*, which the split's own rationale says is where the model looks
first.

**Why it matters: the repo's own evidence is thread-conditioned.** The failures
that reproduce only in a live thread and never fresh are exactly the ones this
explains:

- `constraint_survives_a_live_thread` 12/20 in its thread versus 29/30 fresh
  (`agent-evals.md`, "the condition, not the words").
- `long_capture[poisoned]` (release-blocking), cured by the split but still the
  scenario with the least margin.
- The self-poisoning family in the 2026-07-13 trace review.

**Current practice.** This is now a named failure mode. *Mitigating
Conversational Inertia in Multi-Turn Agents* (arXiv 2602.03664) shows models
attending diagonally to, and imitating, their own prior responses as few-shot
examples. Manus's context-engineering write-up gives the same advice as "don't
few-shot yourself into a rut". Cognition's "share full agent traces, not just
individual messages" is the positive form: the actor should see what the actor
did.

**Recommendation.** Give the planner its own rendering of history. The narrator
keeps the verbatim roles, which is right for a voice. Two arms, in cost order:

- **A. History as data, not as roles.** Fold the recent turns into the opening
  user message as a quoted block ("Recent conversation — Player: … / Glitch: …")
  above `Board state:`. Then no assistant-role prose exists anywhere in the
  planner's context. This keeps `condense`'s policy intact, is about ten lines
  in `_messages`, and leaves the system and tool prefix (the cacheable part)
  untouched.
- **B. History as the planner's own trajectory.** Replay each recent turn's
  actual tool calls as assistant `tool_calls`, with a compact result per call.
  The trace and `StoredMessage` already hold them. This is the Cognition form
  and gives the model correct demonstrations. It costs more tokens, and it needs
  care, because a replayed `legal_moves`-bearing result would reintroduce the
  stale-board copy `turn-memory.md` forbids. Replay the call and its
  `ok`/`legal`/`san`, never board state.

Screen both against control with `probe_planner.py`, seeded with the real
threads (`_LIVE_TRANSCRIPT`, the constraint thread), on `long_capture` ×3,
`constraint_survives_a_live_thread`, `two_threads_similar_asks`, and the
neighbours. This is a context-shape change, not a prompt-text change, so it
does not collide with the "every arm that added a fact made it worse" lesson.
It removes facts the planner should never have been reading.

## 2. The two phases probably evict each other's KV cache every turn

**Read.** The planner prompt is `PLANNER_PROMPT` (~490 tokens) plus 17 tool
schemas (~2,870 tokens). The narrator prompt is the Glitch prompt (~770 tokens)
with no tools. They differ from the first token. A brain turn runs planner →
planner → narrator, and the next turn starts with the planner again. The shared
server runs `--cache-ram 0` with `q8_0` KV (`knight-ask-campaign.md`,
"Decisions"). With one slot, every phase switch discards the other phase's
prefix, and nothing on the host can restore it. #357's trace shows the planner
paying for a full ~3,050-token prompt on the turn's first call.

The cached prefix *within* a phase does hold: the loop only appends, which is
correct. The loss is *across* phases and turns, and it is paid on every brain
turn and every fast-path narration.

**Evidence status: inferred, and checkable without a GPU run.** The trace
already records `cached_tokens` per call (#317). Read it off the deployed trace
for planner calls that follow a narrator call. If it is near zero, this finding
stands. The slot count is in the serving manifest (`total_slots`).

**Current practice.** Manus calls KV-cache hit rate "the single most important
metric for a production-stage AI agent". The 2026 literature, for example
*Don't Break the Cache* (arXiv 2601.06007), treats it the same way.

**Recommendation, in cost order:**

- **Two slots, one per phase.** `--parallel 2` on the shared llama-server, then
  pin `id_slot` per phase in `provider._payload`, or let llama-server's
  prompt-similarity slot selection do it. The KV pool is sized by `-c`, not by
  the slot count, so this costs no extra VRAM. Each slot gets `ctx/2` (65k with
  the current 131k), still twice `input_budget_tokens`. The llama-swap config
  is shared across the fleet, so this is a fleet decision.
- **Or re-enable the host prompt cache** (`--cache-ram`), so an evicted prefix
  is restored from RAM instead of re-prefilled.

Quality risk looks low on the repo's own data: arm A of the knight-ask campaign
(`cache_prompt: false`) and the fresh-versus-warm comparison both found caching
does not change the planner's decisions. So this should be a pure latency
change. Gate it with `plain_move` and `judgment_question` timing and
`latency_report.py`.

## 3. Too much prompt work against a model ceiling; the model swap waits behind it

Across #269, #286, #319, #338, #348, #351 and #357, the planner-understanding
campaigns end in the same place: "no lever" (#357: `said_the_square` 0/20 on
every push-verb ask), wording moves the number more than tier does (#339), and
fixes that land only as *typed* levers (`ask_player`'s enum, `source`,
temperature 0.3). Planner thinking was tried and cost `undo_replace` 4/20
(`agent-evals.md`, the 2026-09-24 table). The prompt comments record why each
sentence stayed or went, arm by arm. That is careful work, but it is prompt
hill-climbing on a 12B whose comprehension is the binding constraint. The 12B
reads "push my e pawn" as naming a square 20/20, and no wording changes that.

Meanwhile #298 (a different brain model) is gated as "only if A–D fail", scoped
to the knight ask alone, and still open. Everything needed to run it already
exists: the `Brain`/`ChatProvider` seams, `probe_planner.py --arm
x:model=<id>`, the frontier tier with dev and held-out splits, and a
`qwen36-35b-a3b` entry already registered in llama-swap.

**Current practice.** 2026 small-model tool-calling moved fast. Qwen reports
BFCL v4 scores of 66.1% for Qwen3.5-9B, a model that fits the card (a vendor
number, not measured here). Practitioner reports describe Gemma 4 malforming
and looping on tool calls under long-context load across llama.cpp, vLLM and
Ollama. Neither report is evidence for *this* app, but together they make the
swap the largest untested lever.

**Recommendation.** Run #298 now, and broaden it: the whole frontier corpus
plus the gated suite, block-interleaved (the card is exclusive), with the
preconditions #298 already lists (the model's own `top_p`/`top_k`, a
thinking-toggle check, `long_capture` ×3). Also consider a split-model arm:
planner on the tool-strongest model that fits, narrator on Gemma for the voice.
Only take it if both fit in VRAM together, since a llama-swap swap between
phases would erase the gain. Until #298 reports, treat further planner-prompt
arms on the 12B as low expected value.

## 4. A/B evals: pair the arms on the sampling seed

**Read.** No request sets `seed` (`provider.py`, `probe_planner.py`,
`evalstats.py`), so every sample of every arm is an independent draw. The repo
already knows the variance problem well. The same unchanged prompt read 5/20,
19/40 and 31/40 in three batches, and day rates swing 60–85%. The mitigation is
interleaving, which cancels drift but not per-sample noise.

**Current practice.** For comparing two prompts on one eval set, use a paired
design: common random numbers (the same seed per sample index in both arms),
then McNemar's test on discordant pairs. When the arms mostly agree, which is
the normal case for a one-sentence or one-field change, the paired test detects
effects that an unpaired two-proportion comparison needs several times the
samples for.

**Recommendation.** Add a per-request `seed` to `LlamaCppProvider` (llama-server
accepts it), pass the sample index as the seed in `probe_planner.py` and the
campaign harness for both arms, and report discordant counts plus a McNemar or
exact-binomial p next to the raw counts. Caveats, stated up front:

- llama.cpp is not bit-reproducible across batch sizes or slots
  (ggml-org/llama.cpp#7052), so pairing is approximate. It is still never worse
  than unpaired.
- Pairing is strongest for small changes. Once the prompt differs, the RNG
  streams diverge from the first differing logit.

Keep interleaving, since it answers a different question (drift). This does not
touch the gate's floor or the sequential rule.

## 5. Keep the tool offer byte-stable; pin the varying tool last

**Measured.** The planner offer is ~11.5k chars (~2,870 tokens), about 80% of
every opening planner prompt. The board state is 120–280 tokens from move 1 to
ply 84. The offer changes with every board, because `ask_player`'s `candidates`
enum is the live `legal_moves`. It currently sits **last**, so consecutive
boards' offers diverge only at ~97% through (char ~11,200 of ~11,500). That is
why #315's per-board offer costs little cache today. It is also an accident of
registration order that no test pins. `claim_draw` is inserted in the middle of
the list when a draw becomes claimable, which shifts every schema after it.

**Recommendation.** Add a unit test asserting that `ask_player` (the one tool
whose schema varies with the board) is the last offered tool. Consider always
offering `claim_draw` and refusing it in the handler (`retry: never`) when
nothing is claimable, instead of withholding it. That is Manus's "mask, don't
remove": the action space stays stable for the cache and for the model's
references to earlier calls. The second change alters the offer, so it runs the
eval gate. It strips no schema key, so it is consistent with the standing
prohibition on schema minimization.

## 6. The "done" round trip (measure before changing)

**Read.** A plain brain-route move is three calls. The second is a planner call
that re-reads the whole prompt to produce the handoff note: #357's trace shows
3,140 prompt tokens and 11 completion tokens. Since #289 the harness builds the
record itself (`handoff.build`) and labels the note "the planner's reading …
(not a record)". The note is load-bearing only on `reply` turns, where no tool
ran.

The cost of this call depends on #2. With the prefix cached it is ~100 new
tokens of prefill plus a short decode, a few hundred ms. Cold, it is a full
prefill. Terminal-tool shortcuts (LangChain's `return_direct`) are the usual
fix, but they are risky here: a planner that emits `undo` alone and then
`make_move` in a second iteration is exactly the composition an early exit
would cut off. The multi-undo history in `llama_brain.py`'s docstring is the
cautionary tale.

**Recommendation.** First read the deployed trace: after an all-`ok`,
mutation-only batch, how often does the next planner call add tools? If
effectively never after a successful `make_move`-only batch (the phase machine
already refuses a second player move), end planning there and let the handoff's
own record stand in for the note. Otherwise, leave it alone once #2 has made it
cheap.

---

## What is already right, and should not change

- **Workflow, not autonomy.** The four routes, three of them deterministic,
  match Anthropic's "workflows before agents" guidance. The model routes
  language and nothing else.
- **The split.** A tool-free narrator that receives a harness-built record
  matches the 2026 convergence (an orchestrator that owns context, plus
  isolated sub-calls that return compressed results). Because the narrator only
  verbalizes, the "share full traces" objection does not apply to it. It does
  apply to the planner (#1).
- **Append-only loop, results-keyed stall rule, typed clarification,
  board-refresh as a planner-only message.** These are the practices the cache
  and context-engineering literature asks for.
- **Typed levers over prose.** `ask_player`'s enum and `make_move`'s `source`
  are the "reasoning in the schema" pattern, and the evidence here keeps
  favouring them over instructions.
- **No framework.** The 2026-07-14 conclusion stands. Nothing above needs one.

## Sources

- Anthropic, [Effective context engineering for AI agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents); [Writing effective tools for AI agents](https://www.anthropic.com/engineering/writing-tools-for-agents)
- Manus, [Context Engineering for AI Agents: Lessons from Building Manus](https://manus.im/blog/Context-Engineering-for-AI-Agents-Lessons-from-Building-Manus)
- Cognition, [Don't Build Multi-Agents](https://cognition.com/blog/dont-build-multi-agents); [Multi-Agents: What's Actually Working](https://cognition.com/blog/multi-agents-working)
- [Mitigating Conversational Inertia in Multi-Turn Agents](https://arxiv.org/html/2602.03664v3) (arXiv 2602.03664)
- [Don't Break the Cache: An Evaluation of Prompt Caching for Long-Horizon Agentic Tasks](https://arxiv.org/pdf/2601.06007) (arXiv 2601.06007); [Spheron: KV cache and prefix caching guide](https://www.spheron.network/blog/context-engineering-production-ai-agents-kv-cache-long-context/)
- llama.cpp [server README](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md) (`seed`, `id_slot`, `--parallel`); [nondeterminism across slots, #7052](https://github.com/ggml-org/llama.cpp/issues/7052)
- Paired evaluation: [Your A/B eval is paired. Your stat test probably isn't.](https://dev.to/alex_spinov/your-ab-eval-is-paired-your-stat-test-probably-isnt-lbk); [Resolution Diagnostics for Paired LLM Evaluation](https://arxiv.org/html/2605.30315v1)
- Local tool-calling landscape (secondary, vendor-reported numbers): [Best local models for tool calling 2026](https://www.promptquorum.com/power-local-llm/best-local-models-tool-calling-2026); [Gemma 4 bug reports, HF forums](https://discuss.huggingface.co/t/gemma-4-bug-fixes-and-research-request/176979)
