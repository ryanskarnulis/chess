# Agent harness audit — 2026-07-14

Scope: the agent's **context, instructions, and tool surface** — where the per-turn
budget goes, what in it is earning its keep, and what is bloat. Everything below is
measured against the shipping code (`main` @ 3c34364) and the real trace data
(`docs/traces-2026-07-13.jsonl`, 46 turns) — token figures are chars/4 estimates
from the actual strings the brain assembles, not guesses.

**Nothing here has been changed yet.** This is the findings document for the joint
review; each finding carries a recommendation and a rough saving, ranked by
impact-per-risk. The proposed order of attack is at the bottom.

---

## The headline number

One brain turn's opening prompt, verbosity=normal, mid-game:

| Component | Size | Share |
| --- | ---: | ---: |
| System prompt (`personality.system_prompt_for`) | ~1,520 tok | 38% |
| Tool definitions (15 tools, `registry.definitions`) | ~1,575 tok | 39% |
| Transcript window (20 turns, measured from real traces) | ~600 tok | 15% |
| Board state (`_agent_state_dict`, mid-game) | ~180–400 tok | 5–10% |
| The player's actual command | ~10 tok | **0.2%** |
| **Total** | **~4,000 tok** | |

**~98% of every request is fixed harness overhead.** For a 12B quantized model,
that overhead is not just latency and KV-cache pressure — it is instruction
competition: every token of persona, dead example, and schema noise is a token
arguing with the tool contract for the model's limited attention. The trace
review's failure modes (zero-tool fabrications, persona-toned lies) are exactly
what instruction competition looks like at this model size.

The realistic target after the diet below is **~2,600–2,800 tok/turn (a 30–35%
cut) with zero capability loss**, plus a structurally smaller prompt for the most
common turn type (the fast-path narrate).

---

## Findings, ranked

### 1. No token observability — the bloat is invisible in traces (fix first)

`provider.py` parses `usage` (prompt/completion/total tokens) off the wire into
`ChatResult.usage` — **and nothing ever reads it.** `trace.turn_record` records
route, tools, and FENs, but not how many model calls a turn cost, how many tokens
each one prefilled, or how long it took. The eval harness had to bolt on a
`CountingProvider` to count round trips because production counts nothing.

Every other finding in this report needed a scratch script to measure. That is
the tell: **we cannot see the thing we are trying to manage.**

**Recommendation:** thread per-model-call `{prompt_tokens, completion_tokens,
duration}` (and the call count) from the brain into `turn_record`. Do this
*before* any diet below, so every change lands with a before/after number in the
same JSONL we already review. Small, mechanical, no model behavior change.

### 2. System prompt: ~400 tok of it teaches phrasings the model never sees anymore

`personality._BASE`'s move-translation section (`personality.py:49-73`,
~1,600 chars ≈ **400 tok**) walks through "pawn to e4" → `make_move("e4")`,
"bishop takes on c6", "d takes e5", "castle kingside", promotions, and capture
phrasings — **all of which `fastparse.parse_move` now resolves deterministically
before the model is ever called.** Since the capture-family and castle work landed,
the brain only receives the leftovers: ambiguous moves, unfamiliar verbs ("grab
the pawn on e6"), mangled STT, and non-move commands. The prompt spends its
largest single block anchoring the model to a distribution it no longer receives.

What in that block is still live:
- the `legal_moves`-matching rule (translate to an entry in the injected list,
  never invent a string) — **keep**, this is load-bearing;
- the STT-repair guidance ("e 4" → "e4", "rook to a one" → "a1") — **keep**,
  the parser does *not* catch spaced squares or number words;
- "make_move at most once per turn / engine replies inside the call" — **keep**;
- "accepted proposal → call make_move now" — **keep** (real observed failure);
- the eight worked examples — **mostly dead.** Every example except the STT ones
  is a phrase the parser swallows.

**Recommendation:** cut the example list to 2–3 examples chosen from what
actually falls through to the model (an STT-mangled square, an ambiguous
capture, an accepted proposal). Saving: **~250–300 tok**, and better anchoring.
Eval-gated (`plain_move`, `long_capture`, `undo_and_replace` are the tripwires).

### 3. Persona is ~45% of the system prompt, and it's implicated in the failures

The layers: `_BASE` ≈ 846 tok, global Glitch personality ≈ **415 tok**, chess
flavor (trolling contract) ≈ **261 tok**. Persona is ~676 tok of a ~1,520-tok
prompt — sitting in context on *every* turn, including the tool-decision turns
where it contributes nothing and competes with the contract.

This is not just cost. The trace review's worst turns are persona-shaped: *"Word.
Game over."* (zero tools, live board), *"Bet."* + fabricated state. A 12B model
told at length to be a chill, deadpan, underreacting character will produce
chill, deadpan, underreacting *lies* — brevity instructions and honesty
instructions pull in opposite directions when the honest answer requires a tool
round trip and the vibey answer doesn't. The code-owned gate and honesty guard
now catch the destructive subset, but the pressure is still in the prompt.

**Recommendation (discuss — this is the one real design decision in the report):**
split the prompt by turn role.
- **Tool-decision turns** (the loop): `_BASE` contract + a two-line persona stub
  ("you are Glitch; keep replies short and casual — full character guide applies
  to commentary"). The loop's closing turn still reads it, so commentary keeps a
  voice, just from a compressed sketch.
- **Narrate turns** (fast path + confirmation commentary): the full persona, and
  much *less* of the contract (see finding 4).

If a split feels like too much machinery, the fallback is compression: the
chess-flavor layer alone can lose half its length (five bullet examples of
trolling technique is a lot of budget for "needle rarely, one dry line, never
pile on"). Saving: **~300–500 tok** per tool turn. Strictly eval-gated — persona
changes must re-run the whole baseline including the pass-rate scenarios.

### 4. `narrate` pays the full harness price for a one-line quip

The fast path is the most common turn type in real play, and above verbosity=low
each such move calls `Brain.narrate` — which sends the **entire** system prompt
(move-translation instructions, destructive-op contract, tool rules — none of
which apply, since narrate is offered no tools and never sees the utterance),
plus the 20-turn transcript, plus board + changes: **~2,300 tok of prefill for a
one-sentence comment.**

**Recommendation:** a dedicated narrate prompt — persona + "comment only on
these results and this board, never assert an outcome the results don't show" —
plus a *shorter* transcript window (banter continuity needs ~5 turns, not 20).
Saving: **~60% of the prefill on the most frequent LLM call in the app**, which
is also a straight latency win on every ordinary move. Low risk: narrate cannot
act, so the blast radius is tone.

### 5. Tool definitions: ~216 tok of pure pydantic noise, plus duplicated guidance

The 15 offered tools serialize to ~1,575 tok. Mechanical cleaning — stripping the
`title` keys FastMCP/pydantic stamps on every property and argument model
(`"title": "analyze_last_moveArguments"` teaches the model nothing), collapsing
`anyOf: [X, {type: null}]` to plain `X` (omission already expresses null for a
non-required arg), and dropping `default: null` — measures **865 chars ≈ 216 tok
(14%)** saved with zero semantic change. One helper in
`tools._derive_schema`, covered by existing schema tests.

Smaller irritants in the same layer:
- `undo` explains "omit for a normal takeback" **twice** — once in the docstring,
  once in the param description (`tools.py:444-459`). Say it once.
- `set_difficulty`'s description enumerates the tiers *and* the schema enum
  lists them again. The description can drop the list and keep the ~ratings.
- `make_move`'s description re-explains result semantics the model will see in
  the result itself.

**Recommendation:** schema-cleaning helper + a one-pass tightening of docstrings.
Saving: **~250–300 tok** total. Eval-gated but very low risk (the baseline
records zero schema corrections; keep it that way).

### 6. Tool surface: 15 tools is defensible, but three of them are one tool

Real-trace usage across 46 turns: `make_move` 10, `new_game` 3, then a long tail —
and `set_voice_output` **never** (it was also the one tool the trace review
flagged as unexercised). The three settings setters (`set_verbosity`,
`set_hints_mode`, `set_voice_output`, ~240 tok combined) are the same shape:
write one enum/bool to `Settings`.

**Recommendation (optional, product call):** merge into one
`set_option(name, value)` tool. Saves ~150 tok and shrinks the model's choice
space by two near-identical entries. Counter-argument: three self-describing
tools are harder to mis-call than one stringly-typed one — if the evals show any
`set_option` confusion, keep the three. This one is genuinely 50/50; flagging it
for the review rather than recommending it outright.

What should **not** change: the `BOARD_STATE_TOOLS` exclusion is measurably
working (zero wasted read calls in 46 traced turns; the eval harness bug that
briefly un-excluded them moved a scenario from 5/5 to 0–3/5 — evidence the
smaller offered list *matters* to this model). The analysis/game/save families
all earned their seats in the traces.

### 7. Transcript window: 20 turns is unexamined generosity

~600 tok of real-trace transcript per turn. The self-poisoning investigation
proved *length* isn't what breaks the model (20 real turns: no effect) — one
poisoned line is — so the window size is purely a cost/continuity trade.
Banter callbacks ("careful. bishop thing.") need a handful of turns, not twenty.

**Recommendation:** `DEFAULT_WINDOW_TURNS` 20 → 10 for the loop (saves ~300
tok), ~5 for narrate. Cheap, reversible, eval-gated by the `long_transcript_*`
scenarios which exist for exactly this.

Related note, no action needed: the window slides once the conversation exceeds
it, which invalidates the llama-server prefix cache below the system prompt each
turn. At 10 turns the re-prefill is smaller too.

### 8. Board-state injection is the one lean layer — leave it alone

`_agent_state_dict` measures ~180–400 tok mid-game and every field has a
documented reason to exist (`legal_moves` is the matching substrate,
`saved_games` is the self-poisoning fix, the UI-only `fens`/`dests` are already
excluded). The only unbounded field is `history` (~208 tok at 120 plies) — 
acceptable; if long games become a thing, cap it at the last ~30 plies with a
`"…N earlier moves"` marker, but that is not today's problem.

### 9. Rules still living prompt-side that the code should own (known, unfinished)

The evals doc already names it: **hints gating is the last prompt-honored
policy**, and it leaks (the one xfail — hints off, model hands over a move
anyway, invented rather than fetched). Same class as the destructive-op gate
before `_gate` existed: a prompt rule a 12B follows ~half the time.

**Recommendation:** code-own it. The cleanest shape mirrors the existing
exclusion machinery: when `hints_mode` is off, don't *offer* `get_best_moves`
(registry already supports per-caller `exclude`), and let the prompt's only job
be tone. Bonus: with hints off the offered list drops another ~108 tok.
The honesty guard's scope is also worth one discussion beat: it catches
game-over/new-game lies, but the traces also showed an invented PGN and invented
move advice — fabrications that aren't destructive claims and so pass the guard.
Fresh-fact injection (the `saved_games` pattern) is the general cure; worth
asking what else the model can currently deny/invent that the state block
doesn't answer. (Candidate: it has no idea what `verbosity`/`hints_mode`
currently *are* — settings state is not injected, so "are hints on?" is a guess.)

### 10. The trace data itself is stale — re-baseline before we tune

All 46 traced turns predate the capture family, the resign route, play-as-black,
`analyze_last_move`'s color fix, the self-poisoning fix, and the honesty guard.
The 38-brain/7-fast route split that shaped this audit's cost model is certainly
different now — most capture phrasings and every resignation have since left the
model's hands. If current performance still feels bad, the *first* diagnostic
question is which route today's bad turns take.

**Recommendation:** one fresh traced session (`CHESSAPP_TRACE_PATH`) as the
first artifact of the joint review — ideally after finding 1 lands, so it
arrives with token counts attached.

---

## Minor / hygiene

- **Double validation:** every tool call is jsonschema-validated in
  `llama_brain._dispatch` *and* again in `registry.dispatch`. Harmless at this
  scale, but the two produce their error strings independently — drift risk.
  Let the brain validate (it owns the correction budget) and have `dispatch`
  trust or share the helper.
- **`_wire_correction` user-role injection** is sound (nothing to attach a tool
  result to), just worth remembering it exists when reading transcripts: a
  "user" message the user never typed.
- **`Settings` state isn't in the agent's view** — see finding 9's candidate.

## Proposed order of attack (for tonight's review)

| # | Change | Saving | Risk |
| --- | --- | ---: | --- |
| 1 | Token/latency observability in traces (finding 1) | measurement | none |
| 2 | Fresh traced session on current `main` (finding 10) | diagnosis | none |
| 3 | Schema cleaning + docstring tightening (finding 5) | ~300 tok | very low |
| 4 | Dead-example cut in `_BASE` (finding 2) | ~275 tok | low, eval-gated |
| 5 | Slim narrate prompt + short narrate window (finding 4) | ~60% of the most common call | low |
| 6 | Transcript window 20→10 (finding 7) | ~300 tok | low, eval-gated |
| 7 | Persona split/compression (finding 3) | ~300–500 tok | **medium — discuss first** |
| 8 | Hints gating in code (finding 9) | +correctness | low |
| 9 | `set_option` merge (finding 6) | ~150 tok | 50/50 — discuss |

Items 3–7 together take the brain turn from ~4,000 to ~2,600–2,800 tok. Every
one of them goes through the eval gate (`CHESSAPP_AGENT_EVALS=1`), and after
item 1 every one of them lands with a measured before/after in the trace log.
