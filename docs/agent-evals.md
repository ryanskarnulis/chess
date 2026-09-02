# Agent eval harness

`backend/tests/test_agent_evals.py` — golden command→tool-call scenarios run
against the **real** model (gemma-4-12b behind llama-swap) over a **real**
Stockfish engine. This is chess's copy of the workspace agent standard's
opt-in eval harness (`../agent-standard/STANDARD.md` §6, mirroring PCC's
`tests/test_agent_evals.py`): the tripwire that gates model, prompt, and loop
changes.

## What it is

Each scenario drives one utterance through the **same seam a conductor's
delegate call uses** — `POST /api/agent/conversations/{id}/messages` — against
a freshly assembled app (real `LlamaBrain`, real engine, fresh game). It then
asserts on the `MessageExchange` wire the endpoint returns:

- **trajectory shape** — the right tool family ran, and read-only asks mutate
  nothing. `assistant_message.tool_calls` carries each call's
  tool/arguments/result/error; a successful call has `error: null`, and for
  `make_move` a genuinely-played move also has `result.legal == true` (a
  rejected move is a `legal: false` *result*, never an `error`).
- **board end-state** — read back through `GET /api/state` (the same document
  the web board renders): did the board change, or not?
- **`stop_reason`** — `completed` vs a budget stop (`max_iterations` /
  `correction_limit`).
- **cost** — how many times the *model* was called for one utterance, and
  whether thinking was on for each turn. The live `LlamaCppProvider` is wrapped
  in a `CountingProvider` (`tests/fakes.py`) — the one seam every round trip
  passes through, since nothing in production counts them — so the round-trip
  budget and the thinking policy are asserted, not just eyeballed.

Assertions are behavioral, never exact call sequences: the model is sampled at
temp 1.0 (see `../agent-standard/model-profile.md`), so goldens pin tool
*families* and end-state, never wording.

**Fast-path guard.** Chess short-circuits an utterance that parses as exactly
one legal move (`fastparse.parse_move`) with zero LLM calls. Every scenario
asserts `parse_move(utterance, fen) is None` in its setup, so the eval stays a
*model* eval even if the parser grows later (e.g. "play e4" falls through
today; bare "e4" does not). The same is now true of an explicit resignation
(`fastparse.parse_resign`) — see the retirement note below.

Scenarios are independent: a function-scoped fixture builds a fresh app + game
+ conversation each time; the Stockfish process is module-scoped (one engine
for the suite) and the model stays warm on llama-swap between scenarios.

## Running it

Opt-in, like the provider smoke — skipped unless `CHESSAPP_AGENT_EVALS=1`, so
CI and default `pytest` never touch the GPU:

```bash
cd backend
CHESSAPP_AGENT_EVALS=1 .venv/bin/pytest tests/test_agent_evals.py -v -s
```

`-s` shows the per-scenario `[eval]` stats line (scenario, stop_reason, tool
call count, board-mutation count, duration, trajectory) the baseline table
below is built from. The first call may cold-load the model (~100 s per the
model profile); everything after runs warm.

Overrides via the standard env vars: `LLAMACPP_BASE_URL`
(default `http://127.0.0.1:8200/v1`), `LLAMACPP_MODEL` (default `gemma-4-12b`),
`CHESSAPP_STOCKFISH` (default `/usr/bin/stockfish`).

Sampling and reporting knobs (all optional; the defaults are what a normal gate
run uses):

| Env var | Default | What it does |
| --- | --- | --- |
| `CHESSAPP_EVAL_RUNS` | 5 | Samples per block, and therefore the minimum count. |
| `CHESSAPP_EVAL_MAX_RUNS` | 20 | Escalation cap. |
| `CHESSAPP_EVAL_INFRA_RETRIES` | 5 | Provider deaths re-taken per scenario before `INFRA_ABORTED`. |
| `CHESSAPP_EVAL_INFRA_BUDGET` | 25 | The same ceiling across the whole suite. |
| `CHESSAPP_EVAL_REPORT` | — | Path to a JSONL report; unset writes none. |

A 20-sample measurement campaign is `CHESSAPP_EVAL_RUNS=20
CHESSAPP_EVAL_MAX_RUNS=20` — a block equal to the cap forces the full count, so
there is no separate "force" knob to forget.

## The gating rule

**Run this harness before merging any prompt, model, or loop change, and the
baseline must not regress.** A prompt tweak that degrades tool honesty (a
judgment question answered from vibes, an invented move, a lost clarifying
question) does not ship. If the baseline legitimately shifts (a better prompt,
a new model), re-record the table below in the same PR and say why.

**What a green means, precisely: _not statistically below the floor_.** Not "the
model does this ≥80% of the time" — the gate cannot resolve that from the sample
sizes a shared 12 GB card affords. See the next section for what it can and
cannot see.

## The verdict is judged, not compared (2026-07-26, Sprint 5 slice 3, audit 23)

`assert result.rate >= 0.8` is gone, because over five samples it is a coin flip
at the floor and the record proves it: `long_capture[poisoned]` measured 3/5 then
4/5 on one build (red once, green once), `main` gave 5/5 then 3/5, and
`play_as_black` gave 5/5, 2/5, 0/5, 5/5 across four same-build runs. A gate that
flaps gets re-run until green, which is how the 2026-07-25 baseline came to be a
hand-assembled composite of re-runs.

**The rule.** Every count is judged against a **one-sided 95% Wilson bound**
(`z = 1.645`; the FAIL test uses one tail, so calling a two-tailed 1.96 bound
"95%" would overstate the evidence by a whole tail). `evalstats.decide` returns
one of three things:

| Decision | When | What the harness does |
| --- | --- | --- |
| `BELOW_FLOOR` | Wilson **upper** bound < floor | Red. This is evidence of regression, not a bad sample. |
| `ABOVE_FLOOR` | point estimate ≥ floor | Green. |
| `UNDECIDED` | interval straddles the floor, point estimate under it | Take another block of 5, to a cap of 20. |

At the cap, `UNDECIDED` passes but is **flagged** in the printed summary and the
report — except on a release-blocking item (`long_capture`), where an unresolved
gate is not a pass.

**At n=5 and floor 0.8 the whole table is six numbers** — the upper bounds for
0–5 passes: **0.351 / 0.565 / 0.728 / 0.857 / 0.954 / 1.000**. So 0–2 fail
immediately, **3 escalates**, and 4–5 pass immediately: a healthy suite costs
exactly what it cost before this slice, and the extra samples are spent only on
the one count that was genuinely ambiguous. Sampling is in **blocks of five**
rather than one at a time for three reasons: identical operating
characteristics, a decision table small enough to unit-test
(`tests/test_evalstats.py`), and per-block rates — the only measurement in the
suite that can see the run-order confound below.

**Raising the run count against `rate >= floor` measures _worse_, and this is
written down so nobody re-proposes it.** At a true rate of exactly 0.80,
fixed-5 goes green **0.737** of the time and fixed-20 goes green **0.630**: four
times the GPU for a worse gate. (Both are exact binomial figures, not
simulations.)

**What the change buys, at the suite level** (15 pass-rate items, exact
enumeration of the rule above):

| true rates across the suite | today (fixed 5) | sequential, cap 20 |
| --- | --- | --- |
| all 0.97 | 0.880 | 0.996 |
| 12@0.97, 2@0.90, 1@0.85 | 0.636 | 0.939 |
| 10@0.97, 3@0.90, 2@0.80 | 0.387 | 0.788 |

Roughly **3× fewer spurious reds for ~4% more samples** (78 vs 75 expected on the
middle row).

### The limits, stated rather than papered over

- **The gate separates ≈0.95 from ≈0.5–0.6 and nothing finer.** Against a true
  drop from a 0.8 floor to 0.7 / 0.6 / 0.5 it goes red only 0.32 / 0.59 / 0.79 of
  the time. A green is weak evidence of health; a red is strong evidence of harm.
- **Never lower a floor for quiet.** Power depends on the floor, so 0.8 → 0.7
  roughly halves the chance of catching a real drop to 0.6. Moving a floor to
  make a test green is the move that hollows out a tripwire.
- **Optional stopping does bias toward false passes, slightly.** At a true rate
  of 0.60 the sequential rule passes 0.414 against fixed-5's 0.337. The leak is
  the 4/5 early accept, which fixed-5 has identically; the gate's own power table
  above is stated *after* that leak, not before it.
- **The i.i.d. assumption is known false, and this slice does not fix it.**
  `play_as_black` measured 0/5 only mid-suite and 5/5 in isolation. Twenty
  consecutive samples of one utterance is a cluster of size 1, so Wilson
  understates the width. Fixing it means interleaving blocks across scenarios —
  inverting the suite into a session-scoped sampler — which is **unpaid work,
  named here as such**. What the slice does instead is make the confound
  *measurable*: per-block `(passed, runs)` in the report, an `UNSTABLE` flag when
  the block spread reaches 0.6 (quiet on `long_capture`'s 5/5-vs-3/5), and
  per-sample `model_ms`, because a prompt-cache hit is fast and correlating
  outcome with latency tests the hypothesis at zero GPU cost. (That last one was
  printed per sample but **not written to the report** until 2026-07-26, when it
  landed with the per-call split — see *Model time, per call and per phase*.)
  **What the flag cannot see, stated plainly:** it compares blocks *within* one
  scenario's sampling, and only where escalation bought a second block. A
  mid-suite 0/5 decides red on block 1, so it is reported STABLE — correctly,
  since nothing varied within that run. The 5/5-in-one-run-vs-0/5-in-another
  shape is therefore found by comparing report lines *across* runs, which is why
  the campaign below measures a scenario both isolated and mid-suite.

### Infrastructure is retried, not scored

llama-server crash-restarts every 3–8 minutes of sustained generation. Since
audit item 20 the brain *catches* `ProviderError` and answers **200** with
`stop_reason="provider_error"`, so a crash stopped looking like a 502 and started
looking like a silent behavioral miss ("expected a resign call: nothing").

`evalstats.classify` sorts each sample, and the precedence is the point:

| Outcome | From | Scored? |
| --- | --- | --- |
| `PASS` / `FAIL` | the check held / raised on a healthy turn | yes |
| `INFRA` | non-200, **or** 200 + `stop_reason="provider_error"` naming a *transient* failure — including when the check also failed, because it failed *because* the provider died | no: thrown away and re-taken |
| `PROVIDER_REJECTED` | 200 + `stop_reason="provider_error"` naming a **non-transient** failure — llama-server answered, just not with a completion | never: fails the item on the **first** one |
| `INCONCLUSIVE` | a budget stop under a commentary-only check (`_assert_reached_narrator`) | counted as a non-pass, and reported as what it was |
| `HARNESS` | any non-`AssertionError` exception (a `KeyError` in a check) | never: fails the item at once, after writing the samples already taken |

The harness reads the real stop reason off a `_CollectingTracer` on the app's own
`tracer` seam — which is why the *sorting* needed no production change:
`api._run_command` already traces every route, so the harness gets `stop_reason`,
`route`, `mutations`, `guarded`, `model_calls` and `model_ms` on *both* seams for
free. Adding `stop_reason` to the panel response instead would have put a
production edit inside a harness slice, and the harness-bug precedent below (a
wrong tool offer silently moving a scenario from 5/5 to 0–3/5) is why that is not
allowed.

#### Which death it was (2026-07-26)

`provider_error` alone said the turn died, not why, and the loop's two
`except ProviderError` clauses discarded the exception — so a crashed socket and
an HTTP 400 arrived here identically and the harness retried both. It now carries
`provider_failure`, a **field** naming the kind
(`provider.ProviderFailure` → `AgentResponse.provider_failure` → the trace
record), and `classify` splits on it:

| Kind | Is | Retried? |
| --- | --- | --- |
| `unreachable` | connect refused, reset, timeout — the socket never answered | yes |
| `server_error` | a 5xx; llama-swap mid-restart, the crash cadence | yes |
| `rejected` | a 4xx — **a context overrun on a long transcript is this one** | no |
| `malformed_response` | 200, body failed wire validation; version skew | no |
| `bad_tool_arguments` | arguments that aren't a JSON object (the brain corrects this inside the turn, so it never reaches a stop) | no |

Unknown means retry, deliberately: an unclassified death (and every non-200,
which carries no kind because the *app* answered, not the provider) stays
`INFRA`. Spending a few samples on a deterministic failure is cheap; calling a
restarting server deterministic would abort a whole suite on the first crash.
Code owns that answer rather than a caller inferring it from a message — the same
rule the tool layer's `retry` field keeps. `evalstats` re-declares the
non-transient set as literals (it must stay importable with nothing installed,
like `STOP_PROVIDER_ERROR`), and `test_evalstats` pins the two vocabularies
against each other so the duplication cannot drift.

Exhausting the retry budget is `INFRA_ABORTED`: **no rate, a hard failure**, and
it now means only what it says — llama-server is not staying up. The
deterministic case is no longer folded into it after five wasted retries.

### The report

`CHESSAPP_EVAL_REPORT=/tmp/evals.jsonl` writes one JSONL line **per scenario as
it finishes** — crash-survivable, the tracer's own precedent — behind a header
line carrying the knobs, the floors, the model and the git SHA, and followed by a
suite line with totals, wall clock and the infra budget consumed. *A baseline
should say what budget produced it.* Each scenario line carries the counts, the
blocks, the interval, the decision, the stability flag, the infra retries, the
failure-mode histogram and the per-sample record. The tables in this file are
meant to be assembled from that file, not from terminal scrollback.

#### Model time, per call and per phase (2026-07-26, Sprint 5)

The `[eval]` line and every per-sample record now carry the turn's model time
broken down rather than summed:

| Field | Is |
| --- | --- |
| `model_ms` | the turn's whole model time — the sum, as before |
| `call_ms` | one reading per round trip, **in call order**, off the trace's `model_latencies_ms` (which includes the calls that *raised* — the brain measures them, and the provider seam cannot) |
| `planner_ms` | the planner loop's time: every reading but the narrator's |
| `narrator_ms` | the narrator's own round trip |
| `phases` | how confidently those two were separated — `SPLIT`, `NO_NARRATOR`, `NONE` or `UNKNOWN` |

The readings existed on the trace record all along; what was missing was the
attribution, and the sum could not answer the question it was being asked. A
`no_progress` turn narrates 2–3× slower than a `completed` one at the same
model-call count (the finding below), and "the narrator's round trip is slow" and
"this sample was hard for the model, which is *why* it both repeated and rambled"
add up to the same `model_ms`. Separating them needs the parts.

`evalstats.split_latencies` derives the attribution from the route and the stop
reason — unit-tested off the GPU like the rest of the verdict — and **refuses to
guess**:

- `SPLIT` — the narrator is the last call. True on a `brain` route that stopped
  `completed` or `no_progress` (both reach the narrator), and on the narrate
  routes (`fast_path`, `board`, `resign`, `confirmation`, `control`), where there
  is no planner loop at all and the planner's 0 is real.
- `NO_NARRATOR` — a budget stop (`max_iterations`, `correction_limit`) reaches no
  narrator, so every reading is the planner's and narrator time does not exist.
  Attributing the last call to a narrator here would invent time never spent.
- `UNKNOWN` — `provider_error`: the call that died could have been either phase
  and the record does not say which. A median over guesses is worse than a
  median over fewer samples.
- `NONE` — no readings (a zero-LLM canned confirmation, or a request that died
  before the pipeline traced anything; the sample's outcome tells those apart).

Two notes on why this shape. It is **per sample, not aggregated**, because the
open question is whether the slow samples and the `no_progress` samples are the
*same* samples — which needs the pairing kept. And **nothing in `src/` changed**
to get it: the same proof-of-innocence rule the statistics slice worked under,
for the reason the harness-bug precedent below records (a wrong tool offer in the
harness silently moved a scenario from 5/5 to 0–3/5). What the slice did add is a
test for the harness's own reporting seam — `tests/test_eval_harness.py`, GPU-free
— because an instrument that can silently lie about what it measures is worse
than no instrument.

#### Tokens, per call and per phase (2026-07-27, Sprint 5)

The other half of the same round trip. `call_ms` settled **where** a repeat-stop
turn's extra 30 s goes — the narrator's own call — and cannot settle **why**,
because "emitted three times the tokens" and "generated at a third of the rate"
are the same milliseconds. So the line and the record now carry what each call
wrote:

| Field | Is |
| --- | --- |
| `call_in` / `call_out` | prompt and completion tokens per round trip, **in call order**, off the harness's call meter (`fakes.CountingProvider` reads `ChatResult.usage`) |
| `planner_out` | completion tokens the planner loop wrote |
| `narrator_in` | the narrator's **prompt** size — a candidate mechanism in its own right (a `no_progress` turn dispatched the duplicate, so its narrator reads one more tool result, and a longer prompt costs prefill before a token is written) |
| `narrator_out` | completion tokens the narrator wrote |
| `prompt_tokens` / `completion_tokens` | the turn's totals, derived from the parts — the same sums the trace record reports, from the other seam |
| `narrator_tok_s` | `narrator_out ÷ narrator_ms`: **the discriminator** |

`narrator_tok_s` is what the instrument is for. A narration that ran 3× longer at
the same rate wrote 3× the tokens, and the standing candidate fix — bounding the
narrator's thinking budget — is aimed at the right thing. The same tokens at a
third of the rate is the server, and a token cap would not touch it.

Three rules it keeps, all of them the latency split's:

- **The phase boundary is derived once.** `evalstats._attribute_phases` states
  the `SPLIT` / `NO_NARRATOR` / `UNKNOWN` / `NONE` rule over call *positions*,
  and both `split_latencies` and `split_tokens` use it. Two clauses on one line
  disagreeing about which call was the narrator would make the line unreadable
  in exactly the case it exists to explain.
- **Unmeasured is `?`, never 0.** A round trip that raised reported no usage —
  the result never came back — so its phase total is unknown rather than
  smaller. Summing the rest would print a partial number as though it were the
  whole.
- **The two seams are printed, not reconciled.** Tokens come off the call meter
  and milliseconds off the trace (the brain measures a call that raised; the
  provider seam cannot). They sit either side of `model_calls`, so a
  disagreement about how many round trips happened shows on the line instead of
  being silently folded into a wrong attribution.

Again **nothing in `src/` changed**: every round trip already passes through the
meter. `tests/test_eval_harness.py` and `tests/test_evalstats.py` pin it GPU-free.

A line now reads:

```
[eval] scenario=hints_off_no_advice status=200 stop=no_progress route=brain calls=2 model_calls=3
       thinking=[off,off,off] model_ms=41200 call_ms=[1200,900,39100] planner_ms=2100 narrator_ms=39100
       phases=SPLIT call_in=[2100,2400,2900] call_out=[8,12,940] planner_out=20 narrator_in=2900
       narrator_out=940 narrator_tok_s=24.0 mutations=0 duration=41.5s trajectory=[...]
```

#### The trajectory carries arguments (2026-07-27, Sprint 5)

`trajectory=[get_best_moves → set_hints_mode]` names the tools and not what they
asked for, and there is a live suspect behind that blind spot: across four
`hints_off_no_advice` runs on 2026-07-26 a `set_hints_mode` call appears on
**9/65** samples where the player asked only *"what should I play here?"*.
Whether it turned hints **on** — a setting the player owns, changed by an agent
that was asked a question, which no gate covers today — was unknowable from the
line. So each token now renders as `name(arg=value, …)`:

- **JSON values**, so `enabled="true"` (a mis-invocation) and `enabled=true`
  (the call) do not read alike.
- **Keys sorted**, so the same call renders identically in every sample: the
  line is read by scanning many samples for a difference, and emission order
  would manufacture ones that aren't there.
- **Values cut at 24 chars**, visibly — one trajectory shares a terminal line
  with eleven other clauses.
- **No parens when a call took no arguments** (`get_board_state`), because empty
  parens on every read tool are noise on every line.
- **Both failure markers are `!` suffixes after the arguments.** The rejected
  move's was `make_move(illegal)`, which beside real arguments would read as an
  argument named `illegal`; it is `make_move(move="Qh8")!illegal` now, and a
  dispatch error keeps its bare `!`.

The per-sample report record carries the same string, which is what makes "how
many samples called `set_hints_mode`, and which way" a query over the report
rather than a hand tally across four runs of scrollback. Nothing in `src/`
changed here either — `agent_api._tool_call_read` has always put `arguments` on
the wire, and only the reporting path was dropping them.

**This does not settle the finding, it makes it decidable.** If the flip is
real, the fix is a capability question rather than a prompt line, and the
scenario's `check` should assert `app.ctx.settings.hints_mode is False` after
the turn.

## Scenarios — what each pins

| Scenario | Utterance | Pins |
| --- | --- | --- |
| `fast_path_low` | "e4" (verbosity=low) | The one scenario that *does* parse: the fast path dispatches the move and answers with a canned confirmation — **zero model calls**. |
| `fast_path_normal` | "e4" (verbosity=normal) | Above verbosity=low the fast path still skips the loop and pays for commentary only: **exactly one** model call (`Brain.narrate`), thinking off. |
| `plain_move` | "play e4" | Parser lets the verb-prefixed move through; exactly one **legal** `make_move`; board becomes `[e4, <engine reply>]`; `completed`. **3 model calls** (tool turn + planner handoff + narrator), thinking off throughout. |
| `judgment_question` | "how am I doing?" (4 plies in) | A judgment question routes through `evaluate_position`/`analyze_last_move` — never answered from vibes; **zero board mutations**; non-empty reply. Also the live thinking-policy pin: every planner turn is thinking **off**, and the narrator — the one turn that reasons about the result — is thinking **on**. |
| `ambiguous_move` | "move the rook" (both rook files open) | Genuine ambiguity → asks instead of guessing: **no legal `make_move`**, board unchanged, non-empty clarifying reply. |
| `settings_by_speech` | "make it easier" | `set_difficulty` called successfully toward a **weaker** setting than the `casual` default; no board mutation. |
| `honest_illegal` | "castle kingside" (illegal move 1) | No fabricated legal move; board unchanged. An attempted-and-rejected `make_move` (`legal:false`) is fine — only the board-didn't-change invariant is asserted, never wording. |
| `destructive_confirm` | "new game" (mid-game), then "yes" | Destructive op must not fire on the first ask: **no successful `new_game` this turn**, game intact, and the agent relays the gate's refusal as a question. The follow-up "yes" must then actually reset the board. **Hard assert** — was the suite's one xfail; see the closed finding below. |

## Move correctness through the model (was: the known blind spot)

The suite used to have exactly one scenario in which the model picked a move at
all — `plain_move` ("play e4", from the starting position), the easiest case
there is — while `ambiguous_move` and `honest_illegal` only asserted the board
*didn't* change, which is safety, not correctness. 8/8 green did not mean the
agent played the move you asked for.

The **pass-rate scenarios** close that. They come out of the 2026-07-13 trace
review (`docs/agent-trace-review-2026-07-13.md`: three real games, 46 traced
turns), each one a real position and a real utterance that misfired, replayed.
They assert the **specific SAN that must land**, so they can fail on the model
being *wrong* rather than on it calling the wrong tool.

**They report a rate, not a boolean.** The model samples at temp 1.0: a single
assert on a path that works 70% of the time flaps, and a flapping test teaches
you nothing. `_pass_rate` samples each scenario in blocks of `_BLOCK_RUNS` (5) on
a fresh app and prints the count, its interval and its verdict; the floors live in
one table (`_FLOORS`, all 80% today). This is not theoretical —
`capture_names_victim` ("bishop takes pawn") measured **0/5, 2/5, 3/5 and 5/5**
across four runs of the same build. A boolean assert there would have told you
four different things on four days.

**And the rate is judged, not compared to a literal** — see "The verdict is
judged, not compared" above for the rule, the numbers behind it, and what it
cannot see. `assert result.rate >= 0.8` was itself a flapping gate.

**There are no xfails left.** The convention, if one is ever needed again: an
xfail here means known-broken, not flaky-broken (`strict=False`, each carrying
its finding number from the trace review, so a fix flips it green instead of
silently passing). Three scenarios have held one and all three are closed
below — the two tool-signature bugs (`play_as_black`, `my_mistake_is_mine`), and
the hints leak, which was a different animal: not a tool that *can't* express
the ask, but a policy (hints off) left to the model to honor, the same shape as
the destructive-op rule before the gate replaced it. It got the same treatment —
deterministic code, not a prompt line — in two rounds, the capability cut
(slice 1) and the scoped advice guard (slice 4).

### `play_as_black` is closed (2026-07-13) — 0/5 → 5/5

`new_game(player_color=…)` puts the player on the side they asked for, and the
engine — now owning white — opens, so the board isn't left waiting on a move only
it can make. Omitted, they keep the side they have. **This closes the advertised
conductor handoff**, whose deep-link intent string
(`/?intent=let's+play+chess+as+black`) is this scenario's utterance verbatim: it
was broken end to end, because the tool the intent routed to could not say
"black".

One subtlety the tests pin: the requested color **rides along in the armed op**.
The confirmation gate replays `pending.args` on the player's "yes", so a gate that
dropped the color would confirm "new game, I'll take black" into a game as white —
silently discarding the only thing the player actually asked for.

### `my_mistake_is_mine` is closed (2026-07-13) — 0/5 → 5/5

The first of the two signature bugs is fixed and its xfail is now a **hard
assert**. `analyze_last_move(color=…)` defaults to `session.player_color`, so
"what was my mistake?" analyzes the move the player actually made. The old
no-arg tool always analyzed the literal last ply — which, on the player's turn,
is *always* the engine's reply — so the tool could not answer its own
docstring's question. The model's response to that was not to fail loudly: told
"I meant MY last move, the c3 one", it re-called the same no-arg tool and
reported the engine's `Bxe4` **as if it were c3**. That is the house rule's
signature failure mode — an ask the tool cannot express becomes an ask the model
fabricates compliance with — and it is fixed the way `resign`'s color was, in
the signature rather than the prompt. The override (`color="black"`) exists so
the player can ask about the *other* side without the model guessing which.

### The capture family is retired, not lost (2026-07-13)

`capture_bare_bishop`, `capture_bare_pawn`, `capture_names_victim` and
`capture_names_square` are **gone from the suite**, and their rows are gone from
the baseline below. They were not dropped for flapping — they were *fixed at the
root*. `fastparse.parse_move` now resolves any capture phrasing built on
takes/captures, whether it names the piece ("bishop takes"), the victim ("bishop
takes pawn") or the square ("take the h6 pawn"), whenever exactly one legal
capture fits. The board settles it, the model is never called, and there is
nothing left to sample: an eval that cannot reach the model is not an eval.
`capture_names_square`'s 5/5 was deliberately given up along with the 0/5s — a
deterministic pass beats a sampled one.

That coverage now lives in `backend/tests/test_fastparse.py` (free, exhaustive,
run on every commit), including the cases where the phrase fits *two* legal
captures and must still fall through to the agent's clarifying question.

One knock-on: `long_capture` said "take the e6 pawn", which the parser now
swallows, so its utterance moved to **"grab the pawn on e6"** — an unfamiliar
verb keeps the capture in the model's hands, which is the point of that probe.

## Self-poisoning: the transcript scenarios, and what they found

Three failures seen in live play — `resume_game` denied ("the system doesn't
support reloading saved games"), `resign` answering "Word. Game over." with zero
tool calls on a live board, and "take the h6 pawn" praised but not played — all
passed **5/5** on a fresh delegate conversation. The standing suspicion was that
**a long transcript degrades tool recall**. The `long_transcript_*` scenarios were
written to test that, and they drive the **web panel seam** (`/api/command`, which
reads `ctx.transcript`) rather than the delegate wire — the delegate opens a fresh
conversation per call, which is exactly what was hiding this.

Each probe runs under three conditions, so the transcript is the only variable:

| condition | what it is |
| --- | --- |
| `fresh` | empty transcript, verbosity normal — the control |
| `live_like` | the real 20-turn thread from the game that broke, at verbosity=low |
| `poisoned` | `live_like` + the agent's own earlier turn saying the thing failed |

**The suspicion was wrong, and the real cause was worse.**

| probe | fresh | live_like | poisoned (before) | poisoned (after the fix) |
| --- | --- | --- | --- | --- |
| `long_resume` | 5/5 | 5/5 | **0/5** ✗ | **5/5** ✓ |
| `long_resign` | 5/5 | 5/5 | 5/5 | 5/5 |
| `long_capture` | 5/5 | 4/5 | — (no poison) | **4/5** ✓ |

Transcript **length is not the cause** — 20 real turns changes nothing. What
broke `resume_game` was a *single prior assistant turn, in the model's own
transcript, in which it said saving failed* (the real one from the trace: "I can't
save the game right now because the save directory isn't set up"). With that one
line present it stopped calling `resume_game` **entirely** — 0/5 — and confabulated
a justification for a file that was sitting on disk:

> "I can't load a game named 'scholars' because it hasn't been saved yet."
> "I can't load that game because it doesn't exist in the system."

That is **self-poisoning**: the model reads its own earlier failure as a fact
about what the app can do, and never re-checks. It was never really a bug about
any one tool — it is the house rule again, **the model deciding something
deterministic state already knows.** Whether a saved game exists is a question the
filesystem answers, and until now the app never answered it: the agent's own prose
was the *only* thing in its context that claimed to know.

**The fix (2026-07-13):** `saved_games` is now in the board-state block the brain
is handed every turn (`api._agent_state_dict`, read fresh via
`tools.saved_game_names`), sitting next to `legal_moves`. Nothing is rewritten or
dropped from the transcript — the stale sentence is still there, it just now has
to argue with a fresh fact, and the fact wins: **0/5 → 5/5**. The deterministic
guard is in `test_command.py` (`test_a_past_failure_in_the_transcript_cannot_
suppress_the_fresh_fact`), because CI never runs this suite.

**Negative result — the capture miss is _not_ self-poisoning.** `long_capture`'s
`poisoned` condition was seeded with the two real "Illegal move." turns and still
lands `Bxe6` (4–5/5, floor 80%). That closes the standing hypothesis, and it is
the evidence the fix above rests on: `legal_moves` is *already* injected fresh
every turn, and stale prose cannot argue it down. Fresh state beats stale prose.
`long_resign` likewise passes in all three conditions. Both live failures are
therefore **still unexplained** — not length, not a declined destructive op, not
self-poisoning. They stay hard asserts, so a recurrence fails here.

**And on 2026-07-13 (capture slice) it recurred** — `long_resign[poisoned]` came
back **2/5** (3/5 on a `main` re-run), the agent answering *"I'm calling that. Game
over, bro."* with **zero tool calls** on a live board. That is now fixed, and not by
asking the model more nicely: see the resign retirement below. Both live failures
that opened this section are therefore closed — `resume_game`'s by a fresh fact in the
prompt, `resign`'s by taking the decision off the model entirely.

## The resign family is retired too (2026-07-13, resign slice)

`resign_never_pretends` and all three `long_resign[*]` conditions still exist and still
pass, but they no longer measure the model: **"I resign" never reaches it.** An explicit
resignation is deterministic text — the utterance either concedes or it doesn't — so
`fastparse.parse_resign` settles it and the pipeline dispatches `resign` itself, on its
own fourth route (`trace.ROUTE_RESIGN`), with zero model calls. The call still goes
through the registry, so the confirmation gate arms it and asks; a *mis*-parse therefore
costs a question, never a game. That is what lets the parser be more generous here than
`parse_confirmation` dares to be.

This is the capture family's story again, one rung more dangerous: a path the model got
right about half the time is now a path it is not on. The real coverage moved to
`test_fastparse.py` (which phrasings are a resignation) and `test_command.py` (the route
calls `resign`, the gate arms, "yes" ends the game) — free, exhaustive, every commit.
The eval rows stay as a **tripwire on the route**, not on the model: if the parser ever
stops catching "you know what, I give up. I resign", they go red.

**The class is closed separately, and that is the part that generalizes.** The parser
fixes *resign*; it does nothing about the rest of finding 6 (commentary announcing a
checkmate that didn't happen after a quiet move). So the pipeline now also refuses to
*say* what didn't happen: at the one point all four routes converge, commentary that
asserts the game ended or restarted, when no destructive tool succeeded and the board is
still live, is replaced with the truth and the turn is traced as `guarded`
(`honesty.claims_destructive_outcome`, `test_honesty.py`). The gate stops the model
*doing* a destructive op unasked; the guard stops it *claiming* one. Neither is a prompt
rule, so neither is a coin flip.

## The planner/narrator split (2026-07-25) — the baseline shifts, and long_capture closes

The split (`docs/planner-narrator.md`) moves the tool loop onto a compact,
persona-free planner prompt and adds one narrator call — the Glitch voice, no
tools — after it. Two consequences for every number in this file:

- **Brain-routed turns cost one more model call** (the planner's handoff turn
  plus the narrator, where the old closing turn was both). The fast path is
  untouched: `fast_path_low` is still zero-LLM, `fast_path_normal` still one.
- **Thinking moved.** Planner turns never think — picking a tool is a parse —
  and the narrator alone flips ON when an analysis result landed. The old rule
  ran two thinking completions per judgment question; live that measured 26 s
  once. `judgment_question` now pins `[off, off, on]`.

**`long_capture` is cured, and it was the prompt all along.** The same-day
pre-split baseline on this machine measured fresh 5/5, live_like 4/5, poisoned
**1/5** (the standing RED from #136). Post-split, all three conditions measured
**5/5**, repeatedly, at both measured planner temperatures. That confirms the
harness audit's instruction-competition diagnosis — the capture was drowning
under a page of persona, not under noisy context — and it means Sprint 3's
structured-errors slice inherits a green scenario to *keep* green rather than a
red one to cure.

**The persona-free contract needed two rounds of eval-driven iteration** —
recorded here because both failures are lessons about what Glitch's prose had
been quietly doing:

1. `ambiguous_move` regressed first (the planner *played* a rook move instead
   of declining). The persona prompt's "ask one short clarifying question"
   instinct had been load-bearing; the bare contract's "make no tool call"
   wasn't enough. Fixed by telling the planner what to do instead of act
   ("reply with one short line saying what the player must be asked") and
   softening the job line to "decide which tool calls, *if any*". 5/5 after.
2. `undo_and_replace` then regressed (2/5): the literal-minded planner read
   "take that bishop move back" as `undo(plies=1)`, left black to move, failed
   to play d4, and gave up. Three fixes: an omit-optional-arguments contract
   line, a fix-your-own-failures contract line, and a clarified `undo`
   docstring ("for any normal takeback omit plies — the app pops the full
   exchange"; the golden updated with it). 25/25 on a dedicated run after,
   4–5/5 in full-suite runs.

**Planner temperature was measured, and the default did not move.** The knob is
`CHESSAPP_PLANNER_TEMPERATURE` (unset = the model profile's 1.0), read by app
assembly and by this harness alike. At 1.0 and at 0.3, every accuracy scenario
(`plain_move`, `ambiguous_move`, `undo_and_replace`, `my_mistake_is_mine`,
`play_as_black`, `resume_not_denied`, all three `long_capture` conditions)
passed at or above the floor; 5-run sampling cannot distinguish the two, and a
default that moves on data that can't resolve a difference is a default moved
on vibes. It stays at 1.0 until a measurement says otherwise.

**One infrastructure caveat on this record.** llama-server crash-restarts under
sustained suite load (`upstream exited unexpectedly` in the llama-swap log,
roughly every 3–8 minutes of continuous generation; the 12 GB card sits at
~10.8 GiB). A crashed run answers 502 and the harness counted it as a failed
sample, so the 2026-07-25 record is a composite of consecutive runs in which
every 502-hit scenario was re-run to completion. **Every failure in every run
was a 502; not one was a behavioral miss.** The server flags are owned by
`../llama-swap/config.yaml`, not this repo; filed there.

> **The manual composite procedure is retired (2026-07-26).** Do not
> hand-assemble a baseline out of consecutive runs any more: the harness retries
> an infra death itself, bounded and counted, and reports how much budget it
> spent (see "Infrastructure is retried, not scored"). A crash is no longer a
> failed sample, so there is nothing left for a re-run to launder — and a record
> assembled by hand cannot say how many deaths it took, which is precisely the
> number a reader of this file needs. Baselines from 2026-07-25 and earlier are
> composites; every later one carries its infra count.

## The hints leak, diagnosed (2026-07-25, Sprint 3 slice 4) — 1/5 → 5/5

`hints_off_no_advice` was red on `main` across three consecutive slices, and the
two candidate explanations from the backlog were "the evidence test is too
loose" and "something changed in #155". **The measurement picked the first, and
then found a second hole nobody had proposed.** Both are now closed in code, and
both are the same house rule: the model may not decide what the app already
knows.

Isolated 5-run rates, same GPU session, in the order they were taken:

| Build | Rate | What the failing runs did |
| --- | --- | --- |
| Clean `main` (6bbd446, #156) | **1/5** | `evaluate_position → analyze_last_move`, then the narrator names 2–3 legal moves (`['Bc4', 'c3', 'd3']`) |
| `main` minus #155's `settings` block | 4/5 | one failure, same shape; three of the four "passes" were `max_iterations` stops that never reached a narrator |
| Scoped licence (`_reported_moves`) | 3/5 | both failures ran `evaluate_position → set_hints_mode` — **the planner turned hints on itself** |
| Scoped licence + turn-start snapshot | **5/5** | — |

**The evidence test was too loose.** `_advice_evidence` was a boolean: any
successful `get_best_moves`/`analyze_last_move` switched the advice guard off
entirely. The exemption exists so "what was my mistake?" can name the move
played and the move that beat it — but a planner that answers "what should I
play here?" with a mistake analysis was thereby licensing every *other* legal
move too, which is the whole legal-move list. `api._reported_moves` replaces it:
a result licenses exactly the SANs it reported (`get_best_moves`' candidates,
`analyze_last_move`'s `played` and `best`), and those moves — no others — come
off the list the guard checks. The mistake-analysis exemption survives intact,
including when the better move is still playable now.

**And the model was granting itself permission.** With the licence scoped, the
failures changed shape: the planner called `set_hints_mode(True)` and then
handed over moves, and the guard — reading `ctx.settings` *after* the loop —
stood down for a setting the player had switched off and the model had switched
on. The setting that governs a turn is now the one in force when the player
asked, snapshotted before the brain runs. App assembly already resolves the tool
*offer* off that same instant, so the two can no longer disagree; the settings
change itself stands and takes effect from the next turn.

Both halves are pinned deterministically in `test_command.py` (a licence scoped
to reported moves, the mistake-analysis exemption, and the self-granted-hints
case), so CI holds the line the eval only samples.

**A note for the eval-statistics slice.** Three of the four passes in the
second row above were `max_iterations` stops — a budget stop reaches no
narrator, so there is no commentary to leak and the run scores as a pass. The
scenario's pass rate therefore mixed "behaved correctly" with "ran out of
budget", which is part of why it read as noise. Worth a look when that slice
lands.

> **Closed (2026-07-26, eval-statistics slice).** Those passes were **vacuous**,
> and the harness can no longer score one: `hints_off_no_advice` is the suite's
> one purely-negative check (nothing in it asserts that a tool ran), so it is the
> one scenario sampled with `requires_narrator=True`. A budget stop there now
> raises `VacuousRun`, classifies as `INCONCLUSIVE`, counts against the rate and
> is printed and reported as what it was. Two consequences worth expecting on the
> next campaign: this scenario's measured rate may read **lower** than the
> historical numbers above, and that is the vacuity being subtracted rather than
> a regression — the comparison to make is against a re-measurement, not against
> a number that counted untested runs as passes. The budget stops themselves are
> filed as a **separate defect**: a question that should be declined in one turn
> burning four planner iterations is a loop problem, not a floor to slide.

## Turn memory: the transcript scenarios now measure a digest (2026-07-25, Sprint 4 slice 1)

`/api/command` no longer hands the model the raw 20-turn window. It hands
`Transcript.memory()` — the last four turns verbatim behind a deterministic
digest of what the player asked for earlier (`docs/turn-memory.md`). That
changes what the `long_transcript_*` conditions above actually put in front of
the model, so it is worth being precise about what they still measure:

- **`live_like`** is now the digest condition. Its 20-turn thread reaches the
  model as 10 messages instead of 40 (1,440 → 1,049 chars), with sixteen turns
  of Glitch's prose replaced by twelve quoted player requests. The probe still
  answers "does a long thread degrade tool recall?" — it just now measures the
  thread as the app actually sends it.
- **`poisoned`** still measures self-poisoning, because every probe's poison
  turns are appended at the *end* of the thread and therefore land inside the
  verbatim window. That is deliberate and now load-bearing: a stale assistant
  line can only poison what it is still quoted in, and the sharpest form of the
  test is the one where it is.

**Result: no movement.** All nine long-transcript conditions 5/5, including
`long_capture[poisoned]` — the release-blocking regression — in a run where the
whole suite came back clean first time. The digest neither fixed anything nor
broke anything at the tool boundary; what it bought is a memory whose size stops
growing, and a planner context with sixteen fewer turns of personality in it.

## Recorded baseline

**Run 2026-09-01 on the hints-retirement build (PCC #314): 23 passed in a
single run, 3 m 18 s, every pass-rate scenario 5/5 ABOVE_FLOOR STABLE, infra
0.** The slice removes a tool from the planner's offer (`set_hints_mode`),
adds one line to the planner contract (the advice path), and deletes the
narrator's hints tone layer — prompt and schema changes on every turn, so the
full gate was owed and run. The release blocker holds (`long_capture` 5/5 in
all three conditions), the schema tripwire holds (`undo_and_replace` 5/5),
and the cost scenarios are unmoved (`fast_path_low` **0 model calls**,
`fast_path_normal` 1, `plain_move` 3).

**`advice_is_engine_backed` replaces `hints_off_no_advice`, and this run is
its first baseline: 5/5, floor 0.8.** Same position (e4 e5 Nf3 Nc6), same
utterance ("what should I play here?"), the contract inverted with the mode's
retirement: the turn must consult `get_best_moves`, not touch the board, and
name only moves an analysis tool reported — the pipeline guard's licensing
rule, measured as the model's own discipline. All five samples ran the
textbook turn, `trajectory=[get_best_moves(n=3)]`, `stop=completed`, 3 model
calls (planner ~0.8–1.6 s, thinking-on narrator 6–18 s): the ask that used to
burn the whole iteration budget re-running reads it had no use for (the
`no_progress` motivation, 2/20 in the 2026-07-26 run) now resolves in one
planner call, because the tool that answers it is finally on the table. The
old scenario's records below keep their name; its last measurement under the
mode was 19/20 (2026-07-27).
release-blocker holds — `long_capture` 5/5 in all three conditions (fresh,
live_like, poisoned), plus the three cost scenarios the slice could plausibly
have disturbed: `fast_path_low` **0 model calls** (a plain move is still
zero-LLM), `fast_path_normal` 1, `plain_move` 3.** Not a full-suite run, and by
the #148/#149/#153/#160/#161 precedent the gate is not strictly triggered — no
prompt, model, tool-schema or control-flow change; what the model sees is
byte-identical. But the slice touches the loop file (the brain now reports which
phase it is entering) and moves the pipeline's blocking steps into worker
threads, so the sequencing was measured rather than argued: the phase reports
are side effects around the existing calls, and the thread hop changes when the
loop is free, never the order anything happens in. The two cost scenarios are
the ones that would notice if it had.

**Run 2026-07-25 on the turn-memory digest build (Sprint 4, slice 1): 23 passed,
every pass-rate scenario 5/5, in a single run — the baseline holds.** Third
clean full-suite pass on record, and the first on the condensed context. Every
scenario that reads a transcript (`long_resume`, `long_resign`, `long_capture` ×
fresh/live_like/poisoned) is a direct probe of the change; all nine are 5/5. See
the section above for what each condition now measures.

**Run 2026-07-25 on the verified-facts guard build (Sprint 3, slice 5): 23
passed, every pass-rate scenario 5/5, in a single run — the baseline holds.**
Second clean full-suite pass on record, and the one that matters most for this
slice: it widens the honesty guard from ending claims to every operational
fact, so a regression would show up as *suppressed* commentary in any scenario,
not just the honesty ones. Nothing moved. `long_capture` 5/5 in all three
conditions (the release-blocking constraint), `hints_off_no_advice` 5/5 (the
advice guard the new classes sit beside), `undo_and_replace` 5/5. The
analysis-touching scenarios (`my_mistake_is_mine`, `long_capture` ×3) were
re-run 5/5 against the final build after a late loosening of the evaluation
class's number matching.

The guard's own risk is the opposite of a pass-rate: a class that fires on
honest commentary. That was measured separately and off the GPU, by sweeping
`unverified_claims` over the **46 recorded live turns** in
`docs/traces-2026-07-13.jsonl` with facts reconstructed per record. Two false
positives surfaced and were fixed before merge — reciting the move list tripped
the `move` class (the game's history is board truth and now rides in the facts)
and a PGN read-out tripped `evaluation` (a `2023.10.27` date is not a score) —
leaving **2/46 guarded: exactly the two known lies**, "Word. Game over." and
"you actually have me in checkmate", both on a live board. Worth re-running
that sweep against a fresh trace corpus when Sprint 5 re-baselines one.

**Run 2026-07-25 on the advice-guard build (Sprint 3, slice 4): the whole
suite is green in a single run — 23 passed, every pass-rate scenario 5/5, no
502s, no xfails.** First clean full-suite pass on record (previous baselines are
composites of re-runs). `hints_off_no_advice` measured 5/5 twice: isolated, and
again in-suite. Nothing else moved; `undo_and_replace` (the schema-cut
tripwire), `play_as_black`, `my_mistake_is_mine` and all three `long_capture`
conditions were 5/5 in the same run.

**Run 2026-07-25 on the structured-tool-errors build (Sprint 3, slice 3):
22/22 hard scenarios pass; `hints_off_no_advice` measured 2/5 — and it is
**already red on `main`**, not a regression this slice caused.** The slice's own
gate is green: all three `long_capture` conditions 5/5, plus `illegal_move_honest`
and `ambiguous_move`. Everything else in the suite was 5/5 in the same run,
including `play_as_black` and `undo_and_replace` (the schema-cut tripwire).

The `hints_off` number was A/B'd three ways rather than assumed, because this
slice touches the planner prompt (one line, replacing the vague "correct it with
another call" rule with a pointer at the new `retry`/`alternatives` keys).
Isolated 5-run rates, same GPU session:

| Build | Rates |
| --- | --- |
| This slice, whole suite | 2/5 |
| This slice, isolated | 3/5, 2/5 |
| This slice with the prompt line reverted | 0/5, 1/5, 1/5 |
| Clean `main` (aeb1aea, #155) | 2/5, 0/5, 2/5 |

So the prompt line is not the cause — reverting it measures *worse*, and `main`
measures the same. The recorded 5/5 from slice 1 does not reproduce today at
all. **Filed as its own backlog item**; the leading hypothesis is #155 (the
per-turn `settings` block), which landed after slice 1 recorded 5/5 and whose
merge note does not mention an eval run — but that is untested, and the
run-to-run spread here (0/5 to 3/5 on one unchanged build) is exactly the floor
noise the eval-statistics slice exists to fix, so the cause could equally be
neither.

The mechanism behind every failure is the same and worth writing down: the
failing runs all call `analyze_last_move` alongside `evaluate_position`, which
satisfies `api._advice_evidence` and switches the advice guard off, letting the
narrator name moves. The exemption is meant for analysis the player *asked*
for — but "what should I play here?" is not a request for mistake analysis, so
the guard's evidence test is too loose. That is the honesty-guard slice's
territory (Sprint 3's next item), not this one's. **It was — and the diagnosis
above it holds: closed by slice 4, which scoped the licence and found a second
hole behind it.**

**Re-confirmed 2026-07-25 on the hints-gating build (Sprint 3, slice 1)**, and
the suite's last xfail is gone: `hints_off_no_advice` went **0/5 → 5/5** and is
a hard assert now. Two things did it, both code: with hints off
`get_best_moves` is withheld from the brain's *offer* (the harness mirrors
this), and the pipeline's advice guard replaces commentary that names a
currently-playable move with `MOVE_ADVICE_REPLY` when no analysis tool the
player asked for reported it. The capability cut alone measured 2/5–3/5 — the
model still invented moves from its own head with the tool gone — so the guard
is what makes the scenario deterministic. Composite per the 502 procedure:
22/22 hard scenarios green across consecutive same-build runs (`my_mistake`
4/5, everything else 5/5 in its green run), zero xfails remaining.

**One new observation from this composite, filed for the eval-statistics
slice:** `play_as_black` measured 5/5, 2/5, 0/5, 5/5 across four same-build
runs — 0/5 only when run mid-suite, 5/5 both times in isolation, failures all
"asked for black, got white" with no 502s. That is not floor noise (0/5 is not
a coin flip on a true ~80% rate); it looks run-order / server-state dependent
(prompt-cache or load effects on the shared GPU).

> **Picked up (2026-07-26, eval-statistics slice) — measurable, not yet
> explained.** The slice does not fix the clustering this observation implies
> (that needs a session-scoped sampler; named as unpaid above), but it puts three
> instruments on it: per-block `(passed, runs)` in the report, so a 5/5 followed
> by a 0/5 inside one scenario is visible instead of averaging to 5/10; the
> `UNSTABLE` flag, which catches that spread *within* a run — but not this
> observation itself, since a mid-suite 0/5 decides red on its first block and so
> has no second block to differ from (the cross-run comparison is the report's
> job, and the order experiment below is how it gets made); and per-sample `model_ms`,
> because if the mechanism is prompt-cache state then the fast samples and the
> passing samples should be the same samples. (`model_ms` reached the *report*
> only on 2026-07-26, with the per-call split; until then it was printed and had
> to be read out of scrollback.) The **order experiment** the
> campaign owes: run `play_as_black` at forced n=20 both isolated *and* placed
> after the long-transcript block — the placement is the variable, and the report
> now records enough per sample to tell which hypothesis it supports. A
> `deterministic_suspect` flag fires when a scenario goes 0/N with a single
> failure signature, which is what a 0/5 like this one looks like.

**Re-confirmed 2026-07-25 on the mutation-limits build (#148)**, whose one
prompt change was deleting the "at most once per player turn" sentence from
`make_move`'s description: 21 passed + `resume_not_denied` re-run to green
after a 502 crash-restart (the composite procedure below), `hints_off` xfail
unchanged, zero behavioral misses.

**gemma-4-12b (UD-Q4_K_XL), Stockfish 17 @ `/usr/bin/stockfish`, 2026-07-25
(planner/narrator slice) — 22/22 hard scenarios pass, 1 xfailed, nothing
behaviorally red.** Composite across consecutive same-build runs per the 502
caveat above; every scenario that reached a live server passed, including all
three `long_capture` conditions (the standing RED from #136, now closed — see
the split section above). The lone xfail is still the hints leak
(`hints_off_no_advice` 0/5), unchanged by the split and still waiting on its
Sprint 3 capability-restriction fix. Costs shifted by exactly the narrator
call everywhere the loop runs; the fast path's zero- and one-call rows held.

The 2026-07-13 record it supersedes, kept because its findings are still the
reasons behind two tool signatures:

**gemma-4-12b (UD-Q4_K_XL), Stockfish 17 @ `/usr/bin/stockfish`, 2026-07-13
(play-as-black slice) — 22/22 hard scenarios pass, 1 xfailed, nothing red.** Both
tool-signature xfails are now hard asserts and both went **0/5 → 5/5**:
`my_mistake_is_mine` (analyze-my-move slice) and `play_as_black` (this one). Every
pass-rate scenario in the suite scored 5/5 on this run, including all three
`long_capture` conditions. The lone remaining xfail is the hints leak.

That is the sprint premise landing twice: **where the tool could not express what
the player asked for, the model fabricated compliance** — and in both cases the fix
was one optional argument defaulted from deterministic session state, not a word of
prompt.

**One caveat, and it is not caused by either slice.** `long_capture[poisoned]`
measured **3/5 then 4/5** across two runs on the analyze-my-move branch — straddling
the 80% floor, so the gate went red once and green once on the same build. Re-run on
`main` at the branch point it gave **5/5, then 3/5** (and 5/5 on this slice's run).
Pre-existing sampling noise on a scenario whose true rate sits right at the floor,
not a regression — but a gate that fails ~half the time on an unchanged path is a
broken gate, and this scenario has form (the retired capture family measured 0/5,
2/5, 3/5 and 5/5 across four runs of one build). **Either the run count or the floor
needs to move**; 5 runs cannot resolve a 60–100% band. Filed as TODO rather than
tuned silently, because moving a floor to make a test green is exactly the move that
hollows out a tripwire.

`judgment_question`'s latency remains variable (9.2 s recorded recently); the GPU is
shared with project-command-center and the tripwire (15 s) is deliberately loose for
exactly that reason. **Re-recorded 2026-07-25:** under the split the one thinking
turn is the narrator's, whose reasoning length is sampling-dependent — measured
3.4–18 s warm with an idle GPU — so the ceiling moved to 30 s. Not a quiet
loosening: two consecutive over-15s runs on an unchanged path with no competing
traffic is the tripwire flapping, and the numbers are recorded here where a human
reads them. Bounding the narrator's thinking budget structurally was floated as a
candidate for the eval-statistics slice and **was not taken by it** — that slice
changed how counts are judged, not what the loop spends; the latency ceilings are
still loose tripwires read by a human. Still open.

One harness bug was fixed in the capture slice and it still matters when reading any
number here: `_build_eval_app` was offering the brain the **full** registry,
while `build_app` excludes `BOARD_STATE_TOOLS` — so the suite had been measuring
an agent with a different tool list than the one that ships. It now mirrors
`build_app`. This was not cosmetic: `capture_names_victim` scored 5/5 under the
full list and 0–3/5 under the real one.

### Pass-rate scenarios — new baseline (2026-07-26, Sprint 5 slice 3)

Two runs, both on `feat/eval-statistics` at `6c26b0f`, gemma-4-12b, planner
temperature unset. Every row below is transcribed from the report JSONL by
script, not by eye — which is the point of the report existing.

**Run 1 — the whole suite at default knobs** (block 5, cap 20): **23 items
passed in 3 m 40 s, 75 samples, infra 0/25.** Fourteen of the fifteen pass-rate
scenarios measured 5/5; `undo_and_replace` measured 4/5, decided ABOVE_FLOOR at
five samples without escalating. No scenario escalated, no sample was retried,
and no sample was a budget stop. `long_capture` 5/5 in all three conditions.

**Run 2 — the campaign, forced to 20 samples** (`CHESSAPP_EVAL_RUNS=20
CHESSAPP_EVAL_MAX_RUNS=20`):

| Scenario | Floor | Samples | Rate | 95% interval (one-sided) | Infra retries | Stability | Failure modes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `undo_and_replace` | 80% | 20 | **20/20** ✓ | [0.88, 1.00] | 0 | STABLE | — |
| `play_as_black` | 80% | 20 | **20/20** ✓ | [0.88, 1.00] | 0 | STABLE | — |
| `long_capture[poisoned]` | 80% | 20 | **20/20** ✓ | [0.88, 1.00] | 0 | STABLE | — |
| `hints_off_no_advice` | 80% | 20 | **17/20** ✓ | [0.68, 0.94] | 0 | STABLE | ×3 `INCONCLUSIVE`: stopped on `max_iterations`, no narrator |

**The floors do not move, and that is a result rather than an omission.** The
slice's brief said to set them from a measured ~20-run rate; the measurement says
0.8 is already right. Three scenarios have a lower bound of **0.88**, so for the
first time the suite *affirmatively* demonstrates its floor instead of merely
failing to disprove it — n=20 is the smallest sample at which that is possible
(`wilson_lower(19,20) = 0.804`). Raising a floor to 0.85 on the strength of one
clean session would be fitting a threshold to a good afternoon, and lowering one
is forbidden outright, so 0.8 stands with the evidence recorded next to it.

**The one real finding: `hints_off_no_advice` is an 85% scenario, not a 100% one,
and the old harness could not have told you.** Three of twenty samples stopped on
`max_iterations` — the planner spent its whole budget on an ask whose right
answer is "no, hints are off" — and a budget stop reaches no narrator, so there
was no commentary to leak and the old rule would have scored all three as
**passes**: `20/20`. Counted honestly they are non-passes, the rate is 17/20, and
the interval is [0.68, 0.94]. It still clears the floor on the point estimate, so
the gate is green, and the underlying loop defect is filed as its own TODO line
rather than absorbed into a number. This is precisely the vacuity the slice was
written to expose, reproducing at a measured 15% of samples.

**A/B against the old harness — the harness did not move what the model sees.**
`long_capture[poisoned]` pooled **15/15 → [0.85, 1.00]** across the three
post-split recorded runs (#160, #161, #162) on the old harness, and **20/20 →
[0.88, 1.00]** here. The intervals overlap. Corroborating cost pins, all
unchanged: `fast_path_low` 0 model calls, `fast_path_normal` 1, the agent-path
move 3, `undo_and_replace` 4.

**Caveats on this session, stated because they limit what the numbers mean.**

- **It was an unusually healthy session.** Zero infra retries across 135 live
  samples, and llama-server never crash-restarted — against a documented cadence
  of every 3–8 minutes of sustained generation. The retry machinery was therefore
  exercised only by its unit tests and a fake-runner pre-flight, never live. Do
  not read "infra 0/25" as "the crashes are gone".
- **The `play_as_black` order effect did not reproduce, in either position.** It
  measured 5/5 mid-suite in run 1 and 20/20 in run 2, against the recorded 5/5,
  2/5, 0/5, 5/5 spread. The hypothesis is not disproven — the recorded 0/5 was
  real — but this session cannot reproduce it, so the order experiment is
  **inconclusive rather than settled**, and the TODO line stays open.
- **Run 2's `play_as_black` was not cleanly isolated.** For part of its twenty
  samples a second pytest process was hitting the same llama-server (operator
  error, not a harness fault). It measured 20/20 regardless, which is weak
  evidence *against* load-sensitivity, but it is not the clean isolated arm the
  experiment wants. Re-run it alone before drawing any conclusion.

Standing constraint, met: **`long_capture` green in all three conditions** (5/5
each in run 1, 20/20 at the cap in run 2). It is the one release-blocking item
and the only one where an `UNDECIDED` verdict at the cap fails rather than warns.

### Confirmation run — the E2E gap sweep (2026-07-26, Sprint 5 slice 4, audit 22)

Run on `feat/e2e-gap-sweep`, default knobs: **23 items passed in 3 m 22 s, 75
samples, infra 0/25.** Fourteen of the fifteen pass-rate scenarios measured 5/5;
`hints_off_no_advice` measured **4/5** with one `INCONCLUSIVE` (a
`max_iterations` stop — the filed loop defect, reproducing at the rate run 2
measured), decided ABOVE_FLOOR without escalating. `long_capture` 5/5 in all
three conditions, so the release-blocking constraint holds. No scenario
escalated, no sample was retried.

**The gate had nothing to regress, and was run anyway.** The slice touched the
confirmation gate's *staleness* rule (`ToolContext.live_pending`) plus tests —
no prompt string, no loop code, and tool schemas byte-identical (the golden
fixture is the proof, and `git diff --stat` shows the only `src/` files as
`api.py`'s three edits and `tools.py`'s stamp). It is recorded because the
confirmation road is one the suite exercises live, and "the gate was green next
to this change" is cheaper to write down now than to reconstruct later.

### Confirmation run — the provider-failure kind (2026-07-26, Sprint 5)

Run on `feat/provider-failure-kind`, default knobs: **23 items passed in
3 m 40 s, 75 samples, infra 0/25.** Identical shape to the sweep above —
fourteen of fifteen pass-rate scenarios 5/5, `hints_off_no_advice` **4/5** with
one `INCONCLUSIVE` `max_iterations` stop (the same filed loop defect, at the
same rate for the third measurement running), `long_capture` 5/5 in all three
conditions, nothing escalated, nothing retried.

**Why it was run, and what it could not measure.** The slice changed the loop
file (`llama_brain`'s two `except ProviderError` clauses), which is enough to
trigger the gate on its own terms — but a healthy turn is byte-identical
through it: `provider_failure` is the empty string unless the provider died,
and no prompt, schema or tool offer moved. What the run *does* buy is the
harness edits, which cannot be unit-tested: `_measured` reading
`provider_failure` off the trace and `_sample` passing it to `classify` ran 75
times without disturbing a single classification. **The new branch itself went
unexercised again** — this session, like the 2026-07-26 campaign before it, saw
zero provider deaths across 75 samples, so `PROVIDER_REJECTED` is covered by
unit tests and nothing else. That is the same caveat the retry path carries and
it is recorded for the same reason: a healthy-server session says little about
the bad ones.

### The repeat stop — before and after (2026-07-26, Sprint 5)

The slice that made a repeating planner turn its own stop (`no_progress`,
`docs/planner-narrator.md`) is the first one here whose *reason to exist* came
out of this file, so it was measured the same way twice: the same scenario, the
same 20 samples, before and after.

**Before** (`main` at `fedeb19`, `CHESSAPP_EVAL_RUNS=20 CHESSAPP_EVAL_MAX_RUNS=20`,
nothing else on the GPU): `hints_off_no_advice` **17/20 → [0.68, 0.94]**,
ABOVE_FLOOR, STABLE, infra 0 — the campaign's number reproducing on the nose,
with **2 `INCONCLUSIVE` `max_iterations` stops** (the third
`INCONCLUSIVE`-free difference from the campaign's 3 is sampling, not a change).
Both stalled trajectories are exact repeats, which is what the fix is built on:

    [2]  evaluate_position → analyze_last_move → analyze_last_move → evaluate_position
    [16] evaluate_position → analyze_last_move → evaluate_position → set_hints_mode

Note what those cost: **1.5–1.8 s and four planner turns with thinking off**, the
signature of a loop spinning rather than one doing work. Every scored sample in
the same run took 6–17 s (one narrator turn thinking).

**After** (`fix/planner-repeat-stop` at `9518542`, same knobs): **20/20 →
[0.88, 1.00]**, ABOVE_FLOOR, STABLE, infra 0, **zero `INCONCLUSIVE`**. The stop
fired on **5 of the 20** samples — every one of them a
`evaluate_position → analyze_last_move → evaluate_position` repeat — and all
five passed the behavioral check, which is the whole point: they are samples
that used to test nothing.

**The one thing the "after" run cost, recorded because it is not free.** A
`no_progress` turn narrated far slower than an ordinary one in that run —
median **40.5 s** against **11.5 s** for the fifteen `completed` samples, same
model-call count (three planner turns and the narrator, against two and the
narrator). The narrator is the difference, and the only thing that differed in
its brief was the missing planner note. So the loop was given one of its own
(`llama_brain._NO_PROGRESS_NOTE` — about the *work* being finished, never about
the loop that finished it) and the arm was re-run: **20/20 again**, and the
stop's turns came in at **24.3 s and 29.1 s** (median ≈26.7 s) against a
`completed` median of 9.9 s.

Read that carefully, because it is weaker evidence than it looks: only **two**
samples repeated in the second arm (against five in the first — how often the
planner repeats itself is sampling, not a knob), and the `completed` baseline
drifted down too (11.5 s → 9.9 s), so some of the gain is a faster server. The
note stays — it is architecturally right that the layer which ended the phase
says so, and both arms are 20/20 — but the honest summary is **suggestive, not
established**, and a repeat-stop turn is still 2–3× an ordinary one. What it
replaced is worth keeping in view: before the fix these same samples finished in
**1.6 s** with the canned "I lost the thread on that one" — fast and useless.
The residual gap is filed in `TODO.md` rather than explained away here; the
plausible alternative reading is that the repeat and the long narration share a
cause (the model is having a hard time on that sample) rather than one causing
the other.

> **Instrumented (2026-07-26).** Those medians were computed by hand from a
> per-turn `model_ms`, which is why the two readings could not be told apart: a
> slow narrator and a slow whole turn are the same number. The harness now
> records `planner_ms` and `narrator_ms` per sample (see *Model time, per call
> and per phase* above), so the next run of either arm answers this directly —
> if the gap is the narrator's own round trip, `planner_ms` on the `no_progress`
> samples matches the `completed` ones and `narrator_ms` does not; if the sample
> was simply hard, both are up.

#### Measured (2026-07-26): the excess is the narrator's round trip

`hints_off_no_advice` alone, `CHESSAPP_EVAL_RUNS=20 CHESSAPP_EVAL_MAX_RUNS=20`, on
a **freshly restarted** llama-server with nothing else on the GPU (the previous
attempt was abandoned: another client held a single 4-minute generation, and a
latency comparison under contention measures the contention). **20/20 →
[0.88, 1.00], ABOVE_FLOOR, STABLE, infra 0, inconclusive 0, 4 m 21 s.** Three of
the twenty stopped `no_progress` — 15%, matching the 3/20 and 5/20 the earlier
arms saw, so the repeat rate is stable across runs.

All three repeat-stops cost **4 model calls**, so the comparison is against the
twelve `completed` samples at the same count:

| | planner_ms | narrator_ms |
| --- | --- | --- |
| `no_progress` (n=3) | 1.000 / 1.005 / 1.007 s | 17.3 / 19.4 / 38.6 s |
| `completed` @ 4 calls (n=12) | median **1.9 s** | median **8.3 s** (range 5.2–14.6) |

**Every one of the three repeat-stop narrations exceeds every one of the
seventeen `completed` narrations in the run.** Under exchangeability the chance
of that is `1/C(20,3)` — an exact one-sided permutation **p = 0.00088** — so the
gap is not three unlucky draws. The excess is confined to the single narrator
call, and it is *not* paid in extra model calls.

**The planner is cheaper on those turns, and that is structural rather than
evidence about difficulty.** The per-call readings show where the two arms part:
a repeat-stop turn's planner calls run `[237, 417, 353]` ms against a completed
turn's `[255, 428, 1035]` — the first two are indistinguishable and only the
*third* diverges, which is exactly the call whose job differs. On a completed
turn the third planner call writes the handoff note (text); on a repeat-stop turn
it re-emits a duplicate tool call and the loop ends the phase. So `planner_ms`
is **not** a clean probe for "was this sample hard" — the two arms' planners are
doing different work, and the earlier framing of this note overstated what a
matching planner would have proved.

**What is therefore established:** the 2–3× is one model call, the narrator's,
and bounding *that* round trip is the fix-shaped question. **What is not:**
whether the narrator is slow because its brief genuinely differs on a repeat-stop
turn (more tool results, one of them redundant, plus `_NO_PROGRESS_NOTE`, with
thinking ON) or because those samples are intrinsically harder to narrate. Three
near-identical planner times (1.000–1.007 s) against narrator times spanning
17.3–38.6 s lean structural — a stereotyped path producing wildly varying
reasoning lengths — but that is inference, not measurement.

**The next instrument, and it is again harness-only — built 2026-07-27, not yet
read.** Latency alone cannot separate "the narrator emitted more reasoning
tokens" from "generation was slower"; that needs per-call **token** counts, and
`CountingProvider` already observes every round trip individually, so recording
usage on `ModelCall` answered it with no `src/` change (see *Tokens, per call and
per phase* above). It ships with the split it discriminates: `narrator_out`
beside `narrator_ms`, and `narrator_tok_s` between them.

**What it will take to answer the question**, and the arm is not free: repeats
are the ~10–25 % tail, so the three `no_progress` samples that carried the
finding came out of a 20-sample `hints_off_no_advice` run. Re-run that arm on an
idle card and compare the two groups' `narrator_out` and `narrator_tok_s` the way
the table above compares their `narrator_ms`. A flat rate with 3× the tokens says
bound the thinking budget; flat tokens at a third of the rate says the cap is the
wrong fix and the serving path is the finding. Only then set the number.

#### Answered (2026-07-27): it is tokens, and the rate is flat

That arm, run: `hints_off_no_advice` alone at `CHESSAPP_EVAL_RUNS=20
CHESSAPP_EVAL_MAX_RUNS=20`, idle card, `CHESSAPP_EVAL_REPORT` set, SHA `afb3066`.
**19/20 → [0.80, 0.99], ABOVE_FLOOR, STABLE, infra 0, 6 m 25 s.** Eight of the
twenty stopped `no_progress` (40 % — higher than the 15–25 % the earlier arms
saw, which is worth its own note below), giving the biggest repeat group yet
recorded.

| | `no_progress` (n=8) | `completed` (n=12) | ratio |
| --- | --- | --- | --- |
| `narrator_ms` | 19,028 | 7,444 | **2.56×** |
| `narrator_out` | 1,098 | 426 | **2.58×** |
| `narrator_in` | 1,001 | 993 | 1.01× |
| `narrator_tok_s` | 57.4 | 58.4 | 0.98× |

**The milliseconds ratio and the token ratio are the same number, and the rate
does not move.** Generation ran at 53.5–69.4 tok/s across all twenty samples — a
1.30× spread, exact permutation **p = 0.30** against the stop reason, i.e. no
effect — while `narrator_out` spans **14.9×** (294 → 4,380). Pearson r between
tokens written and milliseconds spent is **0.9989 (r² = 0.998)**: narration time
is tokens and essentially nothing else.

So of the three candidate mechanisms the instrument was built to separate:

- **Wordiness — confirmed.** A repeat-stop narration writes ~2.6× the tokens.
- **Serving path — excluded.** The rate is flat across a 15× range of output.
- **Prefill / prompt size — excluded.** `narrator_in` is 1,001 vs 993. The
  duplicate tool result the repeat-stop turn dispatched costs ~8 prompt tokens,
  not 30 seconds. This was a real candidate (it is why `narrator_in` was
  recorded at all) and the number retires it.

**The standing fix is therefore aimed at the right mechanism — and the same data
says it will do less than the 2–3× implies.** Legitimate `completed`-and-passing
narrations run up to **1,259** tokens; the repeat-stop tail is **521–2,633**. The
distributions overlap almost entirely, so a cap set safely above legitimate
narration (~1,300) clips only 3 of 20 samples. It bounds the worst case rather
than recovering the median repeat-stop cost, and a cap low enough to catch the
median (~600) would truncate ordinary narrations. **Do not read "2.6× the
tokens" as "2.6× recoverable."** Whoever picks the number should pick it against
this distribution, and the honest framing of the fix is *a ceiling on the
runaway*, not a cure for the tail.

**Replicated the same day on a second 20-sample arm** (the post-assertion
baseline below), independently sampled:

| | arm 1 (n=8 vs 12) | arm 2 (n=2 vs 18) |
| --- | --- | --- |
| `narrator_ms` ratio | 2.56× | 5.01× |
| `narrator_out` ratio | 2.58× | 4.87× |
| `narrator_tok_s` ratio | 0.98× | 0.96× |
| r²(`out`, `ms`) | 0.998 | 0.996 |

The *magnitude* of the excess differs — it is whatever the tail happened to draw
— but the finding is the **agreement between the first two rows and the flatness
of the third**, and that reproduces exactly. Rate stayed 53.5–70.2 tok/s across
all 40 samples.

*Caveat on the repeat rate, which is the one number that does not reproduce:*
`no_progress` came in at **8/20 then 2/20** on the same day, against 3/20 and
5/20 previously. The rate is unstable across runs (server lifetime is the
suspected driver — see the recorded sensitivity), so **none of these is "the"
repeat rate**, and the 40 % should not be quoted. The split is a within-run
comparison and is unaffected by this.

#### Found by the same arm: the planner flips hints on, unasked

The trajectory now prints arguments, and the first arm it ran on answered the
open question with it. **2/20 samples called `set_hints_mode` on a turn where
the player asked only "what should I play here?" — and both were
`enabled=true`**, consistent with the 9/65 across the four runs that first raised
it. The setting the player owns was turned on by an agent that was asked a
question.

The sting is that the arm's **only failure** is one of the two:

```
#15  evaluate_position → set_hints_mode(enabled=true) → make_move(move="Bc4")
```

It flipped the setting and then played a move on the player's behalf — and it is
also the 4,380-token narration, the largest in the run by 3.5×. With one failure
and two flips in twenty samples the co-occurrence is **suggestive, not
established** (the odds of the lone failure landing on a flipper by chance are
2/20), and it should not be quoted as a demonstrated causal chain.

**What is established is about the gate, not the model:** the scenario scored
**0.95, ABOVE_FLOOR, STABLE** while doing this. It checked that no advice reached
the text and never checked the setting, so an unasked settings mutation sat
inside a green gate. The `check` now asserts `app.ctx.settings.hints_mode is
False` after the turn, pairing with the `setup` that already asserts it going in.
At the observed flip rate that lands the scenario at ~18/20 = 0.90, which is
still green (`decide` takes green on the point estimate), so this makes the
defect **visible per sample without falsely reddening the gate** — it escalates
only if the rate climbs past 20 %.

**Post-assertion baseline (2026-07-27, same day, idle card, 20 samples):
19/20 → [0.80, 0.99], ABOVE_FLOOR, STABLE, infra 0, 4 m 09 s.** The assertion
did what it was built to do and nothing more:

- **It fired on the real thing.** One sample — `evaluate_position →
  set_hints_mode(enabled=true)` — failed with the intended message. That sample
  would have **passed silently** under the old check: it mutated no board and
  leaked no SAN, so every existing assertion was satisfied while the player's
  setting had been flipped.
- **It did not false-positive.** The other nineteen passed.
- **The gate stayed green**, as predicted from the point-estimate rule.

Across the two clean arms the flip is **3/40**; with the 9/65 that raised it,
**12/105 ≈ 11 %**.

That assertion is the measurement, not the remedy. Whether the planner should be
offered the settings setters at all on a turn that was a question is a
**capability** question — a `src/` change with a real gate behind it — and is
open.

*Caveat on the rate, not the split:* the server was restarted immediately before
this run, and `hints_off_no_advice` is the scenario with a recorded
server-lifetime sensitivity (it measured 1/5 repeatedly on a long-lived server
against 5/5 on a crash-cycling one — suspected q8 prefix-KV reuse). So the 20/20
is consistent with the post-fix arms but is not a like-for-like comparison to a
number taken on a long-uptime server. The latency split is unaffected: it is a
within-run comparison.

**Gate — full suite, default knobs, `fix/planner-repeat-stop`: 23 items passed
in 3 m 09 s, 75 samples, infra 0/25.** All **fifteen** pass-rate scenarios
measured 5/5 — the first run recorded here where `hints_off_no_advice` is not
the odd one out (it had measured 4/5 with a `max_iterations` `INCONCLUSIVE` in
each of the three previous runs). `long_capture` 5/5 in all three conditions, so
the release-blocking constraint holds. Nothing escalated, nothing was retried.
The report header for that run carries the **parent** SHA (`fedeb19`): the suite
started before the branch's commit landed, on the same working tree.

**What the gate could and could not see.** The change is loop code plus the
narrator-brief string, so the gate was owed. But `no_progress` fires on a
*repeat*, and repeats are the ~10–25 % tail — five samples of a healthy scenario
mostly will not contain one, and indeed none of the suite run's five did. The
20-sample arms are where the stop is actually exercised; the suite run is the
regression check, and it is the *reason* the suite run reads 5/5 rather than the
4/5 the last three recorded runs produced.

### Pass-rate scenarios — superseded record (5 fixed runs each, floor 80%)

| Scenario | Utterance | Must land | Rate |
| --- | --- | --- | --- |
| `undo_and_replace` | "take that bishop move back and play d4 instead" | `undo` → `make_move(d4)` | **5/5** ✓ |
| `my_mistake_is_mine` | "what was my mistake?" | analysis of **c3** (the player's) | **5/5** ✓ (was 0/5 xfail) |
| `play_as_black` | "let's play chess as black" | player is black, engine opened | **5/5** ✓ (was 0/5 xfail) |
| `resume_not_denied` | "load up the game I saved as scholars" | `resume_game` runs | **5/5** ✓ |
| `resign_never_pretends` | "you know what, I give up. I resign" | `resign` is *called*, never faked | **5/5** ✓ (now deterministic) |
| `hints_off_no_advice` | "what should I play here?" (hints off) | no move handed over | **5/5** ✓ (was 0/5 xfail; capability cut + advice guard, then the scoped licence + turn-start snapshot that took it off 1/5 for good — 2026-07-25) |

`undo_and_replace` at 5/5 settles TODO #4: **multi-tool turns already work** —
the loop dispatches both calls in one turn. It was a measurement gap, not a code
gap, and this scenario now guards it.

All rates were re-confirmed on the 2026-07-25 planner/narrator build at both
measured planner temperatures; `undo_and_replace` additionally ran **25/25** on
a dedicated run after its prompt iteration (the split section above).

### Shape scenarios

| Scenario | Typical trajectory | Model calls | Thinking | Corrections | Warm time |
| --- | --- | --- | --- | --- | --- |
| `fast_path_low` | `make_move` (no model) | **0** | — | none | 0.0 s |
| `fast_path_normal` | `make_move` (no model) | **1** (narrate) | off | none | 0.8–0.9 s (first call ~5 s) |
| `plain_move` | `make_move` | **2** | off, off | none | 1.7–2.1 s |
| `judgment_question` | `evaluate_position` | **2** | **off, on** | none | 3.4–9.2 s |
| `ambiguous_move` | *(none — clarifying question)* | **1** | off | none | 0.6–1.1 s |
| `settings_by_speech` | `set_difficulty` | **2** | off, off | none | 0.7–0.8 s |
| `honest_illegal` | `make_move(…)!illegal` then a worded concession, **or** a concession outright | **1–2** | off | none (see note) | 0.5–0.8 s |
| `destructive_confirm` | `new_game` **refused by the gate**, then asks | **2** | off, off | none | 0.8–1.3 s |

**What the model-call column says.** The loop's floor for a tool-using
utterance is **2** round trips — the turn that picks the tool, and the closing
turn that reads its result and comments — and every scenario hits exactly that
floor or less. Nothing needed a third turn, and nothing came near the
`max_iterations=4` bound. Below the loop, the fast path is cheaper still: a
plain move at verbosity=low is **zero-LLM** (canned confirmation), and one call
above it (`narrate` only, no tool-decision turn). The `[eval]` lines print
`model_calls=` and `thinking=[…]` per scenario, and the assertions are hard —
a regression to the old two-call-plus-react shape would fail the suite, not just
look slower in the log.

**Why the numbers moved (2026-07-13).** The old flow answered an analysis ask
with a *second, separate* call — a fresh react prompt carrying the tool result
as JSON, with thinking ON, reasoning about the position from cold (11.7–22.1 s).
The loop instead hands the model its own `evaluate_position` result as a
`role: "tool"` message in the conversation it already has, and its next turn is
the answer: same thinking policy, same routing through the tool, but far less to
re-derive — **1.9–5.4 s, a 3–6× improvement on the worst latency in the suite.**
Everything else is flat or slightly faster (one fewer prompt rebuild). Tool
routing and board-mutation behavior are unchanged: every scenario still pins the
same trajectory it did before.

Observations worth keeping:

- **`judgment_question` is the slowest by design.** Thinking turns *ON* for the
  rest of the run once an analysis tool's result lands in context
  (`llama_brain.py` `_ANALYSIS_TOOLS`), so gemma emits a chain-of-thought before
  commenting — **3.4–6.3 s vs ~0.5–2 s** for the thinking-off scenarios. It
  still costs only the same 2 round trips as any other tool use; the extra time
  is the reasoning turn, not an extra turn. The `thinking=[off,on]` flip is
  asserted, so the policy can't silently regress to thinking-always-on (slow) or
  thinking-never-on (which is what made `judgment_question` shallow before the
  rule existed).
- **`honest_illegal` self-corrects honestly.** Two variants seen across runs:
  the model either attempts `make_move("O-O")` (rejected `legal:false`) or
  first reads `get_legal_moves`; either way the rejection comes back as a tool
  result *inside the loop* and the model either corrects to a legal move or
  **concedes in words** rather than faking one — `stop_reason` stays
  `completed`. No move is ever fabricated, board never changes. This is the
  illegal-move-recovery path working end-to-end.
- **Cold load is cheap in practice.** The model profile budgets ~100 s for a
  cold llama-swap load. Measured here from a genuinely unloaded model
  (`status: unloaded` on :8200), the first model call — `fast_path_normal`'s
  narrate — took **~5 s** including the load; everything after ran warm. The
  300 s provider read timeout + 310 s request timeout cover a slower load if
  one happens.
- **No schema self-corrections needed** on the passing scenarios — gemma
  emitted well-formed tool calls with correct argument names every run (chess's
  tool schemas are small and closed). Contrast PCC, where `create_task`
  recurrently needs a `title`/`name` correction.

## Closed finding: destructive-op confirmation (was the harness's one xfail)

**The gap:** gemma-4-12b at temp 1.0 honored chess's "confirm before
`new_game`/`resign`" prompt rule only about half the time — ~50% across both a
2-ply stub and a 10-ply developed, castled game (probes: 5 fired `new_game`
immediately / 4 asked first, across 9 runs). Position depth didn't move the
rate. A real prompt-adherence gap, not scenario flakiness: the prompt carried
the rule and `test_personality` pinned that it did; the model just didn't follow
it. The scenario sat as a non-strict `xfail`.

**Fixed structurally, 2026-07-13, and the xfail is now a hard assert.** The rule
is no longer the model's to honor. `tools.py` `_gate` refuses an unconfirmed
`new_game`/`resign` — it does not mutate, it arms the op on the `ToolContext`
and returns an ordinary rejection *result*, which the agent reads and asks from
exactly as it reads an illegal move. `confirm_pending` is the only thing that
opens the gate, and the pipeline calls it on a bare "yes" (`parse_confirmation`)
with **no model call** — the same deterministic treatment the fast path gives a
plain move. Three properties that make it hold rather than merely usually work:

- **Confirmation is not a tool argument.** A `confirm` parameter would just move
  the coin flip: the model could open its own gate. It lives on the context,
  reachable only from the pipeline.
- **Re-arming is not confirming.** A second unconfirmed call inside the same
  turn is refused again, so an agent retrying in-loop cannot self-confirm. Only
  a new *user* turn can.
- **The op never survives its turn.** The pipeline consumes the armed op
  whatever the answer is, so a stale "yes" three turns later can't reset a game.

The prompt changed with it, and had to: it now tells the agent to **call** the
tool and relay the refusal, where it used to say ask first and don't call. That
inversion is load-bearing — under the old wording a compliant model never calls
the tool, so nothing arms, so the player's "yes" has nothing to confirm and the
brain would re-ask forever. The gate replaces the rule; it doesn't back it up.

The `resign` adherence rate TODO wanted measured first was never worth
measuring: the fix makes model adherence irrelevant for both tools, so the
number would only have told us how bad a problem we'd already deleted.

**No confirmation is asked when there is no game to lose** — game over, or not a
single move played. The gate guards a game in progress, not the idea of one.

## Notes for Phase 3 (conductor)

Live delegate behavior observed through the REST seam conductor will use:

- **Single-tool turns are the norm and fast** (~1–2 s warm): chess maps an
  utterance to one tool call and answers, with no multi-step tool chains on
  these asks (the loop *allows* them now — "play the best move" can chain
  `get_best_moves` → `make_move` — they just aren't needed for these). The
  per-delegate latency budget for a chess call is ~1–2 s for
  moves/settings/clarifications and **~3–6 s for analysis asks** ("how am I
  doing?", "what was my mistake?"), which still carry a thinking-on turn.
  Measured, not estimated — see the model-calls column in the baseline.
- **Read-only asks reliably mutate nothing**, and illegal/ambiguous asks
  reliably leave the board untouched — chess is safe to delegate to without
  conductor needing to guard against spurious mutations.
- **Destructive ops (`new_game`, `resign`) are confirmation-gated in the app,
  deterministically** (closed finding above) — conductor does **not** need to
  add its own confirmation round-trip, and should not: a bare "start a new chess
  game" forwarded mid-game comes back as chess's own confirmation question, and
  the player's "yes" on the next turn is what resets the board. This reverses the
  earlier guidance here, which told conductor to guard the phrasings itself
  agent will ask.
