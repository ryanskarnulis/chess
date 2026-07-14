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
today; bare "e4" does not).

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

## The gating rule

**Run this harness before merging any prompt, model, or loop change, and the
baseline must not regress.** A prompt tweak that degrades tool honesty (a
judgment question answered from vibes, an invented move, a lost clarifying
question) does not ship. If the baseline legitimately shifts (a better prompt,
a new model), re-record the table below in the same PR and say why.

## Scenarios — what each pins

| Scenario | Utterance | Pins |
| --- | --- | --- |
| `fast_path_low` | "e4" (verbosity=low) | The one scenario that *does* parse: the fast path dispatches the move and answers with a canned confirmation — **zero model calls**. |
| `fast_path_normal` | "e4" (verbosity=normal) | Above verbosity=low the fast path still skips the loop and pays for commentary only: **exactly one** model call (`Brain.narrate`), thinking off. |
| `plain_move` | "play e4" | Parser lets the verb-prefixed move through; exactly one **legal** `make_move`; board becomes `[e4, <engine reply>]`; `completed`. **2 model calls** (tool turn + closing turn), thinking off throughout. |
| `judgment_question` | "how am I doing?" (4 plies in) | A judgment question routes through `evaluate_position`/`analyze_last_move` — never answered from vibes; **zero board mutations**; non-empty reply. Also the live thinking-policy pin: the tool-picking turn is thinking **off**, the turn that reasons about the result is thinking **on**. |
| `ambiguous_move` | "move the rook" (both rook files open) | Genuine ambiguity → asks instead of guessing: **no legal `make_move`**, board unchanged, non-empty clarifying reply. |
| `settings_by_speech` | "make it easier" | `set_difficulty` called successfully toward a **weaker** setting than the `casual` default; no board mutation. |
| `honest_illegal` | "castle kingside" (illegal move 1) | No fabricated legal move; board unchanged. An attempted-and-rejected `make_move` (`legal:false`) is fine — only the board-didn't-change invariant is asserted, never wording. |
| `destructive_confirm` | "new game" (mid-game), then "yes" | Destructive op must not fire on the first ask: **no successful `new_game` this turn**, game intact, and the agent relays the gate's refusal as a question. The follow-up "yes" must then actually reset the board. **Hard assert** — was the suite's one xfail; see the closed finding below. |

## Recorded baseline

**gemma-4-12b (UD-Q4_K_XL), Stockfish 17 @ `/usr/bin/stockfish`, 2026-07-13
— 8/8 scenarios pass, no xfails.** Warm model. `destructive_confirm` is a hard
assert as of the confirmation-gate slice (`fix/destructive-confirm-gate`); every
other scenario's trajectory, model-call count and latency is unchanged against
the previous baseline, which is the result the gate change wanted: it adds a
refusal round-trip to destructive ops and touches nothing else.

| Scenario | Typical trajectory | Model calls | Thinking | Corrections | Warm time |
| --- | --- | --- | --- | --- | --- |
| `fast_path_low` | `make_move` (no model) | **0** | — | none | 0.0 s |
| `fast_path_normal` | `make_move` (no model) | **1** (narrate) | off | none | 0.8–0.9 s (first call ~5 s) |
| `plain_move` | `make_move` | **2** | off, off | none | 1.7–2.1 s |
| `judgment_question` | `evaluate_position` | **2** | **off, on** | none | 3.4–6.3 s |
| `ambiguous_move` | *(none — clarifying question)* | **1** | off | none | 0.6–1.1 s |
| `settings_by_speech` | `set_difficulty` | **2** | off, off | none | 0.7–0.8 s |
| `honest_illegal` | `make_move(illegal)` then a worded concession, **or** a concession outright | **1–2** | off | none (see note) | 0.5–0.8 s |
| `destructive_confirm` | `new_game` **refused by the gate**, then asks | **2** | off, off | none | 0.8 s |

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
