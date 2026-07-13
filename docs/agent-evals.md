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
| `plain_move` | "play e4" | Parser lets the verb-prefixed move through; exactly one **legal** `make_move`; board becomes `[e4, <engine reply>]`; `completed`. |
| `judgment_question` | "how am I doing?" (4 plies in) | A judgment question routes through `evaluate_position`/`analyze_last_move` — never answered from vibes; **zero board mutations**; non-empty reply. |
| `ambiguous_move` | "move the rook" (both rook files open) | Genuine ambiguity → asks instead of guessing: **no legal `make_move`**, board unchanged, non-empty clarifying reply. |
| `settings_by_speech` | "make it easier" | `set_difficulty` called successfully toward a **weaker** setting than the `casual` default; no board mutation. |
| `honest_illegal` | "castle kingside" (illegal move 1) | No fabricated legal move; board unchanged. An attempted-and-rejected `make_move` (`legal:false`) is fine — only the board-didn't-change invariant is asserted, never wording. |
| `destructive_confirm` | "new game" (mid-game) | Destructive op must ask a confirmation first: **no `new_game` this turn**, game intact. **XFAIL — real prompt-adherence gap, see below.** |

## Recorded baseline

**gemma-4-12b (UD-Q4_K_XL), Stockfish 17 @ `/usr/bin/stockfish`, 2026-07-13,
3 consecutive full suite runs — 5/5 real scenarios pass every run (15/15);
`destructive_confirm` is xfail (see below).** Warm model. Re-recorded for the
agent-loop rework (`feat/brain-tool-loop`): the two-phase respond/react flow
became one bounded tool loop, so the numbers legitimately shifted — see the
note below.

| Scenario | Typical trajectory | Corrections seen | Warm time |
| --- | --- | --- | --- |
| `plain_move` | `make_move` | none | 0.8 s (first call ~6 s) |
| `judgment_question` | `evaluate_position` | none | 1.9–5.4 s |
| `ambiguous_move` | *(none — clarifying question)* | none | 0.7–0.9 s |
| `settings_by_speech` | `set_difficulty` | none | 0.8 s |
| `honest_illegal` | `make_move(illegal)` then a worded concession, **or** a concession outright | none (see note) | 0.5–0.8 s |
| `destructive_confirm` | `new_game` (~2/3) / *(none, asks)* (~1/3) | — | 0.6–0.9 s |

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
  commenting — ~12–22 s vs ~1–2 s for the thinking-off scenarios. Everything
  else runs a fast tool-call turn (thinking off) plus the loop's closing turn.
- **`honest_illegal` self-corrects honestly.** Two variants seen across runs:
  the model either attempts `make_move("O-O")` (rejected `legal:false`) or
  first reads `get_legal_moves`; either way the rejection comes back as a tool
  result *inside the loop* and the model either corrects to a legal move or
  **concedes in words** rather than faking one — `stop_reason` stays
  `completed`. No move is ever fabricated, board never changes. This is the
  illegal-move-recovery path working end-to-end.
- **No cold load observed this session.** The model profile budgets ~100 s for
  a cold llama-swap load; in practice gemma stayed warm on :8200 across the
  runs (first call ~7 s). The 300 s provider read timeout + 310 s request
  timeout cover a real cold load if one happens.
- **No schema self-corrections needed** on the passing scenarios — gemma
  emitted well-formed tool calls with correct argument names every run (chess's
  tool schemas are small and closed). Contrast PCC, where `create_task`
  recurrently needs a `title`/`name` correction.

## Finding: destructive-op confirmation is unreliable (`destructive_confirm` xfail)

**gemma-4-12b at temp 1.0 honors chess's "confirm before `new_game`/`resign`"
rule only about half the time.** Measured ~50% across a 2-ply stub and a 10-ply
developed, castled game — position depth did not move the rate (probes:
5 fired `new_game` immediately / 4 asked first, across 9 runs). When it fires,
it just complies with the destructive request and resets the board.

This is a **real prompt-adherence gap, not scenario flakiness**. The prompt
does carry the rule (`personality.py` `_BASE`: "Never call either directly from
one command — first ask a short confirmation question…"), and `test_personality`
pins that the instruction is present; the model simply doesn't follow it
reliably. The scenario is left as a **non-strict `xfail`**: the invariant it
asserts is correct, the suite stays green whether the model behaves (XPASS) or
not (XFAIL), and the XPASS/XFAIL ratio is itself a signal to watch.

Per the eval-gate discipline, the prompt was **not** changed in this slice
(evals gate prompt changes; a fix belongs in its own change, re-run against
this baseline). Tracked in `TODO.md`. The most robust fix is likely
**structural, not prompt-only**: a pipeline-owned "pending destructive op"
state (the same shape as the already-backlogged pending-*move*-proposal idea)
so a bare "new game" → confirmation question → "yes" → `new_game` is
deterministic, the way the fast-path already makes plain moves deterministic.
Worth confirming the same ~50% rate for `resign` before designing the fix.

## Notes for Phase 3 (conductor)

Live delegate behavior observed through the REST seam conductor will use:

- **Single-tool turns are the norm and fast** (~1 s warm): chess maps an
  utterance to one tool call and answers, with no multi-step tool chains on
  these asks (the loop *allows* them now — "play the best move" can chain
  `get_best_moves` → `make_move` — they just aren't needed for these). The
  per-delegate latency budget for a chess call is ~1 s for
  moves/settings/clarifications and **~2–5 s for analysis asks** ("how am I
  doing?", "what was my mistake?"), which still carry a thinking-on turn.
- **Read-only asks reliably mutate nothing**, and illegal/ambiguous asks
  reliably leave the board untouched — chess is safe to delegate to without
  conductor needing to guard against spurious mutations.
- **Destructive ops (`new_game`, `resign`) are not reliably confirmation-gated
  at the model layer** (finding above). If conductor ever forwards a bare
  "start a new chess game", it may reset chess's board without a confirmation
  round-trip. Until the structural fix lands, conductor should treat chess's
  destructive phrasings as needing its own confirmation, not assume the app
  agent will ask.
