# Agent-control audit — external review, 2026-07-25

An external review (ChatGPT, supplied by Ryan 2026-07-25) of how much genuine
control Glitch has over agent-enabled turns. The verbatim report is in the
second half of this document; this first half is the code-checked review of it,
because several of its findings describe things that already exist, and a few
of its prescriptions conflict with the repo's non-negotiables (CLAUDE.md "Core
Architecture Principle" / "Binding Invariants"). `TODO.md` was rebuilt from
this document on 2026-07-25; item numbers below are the report's.

## Verified accurate — the load-bearing findings

- **The board UI bypasses the agent entirely (P0, items 1/4).** `/api/game/move`
  (`api.py`) drives `session.submit_move` + the engine reply directly; a drag
  never produces a Glitch turn. This is *deliberate* for the LLM-off mode the
  app must always support, but the report is right that when the agent is on,
  it is a silent bypass: the agent's "personality in the path" story only
  covers typed/spoken commands.
- **`make_move` is player-move + engine-reply in one atomic tool (item 2).**
  `tools.py make_move` applies the player's move and immediately has the
  engine reply inside the same call. Glitch has no observation point between
  the two — its commentary always reacts to the completed exchange.
- **The final commentary pass is not structurally tool-free (item 9).**
  `llama_brain.py` offers tools on every loop pass ("Tools are always offered:
  the loop ends when the model declines to use them"). The closing turn is
  tool-free only because the model chose not to call any — CLAUDE.md's "it is
  offered no tools" wording overstates what the code enforces. Within
  `max_iterations` the model could act twice.
- **Mutation limits are prose, not code (item 6).** "Call this at most once per
  player turn" lives in the `make_move` docstring — exactly the "rule the
  prompt hopes the model follows" shape this repo has repeatedly measured at
  ~50% adherence and repeatedly moved into code.
- **No board-version checks (item 7).** The web UI, the delegate API, and the
  MCP server all mutate one shared session with no version/expected-FEN
  precondition; concurrent clients can interleave.
- **Recovery semantics are undefined in one spot (item 20).** On the fast path
  the move is applied *before* `brain.narrate`; a provider failure there
  escapes as an unhandled error after the board already changed (and before
  the broadcast).

## Already done — the report didn't know

- **Tracing (item 18):** `CHESSAPP_TRACE_PATH` already records route, utterance,
  FEN before/after, full tool trajectory, `stop_reason`, the honesty-guard
  decision (`guarded`), and per-turn model calls + token counts (#136). Gaps
  worth keeping: turn/correlation IDs, per-model-call latency, an explicit
  mutation count.
- **Deterministic tests vs live evals (item 21):** already exactly the split —
  CI runs the offline suite; the eval harness is opt-in
  (`CHESSAPP_AGENT_EVALS=1`) and gates prompt/model/loop changes. The only new
  bit is a scheduled run.
- **Confirmation locking (item 8):** mostly structural already — the armed op
  never survives its turn, re-arming inside a turn is not confirming, only a
  fresh user "yes" opens the gate. The one divergence is policy, not a gap:
  a new intent *disarms* the pending op rather than being rejected, which is
  arguably better UX than a lock. Not re-planned.
- **Settings injection (item 12), partially:** the system prompt is re-resolved
  per command from live settings (verbosity/hints), and board truth +
  `player_color` + `saved_games` are injected fresh every turn. Difficulty and
  voice-output are not injected — folded into the settings slice.
- **Items 11, 14, 16, 17, 19, 23** were already on the backlog nearly verbatim
  (hints-off as capability restriction; structured tool errors; the
  `long_capture` RED regression as release-blocking; turn summaries; live
  tool-call UI; eval statistics). The replan keeps them, reorganized under the
  report's priorities.

## Reshaped to fit the house rules

- **The coordinator, not Glitch, owns the turn sequence (items 2/3/5).** The
  report has Glitch call `request_engine_reply` / `apply_engine_move` as tools.
  That would make the model responsible for whether the engine replies — a
  thing deterministic code already knows — and a stalled/failed model call
  would stall the game, breaking the LLM-off invariant. The replanned shape:
  a deterministic turn coordinator owns
  `player_move → (observe) → engine_reply → (close)`; Glitch fills the two
  observation slots and can *never* block, skip, or reorder the engine reply
  (timeout/absence of the model degrades to direct mode for that beat). Same
  control point the report wants, without handing the sequence to a 12B.
- **Direct mode already exists (item 1).** Full games with the LLM off are a
  binding invariant. The new work is making the mode *visible and deliberate*
  in the UI, and routing agent-mode board input through the coordinator.
- **Planner/narrator split (item 15).** A full two-phase re-split would
  partially reverse the deliberate one-loop rework (#124) that fixed real
  multi-step failures. The compatible version is (a) the already-planned
  per-phase sampling experiment (lower temp on tool-selection calls only) and
  (b) the P0 observation/close beats, which give narration its own model call
  naturally. Re-evaluate after P0 lands.

## One standing warning the replan must not lose

The tool-schema minimization (old audit finding 5) was attempted 2026-07-21 and
**shelved with prejudice**: stripping pydantic's `title`/`anyOf`/`default` keys
collapsed `undo_and_replace` 88% → ~6%, and the isolation run showed *every*
component of the cut regresses below the floor. gemma-4-12b's tool-use training
evidently expects the pydantic/JSON-Schema shape. **Do not re-attempt on this
model**; re-run `test_eval_undo_and_replace_is_one_turn` as the gate if the
brain model ever changes. Full record in `DONE.md` (2026-07-21) and branch
`chore/clean-tool-schema-noise` history.

## Latency constraint on the P0 work

An observation reaction per board drag adds a model round trip per move on a
local 12B. The report's own mitigations are design requirements from day one,
not options: the observation beat is short or streamed while Stockfish
calculates, it is skippable, and verbosity=low keeps the zero-LLM behavior the
fast path has today. If the observation beat makes a plain move feel slower,
it has failed its acceptance test.

---

# The report, verbatim

Recommended improvements Priority 0 — Make Glitch genuinely control agent-enabled turns

1. Define two explicit operating modes
   * Agent mode: every move, board action, and command passes through a turn coordinator that includes Glitch.
   * Direct mode: the existing fast deterministic player-versus-Stockfish route remains available when AI is disabled or unavailable. Direct mode should be deliberate and visible—not a silent bypass caused by how the user moved a piece.
2. Split the combined `make_move` operation Replace the current combined operation with distinct steps:
   1. `submit_player_move`
   2. Validate and apply the player move
   3. Let Glitch observe the verified result
   4. `request_engine_reply`
   5. Have Stockfish propose a response
   6. `apply_engine_move`
   7. Let Glitch close the turn This is the most important architectural change. It gives Glitch a real point of control without weakening chess legality.
3. Introduce a turn state machine Represent every turn with explicit states such as: `awaiting_player → player_move_applied → agent_observing → engine_calculating → engine_move_applied → completed` Reject actions that do not belong in the current state. Store the turn ID, expected player, board version, and FEN at each transition so timeouts and retries cannot accidentally duplicate moves.
4. Route board controls through the same coordinator A board drag should send a structured move such as `e2e4`, not a natural-language command, but it should still enter the same agent-enabled sequence. Undo, resign, reset, and similar controls should also use the same authorization and confirmation system as spoken or typed commands.
5. Give Glitch an observation point between moves After the player's move is validated, provide Glitch with structured facts:
   * Move played
   * Piece moved or captured
   * Check, checkmate, or draw status
   * New FEN
   * Relevant legal consequences Glitch can then react before requesting Stockfish's reply. To control latency, this reaction can be short or streamed while Stockfish begins calculating.

Priority 1 — Enforce mutation safety in code

6. Limit mutations per turn Code should enforce a maximum of:
   * One accepted player move
   * One accepted engine move
   * One destructive operation After a successful mutation, remove or disable the corresponding tool until the next turn. Do not rely on prompt instructions to enforce this.
7. Add board-version checks to every mutation Every state-changing request should include the board version or expected FEN. Reject it if another request has already changed the board. This prevents duplicate submissions and multi-device race conditions.
8. Lock pending confirmations Once reset, resignation, or another destructive operation is awaiting confirmation:
   * Do not allow another operation to overwrite it.
   * Accept only confirmation, cancellation, or harmless questions.
   * Expire the pending action after an appropriate boundary.
9. Make the final commentary pass tool-free Once all requested work is complete, call the model without tools for its final response. The current loop continues offering tools on subsequent model passes, which makes repeated actions possible even when the documentation says the agent should only comment.
10. Centralize tool validation Use one validation layer for tool schemas, permissions, current turn state, confirmation status, and board version. This reduces drift between API validation, agent validation, and game-session validation.

Priority 2 — Turn settings and honesty into enforceable policy

11. Make hints-off a capability restriction When hints are disabled:

* Remove advice tools such as best-move lookup from the available tool set.
* Do not provide engine evaluations or candidate moves in context.
* Add a final response check for accidental move recommendations. This should resolve the currently failing hints-off evaluation more reliably than prompting alone.

12. Inject current settings into every fresh decision Glitch should receive authoritative values for difficulty, hints, verbosity, voice output, player color, and other active settings. It should never infer these values from old conversation text.
13. Expand the honesty guard The existing guard protects reset and game-ending claims. Extend the same evidence-based approach to claims about:

* Moves and captures
* Check, checkmate, and draws
* Board position and legal moves
* Saved or resumed games
* Settings changes
* Engine analysis Ideally, construct a structured set of verified reply facts and require operational statements to derive from it. Personality can vary the wording, but not the facts.

14. Return richer structured errors An illegal or ambiguous move result should include legal alternatives, ambiguity candidates, the current board version, and whether retrying is safe. This gives Glitch enough fresh information to correct the request without guessing.

Priority 3 — Improve reasoning quality without losing personality

15. Separate decision-making from narration Use two model phases:

* A compact, lower-temperature planning phase for selecting tools.
* A personality-rich narration phase for Glitch's response. This reduces tool-selection variability while preserving the character's voice.

16. Fix the long-capture regression before further prompt trimming Preserve the failing "grab the pawn on e6" scenario as a release-blocking regression test. Further reductions to prompts or schemas should be rejected if they reduce grounded move resolution.
17. Replace raw long history with structured memory Keep the latest few turns verbatim, then summarize older turns into:

* Game events
* User preferences
* Active settings
* Unresolved requests
* Lightweight conversational context Board truth must always come from the current game state, never from conversation history.

Priority 4 — Strengthen observability and user experience

18. Create one trace record per turn Record:

* Input route
* Turn and correlation IDs
* FEN before and after
* Model calls and latency
* Tool requests and results
* Mutation count
* Confirmation and honesty-guard decisions
* Final completion state This will make duplicated moves, bypasses, slow turns, and factual failures much easier to diagnose.

19. Show live progress in the interface Surface concise states such as "validating your move," "Glitch is reacting," and "Stockfish is calculating." This becomes especially important once a turn contains multiple intentional phases.
20. Provide explicit recovery behavior Define what happens if the provider times out after the player move but before the engine reply. The position should remain valid and resumable, with no possibility of silently replaying the player move.

Priority 5 — Make evaluation release-worthy

21. Separate deterministic tests from live-model evaluations Run legality, routing, confirmation, state-machine, and mutation-limit tests in CI. Run stochastic model evaluations on prompt/model changes and on a scheduled basis.
22. Add end-to-end tests for the architectural risks At minimum:

* A board drag passes through Glitch in agent mode.
* Glitch observes the player move before the engine reply.
* A command cannot produce duplicate moves.
* Hints-off never exposes advice.
* Pending destructive operations cannot be overwritten.
* Provider failure leaves a resumable position.
* Stale board versions are rejected.
* Concurrent clients cannot advance the same turn twice.

23. Use stronger statistical evaluation Five-run pass-rate thresholds are too noisy. Use more repetitions or confidence intervals, and report deterministic failures separately from model variance.
24. Complete a real-device walkthrough Test desktop and mobile board controls, speech-to-text, text-to-speech, long games, save/resume, network interruptions, and operation without the model provider.

Recommended implementation order

1. Split player and engine moves and add the turn state machine.
2. Route board controls through the coordinator.
3. Enforce per-turn mutation limits and tool-free commentary.
4. Harden confirmation, settings, hints, and evidence-backed replies.
5. Separate planning from personality and improve context handling.
6. Add unified tracing, CI coverage, live-model evaluations, and real-device QA.

The target architecture should ultimately be: Player intent → validated player move → Glitch observes → Stockfish proposes → trusted code validates → Glitch closes the turn.
