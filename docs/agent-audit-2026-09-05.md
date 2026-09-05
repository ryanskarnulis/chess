# Audit of the chess agent loop, tools, and eval coverage

Audited local `main` at `3038c4e04540cd1121b44127554df2b1d573dff6` (#260), September 5, 2026. No repository files changed. The targeted deterministic suite passed **943 tests**. No live-model evals were run; live pass rates below are explicitly the repository's recorded measurements, not new measurements.

The suspected coverage gap is real, with three corrections: `undo_twice_and_replace` already exists as a non-strict xfail; the long-transcript family and the live constraint scenario use the panel HTTP seam; and `destructive_confirm` starts from a ten-ply game, not an untouched board. The most consequential new evidence is that a refused undo can discard an owed reply, a mid-turn save can resume without that reply, and the guard can suppress a correct clarification or intermediate material count.

Evidence labels: **Read** means directly established from executable code; **reproduced** means additionally exercised with the real registry/coordinator and a scripted provider or a small deterministic probe; **inferred** means a plausible consequence whose frequency or live-model occurrence was not measured. Source links identify the relevant symbol or line. The two reproduction scripts are core probes (appendix A) and additional probes (appendix B). They describe observed current behavior, not desired regression assertions.

The first tool read showed the pre-#260 loop; subsequent reads, Git HEAD, the file checksum, and the tests agreed on the results-keyed implementation. All conclusions here use that verified implementation.

Prioritized findings

| Priority / severity | File and symbol | Defect | Concrete failure: input/state → outcome | Evidence |
|---|---|---|---|---|
| 1 · blocks-a-game | [tools.py:1261](../backend/src/chessapp/tools.py:1261), `build_registry.undo`; [api.py:2159](../backend/src/chessapp/api.py:2159), `_command_turn` | `undo` abandons the open turn before finding out whether the undo can succeed. | Fresh game, batch `make_move(e4), undo(plies=100)` → undo returns `ok:false`, but the command finishes with history `[e4]`, Black to move, coordinator awaiting the player, and no engine reply. | Read + HTTP/scripted reproduction. |
| 2 · blocks-a-game | [tools.py:1457](../backend/src/chessapp/tools.py:1457), `save_game`/`resume_game`; `TurnCoordinator.abandon_turn` | A save can capture the interval between the player move and engine reply, but resume restores no obligation to finish that exchange. | “play e4 and save this as half” with move then save → live board settles at `[e4,e5]`; later “load half” → `[e4]`, player White, Black to move, no automatic reply. | Read + HTTP/scripted reproduction. |
| 3 · blocks-a-game | [mcp_server.py:109](../backend/src/chessapp/mcp_server.py:109), `build_mcp_server`; [tools.py:477](../backend/src/chessapp/tools.py:477), `confirm_pending` | Standalone MCP exposes the confirmation gate but no surface capable of confirming it. | MCP `make_move(e4)`, then `new_game()` → confirmation required; human says yes to the MCP client; another `new_game()` is refused again. The advertised tool list has no confirmation operation, and this process has no HTTP confirmation route. | Read + in-memory MCP protocol reproduction. Applies to gated resign/draw too. |
| 4 · blocks-a-game | [llama_brain.py:402](../backend/src/chessapp/llama_brain.py:402), `LlamaBrain.read_answer` | The confirmation reader accepts a generation cut off by `finish_reason="length"`, contrary to its fail-closed contract. | Pending resignation, free-form reply, provider returns content `confirm` with `finish_reason="length"` → reader returns `CONFIRM`; the pipeline will execute the pending resignation. | Read + reader reproduction; the malformed completion was scripted, not observed from the live model. |
| 5 · blocks-a-game | [tools.py:602](../backend/src/chessapp/tools.py:602), `ToolRegistry.dispatch`; [game.py:410](../backend/src/chessapp/game.py:410), `GameSession.undo` | Schema-valid integral JSON floats are passed unchanged to Python operations requiring an integer, and the resulting `TypeError` escapes. | Batch `make_move(e4), undo(plies=1.0), set_voice_output(true)` → schema accepts `1.0`; slicing raises `TypeError`; e4 remains, voice call is skipped, and the HTTP command raises before normal close/trace. | Read + HTTP/scripted reproduction. No claim about how often Gemma emits this representation. |
| 6 · wrong-answer | [api.py:2249](../backend/src/chessapp/api.py:2249), advice guard; [honesty.py:114](../backend/src/chessapp/honesty.py:114), `names_a_legal_move` | Every unlicensed playable SAN mention on an unchanged board is treated as advice, including a correct clarification. | Fresh board, “move my kings knight”; narrator correctly asks “Do you mean Nf3 or Nh3?” → entire question replaced with the unverified-advice correction. | Read + HTTP/scripted reproduction. |
| 7 · wrong-answer | [api.py:718](../backend/src/chessapp/api.py:718), `_verified_facts`; `_command_turn` at 2191 | Brain-route evidence omits the actual intermediate board that its tools and narrator can describe. | Start `e4 d5`; batch `make_move(exd5), describe_position()` reports player advantage +1; narrator says “You are up a pawn”; engine replies Qxd5 → guard sees only before/final balances 0 and suppresses the correct observation. | Read + HTTP/scripted reproduction. Fast-path `fen_observed` already solves this for that route. |
| 8 · wrong-answer | [llama_brain.py:608](../backend/src/chessapp/llama_brain.py:608), `_exchange_key`; `get_agent_response` at 345 | Result-keyed progress still mistakes a stateful action for a stall when its success payload omits the changed state. | From `e4 e5 Nf3 Nc6`: first response `save_game(checkpoint), undo()`, second response `save_game(checkpoint)`, intended next response `make_move(d4)` → second save rewrites different game bytes but returns the same `{ok:true,name}`; loop narrates and never asks for the move. | Read + HTTP/scripted reproduction. This sequence fits the four-planner-call budget if allowed to continue. |
| 9 · hygiene, affecting reliability measurement | [test_agent_evals.py:1738](../backend/tests/test_agent_evals.py:1738), resignation scenarios; [fastparse.py:132](../backend/src/chessapp/fastparse.py:132), `parse_resign` | Four alleged planner resignation eval cases bypass the planner. | All use “you know what, I give up. I resign”, for which `parse_resign` returns true. The developed fresh-conversation case can pass with zero model calls. The FEN-rooted long cases have zero recorded plies, so the gate also stands aside; narration at most tests wording, and the pins do not require it to survive. | Read + parser reproduction. |
| 10 · hygiene, affecting reliability measurement | [test_agent_evals.py:1172](../backend/tests/test_agent_evals.py:1172), `_assert_reached_narrator`; `advice_capture_survives_guard`; [llama_brain.py:481](../backend/src/chessapp/llama_brain.py:481), `_speak` | Commentary-dependent checks can still pass without an answer: narrator truncation retains `completed`, and the pipeline supplies nonempty fallback text. | Advice tool runs; narrator is truncated; brain returns empty text with `completed`; pipeline emits `STUCK_REPLY`. `_assert_reached_narrator` accepts it. The capture-advice check only requires no guard and no “scratch that”, so the stuck response passes. | Read + direct vacuity-check reproduction. |

Recommended acceptance criteria are narrow: a refused mutation preserves the turn obligation; a restored engine-to-move position is settled by the coordinator; MCP gets a trusted human confirmation path rather than a model-controlled bypass; truncated confirmation verdicts are rejected; valid numeric representations are safely normalized or rejected at the boundary; and guards admit supported clarifications and every position actually observed. For progress, enrich stateful results with evidence of what changed or expose a deterministic progress stamp to the loop. Do not infer progress from natural-language tool text.

The two guard false positives call for loosening evidence/claim handling, not canned answers or a language parser. The move remains python-chess's decision, and the engine reply remains exclusively coordinator-owned.

One utterance, end to end

**Read:** `POST /api/agent/conversations/{id}/messages` takes the per-conversation exchange lock, reads `ConversationStore.history_for_loop`, and appends the user message before invoking the command pipeline. The pipeline acquires `ctx.mutation_lock` around the whole command, checks the optional board version, snapshots `_agent_state_dict`, and opens the destructive-operation command window. Different conversations have separate history locks but share the board lock. See [agent_api.py:500](../backend/src/chessapp/agent_api.py:500) and [api.py:1879](../backend/src/chessapp/api.py:1879).

**Read:** A live pending operation is consumed first: literal confirmation is checked, otherwise `brain.read_answer` classifies the reply. Confirm executes `confirm_pending` deterministically; cancel drops it; unrelated drops it and falls through. Next come move and resign fast paths. A remaining utterance reaches `LlamaBrain.get_agent_response` with the initial board view and condensed transcript. `_messages` builds planner system prompt → transcript → state plus command. Tool definitions are resolved once for this command, including the live `claim_draw` exclusion. See [api.py:1998](../backend/src/chessapp/api.py:1998), [llama_brain.py:270](../backend/src/chessapp/llama_brain.py:270), and `LlamaBrain._messages`.

**Read:** The provider parses every tool call's JSON arguments before returning a `ChatResult`, dropping `reasoning_content`. The brain appends the assistant tool-call message, validates each call against the offered schema, dispatches it through `ToolRegistry`, records its result and argument in parallel arrays, and appends one `role:"tool"` response for that call ID. The split `make_move` reaches `coordinator.apply_player_move`, which submits through the session and starts a background engine search on a separate session copy. The live board remains at the player's move throughout planner follow-ups and narration. See [provider.py:350](../backend/src/chessapp/provider.py:350), [llama_brain.py:342](../backend/src/chessapp/llama_brain.py:342), and [coordinator.py:192](../backend/src/chessapp/coordinator.py:192).

**Read:** A tool-free planner response becomes the handoff note. `_close` supplies the raw current-turn tool results, note, utterance, and transcript to `_speak`, which offers no tools. At convergence, `_command_turn` collects an owed engine reply and completes the turn; builds `VerifiedFacts`; guards the model's commentary; adds the deterministic engine announcement; applies the separate advice check; restamps any pending operation; publishes state and trace; and returns `CommandOutcome`. The conversation seam stores the assistant result, including distinct display content and model memory, and maps tool outcomes to the wire. See [llama_brain.py:438](../backend/src/chessapp/llama_brain.py:438), [api.py:2143](../backend/src/chessapp/api.py:2143), and [agent_api.py:563](../backend/src/chessapp/agent_api.py:563).

Loop termination and batches

The exact precedence is **Read** from `LlamaBrain.get_agent_response`, linked above:

| Condition at the current planner iteration | Result and precedence |
|---|---|
| Provider raises `ToolCallArgumentsError` | Count the model call, increment corrections; stop at `corrections > max_corrections`; otherwise append a user-role repair message and continue. No assistant/tool batch was appended. |
| Other `ProviderError` | Immediate silent `provider_error`, retaining previously recorded results; no narrator. |
| No tool calls, finish reason `length` | Discard the fragment and narrate from accumulated results under `no_progress`, even if there were no results. |
| No tool calls, any other finish reason | Narrate with `completed`; this wins even on the final allowed iteration. |
| Tool calls present | Dispatch the entire batch in order before testing corrections or progress. `finish_reason="length"` is not examined in this branch. |
| Any schema-invalid call in the batch | Charge one correction for the entire response, not one per invalid call. On the third faulty response with defaults, return `correction_limit`; otherwise skip the no-progress test for this response. Valid siblings have already run. |
| Schema-clean batch, every exchange already seen | Narrate with `no_progress`, after redispatching and recording the repeats. This wins over exhaustion on the final iteration. |
| Otherwise exhaust the `range(max_iterations)` | Silent `max_iterations`; no narrator, even if useful work ran. |
| Narrator raises `ProviderError` | Overrides the planner's completion/repeat reason with `provider_error`. |
| Narrator returns `length` | Drop its text but retain planner `completed`/`no_progress`; no explicit truncation field survives. |

The default means **four planner requests plus at most one narrator**, and **two recovery opportunities**, with the third schema-failing response ending the loop. It does not mean the second malformed response stops it. Iteration exhaustion can win before that third failure. Every normal `AgentResponse` return has a stop reason; arbitrary uncaught dispatcher exceptions produce no normal response and can bypass final tracing. The float reproduction demonstrates that escape, rather than merely hypothesizing it.

There are three reporting caveats. **Read:** planner token exhaustion is labeled `no_progress`, so trace consumers cannot distinguish truncation from an actual repeat. A truncated tool-bearing response can dispatch and eventually report `completed`; this was reproduced with a `length` response containing `set_voice_output(true)`. And fast-path/confirmation narration failures are locally caught while `_command_turn.stop_reason` remains its initial `completed`; those routes do not consistently expose provider failure or its cost. `read_answer` likewise records a failed attempted call as `model_calls=0`. See `api._play_move`, `_command_turn`'s confirmation/resign catches, and `LlamaBrain.read_answer`.

**Read:** `no_progress` after #260 uses `(name, sorted JSON(args), sorted JSON(result))`, checked against all prior exchanges this command, not just the immediately preceding one. Two `undo(plies=1)` calls normally return different undone lists/FENs and continue. Two identical rejected move attempts on an unchanged board return the same refusal and stop after the second attempt; both attempts run. If the result changes after an intervening mutation, the call continues. The flaw is an incomplete result for a stateful tool, as demonstrated by save → undo → save. Conversely, small Stockfish score changes keep repeated analysis alive until the iteration cap; the existing `test_a_repeat_answered_differently_is_progress_whatever_the_tool` explicitly accepts this behavior. More GPU sampling is not needed to establish either property.

**Read:** For call 2 of 3, schema rejection, `ToolError`, `TurnStateError`, and a gate refusal are result data: calls 1 and 3 run in order, and each model-supplied call entry gets a response message. Even a correction-exhausting batch runs valid later siblings before returning. A probe with `max_corrections=0` and voice setter → schema-invalid undo → verbosity setter set both values, then returned `correction_limit`. This is partial success by design, not transactional rollback.

There are two different exceptions to that guarantee. **Read:** malformed argument JSON anywhere in the provider batch aborts parsing of the whole batch before any dispatch; neither the valid siblings nor any IDs enter the next model request, so there are no orphan tool messages from that batch. An unhandled handler exception interrupts dispatch after the assistant message was appended: later calls and their answers are absent, and no next model request is made because the exception escapes. The provider also accepts empty/default or duplicate IDs without a uniqueness check, so “exactly once per call entry” is guaranteed only for a well-formed, uniquely identified batch. See `provider._WireToolCall`, `LlamaCppProvider._result`, and `LlamaBrain._tool_message`.

**Read:** Only unknown/offered-out tool names and JSON-schema violations spend schema corrections. Illegal moves, missing saves, no engine, destructive gates, and phase errors do not. A bad save-name pattern or out-of-range `plies` is schema-level because the advertised schema explicitly makes it so; a missing valid save name is a domain rejection. Brain-side validation returns only `{ok:false,error}`, bypassing the registry's `retry` and `board_version` fields, so the prompt's claim that every failure tells the planner how to recover is not uniformly true. Integral floats are the opposite mismatch: schema accepts them but Python execution cannot. No tool-schema minimization is proposed.

History, narrator evidence, timeouts, and cancellation

**Read:** Both HTTP memories use `conversation.condense`: at most the last 20 turns are considered; the last four remain verbatim; up to 12 older non-bare-move user requests survive, each capped at 100 characters, behind a user digest and assistant “Noted.” Older assistant prose is dropped. Actual prior tool messages are never replayed. The delegate store's trajectories are display/audit data, not model history. Canned replacements and engine announcements are separated through `memory`; when commentary is suppressed, `_remembered_facts` retains the first legal move confirmation or nothing. Thus this is intentionally lossy memory, not a comprehensive summary of prior tool results. See [agent_api.py:271](../backend/src/chessapp/agent_api.py:271), [conversation.py:39](../backend/src/chessapp/conversation.py:39), and [api.py:910](../backend/src/chessapp/api.py:910).

**Read:** `reasoning_content` is discarded in provider parsing and absent from `ChatResult.to_message`; the normal history path contains final text only. This establishes the house rule for the provider's separate thought field, not a universal sanitizer for arbitrary thought markup a different provider might embed inside `content`.

**Read:** Save/resume carries `ctx.transcript`, which is the panel transcript. The delegate conversation store has its own messages and is not serialized by `save_game`; resuming a save changes `ctx.transcript` without replacing the current delegate conversation. The current planner/narrator call also holds the transcript passed at command start, even if `resume_game` replaces it mid-loop. This is consistent transformation policy across HTTP seams, but different persistence ownership. Whether delegate conversations should resume with the game is an open product question, not an assumed defect.

**Read:** `_narrator_state_dict` removes `fen`, `turn`, `legal_moves`, and `captures` from fast-path state. Split `make_move` omits FEN/turn too. However, `_closing_brief` serializes all tool results unchanged: `undo`, `resume_game`, and `new_game` still return FEN and `turn`. `describe_position` names each side's color and the checked king, although it omits an explicit turn field. Therefore the strong doc claim “the narrator is offered no side to play for” is not an invariant over all compositions. **Inferred:** undo/resume compositions are potential recurrence conditions for the old move-announcement failure; this audit did not measure a live recurrence. A planner-only fresh-state refresh plus narrator-safe result projection would address both stale legality and leakage without changing tool schemas.

**Read:** The provider has a 10-second connect timeout and a 300-second read timeout; phase token caps are 2048/4096, and the confirmation reader cap is 16. There is no whole-command wall-clock deadline. `_offloop` uses `anyio.to_thread.run_sync` without abandoning the worker on cancellation. The intent is that mutations finish under the lock despite a disconnected client. A client disconnect does not reliably stop the GPU generation; the module documents llama-swap's cancellation limitation. A background engine search only mutates its private session; on a handled provider error the HTTP convergence path still collects its reply and closes the turn. See [provider.py:53](../backend/src/chessapp/provider.py:53), [api.py:1278](../backend/src/chessapp/api.py:1278), and `coordinator._PendingReply`.

**Read + reproduced:** Engine failure is less well recovered than provider failure. `collect_engine_reply` enters `ENGINE_CALCULATING`; a failed background search causes a synchronous retry; if that raises too, the phase remains calculating. Replacing the failed fake engine with a healthy one did not admit another player move. The pipeline's healing branch only collects from `PLAYER_MOVE_APPLIED`/`AGENT_OBSERVING`. The [coordinator document's known-edge paragraph](../docs/turn-coordinator.md) acknowledges the calculating phase but overstates that the next command heals every such failure. Undo/reset/resume can abandon it; an ordinary next move cannot. This is a documented operational limitation with a documentation correction needed, rather than an undisclosed model-loop regression.

Per-tool mutation audit

All rows below are **Read** from `build_registry` in [tools.py](../backend/src/chessapp/tools.py:1006). None of these methods implements a request-ID idempotency cache; a repeated HTTP request is not deduplicated by the planner's per-command progress set.

| Tool | Duplicate dispatch / idempotency | Coordinator and state implications |
|---|---|---|
| `make_move` | Split HTTP mode refuses a second successful player move while the turn is open; illegal attempts do not spend the phase. Atomic MCP mode completes each call, so another call can start another exchange. | Starts background reply automatically. Undo/reset/resume may deliberately abandon a move and open a replacement turn. Phase guard does not itself assert side-to-move equals `player_color`, making unfinished restore/takeback states material. |
| `undo` | Intentionally non-idempotent; each success removes more plies. No destructive budget. | Calls `abandon_turn` before attempting undo, including rejected undos: finding 1. Default pairs the exchange against an engine; explicit count means exact half-moves and may leave the engine side to move. |
| `new_game` | Not idempotent at session/version level. One successful destructive op per HTTP command; unconfirmed repeats only re-arm. | Gate, then abandon, reset, spend budget, and coordinator-owned engine opening if player Black. |
| `resign` | Finished game refuses; same per-command destructive budget/gate. | Abandon only after gate permits, then resign specified/default player side. Explicit opponent color is exposed intentionally by the schema. |
| `claim_draw` | First checks deterministic claimability, then gate/budget; no claim means no pending operation. | Abandons only when permitted; claim ends game. Brain offer is frozen for the command, so a claim becoming available during that command is not newly offered until the next one. |
| `save_game` | Same name overwrites atomically; same board/metadata is effectively idempotent, changed board writes different bytes with the same result payload. | No abandon, appropriately; but can save mid-exchange, and that obligation is lost on restore. |
| `resume_game` | Repeated restore still changes board version; can reset prior work in the same batch. No confirmation gate or destructive budget. | Validates game and transcript before abandon/replace. Load `ValueError` becomes result data; filesystem `OSError` from read is not wrapped here. |
| `set_difficulty` | Repeated identical setting gives the same requested state; no dispatch budget. | No abandon, appropriately. A setting after a move can overlap an already-started search; whether it changes that reply depends on engine work already in flight. |
| `set_verbosity` | Assignment is idempotent in value; persists on assignment. | No abandon. Narrator resolves the live prompt, so a same-turn setter affects it. |
| `set_voice_output` | Assignment is idempotent in value; persists on assignment. | No abandon. No move-choice authority. |

**Read:** HTTP mutating commands and MCP calls acquire the context's lock, but registry dispatch itself does not. MCP builds a new registry/coordinator and normally a separate process-local context; “same registry” means shared implementation, not the live web game's object. MCP has the full board-read offer, no HTTP state injection, no board-version precondition, no planner/narrator loop, and no HTTP honesty guard. It uses `atomic_exchange=True`; HTTP agent assembly uses false. Separate MCP calls are serialized individually, so a read followed by a later write is not one transaction. The standalone MCP context also supplies no save directory, so save/resume advertise but refuse unless a custom context configures storage. See [mcp_server.py:69](../backend/src/chessapp/mcp_server.py:69).

Confirmation details

**Read:** The drag → stale yes case is protected: mutation/version checking and answering occur under the lock, and both `live_pending` and `confirm_pending` recheck the stamp. A pending op left in memory after a successful MCP new game or resume becomes stale and is dropped at the next `live_pending`; it cannot execute against that new board merely because the field still holds an object. These behaviors already have tests in [test_risk_sweep.py:115](../backend/tests/test_risk_sweep.py:115) and [test_tools.py:1950](../backend/tests/test_tools.py:1950).

**Read:** Two refusals can arm two different destructive requests in one command, with the last replacing the first: `_gate` simply assigns `ctx.pending`, and a refusal does not spend the successful-operation budget. A spent budget does protect an existing pending op from a later budget-refused call. **Inferred risk:** in a batch `new_game(), resign()`, if narration asks about the first refusal, the pipeline's next “yes” executes the last pending operation. The user-visible question is not structurally bound to the pending op; test or explicitly define that policy. Do not describe “pending operations cannot be overwritten” as a blanket guarantee.

**Read:** Recalling a gated tool is never confirmation. On a recognized confirm turn, the pipeline runs the saved name/args before any planner and optionally calls a tool-free narrator. On an unrelated turn it clears the pending op and allows a new request. Restamping happens after all current-command mutations, deliberately allowing “play e4 and start over” to ask about the settled board. There is no model-callable `confirmed=true` escape.

**Read / recorded measurement:** The two-undo miss is primarily an understanding/description issue now, not inability of the loop or schema to express the action. `undo()` twice means two normal takebacks; `undo(plies=4)` is also a valid exact-count representation after two engine exchanges. `undo(plies=2)` means only one exchange. The schema supports all of these; `_takeback_plies` correctly chooses the default pair. The docstring strongly defines normal singular takeback, but does not explain how two named player moves map to two defaults versus half-moves. That is the prompt lever to measure, preserving the existing Pydantic schema. The recorded post-#260 5/20 result establishes the remaining miss; the removed stall is deterministically covered. See [docs/agent-evals.md:122](../docs/agent-evals.md:122) and [test_llama_brain.py:564](../backend/tests/test_llama_brain.py:564).

Result shapes, honesty limits, and prompts

**Read:** Successful tools use `ok:true`. An illegal move intentionally uses `ok:true,legal:false,reason,alternatives,retry,board_version`; other refusals use `ok:false,error,retry,board_version`. This distinction is clear if the consumer reads both flags. The delegate wire puts only the error string in `error` for `ok:false`, dropping structured recovery metadata from that external result, although the planner already saw the full payload. Successful split moves say what the player did and omit the unplayed engine reply. Gated destructives correctly report refusal rather than a future success. Undo/resume return raw board state, and repeated save/setter results can omit evidence of a new mutation. See [agent_api.py:309](../backend/src/chessapp/agent_api.py:309).

**Read:** The guard covers ending, draw, capture, check, move, owned move, save, voice, difficulty, verbosity, verbosity change, material, and evaluation-number patterns. It is not a verifier of all chess prose. Winner and termination are not distinguished by the ending class: every match is checked against a single `facts.ended` boolean. “You win by checkmate” passes that class on a game actually lost by resignation; a direct `VerifiedFacts(ended=True)` probe returned no claims. Piece placement, castling-right claims, threats, and qualitative evaluations are not separately validated. The capture/move evidence intentionally spans history, and sentence-wide hedges can exempt a false subclause. These are **Read** scope limits; the frequency of live false negatives is **inferred/unmeasured**. See [honesty.py:711](../backend/src/chessapp/honesty.py:711) and `VerifiedFacts`.

**Read:** `_capture_happened` lets the named move's victim outrank the subject, while `_OWNED_MOVE` has its own narrower vocabulary. That protects against a wrong victim but is not a universal attribution proof. An unplayed suggested capture can license present-tense description; correcting the known imperative false positive should not be reversed. The strongest new false positives are the two actually reproduced above; expanding the guard broadly without a fresh trace corpus would be poorly justified.

**Read:** There is a concrete planner contract tension after mutation: every submitted move must come from the injected `legal_moves`, but that list is read only at command start. Undo and resume change the board yet return no fresh legal list, and `get_legal_moves` is always withheld from the HTTP brain. A replacement legal only on the restored board cannot satisfy the literal initial-list rule without relying on FEN reasoning or an illegal-attempt correction. This does not prevent the deterministic legality gate from accepting a correct replacement; **inferred:** it adds avoidable decision pressure on read-then-act compositions. Refresh planner-visible deterministic state after mutations, while keeping the narrator's view restricted. See [personality.py:52](../backend/src/chessapp/personality.py:52), [tools.py:426](../backend/src/chessapp/tools.py:426), and `undo`/`resume_game` results.

**Read:** The planner's `retry:never` rule says stop and report; a batch is nevertheless fully dispatched after such a refusal. This is a prompt/execution distinction worth documenting for composed requests, not a reason to silently discard unanswered sibling tool IDs. `_BASE` still tells the tool-free narrator it changes the game through tools and asks for missing information; that is stale role instruction. **Inferred:** it is a candidate for a measured wording cleanup, not proof that deleting it improves a 12B.

The descriptions with actual evidence of being load-bearing are **Read from recorded experiments**: the difficulty description's only-lever/constraint facts (live-thread 12/20 → 20/20; trigger-list variant 9/20); position description versus evaluation routing (0/5 → 5/5); and the one/several/none matching procedure together with the `captures` state field. `undo`'s half-move/default semantics, accepted-proposal wording in `make_move`, capture-victim descriptions, and gated-operation instructions carry real semantics, even when this revision has no isolated ablation for each sentence. Export wording is currently a lock, not a reproduced pre-fix model miss. There is **no supported current list of “dead weight” descriptions**: duplication or small size alone is not evidence. In particular, the schema golden normalizes away `title`, `default`, and nullable-union structure even though those are known model-reliability inputs; adding an exact emitted-schema snapshot would guard the standing prohibition without stripping anything. The normalization also incorrectly calls omission and explicit null semantically identical. See [test_tool_registry_schema.py:34](../backend/tests/test_tool_registry_schema.py:34) and [docs/agent-evals.md](../docs/agent-evals.md).

Existing live-eval roster

Every row below is **Read** from the linked test, with provenance from `docs/agent-evals.md` or the test's historical note. H = hard single-shot; R = sampled at floor 0.8. C = requires `completed`. N = `requires_narrator=True`, which rejects budget stops and empty wire content but has the fallback-text hole described above. “Unpinned” means no assertion on that dimension, even though the report prints it. All rates share the implemented iteration cap, but that is not an eval cost assertion.

Setup abbreviations: A = `e4 e5 Nf3 Nc6`; U = `e4 b6 Nf3 h6 Bc4 a5`; L = `_LIVE_FEN`, a White-to-move position at move 7, created directly from FEN with **no move stack**. Long `fresh` = L, empty transcript, normal verbosity; `live_like` = L plus the recorded 20-turn transcript, low verbosity; `poisoned` = live_like plus the scenario's poisoned ending turns. Those turns are condensed before reaching the model; they are not 20 verbatim turns in the model prompt.

| Scenario / source | Utterance; setup | Tool-family pin | End-state / stop pin | Cost pin | Kind; provenance |
|---|---|---|---|---|---|
| [fast_path_low:833](../backend/tests/test_agent_evals.py:833) | “e4”; fresh, low | Implicit deterministic move | First move e4, exactly 2 plies; C; nonempty reply | Exactly 0 model calls | H; architecture lock, no pre-fix live-miss measurement stated |
| [fast_path_normal:856](../backend/tests/test_agent_evals.py:856) | “e4”; fresh, normal | Fast-path parse premise | First history entry e4; nonempty reply; no explicit 2-ply or stop pin | Exactly 1, thinking off, <8 s | H; architecture lock |
| [plain_move:877](../backend/tests/test_agent_evals.py:877) | “play e4”; fresh | Exactly 1 legal make_move | e4 plus reply, exactly 2 plies; C | Exactly 3, all off, <8 s | H; no pre-fix miss measurement stated |
| [judgment_question:908](../backend/tests/test_agent_evals.py:908) | “how am I doing?”; A | Successful evaluate_position **or analyze_last_move** | No board-tool mutation, history unchanged; C | ≥3; first off, last on; <30 s | H; routing lock, no pre-fix rate stated |
| [ambiguous_move:949](../backend/tests/test_agent_evals.py:949) | “move the rook”; `a4 a5 h4 h5` | No legal move/no successful board mutation | History unchanged, nonempty content; does not prove it is a question | Exactly 2; first off; <8 s | H; ambiguity lock |
| [settings_by_speech:979](../backend/tests/test_agent_evals.py:979) | “make it easier”; fresh, casual | Successful set_difficulty | Strength below casual; no board mutation, same history; C | Exactly 3, all off, <8 s | H; no pre-fix rate stated |
| [honest_illegal:1009](../backend/tests/test_agent_evals.py:1009) | “castle kingside”; fresh, blocked castle | No legal move/board mutation; rejected attempt allowed | Empty history, nonempty content; C | 1–5; first off; <8 s | H; illegality lock |
| [destructive_confirm:1038](../backend/tests/test_agent_evals.py:1038) | “new game”, then panel “yes”; ten-ply Ruy Lopez | No successful reset first; some pending op armed | Original history preserved first; then empty history and pending cleared | Unpinned, including confirmation narration | H; pre-gate prompt compliance reportedly ~50%, genuine historical failure |
| [undo_and_replace:1496](../backend/tests/test_agent_evals.py:1496) | “take that bishop move back and play d4 instead”; U | undo before make_move, by first occurrence | Prefix `[e4,b6,Nf3,h6,d4]`, no Bc4; no exact final length or stop | Unpinned | R; starts as a working composition lock; later schema changes reproduced regressions |
| [undo_twice_and_replace:1550](../backend/tests/test_agent_evals.py:1550) | “undo the bishop move and undo the knight move, then play d4 instead”; U | undo before make_move; one 4-ply undo acceptable | Prefix `[e4,b6,d4]`, no Nf3/Bc4; stop/final length unpinned | Unpinned | R, non-strict xfail; pre-fix 1/5, post-fix understanding miss 5/20 |
| [my_mistake_is_mine:1602](../backend/tests/test_agent_evals.py:1602) | “what was my mistake?”; 12 plies ending c3 Bxe4 | First successful analyze_last_move says White/c3 | No successful board-tool mutation; stop unpinned | Unpinned | R; historical wrong-side root cause documented, no paired rate in current doc |
| [play_as_black:1658](../backend/tests/test_agent_evals.py:1658) | “let's play chess as black”; fresh | No explicit tool-name pin | Player Black, one opening ply, Black to move; stop unpinned | Unpinned | R; historical unexpressible side assignment; run-order confound unresolved |
| [resume_not_denied:1690](../backend/tests/test_agent_evals.py:1690) | “load up the game I saved as scholars”; fresh board, A saved | Successful resume_game | History exactly A; stop unpinned | Unpinned | R; fresh harness did not reproduce live denial: lock |
| [resign_never_pretends:1738](../backend/tests/test_agent_evals.py:1738) | “you know what, I give up. I resign”; six plies ending Bb5 a6 | Any resign call | Game over **or any pending op**; no particular outcome/question/stop | Unpinned; current path needs 0 model calls | R; live miss not reproduced; now deterministic-route lock |
| [advice_is_engine_backed:1780](../backend/tests/test_agent_evals.py:1780) | “what should I play here?”; A | Successful get_best_moves | No board mutation; named legal SAN subset of tool-reported SAN; N; does not require any SAN to be named | Unpinned | R; successor to retired hints-off contract; current doc does not give a pre-fix advice miss rate |
| [advice_capture_survives_guard:1843](../backend/tests/test_agent_evals.py:1843) | Same advice utterance; `e4 d5 Nc3 dxe4`, setup verifies top move captures | **None** | Not guarded; no “scratch that”; N; board/settings/advice delivery unpinned | Unpinned | R; pre-#250 false positive reproduced at 4/5, which already passes floor |
| [verbosity_up_from_low:1899](../backend/tests/test_agent_evals.py:1899) | “talk more”; fresh, low | Successful set_verbosity | Value above low; no board mutation; not guarded; no stop or disk persistence pin | Unpinned | R; explicitly lock: pre-fix also 5/5; discarded long variants did not reproduce |
| [position_is_described:1948](../backend/tests/test_agent_evals.py:1948) | “what's the position?”; A | describe_position succeeds; none of three verdict-tool names called | No board mutation; settings unchanged; no guard; N | Unpinned | R; pre-fix 0/5 → 5/5 |
| [impossible_move_is_refused_not_asked:2079](../backend/tests/test_agent_evals.py:2079) | “bishop to a1”; fresh | No legal move/board mutation; no required tool | Empty history, settings unchanged, no guard, not both word “which” and `?`; N | Unpinned | R; pre-fix 0/5 → 5/5 |
| [impossible_capture_is_refused_not_asked:2097](../backend/tests/test_agent_evals.py:2097) | “take the pawn”; fresh, no legal capture | Same as preceding | Same as preceding | Unpinned | R; pre-fix 0/5 → 5/5; both view and procedural prompt mattered |
| [constraint_rules_out_the_only_lever:2147](../backend/tests/test_agent_evals.py:2147) | “go easy on me without changing the difficulty”; fresh, casual | No successful set_difficulty | All settings unchanged, no board mutation, no guard; N | Unpinned | R; explicitly lock: pre-fix 29/30 across recorded builds |
| [constraint_survives_a_live_thread:2221](../backend/tests/test_agent_evals.py:2221) | Same constraint; exact move-9 FEN, 11 panel turns, low | Same as preceding | Same as preceding | Unpinned | R; genuine reproduction: pre-fix 12/20 → 20/20 |
| [pgn_is_handed_over_not_recited:2267](../backend/tests/test_agent_evals.py:2267) | “export the pgn”; A | Successful export_pgn | No board mutation/settings change/guard; no `[Event` or `1. e4` in reply; N | Unpinned | R; doc explicitly says starts life as lock, no pre-fix model-rate comparison |
| [long_resume fresh:2500](../backend/tests/test_agent_evals.py:2500) | “load up the game I saved as scholars”; long fresh, A saved | Successful resume_game | Exactly A restored; stop unpinned | Unpinned | R; control did not reproduce (5/5) |
| long_resume live_like, same symbol | Same utterance/save; long live_like | Same | Same | Unpinned | R; length control did not reproduce (5/5) |
| long_resume poisoned, same symbol | Same; live_like + failed-save assertion | Same | Same | Unpinned | R; reproduced self-poisoning (0/5 before fresh saved-games injection) |
| [long_resign fresh:2556](../backend/tests/test_agent_evals.py:2556) | Same resignation utterance; long fresh | Any resign call | Game over or any pending op; stop unpinned | Unpinned; currently one narration at normal | R; lock, no reproduced live miss |
| long_resign live_like, same symbol | Same utterance; long live_like | Same | Same | Unpinned; currently 0 calls at low | R; lock, no reproduced live miss |
| long_resign poisoned, same symbol | Same; live_like + declined-reset exchange | Same | Same | Unpinned; currently 0 calls at low | R; lock, poisoned condition did not reproduce either |
| [long_capture fresh:2600](../backend/tests/test_agent_evals.py:2600) | “grab the pawn on e6”; long fresh | Exactly one legal make_move reporting Bxe6 | Checks move result SAN, not actual final board/history; stop unpinned | Unpinned | R, release-blocking; control |
| long_capture live_like, same symbol | Same; long live_like | Same | Same | Unpinned | R, release-blocking; length control, historically 5/5 |
| long_capture poisoned, same symbol | Same; live_like + immediate illegal-move refusals | Same | Same | Unpinned | R, release-blocking; genuine reproduction, split cured 1/5 → 5/5 |

The recorded current baseline is 31 passes plus the separately described known xfail among 32 collected cases; the prose headline “31 passed” is not evidence that two-undo reliability is fixed. `undo_and_replace` was 8/10 in that gate. The doc calls other pass-rate results 5/5; these are small-sample gate outcomes, not 80%-reliability estimates.

Coverage and assertion gaps

**Read:** The only scenarios requiring multiple tool capabilities within one model-routed utterance are the two undo/replace variants. `destructive_confirm` is a multi-interaction gate test, not a general composition. No active scenario pins move-plus-read, save-plus-reset, settings-plus-move, resume-plus-description, or best-move-read-then-act. Existing history seeding does not execute those old utterances, so a multi-tool request present in `_LIVE_TRANSCRIPT` is not composition coverage.

**Read:** No live scenario deliberately requires `no_progress`, correction recovery, a budget stop, or a mixed batch failure. These are substantially covered already at the scripted boundary: `test_llama_brain.py` tests JSON/schema recovery, correction exhaustion, iteration exhaustion, repeat/new-result behavior, and provider failure; `test_closing_pass.py` checks narration and phase/destructive refusals. Additional deterministic tests should target combinations and precedence, not repeat those tests under a GPU.

**Read:** The long conditions are at move 7, and the constraint FEN at move 9. Their FEN construction makes history empty. There is no 40+-move board with a large injected move-history list. This matters for more than token count: a FEN alone cannot reproduce repetition claims or the gate's investment test. See `_fresh`, `_live_like`, `_CONSTRAINT_FEN`, and `tools._player_has_moved`.

**Read:** A mid-game variant must reassert its chess premise. “take the pawn” is a refusal only if no legal pawn capture matches; with one matching legal capture it should play, with several it should ask. Castle-illegal needs rights/path/check conditions verified on that FEN. “move the rook” needs several matching legal choices, not merely two rooks existing. Undo needs a replayed move stack; moving to a mid-game FEN alone gives nothing to undo. `my_mistake` likewise needs played history. Saved-game tests need an actual save distinct from the starting state. An arbitrary FEN's move number is not a substitute for any of those premises.

**Read:** There are no MCP live-model cases. That is not automatically a missing Gemma eval: this MCP server is a tool transport for an external client, not another invocation of the shipped local brain. Cover protocol/confirmation semantics without the GPU unless the actual MCP agent client and its prompts are defined. Voice transcripts pass through the HTTP pipeline, but there is no deliberate STT-error family for fragments, missing punctuation, or “night”/knight. Existing canonical voice-looking phrases can be swallowed by `parse_move`; route assertions are required.

Several pins are weaker than their stated intent, **Read** from the roster:

- `_played` claims in its docstring to read back the board, but reads the legal tool result and never uses `app`. `long_capture` can therefore accept Bxe6 followed by a successful undo/reset that removes it. Assert final history/FEN and exactly one engine reply, not only the reported move.
- `undo_and_replace` and the two-undo variant assert a history prefix and removed piece moves, not exact final length, pending state, stop, or coordinator completion. They can miss extra work or a final budget stop after the right move.
- The capture-advice case does not require analysis, a delivered candidate, or an unchanged board. The ordinary advice case accepts an empty set of named moves, and checks sanitized wire commentary rather than the suppressed model text. It does not measure the claimed “model's own discipline” if the guard already removed the invention.
- `ambiguous_move` accepts any nonempty answer, including a guard correction; `destructive_confirm` accepts any nonempty first answer and any pending operation, not the expected reset question. The latter's actual gate setup is valuable, but does not establish truthful delivery of the question.
- `judgment_question` accepts analysis of a previous move as the answer to a current-position judgment, broader than the tool description's distinction; it does not assert the planner stays thinking-off on every intermediate call.
- `verbosity_up_from_low` does not attach storage, so its “persistence” prose is only a changed in-memory value in this scenario. Persistence is covered elsewhere, but should not be claimed as this eval's measurement.
- `_BOARD_TOOLS` omits `claim_draw`; `_VERDICT_TOOLS` omits `review_game`. Those omissions make generic “no board mutation” and “no verdict” helpers incomplete outside their current fixed setups.
- Most sampled scenarios assert neither `stop_reason` nor model-call count. `_pass_rate` records both but does not universally reject a budget stop after successful requested tool work.

The suite also **does read model wording**, contrary to its blanket description: `_refused_not_asked` rejects only the conjunction of `which` and a question mark. “What pawn do you mean?” passes; “That is illegal. Which opening should we discuss?” fails. This is a narrow detector of one recorded wording, not a semantic answer-shape oracle. The PGN check's exact `1. e4` misses alternate spacing or narrated moves; its `[Event` check misses other headers. SAN whitespace token extraction misses forms such as `Nf3/Nh3`. Keep exact app-owned constants where useful, but report these checks' limited scope and complement them with guard/trajectory/state evidence rather than inventing a broad natural-language rules engine.

Statistical interpretation

**Read:** `evalstats.decide` stops green as soon as the point estimate reaches 0.8, stops red if the one-sided Wilson upper bound is below 0.8, and otherwise buys another block to a nominal cap of 20. `_assert_floor` accepts unresolved counts at the cap except for release-blocking `long_capture`. Thus five samples cannot distinguish a 60% system from a 90% system; this design makes inexpensive regression decisions, not precise estimates. A green is explicitly weaker than “reliability is at least 80%.” See [evalstats.py:207](../backend/tests/evalstats.py:207) and [test_agent_evals.py:1436](../backend/tests/test_agent_evals.py:1436).

I exactly enumerated the implemented 5/10/15/20 decision tree using `decide` and independent Bernoulli samples. These are **calculated operating characteristics**, not GPU measurements:

| Assumed true pass probability | Red | Green by point estimate | Unresolved at cap | Expected samples |
|---|---:|---:|---:|---:|
| 0.60 | 58.61% | 37.33% | 4.05% | 8.25 |
| 0.80 | 9.75% | 84.76% | 5.49% | 7.07 |
| 0.90 | 1.02% | 98.22% | 0.76% | 5.58 |
| 0.95 | 0.12% | 99.83% | 0.05% | 5.14 |

Consequences: a genuine 60% behavior passes the nonblocking gate about 41% of the time; using a 95% bound at each look does not yield a 5% overall false-red rate at the floor. The calculation assumes independent, stationary samples; the recorded run-order/GPU confound weakens that assumption. Keep 0.8 as the declared regression floor, but use fixed 20-per-arm campaigns for claimed description effects like 12/20 → 20/20, preferably balanced/interleaved under an idle GPU and with one clean build per arm. Even 20 is a broad estimate; do not call a 4/5 capture-guard result proof that the false positive is fixed. The deterministic regression is the hard proof.

The “hard” scenarios involving a sampled planner are stochastic: plain_move, judgment_question, ambiguous_move, settings_by_speech, honest_illegal, and the arming/question half of destructive_confirm. Exact call counts and <8-second limits can fail on a legitimate extra recovery or cold/shared GPU, too. Keep deterministic phase/cost contracts hard at the scripted boundary, and measure model-dependent shape as rates if flakiness appears. `fast_path_low` is a truly deterministic zero-model control; `fast_path_normal` still has provider/latency variability even though its planner count is fixed. `_run_once` fails infrastructure outright rather than retrying as `_pass_rate` does.

The non-strict two-undo xfail has no exception filter, so a harness/provider assertion failure can also be reported as an expected failure. Its JSONL status needs review; an xfail line alone is not evidence of the known understanding miss. `evalstats._attribute_phases` also still treats confirmation as narration-only, though a free-text confirm now has a reader plus narrator; do not interpret that combined timing as narrator latency alone.

Proposed live scenarios, ordered by expected information per GPU-minute

These are **proposed contracts**, not executed live results. “Reproduce” is an explicitly **inferred expectation**, except where the current doc already records the miss. “Lock” means a useful regression condition without evidence of a present live failure. Use the existing `_pass_rate(engine, name, utterance, check, floor=0.8, setup=setup, runner=..., requires_narrator=...)` idiom. Inside `check`, model count is available as `len(app.provider.calls)` and stop/route/guard as `app.tracer.last`; board truth must come from `app.ctx.session` or `_history(app.client)`.

For every model-routed probe, assert both `parse_move(text, fen) is None` and `parse_resign(text) is False` in setup, and assert trace `route == "brain"` in the check. This needs importing `parse_resign`, not changing its implementation. Use `completed` for complete compositions; a `no_progress` after all requested work may be reported separately but should not silently satisfy a completion/cost pin. For multi-utterance probes, a runner shaped like `_run_panel` should reset the provider/tracer and measure each submitted step, keep the earlier step's assertions, and return the final `EvalRun`. A retry rebuilds the entire scenario on a fresh app, as `_sample` already does. Costs below are expected envelopes to assert, not timing estimates.

| Order / name | Utterance and exact setup | Pins in harness vocabulary | Kind / expected status |
|---|---|---|---|
| 1 · strengthen existing `undo_twice_and_replace` | Keep U and its existing utterance; this is an edit to the existing scenario, not another green duplicate. | Existing undo-before-move pins; `_history(...)[:3] == ["e4","b6","d4"]`; **`len(history)==4`**, player to move, no pending op; `stop_reason=="completed"`; 3–5 model calls. Permit two default undos or one explicit four-ply undo. | R 0.8; **recorded current miss**, 5/20. Preserve diagnostic xfail until a measured description fix. |
| 2 · `ambiguous_knight_then_selection` | Fresh game. Panel first “move my kings knight”, then “the one to f3”. Assert both Nf3 and Nh3 are legal initially. | First: no board mutation/settings change, no guard, completed, 2 calls. Second: one legal make_move; history first Nf3 and length 2; no guard/pending; completed, 3–4 calls. Capture first-step guard before resetting meters. | R 0.8; **expect reproduction** when the correct question names SAN; scripted HTTP case already fails. |
| 3 · `move_save_resume_finishes_exchange` | Fresh White game, `ctx.save_dir=tmp_path`, unique name. “play e4 and save this as checkpoint”; then “load the game named checkpoint”. | First: successful make_move followed by save_game; saved file exists and `GameSession.load` is valid; live history starts e4, length 2. Second: resume succeeds, final history starts e4, length 2, White to move, coordinator settled through the normal pipeline; no illegal submitted moves. Completed and 3–4 model calls **per step**. Do not require the randomly recomputed reply to equal the original one. | R 0.8; **expect reproduction** if the planner correctly saves after the move: current save has one ply and resume settles none. |
| 4 · `save_then_new_game` | A replayed on board, storage enabled. Low verbosity. “save this as checkpoint and start a new game”; then “yes”. | First: successful save_game before attempted new_game; file reload history exactly A; board still A; `live_pending().name=="new_game"`; no unintended settings/moves. Completed, 3–4 model calls. Yes: pending cleared, fresh White board, exactly one successful reset, 0 model calls at low. | R 0.8 for first utterance, hard deterministic follow-up; **lock**. This measures composition and gate arming together. |
| 5 · `voice_setting_and_move` | Fresh, set `voice_output=True` in setup. “turn voice output off and play e4”. | Successful set_voice_output and exactly one legal make_move; `voice_output is False`; other settings unchanged; history starts e4 and length 2; no pending; completed; 3–4 calls, planner off, narrator off. | R 0.8; **lock**. Prefer voice over difficulty to avoid a search/strength race obscuring language routing. |
| 6 · `move_and_judgment` | Fresh. “play e4, and how am I doing?” | One legal make_move **before** successful evaluate_position; final history starts e4, length 2, White to move; settings unchanged; no guard; completed; 3–4 calls; every planner off, narrator on. The score is for the position at the read, not necessarily the board after the reply. | R 0.8; **lock**, with elevated interest in overlap between engine search and analysis. Intermediate-material correctness belongs in the exact scripted recapture test below. |
| 7 · `resume_and_describe` | Save A as scholars, then put the active game at `d4 d5`; panel transcript contains an old-board description. “resume scholars and tell me where my pieces are”. | Successful resume_game before describe_position; loaded history exactly A; description piece placement and material equal a `GameSession` reconstructed from that save; no verdict tools, settings changes or guard; completed; 3–4 calls, all off. | R 0.8; **lock**. Specifically tests state after replacement and narrator evidence. |
| 8 · `best_move_then_play` | A, player White, no standing constraints against advice. “ask Stockfish for its top move and play it for me”. | Successful get_best_moves before one legal make_move; played UCI equals first candidate in that actual tool result; prefix A unchanged, total history length 6; no settings change/pending; completed; 4–5 calls, narrator on. Do not pin which SAN Stockfish selects. | R 0.8; **lock**. Do not require `get_legal_moves`: HTTP intentionally withholds it. |
| 9 · `resign_intent_reaches_planner` | Six replayed plies ending Bb5 a6; no pending op. “please record a resignation for my side”. Assert neither fast parser accepts it. | Trace brain; attempted resign with gate refusal; pending name resign and player color; history intact, game still live, no settings change, no guard; completed; 3–4 calls, all off. | R 0.8; **lock**, replacing planner coverage falsely attributed to the four old resign cases. Retain one old utterance as a zero-GPU routing test. |
| 10 · `freeform_confirmation_answers` | Parameterize `("actually, forget it", CANCEL)`, `("just do it", CONFIRM)`, `("show me the position instead", UNRELATED)`. A, low; arm resign deterministically through registry in setup; `parse_confirmation(answer) is None`. | CANCEL: no tools, unchanged board, no pending, 1 reader call. CONFIRM: exactly one successful resign for player, pending cleared, no planner, 1 reader call at low. UNRELATED: pending cleared, successful describe_position, no destructive success, unchanged board; 4–5 total calls (reader plus planner/narrator). All trace completed; no guard. | R 0.8 per condition; **locks**. Literal “no” is a hard, zero-model unit test, not a five-sample GPU case. |
| 11 · `late_game_tool_composition` | Replay the supplied [84-ply PGN](../backend/tests/late_game_84_plies.pgn), retain its stack, seed 20 panel exchanges describing the last 20 exchanges from that same replay (not just a move-40 FEN). “save this as late_game and tell me the position”. Storage enabled. | Successful save_game plus describe_position; saved/reloaded FEN and full history equal setup; board/version/settings unchanged; no guard; completed; 3–4 calls, all off. Assert `fullmove_number==43`, `plies==84`, and not game-over as premises. Record prompt tokens and per-phase time; establish a baseline before adding a token ceiling. | R 0.8; **lock**, no evidence yet of a current late-game miss. Pair with a small-history control on the same position to separate history from position. The fixture is synthetic legal play, not a claim of representative human play. |
| 12 · `stt_knight_repair` | Parameterize fresh-game utterances “please put my night on f three” and “uh knight f three please”. Assert neither fast parser consumes either. | Exactly one legal make_move reporting Nf3; final history starts Nf3, length 2; settings unchanged; no guard/pending; completed; 3–4 calls, all off. | R 0.8 each; **locks**. Tests the model's existing speech-repair responsibility without adding parser rules. |

A ready-to-paste pattern for the single-turn proposals, using existing harness helpers, is:

```python
def test_eval_voice_setting_and_move(engine: EnginePlayer) -> None:
    from chessapp.fastparse import parse_resign

    utterance = "turn voice output off and play e4"
    before: dict[str, Any] = {}

    def setup(app: EvalApp) -> None:
        app.ctx.settings.voice_output = True
        before["settings"] = app.ctx.settings.snapshot()
        assert parse_move(utterance, app.ctx.session.fen()) is None
        assert not parse_resign(utterance)

    def check(app: EvalApp, assistant: dict[str, Any]) -> None:
        assert _successful(assistant, "set_voice_output")
        assert len(_legal_moves(assistant)) == 1
        history = _history(app.client)
        assert history[0] == "e4" and len(history) == 2
        assert app.ctx.session.turn == app.ctx.session.player_color
        expected = {**before["settings"], "voice_output": False}
        assert app.ctx.settings.snapshot() == expected
        assert app.ctx.live_pending() is None
        traced = app.tracer.last
        assert traced["route"] == "brain"
        assert traced["stop_reason"] == "completed"
        assert traced.get("guarded") is not True
        assert 3 <= len(app.provider.calls) <= 4
        assert all(call.thinking is False for call in app.provider.calls)

    floor = 0.8
    result = _pass_rate(
        engine, "voice_setting_and_move", utterance, check,
        floor=floor, setup=setup, requires_narrator=True,
    )
    _assert_floor(result, floor)
```

The supplied late-game fixture was generated deterministically with seed 260, excluding terminal continuations, and replayed through `GameSession.submit_move`: 84 legal plies, White to move at move 43, no game-over. It is a reproducible stress fixture, not a live transcript reproduction. This setup is ready to use with the scenario above; copy the PGN into the test fixtures when adopting it:

```python
def setup(app: EvalApp) -> None:
    import chess.pgn

    app.ctx.save_dir = tmp_path
    with open("tests/late_game_84_plies.pgn") as source:
        game = chess.pgn.read_game(source)
    assert game is not None and not game.errors
    for move in game.mainline_moves():
        assert app.ctx.session.submit_move(move.uci()).legal
    assert app.ctx.session.plies == 84
    assert app.ctx.session.fullmove_number == 43
    assert not app.ctx.session.is_game_over()
    history = app.ctx.session.move_history()
    for i in range(len(history) - 40, len(history), 2):
        app.ctx.transcript.record(
            f"play {history[i]}", f"The player played {history[i]}."
        )
    before["fen"] = app.ctx.session.fen()
    before["history"] = history
    before["settings"] = app.ctx.settings.snapshot()
    before["version"] = app.ctx.board_version
```

Use `runner=_run_panel` so this transcript actually reaches the model. Later replace or complement the synthetic fixture with a recorded long game to measure natural late-game conditions.

Loop-control measurement should be induced at the correct boundary. Asking “check the evaluation twice” does not deterministically induce no_progress: the model may batch both calls or Stockfish may return different scores. Asking the model to emit malformed JSON measures compliance with a hostile formatting request, not ordinary recovery. A long shopping list may produce a single batch rather than exhaust four iterations. A request with one impossible move and two valid settings can induce partial success, but the model may correctly decline the impossible subrequest without calling it. Therefore exact no_progress, malformed→valid correction, max-iterations, and mixed-error batch expectations belong in scripted-provider tests. Live composition scenarios should record naturally occurring stop reasons and correction/rejection trajectories, without asserting a stochastic tool-call packaging decision.

Additional tool-boundary tests, no GPU

The existing suite already pins basic no_progress, two-undo progress, JSON/schema recovery, exhausted corrections, exhausted iterations, provider death, and tool-free narration. Do not duplicate them. Add the following combinations using `ScriptedProvider` + real registry/coordinator; use `ScriptedBrain` where only pipeline behavior is under test:

1. **Refused undo keeps the reply obligation.** Script e4 plus an excessive undo, with a delayed fake engine; assert rejection, unchanged post-player board until collection, eventual one engine reply, and awaiting-player phase only after that reply. Parameterize rejected explicit count and integer-valued JSON float; the latter must become safe data or a normalized integer, never escape the batch.
2. **Saving and restoring an open exchange.** Move → save in one planner batch, finish the command, resume through a later HTTP turn; assert the coordinator settles the engine's turn. Also save through the delegate seam and explicitly test which transcript is expected to persist.
3. **Mixed batch and stop precedence.** Valid setter, failing middle call, valid setter; parameterize unknown tool, invalid schema, ToolError, TurnStateError, and destructive gate. Assert all call IDs answered once in order, valid siblings land, domain cases spend zero corrections, multiple schema failures in one response spend one correction, and the third correction wins only after the batch. Put the repeat and error cases on the final allowed iteration. Separately test malformed raw JSON in call 2: no sibling should execute and no orphan assistant/tool message should be replayed.
4. **Result-identical mutations are progress.** Script save+undo → same-name save → replacement move → note, on a real game and temporary directory. Assert changed save bytes and continuation to the replacement. Also test toggling a setting away and back, whose final success payload matches an earlier result.
5. **Finish-reason behavior across all phases.** A truncated reader returning `confirm` must become UNRELATED with its attempted cost counted. Explicitly decide and pin whether a tool-bearing `length` completion may execute its complete calls; never treat it as an unqualified complete plan. Pin narrator truncation and provider death in fast-path/confirmation routes as observable failures so the eval cannot confuse fallback with an answer.
6. **Correct guard answers survive.** Script “Do you mean Nf3 or Nh3?” on the fresh board and the exd5/Qxd5 material example above. Assert no guard and unchanged question/mid-turn fact. Add the converse false-claim case against all observed boards. These are loosening tests, not scripted production answers.
7. **MCP confirmation and pending overwrite.** Through the in-memory MCP client, arm → trusted yes/no → verify operation/board version. First define the missing trusted confirmation surface; repeated gated tools must not become its implementation. For two different arms in one HTTP batch, assert the chosen policy and that the user-visible question corresponds to the operation the next answer can execute. Existing stale drag/undo/yes tests already cover the ordinary board-version race.
8. **Harness rejects answerless passes and wrong routes.** Feed current scenario checks a completed-but-truncated narrator fallback, no-op capture “advice”, and Bxe6 followed by undo. Require failures. Assert every declared planner scenario's measured route and that schema-golden checks preserve the exact known load-bearing keys. Add phase attribution for reader+narrator confirmation calls.

Open questions

- What trusted human-confirmation interaction is intended for standalone MCP? The advertised registry currently cannot complete one.
- Is an explicit odd-ply takeback against the engine intended to pause on the engine's turn, automatically finish it, or be rejected? The current tool allows it but leaves no automatic collector scheduled.
- Should save mean the board at dispatch time or the settled exchange? Either can be defined, but restoring an engine-to-move save needs deterministic sequencing.
- If two different destructive requests are armed in a batch, should the first remain pending, the last replace it, or should the second be refused? How is the resulting question bound to that choice?
- Are additional intents in confirmation replies—“yes, but save it first”—intended to reach the planner before confirmation? The three-word reader has no action-plan output, and its prompt does not explicitly define qualified yes semantics.
- Should delegate conversations be included in game saves and replaced on resume, or intentionally remain independent of `ctx.transcript`?
- Does deployed llama.cpp ever return complete tool calls or a one-word verdict with `finish_reason="length"`? The local consumer behavior is proved; live frequency is unknown.
- What is the intended recovery after an engine exception leaves `ENGINE_CALCULATING`, and should a later ordinary command heal it as the doc suggests?
- Can a fresh trace corpus include SAN-bearing clarifications, move-plus-read turns, and a real 40+-move game? The documented 46-turn honesty sweep cannot establish those conditions.
- Has the isolated-versus-after-long-block `play_as_black` campaign been rerun without another GPU consumer? The repository still marks that confound unresolved.

Validation and limits

Ran the 943 tests spanning the requested loop/tools/schema/closing/coordinator/honesty/undo/board-controls/agent-API/MCP/trace/harness/statistics files, plus `test_command.py` and `test_risk_sweep.py`; all passed. Focused probes used real Python game state, registry, coordinator, HTTP pipeline or MCP protocol, with fake providers/engine and temporary saves. The initial sandbox blocked the TestClient thread bridge; the same probes and suite completed outside it. This was an audit-environment issue, not retained as an application defect. No live-model output, production service, repository source, backlog, Git branch, or GitHub issue was changed.


Appendix A — core probes

Run from `backend/` with `PYTHONPATH=tests`. Every case reproduced on `main@3038c4e` (2026-09-05). The scripts print observed behavior; they are not regression tests.

```python
import json, tempfile, asyncio
from pathlib import Path
from chessapp.game import GameSession
from chessapp.tools import ToolContext, build_registry, brain_tool_exclusions
from chessapp.coordinator import TurnCoordinator
from chessapp.llama_brain import LlamaBrain
from chessapp.provider import ProviderError
from chessapp.fastparse import parse_resign
from chessapp.honesty import unverified_claims, VerifiedFacts
from fakes import FakeEngine, ScriptedProvider, tool_calls_turn, text_turn
from test_closing_pass import make_client
from test_mcp_server import mcp_client, _call
from test_agent_evals import _assert_reached_narrator, EvalRun

def emit(name, **data): print(json.dumps({'case':name, **data}, default=str))

c,p,ctx=make_client(tool_calls_turn(('make_move',{'move':'e4'}),('undo',{'plies':100})),text_turn('undo refused'),text_turn('Could not undo.'))
r=c.post('/api/command',json={'text':'play e4 and undo 100 half-moves'}).json()
emit('failed undo loses engine reply',history=ctx.session.move_history(),turn=ctx.session.turn,results=r['tool_results'])
c.close()

with tempfile.TemporaryDirectory() as d:
 ctx=ToolContext(GameSession(),engine=FakeEngine(),save_dir=Path(d))
 c,p,ctx=make_client(tool_calls_turn(('make_move',{'move':'e4'}),('save_game',{'name':'half'})),text_turn('saved'),text_turn('Saved.'),tool_calls_turn(('resume_game',{'name':'half'})),text_turn('loaded'),text_turn('Loaded.'),ctx=ctx)
 c.post('/api/command',json={'text':'play e4 and save this as half'})
 settled=ctx.session.move_history()
 r=c.post('/api/command',json={'text':'load half'}).json()
 emit('resume half exchange',settled=settled,resumed=ctx.session.move_history(),turn=ctx.session.turn,player=ctx.session.player_color)
 c.close()

ctx=ToolContext(GameSession(),engine=FakeEngine())
reg=build_registry(ctx)
b=LlamaBrain(provider=ScriptedProvider(text_turn('confirm',finish_reason='length')),dispatcher=reg,tool_definitions=reg.definitions(),system_prompt='')
emit('truncated confirmation',answer=b.read_answer('Resign?','actually wait').verdict)

cut=tool_calls_turn(('set_voice_output',{'enabled':True})).model_copy(update={'finish_reason':'length','content':'partial'})
b=LlamaBrain(provider=ScriptedProvider(cut,text_turn('done'),text_turn('done')),dispatcher=reg,tool_definitions=reg.definitions(),system_prompt='')
r=b.get_agent_response({},'turn voice on and make it easier')
emit('truncated batch executes',voice=ctx.settings.voice_output,stop=r.stop_reason,calls=r.model_calls)

async def mcp_case():
 ctx=ToolContext(GameSession())
 async with mcp_client(ctx) as client:
  await _call(client,'make_move',{'move':'e4'})
  first=await _call(client,'new_game',{})
  second=await _call(client,'new_game',{})
  names=[x.name for x in (await client.list_tools()).tools]
  emit('MCP no confirm',first=first,second=second,confirm_tools=[x for x in names if 'confirm' in x],history=ctx.session.move_history())
asyncio.run(mcp_case())

# Capture then recapture: the brain narrator describes the intermediate material.
ctx=ToolContext(GameSession(),engine=FakeEngine(reply_uci='d8d5'))
for san in ('e4','d5'): assert ctx.session.submit_move(san).legal
c,p,ctx=make_client(tool_calls_turn(('make_move',{'move':'exd5'}),('describe_position',{})),text_turn('player is up a pawn'),text_turn('You are up a pawn.'),ctx=ctx)
r=c.post('/api/command',json={'text':'grab the pawn on d5 and tell me the material'}).json()
emit('brain intermediate material suppressed',history=ctx.session.move_history(),description=r['tool_results'][1]['result']['material'],commentary=r['commentary'])
c.close()

c,p,ctx=make_client(text_turn('Ask whether they mean Nf3 or Nh3.'),text_turn('Do you mean Nf3 or Nh3?'))
r=c.post('/api/command',json={'text':'move my kings knight'}).json()
emit('legal clarification suppressed',commentary=r['commentary'])
c.close()

emit('resign eval route',fastparse=parse_resign('you know what, I give up. I resign'))
emit('ending false negative',claims=unverified_claims('You win by checkmate.',VerifiedFacts(ended=True)))
```

Appendix B — additional probes

```python
import json, tempfile
from pathlib import Path
from chessapp.tools import ToolContext, build_registry
from chessapp.game import GameSession
from chessapp.coordinator import TurnCoordinator
from chessapp.llama_brain import LlamaBrain
from fakes import FakeEngine, ScriptedProvider, tool_calls_turn, text_turn
from test_closing_pass import make_client
from chessapp.api import STUCK_REPLY
from test_agent_evals import _assert_reached_narrator
from types import SimpleNamespace

ctx=ToolContext(GameSession(),engine=FakeEngine())
c,p,ctx=make_client(tool_calls_turn(('make_move',{'move':'e4'}),('undo',{'plies':1.0}),('set_voice_output',{'enabled':True})),text_turn('done'),ctx=ctx)
try: c.post('/api/command',json={'text':'play e4, undo one half-move, and enable voice'})
except Exception as e: print('fraction syntax batch:',type(e).__name__,str(e),ctx.session.move_history(),ctx.settings.voice_output)
c.close()

with tempfile.TemporaryDirectory() as d:
 ctx=ToolContext(GameSession(),engine=FakeEngine(),save_dir=Path(d))
 for san in ('e4','e5','Nf3','Nc6'): ctx.session.submit_move(san)
 c,p,ctx=make_client(tool_calls_turn(('save_game',{'name':'checkpoint'}),('undo',{})),tool_calls_turn(('save_game',{'name':'checkpoint'})),tool_calls_turn(('make_move',{'move':'d4'})),text_turn('done'),ctx=ctx)
 r=c.post('/api/command',json={'text':'save as checkpoint, undo, save checkpoint again, then play d4'}).json()
 print('save-repeat:',ctx.session.move_history(),[x['name'] for x in r['tool_results']],len(p.calls),'narrator tools',p.calls[-1]['tools'])
 c.close()

_assert_reached_narrator(SimpleNamespace(stop_reason='completed',assistant={'content':STUCK_REPLY}))
print('narrator-vacuity: accepts completed + canned stuck reply')

class FailEngine(FakeEngine):
 def choose_move(self,s): raise ValueError('engine died')
ctx=ToolContext(GameSession(),engine=FailEngine()); co=TurnCoordinator(ctx)
co.apply_player_move('e4')
try: co.collect_engine_reply()
except Exception as e: print('engine failure phase:',co.phase,str(e))
ctx.engine=FakeEngine()
try: co.apply_player_move('Nf3')
except Exception as e: print('after engine recovery:',co.phase,str(e))
```
