# DONE

Completed tasks, newest first, one line each. The full narratives for entries
before 2026-09-01 are in this file's git history; measurement records live in
`docs/agent-evals.md`.

## 2026-09-05

- [x] Claim-draw button (#220 follow-up; TODO "Claim-draw button") — `POST /api/game/claim-draw` mirrors `/api/game/resign`: `_mutation(version)` around `_run_destructive("claim_draw", {})`, so the button asks the gate's own question mid-game (409 + `CONFIRM_QUESTIONS["claim_draw"]`, answered at `/api/game/confirm` from either surface), runs outright where the gate stands aside, and answers a plain 409 with nothing armed when the rules allow no claim (the tool's check, ahead of the gate). The frontend now reads the `claimable_draws` the state document has carried since #220: one Draw button in the bottom bar (Options / Resign / Draw / Hint / Undo) that reads "Claim draw" and dispatches the gated claim while a claim exists, otherwise "Offer draw", disabled until the `offer_draw` path lands (Part B, `docs/draw-offer.md`), and disabled once the game is over. Tests mirror resign's on both sides (api, board controls, board version incl. the stale-version parametrize, trace; BottomBar, useGame, App). No schema change, no prompt change, no gate. (#PR1)
- [x] MCP confirmation surface, run end to end (audit finding 3; TODO "Run the MCP confirmation end to end once") — from a fresh Claude Code 2.1.261 session with only `chessapp.mcp_server` attached (`--mcp-config`, `--strict-mcp-config`, stdio tee'd both ways): the handshake declared a bare `elicitation: {}`, which the server reads as form mode; `make_move e4` came back with Stockfish's reply; the gated `new_game` sent one form-mode `elicitation/create` carrying `CONFIRM_QUESTIONS["new_game"]` verbatim, and Claude Code rendered it as *MCP server "chess" requests your input* with the Confirm box, *Yes, do it* and Accept/Decline while the model's turn waited; ticking and accepting sent `{"action":"accept","content":{"confirm":true}}` and the result was the starting FEN with `confirmed: true`, narrated as a new game; on a second game Decline sent `{"action":"decline"}`, the result was the refusal with `declined: true` and `retry: never`, and `get_board_state` showed the game untouched. The client's model saw only tool results and, after the no, asked the human in chat — which can only lead to another call and another form, as designed. Wire excerpts recorded in `docs/mcp-confirmation-surface.md` ("Verified live"); `docs/turn-coordinator.md`'s limits paragraph, which still said the surface waited on the note, brought current. No code change. (#273)
- [x] MCP confirmation surface, implemented (audit finding 3; TODO "MCP confirmation surface — implement") — the note's sequence, in `mcp_server._mcp_tool`'s call wrapper and nowhere else: a dispatch that comes back with a *newly* armed op releases the lock and puts the app's own question (`tools.CONFIRM_QUESTIONS[op]`, moved out of `api.py` so the board dialog, the free-text reader and MCP read one table) to the client's human by MCP form-mode elicitation, one boolean field where only `accept` with `confirm: true` is a yes; the yes re-takes the lock and calls `confirm_pending`, the existing sole key unchanged, so a board that moved while the form was open runs nothing (`stale: true`) and so does a question a later gated call superseded (MCP is windowless, the newest question stands — the #270 rule seen from this surface, pinned by identity on the armed op rather than by name); a no, a dismissed form, an unticked accept or a client that errored on the ask leaves nothing armed and says `declined: true`; a client whose handshake declared no form elicitation (a bare `elicitation: {}` counts as form, `url` alone does not) gets the refusal with `confirmation_unavailable: true` and its text corrected, nothing armed. The client's model neither emits the request nor answers it, and the advertised schema is byte-identical. Eleven tool-boundary scenarios through the SDK's in-memory client with an `elicitation_callback`, one per acceptance criterion, including a second client reading the board while the form is open and the capability check on every handshake shape; two `test_board_version.py` tests that called the wrapper as a plain function now run the coroutine on a loop; `docs/turn-coordinator.md` gains MCP as the third answering surface. No prompt, model or loop change, so no gate; backend 1814 passed, 47 skipped, lint clean. Claude Code renders MCP elicitation forms with Accept/Decline (its changelog, 2.1.239; the local CLI is 2.1.261); one end-to-end run with our question in that form is left in TODO (#272)
- [x] MCP confirmation surface, designed (audit finding 3; TODO "Later — MCP confirmation surface") — a design note, no code: `docs/mcp-confirmation-surface.md`. Standalone MCP arms the gate and nothing there can answer it, so `new_game`/`resign`/`claim_draw` are unreachable on a game in progress and the refusal's "runs when they say yes" is untrue on that surface. Four options rejected in writing — a `confirm` tool, "call it again to confirm", a gate-off flag for MCP, and making MCP a client of the live game (a different feature, recorded as such) — and one chosen: MCP form-mode elicitation, a server-initiated question the *client* puts to its human and answers with accept/decline/cancel, which the model neither emits nor answers. The wrapper in `mcp_server` dispatches under the lock as today, releases it, elicits the app's own `_CONFIRM_QUESTIONS[op]` (one boolean field), and on a yes calls `confirm_pending` — the existing sole key, unchanged, staleness check included — while decline, cancel and a client without the capability all leave nothing armed and say so in the result. The schema does not change (the byte snapshot and the MCP equivalence test are the guard), the trust boundary is stated (the server trusts the client's declared capability the way a web server trusts the browser's dialog), and the note closes with the acceptance criteria the implementation PR is written against, all at the tool boundary with no GPU (#271)
- [x] Pending-op policy: one question per command (audit §"Confirmation details"; TODO "Later — pending-op policy") — two refusals in one batch armed two ops and the last replaced the first, while the narrated question could be about the first, so the player's next yes ran an op they had not been asked about. Decided and pinned: the first gated call in a command arms its op and its question, and every later gated call in the same command window — the same op again or a different one — is refused with a result naming the question already pending (`pending: <op>`, `retry: never`) and arms nothing, so the narrated question, the question the reader is handed and the op `confirm_pending` runs are one op by construction. Scoped to the command window (`TurnCoordinator.command_open`), because that is the only place two gated calls meet inside one interaction; across interactions the newest question is still the one a yes answers, and a button press or an MCP call stays one dispatch per interaction. Registry tests for the rule, the same-op re-ask, a mutation between the two calls, the windowless surfaces and the window closing; pipeline tests through a real planner batch — the yes runs the op that was asked about, and a free-text answer is read against that op's question. `_BASE` left as is, and said so: the sentence telling the tool-free narrator it changes the game only through its tools is stale role text but harmless to a phase offered no tools, its neighbours in the same bullet (never decide legality, never track the position) still bind the narrator, and cutting it would be a narrator prompt change with no measured symptom to gate against. Gate 47 passed, 11 m 50 s, infra 0 — the day's first clean 47: `undo_twice_and_replace` 4/5 (the residual `plies=1` misread) and `pgn_is_handed_over_not_recited` 4/5 (one recited header block), everything else 5/5, `long_capture` 5/5 ×3, costs unmoved (#270)
- [x] An ambiguous piece ask is asked, not played (TODO "Later — an ambiguous piece ask is sometimes played, not asked") — "move my kings knight" on a fresh board, Nf3 and Nh3 both legal, was played (Nf3, every time) rather than asked about half the time, while "move the rook" with four rook moves was asked. The lever was the planner's one/several/none procedure and the fix is a deletion: its bullet carried a second copy of the `captures` fact between "match the words against `legal_moves`" and "exactly one entry fits: submit it", the bullet above states it once, so the copy is gone and nothing was added. Nothing added is the finding — every arm that put a fact in made the knight ask worse in the same interleaved screen (define "fits" as piece-and-square 8/20 vs 5/20 old; "which of them to play is the player's decision" 13/20; both 18/20; then, forty a sample round-robin on one server, old 26/40 played against the deletion's 3/40, reorder-ask-first 12/40, deletion plus a one-line fact 5/40) — and the screening found that consecutive samples of one prompt on one server are correlated (the old text alone read 5/20, 19/40 and 31/40 in three batches), so arms are compared only interleaved. Harness, alternating blocks of five on one server, 20 an arm: `ambiguous_knight_then_selection` 8/20 → 17/20 (old's twelve misses all a played knight; the deletion's three: two played knights and one correct four-move clarification phrased as a statement, which the guard's decided rule reads as advice), `ambiguous_move` 19/20 → 20/20, `undo_and_replace` 18/20 → 20/20; the knight scenario clears the floor at 20, so its xfail is off. The bullet's neighbouring asks screened clean on the deletion: "take the pawn" and "bishop to a1" refused not asked 30/30, rook asked 20/20, STT knight played 20/20, castle 20/20, undo's first call `undo` with plies omitted 19/20. Gate 46 passed, 1 failed, 12 m 32 s, infra 0: every pass-rate scenario 5/5 with `ambiguous_knight_then_selection` 5/5 unmarked, `long_capture` 5/5 ×3, costs unmoved, the failure `judgment_question`'s 30 s latency ceiling on a correct trajectory (a 32.9 s narrator thought), re-run five times a tree interleaved at 5/5 and 5/5; the same-day control on the unchanged main 46 passed, 1 xpassed, 11 m 51 s, infra 0 (#269)
- [x] Exact emitted-schema snapshot (audit §"Result shapes, honesty limits, and prompts"; TODO "Later — schema snapshot") — the migration golden normalizes away `title`, `default` and the `anyOf`-null union, the three keys whose stripping each collapsed `undo_and_replace` on gemma-4-12b (2026-07-21), so it could not see any of them go. `tests/tool_definitions_emitted.json` is now the exact output of `definitions()`, both `make_move` flavours (the MCP server's atomic exchange and the brain's split one), held byte for byte — key order, whitespace, em-dashes — by `test_emitted_schemas_match_the_snapshot_byte_for_byte`; a deliberate change regenerates it with `CHESSAPP_UPDATE_SCHEMA_SNAPSHOT=1` and reads as a schema change that runs the eval gate, and the failure names the first line that moved. Three positive tests state what the normalizations hide, on the live registry so a regenerated fixture cannot lose them quietly: every derived argument model titled (`set_difficulty`, the hand-written `oneOf`, pinned as the one exception), every optional typed parameter `anyOf: [<type>, null]` with `default: null`, every handler default carried. No schema change, no prompt change, no GPU (#268)
- [x] A named move is a whole takeback, not a plies count (the two-undo reading miss found measuring #260) — "undo the bishop move and undo the knight move, then play d4" reached `undo(plies=2)`, one exchange for two named moves. The lever is `undo`'s description, measured at 20 samples an arm on one base: the old text 19/20 on `undo_and_replace` and 9/20 on `undo_twice_and_replace` (eleven `undo(plies=2)`); an arm that taught the arithmetic ("the player's move and the engine's reply are two") 19/20 via `plies=2` and 4/20 — a number in the text is copied into the argument; an arm with no numbers that said "never a count, once per named move" 14/20 (six `plies=1`) and 15/20; and the shipped text — the old wording minus its two-item enumeration of what the default pops, plus one sentence, call this again for each further named move with plies omitted every time — 20/20 and 18/20, then 20/20 and 16/20 against the old text's 19/20 and 7/20 in alternating blocks on one server (34/40 against 16/40 over both campaigns). The xfail on `undo_twice_and_replace` comes off; the golden changes in the two description strings only and no schema key is touched. Gate 46 passed, 1 xfailed, 11 m 06 s, infra 0: `undo_twice_and_replace` 4/5 unmarked, `undo_and_replace` 5/5, every other pass-rate scenario 5/5 but the knight-ask xfail at 9/15, `long_capture` 5/5 ×3, costs unmoved (#267)
- [x] Eval coverage: the twelve composition and loop-path scenarios (audit §"Proposed live scenarios"; decided: completeness outranks GPU time) — `undo_twice_and_replace` strengthened (exactly four plies, player to move, nothing armed, completed, 3–5 calls); `resign_never_pretends` strengthened into the audit's `resign_intent_reaches_planner` (the gate's refusal is the contract and the armed op carries the player's color); new `ambiguous_knight_then_selection`, `move_save_resume_finishes_exchange`, `save_then_new_game`, `voice_setting_and_move`, `move_and_judgment`, `resume_and_describe`, `best_move_then_play`, `freeform_confirmation_answers` (cancel / confirm / unrelated), `late_game_tool_composition` on the 84-ply fixture (seeded and control) and `stt_knight_repair` (two phrasings), every one pinning route, stop reason, call count, board end-state and a settings snapshot; `_run_steps`, a multi-step panel runner that snapshots earlier steps for the check and retries a dead step as infra; `evalstats` marks a confirmation turn with more than one call UNKNOWN rather than all-narrator; 33 → 47 eval items. Measured on the same harness before and after PR 3 and PR 4: `move_save_resume_finishes_exchange` reproduced finding 2 exactly (0/5, "the restored position owes an engine reply nobody collected") and passes 5/5 on the settled restore; `ambiguous_knight_then_selection` did not do what the audit predicted — the guard was the smaller of its two failure modes, and the planner plays one of the two knights instead of asking about half the time (26 of 50 samples across eight runs, on both sides of the guard fix), so it ships as a measured-miss xfail with the planner's ambiguity procedure filed as the lever; the other nine compositions came in 5/5 on both builds. Post-fix gate 45 passed, 1 failed (that scenario, xfailed in the same PR), 1 xfailed, 11 m 29 s, infra 0; `long_capture` 5/5 ×3, costs unmoved (#265)
- [x] A clarifying question survives the guard, and every board a batch held counts (audit findings 6, 7; the clarification rule decided) — a question sentence naming two or more legal moves is a clarification, not advice (`honesty.unlicensed_advice` replaces `names_a_legal_move`: one named move is still advice however it is punctuated, a statement naming two still hands both over, and every other sentence in the reply is judged on its own), so "Do you mean Nf3 or Nh3?" reaches the player instead of the correction. The brain route's honesty evidence is every position the command held: the command window keeps the board each mutating dispatch left behind, taken off the registry's mutation hook before the state broadcast, and `_verified_facts` checks all of them, so "you are up a pawn" narrated between `exd5` and the engine's `Qxd5` survives while "up a rook" and a move legal on no board the batch held are still guarded; drags, buttons and MCP open no window and record nothing. The ending class deliberately still does not tell winner from termination (the note above `_CLAIM_CLASSES` says what evidence would change that). Seven pipeline tests through the real planner batch plus the predicate's spec. Gate 31 passed, 1 xfailed, 1 xpassed, 6 m 08 s, infra 0: every pass-rate scenario 5/5 but `ambiguous_move` 4/5 (XPASS) and the two-undo xfail 2/5, `long_capture` 5/5 ×3, costs unmoved. `ambiguous_move` re-measured at 20 samples on the fix: 19/20, the one miss a guessed `Rh3` and none eaten by the guard, so the xfail it carried since the morning comes off here (#263)
- [x] A restored position with the engine to move is settled by the coordinator (audit finding 2, decided) — "play e4 and save this" saves between the player's move and the reply, and loading that save, or taking back an odd number of plies, left the engine to move with nobody scheduled to move it. `TurnCoordinator.settle_engine_turn()` replaces `engine_opening_move()`: from `awaiting_player`, with an engine, a live game and the side to move not the player's, it plays the engine's move and comes back awaiting the player without consuming a turn; `undo`, `resume_game` and `new_game` call it after their mutation and report the move as `engine_move` in both registry flavours, `/api/game/undo` does the same (and is attempt-first now, like the tool), and the pipeline announces a settled move with the one deterministic line every engine move gets, so the low-verbosity new-game line no longer spells the opening move itself. An engine that raises inside `collect_engine_reply` leaves the phase at `player_move_applied` instead of `engine_calculating`, so the next command heals the turn (refused move, owed reply played) as `docs/turn-coordinator.md` claimed and now does. Coordinator, tool, route, MCP and pipeline tests, including the audit's probe end to end. PR 5's `move_save_resume_finishes_exchange` reproduced the miss 0/5 on the pre-fix main. Gate 31 passed, 1 xfailed, 1 xpassed, 6 m 37 s, infra 0: `undo_and_replace` 16/20 (every miss the `undo(plies=1)` misread, which the settle now turns into a legal replacement on the wrong board rather than an illegal one), every other pass-rate scenario 5/5, `long_capture` 5/5 ×3, costs unmoved (#266)
- [x] A refusal leaves the turn alone, and the loop's refusals say how to recover (audit findings 1, 4, 5, 8) — `undo` attempts the takeback before abandoning the open turn, so `make_move(e4), undo(plies=100)` in one batch keeps the engine reply e4 is owed instead of leaving Black to move with nobody to move; `read_answer` treats a verdict cut off by the cap as `unrelated` and still bills the round trip; `dispatch` narrows integral JSON floats to the integer their schema names and turns a handler `TypeError` into a `different_args` refusal, so `undo(plies=1.0)` runs and nothing escapes the tool boundary as a 500 on a board that already moved; `save_game` carries `board_version`, so a second save of a changed board is progress to the results-keyed stall rule and the replacement move after it is reached; the loop's own schema refusals are built through the dispatcher's `refusal` (now on the `ToolDispatcher` protocol, `RETRY_*` in `brain.py`) and carry `retry` and `board_version` like every other no. Decided and pinned: a tool-bearing response cut off by the cap executes its complete calls and the loop goes on; a setting toggled on→off→on in three responses still ends the phase after the third lands. Scripted tests at the tool boundary for each, plus the mixed-batch precedence matrix over the five failure kinds the shipped registry can produce (every call id answered once in order, siblings land, only schema failures spend a correction, repeat vs schema error on the final iteration). Gate 31 passed, 2 xfailed, 5 m 48 s, infra 0: `undo_and_replace` 4/5, every other pass-rate scenario 5/5, `long_capture` 5/5 ×3, costs unmoved; the xfails `undo_twice_and_replace` 0/5 (all `undo(plies=2)`) and `ambiguous_move` 2/5, every miss a correct two-move question eaten by the advice guard and none a guessed move (#264)
- [x] The eval harness measures what it claims (audit findings 9, 10 and the weak pins) — every model-routed scenario now asserts in setup that both fast-path parsers stand aside (`_stays_a_model_eval`) and `_pass_rate` pins the traced route on every sample. The four resignation scenarios had been measuring the resign fast path: the same-day control gate on the unchanged harness shows all 20 of their samples at `route=resign` with 0 model calls. They now reach the planner on "please record a resignation for my side" (`resign_never_pretends` 5/5 at 3 calls, the gate arming the resignation every time; `long_resign` 5/5 ×3 at 3 calls, resigning outright because the FEN-rooted long board has no player plies for the gate to guard), and the literal keeps a zero-model hard scenario, `resign_literal_fast_path`. `_assert_reached_narrator` rejects the canned stuck line; `_expect_san` reads the board (one accepted move, nothing else moved, one engine reply, player to move) instead of the tool's report of itself; `_BOARD_TOOLS` derives the game-ending tools from `DESTRUCTIVE_TOOLS` and `_VERDICT_TOOLS` gains `review_game`; `ambiguous_move` and `destructive_confirm` reject a guard correction and pin the armed op; the two-undo xfail is filtered to `AssertionError`; 17 off-GPU harness tests pin the gates and the premises. The stricter harness paid for itself at once: `ambiguous_move`, hard and 1/1 in the control, came in 12/17 across the day (two guessed `Rh3`; two correct "Which one? Rh3 or Rh2?" eaten by the advice guard — finding 6, PR 4), so it ships sampled at the 0.8 floor and xfailed with the measurement. Gate 31 passed, 1 failed (that scenario, before the conversion), 1 xfailed, 6 m 28 s, infra 0; `undo_and_replace` 4/5, `undo_twice_and_replace` 4/10 (xfail), everything else 5/5, `long_capture` 5/5 ×3, costs unmoved (#262)

## 2026-09-04

- [x] A second `undo` in one turn is progress, not a stall (live 2026-07-30 and 2026-08-08) — "undo my knight move and undo the bishop move and then play my knight move" ended after the second undo with the move never played: the planner did the right thing, and the loop's `no_progress` rule cut it off because it keyed a repeat on the call alone, and `undo {}` is the same call every time while popping a different exchange every time. The rule now keys on the call *and its result* — an identical read answered identically still ends the phase, a repeat that comes back different goes on — pinned at the tool boundary on a real `GameSession`. New eval `undo_twice_and_replace` reproduces the trace (pre-fix 1/5, two of the misses the stall); post-fix the stall is gone from every sample, but the model reads two named moves as `undo(plies=2)` three times in four (5/20), so it ships as a non-strict xfail with the `plies` description filed in TODO. Full gate 31 passed, `undo_and_replace` 8/10, costs unmoved (#260)
- [x] A reply no longer moves the board (walkthrough leftover) — Glitch's bubble sits above the board in the column and grew with the reply: a one-line bubble in a 72px row (the spider's own height), every extra line another 25.2px. Measured live: 61px for a four-line reply on a 459px-wide desktop bubble, 112px for the walkthrough's six-line one, and 162px on a phone, where the same words wrap to seven or eight lines in a 258–323px bubble. The bubble is clamped to three lines now (line-height 1.3, a content-box `max-height`) and scrolls inside itself, with the offset put back to the first line for every new reply, and the row reserves exactly a full three-line bubble — 100.1px, +28.1px on the old 72px — so one line and a hundred measure the same and the board never moves. Three lines and not two because a move reply is a reaction, a blank line, then the move, and a two-line clamp would hide the move; the reserve stops at three because the phone has none to spare — on a 440×800 viewport the stack ends 5px above the fixed bar before the first move and 43px above it after. #251's per-interaction re-measure stays, belt-and-braces now rather than the fix: the Copy PGN chip under a reply still grows the row, and reserving for it would cost every turn of every game a row of dead space for something the player asked for once (#259)
- [x] Illegal drags stay silent, decided (walkthrough leftover) — no code change. Chessground puts a dot on every legal destination the moment a piece is picked up and snaps an illegal drop back without firing an event, which is lichess's own convention: the board has already said where the piece may go, and a message after the fact would be scolding the player for something they were shown. If it ever does confuse anyone the message is about fifteen lines away — `api.state.selected` for the piece and `getKeyAtDomPos` for the square it was dropped on (#259)
- [x] "Switch to Black" reads the same on screen as it does aloud (walkthrough leftover) — `.side-picker` and its button both carried `text-transform: capitalize`, which paints "Switch To White" while the accessible name stays "Switch to white" (a transform changes the paint, not the text) and title-cases the "To" into the bargain. The colour is capitalized in the DOM text now and both declarations are gone, so the label and the accessible name are one string; an App test pins the exact name (#259)
- [x] A PGN by chat comes with real headers and a copy button (walkthrough leftover) — "export the pgn" returned `[Event "?"] … [White "?"] [Black "?"]` and the narrator read the whole dump into the bubble, out loud with voice on, with nothing to copy it with. `GameSession` records the day the game began (`started`, additive in the save), `export_pgn` takes composed headers and still owns `Result`, and one composer (`tools.pgn_headers`) fills the Seven Tag Roster for both the tool and `GET /api/game/pgn` so the two can never disagree: the human is "Player" on whichever side they took, the opponent "Glitch (Stockfish, casual)" with the strength spelled the way it was set. The notation is app-owned text now — a copy chip and a collapsed "Show PGN" under the reply — so `export_pgn`'s description says the reply announces it is ready and recites nothing. New eval `pgn_is_handed_over_not_recited` 5/5, full gate 31 passed (#258)
- [x] A constraint the player states is binding (walkthrough leftover) — "go easy on me without changing the difficulty" set beginner anyway. Fixed on `set_difficulty`'s description, not with a guard: the guard checks claims against facts, and knowing a change was excluded means reading the constraint out of the utterance, which is the model's job. The description now carries two facts and no triggers — strength is this setting and nothing else, and a change the player rules out is not made. Fresh, the miss never reproduced (29/30 pre-fix); seeded with the walkthrough's own thread at verbosity low it did, 12/20 — and a first cut that also listed the asks meaning this call went 9/20 there, the trigger list outranking the caveat, while the facts alone went 20/20. New evals `constraint_rules_out_the_only_lever` (fresh, a lock) and `constraint_survives_a_live_thread` (the reproduction). Full gate 30 passed, all 5/5 (#257)
- [x] "what's the position?" gets a description, not a verdict (walkthrough leftover) — new read tool `describe_position` puts the board into one deterministic paragraph for the narrator, which is handed no FEN (#193) and so had nothing to describe from; `evaluate_position`'s description hands the ask over, and the advice guard licenses the one move a description names. Pre-fix 0/5, new eval `position_is_described` 5/5, full gate 28 passed (#256)
- [x] An impossible request is refused, not queried (walkthrough leftover) — "bishop to a1" came back "Which one?" and "take the pawn" "Which pawn?" because the planner contract left asking as the only option for words that fit no legal move; the matching rule is now a procedure (match first; one fits, submit; several, ask; none, refuse), and the per-turn view carries `captures` (what each capturing move takes), the one fact a capture asked for by its victim needs and SAN cannot give. Pre-fix 0/5 both; new evals `impossible_move_is_refused_not_asked` 5/5 and `impossible_capture_is_refused_not_asked` 5/5 — the capture case needed both halves: the rule alone 5/10, the fact alone 0/5 (#256)
- [x] A question turn may not move a setting the player owns — asserted in both new scenarios as a settings snapshot compared after the turn (#256)
- [x] A pending confirmation understands the player's own words (walkthrough #6) — the literal reader still goes first and free, and only an answer it cannot place goes to the model, which returns confirm/cancel/unrelated and never touches a destructive tool; "just do it" resigns, "forget it" doesn't, "undo instead" falls through. Full gate 25 passed (#255)
- [x] Mistake narration stops inventing the captured piece (walkthrough #5) — `analyze_last_move` now reports what the played and best moves each take, and the guard holds a capture claim to what the move it names takes rather than to the game-wide record; full gate 25 passed (#254)
- [x] Container stops in 0.4 s instead of 10 (walkthrough #4) — python-chess drives Stockfish from a non-daemon thread, so an unclosed engine outlived uvicorn and every `compose stop` ended in SIGKILL; `app.serve` owns and closes it. Health probe is a tested module now and flips unhealthy in ~25 s instead of ~71 (#253)
- [x] "talk more" sets verbosity (walkthrough #3) — the live setting is now in the agent's per-turn view and `set_verbosity`'s description says it is the only way to change it; the guard suppresses a narrated change over a turn that called no setter. Eval `verbosity_up_from_low` 5/5, full gate 25 passed (#252)
- [x] Board clicks land on the square you clicked (walkthrough #2) — chessground's cached screen rect went stale whenever the reply above the board moved it; the board now drops the rect at the start of every interaction. Reproduced and verified in Chromium, Firefox and WebKit (#251)
- [x] Guard stops eating correct hints (walkthrough #1) — a bare "take" is advice, not a report, and capture talk hung on an unplayed move describes a line; new eval `advice_capture_survives_guard` 5/5, full gate 24 passed (#250)
- [x] Design stance: code owns truth and safety, the model owns understanding — CLAUDE.md's "prefer code over prompts" rule amended before the walkthrough fixes (#249)
- [x] Phase-4 manual walkthrough — first systematic hands-on pass, live deployment + phone, traced as the fresh baseline; 6 defects and 8 notes filed to TODO (`docs/qa-walkthrough-2026-09-04.md`)

## 2026-09-01

- [x] Docs clean slate: CLAUDE/TODO/DONE/BRIEF and the architecture docs cut to readable length; style-police prompt tests removed (rules are behavioral now — the eval gate, the guards)
- [x] Hints mode retired — hints are on-request and engine-backed; advice guard now enforces evidence, eval `hints_off_no_advice` → `advice_is_engine_backed`, full gate 23 passed / all 5/5 (PCC #314, #245)

## 2026-08-08

- [x] Night-silk board & piece theme — squares, pieces, the six board states, coordinate color (#243)

## 2026-08-07

- [x] Mobile chrome: inline icons, real tap targets, dvh, placeholder fits (#242)
- [x] Board coordinates: opposite-ink labels in the square corners (#241)
- [x] Board frame square again on iOS Safari; column tightened (#240)
- [x] Captures split to their owners; turn pill dropped (#239)

## 2026-08-06

- [x] Master-screen fixes: voice-first command row + merged status row (#238)
- [x] Transport-guarded the remaining `api.ts` helpers (#237)
- [x] Read endpoints serialized against concurrent mutations (#230/#234)
- [x] Refused board move surfaces instead of crashing submit; piece snaps back (#231/#233)
- [x] Hands-free voice survives a dropped request (#232/#235)

## 2026-08-05

- [x] Claimable threefold / fifty-move draws, backend-complete (#220/#228)
- [x] Eval harness seeds saves where the app actually reads them (#227)
- [x] Pending promotion invalidated when the authoritative board changes (#222/#226)
- [x] Delegate messages serialized one exchange per conversation (#221)
- [x] Game saves written atomically (#219)
- [x] Hint arrows bound to the board version they analyzed (#218)

## 2026-08-04

- [x] Direct mode selectable: `CHESSAPP_AGENT=off`; agent-on stays the default; UI told via `agent_available` (#206)

## 2026-07-29

- [x] TTS fetch bounded by a client deadline (from PCC audit #217)

## 2026-07-28

- [x] Voice stops reading SAN as made-up words (pronunciation normalization; PCC #315)
- [x] Observe beat stopped announcing Glitch's next move — data removal on both channels, no new prompt rules (#193)
- [x] mcp pinned below 2 (2.0.0 dropped `mcp.server.fastmcp`)

## 2026-07-27

- [x] Runaway thought loop capped: planner 2048 / narrator 4096 max_tokens, truncation handled
- [x] Narrator offered no side to play for — mid-turn state view carries no turn/legal_moves/FEN (#188)
- [x] Glitch stopped claiming the player's moves as his own — mover attribution + guard direction checks (#185/#186)
- [x] Player's move shows on the board while Glitch is still reacting
- [x] Guard records what it suppressed; two classes stop firing on opinions
- [x] Glitch stopped apologising for things he never said — canned substitutions no longer remembered as his turns
- [x] Sprint-5 measurements read: narrator excess is tokens at flat rate (r²=0.998); wordiness-cap slice rewritten
- [x] Trace trajectory records each call's arguments
- [x] Eval line records per-call output tokens

## 2026-07-26

- [x] Eval line splits planner vs narrator latency
- [x] Declined question no longer burns four planner iterations (`no_progress` stop)
- [x] `llama_brain` surfaces the provider's error message
- [x] Difficulty changes reach the UI and survive restarts
- [x] E2E gap sweep against the audit's risk list — staleness across surfaces fixed via `live_pending`
- [x] Eval statistics worth trusting: Wilson bounds + block-sequential sampling (`tests/evalstats.py`); decision: evals never enter CI

## 2026-07-25 (the agent-control audit sprints)

- [x] Live phase progress in the UI over the state websocket
- [x] Trace gaps closed: `turn_id`, `correlation_id`, derived `mutations`
- [x] Honesty guard material class ("two pawns down" is arithmetic)
- [x] Turn summaries replace the raw transcript window (`conversation.condense`)
- [x] Honesty guard expanded to all verified reply facts (`VerifiedFacts`)
- [x] `hints_off_no_advice` diagnosed and closed (1/5 → 5/5)
- [x] Structured tool errors: every refusal carries `retry` + alternatives
- [x] Live settings injected into the agent's per-turn state
- [x] Code-owned hints gating (capability withheld, not prompt rule)
- [x] Recovery semantics for provider failure mid-turn (landed moves stand)
- [x] Board-version precondition on mutations
- [x] Tool-free closing pass asserted route-by-route (`test_closing_pass.py`)
- [x] Per-turn mutation limits, code-enforced
- [x] Turn state machine + coordinator; engine reply never model-callable
- [x] Full planner/narrator split (cured `long_capture`: poisoned 1/5 → 5/5)
- [x] Move flow split: observe beat while Stockfish computes; deterministic reply announcement
- [x] Board controls route through the coordinator; one confirmation gate for voice and buttons

## 2026-07-13

- [x] Final voice: keep `am_fenrir(2)+am_michael(1)` — judged on real games
- [x] Glitch does swear (watch-item from #95 closed itself)
- [x] Qwen3-TTS voice-design spike dropped — a Kokoro blend won
- [x] Brain owns the bounded tool loop (STANDARD.md §3); pipeline drops phase two (#124)
- [x] Thinking policy measured: round-trips counted, `ChatResult.usage` kept (#125)
- [x] Authoritative player color used everywhere (#126)

## 2026-07-12

- [x] `voice.py` off the OpenAI SDK onto httpx (#119, VOICE-PLAN Phase 2)

## 2026-07-11

- [x] Post-game screen: verdict modal, review, Copy PGN (UI overhaul 3/3)
- [x] Random starting color + pre-game side switch (2/3)
- [x] One stacked layout at every viewport (1/3)
- [x] Agent eval harness + gemma-4-12b baseline (Phase 2 slice 8 — migration epic complete)
- [x] `agent:` block in `app.yaml` (slice 7)
- [x] Conversation persistence + delegate REST endpoint (slice 6)
- [x] httpx `ChatProvider` replaces the OpenAI SDK client (slice 5)
- [x] stdio MCP server off the registry + `.mcp.json` (slice 4)
- [x] Decorator tool registry via `func_metadata` (slice 3)
- [x] Layered personality: global Glitch vendored, chess flavor on top (slice 2)
- [x] LLM env vars renamed to the workspace standard (slice 1)
- [x] Chill pass on Glitch — Ryan's own lexicon (#104)
- [x] iOS Safari VAD out-of-memory fixed: single-threaded ORT wasm (#103)
- [x] Continuous voice death after rebuilds fixed (#99/#100/#101)
- [x] Custom Glitch voice: Kokoro blend on a dedicated TTS server (#97)
- [x] Glitch verified live + prompt tuning (#95)
- [x] Personalities collapsed to Glitch (#93)
- [x] No new-game confirmation after game over + captured-panel cleanup (#91)

## 2026-07-10

- [x] Hands-free voice conversation mode: on-device Silero VAD (#88)
- [x] File-source capture phrases end-to-end ("d takes e5") (#85)
- [x] Deterministic fast-parse path for plain moves (#83)
- [x] STT hardening: chess-vocab prompt + transcript cleanup (#81)
- [x] System prompt hot-path rewrite (#79)
- [x] Domain-retry loop for failed tool calls (#77)
- [x] Agent-facing board state view (~80% prompt cut) (#75)
- [x] Agent voice on phones fixed: audio unlock on the user's tap (#72)
- [x] Chess.com-style mobile UI (#70)

## 2026-07-07

- [x] Beginner tier: weighted move sampler replaces node-limit+blunder (#61)
- [x] Conversation memory, persisted in saves (#59)
- [x] Difficulty tiers with real target strengths (#57)
- [x] Default strength applied; personality is tone only (#55)

## 2026-07-06

- [x] Post-game review panel in the UI (#49)
- [x] Game review: move classification + accuracy (lichess method) (#47)
- [x] Hints + "what was my mistake" explanations (`chessapp.analysis`) (#46)
- [x] Settings by natural speech (acceptance slice) (#45)
- [x] Personality-biased move selection (#44) — later removed with the personality collapse
- [x] Remaining personalities (#43) — later collapsed to Glitch
- [x] Voice fast-path decision: local-only, no Web Speech API (`docs/voice-fast-path-evaluation.md`)
- [x] Voice options: output on/off, mute, talk more/less (#41)
- [x] TTS path: commentary → audio in the browser (#40)
- [x] STT path: mic → transcription → same command pipeline (#39)
- [x] Speaches container in compose (#38)
- [x] Fully-offline gameplay verified in the compose stack
- [x] Frontend served from the app container (#36)

## 2026-07-04

- [x] docker-compose.yml: app + llama-server + voice placeholder
- [x] Backend Dockerfile (multi-stage, Stockfish bundled)
- [x] Frontend CI job (lint + build + tests)
- [x] Command box → agent endpoint + commentary display
- [x] Game controls: new game, undo, resign, difficulty
- [x] Move history + captured-pieces panels
- [x] Pawn-promotion picker
- [x] Interactive board wired to backend state
- [x] React app scaffold (Vite + TS + Chessground)
- [x] `ScriptedBrain` test fake
- [x] App-assembly entrypoint + live personality switching (`chessapp.app`)
- [x] Personalities as system prompts + clarifying-question path
- [x] React-from-new-state step in the game loop
- [x] Defensive tool-call parser + retry loop
- [x] llama-server brain (`LlamaBrain`)
- [x] Brain interface (`chessapp.brain`) (#17)
- [x] `/api/command` endpoint (#18)
- [x] WebSocket `/ws` state channel (#16)
- [x] FastAPI game lifecycle endpoints (#15)
- [x] Settings tools (#14)
- [x] Write tools through the legality gate (#13)
- [x] Tool registry + validated dispatch + read tools (#12)
- [x] Full offline game vs Stockfish, no LLM — acceptance tests (#11)
- [x] Stockfish analysis tools (#10)
- [x] Stockfish bridge via python-chess UCI (#9)
- [x] Export PGN (#8)
- [x] Save / resume (#7)
- [x] Resign / result recording (#6)
- [x] Undo (#5)
- [x] Move history + captures derivation (#4)
- [x] `GameSession` wrapping python-chess (#2)
- [x] Repo, CI, monorepo layout, README, TODO/DONE, scaffold, BRIEF, CLAUDE.md
