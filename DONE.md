# DONE

Completed tasks, newest first, one line each. The full narratives for entries
before 2026-09-01 are in this file's git history; measurement records live in
`docs/agent-evals.md`.

## 2026-09-04

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
