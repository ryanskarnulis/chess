# Phase-4 manual walkthrough — 2026-09-04

The first systematic hands-on pass over the app. Ryan drove the live
deployment (image from `main@309e562`, gemma-4-12b brain, Firefox desktop at
100% and a phone over Wi-Fi); Claude directed, read the turn trace, and took
notes. Automated sweeps covered the rules core and the agent-off game. The
trace written during the session (`turns.jsonl` in the saves volume, rotated
at the start) is the fresh baseline.

## Verdict

The game is solid: rules, board, persistence, review, settings, voice, and
offline play all passed. Six real defects, two of them in the honesty guard
and both hit within minutes of normal play. Fix order below.

## What passed

- **Stack.** Cold `docker compose down/up` healthy in 5 s. TTS 0.4 s. First
  agent turn 16.5 s including the model load. Reachable from a phone through
  the gateway with a valid certificate. Fully offline (WAN unplugged, Wi-Fi on)
  including voice.
- **Rules core.** 40 random games on an agent-off instance, every ply mirrored
  against python-chess: zero divergences. Both castlings, en passant,
  queen and under-promotions, checkmate, stalemate, insufficient material.
  Illegal move → `legal:false` with a reason; stale `version` → 409.
- **Board UI.** New game rolls a side, board flips as black, "Switch to …"
  until the first move, undo takes back the full exchange, captures update,
  game-over dialog with New game / Copy PGN / Review.
- **Text agent.** "knight to f3", "e4", "bishop e2", the typo "cstle" all
  land; board updates before Glitch speaks. Ambiguity gets a question and the
  answer plays the right move. Illegal requests refused, board unchanged.
  Reads match the board. Latency "very acceptable": fast path 0.8–1.0 s model
  time, brain-routed move 2.4 s, reads 3–4 s, eval 7.7 s.
- **Persistence.** Save, real app restart, resume: position, history, and
  captures intact. Undo / resign (with confirmation) / new game by words.
- **Analysis.** "who's winning?" coherent. Post-game review "works great",
  survives dismiss/reopen, Copy PGN copies.
- **Settings.** Tone consistent. Difficulty by words is real and persists
  across restarts (`settings.json`). Voice output on/off works.
- **Voice (phone).** Spoken moves in quiet and noisy rooms, hands-free
  endpointing, half-duplex, autoplay unlock, exit tap, TTS voice and latency,
  mute mid-game, misrecognition fails safe.

## Defects, in fix order

1. **Guard suppresses correct hints whose best move is a capture.** Three of
   three "what should I play?" turns: engine said Rxd1, Glitch said "Take the
   rook. Rxd1 is the move.", the honesty guard read the imperative "Take the
   rook" as a completed-capture claim, found no rook in the captured record,
   and replaced the answer with "Scratch that — I said something the board
   doesn't back up." Root cause in `honesty.py`: bare `take` is in
   `_TAKE_VERBS` and `_CAPTURE_HEDGES` has no imperative/advice hedge. Also
   the likely source of the "random" mid-game suppressions Ryan sees. Fix
   with a hedge for advice phrasing and/or verifying capture claims against
   the suggested move when the turn called `get_best_moves`; regression eval
   on the exact suppressed strings.
2. **Board click hit-testing offset half a square upward** (Firefox desktop,
   100%). Clicking in the top half of a square selects the piece on the
   square below. Drags land. Suspect the board's measured rect diverged from
   the rendered squares after #240/#241/#243. Reproduce in Playwright with
   `elementFromPoint` at square centers; check Chromium.
3. **"talk more" never calls `set_verbosity`.** Twice the model said it would
   give more breakdown; the setting stayed `low` on disk and the guard let
   the claim through. Deterministic fast path for verbosity phrases, plus a
   guard pattern for "more of the breakdown"-style claims.
4. **Container ignores SIGTERM.** `docker compose stop` takes the full 10 s
   grace period and ends in SIGKILL; uvicorn logs shutdown but the process
   stays alive (engine subprocess?), and Docker health kept saying healthy
   while the API was down. Every deploy pays this.
5. **Mistake narration invents the captured piece.** `analyze_last_move` was
   right (Re1 played, Qxe2 best, 265 cp), Glitch said "taken that queen with
   Qxe2" — Qxe2 takes a pawn. Put the captured piece in the tool result and
   verify piece names in capture claims.
6. **Confirmation only understands literal yes/no.** "just do it" after a
   resign question does nothing. Wanted: an unparsed answer to a pending
   question goes back to the model to decide confirm/cancel through the same
   confirm path. Deterministic parse first, model second.

## Smaller notes

- "what's the position?" routes to `evaluate_position` and answers with an
  eval ("You're cooked") rather than a description. Twice.
- PGN by chat is a raw dump with `?` headers and no copy affordance. Fill the
  headers, add a copy chip or point at Options → Copy PGN.
- "go easy on me without changing the difficulty" set beginner anyway. Better
  to name the only lever and ask.
- Illegal-for-every-piece requests ("bishop to a1", "take the pawn" at the
  start) are answered as ambiguous ("Which one?") rather than illegal. Safe
  but misleading; a code rule can pick the rejection template.
- Illegal drags snap back silently. Decide whether that wants a message.
- Desktop stacked column leaves too much dead space.
- "Switch To White" renders in Title Case; the aria label is sentence case.
- One-off: a fast-path move narrated as if no move had landed ("what's the
  move?" then "dxe4.").

## Not covered

- A long game (40+ moves) for late-game tool reliability.
- The exported PGN in an external viewer.
- Android Chrome hands-free, unless the phone was Android.

## Method notes

- Run scratch instances with `CHESSAPP_PORT`, not against the live container;
  the trace file is opened per write, so it can be rotated with `mv` while
  the app runs.
- Never `pkill -f` a pattern that can match `/opt/venv/bin/chessapp`; the
  container's process is visible from the host and it killed the live app.
- The trace answered every "why did Glitch say that" question in one read;
  the guard-suppressed turns keep the suppressed text in `suppressed`.
