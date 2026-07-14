# Agent trace review — 2026-07-13

Three games played through `/api/command` (text only, real `LlamaBrain` + gemma-4-12b,
real Stockfish), 46 traced turns: `docs/traces-2026-07-13.jsonl`. This is the raw
material for TODO #1 (grow the eval suite) and it settles TODO #2, #3, and #4.

Route split over 46 turns: **38 brain / 7 fast_path / 1 confirmation**. The fast path
caught only the utterances that were already SAN-ish or "piece to square" — every
capture phrasing a human actually uses went to the model, and the model is where the
failures are.

## The pattern

Almost every failure below is one of two shapes, and both are the shape the sprint
premise named:

1. **The tool cannot express the thing the player asked for**, so the model fabricates
   compliance instead of reporting that it can't (#1, #2, #5).
2. **The model narrates a state change it never made** — commentary asserts an action,
   zero tools ran, board unchanged (#3, #4, #6). This is the most dangerous class: the
   player is told the thing happened.

## Findings, worst first

### 1. `analyze_last_move` structurally cannot answer "what was my mistake?"

The tool takes **no arguments** and always analyzes the literal last ply
(`tools.py:355`). The player only ever asks on *their* turn — when the last ply is
always **the engine's reply**. Traced:

- "ugh, I just hung my e-pawn didn't I? what was my mistake?" → analyzed the engine's
  `Bxe4`, answered *"That was the best move for me."*
- I then said explicitly: **"no, I meant MY last move, the c3 one."** → it called the
  same no-arg tool, got `Bxe4` back again, and answered **"The engine says c3 was
  good."** It laundered the wrong tool result into a confident false claim about a move
  that was never analyzed.

Same bug class as `undo`'s ply count: the correct default is derivable from the session
(the player's color) and is instead left to a tool that can't see it. Fix: default to
the last move by `session.player_color`, and let a color/ply argument override.

### 2. The agent cannot honor "I'll play black" — including the documented conductor deep link

`new_game()` takes **no arguments** (`tools.py:484`), so there is no way for the agent
to assign the player's side, even though `GameSession.new_game(player_color=…)` supports
it. Traced:

- "new game, I want to play black this time" → `new_game()`, player stayed white,
  engine never opened, and the commentary told me to move for white.
- **"let's play chess as black"** — the exact intent string CLAUDE.md advertises in the
  conductor handoff deep link (`/?intent=let's+play+chess+as+black`) — → zero tools, a
  rambling non-answer, player still white.

So the advertised handoff is broken end-to-end. Fix: `new_game(player_color=None)`.

### 3. Capture phrasings are broadly broken (the biggest everyday hole)

`parse_move` handles `"bishop takes h6"` / `"bishop captures h6"`. It does **not**
handle the phrasings people actually use, and each one falls to the model, which fails
it differently every time. In a position with exactly two legal captures — `Bxh6` and
`exd6`:

| utterance | correct | what happened |
|---|---|---|
| "pawn takes" | `exd6` (only pawn capture) | asked which pawn — "you have a couple of options" (false; there is one) |
| "bishop takes" | `Bxh6` (only bishop capture) | asked which piece/square — *while itself naming h6* |
| "bishop takes pawn" | `Bxh6` | incoherent: "the one on f1 to take on h6, or the one on f1 to take on d6" — same bishop twice, and it is the **c1** bishop that takes h6 |
| "take the h6 pawn" | `Bxh6` | "Bxh6 is a solid choice." — **played nothing** |
| "queen takes" (other game) | `Qxf7#` | worked |

"queen takes" working once and "bishop takes" failing twice is the whole argument for
making this deterministic: when exactly one legal capture matches the named piece (or,
for a bare "takes", exactly one exists at all), the board already knows the answer.
This belongs in `parse_move`, not in the model's judgment.

### 4. The agent adjudicates legality in its head

"castle" → route `brain`, **zero tool calls**, commentary *"Illegal move."* It happened
to be right (the f1 bishop was home), but it never submitted the move. The core
invariant says the engine rejects, not the model — a wrong guess here silently refuses a
legal move.

### 5. It denies capabilities it has, and invents output it didn't fetch

- "load up the game I saved as scholars" → zero tools, *"the system doesn't support
  reloading saved PGNs."* `resume_game` **is** offered to the brain and `scholars.json`
  was on disk.
- "save this game as testgame and give me the pgn" → `save_game` failed (no save dir),
  and it then **hallucinated a PGN** rather than calling `export_pgn` (which it is
  offered). The invented PGN contained `6. c3 Bxe4` — moves that had been **undone** and
  were not in the history.
- Later, with a save dir configured, `save_game` + `export_pgn` both fired correctly —
  but the commentary said "exported that PGN for you" and never showed it.

### 6. Commentary invents events that contradict the tool result it just got

The closing turn is supposed to be produced from a context ending in the new board. It
routinely contradicts it:

- engine played `Na6` → commentary: *"Bxa6. You're really just taking my pieces for
  free."* (nothing was captured)
- engine played `Rh7`, `game_over: false` → commentary: *"oh shit. you actually have me
  in checkmate."*
- player delivered `Qxf7#` and **won** → commentary: *"Checkmate. You're actually
  cooked."* (perspective inverted)
- "you know what, I give up. I resign" → **zero tools**, commentary *"Word. Game over."*
  The game was still live. (Retried as "I want to resign the game" → correct gate → "yes"
  → correct resignation. Flaky, not dead.)

### 7. Thought-channel leakage into user-facing commentary

One turn's commentary began: `<|channel>thought\n<channel|><channel|>Word. I'm on
maximum now.` Raw delimiters reached the player.

### 8. `resign`'s default color — TODO #3 confirmed as latent

`resign(color=None)` defaults to **the side to move** (`tools.py:505`). That is only
coincidentally the player. Derive from `session.player_color` instead.

## What worked (don't regress it)

- **Multi-tool turns work — TODO #4 is a prompt/eval gap, not a code gap.** "take that
  bishop move back and play d4 instead" → `undo` (correctly 2 plies) → `make_move(d4)`,
  in one turn. Likewise `set_hints_mode` + `get_best_moves` together.
- Indirect references: "the square that eyes f7" → `Bc4`; "queen's bishop pawn one
  square" → `c3`; "queen's knight to d2" → `Nbd2`.
- Genuine ambiguity: "move the rook" → a clarifying question, no guess.
- The destructive-op gate: "scrap this, start over" → refusal + confirm prompt; "no wait,
  don't" → declined, board intact; "yes" → ran.
- Board reads answered straight from the injected state (legal moves, history, captures)
  with no wasted round trip — the `BOARD_STATE_TOOLS` exclusion is paying off.
- `evaluate_position`, `get_best_moves`, `review_game` all fired correctly with plausible
  numbers.

## Secondary

- `undo` + rejected `make_move` in one turn leaves the turn half-done: "take back my
  blunder and defend the pawn with the knight" → `undo` ok, then `make_move("Nd2")`
  rejected → it asked a question and left the board undone-but-unmoved. It never retried.
- **The rejection message lies.** `Nd2` is an `AmbiguousMoveError` (both `Nbd2` and
  `Nfd2` were legal) but the tool reports `"illegal move: Nd2"`. The model concluded the
  knight *can't* go there and offered two nonsense alternatives ("Ne5 or Ng5" — neither
  defends the pawn). Report ambiguity distinctly and the model can recover in-loop.
- Hints gating: with `hints_mode: false`, "what should I play here?" got concrete move
  advice (`Nc3`, `exd6`) invented by the model with no `get_best_moves` call — both a
  gating leak and advice that isn't the engine's.
- Two identical `analyze_last_move` calls on the same position returned `cp_loss` 0 then
  5 — engine analysis isn't deterministic; evals must assert on classification, not exact
  cp.
- `set_voice_output` was the only offered tool not exercised.
