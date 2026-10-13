# Glitch's sight: board facts and looking ahead

Design note for the Glitch tier's next step (#494, roadmap #366 step 20).
**Status: proposed 2026-10-12, not yet agreed.** It builds on the tier as
built (`docs/glitch-difficulty.md`) and amends that note's decision 2, what
the mover sees.

## Why

The first real game in the tier (2026-10-12, game `927c238e`, Glitch black,
mated on move 22) was lost by misreading the board, not by failing to see
deep. Stockfish's read of his worst moves, next to the intent he gave:

| Move | What happened | His intent |
|---|---|---|
| 13… Nxd4 | knight for a pawn; e3 recaptures | "Capturing the central pawn … to gain a material advantage" |
| 17… Qxd4 | queen for a pawn; c3 recaptures | "Capture the hanging pawn on d4 with the queen." |
| 19… Bf6 | walks into Nxf6+ | "Repositioning the bishop to a more active diagonal" |
| 21… Rd8 | allows Qxd8#, mate in one | "… preparing to defend against White's …" |

Every loss shows within one move. He believed d4 was hanging when a pawn
defended it. A language model holds a board badly in its head: it loses
track of what attacks and defends what, and of what a move changes. A human
at his level makes the same mistakes for the same reason, and fixes them by
looking harder at the board, not by consulting an engine.

So the mover gets two things a human has at the board: a clear view of the
position as it stands (**board facts**), and the ability to move the pieces
in his head and see where they land (**`try_line`**).

## Decisions

### 1. The line: python-chess is the board, Stockfish is the engine

The tier's promise stands: "the model chooses, and no engine helps it
choose." What the mover may be given is anything the **rules** say about a
position on the board or a line he names himself:

- what attacks and defends each piece, and which pieces are pinned;
- which pieces hang, by a fixed static rule (decision 2);
- the moves the other side has that capture or give check;
- whether a position is check, checkmate, stalemate or a rules draw;
- the material count, in the usual 1/3/3/5/9.

What he is never given is anything that **judges or searches**:

- no evaluation, centipawns or win chances;
- no best move, ranking or "safe moves" filter of the legal list;
- no search the code chooses: code never picks the other side's reply in a
  line, and there is no exchange calculator (SEE) that resolves a whole
  trade for him;
- nothing from Stockfish, the notes, the opening's name or the conversation
  (decision 2 of `docs/glitch-difficulty.md` is otherwise unchanged).

The test for a new fact: could a careful human read it off the board, one
move at a time, without calculating a line? If it needs a line, it belongs
to `try_line`, where he names the line.

### 2. Board facts: a deterministic module, in the mover's prompt

A new module, `board_facts.py`, builds the facts from a `chess.Board` alone.
It is pure python-chess, thoroughly tested, and knows nothing of the model.

**Hanging**, statically: a piece (not the king) is hanging when it is
attacked and either undefended or attacked by a piece worth less. That is
the glance a player gives the board. It is wrong sometimes (an x-ray, a
pinned defender), and it says nothing about what to do. Attackers and
defenders are direct ones only; batteries and x-rays are for `try_line`.

The facts go in the mover's user message after the board and before the
legal moves. Sketch, from the position before 17… Qxd4:

```
Board facts (from the rules; no judgement of the position):
Material: White 32, Black 29.
Black's pieces under attack: none.
White's pieces under attack:
  pawn d4: attacked by Qd7; defended by Rd1, pawn c3
  pawn h4: attacked by Be7; defended by Bg3
If White were to move now: captures none; checks Nf6+.
```

The exact wording is the build step's to settle and measure. The position
changes every call anyway, so the prompt stays append-only (#362): fixed
instructions, then the growing moves, then this position's section.

### 3. `try_line(lines)`: he names a line, code plays it out

A second mover tool, beside `play_move`:

- `lines`: up to 3 lines per call, each up to 6 plies in SAN, starting with
  his own move from the current position.
- Code plays each line on a copy of the board. A ply that is not legal stops
  that line, with the reason and the legal moves like it (the same refusal
  `play_move` gives).
- For each line it returns the plies played, the board at the end, the board
  facts there, check / mate / stalemate / draw, and the material count.
- It never extends a line, picks a reply, or says which line is better.

Each call is a model round trip on the shared GPU, so the profile caps them
(decision 4). Past the cap, `try_line` refuses with "no looks left; play your
move". The tool stays in the list, so the request's prefix still caches. A
look never counts against `MAX_ATTEMPTS`, which stays the budget for illegal
`play_move`s. The run's `deadline_s` covers looks too, and passing it is the
usual fallback (Stockfish `beginner`).

The game is never touched. Lines are played on a copy, and `play_move` alone
reaches the coordinator's legality gate (#471). The turn record's `mover`
gains `looks` (each line asked and what came back), and `intent` can cite a
line he looked at.

### 4. Model knowledge goes in the profile

Two new `[phases.mover]` keys (`docs/model-profiles.md`):

- `board_facts = true | false`: whether the facts section is in the prompt.
- `look_calls = N`: how many `try_line` calls a run may make. `0` leaves
  the tool out of the request.

A stronger model may need neither, and an arm without them is a profile
edit. The defaults are set from measurement (decision 5), per model. The
deadline is revisited with them: the 26B's 120 s holds about four budgeted
calls.

### 5. Measure first: a blunder suite

`mover_strength.py` takes hours and its bounds are hundreds of Elo wide, too
slow and too coarse to compare arms. A new manual command,
`scripts/mover_blunders.py`, asks the mover for a move in a fixed corpus of
positions and judges each move afterwards with Stockfish at a fixed depth:
the centipawns it gives up against the best move, and a **blunder** when that
is 150 or more.

- **Corpus:** positions where the mover has gone wrong (the four above,
  traced Glitch-tier games, the strength script's games), plus positions with
  something free to take, so an arm that only makes him timid scores badly.
  The mover sees exactly what it sees in a game, built from each position's
  moves.
- **Report:** blunder rate, mean centipawn loss, seconds per move, looks per
  move, fallbacks; several samples per position, with arms interleaved per
  position (consecutive samples on one server are correlated).
- **History:** one line per run in `docs/mover-blunders-history.jsonl`, keyed
  by model and `profile_sha`, like the strength history.
- Stockfish only judges, after the move is chosen. Nothing it says reaches
  the mover. Never in CI; it needs the GPU.

The baseline is the tier as it plays today, recorded before any change.

### 6. What does not change

The planner, the narrator and their prompts; the fallback; the draw-offer
rule; the player's hints, analysis and review; personality (tone only). The
Stockfish tiers are untouched: this is the mover's context alone.

## Order of work

Sub-issues of #494, in order:

1. #495: the blunder suite and its baseline (decision 5).
2. #496: `board_facts.py`, deterministic and tested, not yet wired (decision 2).
3. #497: board facts in the mover's prompt, behind `board_facts`; decision 2 of
   `docs/glitch-difficulty.md` amended; request bytes re-pinned; measured
   against the baseline.
4. #498: `try_line`, behind `look_calls` (decision 3).
5. #499: measure the looks arm, set each model's defaults (`board_facts`,
   `look_calls`, `deadline_s`), a strength run, a live game with context
   capture, and the close-out.

If a step measures worse, stop and discuss (roadmap #366). A null result is
a valid outcome: the facts stay off and the note records why.

## Measuring it

- **Blunder suite** per step from step 3 on, against the baseline.
- **Strength:** `mover_strength.py` once the defaults are set, appended to
  `docs/mover-strength-history.jsonl`.
- **Latency:** seconds per move and fallback rate from both scripts, on an
  idle GPU.
- **Eval gate:** the planner and narrator bytes don't change, but the mover's
  prompt is a prompt, so the gate runs once at step 5, before the defaults
  turn on.
- **Live:** one game in the tier with context capture on, checking the
  mover's requests carry only what decision 1 allows.
