# Glitch's sight: board facts and looking ahead

Design note for the Glitch tier's next step (#494, roadmap #366 step 20).
**Status: agreed 2026-10-12; decisions 1 and 2 amended the same day
(#496); decision 3 built 2026-10-13 (#498). Done 2026-10-14 (#499): on
the default brain neither the facts nor the looks earned their place, so both
stay off and the tools wait for a stronger model ("Result of steps 4 and
5").** It builds on the tier as built (`docs/glitch-difficulty.md`) and
amends that note's decision 2, what the mover sees.

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

### 1. The line: his eyes, not his judgement

The tier's promise stands: "the model chooses, and no engine helps it
choose." Today he plays blindfold: he gets the moves, a FEN and a grid of
letters, and has to work out from them what attacks and defends what. Board
facts give him what a player sees by looking at the board, and nothing a
player has to work out. The test for a fact: **could he point at it on the
board without moving a piece in his head and without deciding anything?**

What passes:

- where every piece stands;
- what directly attacks each piece, and what directly defends it;
- which pieces are pinned to their king, and by what;
- whether the side to move is in check, and from what;
- the material count, in the usual 1/3/3/5/9 (in traces only: in the
  prompt it sent the 12B after material to win back, #497).

What fails, and is never in the facts:

- any verdict on a piece: no "hanging", "safe" or "loose" label, however
  simple the rule (amended 2026-10-12, #496). Seeing "d4: attacked by Qd7;
  defended by Rd1, pawn c3" and deciding whether to take is his job;
- the other side's moves: no list of their captures or checks "as if it were
  their move". That is a move generator looking one ply ahead, not something
  seen, and it describes the position before his move, not after it;
- mate, stalemate or a draw marked on any move. In a position he is to move
  in, the game is not over;
- anything that judges or searches: no evaluation, best move, ranking,
  "safe moves" filter, exchange calculator (SEE), or reply code picks;
- nothing from Stockfish, the notes, the opening's name or the conversation
  (decision 2 of `docs/glitch-difficulty.md` is otherwise unchanged).

Threats and "what happens if I play this" are found by looking ahead, with
`try_line` (decision 3), where he names the line. A line that ends in mate
is reported as mate (#498): once the line is played the game on that board is
over, which the rules say, not a judgement.

### 2. Board facts: a deterministic module, in the mover's prompt

`board_facts.py` builds the facts from a `chess.Board` alone. It is pure
python-chess, never imports the engine, is thoroughly tested, and knows
nothing of the model. It returns a structured form for traces and tests, and
a text rendering for the prompt.

Attackers and defenders are direct ones only, cheapest first. A piece behind
another on the same line (a battery, an x-ray) is not counted, and a pinned
defender still counts, with the pin listed separately. A pawn that has just
moved two squares is attacked by a pawn that may take it en passant. The king
is never listed as attacked (check covers that), but attacks and defends
like any piece. Only attacked pieces get attackers and defenders; every piece
is placed, but a full attack map of the board would make the prompt long, and
the 12B has done worse with long prompts.

The facts go in the mover's user message after the board and before the
legal moves (#497), the side to move first. From the position before
17… Qxd4:

```
Board facts (what is on the board; nothing here judges it):
Black: Kg8, Qd7, Ra8, Rd8, Be7, pawns a7 b7 c6 e6 f7 g7 h7.
White: Kc1, Qe2, Rd1, Re1, Bg3, Ne4, pawns a2 b2 c3 d4 f2 g2 h4.
Black's pieces under attack: none.
White's pieces under attack:
  pawn d4: attacked by Qd7; defended by pawn c3, Rd1
  pawn h4: attacked by Be7; defended by Bg3
Pinned to the king: none.
```

#497 may still adjust the wording with its measurement. The position
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

**As built (#498).** `[phases.mover] look_calls`, 0 in every profile, and at
0 the request is byte for byte what it was. With looks on:

- The system prompt gains one paragraph on `try_line`, asking for lines that
  start with his move and go on with the replies he expects, like
  "Qxd4 cxd4". Whether he writes the replies is the open question for #499.
- A line is read as notation: move numbers ("17... Qxd4 18. cxd4"), commas
  and `!`/`?` are skipped, and SAN or UCI is accepted, like `play_move`.
- Each line comes back as text: the line numbered as a score sheet writes
  it, the FEN and board at its end, whose move, how the game stands (mate,
  stalemate, a draw, a draw that may be claimed: the game's own moves count
  toward repetitions), the board facts there, and the material at the end and
  before the line. The facts at a line's end are shown whatever
  `board_facts` says: they are how a one-move look shows "queen d4: attacked
  by pawn c3". Material is shown here though it left the prompt (#497): at
  the end of a line he chose, the count is the outcome he asked about. Both
  are first in line to change if #499 finds them leading him astray.
- A ply past mate is not played, and the line says so. Lines past 3 and
  plies past 6 are named as not played.
- A reply with at least one look in it is not an attempt; any other reply
  is. A look past the cap, or with no lines, is refused and that reply is an
  attempt, so a run makes at most `MAX_ATTEMPTS` + `look_calls` calls.
- Calls in one reply run in order: a `play_move` after a `try_line` plays,
  and anything after it is told the move is already played.
- A reply that looked and played nothing ends his turn: a user turn follows
  the look's answer ("Play your move with play_move, or look again with
  try_line."), so his next reply starts a turn of its own (#499, below).
- The blunder suite records each sample's looks and reports looks per move.

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

- **Corpus:** positions where the mover has gone wrong, harvested from games
  he played (`harvest`, from a saved session, a PGN or a strength report;
  a fallback's move is never counted as his). Positions where the other side
  had just given something up are tagged `free`, so an arm that only makes
  him timid scores badly. Decided positions (best move beyond ±500) are left
  out: losing faster there is not measurable, which drops 19… Bf6 and
  21… Rd8. The mover sees exactly what it sees in a game, built from each
  position's moves.
- **Report:** blunder rate, mean centipawn loss, seconds per move, looks per
  move, fallbacks; several samples per position, with arms interleaved per
  position (consecutive samples on one server are correlated).
- **History:** one line per run in `docs/mover-blunders-history.jsonl`, keyed
  by model and `profile_sha`, like the strength history.
- Stockfish only judges, after the move is chosen. Nothing it says reaches
  the mover. Never in CI; it needs the GPU.

The baseline is the tier as it plays today, recorded before any change, on
the default brain, gemma-4-12b. The 26B and qwen38 are for experiments and
get no baseline unless one is wanted.

**Baseline (2026-10-12, #495).** Corpus `9b35b985c627` of 9 positions: the four
from game `927c238e` (7… O-O, 11… Bxe4, 13… Nxd4, 17… Qxd4) and five from
one 12B strength game against Stockfish 1320 (lost in 30 moves). Three
samples each, judged at depth 18:

| model | blunders | cp loss mean / median | fallbacks | s/move p50 / p90 |
|---|---|---|---|---|
| gemma-4-12b | 23/27, 85% [71%, 93%] | 292 / 310 | 0/27 | 17.6 / 20.0 |

High by construction, since every position is one he blundered in, which
leaves the room a better arm needs to show. Small too: grow it with
`harvest` from future Glitch-tier games (the live trace's `live.json`, or a
strength report) before a result is read as more than a direction.

### 6. What does not change

The planner, the narrator and their prompts; the fallback; the draw-offer
rule; the player's hints, analysis and review; personality (tone only). The
Stockfish tiers are untouched: this is the mover's context alone.

## Result of step 3 (#497): no gain on the 12B, facts stay off

The key shipped as `[phases.mover] board_facts`, off by default, with both
request forms pinned. On gemma-4-12b the facts made the mover worse, so no
profile turns them on.

**Blunder suite** (corpus `9b35b985c627`, 5 samples a position, arms
interleaved, idle GPU, 2026-10-13; `docs/mover-blunders-history.jsonl`). The
facts were this note's first wording, material line included:

| arm | blunders | cp loss mean / median | fallbacks | s/move p50 / p90 |
|---|---|---|---|---|
| off | 27/44, 61% [49%, 72%] | 226 / 201 | 1/45 | 18.3 / 21.1 |
| facts | 33/43, 77% [65%, 86%] | 265 / 278 | 2/45 | 18.6 / 20.7 |

The off arm's `profile_sha` differs from the baseline's only because the
profile now describes `board_facts`. The facts helped at 13… Nxd4 (the best
move, f5, 3 of 4) and hurt most at 17… Qxd4 (5 of 5 with facts, 0 of 5
without).

**Screen on 17… Qxd4 alone** (8 to 10 samples an arm a round, five rounds;
blunder = Qxd4, or Bxh4, which Bg3 takes back the same way):

| arm | material line | thinking budget | blunders |
|---|---|---|---|
| off | | 1024 | 16/38, 42% |
| facts as first written | yes | 1024 | 12/13 |
| + a system-prompt line on what a defender can do | yes | 1024 | 6/8 |
| reworded: "White can take back on d4 with pawn c3 (1), Rd1 (5)" | yes | 1024 | 6/8 |
| the same + the system-prompt line | yes | 1024 | 7/8 |
| facts as first written | no | 1024 | 19/28, 68% |
| defenders listed first | no | 1024 | 7/8 |
| facts after the legal moves | no | 1024 | 4/8 |
| defenders first, after the legal moves | no | 1024 | 7/8 |
| each defender's capture: "pawn c3 (cxd4), Rd1 (Rxd4)" | no | 1024 | 9/10 |
| the same | no | 2048 | 9/20 |
| facts as first written | no | 2048 | 6/9 |
| off | | 2048 | 7/10 |

One round of 8 swings widely (off alone ran 0/5 to 6/8), but three things
held:

- **The material line drove the grab.** His intents read "regain material",
  "restore material balance". It is gone from the prompt rendering.
- **The other side's attacked pieces read as a capture menu.** Bxh4 came up
  13 times across the facts arms and never in 48 samples without them, its
  intent "capture the undefended h4 pawn" beside "defended by Bg3 (Bxh4)".
- **He reads the facts and cannot use them.** Context capture shows him
  quoting "defended by c3 and Rd1", then reasoning "White cannot take back
  with the rook on d1 because it's on d1" and "c3 is a pawn". Nearly every
  call, with facts or without, spends the whole thinking budget; doubling
  it doubled the seconds a move (~34 s) and did not help without facts.

So the gap is the step from "defended by c3" to "cxd4 takes my queen", which
is what `try_line` (#498) shows him directly: he plays the line on a copy and
sees the position after it. The facts are re-measured beside the looks in
#499, where listing only his own attacked pieces (no capture menu) is the
first variant to try.

## Result of steps 4 and 5 (#498, #499): no gain on the 12B, looks stay off

`try_line` works as designed: lines are played, refused and reported
correctly. The 12B cannot make use of it. Two suite runs, corpus
`9b35b985c627`, 5 samples a position, arms interleaved, idle GPU,
2026-10-14 (`docs/mover-blunders-history.jsonl`). "Bad" counts a blunder or
a fallback, since a fallback's move is not his and drops out of the blunder
rate:

| run | arm | blunders | bad | fallbacks | s/move p50 / p90 | looks a move |
|---|---|---|---|---|---|---|
| look continues his turn | off | 34/42 | 37/45 | 3 | 18.0 / 21.5 | |
| | `look_calls = 2` | 26/36 | 35/45 | 9 | 21.7 / 68.8 | 0.8 |
| a fresh turn after a look | off | 31/42 | 34/45 | 3 | 19.7 / 30.8 | |
| | `look_calls = 2` | 29/39 | 35/45 | 6 | 39.4 / 88.7 | 0.9 |

The off arm alone has read 61% to 85% blunders across this note's runs, so
a few points either way is noise. What held:

- **He looks, and imagines the wrong replies.** He looked on four moves in
  five, almost always once, with three lines. At 17… Qxd4 the reply that
  loses the queen, cxd4, never appeared in any line. He tried Nf6+, moves
  that were not legal there, and once `Qxd4 c3`: the defending pawn's
  square, not its capture. Seeing where a line ends does not help a player
  who cannot come up with the line. The one clear gain was m12b-7w in the
  first run (dxe5 five times of five, against 4/5 blunders off), and it did
  not repeat in the second.
- **Gemma 4's template opened his thought for him after a look.** With
  thinking on and a tool answer last, the template ends the prompt with
  `<|channel>thought\n` rather than a fresh `<|turn>model`. The 12B closed
  it at once (about 35 tokens: the `play_move` he had already chosen, no
  reasoning about what he saw), and about one time in seven looped on empty
  thought markers to `max_tokens`. That was the first run's extra
  fallbacks. Replaying those calls without `thinking_budget` behaved the
  same, so the budget was not the cause.
- **A fresh turn fixed the mechanism, not the result.** Ending his turn
  after a look (a short user turn, so the next reply opens its own
  thought) brought the reasoning back, about 1,060 tokens after a look, and
  left one token-cap fallback, against off's three. Bad moves did not
  change, a move took twice as long, and three ~20 s calls overran the
  90 s deadline four times. The fresh turn stays: it is right for any model,
  and nothing reaches it while looks are off.

So no default changes. `board_facts` and `look_calls` stay off in every
profile, and the gate, the strength run and the live game step 5 planned
around new defaults do not apply. The facts variant #499 listed (his own
attacked pieces only) was not run: the gap both runs show is coming up with
the other side's reply, which no listing of the position supplies. Both
tools, the suite and its corpus stay for the next brain; a model that can
read "defended by pawn c3" as "cxd4" is the one to measure them on.

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
a valid outcome: the facts stay off and the note records why. Steps 3 and 5
were both null on the 12B: see their results above.

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
