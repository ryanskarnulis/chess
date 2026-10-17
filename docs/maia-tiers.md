# Maia-3 tiers

The tiers below `maximum` play [Maia-3](https://github.com/CSSLab/maia3), a
model trained to play like a human of a given rating, in place of the
Stockfish settings they used to play (#510). This note is what the spike
measured on 2026-10-17 (#511), the tier map Ryan chose from it, and how it is
built (#512).

## What Maia-3 is

A transformer trained on Lichess **blitz** games (Jan 2023 – Jul 2025,
positions under 30 s on the clock dropped), resampled so every rating is
equally represented. The ratings are binned 600–2600 in steps of 100, plus one
bin below 600 and one above 2600 (paper, appendix A.1). The Elo option is a
linear blend of two learned embeddings clamped to 0–5000, so any number is
accepted, but only **600–2600** is what it was taught.

It is a UCI engine (`maia3-uci`), with no search: one forward pass per move.
Options: `Elo` (or `SelfElo`/`OppoElo`), `Temperature` (0 is its single most
likely move, the default 1.0 samples its policy), `TopP` (nucleus sampling,
1.0 is off) and `MultiPV`. With `--use-uci-history` it reads the last eight
positions from the `position ... moves` command, so it must be handed the
game's moves, not a bare FEN (today's `EnginePlayer.choose_move` builds its
board from the FEN). License AGPL-3.0: fine for a self-hosted app, and a
separate UCI process keeps it apart from our code.

## Cost (79M on CPU)

On the host (i9-11900KF, 16 threads; the app container has no CPU limit),
three self-play games, 224 moves per row:

| model | threads | median / move | p90 | RSS |
|---|---|---|---|---|
| 79M | 1 | 428 ms | 432 ms | 588 MB |
| 79M | 4 | 151 ms | 160 ms | 602 MB |
| 79M | 8 | 116 ms | 120 ms | 598 MB |
| 5M | 4 | 24 ms | 28 ms | 340 MB |

It loads in about 1.2 s. In the image it costs CPU-only torch (724 MB
unpacked in a venv) plus the 302 MB checkpoint, against today's 511 MB image.

## Strength

`scripts/calibrate_difficulty.py` takes `maia:<elo>[:<temperature>[:<top_p>]]`
contenders. Every game below is against the container's own Stockfish 17.1 at
the app's 0.1 s per move, so a Stockfish tier is exactly what the app plays
today. Games are scored from Maia's side; the gap is Maia minus the opponent
on the logistic Elo curve. Twenty games carry about ±100 Elo near an even
score and much more near a sweep.

**Sampling the full policy (`TopP 1.0`)**:

| Maia | vs | games | Maia scored | gap |
|---|---|---|---|---|
| 0 | beginner | 20 | 7 | −108 |
| 100 | beginner | 40 | 23.5 | +61 |
| 300 | casual | 20 | 5.5 | −168 |
| 450 | casual | 40 | 25.5 | +98 |
| 600 | beginner | 20 | 17 | +301 |
| 600 | casual | 20 | 15.5 | +215 |
| 1000 | casual | 20 | 16.5 | +269 |
| 1000 | intermediate | 20 | 2 | −382 |
| 1400 | intermediate | 20 | 5 | −191 |
| 1800 | intermediate | 40 | 21.5 | +26 |
| 1800 | advanced | 20 | 2 | −382 |
| 2200 | advanced | 20 | 8 | −70 |
| 2400 | advanced | 40 | 16 | −70 |
| 2600 | advanced | 40 | 19 | −17 |
| 3000 | advanced | 40 | 16.5 | −61 |
| 2200 | elo:2400 | 20 | 2.5 | −338 |

Maia against itself orders cleanly: 450 beat 100 33/40, 1000 beat 600 17.5/20,
1400 beat 1000 16.5/20, 2200 beat 1800 14/20 (1800 against 1400 was close,
11/20). It stops gaining at the top of its training range: 2400, 2600 and 3000
all score about even with `advanced`.

**Its single most likely move is far stronger.** `Temperature 0` at 1000 beat
the same setting sampled 16/20 (+241) and beat `intermediate` 13/20; at 1800 it
beat `advanced` 13/20. A known Maia effect: the top move averages out the
mistakes a player of that rating makes. Tiers sample.

## Shuffling below 600, and `TopP`

Sample games at low settings did not read as human. At 100 and 450 Maia
shuffled (`Ra2 Ra1 Ra2 Ra1`, `Qf8 Qd8 Qf8 Qd8`, `Kh8 Kg8 Kh8`), opened 1. g4
2. Nh3 3. Bg2 4. O-O 5. Kh1, and could not mate a queen up. Its top five
first moves at 600 are ordinary (e4, d4, e3, Nf3, c4), so the odd moves come
from sampling the long tail of a flat low-rated policy. Raising the
temperature (1.5, 2.0 at 600) weakened it less than lowering the Elo and
flattens that tail further.

`TopP 0.9` cuts the tail. The share of a side's moves that send a piece
straight back to the square it just left, over six games against `casual`:

| Maia | undo rate |
|---|---|
| 600, `TopP 1.0` | 9.1% (24/263) |
| 600, `TopP 0.9` | 1.7% (3/177) |
| 600, `TopP 0.8` | 0.5% (1/192) |
| 1000, `TopP 1.0` | 1.5% (3/199) |
| 1000, `TopP 0.9` | 0.7% (1/150) |
| 1800, `TopP 1.0` | 1.6% (3/193) |
| `casual` (Stockfish sampler) | 3.1% (15/485) |

**With `TopP 0.9`** (40 games each):

| Maia | vs | Maia scored | gap |
|---|---|---|---|
| 600 | beginner | 36 | +382 |
| 600 | casual | 35.5 | +359 |
| 600, temperature 1.5 | beginner | 37 | +436 |
| 1000 | casual | 38 | +512 |
| 1000 | intermediate | 7 | −269 |
| 1000 | Maia 600 | 36 | +382 |
| 1000 | Maia 1400 | 11 | −168 |
| 1400 | intermediate | 20 | ±0 |
| 1800 | intermediate | 25.5 | +98 |
| 2200 | advanced | 23.5 | +61 |
| 2600 | advanced | 29.5 | +179 |

Cutting the tail makes every setting stronger, by roughly 300–400 Elo.

## The tier map (decided 2026-10-17, for #512)

Every Maia tier at temperature 1.0 and `TopP 0.9`, a sampled human move with
the tail cut:

| tier | today | becomes | against today's tier | step from the tier below |
|---|---|---|---|---|
| `maximum` | Stockfish skill 20 | unchanged | — | — |
| `advanced` | `UCI_Elo` 2000 | Maia 2200 | +61 | — |
| `intermediate` | `UCI_Elo` 1500 | Maia 1400 | ±0 | +168 (Maia 1000) |
| `casual` | sampler, T 250 | Maia 1000 | +512 | +382 (Maia 600) |
| `beginner` | sampler, T 700 | Maia 600 | +382 | — |
| `glitch` fallback | as `beginner` | as `beginner`: Maia 600 | — | — |

All 40 games. The two upper tiers swap one for one at about today's strength.
The two at the bottom cannot: no setting Maia was trained on, played without
its shuffling, is as weak as today's `casual`. Ryan's call: the bottom of the
ladder gets harder and plays like a person, `beginner` at the bottom of Maia's
trained range (600) and `casual` at 1000, evenly spaced below `intermediate`.

The Glitch fallback stays "play as `beginner`", so it becomes Maia 600 too.
That may be stronger than the mover plays on his own (the 12B has only an
upper bound, below about 1373 on the Stockfish scale), which relaxes decision 4
in `docs/glitch-difficulty.md`; Ryan accepted that, and #512 rewords it.

The Maia tiers' Elo labels are not our tier targets: Maia's ratings are
Lichess blitz, and `TopP` shifts strength by hundreds of points, so the map is
read off games against today's tiers, not off the label.

## As built (#512)

- `maia.py`: `MaiaEngine`, one `maia3-uci` process (the 79M model, CPU, four
  torch threads, `--use-uci-history`), handed the game's moves from the
  position it began at and all three options on every call. A dead process
  is started again once, as Stockfish is (#329).
- `engine.py`: each tier's `DifficultyProfile.maia` holds its `MaiaSettings`.
  `EnginePlayer.set_tier` still sets Stockfish up at the tier's old settings,
  and `choose_move` plays Maia's move when one is attached, or Stockfish's when
  it is not, or when Maia fails (logged `maia_move_failed_stockfish_plays`).
  `maximum`, a raw Elo and a raw skill level are Stockfish's.
- `CHESSAPP_MAIA` names the executable and `CHESSAPP_MAIA_CHECKPOINT` the
  weights; unset, or a process that will not start, means no Maia (logged
  `maia_unavailable`). The image installs Maia-3 in its own venv
  (`/opt/maia`, CPU torch, pinned commit and weights) and sets both.
- Who played is said truthfully: the PGN header names "Maia-3" for its tiers,
  and the narrator calls the `glitch` fallback "the backup engine", which is
  true whichever engine played it. Reviews, hints and analysis stay
  Stockfish's.

## Reproduce

From `backend/`, with Maia-3 installed in its own venv (`uv venv`, CPU torch
from `https://download.pytorch.org/whl/cpu`, `pip install -e maia3`,
`maia3-cache --model maia3-79m`) and the container's Stockfish first on PATH
(`docker cp chess-app-1:/usr/games/stockfish`):

    python scripts/calibrate_difficulty.py --games 40 --jobs 7 --max-plies 300 \
        --maia "<venv>/bin/maia3-uci --model maia3-79m --device cpu --no-use-amp --use-uci-history" \
        --pair maia:1400:1:0.9 intermediate --pair maia:2200:1:0.9 advanced

Seven pairings at once kept the load near 5 of 16 threads, so Stockfish's 0.1 s
was not starved.
