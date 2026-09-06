# Draw offers

Design note, 2026-09-05. Part B of "draws become playable from every surface";
Part A (the claim-draw button, #274) covered the draws the *rules* hand the
player. This is the other kind: the player offers, and the opponent accepts or
declines. Read `docs/turn-coordinator.md` first for the gate, the budget and the
reply beat this leans on.

## The ask

"Eh, wanna just call it a draw?" — or the bottom bar's Draw button while no
rules-based claim exists. Two surfaces, one answer, and the answer has to be
Glitch's *as an opponent*: he accepts when a draw makes sense for him too, and
otherwise plays on. A claim ends the game because the rules say so; an offer
ends it only if both sides agree, so somebody has to decide for the engine.

## Who decides: code, not the model

The house rule (CLAUDE.md): code owns truth and safety, the model owns
understanding. Whether an offer is *made* is understanding — the planner
recognises "call it a draw", "split the point", "I'd take a draw here" and
routes it to one tool, exactly as it routes a resignation. Whether the offer is
*accepted* is a verdict about the position, and a verdict the model gives is
one the model can be talked into ("c'mon, it's obviously drawn"), can get wrong
from a FEN it cannot read, and can vary with the personality it is playing.
Three things the rule therefore never reads:

- **the model's judgment** — the tool computes the answer before any narration
  happens, and the narrator only voices it;
- **personality** — tone only, never move choice, difficulty or settings, and
  not this either;
- **the difficulty tier** — the offer is a question about the position, not
  about how hard Glitch is playing. A beginner-tier Glitch accepts exactly the
  draws a maximum-tier one does; the number he judges by is Stockfish's own
  full-strength analysis (`evaluate_position`, depth 12), not the throttled
  move sampler.

## The rule

`chessapp/draw_offer.py`, one pure function over `(session, evaluation)` so
every branch is pinned without an engine, plus the constants:

```
DRAW_OFFER_BAND_CP = 50            # |engine-POV eval| ≤ this counts as level
ENDGAME_MAX_NON_PAWN_MATERIAL = 8  # per side, in pawns; queens disqualify outright
```

Accept **iff all four** hold, else decline with the first reason that fails,
in this order:

| # | Condition | Reason when it fails |
| --- | --- | --- |
| 1 | The player has moved (`tools._player_has_moved`, the gate's own notion of investment) | `too_early` |
| 2 | Engine-POV eval ≥ −50 cp, mates included (`pov_cp` maps a mate onto the `MATE_CP` scale, so "no forced mate either way" is the same check) | `engine_ahead` (eval > +50 from the engine's side) |
| 3 | Engine-POV eval ≤ +50 cp | `player_ahead` (eval < −50 from the engine's side) |
| 4 | It is an endgame by material: no queens on the board and neither side holds more than 8 pawns' worth of non-pawn material (a rook and a minor piece, or two minors) | `not_an_endgame` |

The eval reasons come before the material one on purpose: "you're ahead" is
the more useful thing to hear, and it is what the player's fixture asks for.
`player_ahead` is a decline, not a courtesy — a player who is winning may
*offer* a draw, but taking it would be Glitch grabbing half a point he has not
earned, and the rule is about what makes sense for him. Nothing else is a
factor: not the move number, not the halfmove clock, not who is to move.

Pinned by four positions (`tests/test_draw_offer.py`), each one constant's
edge: a dead-drawn rook endgame (accept); a flat middlegame with every piece
on (decline, `not_an_endgame`); an endgame the engine is winning (decline,
`engine_ahead`); a position the player is winning (decline, `player_ahead`).
Plus the band and the material threshold at their boundaries with synthetic
evaluations, and one Stockfish-backed test per fixture (skipped without the
binary, like the other engine tests).

## The session

`GameSession.agree_draw()` — session-level like `resign()` and
`claim_draw()`, because boards model neither agreement nor claiming.
`outcome()` reports termination `agreement`, winner `None`, result
`1/2-1/2`; `is_game_over()` honours it; `undo` refuses a game ended by
agreement the way it refuses a resigned one; `to_dict`/`from_dict` carry
`draw_agreed` (additive, older saves default false); the PGN `Result` comes
off `outcome()` as it already does. Refuses on a finished game.

## The tool

`offer_draw()` in `tools.py`, **offered to the brain on every live turn**
(not in `brain_tool_exclusions`: an offer is always a thing a player can say).
Result:

```
{ok: true, accepted: bool, reason: null | "too_early" | "engine_ahead" |
 "player_ahead" | "not_an_endgame",
 evaluation: {cp_engine_pov: int, mate_in: int | null},
 material: {queens: bool, non_pawn: {white: int, black: int}, balance: int},
 outcome?: {termination: "agreement", winner: null, result: "1/2-1/2"}}
```

Description in `resign`'s register: call it as soon as the player offers a
draw, do not decide the answer yourself, relay whether it was accepted and
why, and never call it twice in one turn. The narrator gets its words from
those fields only. The honesty guard already covers the two lies that matter:
"game over"/"we drew" on a decline is an `ending`/`draw` claim the board does
not back (`UNTRUE_CLAIM_REPLY`), and "you were winning" needs a number the
turn reported — `_analysis_numbers` learns `offer_draw`'s `cp_engine_pov`
(both signs, since the narrator may state it from either side) so an honest
quote survives and an invented one does not.

Finished game → `ToolError`, `retry: never`, like `resign`. No engine → the
same `_require_engine` refusal every analysis tool gives: nobody is there to
accept.

## Gating decision: not gated, but budgeted

`offer_draw` is **not** in `DESTRUCTIVE_TOOLS`: no `_gate`, no
`CONFIRM_QUESTIONS` entry, no 409-and-ask. A declined offer changes nothing,
and an accepted one ends only a position the rule has already judged level and
an endgame — the player asked for exactly that outcome in the same breath.
Asking "you offered a draw, and he accepts; agree?" would be a question about
nothing.

What it does take is the destructive *budget*: `require_destructive_budget()`
before it evaluates, and on acceptance `abandon_turn()` → `agree_draw()` →
`record_destructive_op()`, in that order like `resign`. So "new game and offer
a draw" cannot end the fresh game (budget refused, nothing evaluated), and
"offer a draw and resign" ends the game exactly once. A decline spends
nothing — nothing was thrown away — so a second offer in the same command is
answered again, identically; the description's "never call twice" is the
lever there, not the budget.

Because it is not gated, MCP needs no elicitation for it: the standalone
server gets the tool from the registry for free and dispatches it under the
same lock, and its result is the same answer.

## The coordinator

An offer can arrive with a turn open — "play e4, and wanna call it a draw?" —
so the engine's reply is owed while the offer is judged.

- **Declined with the engine to move:** the tool touches no turn state. The
  command converges as it does for any read (`api._run_command`'s owed-reply
  branch): the reply that has been computing in the background is collected,
  the turn completes, and the app appends its own announcement. The player
  hears the decline and then `e5.` — the reply is still owed, and it is still
  played. The evaluation is of the board with the engine to move, which is the
  position the offer was made on.
- **Accepted with the engine to move:** `abandon_turn()` drops the pending
  computation (the same discard an undo or a resignation makes), the session
  ends in agreement, and no reply is announced. The `ending`/`draw` facts hold
  for the narrator's line.
- **No turn open** (the ordinary case: player to move, the offer is the whole
  utterance): nothing to collect, nothing to abandon between turns.
- **Restored boards** (`settle_engine_turn`) are not involved: an offer
  restores nothing.

## The endpoint and the button

`POST /api/game/offer-draw` (body: the version precondition only) dispatches
`offer_draw` through the registry under `_mutation(version)`, traced as a
control interaction like the other buttons, and returns `{accepted, reason,
evaluation, material, outcome?, state}`; a refusal (finished game, no engine)
is a 409 with the tool's message. Windowless, so the budget is a no-op there,
as for every button.

The Draw button (Part A) already reads "Offer draw" while `claimable_draws` is
empty; PR 4 enables it and routes it here. The answer is shown as one short
deterministic line composed from `accepted` and `reason` — no model on this
path, exactly as the resign and new-game buttons have none. Wording is PR 4's,
but the shape is fixed: accepted → the result; declined → who declined and the
reason in the player's language ("he's ahead", "you're the one ahead", "too
much on the board still", "make a move first").

## What this changes for the evals

Adding a tool changes the emitted schema: the byte snapshot is regenerated
deliberately (`CHESSAPP_UPDATE_SCHEMA_SNAPSHOT=1`), BRIEF.md's tool list gains
`offer_draw` under Writes, and the gate runs before merge because a changed
tool list has collapsed `undo_and_replace` before. Two scenarios join
`docs/agent-evals.md`: `offer_draw_routes` (a mid-game offer reaches
`offer_draw` — not `resign`, not `claim_draw`; the narration is not guarded
and the game is not over, because a middlegame with the pieces on is declined
by construction) and `offer_draw_accepted` (a seeded dead-drawn rook endgame,
two plies played so the player has moved; the offer is accepted, the game ends
in agreement, the narration is not guarded). The recorded baseline is
re-measured with the code PR.

## Not in scope

Glitch offering a draw himself; a draw *offer* by the engine's side over MCP;
tuning the constants by tier. Each would be a design change to the rule above,
recorded here first.
