# Persistence and operation identity

What this app keeps, where, and what a client can rely on when it acts on the
board. Written for #291 (astra audit 2026-09-16, F11), which found these
boundaries correct for one player at one screen but only *implied*. This page
makes them explicit, so an agent that delegates to the app, or a second
client, knows what it is getting.

## What persists

Under `CHESSAPP_SAVE_DIR` (the `/data/saves` volume in the container):

| File | Holds | Written |
|---|---|---|
| `settings.json` | difficulty, verbosity, voice | on every settings change, best-effort |
| `games/<name>.json` | a named save: the game plus the panel transcript | by `save_game`, atomically |
| `live.json` | the live game: board, panel transcript, `game_id`, board version | on every change, atomically, best-effort |
| `conversations.json` | every delegate thread: turns, soft deletes, id counters, idempotency keys | on every change, atomically, best-effort |

**Restart** (#291). `live.json` is written whenever the board version or the
panel transcript changes — from the mutation guard's exit and from the
state broadcast, both under the mutation lock, so it is one coherent
snapshot — and restored by `build_app` before anything reads the session
(`tools.restore_live_checkpoint`). Restoring replays every move through the
legality gate, so a tampered file cannot put up a board the rules never
allowed; a missing, unreadable or invalid one means a fresh board and a
warning, never a failed start. A checkpoint taken while the engine was
thinking is settled on restore, as a resumed save is. The file sits at the
save dir's root, outside `games/`, and nests the session under a key, so it
is never listed as a save nor swept up by the legacy-save migration.

What does **not** survive a restart:

- **The pending question.** It was asked in a conversation about a board on
  a screen; after a restart nothing is armed, and the player asks again. A
  "yes" can never meet a question the restarted app did not ask.
- **An in-flight turn.** A command the process died inside has whatever
  outcome the checkpoint recorded; nothing re-runs it.

## The gate: what asks before it runs

`tools.GATED_TOOLS` is every tool that can arm a confirmation. A refused call
is ordinary result data (`ok: false`, `retry: never`, "confirmation required:
…"); the model relays the question and stops, and only the player's yes —
read deterministically, or by the answer reader and then run through
`tools.confirm_pending` — opens the gate. The question is the app's own
(`tools.CONFIRM_QUESTIONS`), the same on every surface.

| Tool | Asks when | Budget | Ends a game? |
|---|---|---|---|
| `new_game` | a game is at stake | spends | resets one |
| `resign` | a game is at stake | spends | yes |
| `claim_draw` | a game is at stake (and a claim exists) | spends | yes |
| `resume_game` | a game is at stake (and the save loads) | spends | no |
| `save_game` | the name already exists, unless it is `autosave` or this command wrote it | never | no |

"A game is at stake" means the player has moved and the game is not over:
nothing on a fresh board, or on one holding only the engine's opening move,
is worth a question.

**Resume** (#291). Loading a save swaps out the game on the board, so it is
gated exactly like a reset. The save is checked first — a missing name or an
unreadable file is refused without arming anything, so a yes can never be
the answer to a load that would then fail. It spends the command's
destructive budget: "load scholars and start a new game" throws one board
away, not two.

**Overwrite** (#291). Saving under a name that exists asks before replacing
the file; the file is untouched until the yes. The stake is the *save*, not
the board, so this asks whatever state the game is in, and the refusal says
"would replace the existing save '<name>'" rather than "would end the
current game". Two exemptions, each because nothing the player owned is lost:

- `autosave`, the default slot an unnamed "save" writes over by design;
- a name the *same command* already wrote — "save as checkpoint, undo, save
  it again, then play d4" is one ask (audit 2026-09-05, finding 8). Tracked
  on the coordinator's command window, so a surface without one (the MCP
  server, one call per interaction) asks on the second save of a name.

A successful save reports `replaced: true|false`.

`DESTRUCTIVE_TOOLS` stays the three game-ending ops: it is what the honesty
guard certifies as "the game ended or restarted", and a confirmed resume or
save must never read as "game over".

## Who may answer

A question is stamped with the board it is about (`board_version`) and the
conversation it was asked in (`origin`): the panel, one delegate thread
(`delegate:<id>`), or the standalone MCP server (`mcp`). Only that origin
can answer it, and only while that board is still on screen (#281,
`ToolContext.live_pending`). A delegate therefore cannot confirm a question
it did not ask.

## Board versions are opt-in

Every mutating HTTP request may carry `version` (`api.VersionedRequest`), and
a stale one is a 409 with the board untouched. A restored checkpoint comes
back one version past the one it was taken at, so a number held across a
restart is stale rather than silently meaning a different position. It is optional by design
(audit item 7): the web UI sends it, but a caller that omits it acts on
whatever board is live when its request lands. A client that wants "act on
the board I saw, or not at all" must send it. A delegate message
(`MessageCreate.version`) may carry it too.

**Delegate threads** (#291). `agent_api.ConversationStore` rewrites
`conversations.json` after every change and reloads it at startup, so a
conductor's thread — and its replayed history — survives a restart like the
board does. Stored roles are validated on load (only `user` and
`assistant`), so a tampered file cannot inject a system turn into the
model's context; an unreadable file is an empty store and a warning.

## Retries: `Idempotency-Key`

`POST /api/agent/conversations/{id}/messages` takes an optional
`Idempotency-Key` header (1–128 characters), stored on the user turn and
scoped to the thread. A retry under the same key never runs the pipeline:

| The first attempt… | A retry gets |
|---|---|
| finished (its answer is stored) | 200, the stored exchange, byte for byte — after a timeout or a restart alike |
| never finished (502, a stale 409, a crash mid-run) | 409 "did not complete": whatever it did is on the board, so read the state and send a **new** key |
| used the key for different words | 422 |

With a key and `game_id` (below) a conductor gets the acceptance #291 asked
for: a retry recovers a known outcome without acting twice, and never lands
on a game it was not playing. Without a key, a retry is a new exchange, as
before.

## Game identity

`state.game_id` names the game on the board (`GameSession.game_id`, 32 hex
characters). It changes on a new game and on a resume — resuming one save
twice gives two games — and survives a move, a takeback and a restart.
Every mutating request, and a delegate message, may carry `game_id` as a
precondition beside `version`: a mismatch is the same 409 (`stale: true`,
with the current `game_id` and state), so a delegate that sends it can never
silently act on a game other than the one it was playing. Trace records
carry it too, which is what ties turns to one game across restarts.

## `X-Agent-Actor` is a label

`agent_api.resolve_actor` stamps a recognised delegate actor into the audit
log and falls back to the loop's identity otherwise. It is **not**
authentication: any caller that reaches the port can send the header, the
app trusts its network, and any access claim beyond "someone on the home
network" needs the external gateway verified first — that it authenticates
callers and strips a forged header on the way in.

## The standalone MCP server is its own game

`chessapp.mcp_server` builds its own `ToolContext` (`build_mcp_context`):
its own board, its own pending question, no save directory — so no save,
resume or settings persistence — and nothing shared with the HTTP app's
state. They are deliberately not bridged; an MCP client playing chess is not
playing the game on the web board.
