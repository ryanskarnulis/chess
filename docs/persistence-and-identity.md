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

Everything else lives in process memory: the live board, the pending
question, and the delegate conversations. A restart loses them. (#291 PR 2
and PR 3 change this; this table is updated as they land.)

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
a stale one is a 409 with the board untouched. It is optional by design
(audit item 7): the web UI sends it, but a caller that omits it acts on
whatever board is live when its request lands. A client that wants "act on
the board I saw, or not at all" must send it. A delegate message
(`MessageCreate.version`) may carry it too.

## `X-Agent-Actor` is a label

`agent_api.resolve_actor` stamps a recognised delegate actor into the audit
log and falls back to the loop's identity otherwise. It is **not**
authentication: the app trusts its network, and any access claim beyond
"someone on the home network" needs the external gateway verified first.

## The standalone MCP server is its own game

`chessapp.mcp_server` builds its own `ToolContext` (`build_mcp_context`):
its own board, its own pending question, no save directory — so no save,
resume or settings persistence — and nothing shared with the HTTP app's
state. They are deliberately not bridged; an MCP client playing chess is not
playing the game on the web board.
