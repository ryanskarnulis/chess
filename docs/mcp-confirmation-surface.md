# The MCP confirmation surface

Design note, 2026-09-05 — audit finding 3 (`docs/agent-audit-2026-09-05.md`).
Filed without code (#271) and agreed the same day; **implemented in #272**
against the acceptance criteria at the end, in `mcp_server._mcp_tool`'s call
wrapper and nowhere else. One name moved in the implementation:
`_CONFIRM_QUESTIONS` left `api.py` for `tools.CONFIRM_QUESTIONS`, so the board
dialog, the free-text reader and the MCP server read one table — where this
note says `_CONFIRM_QUESTIONS[op]`, read `CONFIRM_QUESTIONS[op]`. One
concurrency rule the implementation added beyond the text below: a yes runs
only if the armed op is still *this* call's (identity, not name), because with
the lock released across the human's wait a later gated call can arm its own
question, and the newest question is the one a yes answers. On the client:
Claude Code renders MCP elicitation forms with Accept/Decline (its changelog
fixes their fullscreen layout in 2.1.239, and the CLI here is 2.1.261), so the
capability this note leaned on is real, and the end-to-end run with our
question in that form happened on 2026-09-05 — both buttons, recorded at the
end of this note ("Verified live").

## The finding

`python -m chessapp.mcp_server` (the repo's `.mcp.json`, Claude Code as the
client) is a stdio MCP server over the same `ToolRegistry` the in-app agent
plays through, on a game of its own (`build_mcp_context`: its own
`ToolContext`, no save directory, Stockfish from the environment). The
destructive gate (`tools._gate`) is the same too: once the player has invested
a move, `new_game`, `resign` and `claim_draw` come back `confirmation
required`, arm `ctx.pending`, and wait for a yes.

On this surface no yes can arrive. The two answering surfaces the app has —
the spoken turn (`api._command_turn`: a bare yes, or a free-text reply the
brain reads against the pipeline's question) and the board dialog
(`/api/game/confirm`) — both live in the HTTP process, and the MCP process has
neither. The advertised tool list carries no confirmation operation, by
design (below), so a client that asks again is refused again: the human says
"yes" to Claude Code, Claude Code calls `new_game` again, and the gate refuses
again. Reproduced in the audit's appendix A (`MCP no confirm`). The three
gated tools are unreachable on a game in progress, and the refusal's text —
"it runs when they say yes" — is not true here.

## What must stay true

- **Code owns safety; nothing the model can emit opens the gate.**
  `confirm_pending` is the sole key (`ToolContext._confirming` is not
  reachable from any tool argument), and it is turned only by a surface a
  human operates. That rule is the reason the gate exists: the prompt asked
  the model to confirm first and it complied about half the time.
- **The advertised schema is frozen.** MCP advertises the registry's exact
  definitions (`test_list_tools_matches_registry_definitions`, and since #268
  the byte-for-byte snapshot). A confirmation must not add a tool, an
  argument, or a description sentence — the eval floor on gemma-4-12b is
  measured against this schema, and the standing prohibition forbids touching
  its keys.
- **The MCP module is transport wiring.** Game truth, the gate and the budget
  stay in `tools.py` and `coordinator.py`; whatever confirms on MCP is a
  caller of `confirm_pending`, exactly as `/api/game/confirm` is.
- **A confirmation is an answer about a position.** `PendingOp` carries the
  board version it was armed against and `confirm_pending` reads it through
  `live_pending`, so a yes cannot run an op about a board that has since moved.
  Whatever the surface, the answer is applied under `ctx.mutation_lock` with
  that check, as the web dialog's is.

## Options considered

1. **A `confirm` tool.** Rejected. A tool is model-callable by definition; the
   client's model would call it straight after reading the refusal, which is
   the same failure the gate was built against, now with a second tool. It
   would also change the advertised schema.
2. **"Call it again to confirm."** Rejected, for the same reason and more
   sharply: the audit names it ("repeated gated tools must not become its
   implementation"). The model already calls again about half the time when
   told not to; a convention that turns the repeat into consent turns the
   model's most common mistake into a game thrown away.
3. **Disable the gate for MCP** (an environment flag: "the operator is a human
   at a terminal"). Rejected. The MCP client is an agent, not the human; the
   human's intent is what the gate guards, and the agent's tool call is
   exactly the thing that is not evidence of it.
4. **Make MCP a client of the running app's game** (an HTTP client of
   `chessapp`, the confirmation happening in the browser's dialog). A real
   design, and a different one: it changes what MCP *is* (a second controller
   of the live board rather than a standalone game), needs the app running,
   and drags the board-version handshake into MCP. Not what finding 3 asks
   for; recorded as the shape a future "drive the live game from Claude Code"
   feature would take, where URL-mode elicitation (below) is the natural
   confirmation.
5. **MCP elicitation, form mode. Chosen.** The protocol (spec 2025-06-18)
   gives a server a way to ask the *human* something mid-call: an
   `elicitation/create` request the *client* answers by presenting the
   server's message to the user and returning `accept` (with form content),
   `decline` or `cancel`. The model produces neither the request nor the
   answer — the request is the server's, the answer is the client UI's. The
   SDK we pin (`mcp` 1.28, `<2`) implements it: `Context.elicit(message,
   schema)` returns an `ElicitationResult` whose `.action` is one of the three,
   and a client declares support in its capabilities
   (`ClientCapabilities.elicitation.form`), which the server reads with
   `session.check_client_capability(...)`. Whether the client can ask is
   never assumed: the handshake says, and a client that cannot gets the
   fallback below. Claude Code — the client `.mcp.json` names — declares form
   elicitation and presents it to the user as a prompt in current releases;
   the implementation PR confirms that on its first connection before
   anything is relied on.

## The design

**Where.** In `mcp_server._mcp_tool`'s call wrapper, and nowhere else. The
registry, the gate and `confirm_pending` do not change. The wrapper already
owns the one MCP-specific thing (the lock); the confirmation is the second.

**Sequence, for a call of a gated tool.**

1. Dispatch under `ctx.mutation_lock`, exactly as today. On a fresh or
   finished board the gate stands aside and the op runs — nothing new. On a
   game worth guarding the gate refuses and arms `ctx.pending` for *this* op
   (MCP is windowless, so each call arms its own question — the newest
   question is the one that stands, as on the buttons).
2. **Release the lock**, then, if `ctx.pending` names this call's op and the
   client declared form-mode elicitation: elicit. The message is the app's
   own question for the op — `_CONFIRM_QUESTIONS[op]`, the same words the
   board dialog and the free-text reader use — never a model paraphrase. The
   form schema has one boolean field, `confirm`, described as "Yes, do it";
   only `action == "accept"` with `confirm` true counts as a yes. A human
   waits on this step; the lock must not be held across a human-speed wait,
   and a concurrent read (`get_board_state`) must answer while the question
   is open.
3. **Yes:** re-acquire the lock and run `confirm_pending(registry, ctx)` — the
   existing sole key, unchanged, which re-reads the op through `live_pending`
   and so runs nothing about a board that moved while the question was open
   (a board that moved answers with the refusal and `stale: true`). The tool
   result is the op's own result (`ok: true`, the fresh board / the outcome),
   plus `confirmed: true` so the client can tell a confirmed run from an
   ungated one.
4. **Decline or cancel (or the client's timeout):** `ctx.pending = None`;
   the result is the gate's refusal with `declined: true`. Nothing ran and
   nothing is left armed for a later call to stumble on.
5. **No elicitation capability:** the refusal returns as today with
   `confirmation_unavailable: true` and its text corrected — "this client
   cannot confirm; nothing was armed" — and `ctx.pending` is cleared. An op
   that nothing can ever confirm must not sit armed: it was the lie in the
   current text, and on this surface a stale armed op only misleads the next
   caller.

**What the model sees.** The client's model sees a tool result, as now. On a
yes it is the op's success; on a no it is a refusal marked declined; on a
client that cannot ask, a refusal that says so. In none of the three does the
model learn a way to confirm, because there is none: the yes travelled from
the human to the client UI to the server and never through the model's
context.

**The trust boundary, stated.** The server trusts the client's declared
capability the way a web server trusts the browser to have shown its dialog.
A client that declares form elicitation and answers it without a human is
misrepresenting itself to the protocol; that is outside anything this process
can verify, and it is no worse than today's "unreachable", while a conforming
client — Claude Code — gets the human's answer. The SDK's own docstring notes
an agentic client "might decide how to handle the elicitation"; the design
does not pretend otherwise, and this is the paragraph to reread before
trusting a new client.

**URL mode, later.** `Context.elicit_url` directs the human to a URL for an
out-of-band interaction the model must not see. It is the right shape for
option 4 (the live game's own confirm dialog) and for nothing in the
standalone server today; noted so the two are not confused.

## Acceptance criteria for the implementation PR

All at the tool boundary, no GPU (no model is in this loop, and the schema
does not change — the byte snapshot and the MCP schema-equivalence test stay
green as they are):

- Through the in-memory client with an `elicitation_callback` (the SDK's
  `create_connected_server_and_client_session(..., elicitation_callback=)`):
  a gated `new_game`, `resign` and `claim_draw` on a game in progress (for
  the claim, a repetition the player themselves played into — the gate stands
  aside on a FEN-rooted position with no player plies, `tools._player_has_moved`,
  and `claim_draw` checks claimability before the gate, so an unclaimable
  position refuses without arming and must not elicit) each
  elicit exactly `_CONFIRM_QUESTIONS[op]`; `accept` with `confirm: true` runs
  the op (board reset / game over) with `confirmed: true`; `accept` with
  `confirm: false`, `decline` and `cancel` each run nothing and leave
  `ctx.pending` clear with `declined: true`.
- A client without the elicitation capability (the callback left unset, so
  the session does not declare it) gets the refusal with
  `confirmation_unavailable: true`, nothing runs, `ctx.pending` is clear.
- The lock is not held across the question: an elicitation callback that
  itself calls `get_board_state` before answering gets its answer.
- A board that moved while the question was open (the callback plays a move
  through the client before accepting) runs nothing and says `stale: true`.
- A fresh or finished board still runs the op with no elicitation at all.
- `list_tools` is byte-identical to before: no tool, argument or description
  changed (`test_list_tools_matches_registry_definitions`, plus
  `test_emitted_schemas_match_the_snapshot_byte_for_byte`).
- `docs/turn-coordinator.md`'s "Board controls" gains MCP as the third
  answering surface; `TODO.md`'s "Later — MCP confirmation surface" moves to
  DONE with the PR.

## Open questions this note does not decide

- Whether MCP should ever share the live app's game (option 4). Separate
  design; nothing here forecloses it.
- Whether the delegate wire (`/api/agent/...`) wants the same elicitation-style
  round trip for a client that is not the web panel. Today it inherits the
  spoken-turn answer path, which is the right one for a conversation.

## Verified live (2026-09-05)

The check `TODO.md` carried after #272, run once against the real client: a
fresh Claude Code 2.1.261 session in an empty folder with only this server
attached (`--mcp-config` + `--strict-mcp-config`, the server's stdio tee'd
both ways for the record, `CHESSAPP_STOCKFISH` set so the engine replied).
The wire, verbatim except for whitespace and the client's `_meta` request ids.

The handshake declared the bare `elicitation: {}` shape (protocol
`2025-11-25`, no `form` or `url` key), which `_declares_form_elicitation`
reads as form mode:

```json
{"roots":{"listChanged":true},"elicitation":{}}
```

`make_move {"move":"e4"}` ran and came back with Stockfish's reply, so the
gate had a player's investment to guard. `new_game {}` then produced one
server → client request — `CONFIRM_QUESTIONS["new_game"]` verbatim, one
boolean:

```json
{"method":"elicitation/create","params":{"mode":"form",
 "message":"That ends the game in progress. Start a new one?",
 "requestedSchema":{"type":"object","properties":{"confirm":{"type":"boolean",
 "title":"Confirm","description":"Yes, do it","default":false}},
 "required":["confirm"]}}}
```

Claude Code rendered it as *MCP server "chess" requests your input*, the
question, a `Confirm` checkbox captioned *Yes, do it*, and Accept / Decline,
with the model's turn parked at "Calling chess" until the human answered.
Ticking the box and choosing Accept sent the yes, and the tool result was the
op's own: the starting FEN with `confirmed: true`, which the model narrated as
a new game. The yes travelled human → client UI → server; the model saw only
the result.

```json
{"action":"accept","content":{"confirm":true}}
{"ok":true,"engine_move":null,"fen":"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1","turn":"white","confirmed":true}
```

After another `e4`, a second `new_game {}` asked again. Choosing Decline sent
a bare decline; the tool result was the refusal marked `declined: true`,
`retry: never`, and `get_board_state` then showed the game untouched. The
model read the refusal and asked the human in chat — the design working: a
chat "yes" can only make it call `new_game` again and be asked again in the
form.

```json
{"action":"decline"}
{"ok":false,"error":"new_game did not run: the player did not confirm. Nothing ran and nothing is armed.","retry":"never","board_version":5,"declined":true}
```

Not exercised live, covered at the tool boundary by `tests/test_mcp_server.py`:
cancel (Esc), an accept with the box unticked, a board that moved while the
form was open, a superseding question, and a client without the capability.
