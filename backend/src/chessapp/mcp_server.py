"""stdio MCP server: `ToolRegistry` over the MCP protocol.

Run with `python -m chessapp.mcp_server` from `backend/` (the repo's
`.mcp.json` does exactly that, giving Claude Code the same tools the in-app
agent uses). The tools live in `chessapp.tools`; this module is only
transport wiring.

Unlike PCC's server (which hands each registry function straight to
FastMCP's `add_tool` and lets it re-derive the schema from the signature),
chess's registry already carries the schema it wants advertised —
`Tool.parameters`, including the hand-written `oneOf` on `set_difficulty`
that no plain signature can express (see `tools._derive_schema`). Re-deriving
from a wrapper function's signature would produce a *different*, looser
schema, so each MCP tool is built directly as a
`mcp.server.fastmcp.tools.base.Tool` with `parameters` set to the registry's
schema verbatim, and a permissive pass-through `fn_metadata` that performs no
validation of its own — every call reaches `ToolRegistry.dispatch`
untouched, and `dispatch`'s own `jsonschema.validate` is the only place a bad
call can be rejected. That is also why a schema violation or domain failure
never becomes an MCP tool error (`isError`): `dispatch` already turns both
into `{"ok": False, "error": ...}`, and this module never raises on top of
that.

The one thing this surface adds is a way for a *human* to answer the
destructive gate (`tools._gate`; `docs/mcp-confirmation-surface.md`). The gate
refuses `new_game`, `resign` and `claim_draw` on a game the player has invested
in and arms the op for a yes — and on this surface, alone of the three, no yes
could arrive: the spoken turn and the board dialog both live in the HTTP
process. So a gated call that comes back armed goes on to ask the client's
human through MCP elicitation (form mode): a server-initiated question the
client presents to its user and answers accept, decline or cancel. The
client's *model* neither emits the request nor answers it, which keeps the
rule the gate exists for — nothing the model can emit opens it — and the
advertised tool list does not change: no confirm tool, no argument, no
description sentence, so the schema the eval floor is measured against is the
schema a client sees (`test_list_tools_matches_registry_definitions`, and the
byte snapshot in `tests/tool_definitions_emitted.json`).
"""

from __future__ import annotations

import os
from typing import Any

from mcp.server.fastmcp import Context, FastMCP
from mcp.server.fastmcp.tools.base import Tool as MCPTool
from mcp.server.fastmcp.utilities.func_metadata import ArgModelBase, FuncMetadata
from mcp.shared.exceptions import McpError
from mcp.types import ClientCapabilities
from pydantic import ConfigDict

from chessapp.brain import RETRY_NEVER
from chessapp.engine import EnginePlayer
from chessapp.game import GameSession
from chessapp.tools import (
    CONFIRM_QUESTIONS,
    PendingOp,
    ToolContext,
    ToolRegistry,
    build_registry,
    confirm_pending,
)

INSTRUCTIONS = (
    "Chess: a local, agent-first chess app. These tools are the exact same "
    "boundary the in-app agent plays through — moves, board state, "
    "Stockfish analysis, saves, and settings. The engine, not the caller, "
    "decides legality; illegal moves and domain failures come back as "
    '{"ok": false, "error": ...} rather than a protocol error.'
)

# The keyword FastMCP injects the request's `Context` under (`Tool.context_kwarg`).
# Popped before dispatch, so the registry's `jsonschema.validate` sees exactly
# the client's arguments; named so that no tool argument can collide with it.
_CONTEXT_KWARG = "_mcp_context"

# The form the client puts to its human: one boolean, and only `true` is a yes.
# Hand-written JSON rather than a pydantic model, so an accept that carries no
# field, or the wrong kind, reads as "not a yes" instead of raising a validation
# error — the answer is a human's, and the only thing that runs the op is that
# human's explicit true.
_CONFIRM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "confirm": {
            "type": "boolean",
            "title": "Confirm",
            "description": "Yes, do it",
            "default": False,
        }
    },
    "required": ["confirm"],
}


class _PassthroughArgs(ArgModelBase):
    """An arg model that accepts anything and changes nothing.

    Real validation belongs to `ToolRegistry.dispatch`'s own
    `jsonschema.validate` against the tool's real schema; this model exists
    only because FastMCP's `Tool.run` always calls through an `arg_model`.
    `extra="allow"` means no key is ever rejected here, and overriding
    `model_dump_one_level` (which by default reports only *declared*
    fields — none, in this model) hands every key straight back so nothing
    given to a tool call is lost on the way to `dispatch`.
    """

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    def model_dump_one_level(self) -> dict[str, Any]:
        return self.model_dump()


_PASSTHROUGH_METADATA = FuncMetadata(arg_model=_PassthroughArgs)


def _declares_form_elicitation(capabilities: ClientCapabilities | None) -> bool:
    """Whether the client said, at the handshake, that it can put a form to its
    human. Never assumed: a client that cannot ask gets the fallback in
    `_confirm` rather than a question nothing will answer.

    Spec 2025-11-25 splits the capability into `form` and `url` modes; a client
    on the 2025-06-18 shape declares a bare `elicitation: {}` and means form,
    the only mode there was. A client declaring `url` alone cannot show a form
    and is treated as unable to confirm.
    """
    if capabilities is None or capabilities.elicitation is None:
        return False
    elicitation = capabilities.elicitation
    return elicitation.form is not None or elicitation.url is None


def _client_capabilities(mcp: Context | None) -> ClientCapabilities | None:
    if mcp is None:
        return None
    params = mcp.session.client_params
    return None if params is None else params.capabilities


def _declined(refusal: dict[str, Any], armed: PendingOp, reason: str) -> dict[str, Any]:
    """The gate's refusal, marked as answered with a no. Nothing ran."""
    return {
        **refusal,
        "error": (
            f"{armed.name} did not run: {reason}. Nothing ran and nothing is armed."
        ),
        "declined": True,
    }


async def _confirm(
    mcp: Context | None,
    registry: ToolRegistry,
    ctx: ToolContext,
    armed: PendingOp,
    refusal: dict[str, Any],
) -> dict[str, Any]:
    """Put the gate's question to the client's human; run the op on their yes.

    The sequence of `docs/mcp-confirmation-surface.md` ("The design"). The lock
    was released with the dispatch and stays released across the question — a
    human waits on this step, and a concurrent call must be answered while the
    question is open. The question is the app's own (`CONFIRM_QUESTIONS`), the
    same words the board dialog shows and the free-text reader judges against,
    never a paraphrase. On a yes the lock is taken again around
    `confirm_pending` — the existing sole key, unchanged — which re-reads the op
    through `live_pending`, so a board that moved while the question was open
    runs nothing (`stale: true`); so does a question that is no longer the one
    standing, because a later gated call asked another (MCP is windowless: the
    newest question is the one a yes answers, as on the buttons, and the
    earlier yes must not run the later op). Every other outcome — a no, a
    dismissed form, a form accepted with the box unticked, a client that
    errored on the ask or cannot ask at all — runs nothing and leaves nothing
    armed: an op nothing can confirm must not sit armed for a later call to
    stumble on, which was the untruth in the gate's "runs when they say yes" on
    this surface.
    """
    try:
        if not _declares_form_elicitation(_client_capabilities(mcp)):
            return {
                **refusal,
                "error": (
                    f"confirmation required: {armed.name} would end the current "
                    "game, and this client cannot ask the player to confirm, so "
                    "nothing was armed and nothing ran."
                ),
                "confirmation_unavailable": True,
            }
        assert mcp is not None  # a client with capabilities came through a request
        try:
            answer = await mcp.session.elicit_form(
                CONFIRM_QUESTIONS[armed.name],
                _CONFIRM_SCHEMA,
                related_request_id=mcp.request_context.request_id,
            )
        except McpError as exc:
            return _declined(
                refusal,
                armed,
                f"the client could not ask the player ({exc.error.message})",
            )
        content = answer.content or {}
        if answer.action != "accept" or content.get("confirm") is not True:
            return _declined(refusal, armed, "the player did not confirm")
        with ctx.mutation_lock:
            # Identity, not name: the yes answers *this* question. A later call
            # that armed its own question replaced this one, and a board that
            # moved is caught one layer down, in `live_pending`.
            confirmed = confirm_pending(registry, ctx) if ctx.pending is armed else None
        if confirmed is None:
            return registry.refusal(
                f"{armed.name} did not run: the question is no longer the one "
                "standing — the board moved while the player was being asked, "
                "or a later call asked a different question. Nothing is armed "
                "for it; if the player still wants it, they can ask again.",
                RETRY_NEVER,
                stale=True,
            )
        _name, result = confirmed
        if result.get("ok") is True:
            return {**result, "confirmed": True}
        return result
    finally:
        # Settled for good, whichever way it went — but only *this* question:
        # a newer one another call armed while this one was open is theirs.
        if ctx.pending is armed:
            ctx.pending = None


def _mcp_tool(
    name: str,
    description: str,
    parameters: dict[str, Any],
    registry: ToolRegistry,
    ctx: ToolContext,
) -> MCPTool:
    """One FastMCP tool: registry schema verbatim, body is `registry.dispatch`.

    The call always goes through `dispatch`, never the registry handler
    directly — `dispatch` is what turns bad args/domain errors into
    `{"ok": False, ...}` instead of an exception.

    …and through `ctx.mutation_lock`, which is this surface's whole share of the
    board-version work (audit item 7). MCP gets **no `version` parameter**: its
    tools are advertised from the very schema objects the brain is offered, and
    that schema shape is frozen by the eval floor on gemma-4-12b (TODO.md's
    standing warning about the shelved minimization). What MCP has instead is
    the other two guarantees — `make_move` is the *atomic* exchange, so a call
    can never leave a turn half-played, and the lock makes each call indivisible
    against every other client on the session. Two concurrent MCP calls
    therefore take turns; what they cannot do is check one board and act on
    another. Held around reads too, which cost nothing and keeps the rule "one
    call, one hold" rather than a list of which tools mutate.

    The wrapper's second MCP-specific job is the confirmation (`_confirm`): a
    dispatch that comes back with a *newly* armed op is a gated call the gate
    refused, and on this surface the only thing that can answer it is the
    client's human, asked right here. The registry, the gate and
    `confirm_pending` do not change; this is a caller of `confirm_pending`,
    exactly as `/api/game/confirm` is.
    """

    async def _call(**kwargs: Any) -> dict[str, Any]:
        mcp: Context | None = kwargs.pop(_CONTEXT_KWARG, None)
        armed_before = ctx.pending
        with ctx.mutation_lock:
            result = registry.dispatch(name, kwargs)
        armed = ctx.pending
        if armed is None or armed is armed_before:
            return result
        return await _confirm(mcp, registry, ctx, armed, result)

    return MCPTool(
        fn=_call,
        name=name,
        description=description,
        parameters=parameters,
        fn_metadata=_PASSTHROUGH_METADATA,
        is_async=True,
        context_kwarg=_CONTEXT_KWARG,
    )


def build_mcp_server(ctx: ToolContext) -> FastMCP:
    """A FastMCP server exposing exactly `build_registry(ctx)`'s tools.

    Deliberately the registry's *atomic* default (`atomic_exchange=True`): the
    same tool boundary the app offers, sequenced differently. An MCP call has no
    pipeline behind it to run the app's observe/close beats, so `make_move` must
    finish the exchange itself — otherwise a call would leave this process's game
    parked mid-turn with a reply nobody is going to collect.
    """
    registry = build_registry(ctx)
    tools = [
        _mcp_tool(
            definition["function"]["name"],
            definition["function"]["description"],
            definition["function"]["parameters"],
            registry,
            ctx,
        )
        for definition in registry.definitions()
    ]
    return FastMCP("chess", instructions=INSTRUCTIONS, tools=tools)


def _engine_from_env() -> EnginePlayer | None:
    """Same discovery as `chessapp.app`'s: unset `CHESSAPP_STOCKFISH` means
    no engine, not a crash."""
    path = os.environ.get("CHESSAPP_STOCKFISH")
    return EnginePlayer(path=path) if path else None


def build_mcp_context() -> ToolContext:
    """A `ToolContext` for the MCP server, mirroring `build_app`'s assembly:
    Stockfish from the environment (optional), and — the same "never leave
    an engine unconfigured" invariant `build_app` upholds — the default
    difficulty tier applied to it immediately, since Stockfish's own default
    is full strength."""
    ctx = ToolContext(session=GameSession(), engine=_engine_from_env())
    if ctx.engine is not None:
        if ctx.settings.tier is not None:
            ctx.engine.set_tier(ctx.settings.tier)
        elif ctx.settings.skill_level is not None:
            ctx.engine.set_skill_level(ctx.settings.skill_level)
        elif ctx.settings.elo is not None:
            ctx.engine.set_elo(ctx.settings.elo)
    return ctx


def main() -> None:  # pragma: no cover - thin runtime shim
    build_mcp_server(build_mcp_context()).run()  # stdio transport


if __name__ == "__main__":  # pragma: no cover
    main()
