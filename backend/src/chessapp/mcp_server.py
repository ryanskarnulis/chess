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
"""

from __future__ import annotations

import os
from typing import Any

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.tools.base import Tool as MCPTool
from mcp.server.fastmcp.utilities.func_metadata import ArgModelBase, FuncMetadata
from pydantic import ConfigDict

from chessapp.engine import EnginePlayer
from chessapp.game import GameSession
from chessapp.tools import ToolContext, ToolRegistry, build_registry

INSTRUCTIONS = (
    "Chess: a local, agent-first chess app. These tools are the exact same "
    "boundary the in-app agent plays through — moves, board state, "
    "Stockfish analysis, saves, and settings. The engine, not the caller, "
    "decides legality; illegal moves and domain failures come back as "
    '{"ok": false, "error": ...} rather than a protocol error.'
)


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


def _mcp_tool(
    name: str, description: str, parameters: dict[str, Any], registry: ToolRegistry
) -> MCPTool:
    """One FastMCP tool: registry schema verbatim, body is `registry.dispatch`.

    The call always goes through `dispatch`, never the registry handler
    directly — `dispatch` is what turns bad args/domain errors into
    `{"ok": False, ...}` instead of an exception.
    """

    def _call(**kwargs: Any) -> dict[str, Any]:
        return registry.dispatch(name, kwargs)

    return MCPTool(
        fn=_call,
        name=name,
        description=description,
        parameters=parameters,
        fn_metadata=_PASSTHROUGH_METADATA,
        is_async=False,
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
