"""stdio MCP server, driven through the real protocol layer.

``mcp_client`` speaks JSON-RPC to the FastMCP server over an in-memory
transport — the same request/validation/serialization path a real client
(Claude Code over stdio) exercises, minus the pipe. Mirrors PCC's
``test_mcp_tools.py``.

The contract under test: the MCP server's tools are the *same* tools the
in-app agent loop calls (schemas from ``ToolRegistry.definitions()``, calls
through ``ToolRegistry.dispatch()``), and the server can be assembled with or
without a live Stockfish engine.
"""

from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import pytest
from mcp import ClientSession
from mcp.shared.memory import create_connected_server_and_client_session

from chessapp.game import GameSession
from chessapp.mcp_server import build_mcp_server
from chessapp.tools import ToolContext, build_registry
from fakes import FakeEngine

# --- schema-equivalence normalization (same normalizations as slice 3's
# tests/test_tool_registry_schema.py golden test) -------------------------

_NULL = {"type": "null"}


def _normalize(node: Any) -> Any:
    if isinstance(node, dict):
        cleaned = {
            k: _normalize(v) for k, v in node.items() if k not in ("title", "default")
        }
        any_of = cleaned.get("anyOf")
        if isinstance(any_of, list) and len(any_of) == 2 and _NULL in any_of:
            other = next(branch for branch in any_of if branch != _NULL)
            merged = {k: v for k, v in cleaned.items() if k != "anyOf"}
            merged.update(other)
            return merged
        return cleaned
    if isinstance(node, list):
        return [_normalize(x) for x in node]
    return node


def _canonical_description(description: str) -> str:
    return re.sub(r"\s+", " ", description).strip()


@asynccontextmanager
async def mcp_client(ctx: ToolContext) -> AsyncIterator[ClientSession]:
    """In-memory client session against a server built for `ctx`.

    Entered inside the test body, not a fixture: anyio cancel scopes must be
    exited in the task that entered them, and a pytest fixture's teardown
    runs in a different task.
    """
    mcp = build_mcp_server(ctx)
    async with create_connected_server_and_client_session(mcp._mcp_server) as client:
        yield client


async def _call(client: ClientSession, tool: str, args: dict[str, Any]) -> Any:
    result = await client.call_tool(tool, args)
    assert not result.isError, f"{tool} unexpectedly errored: {result.content}"
    assert result.content and result.content[0].type == "text"
    return json.loads(result.content[0].text)


async def test_list_tools_matches_registry_definitions():
    """Every registry tool is exposed, with the same name, description, and
    input schema (modulo the golden test's harmless normalizations)."""
    ctx = ToolContext(session=GameSession())
    registry_definitions = build_registry(ctx).definitions()
    assert len(registry_definitions) == 20

    async with mcp_client(ctx) as client:
        listed = await client.list_tools()

    mcp_by_name = {t.name: t for t in listed.tools}
    assert mcp_by_name.keys() == {d["function"]["name"] for d in registry_definitions}

    for definition in registry_definitions:
        fn = definition["function"]
        tool = mcp_by_name[fn["name"]]
        assert _canonical_description(tool.description) == _canonical_description(
            fn["description"]
        )
        assert _normalize(tool.inputSchema) == _normalize(fn["parameters"])


async def test_make_move_happy_path_returns_real_game_data():
    """A legal move played through MCP updates the real (server-owned) game
    and gets the engine's reply, same as the in-app tool boundary."""
    engine = FakeEngine(reply_uci="e7e5")
    ctx = ToolContext(session=GameSession(), engine=engine)

    async with mcp_client(ctx) as client:
        board = await _call(client, "get_board_state", {})
        assert board["fen"].startswith("rnbqkbnr/pppppppp")

        result = await _call(client, "make_move", {"move": "e4"})
        assert result["legal"] is True
        assert result["san"] == "e4"
        assert result["engine_move"]["san"] == "e5"
        assert result["engine_move"]["uci"] == "e7e5"

    # The move landed on the context's own session (the server's game).
    assert ctx.session.move_history() == ["e4", "e5"]


async def test_an_odd_takeback_comes_back_settled():
    """An MCP call has no pipeline behind it, which is why `make_move` here runs
    the whole exchange — and the same reason a takeback that leaves the engine
    to move must not park the game there. Settling is not an exchange (nobody
    moved for the player), so it belongs to the coordinator on this surface
    exactly as it does in the app: one ply off 1.e4 c5 comes back as a board the
    client can go on playing."""
    ctx = ToolContext(session=GameSession(), engine=FakeEngine(reply_uci="e7e5"))
    for san in ("e4", "c5"):
        assert ctx.session.submit_move(san).legal

    async with mcp_client(ctx) as client:
        result = await _call(client, "undo", {"plies": 1})

    assert result["undone"] == ["c5"]
    assert result["engine_move"]["san"] == "e5"
    assert result["turn"] == "white"
    assert ctx.session.move_history() == ["e4", "e5"]


async def test_invalid_args_come_back_as_ok_false_not_a_tool_error():
    """A schema violation (set_difficulty's oneOf) surfaces as a normal
    `{"ok": False}` result, not an MCP tool error."""
    ctx = ToolContext(session=GameSession())

    async with mcp_client(ctx) as client:
        result = await client.call_tool("set_difficulty", {})
        assert not result.isError
        payload = json.loads(result.content[0].text)
        assert payload["ok"] is False


async def test_domain_failure_comes_back_as_ok_false_not_a_tool_error(tmp_path):
    """A domain failure (illegal save name / missing save) never crashes the
    server — same `{"ok": False, "error": ...}` envelope as the in-app path."""
    ctx = ToolContext(session=GameSession(), save_dir=tmp_path)

    async with mcp_client(ctx) as client:
        result = await client.call_tool("resume_game", {"name": "does-not-exist"})
        assert not result.isError
        payload = json.loads(result.content[0].text)
        assert payload["ok"] is False
        assert "does-not-exist" in payload["error"]


async def test_analysis_tool_reports_engine_unavailable_without_stockfish():
    """Engine-less context: analysis tools fail gracefully, not crash."""
    ctx = ToolContext(session=GameSession(), engine=None)

    async with mcp_client(ctx) as client:
        result = await client.call_tool("evaluate_position", {})
        assert not result.isError
        payload = json.loads(result.content[0].text)
        assert payload["ok"] is False
        assert "engine" in payload["error"].lower()


def test_build_mcp_context_is_engine_less_when_stockfish_discovery_finds_nothing(
    monkeypatch: pytest.MonkeyPatch,
):
    """`build_mcp_context` mirrors `build_app`'s engine discovery: no
    Stockfish configured/found means an engine-less context, not a crash."""
    import chessapp.mcp_server as mcp_server_module

    monkeypatch.setattr(mcp_server_module, "_engine_from_env", lambda: None)

    ctx = mcp_server_module.build_mcp_context()

    assert ctx.engine is None
