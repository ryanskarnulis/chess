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
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any

import anyio
import pytest
from mcp import ClientSession
from mcp.shared.memory import create_connected_server_and_client_session
from mcp.types import (
    INVALID_REQUEST,
    ClientCapabilities,
    ElicitationCapability,
    ElicitResult,
    ErrorData,
    FormElicitationCapability,
    UrlElicitationCapability,
)

from chessapp.game import GameSession
from chessapp.mcp_server import _declares_form_elicitation, build_mcp_server
from chessapp.tools import CONFIRM_QUESTIONS, ToolContext, build_registry
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
async def mcp_client(
    ctx: ToolContext, elicitation_callback: Callable[..., Any] | None = None
) -> AsyncIterator[ClientSession]:
    """In-memory client session against a server built for `ctx`.

    Entered inside the test body, not a fixture: anyio cancel scopes must be
    exited in the task that entered them, and a pytest fixture's teardown
    runs in a different task.

    `elicitation_callback` is the client's human. With one, the session
    declares form elicitation at the handshake and answers the server's
    questions through it; without one it declares nothing, which is exactly
    how a client that cannot ask looks to the server.
    """
    mcp = build_mcp_server(ctx)
    async with create_connected_server_and_client_session(
        mcp._mcp_server, elicitation_callback=elicitation_callback
    ) as client:
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


# --- the confirmation surface (docs/mcp-confirmation-surface.md) ---------------
#
# The destructive gate refuses `new_game`, `resign` and `claim_draw` on a game
# the player has invested in and arms the op for a yes. On the standalone MCP
# server nothing could answer it, so the call wrapper asks the client's human
# through form-mode elicitation and calls `confirm_pending` on their yes. All at
# the tool boundary, no GPU: no model is in this loop.


def _invested(ctx: ToolContext) -> ToolContext:
    """A game worth guarding: the player has moved, and the engine replied."""
    for san in ("e4", "e5"):
        assert ctx.session.submit_move(san).legal
    return ctx


def _drawish(ctx: ToolContext) -> ToolContext:
    """A game the player repeated into a claimable threefold draw — their own
    plies, so the gate guards it, and a draw the rules let them claim."""
    for _ in range(2):
        for san in ("Nf3", "Nf6", "Ng1", "Ng8"):
            assert ctx.session.submit_move(san).legal
    assert ctx.session.claimable_draws() == ("threefold_repetition",)
    return ctx


class _Human:
    """The client's human, as an elicitation callback: records what it was
    asked and answers as scripted. The client's *model* is nowhere in this."""

    def __init__(self, action: str = "accept", content: dict[str, Any] | None = None):
        self.action = action
        self.content = content
        self.asked: list[Any] = []

    async def __call__(self, context: Any, params: Any) -> ElicitResult:
        self.asked.append(params)
        return ElicitResult(action=self.action, content=self.content)


@pytest.mark.parametrize(
    ("op", "prepare", "ended"),
    [
        pytest.param(
            "new_game",
            _invested,
            lambda ctx, result: (
                ctx.session.move_history() == []
                and result["fen"].startswith("rnbqkbnr/pppppppp")
            ),
            id="new_game",
        ),
        pytest.param(
            "resign",
            _invested,
            lambda ctx, result: (
                ctx.session.is_game_over() and result["outcome"]["result"] == "0-1"
            ),
            id="resign",
        ),
        pytest.param(
            "claim_draw",
            _drawish,
            lambda ctx, result: (
                ctx.session.is_game_over()
                and result["outcome"]["termination"] == "threefold_repetition"
            ),
            id="claim_draw",
        ),
    ],
)
async def test_a_gated_call_asks_the_human_the_apps_own_question_and_a_yes_runs_it(
    op, prepare, ended
):
    """Each gated tool on a game in progress elicits exactly the question the
    board dialog would show — the app's words, never a paraphrase — as one
    boolean form; `accept` with the box ticked runs the op through
    `confirm_pending`, and the result is the op's own plus `confirmed`."""
    ctx = prepare(ToolContext(session=GameSession(), engine=FakeEngine()))
    human = _Human("accept", {"confirm": True})

    async with mcp_client(ctx, elicitation_callback=human) as client:
        result = await _call(client, op, {})

    [asked] = human.asked
    assert asked.mode == "form"
    assert asked.message == CONFIRM_QUESTIONS[op]
    assert set(asked.requestedSchema["properties"]) == {"confirm"}
    assert asked.requestedSchema["properties"]["confirm"]["type"] == "boolean"
    assert result["ok"] is True
    assert result["confirmed"] is True
    assert ended(ctx, result)
    assert ctx.pending is None


@pytest.mark.parametrize(
    ("action", "content"),
    [
        pytest.param("accept", {"confirm": False}, id="accepted-unticked"),
        pytest.param("accept", {}, id="accepted-empty"),
        pytest.param("decline", None, id="declined"),
        pytest.param("cancel", None, id="cancelled"),
    ],
)
async def test_anything_but_a_yes_runs_nothing_and_leaves_nothing_armed(
    action, content
):
    """Only `accept` with `confirm` true is a yes. A declined or dismissed
    form, or one accepted with the box unticked or missing, runs nothing; the
    result is the gate's refusal marked `declined`, and nothing stays armed for
    a later call to stumble on."""
    ctx = _invested(ToolContext(session=GameSession(), engine=FakeEngine()))
    human = _Human(action, content)

    async with mcp_client(ctx, elicitation_callback=human) as client:
        result = await _call(client, "new_game", {})

    assert len(human.asked) == 1
    assert result["ok"] is False
    assert result["declined"] is True
    assert "confirmed" not in result
    assert ctx.session.move_history() == ["e4", "e5"]
    assert ctx.pending is None


async def test_a_client_that_fails_to_ask_is_read_as_no():
    """A client that declared the capability but errors on the ask never got a
    human's yes: nothing runs, nothing stays armed, and the result says what
    the client answered."""
    ctx = _invested(ToolContext(session=GameSession(), engine=FakeEngine()))

    async def broken(context: Any, params: Any) -> ErrorData:
        return ErrorData(code=INVALID_REQUEST, message="no form UI here")

    async with mcp_client(ctx, elicitation_callback=broken) as client:
        result = await _call(client, "new_game", {})

    assert result["ok"] is False
    assert result["declined"] is True
    assert "no form UI here" in result["error"]
    assert ctx.session.move_history() == ["e4", "e5"]
    assert ctx.pending is None


async def test_a_client_that_cannot_ask_is_told_so_and_nothing_stays_armed():
    """No elicitation capability at the handshake: the refusal comes back as
    before, but truthful — this client cannot confirm, so nothing was armed —
    instead of the gate's "runs when they say yes", which no yes could reach."""
    ctx = _invested(ToolContext(session=GameSession(), engine=FakeEngine()))

    async with mcp_client(ctx) as client:  # no callback: nothing declared
        result = await _call(client, "resign", {})

    assert result["ok"] is False
    assert result["confirmation_unavailable"] is True
    assert "cannot ask the player" in result["error"]
    assert not ctx.session.is_game_over()
    assert ctx.pending is None


async def test_the_lock_is_not_held_while_the_question_is_open():
    """A human waits on the question, so the mutation lock must not: a read on
    the same board, from another session, is answered while the form is open.
    The second client's SDK session cannot be driven from inside the first's
    callback (the client handles a server request inline in its receive loop),
    so the read comes from a second client on the same context."""
    ctx = _invested(ToolContext(session=GameSession(), engine=FakeEngine()))
    asked = anyio.Event()
    answer = anyio.Event()

    async def human(context: Any, params: Any) -> ElicitResult:
        asked.set()
        await answer.wait()
        return ElicitResult(action="accept", content={"confirm": True})

    async with (
        mcp_client(ctx, elicitation_callback=human) as asking,
        mcp_client(ctx) as other,
    ):
        outcome: dict[str, Any] = {}

        async def ask_for_a_new_game() -> None:
            outcome.update(await _call(asking, "new_game", {}))

        async with anyio.create_task_group() as tg:
            tg.start_soon(ask_for_a_new_game)
            await asked.wait()
            # The question is open. A read on the same board answers now.
            board = await _call(other, "get_board_state", {})
            assert board["fen"].startswith("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP")
            answer.set()

    assert outcome["ok"] is True
    assert outcome["confirmed"] is True
    assert ctx.session.move_history() == []


async def test_a_board_that_moved_while_the_question_was_open_runs_nothing():
    """A confirmation is an answer about a position. When the board moves
    while the form is open — here another client plays a whole exchange — the
    yes arrives about a game that is gone: `confirm_pending` reads the op
    through `live_pending`, runs nothing, and the result says `stale`."""
    ctx = _invested(
        ToolContext(session=GameSession(), engine=FakeEngine(reply_uci="b8c6"))
    )

    async with mcp_client(ctx) as other:

        async def human(context: Any, params: Any) -> ElicitResult:
            moved = await _call(other, "make_move", {"move": "Nf3"})
            assert moved["legal"] is True
            return ElicitResult(action="accept", content={"confirm": True})

        async with mcp_client(ctx, elicitation_callback=human) as asking:
            result = await _call(asking, "new_game", {})

    assert result["ok"] is False
    assert result["stale"] is True
    assert "confirmed" not in result
    assert ctx.session.move_history() == ["e4", "e5", "Nf3", "Nc6"]
    assert ctx.pending is None


async def test_a_question_superseded_by_a_later_one_runs_nothing_on_its_yes():
    """MCP opens no command window, so with the lock released across the human
    wait two gated calls can both ask. The newest question is the one that
    stands (as on the buttons): the later call's yes runs its op, and the
    earlier question's yes must not run it again or run anything else — the
    one-question rule of #270, seen from this surface."""
    ctx = _invested(ToolContext(session=GameSession(), engine=FakeEngine()))
    first_asked = anyio.Event()
    first_answer = anyio.Event()

    async def first_human(context: Any, params: Any) -> ElicitResult:
        first_asked.set()
        await first_answer.wait()
        return ElicitResult(action="accept", content={"confirm": True})

    async with (
        mcp_client(ctx, elicitation_callback=first_human) as first,
        mcp_client(
            ctx, elicitation_callback=_Human("accept", {"confirm": True})
        ) as second,
    ):
        outcome: dict[str, Any] = {}

        async def ask_for_a_new_game() -> None:
            outcome.update(await _call(first, "new_game", {}))

        async with anyio.create_task_group() as tg:
            tg.start_soon(ask_for_a_new_game)
            await first_asked.wait()
            resigned = await _call(second, "resign", {})
            first_answer.set()

    assert resigned["ok"] is True and resigned["confirmed"] is True
    assert ctx.session.is_game_over(), "the later question's yes ran its op"
    assert outcome["ok"] is False
    assert outcome["stale"] is True
    assert ctx.session.move_history() == ["e4", "e5"], "not reset"
    assert ctx.pending is None


async def test_a_fresh_or_finished_board_runs_the_op_without_asking():
    """The gate stands aside where there is nothing to lose — an untouched
    board, or a finished game — and so does the wrapper: no elicitation, the
    op just runs, and the result carries no `confirmed`."""
    human = _Human("decline")  # would say no, were it ever asked

    fresh = ToolContext(session=GameSession(), engine=FakeEngine())
    async with mcp_client(fresh, elicitation_callback=human) as client:
        result = await _call(client, "new_game", {"player_color": "white"})
    assert result["ok"] is True
    assert "confirmed" not in result

    finished = _invested(ToolContext(session=GameSession(), engine=FakeEngine()))
    finished.session.resign("white")
    async with mcp_client(finished, elicitation_callback=human) as client:
        result = await _call(client, "new_game", {})
    assert result["ok"] is True
    assert "confirmed" not in result
    assert finished.session.move_history() == []

    assert human.asked == []


async def test_an_unclaimable_draw_refuses_without_asking_or_arming():
    """`claim_draw` checks claimability before the gate, so a position with no
    claim refuses outright: nothing is armed, and nobody is asked a question
    whose yes could not run."""
    ctx = _invested(ToolContext(session=GameSession(), engine=FakeEngine()))
    human = _Human("accept", {"confirm": True})

    async with mcp_client(ctx, elicitation_callback=human) as client:
        result = await _call(client, "claim_draw", {})

    assert result["ok"] is False
    assert "no draw" in result["error"]
    assert human.asked == []
    assert ctx.pending is None
    assert not ctx.session.is_game_over()


async def test_the_advertised_tools_carry_no_confirmation():
    """The schema is frozen (the eval floor is measured against it): no confirm
    tool and no confirm argument. The yes travels human → client UI → server
    and never through the model's tool list."""
    ctx = ToolContext(session=GameSession())

    async with mcp_client(ctx) as client:
        listed = await client.list_tools()

    assert len(listed.tools) == 20
    assert not [t.name for t in listed.tools if "confirm" in t.name]
    for tool in listed.tools:
        assert "confirm" not in (tool.inputSchema.get("properties") or {})


@pytest.mark.parametrize(
    ("capabilities", "can_ask"),
    [
        pytest.param(None, False, id="no-handshake"),
        pytest.param(ClientCapabilities(), False, id="no-elicitation"),
        pytest.param(
            ClientCapabilities(elicitation=ElicitationCapability()),
            True,
            id="legacy-bare-elicitation-means-form",
        ),
        pytest.param(
            ClientCapabilities(
                elicitation=ElicitationCapability(form=FormElicitationCapability())
            ),
            True,
            id="form",
        ),
        pytest.param(
            ClientCapabilities(
                elicitation=ElicitationCapability(url=UrlElicitationCapability())
            ),
            False,
            id="url-only",
        ),
        pytest.param(
            ClientCapabilities(
                elicitation=ElicitationCapability(
                    form=FormElicitationCapability(), url=UrlElicitationCapability()
                )
            ),
            True,
            id="form-and-url",
        ),
    ],
)
def test_only_a_client_that_declares_form_elicitation_can_be_asked(
    capabilities, can_ask
):
    """Whether the client can ask is read off the handshake, never assumed: a
    bare `elicitation: {}` (the 2025-06-18 shape) means form, `url` alone does
    not, and no capability at all means the fallback refusal."""
    assert _declares_form_elicitation(capabilities) is can_ask
