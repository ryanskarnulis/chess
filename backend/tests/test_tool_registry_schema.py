"""Schema-equivalence + decorator-derivation contract for the tool registry.

The tool layer migrated from hand-written JSON-schema `Tool` dataclasses to a
decorator that derives each tool's name (`__name__`), description (docstring),
and argument schema (FastMCP's `func_metadata`, from the typed signature). The
plan requires `definitions()` output to stay schema-equivalent across that
migration — pinned here against a golden fixture dumped from the pre-migration
code (`tests/tool_definitions_golden.json`).

Equivalence is compared modulo four harmless, documented normalizations:
Pydantic's advisory `title` and `default` keys and the nullable-union form it
emits for an optional typed param (all in `_normalize`), plus whitespace runs
in each tool's `description` (in `_canonical_description`) — a wrapped docstring
joins lines with newlines where the old hand-written string joined with spaces,
which is cosmetically identical to the model.

Those normalizations are exactly the keys the standing prohibition protects.
The 2026-07-21 minimization attempt (DONE.md; `docs/agent-evals.md` "Standing
results") found that stripping any one of Pydantic's `title`, the `default`,
or the `anyOf`-null union off the emitted schema independently collapsed
`undo_and_replace` on gemma-4-12b — and this golden, by design, cannot see any
of them go. So a second fixture sits beside it: `tool_definitions_emitted.json`
is the *exact* output of `definitions()`, both flavours of `make_move`, held
byte for byte by `test_emitted_schemas_match_the_snapshot_byte_for_byte` —
key order, whitespace and all, because the wire carries the dicts verbatim
(`provider.LlamaCppProvider._payload`) and the model reads what the wire
carries. Regenerating it is a deliberate act (`CHESSAPP_UPDATE_SCHEMA_SNAPSHOT=1`)
and a schema change, which runs the eval gate before it merges.
"""

import json
import os
import re
from pathlib import Path
from typing import Annotated, Any, Literal

import pytest
from pydantic import Field

from chessapp.game import GameSession
from chessapp.tools import UNDO_PLIES_MAX, ToolContext, ToolRegistry, build_registry

GOLDEN = Path(__file__).parent / "tool_definitions_golden.json"
EMITTED = Path(__file__).parent / "tool_definitions_emitted.json"

_NULL = {"type": "null"}


def _normalize(node: Any) -> Any:
    """Collapse schema-equivalent representations to one canonical form.

    - Drop `title`: Pydantic names every model and field; it is advisory and
      never constrains validation.
    - Drop `default`: advisory only — jsonschema validation does not inject
      defaults and the handler's own signature default is unchanged, so the
      accepted-argument set is identical with or without it.
    - Unwrap `{"anyOf": [S, {"type": "null"}]}` (Pydantic's shape for an
      optional typed param) to S. The golden expresses the same optionality
      by omitting the key from `required`; both forms accept omission and
      constrain a present value identically.
    """
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


def _canonical_description(definition: dict[str, Any]) -> dict[str, Any]:
    """Collapse whitespace runs in the tool description (docstring wrapping is
    cosmetic; the model sees the same text)."""
    fn = definition["function"]
    fn["description"] = re.sub(r"\s+", " ", fn["description"]).strip()
    return definition


def _by_name(definitions: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        d["function"]["name"]: _canonical_description(_normalize(d))
        for d in definitions
    }


def test_definitions_match_golden():
    """Every tool's derived schema is schema-equivalent to the pre-migration
    golden — same names, descriptions, constraints, required lists, closed
    schemas — modulo the documented normalizations."""
    golden = json.loads(GOLDEN.read_text())
    live = build_registry(ToolContext(session=GameSession())).definitions()

    golden_by_name = _by_name(golden)
    live_by_name = _by_name(live)

    assert live_by_name.keys() == golden_by_name.keys()
    for name in golden_by_name:
        assert live_by_name[name] == golden_by_name[name], f"schema drift in {name}"


# --- the exact emitted schema ------------------------------------------------


def _emitted() -> dict[str, list[dict[str, Any]]]:
    """Both flavours of the live registry, exactly as `definitions()` emits them.

    `make_move`'s description differs between the two (`tools._make_move_doc`):
    the atomic exchange is what the MCP server advertises, the split one is what
    the app's brain is offered. Everything else is identical, and the snapshot
    holds both so neither surface can drift unseen.
    """
    return {
        "atomic_exchange": build_registry(
            ToolContext(session=GameSession())
        ).definitions(),
        "split_exchange": build_registry(
            ToolContext(session=GameSession()), atomic_exchange=False
        ).definitions(),
    }


def _serialized(emitted: dict[str, list[dict[str, Any]]]) -> str:
    """The snapshot's one serialization: emitted key order kept (no
    `sort_keys`), non-ASCII kept (the em-dashes the model reads), trailing
    newline so the file is a well-formed text file."""
    return json.dumps(emitted, indent=2, ensure_ascii=False) + "\n"


def _first_difference(expected: str, actual: str) -> str:
    """The first line where the two serializations part, for the failure."""
    for number, (want, got) in enumerate(
        zip(expected.splitlines(), actual.splitlines(), strict=False), start=1
    ):
        if want != got:
            return f"line {number}:\n  snapshot: {want!r}\n  emitted:  {got!r}"
    return (
        f"lengths differ: snapshot {len(expected.splitlines())} lines, "
        f"emitted {len(actual.splitlines())}"
    )


def test_emitted_schemas_match_the_snapshot_byte_for_byte():
    """`definitions()` emits exactly `tool_definitions_emitted.json`.

    The golden above normalizes away `title`, `default` and the nullable
    union, so a change to any of them passes it — and each of those, stripped
    on its own, put `undo_and_replace` under the floor on gemma-4-12b
    (2026-07-21). This is the tripwire the golden cannot be: a byte-for-byte
    comparison of the serialized definitions, both `make_move` flavours, so
    the failure names the first line that moved and nothing about the model
    has to be re-measured to find out that something did.

    A deliberate change regenerates the file — run this test once with
    `CHESSAPP_UPDATE_SCHEMA_SNAPSHOT=1` — and the diff of the fixture in the PR
    *is* the schema change, to be read line by line and gated on the eval
    suite before it merges. Every tool's docstring is in here too, so a
    description edit shows up as the same kind of diff.
    """
    actual = _serialized(_emitted())
    if os.environ.get("CHESSAPP_UPDATE_SCHEMA_SNAPSHOT"):
        EMITTED.write_text(actual, encoding="utf-8")
    expected = EMITTED.read_text(encoding="utf-8")
    assert actual == expected, (
        "the emitted tool schemas drifted from tests/tool_definitions_emitted.json "
        "— every key here reaches the model verbatim, and title/default/anyOf-null "
        "are the ones the standing prohibition protects. If the change is "
        "deliberate, regenerate with CHESSAPP_UPDATE_SCHEMA_SNAPSHOT=1, read the "
        "fixture diff, and run the eval gate. First difference at "
        + _first_difference(expected, actual)
    )


def test_the_snapshot_is_the_serialization_this_test_would_write():
    """The fixture on disk is byte-identical to a fresh dump of *itself* — so a
    hand edit that reorders keys, escapes an em-dash or drops the trailing
    newline is caught as a fixture problem, separately from a live drift."""
    text = EMITTED.read_text(encoding="utf-8")
    assert _serialized(json.loads(text)) == text


# What the normalizations hide, stated positively: the shapes Pydantic emits for
# an optional typed parameter and for a defaulted one, which the 2026-07-21
# arms each stripped and each time collapsed `undo_and_replace`. Pinned on the
# live registry rather than on the file, so a regenerated snapshot that lost
# one of them still fails here, env var or no env var.

_NULL_BRANCH = {"type": "null"}

# (tool, parameter) → the typed branch that sits beside the null one.
_OPTIONAL_TYPED_PARAMS: dict[tuple[str, str], dict[str, Any]] = {
    ("analyze_last_move", "color"): {"enum": ["white", "black"], "type": "string"},
    ("undo", "plies"): {"maximum": UNDO_PLIES_MAX, "minimum": 1, "type": "integer"},
    ("new_game", "player_color"): {"enum": ["white", "black"], "type": "string"},
    ("resign", "color"): {"enum": ["white", "black"], "type": "string"},
}

# (tool, parameter) → the advisory default the schema still carries.
_DEFAULTED_PARAMS: dict[tuple[str, str], Any] = {
    ("get_best_moves", "n"): 3,
    ("save_game", "name"): "autosave",
    ("resume_game", "name"): "autosave",
}


def _live_parameters() -> dict[str, dict[str, Any]]:
    return {
        d["function"]["name"]: d["function"]["parameters"]
        for d in build_registry(ToolContext(session=GameSession())).definitions()
    }


def test_every_derived_tool_keeps_pydantic_s_title():
    """Each derived argument model is titled `<tool>Arguments` — the first key
    the minimization stripped, and the first one to collapse the tripwire.
    `set_difficulty` is the one hand-written schema (the `parameters=` escape
    hatch, for its `oneOf`) and has never carried a title; pinned as the one
    exception rather than skipped, so a second hand-written schema is noticed."""
    parameters = _live_parameters()
    assert "title" not in parameters.pop("set_difficulty")
    for name, schema in parameters.items():
        assert schema.get("title") == f"{name}Arguments", name


def test_optional_typed_params_keep_the_nullable_union_and_the_null_default():
    """An optional typed parameter is `anyOf: [<type>, null]` with
    `default: null` and a `title` — never the bare type with the key merely
    left out of `required`, which the golden treats as the same thing and the
    model does not. (A field description rides along only where the handler
    wrote one; the byte snapshot holds those.)"""
    parameters = _live_parameters()
    for (tool, param), typed in _OPTIONAL_TYPED_PARAMS.items():
        prop = parameters[tool]["properties"][param]
        assert prop["anyOf"] == [typed, _NULL_BRANCH], (tool, param, prop)
        assert "default" in prop and prop["default"] is None, (tool, param, prop)
        assert prop["title"], (tool, param, prop)
        assert param not in parameters[tool].get("required", []), (tool, param)


def test_defaulted_params_keep_their_default():
    """A parameter with a handler default carries it in the schema too."""
    parameters = _live_parameters()
    for (tool, param), default in _DEFAULTED_PARAMS.items():
        prop = parameters[tool]["properties"][param]
        assert prop["default"] == default, (tool, param, prop)
        assert param not in parameters[tool].get("required", []), (tool, param)


def test_set_difficulty_keeps_oneof_via_override():
    """`set_difficulty`'s exactly-one-of tier/skill_level/elo (oneOf) cannot be
    derived from a plain signature; it rides the decorator's `parameters=`
    escape hatch and must survive in the emitted schema."""
    live = build_registry(ToolContext(session=GameSession())).definitions()
    params = next(
        d["function"]["parameters"]
        for d in live
        if d["function"]["name"] == "set_difficulty"
    )
    assert params["oneOf"] == [
        {"required": ["tier"]},
        {"required": ["skill_level"]},
        {"required": ["elo"]},
    ]


# --- decorator derivation ---------------------------------------------------


def test_tool_decorator_derives_name_and_description_and_schema():
    """The decorator takes the tool's name from `__name__`, its description
    from the docstring, and its argument schema from the typed signature."""
    registry = ToolRegistry()

    @registry.tool()
    def sample(n: Annotated[int, Field(ge=1, le=5, description="count")] = 2):
        """A sample tool."""
        return {"ok": True, "n": n}

    (definition,) = registry.definitions()
    fn = definition["function"]
    assert fn["name"] == "sample"
    assert fn["description"] == "A sample tool."
    prop = fn["parameters"]["properties"]["n"]
    assert prop["type"] == "integer"
    assert prop["minimum"] == 1
    assert prop["maximum"] == 5
    assert prop["description"] == "count"
    # Closed schema: extra args are rejected at dispatch, as before.
    assert fn["parameters"]["additionalProperties"] is False
    assert registry.dispatch("sample", {})["n"] == 2
    assert registry.dispatch("sample", {"n": 9})["ok"] is False


def test_tool_decorator_requires_docstring():
    registry = ToolRegistry()

    with_no_doc = lambda: {"ok": True}  # noqa: E731
    with_no_doc.__name__ = "no_doc"
    with pytest.raises(ValueError):
        registry.tool()(with_no_doc)


def test_tool_decorator_accepts_literal_enum():
    """A `Literal` param becomes a string enum in the schema."""
    registry = ToolRegistry()

    @registry.tool()
    def pick(choice: Literal["a", "b"]):
        """Pick one."""
        return {"ok": True, "choice": choice}

    prop = registry.definitions()[0]["function"]["parameters"]["properties"]["choice"]
    assert prop["enum"] == ["a", "b"]
    assert registry.dispatch("pick", {"choice": "c"})["ok"] is False
