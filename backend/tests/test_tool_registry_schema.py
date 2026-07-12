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
"""

import json
import re
from pathlib import Path
from typing import Annotated, Any, Literal

import pytest
from pydantic import Field

from chessapp.game import GameSession
from chessapp.tools import ToolContext, ToolRegistry, build_registry

GOLDEN = Path(__file__).parent / "tool_definitions_golden.json"

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
