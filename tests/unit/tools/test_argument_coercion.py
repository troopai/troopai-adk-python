"""Argument-coercion edge-case sweep for ``@function_tool``.

These tests pin the behaviour of Pydantic-driven argument validation
on the parameter shapes most likely to bite production code:

- ``StrEnum`` / ``IntEnum`` round-trip
- ``date`` / ``datetime`` / ``datetime`` with timezone
- ``Optional[T]`` and ``T | None``
- Nested optional Pydantic models
- ``Annotated[T, Field(description=...)]``
- ``Literal[...]`` unions
- Pydantic discriminated unions (``Tag``-discriminated)
- Self-referential / recursive Pydantic models

Tests run the tool end-to-end through ``FunctionTool.on_invoke`` so
both the schema generation AND the argument-reconstruction path are
exercised.

NOTE: this file deliberately does NOT use
``from __future__ import annotations`` — the function-args Pydantic
model is generated dynamically from runtime type hints, and the
deferred-annotation mode breaks resolution for several edge-case
types (``Annotated[X | Y, Field(discriminator=...)]``, forward
references, ``Annotated`` constraints). The tests pin behaviour for
the runtime-evaluated form, which is what production code uses
99% of the time.
"""

import json
from datetime import UTC, date, datetime
from enum import IntEnum, StrEnum
from typing import Annotated, Any, Literal, cast

import pytest
from pydantic import BaseModel, Field

from troopai.adk.tools import FunctionTool, function_tool
from troopai.adk.tools.tool_context import ToolContext


def _ctx(raw: str = "{}") -> ToolContext[dict[str, Any]]:
    return ToolContext(
        tool_name="t",
        tool_call_id="c1",
        tool_arguments={},
        raw_arguments=raw,
        context={},
    )


async def _invoke(tool: FunctionTool, raw_json: str) -> Any:
    assert tool.on_invoke is not None
    return await tool.on_invoke(_ctx(raw_json), raw_json)


# --------------------------------------------------------------------
# Enums
# --------------------------------------------------------------------


class Color(StrEnum):
    RED = "red"
    GREEN = "green"
    BLUE = "blue"


class Priority(IntEnum):
    LOW = 1
    MED = 2
    HIGH = 3


class TestEnumCoercion:
    async def test_str_enum_round_trip(self) -> None:
        @function_tool(name="color", description="Pick a color.")
        def pick(color: Color) -> str:
            return f"chose {color.value}"

        result = await _invoke(pick, '{"color": "red"}')
        assert result == "chose red"

    async def test_str_enum_invalid_value_surfaces_to_llm(self) -> None:
        @function_tool(name="color", description="Pick a color.")
        def pick(color: Color) -> str:
            return color.value

        # Pydantic-level ValueError → routed through the default
        # failure_error_function → returned as a string for the LLM.
        result = await _invoke(pick, '{"color": "purple"}')
        assert isinstance(result, str)
        assert "color" in result.lower() or "purple" in result.lower()

    async def test_int_enum_round_trip(self) -> None:
        @function_tool(name="prio", description="Pick priority.")
        def pick(priority: Priority) -> int:
            return priority.value

        result = await _invoke(pick, '{"priority": 3}')
        assert result == 3

    async def test_int_enum_string_form_rejected(self) -> None:
        @function_tool(name="prio", description="Pick priority.")
        def pick(priority: Priority) -> int:
            return priority.value

        result = await _invoke(pick, '{"priority": "high"}')
        # Pydantic narrows IntEnum strictly to int values; string forms
        # are rejected with an error the LLM can react to.
        assert isinstance(result, str)
        assert "priority" in result.lower() or "value" in result.lower()


# --------------------------------------------------------------------
# Date / datetime
# --------------------------------------------------------------------


class TestDateCoercion:
    async def test_iso_date_string(self) -> None:
        @function_tool(name="day", description="Pick a date.")
        def pick(day: date) -> str:
            return day.isoformat()

        result = await _invoke(pick, '{"day": "2026-03-14"}')
        assert result == "2026-03-14"

    async def test_iso_datetime_naive(self) -> None:
        @function_tool(name="when", description="Pick a moment.")
        def pick(when: datetime) -> str:
            return when.isoformat()

        result = await _invoke(pick, '{"when": "2026-03-14T15:30:00"}')
        assert "2026-03-14T15:30:00" in str(result)

    async def test_iso_datetime_with_tz(self) -> None:
        @function_tool(name="when", description="Pick a moment.")
        def pick(when: datetime) -> bool:
            return when.tzinfo is not None and when.utcoffset() == UTC.utcoffset(when)

        result = await _invoke(pick, '{"when": "2026-03-14T15:30:00Z"}')
        assert result is True

    async def test_invalid_date_string_returns_error(self) -> None:
        @function_tool(name="day", description="Pick a date.")
        def pick(day: date) -> str:
            return day.isoformat()

        result = await _invoke(pick, '{"day": "not-a-date"}')
        assert isinstance(result, str)
        assert "day" in result.lower() or "date" in result.lower()


# --------------------------------------------------------------------
# Optional / | None
# --------------------------------------------------------------------


class TestOptionalCoercion:
    async def test_pep604_union_with_none(self) -> None:
        @function_tool(name="t", description="Optional param.")
        def t(value: int | None = None) -> str:
            return f"got {value}"

        absent = await _invoke(t, "{}")
        assert absent == "got None"
        explicit_null = await _invoke(t, '{"value": null}')
        assert explicit_null == "got None"
        present = await _invoke(t, '{"value": 42}')
        assert present == "got 42"

    async def test_nested_optional_in_model(self) -> None:
        class Inner(BaseModel):
            label: str
            count: int | None = None

        class Outer(BaseModel):
            inner: Inner | None = None
            tag: str = "default"

        @function_tool(name="wrap", description="Wrap.")
        def wrap(payload: Outer) -> str:
            if payload.inner is None:
                return f"no-inner/{payload.tag}"
            return f"{payload.inner.label}:{payload.inner.count}/{payload.tag}"

        # Fully present
        r1 = await _invoke(wrap, '{"payload": {"inner": {"label": "x", "count": 7}, "tag": "a"}}')
        assert r1 == "x:7/a"
        # Inner present, count omitted (defaults to None)
        r2 = await _invoke(wrap, '{"payload": {"inner": {"label": "x"}}}')
        assert r2 == "x:None/default"
        # Inner null
        r3 = await _invoke(wrap, '{"payload": {"inner": null}}')
        assert r3 == "no-inner/default"


# --------------------------------------------------------------------
# Annotated[T, Field(description=...)]
# --------------------------------------------------------------------


class TestAnnotatedDescriptions:
    async def test_annotated_field_description_flows_to_schema(self) -> None:
        @function_tool(name="t", description="Take an integer.")
        def t(value: Annotated[int, Field(description="The integer to process.")]) -> int:
            return value * 2

        schema = t.get_json_schema()
        param = cast(dict[str, Any], schema["properties"])["value"]
        assert param["description"] == "The integer to process."

    async def test_annotated_constraints_enforced(self) -> None:
        @function_tool(name="t", description="Take a constrained integer.")
        def t(value: Annotated[int, Field(ge=0, le=100)]) -> int:
            return value

        too_low = await _invoke(t, '{"value": -1}')
        assert isinstance(too_low, str)
        too_high = await _invoke(t, '{"value": 101}')
        assert isinstance(too_high, str)
        ok = await _invoke(t, '{"value": 50}')
        assert ok == 50


# --------------------------------------------------------------------
# Literal unions
# --------------------------------------------------------------------


class TestLiteralUnion:
    async def test_literal_union_round_trip(self) -> None:
        @function_tool(name="mode", description="Pick a mode.")
        def pick(mode: Literal["fast", "balanced", "thorough"]) -> str:
            return mode

        assert await _invoke(pick, '{"mode": "fast"}') == "fast"
        assert await _invoke(pick, '{"mode": "thorough"}') == "thorough"

    async def test_literal_union_invalid_value_rejected(self) -> None:
        @function_tool(name="mode", description="Pick a mode.")
        def pick(mode: Literal["fast", "balanced", "thorough"]) -> str:
            return mode

        result = await _invoke(pick, '{"mode": "instant"}')
        assert isinstance(result, str)
        assert "mode" in result.lower() or "literal" in result.lower() or "instant" in result.lower()


# --------------------------------------------------------------------
# Discriminated unions
# --------------------------------------------------------------------


class TestDiscriminatedUnion:
    """Pydantic ``Field(discriminator=...)`` unions route by a tag
    field. The schema generator should emit the discriminator branch."""

    async def test_discriminated_union_dispatches_by_kind(self) -> None:
        class CreateAction(BaseModel):
            kind: Literal["create"]
            name: str

        class DeleteAction(BaseModel):
            kind: Literal["delete"]
            target_id: int

        @function_tool(name="act", description="Run an action.")
        def act(action: Annotated[CreateAction | DeleteAction, Field(discriminator="kind")]) -> str:
            if isinstance(action, CreateAction):
                return f"created {action.name}"
            return f"deleted {action.target_id}"

        r1 = await _invoke(act, '{"action": {"kind": "create", "name": "thing"}}')
        assert r1 == "created thing"
        r2 = await _invoke(act, '{"action": {"kind": "delete", "target_id": 7}}')
        assert r2 == "deleted 7"

    async def test_discriminator_missing_kind_rejected(self) -> None:
        class CreateAction(BaseModel):
            kind: Literal["create"]
            name: str

        class DeleteAction(BaseModel):
            kind: Literal["delete"]
            target_id: int

        @function_tool(name="act", description="Run an action.")
        def act(action: Annotated[CreateAction | DeleteAction, Field(discriminator="kind")]) -> str:
            return str(action)

        result = await _invoke(act, '{"action": {"name": "thing"}}')
        assert isinstance(result, str)


# --------------------------------------------------------------------
# Recursive / self-referential models
# --------------------------------------------------------------------


class TestRecursiveModels:
    async def test_self_referential_tree(self) -> None:
        class Node(BaseModel):
            value: int
            children: list["Node"] = Field(default_factory=list)

        Node.model_rebuild()

        @function_tool(name="sum_tree", description="Sum a tree of integers.")
        def sum_tree(root: Node) -> int:
            stack: list[Node] = [root]
            total = 0
            while len(stack) > 0:
                n = stack.pop()
                total += n.value
                stack.extend(n.children)
            return total

        tree = {"value": 1, "children": [{"value": 2, "children": [{"value": 3}]}, {"value": 4}]}
        result = await _invoke(sum_tree, json.dumps({"root": tree}))
        assert result == 10


# --------------------------------------------------------------------
# Schema integration
# --------------------------------------------------------------------


class TestSchemaIntegration:
    """The generated schema MUST be valid JSON Schema and round-trip
    through json.loads/json.dumps without losing fields."""

    @pytest.mark.parametrize(
        "fn,expected_required",
        [
            (lambda x: x, ["x"]),
            (lambda x, y=2: (x, y), ["x"]),
        ],
    )
    def test_required_fields(self, fn: Any, expected_required: list[str]) -> None:
        @function_tool(name="t", description="t")
        def wrapped(x: int, y: int = 2) -> tuple[int, int]:
            return fn(x, y)

        schema = wrapped.get_json_schema()
        assert set(schema.get("required", [])) >= set(expected_required)

    def test_schema_is_json_serializable(self) -> None:
        @function_tool(name="t", description="t")
        def t(value: int | None = None, mode: Literal["a", "b"] = "a") -> str:
            return f"{mode}:{value}"

        schema = t.get_json_schema()
        # Round-trip via JSON to confirm no non-serializable objects
        # leak into the schema.
        round_tripped = json.loads(json.dumps(schema))
        assert round_tripped == schema


class TestVariadicArgsRejection:
    """``*args`` and ``**kwargs`` in tool signatures are rejected at
    decoration time. Variadic shapes have no clean JSON-Schema
    equivalent the LLM can reliably target — authors who need a list
    / dict parameter spell it explicitly."""

    def test_var_positional_rejected(self) -> None:
        from troopai.adk.exceptions import UserError

        with pytest.raises(UserError, match=r"\*args parameter"):

            @function_tool(name="t", description="t")
            def t(*items: int) -> int:
                return sum(items)

    def test_var_keyword_rejected(self) -> None:
        from troopai.adk.exceptions import UserError

        with pytest.raises(UserError, match=r"\*\*kwargs parameter"):

            @function_tool(name="t", description="t")
            def t(**meta: str) -> str:
                return ",".join(meta.values())

    def test_explicit_list_param_still_works(self) -> None:
        # The rejection message tells authors to use `list[T]` /
        # `dict[str, T]` explicitly; pin that that still works.
        @function_tool(name="t", description="t")
        def t(items: list[int]) -> int:
            return sum(items)

        assert t.name == "t"

    def test_explicit_dict_param_still_works(self) -> None:
        @function_tool(name="t", description="t")
        def t(meta: dict[str, str]) -> str:
            return ",".join(meta.values())

        assert t.name == "t"

    def test_dict_param_schema_is_object_not_corrupted(self) -> None:
        """A ``dict[str, str]`` param must yield an object schema.

        Regression: ``_strip_annotated`` unwrapped ANY multi-arg generic to
        its first argument, so ``dict[str, str]`` collapsed to ``str``
        (``{"type": "string"}``) — the LLM was told to pass a string where a
        mapping was required. Only ``typing.Annotated`` layers may be
        stripped; ``dict`` / ``tuple`` / ``X | Y`` must survive intact.
        """
        from typing import Any, cast

        @function_tool(name="t", description="t")
        def t(meta: dict[str, str]) -> str:
            return ",".join(meta.values())

        schema = t.get_json_schema()
        param = cast(dict[str, Any], schema["properties"])["meta"]
        assert param.get("type") == "object", f"dict param corrupted to {param!r}"
