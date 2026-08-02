"""Regression tests for Wave-B bug findings in schemas/."""

from __future__ import annotations

import copy
from typing import Any, Literal, Union

import pytest
from pydantic import BaseModel

from troopai.adk.schemas.agent_output_schema import AgentOutputSchema
from troopai.adk.schemas.utils import (
    _EMPTY_SCHEMA,
    _ensure_strict_schema,
    ensure_strict_schema,
    normalize_schema,
)

# ---------------------------------------------------------------------------
# Finding 1 — _EMPTY_SCHEMA.copy() is shallow
# ---------------------------------------------------------------------------


class TestEmptySchemaDeepCopy:
    """ensure_strict_schema({}) must return an independent copy each call."""

    def test_mutating_returned_schema_does_not_corrupt_module_constant(self) -> None:
        """Two calls to ensure_strict_schema({}) must be independent objects."""
        result1 = ensure_strict_schema({})
        result2 = ensure_strict_schema({})

        # Mutate result1 — must NOT affect result2 or the module constant
        result1["properties"]["injected"] = {"type": "string"}
        result1["required"].append("injected")

        assert "injected" not in result2["properties"]
        assert "injected" not in result2["required"]
        assert "injected" not in _EMPTY_SCHEMA["properties"]
        assert "injected" not in _EMPTY_SCHEMA["required"]

    def test_empty_schema_constant_untouched_after_use(self) -> None:
        """The _EMPTY_SCHEMA module constant must stay pristine after calls."""
        original_props = copy.deepcopy(_EMPTY_SCHEMA["properties"])
        original_required = copy.deepcopy(_EMPTY_SCHEMA["required"])

        result = ensure_strict_schema({})
        result["properties"]["x"] = {"type": "integer"}
        result["required"].append("x")

        assert _EMPTY_SCHEMA["properties"] == original_props
        assert _EMPTY_SCHEMA["required"] == original_required


# ---------------------------------------------------------------------------
# Finding 2 — get_type_hints broad except
# ---------------------------------------------------------------------------


class TestGetTypeHintsBroadExcept:
    """get_type_hints failure should only catch NameError, not all exceptions."""

    def test_name_error_forward_ref_still_yields_empty_hints(self) -> None:
        """A forward-ref NameError results in empty type_hints (no crash)."""
        from troopai.adk.schemas.function_schema import function_schema

        # A function with a bad forward-reference annotation — eval fails
        # with NameError.  function_schema should still succeed (graceful
        # degradation), and the param should map to Any.
        def fn_with_bad_ref(x: NonExistentType) -> str:  # type: ignore[name-defined]  # noqa: F821
            return str(x)

        fs = function_schema(fn_with_bad_ref)
        # Should succeed without raising
        assert fs.name == "fn_with_bad_ref"

    def test_attribute_error_is_not_swallowed(self) -> None:
        """AttributeError from get_type_hints must NOT be silently swallowed."""
        from unittest.mock import patch

        from troopai.adk.schemas.function_schema import function_schema as fn_schema

        def simple_fn(x: int) -> str:
            return str(x)

        # Patch get_type_hints at the module namespace where function_schema uses it
        with (
            patch("troopai.adk.schemas.function_schema.get_type_hints", side_effect=AttributeError("broken")),
            pytest.raises(AttributeError, match="broken"),
        ):
            fn_schema(simple_fn)


# ---------------------------------------------------------------------------
# Finding 3 — additionalProperties truthy check
# ---------------------------------------------------------------------------


class TestAdditionalPropertiesTruthyCheck:
    """additionalProperties dict sub-schemas must not be rejected."""

    def test_additional_properties_dict_sub_schema_not_rejected(self) -> None:
        """additionalProperties: {"type": "string"} is valid and must not raise."""
        schema: dict[str, Any] = {
            "type": "object",
            "properties": {"meta": {"type": "string"}},
            "additionalProperties": {"type": "string"},
        }
        # Must NOT raise UserError
        result = _ensure_strict_schema(schema, path=(), root=schema)
        # The dict sub-schema should remain (not raise)
        assert result["additionalProperties"] == {"type": "string"}

    def test_additional_properties_true_still_raises(self) -> None:
        """additionalProperties: True must still raise UserError."""
        from troopai.adk.exceptions import UserError

        schema: dict[str, Any] = {
            "type": "object",
            "properties": {"x": {"type": "string"}},
            "additionalProperties": True,
        }
        with pytest.raises(UserError):
            _ensure_strict_schema(schema, path=(), root=schema)

    def test_additional_properties_false_not_rejected(self) -> None:
        """additionalProperties: False must not raise."""
        schema: dict[str, Any] = {
            "type": "object",
            "properties": {"x": {"type": "string"}},
            "additionalProperties": False,
        }
        result = _ensure_strict_schema(schema, path=(), root=schema)
        assert result["additionalProperties"] is False


# ---------------------------------------------------------------------------
# Finding 4 — allOf single-element flatten uses update()
# ---------------------------------------------------------------------------


class TestAllOfSingleElementFlatten:
    """allOf[0] merge must preserve parent-level keys."""

    def test_parent_title_preserved_after_allof_flatten(self) -> None:
        """Parent 'title' must survive single-element allOf flattening."""
        schema: dict[str, Any] = {
            "title": "ParentTitle",
            "description": "Parent description",
            "allOf": [
                {
                    "type": "object",
                    "title": "InnerTitle",
                    "properties": {"x": {"type": "string"}},
                }
            ],
        }
        result = _ensure_strict_schema(copy.deepcopy(schema), path=(), root=schema)

        assert result.get("title") == "ParentTitle", f"title was clobbered: {result}"
        assert result.get("description") == "Parent description", f"description was clobbered: {result}"
        assert "allOf" not in result

    def test_parent_description_not_overwritten_by_allof_child(self) -> None:
        """Parent description beats child description on allOf flatten."""
        schema: dict[str, Any] = {
            "description": "parent desc",
            "allOf": [{"type": "object", "description": "child desc", "properties": {}}],
        }
        result = _ensure_strict_schema(copy.deepcopy(schema), path=(), root=schema)
        assert result["description"] == "parent desc"


# ---------------------------------------------------------------------------
# Finding 5 — normalize_schema injects type:object onto anyOf/oneOf/allOf
# ---------------------------------------------------------------------------


class TestNormalizeSchemCompositionKeyword:
    """normalize_schema must not inject type:object when composition keyword present."""

    def test_anyof_root_does_not_get_type_object(self) -> None:
        """{'anyOf': [...]} must not receive type:'object'."""
        schema = {"anyOf": [{"type": "string"}, {"type": "integer"}]}
        result = normalize_schema(schema)
        assert "type" not in result, f"type was injected: {result}"
        assert "properties" not in result
        assert "required" not in result

    def test_oneof_root_does_not_get_type_object(self) -> None:
        """{'oneOf': [...]} must not receive type:'object'."""
        schema = {"oneOf": [{"type": "string"}, {"type": "null"}]}
        result = normalize_schema(schema)
        assert "type" not in result

    def test_allof_root_does_not_get_type_object(self) -> None:
        """{'allOf': [...]} must not receive type:'object'."""
        schema = {"allOf": [{"type": "object", "properties": {}}]}
        result = normalize_schema(schema)
        assert "type" not in result

    def test_plain_object_still_gets_defaults(self) -> None:
        """Plain object schema without composition still gets defaults."""
        schema = {"properties": {"x": {"type": "string"}}}
        result = normalize_schema(schema)
        assert result["type"] == "object"
        assert "required" in result


# ---------------------------------------------------------------------------
# Finding 6 — validate_json outer except masks infra errors
# ---------------------------------------------------------------------------


class TestValidateJsonExceptNarrow:
    """validate_json outer except must only catch Pydantic ValidationError."""

    def test_attribute_error_in_type_adapter_propagates(self) -> None:
        """AttributeError from TypeAdapter.validate_python must not be swallowed."""
        import unittest.mock as mock

        class MyModel(BaseModel):
            val: int

        schema = AgentOutputSchema(MyModel)

        # Patch validate_python to raise AttributeError
        with (
            mock.patch.object(
                schema._type_adapter,
                "validate_python",
                side_effect=AttributeError("internal broken"),
            ),
            pytest.raises(AttributeError, match="internal broken"),
        ):
            schema.validate_json('{"val": 1}')

    def test_pydantic_validation_error_still_raises_value_error(self) -> None:
        """Pydantic ValidationError from bad data becomes a ValueError."""

        class MyModel(BaseModel):
            val: int

        schema = AgentOutputSchema(MyModel)
        with pytest.raises(ValueError, match="does not match"):
            schema.validate_json('{"val": "not-an-int"}')


# ---------------------------------------------------------------------------
# Finding 7 — wrapped-type unwrap fallback returns wrong value
# ---------------------------------------------------------------------------


class TestWrappedTypeUnwrapFallback:
    """When wrapped response lacks 'response' key, raise ValueError."""

    def test_missing_response_key_raises_value_error(self) -> None:
        """A wrapped schema that produces no 'response' key must raise ValueError."""
        schema = AgentOutputSchema(int)

        # Patch _type_adapter to return a dict WITHOUT the 'response' key
        import unittest.mock as mock

        with (
            mock.patch.object(
                schema._type_adapter,
                "validate_python",
                return_value={"wrong_key": 42},
            ),
            pytest.raises(ValueError, match="[Ee]xpected wrapped response"),
        ):
            schema.validate_json('{"response": 42}')

    def test_valid_wrapped_response_still_unwraps(self) -> None:
        """Normal wrapped response still unwraps correctly."""
        schema = AgentOutputSchema(int)
        result = schema.validate_json('{"response": 99}')
        assert result == 99


# ---------------------------------------------------------------------------
# Finding 8 — inner discriminator-repair except discards exception
# ---------------------------------------------------------------------------


class TestDiscriminatorRepairExceptionChaining:
    """Repair-attempt exception must be chained into the raised ValueError."""

    def test_repair_exception_chained(self) -> None:
        """When second validate_python (after repair) fails, the repair error is chained."""
        from unittest.mock import patch

        class ModelA(BaseModel):
            kind: Literal["a"] = "a"
            value: str

        schema = AgentOutputSchema(Union[ModelA, ModelA])

        repair_exc = RuntimeError("repair boom")

        # _try_repair_discriminators must return a DIFFERENT object (not data)
        # so the code takes the `if repaired is not data:` branch and calls
        # the second validate_python — which we make raise repair_exc.
        repaired_dict = {"kind": "still-wrong", "value": "x"}

        original_validate = schema._type_adapter.validate_python
        call_count = [0]

        def patched_validate(data: Any, **kw: Any) -> Any:
            call_count[0] += 1
            if call_count[0] == 1:
                # First call — raise ValidationError via the real adapter on bad data
                return original_validate({"kind": "z", "value": "x"})
            # Second call (repair attempt) — raise the repair error
            raise repair_exc

        with (
            patch.object(schema, "_try_repair_discriminators", return_value=repaired_dict),
            patch.object(schema._type_adapter, "validate_python", side_effect=patched_validate),
        ):
            try:
                schema.validate_json('{"response": {"kind": "z", "value": "x"}}')
            except ValueError as exc:
                # The repair_exc must appear in the exception chain
                causes = []
                current: BaseException | None = exc
                while current is not None:
                    causes.append(current)
                    current = current.__cause__ or current.__context__
                assert any(c is repair_exc for c in causes), f"repair_exc not in chain: {causes}"
            else:
                pytest.fail("Expected ValueError")


# ---------------------------------------------------------------------------
# Finding 9 — output_schema is a public mutable attribute (design-fork)
# Deferred — see report
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Finding 10 — FunctionToolSchema uses Union[] instead of PEP-604
# ---------------------------------------------------------------------------


class TestFunctionToolSchemaPep604:
    """FunctionToolSchema must use PEP-604 union syntax."""

    def test_function_tool_schema_uses_pep604(self) -> None:
        """FunctionToolSchema should be type[BaseModel] | dict[str, Any]."""
        import types

        from troopai.adk.schemas.function_schema import FunctionToolSchema

        # In Python 3.10+ a PEP-604 union has type UnionType, not typing.Union.
        # We just verify it's a valid type alias that accepts both arms.
        assert FunctionToolSchema is not None
        # The alias must be usable for isinstance checks indirectly via get_origin
        from typing import get_origin

        # PEP 604: get_origin returns types.UnionType for X | Y
        # typing.Union: get_origin returns typing.Union
        # We require the PEP 604 form (types.UnionType)
        assert get_origin(FunctionToolSchema) is types.UnionType, (
            f"FunctionToolSchema should use X | Y union syntax (types.UnionType), "
            f"got origin={get_origin(FunctionToolSchema)!r}"
        )
