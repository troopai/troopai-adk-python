"""Tests for SchemaEnforcement.COMPACT and Union type naming."""

from __future__ import annotations

import copy
from typing import Literal, Union

import pytest
from pydantic import BaseModel, Field

from troopai.adk.schemas.agent_output_schema import AgentOutputSchema
from troopai.adk.schemas.utils import (
    SchemaEnforcement,
    _convert_const_to_enum,
    _strip_schema_metadata,
    enforce_schema,
    ensure_compact_schema,
    ensure_strict_schema,
)


class SimpleModel(BaseModel):
    name: str
    age: int


class DiscriminatedA(BaseModel):
    kind: Literal["a"] = "a"
    value: str = Field(description="The value for A.")


class DiscriminatedB(BaseModel):
    kind: Literal["b"] = "b"
    score: int = Field(description="The score for B.")


class TestCompactStripsMetadata:
    """Tests for _strip_schema_metadata and ensure_compact_schema."""

    def test_compact_strips_titles(self) -> None:
        """COMPACT removes all 'title' keys from schema."""
        schema = {
            "title": "SimpleModel",
            "type": "object",
            "properties": {
                "name": {"title": "Name", "type": "string"},
                "age": {"title": "Age", "type": "integer"},
            },
        }
        _strip_schema_metadata(schema)

        assert "title" not in schema
        assert "title" not in schema["properties"]["name"]
        assert "title" not in schema["properties"]["age"]

    def test_compact_strips_const_descriptions(self) -> None:
        """COMPACT removes 'description' from const-valued properties."""
        schema = {
            "properties": {
                "kind": {
                    "const": "refund",
                    "description": "Always 'refund'.",
                    "title": "Kind",
                },
            },
        }
        _strip_schema_metadata(schema)

        prop = schema["properties"]["kind"]
        assert "description" not in prop
        assert "title" not in prop
        assert prop["const"] == "refund"

    def test_compact_preserves_non_const_descriptions(self) -> None:
        """COMPACT keeps 'description' on properties without 'const'."""
        schema = {
            "properties": {
                "name": {
                    "type": "string",
                    "description": "User name.",
                },
            },
        }
        _strip_schema_metadata(schema)

        assert schema["properties"]["name"]["description"] == "User name."

    def test_compact_applies_strict_rules(self) -> None:
        """COMPACT first applies strict-mode compliance, then strips metadata."""
        schema = {
            "title": "TestSchema",
            "type": "object",
            "properties": {
                "name": {"title": "Name", "type": "string"},
            },
        }
        result = ensure_compact_schema(copy.deepcopy(schema))

        # Strict rules applied: additionalProperties, required
        assert result.get("additionalProperties") is False
        assert result.get("required") == ["name"]
        # Metadata stripped
        assert "title" not in result
        assert "title" not in result["properties"]["name"]

    def test_compact_is_strict_via_enforce_schema(self) -> None:
        """enforce_schema(COMPACT) calls ensure_compact_schema."""
        schema = {
            "title": "Test",
            "type": "object",
            "properties": {
                "x": {"title": "X", "type": "string"},
            },
        }
        result = enforce_schema(copy.deepcopy(schema), SchemaEnforcement.COMPACT)

        assert "title" not in result
        assert result.get("additionalProperties") is False

    def test_compact_recurses_into_defs(self) -> None:
        """COMPACT strips titles from $defs definitions."""
        schema = {
            "title": "Root",
            "type": "object",
            "properties": {},
            "$defs": {
                "SubModel": {
                    "title": "SubModel",
                    "type": "object",
                    "properties": {
                        "val": {"title": "Val", "type": "string"},
                    },
                },
            },
        }
        _strip_schema_metadata(schema)

        assert "title" not in schema
        assert "title" not in schema["$defs"]["SubModel"]
        assert "title" not in schema["$defs"]["SubModel"]["properties"]["val"]

    def test_compact_recurses_into_anyof(self) -> None:
        """COMPACT strips titles from anyOf variants."""
        schema = {
            "title": "Union",
            "anyOf": [
                {"title": "VariantA", "type": "object", "properties": {}},
                {"title": "VariantB", "type": "string"},
            ],
        }
        _strip_schema_metadata(schema)

        assert "title" not in schema
        assert "title" not in schema["anyOf"][0]
        assert "title" not in schema["anyOf"][1]


class TestCompactIsStrict:
    """Tests that COMPACT is treated as strict for provider hints."""

    def test_agent_output_schema_compact_is_strict(self) -> None:
        """AgentOutputSchema with COMPACT reports is_strict_json_schema=True."""
        schema = AgentOutputSchema(SimpleModel, SchemaEnforcement.COMPACT)
        assert schema.is_strict_json_schema() is True

    def test_agent_output_schema_strict_is_strict(self) -> None:
        """STRICT still reports True (regression check)."""
        schema = AgentOutputSchema(SimpleModel, SchemaEnforcement.STRICT)
        assert schema.is_strict_json_schema() is True

    def test_agent_output_schema_normalized_not_strict(self) -> None:
        """NORMALIZED is not strict."""
        schema = AgentOutputSchema(SimpleModel, SchemaEnforcement.NORMALIZED)
        assert schema.is_strict_json_schema() is False


class TestUnionNameSanitized:
    """Tests for Union type naming in AgentOutputSchema."""

    def test_union_name_produces_clean_string(self) -> None:
        """Union type produces a sanitized underscore-joined name."""
        schema = AgentOutputSchema(Union[DiscriminatedA, DiscriminatedB])

        name = schema.name()
        assert "DiscriminatedA" in name
        assert "DiscriminatedB" in name
        # Should NOT contain dots, brackets, or spaces
        assert "." not in name
        assert "[" not in name
        assert "]" not in name
        assert " " not in name

    def test_single_model_name_unchanged(self) -> None:
        """Non-Union types still use __name__ directly."""
        schema = AgentOutputSchema(SimpleModel)
        assert schema.name() == "SimpleModel"

    def test_none_schema_name_is_str(self) -> None:
        """None output_schema returns 'str'."""
        schema = AgentOutputSchema(None)
        assert schema.name() == "str"


class TestConstToEnum:
    """Tests for _convert_const_to_enum (Gemini provider compatibility)."""

    def test_const_converted_to_single_enum(self) -> None:
        """const: 'refund' becomes enum: ['refund']."""
        schema = {"const": "refund", "type": "string"}
        _convert_const_to_enum(schema)

        assert "const" not in schema
        assert schema["enum"] == ["refund"]

    def test_const_in_properties(self) -> None:
        """const values inside properties are converted."""
        schema = {
            "properties": {
                "kind": {"const": "billing", "type": "string"},
                "name": {"type": "string"},
            },
        }
        _convert_const_to_enum(schema)

        assert schema["properties"]["kind"]["enum"] == ["billing"]
        assert "const" not in schema["properties"]["kind"]
        assert "enum" not in schema["properties"]["name"]

    def test_const_in_anyof_variants(self) -> None:
        """const values inside anyOf variants are converted."""
        schema = {
            "anyOf": [
                {"properties": {"kind": {"const": "a"}}},
                {"properties": {"kind": {"const": "b"}}},
            ],
        }
        _convert_const_to_enum(schema)

        assert schema["anyOf"][0]["properties"]["kind"]["enum"] == ["a"]
        assert schema["anyOf"][1]["properties"]["kind"]["enum"] == ["b"]

    def test_strict_schema_applies_const_to_enum(self) -> None:
        """ensure_strict_schema converts const → enum as part of processing."""
        schema = {
            "type": "object",
            "properties": {
                "kind": {"const": "refund"},
            },
        }
        result = ensure_strict_schema(copy.deepcopy(schema))

        assert result["properties"]["kind"]["enum"] == ["refund"]
        assert "const" not in result["properties"]["kind"]

    def test_no_const_unchanged(self) -> None:
        """Schema without const is not modified."""
        schema = {"type": "string", "enum": ["a", "b"]}
        original = dict(schema)
        _convert_const_to_enum(schema)
        assert schema == original


class TestDiscriminatorRepair:
    """Tests for _try_repair_discriminators (Gemini const/Literal fix)."""

    def test_repair_capitalized_discriminator(self) -> None:
        """'Refund' → 'refund' when Literal['refund'] is expected."""
        schema = AgentOutputSchema(Union[DiscriminatedA, DiscriminatedB])

        # Gemini-style output: "A" instead of "a"
        result = schema.validate_json('{"response": {"kind": "A", "value": "test"}}')
        assert result.kind == "a"
        assert result.value == "test"

    def test_repair_uppercase_discriminator(self) -> None:
        """Full uppercase 'B' → 'b'."""
        schema = AgentOutputSchema(Union[DiscriminatedA, DiscriminatedB])

        result = schema.validate_json('{"response": {"kind": "B", "score": 42}}')
        assert result.kind == "b"
        assert result.score == 42

    def test_no_repair_needed_passes_through(self) -> None:
        """Correct casing validates without repair."""
        schema = AgentOutputSchema(Union[DiscriminatedA, DiscriminatedB])

        result = schema.validate_json('{"response": {"kind": "a", "value": "correct"}}')
        assert result.kind == "a"

    def test_unrepairable_value_raises(self) -> None:
        """Completely wrong discriminator value still raises."""
        schema = AgentOutputSchema(Union[DiscriminatedA, DiscriminatedB])

        with pytest.raises(ValueError, match="does not match"):
            schema.validate_json('{"response": {"kind": "unknown", "value": "x"}}')

    def test_repair_non_union_type_no_crash(self) -> None:
        """Repair is a no-op for non-Union types."""
        schema = AgentOutputSchema(SimpleModel)
        result = schema.validate_json('{"name": "test", "age": 25}')
        assert result.name == "test"


# ── COMPACT strips descriptions from discriminator fields ─────────────


class TestCompactStripsDiscriminatorDescriptions:
    """Regression for the const→enum ordering bug.

    ``ensure_strict_schema`` converts ``{"const": X}`` to
    ``{"enum": [X]}`` before ``_strip_schema_metadata`` runs.  The
    fixed guard must match both the original ``const`` form AND the
    converted single-element ``enum`` form so that descriptions on
    Pydantic ``Literal`` discriminator fields are stripped in COMPACT
    mode.
    """

    def test_compact_strips_description_from_literal_discriminator(self) -> None:
        """COMPACT must strip 'description' from a Literal discriminator field.

        Pydantic generates ``{"const": "bar", "description": "..."}``
        for ``kind: Literal['bar'] = Field(description='...')``.
        ``ensure_strict_schema`` converts the ``const`` to
        ``{"enum": ["bar"]}`` first.  The fix must then strip the
        description from the resulting ``enum``-of-one property.
        """
        from pydantic import TypeAdapter

        class Bar(BaseModel):
            kind: Literal["bar"] = Field(description="The discriminator type")
            value: str

        raw_schema = TypeAdapter(Bar).json_schema()
        # Raw schema has const/description on the kind field.
        result = ensure_compact_schema(copy.deepcopy(raw_schema))

        kind_prop = result.get("properties", {}).get("kind", {})
        assert "description" not in kind_prop, (
            "COMPACT mode must strip 'description' from discriminator fields even after const→enum conversion"
        )

    def test_compact_strips_description_from_enum_of_one(self) -> None:
        """_strip_schema_metadata must strip description from single-element enum."""
        from troopai.adk.schemas.utils import _strip_schema_metadata

        schema: dict = {
            "properties": {
                "kind": {
                    "enum": ["bar"],
                    "description": "The discriminator type",
                    "title": "Kind",
                },
            },
        }
        _strip_schema_metadata(schema)

        kind_prop = schema["properties"]["kind"]
        assert "description" not in kind_prop, (
            "_strip_schema_metadata must remove 'description' from enum-of-one properties"
        )
        assert "title" not in kind_prop

    def test_compact_preserves_description_on_multi_value_enum(self) -> None:
        """Descriptions on enum properties with multiple values are kept."""
        from troopai.adk.schemas.utils import _strip_schema_metadata

        schema: dict = {
            "properties": {
                "status": {
                    "enum": ["active", "inactive", "pending"],
                    "description": "Current status.",
                },
            },
        }
        _strip_schema_metadata(schema)

        assert schema["properties"]["status"]["description"] == "Current status."
