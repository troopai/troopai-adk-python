"""Regression tests for schema-normalization and output-schema sweep fixes.

- ``normalize_schema`` must not synthesise a ``"Property <name>"`` description
  (a framework-added token the developer never opted into).
- ``normalize_schema`` must preserve a boolean JSON Schema property
  (``true`` / ``false``) instead of clobbering it to a string type.
- ``AgentOutputSchema.json_schema()`` must return an independent copy so a
  caller mutating the schema in place cannot corrupt the cached dict.
"""

from __future__ import annotations

from pydantic import BaseModel

from troopai.adk.schemas.agent_output_schema import AgentOutputSchema
from troopai.adk.schemas.utils import normalize_schema


class TestNormalizeNoSyntheticDescription:
    def test_untyped_property_gets_type_but_no_description(self) -> None:
        out = normalize_schema({"type": "object", "properties": {"x": {}}})
        assert out["properties"]["x"] == {"type": "string"}

    def test_typed_property_gets_no_injected_description(self) -> None:
        out = normalize_schema({"type": "object", "properties": {"name": {"type": "string"}}})
        assert "description" not in out["properties"]["name"]

    def test_explicit_description_is_preserved(self) -> None:
        out = normalize_schema(
            {"type": "object", "properties": {"name": {"type": "string", "description": "the name"}}}
        )
        assert out["properties"]["name"]["description"] == "the name"

    def test_malformed_non_dict_property_has_no_description(self) -> None:
        out = normalize_schema({"type": "object", "properties": {"weird": 123}})
        assert out["properties"]["weird"] == {"type": "string"}


class TestNormalizePreservesBooleanSchema:
    def test_boolean_true_property_preserved(self) -> None:
        out = normalize_schema({"type": "object", "properties": {"anything": True}})
        assert out["properties"]["anything"] is True

    def test_boolean_false_property_preserved(self) -> None:
        out = normalize_schema({"type": "object", "properties": {"forbidden": False}})
        assert out["properties"]["forbidden"] is False


class TestJsonSchemaReturnsIndependentCopy:
    def test_mutating_returned_schema_does_not_corrupt_cache(self) -> None:
        class Model(BaseModel):
            value: int

        schema = AgentOutputSchema(Model)
        first = schema.json_schema()
        first["properties"]["INJECTED"] = {"type": "string"}
        first["__mutated__"] = True

        second = schema.json_schema()
        assert "INJECTED" not in second.get("properties", {})
        assert "__mutated__" not in second
