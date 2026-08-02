"""Tests for :mod:`troopai.adk.workflows.temporal.serialization`.

Covers:
- DataConverter roundtrips a dataclass payload.
- DataConverter roundtrips ``None``.
"""

from __future__ import annotations

import dataclasses

import pytest

temporalio = pytest.importorskip("temporalio")

from temporalio.converter import DataConverter

from troopai.adk.workflows.temporal.serialization import (
    build_troopai_data_converter,
    config_from_json_dict,
    config_to_json_dict,
)


@dataclasses.dataclass
class _SamplePayload:
    name: str
    value: int


class TestTroopAIPayloadConverterRoundtrips:
    def test_troopai_payload_converter_roundtrips_dataclass(self) -> None:
        converter: DataConverter = build_troopai_data_converter()
        original = _SamplePayload(name="hello", value=42)

        payloads = converter.payload_converter.to_payloads([original])
        restored = converter.payload_converter.from_payloads(payloads, [type(original)])

        assert len(restored) == 1
        roundtripped = restored[0]
        assert isinstance(roundtripped, _SamplePayload)
        assert roundtripped.name == original.name
        assert roundtripped.value == original.value

    def test_troopai_payload_converter_handles_none(self) -> None:
        converter: DataConverter = build_troopai_data_converter()

        payloads = converter.payload_converter.to_payloads([None])
        restored = converter.payload_converter.from_payloads(payloads, [type(None)])

        assert len(restored) == 1
        assert restored[0] is None


class TestConfigJsonRoundtrip:
    """config_to_json_dict / config_from_json_dict survive json.dumps round-trip.

    Regression: the durable-LLM path json.dumps'd the raw LLMConfig fields, so a
    set retry_policy (frozenset retry_on) or an httpx.Timeout made json.dumps
    raise and crashed the workflow at its first LLM turn.
    """

    def test_plain_config_roundtrips(self) -> None:
        import json

        from troopai.adk.llms.llm_config import LLMConfig

        cfg = LLMConfig(temperature=0.7, max_output_tokens=256)
        out = config_from_json_dict(json.loads(json.dumps(config_to_json_dict(cfg))))
        assert out.temperature == 0.7
        assert out.max_output_tokens == 256

    def test_retry_policy_roundtrips(self) -> None:
        import json

        from troopai.adk.llms.llm_config import LLMConfig
        from troopai.adk.types.llms.retry_policy import LLMRetryPolicy

        cfg = LLMConfig(
            retry_policy=LLMRetryPolicy(max_retries=5, retry_on=frozenset(["rate_limit", "timeout"])),
        )
        # A raw json.dumps of the frozenset would raise; config_to_json_dict coerces it.
        out = config_from_json_dict(json.loads(json.dumps(config_to_json_dict(cfg))))
        assert out.retry_policy is not None
        assert out.retry_policy.max_retries == 5
        assert out.retry_policy.retry_on == frozenset(["rate_limit", "timeout"])

    def test_httpx_timeout_roundtrips(self) -> None:
        import json

        import httpx

        from troopai.adk.llms.llm_config import LLMConfig

        cfg = LLMConfig(timeout=httpx.Timeout(connect=5.0, read=10.0, write=10.0, pool=2.0))
        out = config_from_json_dict(json.loads(json.dumps(config_to_json_dict(cfg))))
        assert isinstance(out.timeout, httpx.Timeout)
        assert out.timeout.read == 10.0

    def test_float_timeout_passes_through(self) -> None:
        import json

        from troopai.adk.llms.llm_config import LLMConfig

        cfg = LLMConfig(timeout=30.0)
        out = config_from_json_dict(json.loads(json.dumps(config_to_json_dict(cfg))))
        assert out.timeout == 30.0


# ---------------------------------------------------------------------------
# Tool + output-schema transport across the activity boundary
# ---------------------------------------------------------------------------

from pydantic import BaseModel

from troopai.adk.schemas.agent_output_schema import AgentOutputSchema
from troopai.adk.tools import function_tool
from troopai.adk.workflows.temporal.serialization import (
    ForwardedOutputSchema,
    output_schema_from_json_dict,
    output_schema_to_json_dict,
    tool_from_json_dict,
    tool_to_json_dict,
)


@function_tool(name="lookup", description="Look up a record by id")
def _lookup(record_id: str) -> str:
    """Return a record string for the given id."""
    return f"record {record_id}"


class TestToolJsonRoundtrip:
    """A FunctionTool's LLM-facing definition survives the activity boundary."""

    def test_definition_carries_name_description_parameters(self) -> None:
        definition = tool_to_json_dict(_lookup)
        assert definition is not None
        assert definition["name"] == "lookup"
        assert definition["description"] == "Look up a record by id"
        # record_id must appear in the serialized parameter schema
        assert "record_id" in json_dumps_keys(definition["parameters"])

    def test_roundtrip_preserves_wire_schema(self) -> None:
        rebuilt = tool_from_json_dict(tool_to_json_dict(_lookup))  # type: ignore[arg-type]
        assert rebuilt.name == "lookup"
        assert rebuilt.description == "Look up a record by id"
        # The reconstructed (definition-only) tool produces the same JSON schema
        # the model would have seen from the original.
        assert rebuilt.get_json_schema() == _lookup.get_json_schema()
        # It is definition-only — no executable crosses the boundary.
        assert rebuilt.on_invoke is None

    def test_hosted_tool_is_not_forwardable(self) -> None:
        from troopai.adk.tools.hosted.code_execution_tool import CodeExecutionTool

        # Provider-hosted tools cannot be reconstructed as function definitions;
        # the serializer returns None so the caller skips (and logs) them.
        assert tool_to_json_dict(CodeExecutionTool()) is None


class _Weather(BaseModel):
    city: str
    temp_c: float


class TestOutputSchemaJsonRoundtrip:
    """A structured-output schema survives the activity boundary."""

    def test_roundtrip_preserves_schema_strict_and_name(self) -> None:
        original = AgentOutputSchema(_Weather)
        rebuilt = output_schema_from_json_dict(output_schema_to_json_dict(original))

        assert isinstance(rebuilt, ForwardedOutputSchema)
        assert rebuilt.json_schema() == original.json_schema()
        assert rebuilt.is_strict_json_schema() == original.is_strict_json_schema()
        assert rebuilt.name() == original.name()
        assert rebuilt.is_plain_text() is False

    def test_validate_json_parses_constrained_output(self) -> None:
        rebuilt = output_schema_from_json_dict(output_schema_to_json_dict(AgentOutputSchema(_Weather)))
        # The provider already constrained the output; validate_json just parses.
        assert rebuilt.validate_json('{"city": "Paris", "temp_c": 18.0}') == {
            "city": "Paris",
            "temp_c": 18.0,
        }


def json_dumps_keys(obj: object) -> str:
    """Flatten a nested schema to a string for cheap key-presence assertions."""
    import json

    return json.dumps(obj)
