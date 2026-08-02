"""Tests for reference-bearing config fields: llm, tools, output_schema.

A tool entry is a bare dotted ``ref`` string resolving to a FunctionTool.
``output_schema`` is a bare string (default enforcement) or ``{ref,
enforcement}``. ``llm`` as a string is a model-name passed straight through.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from troopai.adk.config import build_agent
from troopai.adk.config.resolver import resolve_function_tool, resolve_output_schema
from troopai.adk.exceptions import ConfigResolutionError
from troopai.adk.schemas import SchemaEnforcement
from troopai.adk.tools import FunctionTool
from troopai.adk.types.config import AgentConfig
from troopai.adk.types.config.references import HandoffRef, OutputSchemaRef

from .sample_symbols import SampleOutput, sample_tool

_TOOL_REF = "tests.unit.config.sample_symbols:sample_tool"
_SCHEMA_REF = "tests.unit.config.sample_symbols:SampleOutput"
_NON_TOOL_REF = "tests.unit.config.sample_symbols:NOT_A_TOOL"


def _build(data: dict[str, object]):
    return build_agent(AgentConfig.model_validate(data))


class TestLLMReference:
    def test_string_model_passthrough(self) -> None:
        agent = _build({"name": "a", "system_prompt": "p", "llm": "gpt-4o"})
        assert agent.llm == "gpt-4o"

    def test_absent_llm_is_none(self) -> None:
        agent = _build({"name": "a", "system_prompt": "p"})
        assert agent.llm is None


class TestToolReferences:
    def test_single_function_tool(self) -> None:
        agent = _build({"name": "a", "system_prompt": "p", "tools": [_TOOL_REF]})
        assert len(agent.tools) == 1
        assert agent.tools[0] is sample_tool

    def test_absent_tools_is_empty(self) -> None:
        agent = _build({"name": "a", "system_prompt": "p"})
        assert len(agent.tools) == 0

    def test_non_tool_ref_raises(self) -> None:
        with pytest.raises(ConfigResolutionError):
            _build({"name": "a", "system_prompt": "p", "tools": [_NON_TOOL_REF]})

    def test_resolve_function_tool_returns_tool(self) -> None:
        assert isinstance(resolve_function_tool(_TOOL_REF), FunctionTool)

    def test_resolve_function_tool_rejects_non_tool(self) -> None:
        with pytest.raises(ConfigResolutionError):
            resolve_function_tool(_NON_TOOL_REF)


class TestOutputSchemaReference:
    def test_bare_string_defaults_strict(self) -> None:
        schema = resolve_output_schema(_SCHEMA_REF, SchemaEnforcement.STRICT)
        assert schema.output_schema is SampleOutput

    def test_object_form_with_enforcement(self) -> None:
        agent = _build(
            {
                "name": "a",
                "system_prompt": "p",
                "output_schema": {"ref": _SCHEMA_REF, "enforcement": "normalized"},
            }
        )
        assert agent.output_schema is not None

    def test_bare_string_on_agent(self) -> None:
        agent = _build({"name": "a", "system_prompt": "p", "output_schema": _SCHEMA_REF})
        assert agent.output_schema is not None

    def test_absent_output_schema_is_none(self) -> None:
        agent = _build({"name": "a", "system_prompt": "p"})
        assert agent.output_schema is None

    def test_non_class_ref_raises(self) -> None:
        with pytest.raises(ConfigResolutionError):
            resolve_output_schema(_NON_TOOL_REF, SchemaEnforcement.STRICT)


class TestRefMinLength:
    def test_output_schema_ref_empty_rejected(self) -> None:
        with pytest.raises(ValidationError):
            OutputSchemaRef.model_validate({"ref": ""})

    def test_handoff_ref_empty_target_rejected(self) -> None:
        with pytest.raises(ValidationError):
            HandoffRef.model_validate({"target": ""})
