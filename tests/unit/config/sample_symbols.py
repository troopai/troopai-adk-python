"""Importable symbols referenced by config-resolution tests.

These exist at a stable dotted path so reference-resolution tests can point
``ref`` strings at real objects (a FunctionTool, an output-schema class) and
at a non-tool symbol for the negative cases.
"""

from __future__ import annotations

from pydantic import BaseModel

from troopai.adk.agents.agent_guardrails import (
    AgentGuardrailFunctionOutput,
    AgentInputGuardrailData,
    AgentOutputGuardrailData,
    agent_input_guardrail,
    agent_output_guardrail,
)
from troopai.adk.prompts.system_prompt import DynamicSystemPromptData
from troopai.adk.tools import function_tool


@function_tool
def sample_tool(query: str) -> str:
    """Echo the query back (a trivial tool for tests)."""
    return query


class SampleOutput(BaseModel):
    """A structured output schema for tests."""

    answer: str


NOT_A_TOOL = 42
"""A non-tool, non-class symbol for negative-path tests."""


def always_true(result: object) -> bool:
    """A trivial edge-condition predicate for graph `when` tests."""
    return True


@agent_input_guardrail
def my_input_guard(data: AgentInputGuardrailData) -> AgentGuardrailFunctionOutput:
    """Sample input guardrail (never trips) for guardrail-ref tests."""
    return AgentGuardrailFunctionOutput(tripwire_triggered=False)


@agent_output_guardrail
def my_output_guard(data: AgentOutputGuardrailData) -> AgentGuardrailFunctionOutput:
    """Sample output guardrail (never trips) for guardrail-ref tests."""
    return AgentGuardrailFunctionOutput(tripwire_triggered=False)


def build_prompt(data: DynamicSystemPromptData) -> str:
    """Sample dynamic system prompt for dynamic-prompt-ref tests."""
    return "Dynamic prompt."
