"""Tests for the unified resolved_action() contract across guardrail levels."""

from __future__ import annotations

from troopai.adk.agents.agent_guardrails import (
    AgentGuardrailFunctionOutput,
    AgentGuardrailSeverity,
)
from troopai.adk.flows.step_guardrails import FlowStepGuardrailVerdict
from troopai.adk.tools.tool_guardrails import ToolGuardrailFunctionOutput
from troopai.adk.types.guardrails import GuardrailAction, GuardrailSpan


class TestAgentResolvedAction:
    def test_pass_by_default(self) -> None:
        assert AgentGuardrailFunctionOutput().resolved_action() is GuardrailAction.PASS

    def test_tripwire_raises(self) -> None:
        out = AgentGuardrailFunctionOutput(tripwire_triggered=True)
        assert out.resolved_action() is GuardrailAction.RAISE

    def test_severity_error_raises(self) -> None:
        out = AgentGuardrailFunctionOutput(severity=AgentGuardrailSeverity.ERROR)
        assert out.resolved_action() is GuardrailAction.RAISE

    def test_severity_warning_passes_even_with_tripwire(self) -> None:
        # Severity overrides tripwire for the halt decision, so a WARNING passes.
        out = AgentGuardrailFunctionOutput(tripwire_triggered=True, severity=AgentGuardrailSeverity.WARNING)
        assert out.resolved_action() is GuardrailAction.PASS

    def test_transform_takes_precedence(self) -> None:
        # A transform-mode verdict also sets tripwire as a fallback; the resolved
        # action is still TRANSFORM because the replacement is present.
        out = AgentGuardrailFunctionOutput(
            transformed_output="[REDACTED]",
            tripwire_triggered=True,
            changed_spans=[GuardrailSpan(start=0, end=10, reason="pii")],
        )
        assert out.resolved_action() is GuardrailAction.TRANSFORM

    def test_carrier_fields_default_to_none(self) -> None:
        out = AgentGuardrailFunctionOutput()
        assert out.transformed_output is None
        assert out.changed_spans is None


class TestToolResolvedAction:
    def test_allow_passes(self) -> None:
        verdict = ToolGuardrailFunctionOutput.allow()
        assert verdict.resolved_action() is GuardrailAction.PASS

    def test_reject_content_transforms(self) -> None:
        verdict = ToolGuardrailFunctionOutput.reject_content("blocked")
        assert verdict.resolved_action() is GuardrailAction.TRANSFORM

    def test_raise_exception_raises(self) -> None:
        verdict = ToolGuardrailFunctionOutput.raise_exception()
        assert verdict.resolved_action() is GuardrailAction.RAISE


class TestFlowResolvedAction:
    def test_allow_passes(self) -> None:
        verdict = FlowStepGuardrailVerdict.allow()
        assert verdict.resolved_action() is GuardrailAction.PASS

    def test_reject_content_raises(self) -> None:
        verdict = FlowStepGuardrailVerdict.reject_content("nope")
        assert verdict.resolved_action() is GuardrailAction.RAISE

    def test_raise_exception_raises(self) -> None:
        verdict = FlowStepGuardrailVerdict.raise_exception(ValueError("boom"))
        assert verdict.resolved_action() is GuardrailAction.RAISE
