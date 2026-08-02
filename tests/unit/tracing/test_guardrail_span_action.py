"""Tests for the guardrail tracing span's action field."""

from __future__ import annotations

from troopai.adk.types.guardrails import GuardrailAction
from troopai.adk.types.tracing.span_data import GuardrailSpanData


class TestGuardrailSpanDataAction:
    def test_action_absent_from_export_by_default(self) -> None:
        exported = GuardrailSpanData(name="g", triggered=True).export()
        assert "action" not in exported
        assert exported == {"type": "guardrail", "name": "g", "triggered": True}

    def test_action_included_when_set(self) -> None:
        exported = GuardrailSpanData(name="g", triggered=True, action=GuardrailAction.RAISE).export()
        assert exported["action"] == GuardrailAction.RAISE
