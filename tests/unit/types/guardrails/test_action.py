"""Tests for the shared guardrail action vocabulary and observability span."""

from __future__ import annotations

import dataclasses

from troopai.adk.types.guardrails import GuardrailAction, GuardrailSpan


class TestGuardrailAction:
    def test_members(self) -> None:
        assert {action.value for action in GuardrailAction} == {
            "pass",
            "raise",
            "transform",
        }

    def test_str_enum_value_equality(self) -> None:
        # StrEnum members equal their string value, so the action serialises
        # cleanly into the audit record and tracing payload.
        assert GuardrailAction.PASS == "pass"
        assert GuardrailAction.RAISE == "raise"
        assert GuardrailAction.TRANSFORM == "transform"


class TestGuardrailSpan:
    def test_construction(self) -> None:
        span = GuardrailSpan(start=3, end=8, reason="email")
        assert span.start == 3
        assert span.end == 8
        assert span.reason == "email"

    def test_hashable(self) -> None:
        # frozen=True makes the span hashable, so it can ride inside the frozen
        # audit record's changed_spans tuple.
        span = GuardrailSpan(start=0, end=1, reason="x")
        assert hash(span) == hash(GuardrailSpan(start=0, end=1, reason="x"))
        assert len({span, GuardrailSpan(start=0, end=1, reason="x")}) == 1

    def test_fields_keyword_only(self) -> None:
        for declared in dataclasses.fields(GuardrailSpan):
            assert declared.kw_only is True
