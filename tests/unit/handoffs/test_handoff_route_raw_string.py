"""Regression guard: ``HandoffRoute.resolve`` must tolerate a raw ``str``.

Scenario: ``turn_resolution.py`` falls back to the raw LLM response
string when ``AgentOutputSchema.validate_json`` raises. The raw string
is then handed straight to ``HandoffRoute.resolve``. Every branch in
``resolve`` MUST guard with ``isinstance`` before touching attributes —
otherwise a malformed-JSON LLM response becomes a run-ending
``AttributeError``.

These tests feed a raw ``str`` in and assert one of two safe outcomes:

1. A defined fallback (``.otherwise``) matches and is returned, OR
2. ``UnhandledIntentError`` is raised (type-safe termination).

Neither ``AttributeError`` nor ``TypeError`` is acceptable.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from troopai.adk.handoffs.handoff_route import (
    HandoffRoute,
    UnhandledIntentError,
    handoff_route,
)
from troopai.adk.types.intents import Intent


class _RefundIntent(Intent):
    reason: str = ""


class _BillingIntent(Intent):
    amount: float = 0.0


def _mock_agent(name: str = "test_agent") -> MagicMock:
    agent = MagicMock()
    agent.name = name
    agent.description = None
    return agent


class TestHandoffRouteRawString:
    def test_raw_string_with_otherwise_returns_fallback(self) -> None:
        # Fallback path: .otherwise handles the non-Intent input cleanly.
        refunds = _mock_agent("refunds")
        general = _mock_agent("general")
        route = handoff_route(
            (_RefundIntent, refunds),
            otherwise=general,
        )

        result = asyncio.run(route.resolve("raw LLM string that failed validation"))  # type: ignore[arg-type]

        assert result is not None
        assert result.target is general

    def test_raw_string_without_otherwise_raises_unhandled(self) -> None:
        # No fallback: must raise UnhandledIntentError, NOT AttributeError.
        refunds = _mock_agent("refunds")
        route: HandoffRoute[MagicMock, None] = HandoffRoute()
        route.when(_RefundIntent).to(refunds)

        with pytest.raises(UnhandledIntentError, match="str"):
            asyncio.run(route.resolve("raw LLM string"))  # type: ignore[arg-type]

    def test_raw_string_does_not_match_any_typed_rule(self) -> None:
        # Ensures isinstance(intent, intent_types) correctly rejects str
        # for ALL registered rule entries (not just the first one).
        refunds = _mock_agent("refunds")
        billing = _mock_agent("billing")
        general = _mock_agent("general")
        route = handoff_route(
            (_RefundIntent, refunds),
            (_BillingIntent, billing),
            otherwise=general,
        )

        result = asyncio.run(route.resolve("arbitrary string"))  # type: ignore[arg-type]

        # Must have landed on otherwise — not on either typed rule.
        assert result is not None
        assert result.target is general
        assert result.target is not refunds
        assert result.target is not billing

    def test_empty_string_also_safe(self) -> None:
        # Edge case: empty string must not short-circuit any isinstance check.
        general = _mock_agent("general")
        route: HandoffRoute[MagicMock, None] = HandoffRoute()
        route.otherwise(general)

        result = asyncio.run(route.resolve(""))  # type: ignore[arg-type]

        assert result is not None
        assert result.target is general

    def test_unhandled_error_message_uses_type_name_not_attribute_access(self) -> None:
        # The error formatter uses type(intent).__name__, which works for
        # any object including raw str. If someone refactors it to
        # intent.__class__.__name__ or intent.name the test still passes,
        # but if they introduce intent.some_attr it crashes BEFORE the
        # message is built and this assertion catches the regression.
        route: HandoffRoute[MagicMock, None] = HandoffRoute()
        route.when(_RefundIntent).to(_mock_agent("refunds"))

        with pytest.raises(UnhandledIntentError) as exc_info:
            asyncio.run(route.resolve("some-raw-string"))  # type: ignore[arg-type]

        # 'str' is the type name of a raw string — proves the formatter
        # went through type(intent).__name__ without crashing.
        assert "str" in str(exc_info.value)
