"""Tests for Handoff.get_name() tool-name sanitization.

Provider tool-name validation (OpenAI / Anthropic) requires names to
match ``^[a-zA-Z0-9_-]+$``. The auto-generated handoff tool name flows
verbatim through ``to_tool()`` to the provider wire format, so any
agent name with characters outside that set (``/``, ``#``, ``.``,
parentheses, ``&``, non-ASCII) would otherwise produce a tool name the
provider API rejects.
"""

from __future__ import annotations

import re
from unittest.mock import MagicMock

import pytest

from troopai.adk.handoffs.handoff import HANDOFF_TOOL_PREFIX, Handoff

# Provider-side tool-name validation pattern (OpenAI / Anthropic).
_VALID_TOOL_NAME = re.compile(r"^[a-zA-Z0-9_-]+$")


def _mock_agent(name: str) -> MagicMock:
    """Create a mock Agent with the required name/description attributes."""
    agent = MagicMock()
    agent.name = name
    agent.description = None
    return agent


class TestHandoffGetNameSanitization:
    """Auto-generated handoff tool names must be provider-wire-safe."""

    @pytest.mark.parametrize(
        ("agent_name", "expected"),
        [
            ("Refund/Billing Agent", "transfer_to_refund_billing_agent"),
            ("Agent #1", "transfer_to_agent_1"),
            ("Acct.Mgmt", "transfer_to_acct_mgmt"),
            ("Agent (Premium)", "transfer_to_agent_premium"),
            ("R&D Agent", "transfer_to_r_d_agent"),
            # Plain alphanumeric + space names stay unchanged in shape.
            ("Refunds", "transfer_to_refunds"),
            ("Customer Support", "transfer_to_customer_support"),
            # camelCase / PascalCase split like as_tool() names.
            ("CustomerSupport", "transfer_to_customer_support"),
        ],
    )
    def test_get_name_sanitizes_invalid_characters(self, agent_name: str, expected: str) -> None:
        """Names with /, #, ., parens, & are stripped to a wire-safe tool name."""
        h = Handoff(target=_mock_agent(agent_name))

        name = h.get_name()

        assert name == expected
        assert _VALID_TOOL_NAME.match(name) is not None, f"{name!r} is not a valid provider tool name"

    def test_to_tool_carries_sanitized_name(self) -> None:
        """The sanitized name flows through to the emitted FunctionTool."""
        h = Handoff(target=_mock_agent("Refund/Billing Agent"))

        tool = h.to_tool()

        assert tool.name == "transfer_to_refund_billing_agent"
        assert _VALID_TOOL_NAME.match(tool.name) is not None

    def test_explicit_name_is_not_resniffed(self) -> None:
        """An explicit Handoff(name=...) is returned verbatim (developer-owned)."""
        h = Handoff(target=_mock_agent("Refund/Billing Agent"), name="my_custom_tool")

        assert h.get_name() == "my_custom_tool"

    def test_prefix_constant_is_used(self) -> None:
        """The generated name keeps the documented transfer_to_ prefix."""
        h = Handoff(target=_mock_agent("Refunds"))

        assert h.get_name().startswith(HANDOFF_TOOL_PREFIX)


class TestHandoffGetNameNonLatin:
    """Non-Latin target names must stay distinct and provider-wire-safe.

    A name written entirely in a non-Latin script sanitises to the empty
    string, so a naive ``transfer_to_{snake}`` collapses every such target
    to a bare ``transfer_to_`` — colliding across distinct agents. A stable
    per-name digest keeps distinct targets distinct.
    """

    def test_distinct_non_latin_names_yield_distinct_tool_names(self) -> None:
        n1 = Handoff(target=_mock_agent("退款代理")).get_name()
        n2 = Handoff(target=_mock_agent("账单代理")).get_name()

        assert n1 != n2
        # Neither collapses to the bare prefix.
        assert n1 != HANDOFF_TOOL_PREFIX
        assert n2 != HANDOFF_TOOL_PREFIX
        assert n1.startswith(HANDOFF_TOOL_PREFIX)
        assert n2.startswith(HANDOFF_TOOL_PREFIX)
        # Still valid provider tool names ([a-zA-Z0-9_-] only).
        assert _VALID_TOOL_NAME.match(n1) is not None
        assert _VALID_TOOL_NAME.match(n2) is not None

    def test_non_latin_name_is_deterministic(self) -> None:
        # The digest is stable: the same name always maps to the same tool
        # name, so the tool identity is reproducible across constructions.
        assert Handoff(target=_mock_agent("退款代理")).get_name() == Handoff(target=_mock_agent("退款代理")).get_name()
