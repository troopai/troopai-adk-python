"""Attribute + type-richness contract for FunctionTool, Handoff, and HandoffConfig.

Covered behaviour:
1. ``TokenBudget`` construction invariants + ``drop_policy`` field.
2. ``HandoffCollapseMode`` enum values.
3. ``HandoffConfig.__post_init__`` normalization of scalar inputs
   (``int`` → ``TokenBudget``; ``bool`` → ``HandoffCollapseMode``).
4. ``HandoffConfig`` rejecting non-int / non-TokenBudget / bool
   budget shapes that would otherwise coerce silently.
5. ``FunctionTool.metadata`` + ``FunctionTool.tool_namespace``.
6. ``Handoff.metadata`` + ``Handoff.agent_name``.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from troopai.adk.handoffs.handoff import Handoff
from troopai.adk.handoffs.handoff_collapse_mode import HandoffCollapseMode
from troopai.adk.handoffs.handoff_config import HandoffConfig
from troopai.adk.tools import FunctionTool, function_tool
from troopai.adk.tools.token_budget import TokenBudget


class TestTokenBudget:
    def test_default_drop_policy_is_preserve_system(self) -> None:
        b = TokenBudget(max_tokens=10_000)
        assert b.drop_policy == "preserve_system"

    def test_custom_drop_policy(self) -> None:
        b = TokenBudget(max_tokens=10_000, drop_policy="oldest_first")
        assert b.drop_policy == "oldest_first"

    def test_zero_max_tokens_rejected(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            TokenBudget(max_tokens=0)

    def test_negative_max_tokens_rejected(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            TokenBudget(max_tokens=-100)

    def test_frozen(self) -> None:
        import dataclasses

        b = TokenBudget(max_tokens=10_000)
        with pytest.raises(dataclasses.FrozenInstanceError):
            b.max_tokens = 5_000  # type: ignore[misc]


class TestHandoffCollapseMode:
    def test_enum_values(self) -> None:
        assert HandoffCollapseMode.OFF.value == "off"
        assert HandoffCollapseMode.SYSTEM_MESSAGE.value == "system_message"
        assert HandoffCollapseMode.USER_MESSAGE.value == "user_message"


class TestHandoffConfigCoercion:
    def test_default_budget_is_typed(self) -> None:
        c = HandoffConfig()
        assert isinstance(c.budget, TokenBudget)
        assert c.budget.max_tokens == 20_000
        assert c.budget.drop_policy == "preserve_system"

    def test_default_collapse_is_off(self) -> None:
        c = HandoffConfig()
        assert c.collapse == HandoffCollapseMode.OFF

    def test_int_budget_coerced_to_token_budget(self) -> None:
        c = HandoffConfig(budget=5_000)
        assert isinstance(c.budget, TokenBudget)
        assert c.budget.max_tokens == 5_000
        assert c.budget.drop_policy == "preserve_system"

    def test_typed_token_budget_preserved(self) -> None:
        b = TokenBudget(max_tokens=5_000, drop_policy="oldest_first")
        c = HandoffConfig(budget=b)
        assert c.budget is b

    def test_none_budget_preserved(self) -> None:
        c = HandoffConfig(budget=None)
        assert c.budget is None

    def test_bool_budget_rejected(self) -> None:
        # bool is a subclass of int — without an explicit guard a True
        # would silently coerce to TokenBudget(max_tokens=1).
        with pytest.raises(TypeError, match="cannot be bool"):
            HandoffConfig(budget=True)  # type: ignore[arg-type]

    def test_true_collapse_coerced_to_system_message(self) -> None:
        c = HandoffConfig(collapse=True)
        assert c.collapse == HandoffCollapseMode.SYSTEM_MESSAGE

    def test_false_collapse_coerced_to_off(self) -> None:
        c = HandoffConfig(collapse=False)
        assert c.collapse == HandoffCollapseMode.OFF

    def test_typed_collapse_mode_preserved(self) -> None:
        c = HandoffConfig(collapse=HandoffCollapseMode.USER_MESSAGE)
        assert c.collapse == HandoffCollapseMode.USER_MESSAGE


class TestFunctionToolAttributes:
    def test_default_metadata_empty(self) -> None:
        @function_tool(name="t", description="t")
        def t(x: int) -> int:
            return x

        assert dict(t.metadata) == {}

    def test_default_tool_namespace_none(self) -> None:
        @function_tool(name="t", description="t")
        def t(x: int) -> int:
            return x

        assert t.tool_namespace is None

    def test_metadata_attached_via_function_tool_constructor(self) -> None:
        # Direct construction supports metadata; the decorator helper
        # doesn't expose every field but FunctionTool(...) does.
        tool = FunctionTool(
            name="t",
            description="t",
            schema={"type": "object", "properties": {}},
            metadata={"owner": "platform", "cost_tier": "expensive"},
        )
        assert tool.metadata["owner"] == "platform"
        assert tool.metadata["cost_tier"] == "expensive"

    def test_tool_namespace_attached(self) -> None:
        tool = FunctionTool(
            name="search",
            description="t",
            schema={"type": "object", "properties": {}},
            tool_namespace="retrieval",
        )
        assert tool.tool_namespace == "retrieval"


class TestHandoffAttributes:
    def test_default_metadata_empty(self) -> None:
        agent = MagicMock()
        agent.name = "billing"
        agent.description = None
        h = Handoff(target=agent)
        assert dict(h.metadata) == {}

    def test_metadata_attached(self) -> None:
        agent = MagicMock()
        agent.name = "billing"
        agent.description = None
        h = Handoff(target=agent, metadata={"team": "finance"})
        assert h.metadata["team"] == "finance"

    def test_agent_name_property(self) -> None:
        agent = MagicMock()
        agent.name = "refunds"
        agent.description = None
        h = Handoff(target=agent)
        assert h.agent_name == "refunds"
