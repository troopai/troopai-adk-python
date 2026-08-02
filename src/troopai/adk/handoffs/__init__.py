"""Handoff system for routing between agents.

Code-orchestrated (deterministic Intent routing):
    agent.handoffs = handoff_route(
        (RefundIntent, refunds_agent),
        (BillingIntent, billing_agent),
        otherwise=general_agent,
    )

LLM-orchestrated (agents as transfer tools):
    agent.handoffs = [refunds_agent, billing_agent]
    # or with config:
    agent.handoffs = [
        handoff(target=refunds_agent, description="Handle refunds"),
        billing_agent,
    ]
"""

from __future__ import annotations

from troopai.adk.handoffs.handoff import HANDOFF_TOOL_PREFIX, Handoff, handoff
from troopai.adk.handoffs.handoff_config import HandoffConfig
from troopai.adk.handoffs.handoff_input_data import HandoffInputData
from troopai.adk.handoffs.handoff_prompt import (
    RECOMMENDED_PROMPT_PREFIX,
    prompt_with_handoff_instructions,
)
from troopai.adk.handoffs.handoff_route import HandoffRoute, handoff_route
from troopai.adk.handoffs.handoff_strategy import HandoffStrategy
from troopai.adk.handoffs.handoff_target import (
    HandoffEnabledCallback,
    HandoffInputFilter,
    HandoffTarget,
    OnHandoffCallback,
    OnHandoffWithData,
    OnHandoffWithInput,
    OnHandoffWithoutInput,
    THandoffInput,
)

__all__ = [
    # Constants
    "HANDOFF_TOOL_PREFIX",
    # Prompt helpers
    "RECOMMENDED_PROMPT_PREFIX",
    # Core classes
    "Handoff",
    # Configuration
    "HandoffConfig",
    # Type aliases
    "HandoffEnabledCallback",
    "HandoffInputData",
    "HandoffInputFilter",
    "HandoffRoute",
    "HandoffStrategy",
    "HandoffTarget",
    "OnHandoffCallback",
    "OnHandoffWithData",
    "OnHandoffWithInput",
    "OnHandoffWithoutInput",
    "THandoffInput",
    # Factory functions
    "handoff",
    "handoff_route",
    "prompt_with_handoff_instructions",
]
