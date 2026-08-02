"""Agent module for TroopAI Agents ADK.

This module provides the core Agent class and agent-level guardrails for
building autonomous AI agent.

Example:
    from troopai.adk.agents import (
        Agent,
        AgentGuardrails,
        AgentGuardrailFunctionOutput,
        AgentInputGuardrailData,
        agent_input_guardrail,
    )

    @agent_input_guardrail
    async def block_secrets(
        data: AgentInputGuardrailData,
    ) -> AgentGuardrailFunctionOutput:
        triggered = "password" in str(data.user_prompt).lower()
        return AgentGuardrailFunctionOutput(tripwire_triggered=triggered)

    # Create an agent with guardrails
    agent = Agent(
        name="Customer Support",
        system_prompt="Help customers with their questions.",
        guardrails=AgentGuardrails(input=[block_secrets]),
    )
"""

from troopai.adk.agents.agent import Agent, BaseAgent
from troopai.adk.agents.agent_guardrails import (
    AgentGuardrailFunctionOutput,
    AgentGuardrailResults,
    AgentGuardrails,
    AgentGuardrailSeverity,
    AgentGuardrailTimeoutInfo,
    AgentInputGuardrail,
    AgentInputGuardrailData,
    AgentInputGuardrailResult,
    AgentOutputGuardrail,
    AgentOutputGuardrailData,
    AgentOutputGuardrailResult,
    AgentTimeoutPolicy,
    agent_input_guardrail,
    agent_output_guardrail,
)
from troopai.adk.agents.middleware import Middleware

__all__ = [
    # Agent
    "Agent",
    # Guardrail classes
    "AgentGuardrailFunctionOutput",
    "AgentGuardrailResults",
    # Guardrail enums
    "AgentGuardrailSeverity",
    "AgentGuardrailTimeoutInfo",
    # Per-Agent guardrail config
    "AgentGuardrails",
    "AgentInputGuardrail",
    "AgentInputGuardrailData",
    "AgentInputGuardrailResult",
    "AgentOutputGuardrail",
    "AgentOutputGuardrailData",
    "AgentOutputGuardrailResult",
    "AgentTimeoutPolicy",
    "BaseAgent",
    # Middleware config
    "Middleware",
    # Guardrail decorators
    "agent_input_guardrail",
    "agent_output_guardrail",
]
