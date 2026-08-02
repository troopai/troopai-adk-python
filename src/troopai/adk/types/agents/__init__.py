"""Agent-related type definitions.

Provides types for the ``Agent.as_tool()`` feature: default input schema
and callback type aliases for output extraction and input building.
"""

from troopai.adk.types.agents.agent_as_tool_types import (
    AgentToolInput,
    AgentToolInputBuilder,
    AgentToolOutputExtractor,
)

__all__ = [
    "AgentToolInput",
    "AgentToolInputBuilder",
    "AgentToolOutputExtractor",
]
