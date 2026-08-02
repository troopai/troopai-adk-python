"""Schema models for declarative agent configuration.

The Pydantic models here define the JSON config format and are the single
source of truth for the generated JSON Schema.
"""

from __future__ import annotations

from troopai.adk.types.config.agent_config import (
    CODE_ONLY_KEYS,
    AgentConfig,
    VerboseConfigRef,
)
from troopai.adk.types.config.graph_config import GraphEdgeRef, GraphNodeRef, GraphRef
from troopai.adk.types.config.guardrail_config import (
    DottedGuardrailRef,
    GuardrailsConfig,
)
from troopai.adk.types.config.llm_config import (
    AnthropicConfigBlock,
    AnthropicProviderBlock,
    GeminiConfigBlock,
    GeminiProviderBlock,
    LiteLLMConfigBlock,
    LiteLLMProviderBlock,
    LLMConfigBlock,
    LLMProviderConfig,
    LLMRetryPolicyBlock,
    OpenAIChatConfigBlock,
    OpenAIChatProviderBlock,
    OpenAIResponsesConfigBlock,
    OpenAIResponsesProviderBlock,
)
from troopai.adk.types.config.prompt_config import DynamicPromptRef
from troopai.adk.types.config.references import HandoffRef, OutputSchemaRef
from troopai.adk.types.config.swarm_config import (
    AndTerminationRef,
    ExplicitDoneTerminationRef,
    HandoffToTerminationRef,
    MaxTurnsTerminationRef,
    OrTerminationRef,
    PolicyRef,
    SwarmConfigRef,
    SwarmRef,
    TerminationRef,
)
from troopai.adk.types.config.tool_config import HostedToolRef
from troopai.adk.types.config.topology_config import AgentNodeConfig, TopologyConfig

__all__ = [
    "CODE_ONLY_KEYS",
    "AgentConfig",
    "AgentNodeConfig",
    "AndTerminationRef",
    "AnthropicConfigBlock",
    "AnthropicProviderBlock",
    "DottedGuardrailRef",
    "DynamicPromptRef",
    "ExplicitDoneTerminationRef",
    "GeminiConfigBlock",
    "GeminiProviderBlock",
    "GraphEdgeRef",
    "GraphNodeRef",
    "GraphRef",
    "GuardrailsConfig",
    "HandoffRef",
    "HandoffToTerminationRef",
    "HostedToolRef",
    "LLMConfigBlock",
    "LLMProviderConfig",
    "LLMRetryPolicyBlock",
    "LiteLLMConfigBlock",
    "LiteLLMProviderBlock",
    "MaxTurnsTerminationRef",
    "OpenAIChatConfigBlock",
    "OpenAIChatProviderBlock",
    "OpenAIResponsesConfigBlock",
    "OpenAIResponsesProviderBlock",
    "OrTerminationRef",
    "OutputSchemaRef",
    "PolicyRef",
    "SwarmConfigRef",
    "SwarmRef",
    "TerminationRef",
    "TopologyConfig",
    "VerboseConfigRef",
]
