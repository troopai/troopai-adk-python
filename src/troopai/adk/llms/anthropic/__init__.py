"""Anthropic native LLM implementation.

Calls the Anthropic Messages API directly via the ``anthropic`` SDK
— no litellm indirection.

Provider-native capabilities (web search, computer use, text editor,
etc.) are NOT exposed as framework tool classes. Pass the raw provider
JSON through ``LLMConfig.extra_body`` / ``extra_args`` instead.

Usage::

    from troopai.adk.llms.anthropic import AnthropicLLM, AnthropicConfig
    from troopai.adk.tools import function_tool


    @function_tool(name="lookup", description="Look up a record")
    def lookup(record_id: str) -> str:
        return f"Record {record_id}"


    agent = Agent(
        llm=AnthropicLLM(model="claude-sonnet-4-20250514"),
        tools=[lookup],
        llm_config=AnthropicConfig(
            temperature=0.2,
            thinking={"type": "enabled", "budget_tokens": 2048},
            auto_cache_control=True,
        ),
    )
"""

from troopai.adk.llms.anthropic.anthropic_config import AnthropicConfig
from troopai.adk.llms.anthropic.anthropic_model import AnthropicLLM

__all__ = ["AnthropicConfig", "AnthropicLLM"]
