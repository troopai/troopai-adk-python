"""Assemble an ``Agent`` from a validated ``AgentConfig``.

Each config field is translated into an explicit ``Agent(...)`` keyword
argument. Optional fields the config did not declare are left at the
``Agent`` dataclass's own defaults — the assembler never injects a value the
operator did not choose.
"""

from __future__ import annotations

import logging

from troopai.adk.agents.agent import Agent
from troopai.adk.agents.agent_guardrails import AgentGuardrails
from troopai.adk.config.guardrails import build_guardrails
from troopai.adk.config.hosted_tools import build_hosted_tool
from troopai.adk.config.providers import PROVIDER_REGISTRY, build_agnostic_config
from troopai.adk.config.resolver import resolve_dynamic_prompt, resolve_function_tool, resolve_output_schema
from troopai.adk.exceptions import ConfigResolutionError
from troopai.adk.llms.llm import LLM
from troopai.adk.llms.llm_config import LLMConfig
from troopai.adk.prompts.system_prompt import DynamicSystemPrompt, SystemPrompt
from troopai.adk.schemas import SchemaEnforcement
from troopai.adk.schemas.agent_output_schema import AgentOutputSchema
from troopai.adk.skills.activation import SkillActivation
from troopai.adk.tools import Tool
from troopai.adk.types.config.agent_config import AgentConfig
from troopai.adk.types.config.prompt_config import DynamicPromptRef
from troopai.adk.types.config.tool_config import HostedToolRef
from troopai.adk.verbose.config import VerboseConfig

logger = logging.getLogger(__name__)


def _build_tool(item: str | HostedToolRef) -> Tool:
    """Resolve one ``tools`` entry to a ``Tool``.

    A string is a dotted ``FunctionTool`` reference; a ``HostedToolRef`` is a
    provider-hosted tool built from the registry.
    """
    if isinstance(item, str):
        return resolve_function_tool(item)
    return build_hosted_tool(item)


def _build_system_prompt(
    prompt: str | SystemPrompt | DynamicPromptRef,
) -> str | SystemPrompt | DynamicSystemPrompt:
    """Resolve the config system prompt; a ``DynamicPromptRef`` → callable."""
    if isinstance(prompt, DynamicPromptRef):
        return resolve_dynamic_prompt(prompt.dynamic)
    return prompt


def build_llm(config: AgentConfig) -> tuple[str | LLM | None, LLMConfig | None]:
    """Resolve the config's ``llm`` / ``llm_config`` to Agent constructor values.

    A string (or absent) ``llm`` passes through, paired with the agnostic
    ``llm_config`` block. A typed provider block dispatches to its factory,
    which returns a concrete ``LLM`` and its runtime config.

    Args:
        config: A validated :class:`AgentConfig`.

    Returns:
        ``(llm, llm_config)`` for ``Agent(llm=…, llm_config=…)``.

    Raises:
        ConfigResolutionError: If a provider block names a provider with no
            registered factory, if the provider's optional dependency is not
            installed, or if a config-block field has no matching runtime
            config field.
    """
    if config.llm is None or isinstance(config.llm, str):
        return config.llm, build_agnostic_config(config.llm_config)

    factory = PROVIDER_REGISTRY.get(config.llm.provider)
    if factory is None:
        available = ", ".join(sorted(PROVIDER_REGISTRY))
        raise ConfigResolutionError(
            f"No factory registered for LLM provider {config.llm.provider!r}. Available: {available}."
        )
    try:
        return factory(config.llm)
    except ImportError as exc:
        raise ConfigResolutionError(
            f"The {config.llm.provider!r} provider requires its optional dependency, which is not "
            f"installed ({exc}). Install the matching provider package to load this config."
        ) from exc


def build_agent(config: AgentConfig) -> Agent:
    """Build an ``Agent`` from a validated configuration model.

    Args:
        config: A validated :class:`AgentConfig`. Inter-agent ``handoffs`` on
            an :class:`AgentNodeConfig` are NOT wired here — the topology
            loader assigns them in a second pass.

    Returns:
        The constructed :class:`~troopai.adk.agents.agent.Agent`.

    Raises:
        ConfigResolutionError: If a tool or output-schema reference cannot be
            resolved or resolves to the wrong type.
    """
    verbose: VerboseConfig | None = None
    if config.verbose is not None:
        verbose = VerboseConfig(
            enabled=config.verbose.enabled,
            mode=config.verbose.mode,
            use_color=config.verbose.use_color,
            use_rich=config.verbose.use_rich,
            show_timestamps=config.verbose.show_timestamps,
        )

    output_schema: AgentOutputSchema | None = None
    if isinstance(config.output_schema, str):
        output_schema = resolve_output_schema(config.output_schema, SchemaEnforcement.STRICT)
    elif config.output_schema is not None:
        output_schema = resolve_output_schema(
            config.output_schema.ref, SchemaEnforcement(config.output_schema.enforcement)
        )

    llm, llm_config = build_llm(config)
    guardrails = build_guardrails(config.guardrails) if config.guardrails is not None else AgentGuardrails()

    agent: Agent = Agent(
        name=config.name,
        description=config.description,
        system_prompt=_build_system_prompt(config.system_prompt),
        llm=llm,
        llm_config=llm_config,
        tools=[_build_tool(item) for item in config.tools],
        skill_activation=SkillActivation(config.skill_activation),
        tool_use_behavior=config.tool_use_behavior,
        output_schema=output_schema,
        guardrails=guardrails,
        verbose=verbose,
    )
    logger.debug("Assembled agent %r from config", config.name)
    return agent
