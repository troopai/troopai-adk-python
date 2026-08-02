"""Serialize an ``Agent``'s static surface back to a config dict.

The inverse of ``build_agent`` for the data-heavy, declarative surface. It is
explicitly lossy: behavior that exists only as Python callables/objects
(function-tool bodies, guardrail functions, dynamic prompts) and secrets
(``api_key``, a hosted tool's ``authorization`` / ``headers``, and the
``LLMConfig`` ``extra_*`` escape-hatch maps that commonly carry credentials)
cannot/should not be serialized — those fields are omitted and the omission
is logged at debug. The result validates as an ``AgentConfig``.
"""

from __future__ import annotations

import dataclasses
import logging
from enum import Enum
from typing import Any, Protocol, runtime_checkable

from troopai.adk.agents.agent import Agent
from troopai.adk.llms.llm import LLM
from troopai.adk.llms.llm_config import LLMConfig
from troopai.adk.prompts.system_prompt import SystemPrompt
from troopai.adk.schemas.agent_output_schema import AgentOutputSchema
from troopai.adk.skills.activation import SkillActivation
from troopai.adk.tools.hosted.code_execution_tool import CodeExecutionTool
from troopai.adk.tools.hosted.file_search_tool import FileSearchTool
from troopai.adk.tools.hosted.hosted_tool import HostedTool
from troopai.adk.tools.hosted.image_generation_tool import ImageGenerationTool
from troopai.adk.tools.hosted.mcp_tool import HostedMCPTool
from troopai.adk.tools.hosted.url_context_tool import URLContextTool
from troopai.adk.tools.hosted.web_search_tool import WebSearchTool
from troopai.adk.types.llms.retry_policy import LLMRetryPolicy
from troopai.adk.types.tools.tool_use_behavior import StopAtTools

logger = logging.getLogger(__name__)

# Reverse of the hosted-tool registry: concrete class → config ``type`` name.
_HOSTED_TOOL_TYPES: tuple[tuple[type[HostedTool], str], ...] = (
    (WebSearchTool, "web_search"),
    (CodeExecutionTool, "code_execution"),
    (FileSearchTool, "file_search"),
    (ImageGenerationTool, "image_generation"),
    (URLContextTool, "url_context"),
    (HostedMCPTool, "hosted_mcp"),
)


@runtime_checkable
class _ModelLLM(Protocol):
    """Structural type for an LLM that exposes a ``model`` identifier.

    The ``LLM`` ABC does not type ``model``, but every concrete provider
    implementation exposes it as a ``str`` property. Used to read the model
    off a reverse-mapped LLM instance without importing the concrete class.
    """

    @property
    def model(self) -> str: ...


# (provider name, "module:ClassName") — the inverse of the provider registry.
_PROVIDER_CLASSES: tuple[tuple[str, str], ...] = (
    ("anthropic", "troopai.adk.llms.anthropic.anthropic_model:AnthropicLLM"),
    ("openai-responses", "troopai.adk.llms.openai.openai_responses_model:OpenAIResponsesLLM"),
    ("openai-chat", "troopai.adk.llms.openai.openai_chatcompletions_model:OpenAIChatCompletionsLLM"),
    ("gemini", "troopai.adk.llms.gemini.gemini_model:GeminiLLM"),
    ("litellm", "troopai.adk.llms.litellm.litellm_model:LiteLLM"),
)


def _provider_name(llm: LLM) -> str | None:
    """Reverse-map an LLM instance to its config provider name, or ``None``."""
    import importlib

    for name, spec in _PROVIDER_CLASSES:
        module_path, cls_name = spec.split(":")
        try:
            module = importlib.import_module(module_path)
        except ImportError:
            logger.debug("dump_agent: provider module %r not installed; skipping", module_path)
            continue
        cls = getattr(module, cls_name, None)
        if cls is None:
            logger.debug("dump_agent: %r not found in %r; provider table may be stale", cls_name, module_path)
            continue
        if isinstance(llm, cls):
            return name
    return None


# LLMConfig escape-hatch maps that commonly carry credentials; never dumped.
_LLM_CONFIG_OMIT: frozenset[str] = frozenset({"extra_headers", "extra_body", "extra_query", "extra_args"})


def _dump_llm_config(config: LLMConfig) -> dict[str, Any]:
    """Dump an ``LLMConfig`` (or subclass) to JSON-safe set fields.

    ``api_key`` is not an ``LLMConfig`` field (it lives on the LLM instance),
    so no secret is emitted there. The ``extra_*`` escape-hatch maps
    (``extra_headers`` / ``extra_body`` / ``extra_query`` / ``extra_args``)
    are also omitted — they commonly carry credentials and are the least
    useful to round-trip declaratively. ``retry_policy`` flattens to a dict
    (``retry_on`` as a sorted list); a ``StrEnum`` (``ToolExecutionMode``)
    becomes its value; a non-numeric ``timeout`` (``httpx.Timeout``) is
    omitted as non-JSON.
    """
    out: dict[str, Any] = {}
    for field in dataclasses.fields(config):
        value = getattr(config, field.name)
        if value is None:
            continue
        if field.name in _LLM_CONFIG_OMIT:
            logger.debug("dump_agent: omitting escape-hatch field %r (may carry credentials)", field.name)
            continue
        if isinstance(value, LLMRetryPolicy):
            policy = {
                f.name: getattr(value, f.name) for f in dataclasses.fields(value) if getattr(value, f.name) is not None
            }
            if policy.get("retry_on") is not None:
                policy["retry_on"] = sorted(policy["retry_on"])
            out[field.name] = policy
        elif field.name == "timeout" and not isinstance(value, (int, float)):
            logger.debug("dump_agent: non-numeric timeout cannot be dumped; omitting")
        elif isinstance(value, Enum):
            out[field.name] = value.value
        else:
            out[field.name] = value
    return out


def _dump_hosted_tool(tool: HostedTool) -> dict[str, Any] | None:
    """Dump a hosted tool to ``{type, args}`` with its set (non-default) fields."""
    for cls, type_name in _HOSTED_TOOL_TYPES:
        if isinstance(tool, cls):
            args: dict[str, Any] = {}
            for field in dataclasses.fields(tool):
                # ``repr=False`` marks sensitive fields (e.g. HostedMCPTool's
                # ``authorization`` / ``headers``); never serialize a secret.
                if not field.repr:
                    if getattr(tool, field.name) is not None:
                        logger.debug("dump_agent: omitting secret field %r of %s", field.name, type(tool).__name__)
                    continue
                value = getattr(tool, field.name)
                if field.default is not dataclasses.MISSING:
                    default = field.default
                elif field.default_factory is not dataclasses.MISSING:
                    default = field.default_factory()
                else:
                    default = None
                if value is not None and value != default:
                    args[field.name] = value
            return {"type": type_name, "args": args}
    logger.debug("dump_agent: hosted tool %s not in the dump table; omitting", type(tool).__name__)
    return None


def _dump_output_schema(schema: AgentOutputSchema) -> dict[str, Any] | None:
    """Dump an output schema to ``{ref, enforcement}`` from its wrapped type."""
    output_type = schema.output_schema
    if output_type is None:
        logger.debug("dump_agent: output_schema has no wrapped type; omitting")
        return None
    ref = f"{output_type.__module__}:{output_type.__qualname__}"
    return {"ref": ref, "enforcement": schema.schema_enforcement.value}


def dump_agent(agent: Agent) -> dict[str, Any]:
    """Dump an ``Agent``'s declarative surface to a config dict.

    Args:
        agent: The agent to serialize.

    Returns:
        A dict that validates as an ``AgentConfig`` for the recoverable
        fields. Code-only fields (function tools, guardrails, dynamic
        prompts) and secrets are omitted.
    """
    data: dict[str, Any] = {"name": agent.name}

    if agent.description is not None:
        data["description"] = agent.description

    llm = agent.llm
    if isinstance(llm, str):
        data["llm"] = llm
        if agent.llm_config is not None:
            data["llm_config"] = _dump_llm_config(agent.llm_config)
    elif isinstance(llm, LLM):
        provider = _provider_name(llm)
        # The _ModelLLM check is defensive: every registered provider exposes
        # ``model``, so it only guards a future provider that omits it.
        if provider is None or not isinstance(llm, _ModelLLM):
            logger.debug("dump_agent: custom LLM %s has no provider mapping; omitting", type(llm).__name__)
        else:
            block: dict[str, Any] = {"provider": provider, "model": llm.model}
            if agent.llm_config is not None:
                block["config"] = _dump_llm_config(agent.llm_config)
            data["llm"] = block

    prompt = agent.system_prompt
    if isinstance(prompt, str):
        data["system_prompt"] = prompt
    elif isinstance(prompt, SystemPrompt):
        data["system_prompt"] = prompt.model_dump(exclude_none=True)
    elif prompt is not None:
        logger.debug("dump_agent: dynamic/callable system_prompt cannot be dumped; omitting")

    if isinstance(agent.output_schema, AgentOutputSchema):
        dumped_schema = _dump_output_schema(agent.output_schema)
        if dumped_schema is not None:
            data["output_schema"] = dumped_schema
    elif agent.output_schema is not None:
        logger.debug("dump_agent: non-AgentOutputSchema output_schema cannot be dumped; omitting")

    if agent.skill_activation is not SkillActivation.LAZY:
        data["skill_activation"] = agent.skill_activation.value

    behavior = agent.tool_use_behavior
    if isinstance(behavior, StopAtTools):
        data["tool_use_behavior"] = dataclasses.asdict(behavior)
    elif behavior == "stop_on_first_tool":
        data["tool_use_behavior"] = behavior
    elif not isinstance(behavior, str):
        logger.debug("dump_agent: callable tool_use_behavior cannot be dumped; omitting")
    # "run_llm_again" is the Agent default and is omitted (round-trips as default).

    tool_entries: list[dict[str, Any]] = []
    omitted_tools = 0
    for tool in agent.tools:
        dumped_tool = _dump_hosted_tool(tool) if isinstance(tool, HostedTool) else None
        if dumped_tool is not None:
            tool_entries.append(dumped_tool)
        else:
            omitted_tools += 1
    if omitted_tools > 0:
        logger.debug("dump_agent: omitted %d non-hosted/unrecoverable tool(s)", omitted_tools)
    if len(tool_entries) > 0:
        data["tools"] = tool_entries

    return data
