"""Provider-hosted tool registry and factories for declarative configs.

A hosted-tool factory turns a :class:`HostedToolRef` (``{type, args}``) into
a concrete ``HostedTool`` instance by calling the framework's own dataclass
constructor with the free-form ``args``. The dataclass is the single source
of truth — a bad or unknown arg surfaces as a ``ConfigResolutionError`` at
construction rather than being re-validated against a duplicated schema.

``register_hosted_tool`` exposes the registry for extension, mirroring the
LLM provider registry. ``ComputerTool`` is intentionally absent from both the
``HostedToolType`` literal and this registry: it requires a live Python
``Computer`` object that cannot be serialized to JSON.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from troopai.adk.exceptions import ConfigResolutionError
from troopai.adk.tools import Tool
from troopai.adk.tools.hosted.code_execution_tool import CodeExecutionTool
from troopai.adk.tools.hosted.file_search_tool import FileSearchTool
from troopai.adk.tools.hosted.hosted_tool import HostedTool
from troopai.adk.tools.hosted.image_generation_tool import ImageGenerationTool
from troopai.adk.tools.hosted.mcp_tool import HostedMCPTool
from troopai.adk.tools.hosted.url_context_tool import URLContextTool
from troopai.adk.tools.hosted.web_search_tool import WebSearchTool
from troopai.adk.types.config.tool_config import HostedToolRef

logger = logging.getLogger(__name__)

HostedToolFactory = Callable[[dict[str, Any]], Tool]
"""Factory taking a hosted tool's ``args`` dict, returning the built tool.

Each built-in returns a concrete provider-hosted ``Tool`` (a ``HostedTool``
subclass); the return is typed ``Tool`` because that is what ``Agent.tools``
consumes."""

HOSTED_TOOL_REGISTRY: dict[str, HostedToolFactory] = {}
"""Registry mapping a hosted-tool ``type`` name to its factory."""


def register_hosted_tool(name: str, factory: HostedToolFactory) -> None:
    """Register (or override) the factory for a hosted-tool type name.

    Mirrors ``register_llm_provider``. A name the schema's ``HostedToolRef``
    ``type`` literal does not accept leaves an unreachable entry (Pydantic
    rejects an unknown ``type`` before dispatch), so in practice this
    overrides how a known type is built.

    Args:
        name: The hosted-tool ``type`` value (e.g. ``"web_search"``).
        factory: Callable taking the ``args`` dict and returning a ``HostedTool``.
    """
    HOSTED_TOOL_REGISTRY[name] = factory
    logger.debug("Registered hosted-tool factory: %r", name)


def build_hosted_tool(ref: HostedToolRef) -> Tool:
    """Build a provider-hosted tool from a validated :class:`HostedToolRef`.

    Args:
        ref: The validated hosted-tool reference.

    Returns:
        The constructed provider-hosted ``Tool``.

    Raises:
        ConfigResolutionError: If the ``type`` has no registered factory, or
            the ``args`` are rejected by the tool's constructor (unknown
            field, missing required field, or a ``__post_init__`` invariant).
    """
    factory = HOSTED_TOOL_REGISTRY.get(ref.type)
    if factory is None:
        available = ", ".join(sorted(HOSTED_TOOL_REGISTRY))
        raise ConfigResolutionError(f"No factory registered for hosted tool {ref.type!r}. Available: {available}.")
    try:
        tool = factory(ref.args)
    except (TypeError, ValueError) as exc:
        raise ConfigResolutionError(
            f"Could not build hosted tool {ref.type!r} from args {sorted(ref.args)} ({exc})."
        ) from exc
    if not isinstance(tool, HostedTool):
        raise ConfigResolutionError(
            f"Hosted-tool factory for {ref.type!r} returned {type(tool).__name__}, expected a HostedTool subclass."
        )
    return tool


register_hosted_tool("web_search", lambda args: WebSearchTool(**args))
register_hosted_tool("code_execution", lambda args: CodeExecutionTool(**args))
register_hosted_tool("file_search", lambda args: FileSearchTool(**args))
register_hosted_tool("image_generation", lambda args: ImageGenerationTool(**args))
register_hosted_tool("url_context", lambda args: URLContextTool(**args))
register_hosted_tool("hosted_mcp", lambda args: HostedMCPTool(**args))
