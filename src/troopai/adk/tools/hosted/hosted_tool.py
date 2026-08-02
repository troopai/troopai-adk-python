"""Base class for cross-provider hosted-tool classes.

Hosted tools represent capabilities executed by the LLM provider
server-side — web search, code execution, file search, image
generation, URL retrieval, and similar. The framework does NOT
execute these; it forwards typed config to each provider's wire
format via the matching converter's ``isinstance`` dispatch.

A single class is shared across providers (``WebSearchTool``, never
``AnthropicWebSearchTool``). Per-provider attributes are documented
on each subclass with a "**Provider only.**" line. Attributes set on
a subclass that the active provider does not support are silently
dropped at the converter boundary, with a debug log — this lets one
instance flow across providers without surprise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar


@dataclass(kw_only=True)
class HostedTool:
    """Base class for cross-provider hosted-tool config dataclasses.

    Every concrete subclass MUST:

    1. Be a ``@dataclass(kw_only=True)``.
    2. Document a "Provider matrix" section in its class docstring
       listing every provider that ships the capability and pointing
       to the relevant attributes.
    3. End each attribute docstring with a ``**<Provider> only.**`` or
       ``**<Provider> + <Provider>.**`` line when the attribute applies
       to a strict subset of providers. If an attribute applies
       everywhere, no provider line is needed.

    The framework's ``Tool`` Union accepts every concrete subclass as a
    sibling of ``FunctionTool`` and the local-execution
    ``BuiltinTool`` subclasses. Each provider's converter dispatches by
    ``isinstance``: when a converter encounters a hosted tool it does
    not support, it raises :class:`UnsupportedHostedToolError` rather
    than silently dropping the request. Silent drops are forbidden.

    Class attributes:
        SUPPORTED_PROVIDERS: A tuple of provider identifiers that ship
            this capability natively. Used by
            :class:`UnsupportedHostedToolError` to give a helpful
            message when a converter rejects the tool. Identifiers
            are framework-internal short names (``"anthropic"``,
            ``"openai-responses"``, ``"openai-chatcompletions"``,
            ``"gemini"``, ``"litellm"``).
    """

    SUPPORTED_PROVIDERS: ClassVar[tuple[str, ...]] = ()
