"""Schema model for declarative provider-hosted tools.

A ``HostedToolRef`` names a provider-hosted tool by ``type`` and carries
free-form ``args``. The args are validated at build time by the framework's
own ``HostedTool`` dataclass (the single source of truth) rather than
re-mirrored here, so a typo in an arg surfaces as a guiding error when the
tool is constructed. The ``type`` is a closed set enforced by Pydantic;
``register_hosted_tool`` replaces the factory for a known ``type`` value
(unknown values are unreachable from JSON), and user-defined tools use the
dotted function-``ref`` form instead.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

HostedToolType = Literal[
    "web_search",
    "code_execution",
    "file_search",
    "image_generation",
    "url_context",
    "hosted_mcp",
]


class HostedToolRef(BaseModel):
    """Reference to a provider-hosted tool by type name with free-form args.

    Attributes:
        type: The hosted-tool type name (a key in the hosted-tool registry).
        args: Constructor arguments forwarded to the ``HostedTool`` subclass.
    """

    model_config = ConfigDict(extra="forbid")

    type: HostedToolType
    """The hosted-tool type name (a key in the hosted-tool registry)."""

    args: dict[str, Any] = Field(default_factory=dict)
    """Constructor arguments forwarded verbatim to the ``HostedTool`` subclass."""
