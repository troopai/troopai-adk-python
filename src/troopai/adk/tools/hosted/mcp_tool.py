"""Hosted MCP tool — provider-side Model Context Protocol execution.

Unlike the client-side ``MCPToolset`` (which wraps a local
``MCPServer`` and runs every tool call through the agent's Python
process), ``HostedMCPTool`` declares a remote MCP server that the
provider invokes directly. This eliminates the round-trip through
the client and works for servers reachable from the provider's
infrastructure — typical for SaaS connectors and public MCP services.

Today only the OpenAI Responses API ships hosted-MCP execution.
``MCPToolset`` remains the right choice for Anthropic, Gemini, and
all stdio / private-network deployments.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import ClassVar, Literal

from troopai.adk.tools.hosted.hosted_tool import HostedTool


@dataclass(kw_only=True)
class HostedMCPTool(HostedTool):
    """Provider-hosted MCP server reference.

    Provider matrix:
    - **OpenAI Responses**: native via the ``mcp`` tool type. Honours
      every attribute below.
    - All other providers raise :class:`UnsupportedHostedToolError`.
      Use ``MCPToolset`` (client-side) for Anthropic / Gemini /
      stdio / private-network MCP servers.

    Refs:
        - OpenAI hosted MCP: https://platform.openai.com/docs/guides/tools-remote-mcp
    """

    SUPPORTED_PROVIDERS: ClassVar[tuple[str, ...]] = ("openai-responses",)

    server_label: str
    """A label identifying this MCP server in tool calls (required).
    **OpenAI Responses only.**"""

    server_url: str | None = None
    """The MCP server URL. Exactly one of ``server_url`` or
    ``connector_id`` MUST be set. **OpenAI Responses only.**"""

    connector_id: str | None = None
    """OpenAI service-connector identifier (``connector_dropbox``,
    ``connector_gmail``, ``connector_googlecalendar``,
    ``connector_googledrive``, ``connector_microsoftteams``,
    ``connector_outlookcalendar``, ``connector_outlookemail``,
    ``connector_sharepoint``). Exactly one of ``server_url`` or
    ``connector_id`` MUST be set. **OpenAI Responses only.**"""

    server_description: str | None = None
    """Optional natural-language description of the MCP server,
    surfaced to the model. **OpenAI Responses only.**"""

    headers: Mapping[str, str] | None = field(default=None, repr=False)
    """Static HTTP headers attached to every request to ``server_url``.
    Typical use: ``{"Authorization": "Bearer ..."}`` for API-key auth.
    For OAuth tokens prefer :attr:`authorization`. ``repr=False`` so
    the token is not leaked through ``str(tool)`` or default logging.
    **OpenAI Responses only.**"""

    authorization: str | None = field(default=None, repr=False)
    """OAuth access token to send with each request. The application
    is responsible for the OAuth dance; this field carries the
    resulting bearer token. ``repr=False`` so the token is not leaked
    through ``str(tool)`` or default logging.
    **OpenAI Responses only.**"""

    require_approval: Literal["always", "never"] | Mapping[str, list[str]] | None = None
    """Approval policy for the model's tool calls.

    - ``"always"`` — every call requires approval.
    - ``"never"`` — auto-approve all calls.
    - Mapping ``{"always": [tool_names], "never": [tool_names]}``
      — per-tool fine-grained policy.

    When approval is required, the Responses API emits a typed
    ``MCPApprovalRequestItem`` in :attr:`RunResult.new_items` (the
    framework lifts the wire-format ``mcp_approval_request`` item out
    of the generic provider_item channel). Reply by appending an
    ``MCPApprovalResponseItem(raw=MCPApprovalResponse(...))`` to the
    next turn's input — :meth:`MCPApprovalResponseItem.to_param`
    builds the matching wire-format ``mcp_approval_response`` so the
    provider sees the approve/reject decision.

    MCP approval currently requires the application to orchestrate the
    round-trip manually; loop-driven deferral (surfacing
    ``MCPApprovalRequestItem`` as ``DeferredToolCall`` entries resolved
    via ``RunState.approve(...)`` / ``state.reject(...)``) is not wired
    for this tool.
    **OpenAI Responses only.**"""

    allowed_tools: list[str] | None = field(default=None)
    """Static allowlist of tool names the model may call. ``None``
    (default) means "no filter — every tool the server publishes is
    available." For dynamic / context-aware filtering on the
    client-side path, use ``MCPToolset.tool_filter`` instead.
    **OpenAI Responses only.**"""

    defer_loading: bool = False
    """When ``True``, the server's tool list is not eagerly fetched —
    the model sees only the tool descriptor and must trigger
    discovery via the matching ``ToolSearchTool``. Saves prompt
    tokens for large tool catalogues. **OpenAI Responses only.**"""

    def __post_init__(self) -> None:
        """Enforce the documented XOR constraint on server_url and connector_id."""
        if self.server_url is None and self.connector_id is None:
            raise ValueError("HostedMCPTool requires exactly one of server_url or connector_id to be set; got neither")
        if self.server_url is not None and self.connector_id is not None:
            raise ValueError("HostedMCPTool requires exactly one of server_url or connector_id; both were set")
