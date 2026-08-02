"""MCP-layer exception hierarchy.

All exceptions surfaced by the MCP layer derive from ``MCPError``,
which itself derives from the ADK's ``TroopAIError`` so user
``except TroopAIError`` catches still match.

The split between ``MCPConnectionError`` (lifecycle / transport
problems) and ``MCPToolCallError`` (server returned ``isError=True``
for one tool call) lets callers decide whether to abort the run or
let the framework's standard tool-error path convert the failure
into a model-visible message.
"""

from __future__ import annotations

from troopai.adk.exceptions import TroopAIError


class MCPError(TroopAIError):
    """Base for all MCP-layer errors."""


class MCPConnectionError(MCPError):
    """Raised when an MCP server is unreachable or fails to initialise.

    Wraps stdio subprocess spawn errors, HTTP connect failures, MCP
    SDK ``initialize()`` failures, and any other connection-lifecycle
    problem. Callers may treat this as fatal (the toolset cannot
    contribute tools) or retry at a higher layer.
    """


class MCPToolCallError(MCPError):
    """Raised when an MCP server returns ``isError=True`` for a tool call.

    Carries the originating ``tool_name`` and ``server`` so error
    messages can identify which integration failed without scanning
    a stack trace. Flows through the framework's standard tool-error
    path: when ``RunConfig.fail_on_tool_error=True`` (the default) it
    propagates; set it to ``False`` to convert it to a model-visible
    string via ``default_tool_error_function``.
    """

    def __init__(self, tool_name: str, server: str, content: str) -> None:
        self.tool_name = tool_name
        self.server = server
        self.content = content
        super().__init__(f"MCP tool '{tool_name}' on server '{server}' returned an error: {content}")


class MCPToolNotFoundError(MCPError):
    """Raised when a requested tool is not on the server's tool list.

    Most commonly fires when a cached tool list is stale and the LLM
    requests a tool the server has since removed.
    """


class MCPSchemaConversionError(MCPError):
    """Raised when an MCP tool's ``inputSchema`` cannot be converted.

    ``inputSchema`` is passed through unchanged so this is rare,
    but a malformed schema dict from a non-conformant server
    (missing ``type``, unparseable ``$defs``) surfaces here instead
    of failing later inside the LLM provider.
    """


class UnsupportedTransportError(MCPError):
    """Raised when an MCP transport is requested that the build does not support.

    Reserved for transports added behind extras or feature flags. The
    stdio + streamable-HTTP transports are always available when the
    ``mcp`` extra is installed.
    """
