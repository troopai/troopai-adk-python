"""Model Context Protocol (MCP) integration.

Optional-import gate: when the ``mcp`` extra is not installed, every
exported name is bound to ``None``. Callers can compare any export
against ``None`` to detect availability without ``ImportError``
handling.

Top-level surfaces:

- ``MCPServer`` — abstract contract for a single server.
- ``MCPServerStdio`` / ``MCPServerStreamableHttp`` /
  ``MCPServerWebsocket`` / ``MCPServerSse`` — concrete transports.
- ``MCPServerManager`` — multi-server lifecycle holder with
  ref-counted ``acquire`` / ``release``.
- ``ToolFilter`` / ``ToolFilterContext`` — per-tool predicate types.
- ``HeaderProvider`` — per-request HTTP header callable.
- ``ElicitationHandler`` — async callback for MCP elicitation.
- ``MCPError`` and subclasses — exception hierarchy.

The agent-facing adapter ``MCPToolset`` lives under
``troopai.adk.tools.toolsets.mcp_toolset`` because it is a ``Toolset``
subclass; that placement keeps the toolset module a single source
of truth for everything an agent's ``tools`` list accepts.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from troopai.adk.mcp.auth import HeaderProvider
    from troopai.adk.mcp.elicitation import ElicitationHandler
    from troopai.adk.mcp.exceptions import (
        MCPConnectionError,
        MCPError,
        MCPSchemaConversionError,
        MCPToolCallError,
        MCPToolNotFoundError,
        UnsupportedTransportError,
    )
    from troopai.adk.mcp.filters import ToolFilter, ToolFilterContext
    from troopai.adk.mcp.http import MCPServerStreamableHttp, MCPServerStreamableHttpParams
    from troopai.adk.mcp.manager import MCPServerManager
    from troopai.adk.mcp.mcp_server import (
        MCPServer,  # pyright: ignore[reportUnusedImport]
        MCPServerWithClientSession,
    )
    from troopai.adk.mcp.sse import MCPServerSse, MCPServerSseParams
    from troopai.adk.mcp.stdio import MCPServerStdio, MCPServerStdioParams
    from troopai.adk.mcp.websocket import MCPServerWebsocket, MCPServerWebsocketParams
else:
    try:
        from troopai.adk.mcp.auth import HeaderProvider
        from troopai.adk.mcp.elicitation import ElicitationHandler
        from troopai.adk.mcp.exceptions import (
            MCPConnectionError,
            MCPError,
            MCPSchemaConversionError,
            MCPToolCallError,
            MCPToolNotFoundError,
            UnsupportedTransportError,
        )
        from troopai.adk.mcp.filters import ToolFilter, ToolFilterContext
        from troopai.adk.mcp.http import MCPServerStreamableHttp, MCPServerStreamableHttpParams
        from troopai.adk.mcp.manager import MCPServerManager
        from troopai.adk.mcp.mcp_server import MCPServer, MCPServerWithClientSession
        from troopai.adk.mcp.sse import MCPServerSse, MCPServerSseParams
        from troopai.adk.mcp.stdio import MCPServerStdio, MCPServerStdioParams
        from troopai.adk.mcp.websocket import MCPServerWebsocket, MCPServerWebsocketParams
    except ImportError as _exc:
        # Only swallow "top-level mcp package not installed". Any
        # other ImportError (typo inside this package, transitive
        # dep blowup) MUST surface so the failure is visible at
        # development time.
        if getattr(_exc, "name", None) != "mcp":
            raise
        ElicitationHandler = None  # type: ignore[assignment,misc]
        HeaderProvider = None  # type: ignore[assignment,misc]
        MCPConnectionError = None  # type: ignore[assignment,misc]
        MCPError = None  # type: ignore[assignment,misc]
        MCPSchemaConversionError = None  # type: ignore[assignment,misc]
        MCPToolCallError = None  # type: ignore[assignment,misc]
        MCPToolNotFoundError = None  # type: ignore[assignment,misc]
        UnsupportedTransportError = None  # type: ignore[assignment,misc]
        ToolFilter = None  # type: ignore[assignment,misc]
        ToolFilterContext = None  # type: ignore[assignment,misc]
        MCPServerStreamableHttp = None  # type: ignore[assignment,misc]
        MCPServerStreamableHttpParams = None  # type: ignore[assignment,misc]
        MCPServerManager = None  # type: ignore[assignment,misc]
        MCPServer = None  # type: ignore[assignment,misc]
        MCPServerWithClientSession = None  # type: ignore[assignment,misc]
        MCPServerSse = None  # type: ignore[assignment,misc]
        MCPServerSseParams = None  # type: ignore[assignment,misc]
        MCPServerStdio = None  # type: ignore[assignment,misc]
        MCPServerStdioParams = None  # type: ignore[assignment,misc]
        MCPServerWebsocket = None  # type: ignore[assignment,misc]
        MCPServerWebsocketParams = None  # type: ignore[assignment,misc]


__all__ = [
    "ElicitationHandler",
    "HeaderProvider",
    "MCPConnectionError",
    "MCPError",
    "MCPSchemaConversionError",
    # MCPServer is intentionally excluded: the ABC's abstract methods return
    # mcp SDK wire types, making it an internal surface. Extend
    # MCPServerWithClientSession instead (see mcp_server.py).
    "MCPServerManager",
    "MCPServerSse",
    "MCPServerSseParams",
    "MCPServerStdio",
    "MCPServerStdioParams",
    "MCPServerStreamableHttp",
    "MCPServerStreamableHttpParams",
    "MCPServerWebsocket",
    "MCPServerWebsocketParams",
    "MCPServerWithClientSession",
    "MCPToolCallError",
    "MCPToolNotFoundError",
    "ToolFilter",
    "ToolFilterContext",
    "UnsupportedTransportError",
]
