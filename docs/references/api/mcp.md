(references/api/mcp)=

# MCP

Model Context Protocol integration: server transports, lifecycle
management, and tool filtering.

The `mcp` package is an optional extra. When it is not installed, every
name below is bound to `None` so callers can detect availability without
`ImportError` handling.

## Servers and transports

```{eval-rst}
.. autoclass:: troopai.adk.mcp.MCPServerWithClientSession
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.mcp.MCPServerStdio
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.mcp.MCPServerStdioParams
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.mcp.MCPServerStreamableHttp
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.mcp.MCPServerStreamableHttpParams
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.mcp.MCPServerSse
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.mcp.MCPServerSseParams
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.mcp.MCPServerWebsocket
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.mcp.MCPServerWebsocketParams
   :members:
   :show-inheritance:
```

## Lifecycle

```{eval-rst}
.. autoclass:: troopai.adk.mcp.MCPServerManager
   :members:
   :show-inheritance:
```

## Filters

```{eval-rst}
.. autodata:: troopai.adk.mcp.ToolFilter

.. autoclass:: troopai.adk.mcp.ToolFilterContext
   :members:
   :show-inheritance:
```

## Auth and elicitation

```{eval-rst}
.. autodata:: troopai.adk.mcp.HeaderProvider

.. autodata:: troopai.adk.mcp.ElicitationHandler
```

## Exceptions

```{eval-rst}
.. autoclass:: troopai.adk.mcp.MCPError
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.mcp.MCPConnectionError
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.mcp.MCPToolCallError
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.mcp.MCPToolNotFoundError
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.mcp.MCPSchemaConversionError
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.mcp.UnsupportedTransportError
   :members:
   :show-inheritance:
```

The agent-facing adapter `MCPToolset` is a `Toolset` subclass and lives
under `troopai.adk.tools.toolsets.mcp_toolset`. Usage lives in the
[MCP guide](../../mcp/mcp.md).
