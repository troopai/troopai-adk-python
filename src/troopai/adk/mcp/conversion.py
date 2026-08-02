"""Convert MCP wire types to ADK framework types.

The two functions here are the only places in the framework that
import ``mcp.types``: every other module sees only ``FunctionTool``
(Layer 1) and ``MCPListToolsItem`` (Layer 3). This keeps the
type-layer boundary clean — the wire protocol is a transport
implementation detail.

Design choices:

- ``inputSchema`` is passed through unchanged by default. Most
  providers handle intra-document ``$ref`` natively. The optional
  ``inline_refs`` flag on ``MCPToolset`` invokes the resolver in
  ``schema_resolver.py`` for non-conformant servers.
- Text ``CallToolResult.content[]`` blocks form the model-visible
  string. Non-text content blocks are surfaced via the artifact
  channel (``FunctionToolCallResult.artifact``). When
  ``use_structured_content=True`` is set on the server,
  ``structuredContent`` is included in that artifact too.
- ``isError=True`` raises ``MCPToolCallError``. The framework's
  standard tool-error path re-raises it when
  ``RunConfig.fail_on_tool_error=True`` (the default); set it to
  ``False`` to convert the error into a model-visible message.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from mcp.types import TextContent

from troopai.adk.mcp.exceptions import MCPToolCallError
from troopai.adk.schemas import SchemaEnforcement
from troopai.adk.tools.function_tool import FunctionTool

if TYPE_CHECKING:
    from mcp import Tool as MCPTool
    from mcp.types import CallToolResult

    from troopai.adk.mcp.mcp_server import MCPServer
    from troopai.adk.tools.tool_context import ToolContext
    from troopai.adk.types.tools import ApprovalPolicy

logger = logging.getLogger(__name__)


def mcp_tool_to_function_tool(
    mcp_tool: MCPTool,
    server: MCPServer,
    *,
    use_structured_content: bool = False,
    inline_refs: bool = False,
    requires_approval: ApprovalPolicy | None = None,
) -> FunctionTool:
    """Wrap an ``mcp.types.Tool`` as a ``FunctionTool``.

    The returned tool's ``on_invoke`` parses the LLM's raw JSON args
    and delegates to ``server.call_tool``. Schema enforcement is
    ``NONE`` so the MCP server's exact ``inputSchema`` reaches the
    LLM provider — strict-mode normalisation would silently mutate
    it, which contradicts the MCP server's contract.

    Args:
        mcp_tool: The MCP tool descriptor returned by ``list_tools``.
        server: The originating server, captured by closure for
            invocation. The server's name is recorded for error
            messages and observability.
        use_structured_content: When ``True``, include MCP
            ``structuredContent`` in the tool artifact. Non-text
            content blocks are always preserved in the artifact
            channel; the LLM still receives only the joined text body.
        inline_refs: When ``True``, intra-document ``$ref`` pointers
            inside ``inputSchema`` are resolved (inlined) before the
            schema is sent to the LLM provider. Use this for MCP
            servers that emit ``$ref`` against ``$defs`` and target
            providers that do not honour them.
        requires_approval: Controls the HITL approval gate on the
            converted tool.  A static ``True`` / ``False`` sets a
            fixed policy; a callable receives the ``ToolContext`` at
            each invocation and returns a ``bool`` (sync or async) for
            per-call decisions (e.g. require approval only in
            production).  ``None`` (default) maps to ``False``.

    Returns:
        A ``FunctionTool`` whose ``schema`` is the (optionally
        ref-resolved) MCP ``inputSchema`` and whose ``on_invoke``
        invokes the server.
    """
    server_name = server.name
    tool_name = mcp_tool.name
    description = mcp_tool.description or ""
    raw_schema: dict[str, Any] = dict(mcp_tool.inputSchema or {})

    # ``properties`` is required by most LLM providers' tool-schema
    # validators even when the tool takes no arguments. MCP servers
    # may legally omit it for parameterless tools; pad here so the
    # provider does not reject the schema.
    if "properties" not in raw_schema:
        raw_schema["properties"] = {}

    if inline_refs:
        from troopai.adk.mcp.schema_resolver import inline_intra_document_refs

        raw_schema = inline_intra_document_refs(raw_schema, tool_name=tool_name)

    approval = requires_approval if requires_approval is not None else False

    async def _on_invoke(
        ctx: ToolContext[Any],
        raw_args: str,
    ) -> tuple[str, dict[str, Any] | None]:
        del ctx
        args = _parse_args(raw_args, tool_name, server_name)
        result = await server.call_tool(tool_name, args)
        text = call_tool_result_to_str(
            result,
            tool_name=tool_name,
            server_name=server_name,
        )
        artifact = call_tool_result_to_artifact(
            result,
            include_structured_content=use_structured_content,
        )
        return text, artifact

    return FunctionTool(
        name=tool_name,
        description=description,
        schema=raw_schema,
        schema_enforcement=SchemaEnforcement.NONE,
        on_invoke=_on_invoke,
        response_format="content_and_artifact",
        requires_approval=approval,
    )


def call_tool_result_to_str(
    result: CallToolResult,
    *,
    tool_name: str,
    server_name: str,
) -> str:
    """Serialise a ``CallToolResult`` into the string a tool returns.

    ``isError=True`` raises ``MCPToolCallError`` so the framework's
    tool-error pathway produces the model-visible message. Text
    parts are concatenated; non-text parts are omitted from this
    string and retained by :func:`call_tool_result_to_artifact`.

    Args:
        result: The raw MCP call result.
        tool_name: Originating tool name (for error messages).
        server_name: Originating server name (for error messages).

    Returns:
        The joined text content. Empty string when no text parts
        were returned.
    """
    text_parts: list[str] = []
    for part in result.content:
        if isinstance(part, TextContent):
            text_parts.append(part.text)
        else:
            logger.debug(
                "MCP tool %r on server %r returned non-text part type=%r; omitting from text output",
                tool_name,
                server_name,
                getattr(part, "type", None),
            )

    body = "\n".join(text_parts)
    if result.isError:
        raise MCPToolCallError(tool_name, server_name, body or "(no error message)")
    return body


def call_tool_result_to_artifact(
    result: CallToolResult,
    *,
    include_structured_content: bool = False,
) -> dict[str, Any] | None:
    """Serialise non-text MCP result data for the tool artifact channel."""
    non_text_content: list[dict[str, Any]] = [
        _serialise_content_block(part) for part in result.content if not isinstance(part, TextContent)
    ]
    artifact: dict[str, Any] = {}
    if len(non_text_content) > 0:
        artifact["content"] = non_text_content
    if include_structured_content:
        structured_content = getattr(result, "structuredContent", None)
        if structured_content is not None:
            artifact["structuredContent"] = structured_content
    return artifact if len(artifact) > 0 else None


def _serialise_content_block(part: Any) -> dict[str, Any]:
    """Convert an MCP Pydantic content block into JSON-compatible data."""
    if hasattr(part, "model_dump"):
        dumped = part.model_dump(by_alias=True, mode="json", exclude_none=True)
        if isinstance(dumped, dict):
            return dumped
    if hasattr(part, "dict"):
        dumped = part.dict(by_alias=True, exclude_none=True)
        if isinstance(dumped, dict):
            return dumped
    return {"type": str(getattr(part, "type", type(part).__name__))}


def _parse_args(raw_args: str, tool_name: str, server_name: str) -> Mapping[str, Any] | None:
    """Parse the LLM's raw JSON args string into a mapping.

    Empty string or ``"null"`` returns ``None`` so a parameterless
    tool's call_tool receives no arguments. Malformed JSON raises
    ``MCPToolCallError`` carrying the parse error so the failure
    surfaces as a tool error rather than an unhandled exception.

    Args:
        raw_args: The JSON string the LLM produced for this tool call.
        tool_name: Tool name used in error messages.
        server_name: Server name used in error messages.

    Returns:
        A mapping of argument names to values, or ``None`` when the
        tool takes no arguments.

    Raises:
        MCPToolCallError: If ``raw_args`` is not valid JSON, or if the
            parsed value is not a JSON object (dict).
    """
    if len(raw_args) == 0 or raw_args == "null":
        return None
    try:
        parsed = json.loads(raw_args)
    except json.JSONDecodeError as exc:
        raise MCPToolCallError(
            tool_name,
            server_name,
            f"invalid JSON arguments: {exc.msg}",
        ) from exc
    if isinstance(parsed, dict):
        return parsed
    raise MCPToolCallError(
        tool_name,
        server_name,
        f"arguments must be a JSON object, got {type(parsed).__name__}",
    )
