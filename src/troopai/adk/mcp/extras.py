"""Optional MCP-derived tools surfaced via ``MCPToolset``.

Two opt-in surfaces:

- ``build_resource_tool`` — a synthetic ``FunctionTool`` (Google ADK
  pattern) that lets the LLM request a resource by URI and receives
  the raw bytes / text in the response.
- ``build_prompt_tools`` — converts every server-side MCP prompt to
  a ``FunctionTool`` (Microsoft Agent Framework pattern). Each call
  returns the assembled prompt text.

Both are gated behind opt-in flags on ``MCPToolset`` (``use_mcp_resources``,
``expose_prompts_as_tools``) so the default surface stays minimal.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from troopai.adk.mcp.exceptions import MCPToolCallError
from troopai.adk.schemas import SchemaEnforcement
from troopai.adk.tools.function_tool import FunctionTool

if TYPE_CHECKING:
    from troopai.adk.mcp.mcp_server import MCPServer
    from troopai.adk.tools.tool_context import ToolContext
    from troopai.adk.types.tools import ApprovalPolicy

logger = logging.getLogger(__name__)


def build_resource_tool(server: MCPServer, *, requires_approval: ApprovalPolicy = False) -> FunctionTool:
    """Return a synthetic ``read_mcp_resource`` tool bound to ``server``.

    The LLM sees a single function with one argument, ``uri``, and
    gets back the resource's text contents. Binary resources are
    surfaced as a JSON string carrying the MIME type and
    base64-encoded payload — the LLM can ask the application to
    decode if needed.

    Args:
        server: The MCP server whose resources the tool will expose.
        requires_approval: HITL approval policy applied to the
            converted tool. A static ``True`` / ``False`` sets a fixed
            policy; a callable receives the ``ToolContext`` at each
            invocation and returns a ``bool`` (sync or async).

    Returns:
        A ``FunctionTool`` named ``read_<server>_resource`` whose
        ``on_invoke`` calls ``server.read_resource(uri)``.
    """
    server_name = server.name
    name = f"read_{_sanitize(server_name)}_resource"
    description = (
        f"Read a resource by URI from the MCP server '{server_name}'. "
        "Use this when the user asks about content (files, documents, "
        "rendered views) the server exposes via its resources surface."
    )

    async def _on_invoke(ctx: ToolContext[Any], raw_args: str) -> str:
        del ctx
        try:
            args = json.loads(raw_args) if len(raw_args) > 0 else {}
        except json.JSONDecodeError as exc:
            raise MCPToolCallError(name, server_name, f"invalid JSON arguments: {exc.msg}") from exc
        uri = args.get("uri") if isinstance(args, dict) else None
        if not isinstance(uri, str) or len(uri) == 0:
            raise MCPToolCallError(name, server_name, "argument 'uri' (string) is required")
        result = await server.read_resource(uri)
        return _resource_result_to_str(result, uri=uri, server_name=server_name)

    return FunctionTool(
        name=name,
        description=description,
        schema={
            "type": "object",
            "properties": {
                "uri": {"type": "string", "description": "Resource URI to read."},
            },
            "required": ["uri"],
        },
        schema_enforcement=SchemaEnforcement.NONE,
        on_invoke=_on_invoke,
        requires_approval=requires_approval,
    )


async def build_prompt_tools(server: MCPServer, *, requires_approval: ApprovalPolicy = False) -> list[FunctionTool]:
    """Convert every MCP prompt on ``server`` to a ``FunctionTool``.

    Each prompt becomes one tool whose schema is derived from the
    prompt's declared arguments. Calling the tool runs
    ``server.get_prompt(name, args)`` and returns the assembled
    prompt text — typically used by the LLM to compose follow-up
    queries.

    Args:
        server: The MCP server whose prompts to convert.
        requires_approval: HITL approval policy applied to every
            converted prompt tool (see :func:`build_resource_tool`).

    Returns:
        A list of ``FunctionTool`` instances, one per prompt returned
        by ``server.list_prompts()``. Empty when the server has no
        prompts.
    """
    listing = await server.list_prompts()
    tools: list[FunctionTool] = []
    for prompt in listing.prompts:
        tools.append(_build_prompt_tool(server, prompt, requires_approval=requires_approval))
    return tools


def _build_prompt_tool(server: MCPServer, prompt: Any, *, requires_approval: ApprovalPolicy = False) -> FunctionTool:
    server_name = server.name
    prompt_name = prompt.name
    tool_name = f"prompt_{_sanitize(prompt_name)}"
    description = prompt.description or f"MCP prompt '{prompt_name}' from '{server_name}'"

    properties: dict[str, Any] = {}
    required: list[str] = []
    for arg in prompt.arguments or []:
        properties[arg.name] = {
            "type": "string",
            "description": arg.description or "",
        }
        if arg.required:
            required.append(arg.name)

    async def _on_invoke(ctx: ToolContext[Any], raw_args: str) -> str:
        del ctx
        try:
            args = json.loads(raw_args) if len(raw_args) > 0 else {}
        except json.JSONDecodeError as exc:
            raise MCPToolCallError(tool_name, server_name, f"invalid JSON arguments: {exc.msg}") from exc
        result = await server.get_prompt(prompt_name, args if isinstance(args, dict) else {})
        return _prompt_result_to_str(result)

    return FunctionTool(
        name=tool_name,
        description=description,
        schema={
            "type": "object",
            "properties": properties,
            "required": required,
        },
        schema_enforcement=SchemaEnforcement.NONE,
        on_invoke=_on_invoke,
        requires_approval=requires_approval,
    )


def _resource_result_to_str(result: Any, *, uri: str, server_name: str) -> str:
    """Render a ``ReadResourceResult`` as a single string for the LLM.

    Args:
        result: The ``ReadResourceResult`` returned by the server.
        uri: The resource URI (for debug logging on empty results).
        server_name: The originating server name (for debug logging).

    Returns:
        Text contents joined by newlines. Binary entries are encoded as
        a JSON object with ``mimeType`` and ``blobBase64`` keys. Empty
        string when the server returned no content items.
    """
    contents = list(getattr(result, "contents", []) or [])
    if len(contents) == 0:
        logger.debug("MCP resource %r on server %r returned empty contents", uri, server_name)
        return ""
    rendered: list[str] = []
    for entry in contents:
        text = getattr(entry, "text", None)
        if isinstance(text, str):
            rendered.append(text)
            continue
        blob = getattr(entry, "blob", None)
        mime = getattr(entry, "mimeType", "application/octet-stream")
        if blob is not None:
            rendered.append(json.dumps({"mimeType": mime, "blobBase64": blob}))
    return "\n".join(rendered)


def _prompt_result_to_str(result: Any) -> str:
    """Concatenate every text part of a ``GetPromptResult``'s messages.

    Args:
        result: The ``GetPromptResult`` returned by ``server.get_prompt``.

    Returns:
        All text message parts joined by newlines. Empty string when
        no messages carry text content.
    """
    rendered: list[str] = []
    for message in getattr(result, "messages", []):
        content = getattr(message, "content", None)
        text = getattr(content, "text", None)
        if isinstance(text, str):
            rendered.append(text)
    return "\n".join(rendered)


def _sanitize(name: str) -> str:
    """Lower-case, replace non-identifier chars with ``_`` for tool names."""
    out = []
    for ch in name.lower():
        if ch.isalnum() or ch == "_":
            out.append(ch)
        else:
            out.append("_")
    return "".join(out).strip("_") or "x"
