"""``MCPToolset`` — adapt an MCP server to the ADK toolset surface.

An ``MCPToolset`` wraps a single ``MCPServer`` and exposes its tools
to the agent runtime via the standard ``Toolset.get_tools(ctx)``
contract. This is the public entry point: developers add
``MCPToolset(server=...)`` to an agent's ``tools`` list and the
runner takes care of connection, tool discovery, filtering, and
disposal.

Composition with the existing toolset wrappers works out of the box:

    MCPToolset(server=stdio_server).prefixed("svc").filtered(my_pred)

— the prefix and filter wrap the toolset transparently, sharing the
same per-turn ``get_tools`` flow that every other toolset uses.

Lifecycle: ``auto_connect=True`` (the default) lazy-connects on the
first ``get_tools`` call; the runner's ``finally`` block calls
``adispose()`` to close the server. ``auto_connect=False`` is the
escape hatch for code that owns the connection externally (e.g. a
shared ``MCPServerManager``).
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from inspect import isawaitable
from typing import TYPE_CHECKING, Any, override

from troopai.adk.tools.toolsets.abstract import Toolset

if TYPE_CHECKING:
    from troopai.adk.mcp.filters import ToolFilter
    from troopai.adk.mcp.mcp_server import MCPServer
    from troopai.adk.run.context import RunContext
    from troopai.adk.tools.function_tool import FunctionTool
    from troopai.adk.types.tools import ApprovalPolicy

logger = logging.getLogger(__name__)


@dataclass
class MCPToolset(Toolset):
    """Toolset adapter that materialises an ``MCPServer``'s tools.

    Each turn ``get_tools`` calls ``server.list_tools()`` (cache-aware
    on the server side), converts every entry to a ``FunctionTool``
    via ``mcp_tool_to_function_tool``, applies the optional
    ``tool_filter``, and returns the result as a name→tool dict.

    Attributes:
        server: The underlying MCP server.
        auto_connect: When ``True`` (default), the toolset opens the
            server's connection on the first ``get_tools`` call and
            disposes it via ``adispose``. When ``False``, the caller
            is responsible for ``connect``/``cleanup`` (typically via
            ``MCPServerManager`` or an explicit ``async with``).
        tool_filter: Optional per-tool predicate. Receives a
            ``ToolFilterContext`` carrying the server name and the
            current ``RunContext``; returns ``True`` to keep the
            tool. Sync or async; exceptions are treated as
            fail-closed (tool excluded, warning logged).
    """

    server: MCPServer
    """The MCP server this toolset adapts."""

    auto_connect: bool = True
    """When ``True``, lazy-connect on first ``get_tools`` and dispose on run end."""

    tool_filter: ToolFilter | None = None
    """Optional per-tool predicate evaluated each turn."""

    use_structured_content: bool = False
    """When ``True``, MCP ``structuredContent`` is surfaced via the
    ``FunctionToolCallResult.artifact`` channel."""

    inline_refs: bool = False
    """When ``True``, intra-document ``$ref`` pointers in the MCP
    server's ``inputSchema`` are inlined before being sent to the LLM
    provider. Use this for non-conformant servers and providers that
    do not honour ``$ref``."""

    use_mcp_resources: bool = False
    """When ``True``, exposes a synthetic ``read_<server>_resource``
    tool the LLM can call to fetch MCP resources by URI."""

    expose_prompts_as_tools: bool = False
    """When ``True``, every server-side MCP prompt is converted to a
    ``FunctionTool`` callable by the LLM."""

    requires_approval: ApprovalPolicy = False
    """Approval policy forwarded to every converted ``FunctionTool``.

    A static ``True`` / ``False`` applies the same policy to all tools.
    A callable receives the ``ToolContext`` at each invocation and
    returns a ``bool`` (sync or async) for per-call decisions — for
    example, requiring approval only when the run context indicates a
    production environment.

    The same callable identity is stored on all converted tools; it is
    not called during conversion, only at invocation time by the
    framework's HITL evaluation path."""

    _connected: bool = field(default=False, init=False, repr=False)
    """Backing field for :attr:`is_connected` — read-only lifecycle
    predicate surfaced via the public property below. Callers read
    ``toolset.is_connected``, never ``toolset._connected``."""

    _disposed: bool = field(default=False, init=False, repr=False)
    """Backing field for :attr:`is_disposed`. See ``_connected``."""

    _connect_lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)
    """Synchronisation primitive — never observable; no public
    accessor by design. Exposing a lock would couple callers to
    the locking strategy."""

    # ------------------------------------------------------------------
    # Public state-observation surface
    # ------------------------------------------------------------------

    @property
    def is_connected(self) -> bool:
        """Whether the underlying server has been connected via this toolset.

        ``True`` after the first successful ``get_tools()`` call when
        ``auto_connect=True`` and before ``adispose``. Useful for
        debugging multi-toolset configurations.
        """
        return self._connected

    @property
    def is_disposed(self) -> bool:
        """Whether ``adispose()`` has been called on this toolset.

        Once disposed, ``get_tools()`` returns an empty dict and the
        underlying server connection (if owned by this toolset) is
        closed. Disposal is one-way; the toolset cannot be revived.
        """
        return self._disposed

    @override
    async def get_tools(
        self,
        ctx: RunContext[Any] | None = None,
    ) -> dict[str, FunctionTool]:
        """Return the converted ``FunctionTool`` dict for this turn.

        Connects the server if ``auto_connect=True`` and not yet
        connected. Returns an empty dict when the toolset has been
        disposed (logs a warning so accidental reuse after run end
        is findable).
        """
        if self._disposed:
            logger.warning(
                "MCPToolset.get_tools() called after adispose; returning empty dict (server=%r)",
                self.server.name,
            )
            return {}

        await self._ensure_connected()

        # Local imports keep the module import-cycle-free at startup;
        # the conversion module imports the function_tool dataclass.
        from troopai.adk.mcp.conversion import mcp_tool_to_function_tool

        mcp_tools = await self.server.list_tools()
        result: dict[str, FunctionTool] = {}
        for mcp_tool in mcp_tools:
            ft = mcp_tool_to_function_tool(
                mcp_tool,
                self.server,
                use_structured_content=self.use_structured_content,
                inline_refs=self.inline_refs,
                requires_approval=self.requires_approval,
            )
            if self.tool_filter is not None and not await _eval_filter(self.tool_filter, self.server.name, ctx, ft):
                continue
            result[ft.name] = ft

        # Optional resource + prompt surfaces (opt-in). These are tool surfaces
        # too, so the same ``tool_filter`` MUST gate them — otherwise a filter
        # that denies a server still exposes its resource-read + prompt tools.
        if self.use_mcp_resources:
            from troopai.adk.mcp.extras import build_resource_tool

            resource_tool = build_resource_tool(self.server, requires_approval=self.requires_approval)
            if self.tool_filter is None or await _eval_filter(self.tool_filter, self.server.name, ctx, resource_tool):
                result[resource_tool.name] = resource_tool

        if self.expose_prompts_as_tools:
            from troopai.adk.mcp.extras import build_prompt_tools

            for prompt_tool in await build_prompt_tools(self.server, requires_approval=self.requires_approval):
                if self.tool_filter is None or await _eval_filter(self.tool_filter, self.server.name, ctx, prompt_tool):
                    result[prompt_tool.name] = prompt_tool

        return result

    @override
    async def adispose(self) -> None:
        """Clean up the server if this toolset connected it.

        Idempotent. Only invokes ``server.cleanup()`` when this
        toolset was the one that connected (``auto_connect=True``
        and the connection was opened). Externally-managed
        connections (``auto_connect=False``) are left alone — their
        owner is responsible.
        """
        if self._disposed:
            return
        self._disposed = True
        if self._connected and self.auto_connect:
            try:
                await self.server.cleanup()
            except Exception:
                logger.warning(
                    "MCPToolset cleanup failed for server %r",
                    self.server.name,
                    exc_info=True,
                )

    async def _ensure_connected(self) -> None:
        """Connect the underlying server if needed.

        Always acquires ``_connect_lock`` so concurrent ``get_tools``
        calls cannot double-connect. The lock is uncontended on the
        warm path (the cheap ``if self._connected: return`` check
        inside the lock is the actual fast-exit) so the cost is one
        extra ``async with`` per turn.
        """
        if not self.auto_connect:
            # Externally managed — assume the caller has connected.
            return
        async with self._connect_lock:
            if self._connected:
                return
            await self.server.connect()
            self._connected = True


async def _eval_filter(
    tool_filter: ToolFilter,
    server_name: str,
    ctx: RunContext[Any] | None,
    tool: FunctionTool,
) -> bool:
    """Evaluate ``tool_filter`` fail-closed.

    Sync and async predicates both supported. Any exception raised
    during evaluation is logged at WARNING and the tool is excluded —
    a buggy filter cannot leak unauthorised tools to the LLM.
    """
    from troopai.adk.mcp.filters import ToolFilterContext

    filter_ctx = ToolFilterContext(server_name=server_name, run_context=ctx)
    try:
        verdict = tool_filter(filter_ctx, tool)
        if isawaitable(verdict):
            verdict = await verdict
    except Exception:
        logger.warning(
            "MCPToolset tool_filter raised for tool %r on server %r; excluding",
            tool.name,
            server_name,
            exc_info=True,
        )
        return False
    return bool(verdict)
