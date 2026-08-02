"""SSE-transport MCP server (MCP-spec-deprecated).

The MCP spec marks the SSE transport as **deprecated** in favour of
streamable HTTP — new servers SHOULD use ``MCPServerStreamableHttp``.
This class supports servers that have not yet migrated to streamable
HTTP.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, override

from mcp.client.sse import sse_client

from troopai.adk.mcp.exceptions import MCPConnectionError
from troopai.adk.mcp.mcp_server import MCPServerWithClientSession, extract_first_exception
from troopai.adk.mcp.run_hooks_bridge import (
    fire_on_mcp_connect,
    fire_on_mcp_connected,
    fire_on_mcp_error,
)

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class MCPServerSseParams:
    """Parameters for an SSE-transport MCP server (MCP-spec-deprecated).

    Attributes:
        url: The SSE endpoint URL.
        headers: Static headers attached to every request.
        timeout_seconds: HTTP-level per-request timeout.
        sse_read_timeout_seconds: Maximum wait between SSE events.
        call_tool_timeout_seconds: Per-call read timeout for
            ``call_tool`` requests, in seconds.
    """

    url: str
    """The SSE endpoint URL."""

    headers: dict[str, str] | None = field(default=None, repr=False)
    """Static headers attached to every request. ``repr=False`` so
    bearer tokens cannot leak through ``str(params)`` or default
    logging."""

    timeout_seconds: float = 5.0
    """HTTP-level per-request timeout (default 5 s, matches MCP SDK default)."""

    sse_read_timeout_seconds: float = 300.0
    """Max wait between SSE events (default 300 s)."""

    call_tool_timeout_seconds: float | None = None
    """Per-call read timeout for ``call_tool`` requests, in seconds.

    ``None`` (default) disables the timeout — the call waits
    indefinitely for the server to respond. Set a finite value (e.g.
    ``120.0``) to bound how long a stalled SSE server can block the
    runner turn.
    """


class MCPServerSse(MCPServerWithClientSession):
    """MCP server over the SSE transport (MCP-spec-deprecated).

    .. deprecated::
        Prefer :class:`MCPServerStreamableHttp`. The MCP project
        marks SSE deprecated; this class supports servers that have
        not migrated to streamable HTTP.
    """

    def __init__(
        self,
        *,
        name: str,
        params: MCPServerSseParams,
        cache_tools_list: bool = True,
        use_structured_content: bool = False,
        llm: Any | None = None,
        elicitation_callback: Any | None = None,
    ) -> None:
        """Initialise the SSE MCP server.

        Args:
            name: Human-readable server name used in log messages and
                error reports.
            params: SSE endpoint URL, headers, and timeout parameters.
            cache_tools_list: When ``True`` (default), the tool list
                is cached between turns and invalidated by push
                notifications.
            use_structured_content: When ``True``, pass
                ``structuredContent`` through to the artifact channel.
                Default ``False`` avoids duplicate content.
            llm: Optional ``LLM`` for serving MCP
                ``sampling/createMessage`` requests.
            elicitation_callback: Optional async callable invoked on
                ``elicitation/create`` requests from the server.
        """
        super().__init__(
            name=name,
            cache_tools_list=cache_tools_list,
            use_structured_content=use_structured_content,
            llm=llm,
            elicitation_callback=elicitation_callback,
        )
        self._params = params
        self._exit_stack: AsyncExitStack | None = None

    @override
    async def connect(self) -> None:
        """Open the SSE transport and initialise the MCP session."""
        async with self._connect_lock:
            if self._session is not None:
                return
            await fire_on_mcp_connect(self._name)
            stack = AsyncExitStack()
            try:
                streams = await stack.enter_async_context(
                    sse_client(
                        self._params.url,
                        headers=self._params.headers,
                        timeout=self._params.timeout_seconds,
                        sse_read_timeout=self._params.sse_read_timeout_seconds,
                    )
                )
                read, write = streams
                timeout = (
                    timedelta(seconds=self._params.call_tool_timeout_seconds)
                    if self._params.call_tool_timeout_seconds is not None
                    else None
                )
                session = await stack.enter_async_context(
                    self._make_client_session(read, write, read_timeout_seconds=timeout)
                )
                await self._attach_session(session)
                self._exit_stack = stack
                logger.info("MCP SSE server %r connected (%s)", self._name, self._params.url)
                await fire_on_mcp_connected(self._name)
            except (Exception, asyncio.CancelledError) as exc:
                # See ``MCPServerStreamableHttp.connect`` for the full
                # rationale.
                root_exc: BaseException = exc
                try:
                    await stack.aclose()
                except (Exception, BaseExceptionGroup) as close_exc:
                    if isinstance(exc, asyncio.CancelledError):
                        root_exc = extract_first_exception(close_exc)
                if isinstance(root_exc, asyncio.CancelledError):
                    task = asyncio.current_task()
                    if task is not None and task.cancelling() > 0:
                        raise
                logger.error("MCP SSE server %r failed to connect: %s", self._name, root_exc)
                await fire_on_mcp_error(self._name, root_exc)
                raise MCPConnectionError(f"MCP SSE server '{self._name}' failed to connect: {root_exc}") from root_exc

    @override
    async def cleanup(self) -> None:
        """Close the SSE session. Idempotent.

        Anyio's ``RuntimeError("Attempted to exit ...cancel scope...")``
        is demoted to DEBUG. Same rationale as
        ``MCPServerStdio.cleanup``. ``startswith("Attempted to exit")``
        keeps the MCP SDK's ``"No active cancel scope"`` (a
        different error class with different semantics) at WARNING.
        """
        async with self._connect_lock:
            if self._exit_stack is None:
                return
            await self._detach_session()
            try:
                await self._exit_stack.aclose()
            except RuntimeError as exc:
                if str(exc).startswith("Attempted to exit"):
                    logger.debug(
                        "MCP SSE server %r cleanup: anyio cancel-scope mismatch (SSE stream already closed)",
                        self._name,
                    )
                else:
                    logger.warning("MCP SSE server %r cleanup raised", self._name, exc_info=True)
            except Exception:
                logger.warning("MCP SSE server %r cleanup raised", self._name, exc_info=True)
            finally:
                self._exit_stack = None
                logger.info("MCP SSE server %r disconnected", self._name)
