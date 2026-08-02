"""WebSocket transport MCP server.

Connects via the MCP SDK's ``websocket_client``. Bidirectional and
long-lived; useful for low-latency setups and servers that want to
push notifications without an SSE channel.

Requires the optional ``websockets`` package
(``pip install 'troopai-adk-python[mcp,websockets]'``).
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, override

from troopai.adk.mcp.exceptions import MCPConnectionError, UnsupportedTransportError
from troopai.adk.mcp.mcp_server import MCPServerWithClientSession, extract_first_exception
from troopai.adk.mcp.run_hooks_bridge import (
    fire_on_mcp_connect,
    fire_on_mcp_connected,
    fire_on_mcp_error,
)

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class MCPServerWebsocketParams:
    """Parameters for a WebSocket-transport MCP server.

    Attributes:
        url: The MCP WebSocket endpoint URL (typically ``ws://`` or ``wss://``).
        call_tool_timeout_seconds: Per-call read timeout for
            ``call_tool`` requests, in seconds.
    """

    url: str
    """The WebSocket endpoint URL (``ws://`` / ``wss://``)."""

    call_tool_timeout_seconds: float | None = field(default=None)
    """Per-call read timeout for ``call_tool`` requests, in seconds.

    ``None`` (default) disables the timeout — the call waits
    indefinitely for the server to respond. Set a finite value (e.g.
    ``120.0``) to bound how long a stalled WebSocket server can block
    the runner turn.
    """


class MCPServerWebsocket(MCPServerWithClientSession):
    """MCP server over the WebSocket transport.

    The MCP SDK's ``websocket_client`` performs no authentication —
    use a reverse proxy (TLS-terminating + bearer-token gating) when
    deploying against untrusted networks. Custom auth headers are not
    yet supported by the upstream client.
    """

    def __init__(
        self,
        *,
        name: str,
        params: MCPServerWebsocketParams,
        cache_tools_list: bool = True,
        use_structured_content: bool = False,
        llm: Any | None = None,
        elicitation_callback: Any | None = None,
    ) -> None:
        """Initialise the WebSocket MCP server.

        Args:
            name: Human-readable server name used in log messages and
                error reports.
            params: WebSocket endpoint URL parameters.
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
        """Open the WebSocket transport and initialise the MCP session."""
        try:
            from mcp.client.websocket import websocket_client
        except ImportError as exc:
            raise UnsupportedTransportError(
                "WebSocket transport requires the 'websockets' package. "
                "Install with: pip install 'troopai-adk-python[mcp]' websockets"
            ) from exc

        async with self._connect_lock:
            if self._session is not None:
                return
            await fire_on_mcp_connect(self._name)
            stack = AsyncExitStack()
            try:
                read, write = await stack.enter_async_context(websocket_client(self._params.url))
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
                logger.info("MCP WebSocket server %r connected (%s)", self._name, self._params.url)
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
                logger.error("MCP WebSocket server %r failed to connect: %s", self._name, root_exc)
                await fire_on_mcp_error(self._name, root_exc)
                raise MCPConnectionError(
                    f"MCP WebSocket server '{self._name}' failed to connect: {root_exc}"
                ) from root_exc

    @override
    async def cleanup(self) -> None:
        """Close the WebSocket session. Idempotent.

        Anyio's ``RuntimeError("Attempted to exit ...cancel scope...")``
        is demoted to DEBUG. Same rationale as
        ``MCPServerStdio.cleanup``. ``startswith("Attempted to exit")``
        keeps the MCP SDK's ``"No active cancel scope"`` at WARNING.
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
                        "MCP WebSocket server %r cleanup: anyio cancel-scope mismatch (socket already closed)",
                        self._name,
                    )
                else:
                    logger.warning("MCP WebSocket server %r cleanup raised", self._name, exc_info=True)
            except Exception:
                logger.warning("MCP WebSocket server %r cleanup raised", self._name, exc_info=True)
            finally:
                self._exit_stack = None
                logger.info("MCP WebSocket server %r disconnected", self._name)
