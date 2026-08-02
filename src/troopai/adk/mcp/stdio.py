"""Stdio-transport MCP server.

Spawns a subprocess and communicates with it over its stdin/stdout
using the MCP SDK's ``stdio_client``. Lifecycle is bound to an
``AsyncExitStack`` so the subprocess is reliably terminated when
``cleanup()`` runs, even if the run raised mid-tool-call.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, override

from mcp.client.stdio import StdioServerParameters, stdio_client

from troopai.adk.mcp.exceptions import MCPConnectionError
from troopai.adk.mcp.mcp_server import MCPServerWithClientSession, extract_first_exception
from troopai.adk.mcp.run_hooks_bridge import (
    fire_on_mcp_connect,
    fire_on_mcp_connected,
    fire_on_mcp_error,
)

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class MCPServerStdioParams:
    """Parameters for spawning a stdio MCP server subprocess.

    Attributes:
        command: The executable to run (e.g. ``"npx"`` or
            ``"/usr/local/bin/my-mcp-server"``).
        args: Command-line arguments passed to the executable.
        env: Additional environment variables. ``None`` (default)
            inherits the parent process's environment plus the MCP
            SDK's defaults; a non-``None`` dict is merged on top of
            the SDK defaults.
        cwd: Working directory for the subprocess. ``None`` (default)
            inherits the parent's CWD.
        encoding: Text encoding for stdio framing. ``"utf-8"`` is
            standard; servers using a different codepage need to set
            this to match.
    """

    command: str
    """Executable to run as the MCP server (resolved on ``PATH``)."""

    args: list[str] = field(default_factory=list)
    """Command-line arguments passed to ``command``."""

    env: dict[str, str] | None = None
    """Extra env vars merged with the SDK defaults; ``None`` inherits."""

    cwd: str | None = None
    """Working directory for the subprocess; ``None`` inherits."""

    encoding: str = "utf-8"
    """Text encoding for stdio framing; default ``"utf-8"`` matches MCP spec."""

    call_tool_timeout_seconds: float | None = None
    """Per-call read timeout for ``call_tool`` requests, in seconds.

    ``None`` (default) disables the timeout — the call waits
    indefinitely for the subprocess to respond. Set a finite value
    (e.g. ``120.0``) to bound how long a stalled or deadlocked
    subprocess can block the runner turn.
    """


class MCPServerStdio(MCPServerWithClientSession):
    """MCP server communicating over the stdin/stdout of a subprocess.

    The subprocess is spawned on ``connect()`` and terminated on
    ``cleanup()``. The session is initialised inside ``connect()`` so
    capabilities and the first tool list are available before the
    first ``list_tools`` call.

    Example::

        server = MCPServerStdio(
            name="filesystem",
            params=MCPServerStdioParams(
                command="npx",
                args=["@modelcontextprotocol/server-filesystem", "/tmp/scratch"],
            ),
        )
        async with server:
            tools = await server.list_tools()
    """

    def __init__(
        self,
        *,
        name: str,
        params: MCPServerStdioParams,
        cache_tools_list: bool = True,
        use_structured_content: bool = False,
        llm: Any | None = None,
        elicitation_callback: Any | None = None,
    ) -> None:
        """Initialise the stdio MCP server.

        Args:
            name: Human-readable server name used in log messages and
                error reports.
            params: Subprocess command, arguments, environment, and
                encoding parameters.
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
        """Spawn the subprocess and open an MCP ``ClientSession`` on it.

        Held under ``_connect_lock`` so concurrent ``connect()`` calls
        are serialised — only the first call actually spawns; later
        callers see the established session and return.
        """
        async with self._connect_lock:
            if self._session is not None:
                return
            await fire_on_mcp_connect(self._name)
            stack = AsyncExitStack()
            try:
                params = StdioServerParameters(
                    command=self._params.command,
                    args=self._params.args,
                    env=self._params.env,
                    cwd=self._params.cwd,
                    encoding=self._params.encoding,
                )
                read, write = await stack.enter_async_context(stdio_client(params))
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
                logger.info("MCP stdio server %r connected (%s)", self._name, self._params.command)
                await fire_on_mcp_connected(self._name)
            except (Exception, asyncio.CancelledError) as exc:
                # See ``MCPServerStreamableHttp.connect`` for the full
                # rationale: SDK-internal task-group cancellations
                # surface as ``CancelledError`` instead of the
                # underlying spawn/IO error; ``stack.aclose()`` then
                # propagates the leaf as a ``BaseExceptionGroup``.
                # Capture the close-time error so callers see a
                # concrete root cause via ``MCPConnectionError``.
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
                logger.error("MCP stdio server %r failed to connect: %s", self._name, root_exc)
                await fire_on_mcp_error(self._name, root_exc)
                raise MCPConnectionError(f"MCP stdio server '{self._name}' failed to connect: {root_exc}") from root_exc

    @override
    async def cleanup(self) -> None:
        """Cancel the notification subscriber and tear down the subprocess.

        Idempotent — safe to call before ``connect`` or after a prior
        ``cleanup``. Errors during teardown are logged but never
        raised so the caller's ``finally`` block is not blocked.

        Anyio's ``RuntimeError("Attempted to exit ...cancel scope...")``
        is demoted to DEBUG — anyio raises it when a background SDK
        task in ``stdio_client`` fails during async-generator cleanup,
        even when our caller did the right thing (LIFO disposal,
        same-task cleanup). The subprocess and streams are already
        torn down by the time the scope mismatch surfaces; treating
        it as an actionable warning produces false alarms. The
        prefix-match is intentional: the MCP SDK's
        ``"No active cancel scope"`` error (raised by
        ``RequestResponder`` outside its context) signals an
        in-flight-request abandonment with different resource
        semantics — that one stays at WARNING. Mirrors OpenAI Agents
        SDK ``MCPServerSession.cleanup``
        (``agents/mcp/server.py``).
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
                        "MCP stdio server %r cleanup: anyio cancel-scope mismatch (subprocess already torn down)",
                        self._name,
                    )
                else:
                    logger.warning("MCP stdio server %r cleanup raised", self._name, exc_info=True)
            except Exception:
                logger.warning("MCP stdio server %r cleanup raised", self._name, exc_info=True)
            finally:
                self._exit_stack = None
                logger.info("MCP stdio server %r disconnected", self._name)
