"""Streamable HTTP transport MCP server.

Connects to an MCP server over HTTP using the modern
``streamablehttp_client`` (HTTP POST + SSE for server-pushed
messages). Per-request header injection is implemented via an httpx
request event hook that reads from the
``active_header_provider`` ``ContextVar`` — concurrent agent runs
each see their own headers without cross-contamination.

Reference: https://modelcontextprotocol.io/specification/2025-06-18/basic/transports#streamable-http
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, override

import httpx
from mcp.client.streamable_http import streamable_http_client, streamablehttp_client

from troopai.adk.mcp.auth import HeaderProvider, active_header_provider
from troopai.adk.mcp.exceptions import MCPConnectionError
from troopai.adk.mcp.mcp_server import MCPServerWithClientSession, extract_first_exception
from troopai.adk.mcp.run_hooks_bridge import (
    fire_on_mcp_connect,
    fire_on_mcp_connected,
    fire_on_mcp_error,
)

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class MCPServerStreamableHttpParams:
    """Parameters for a streamable HTTP MCP transport.

    Attributes:
        url: The MCP server endpoint URL (typically ending in ``/mcp``).
        headers: Static headers attached to every request. Merged
            with any per-request headers from ``header_provider``;
            provider headers win on key collision.
        header_provider: Optional callable returning per-request
            headers. Called fresh on every outbound HTTP request,
            so token rotation works without re-creating the session.
            May be sync or async.
        timeout_seconds: Per-request timeout (HTTP-level). Default
            30 s matches the MCP SDK's default.
        sse_read_timeout_seconds: Maximum time the client will wait
            for the next SSE event before giving up. Default 300 s
            handles long-running tool calls.
        terminate_on_close: When ``True`` (default), send the MCP
            session-terminate signal on cleanup so the server can
            release resources. Set ``False`` only if your server
            needs to keep the session alive across reconnects.
    """

    url: str
    """MCP server endpoint URL (e.g. ``"https://api.example.com/mcp"``)."""

    headers: dict[str, str] | None = field(default=None, repr=False)
    """Static headers attached to every request. ``repr=False`` so
    bearer tokens cannot leak through ``str(params)`` or default
    logging."""

    header_provider: HeaderProvider | None = None
    """Optional per-request callable returning fresh headers (sync or async)."""

    timeout_seconds: float = 30.0
    """HTTP-level per-request timeout (default 30 s, matches MCP SDK)."""

    sse_read_timeout_seconds: float = 300.0
    """Max wait between SSE events (default 300 s, accommodates long calls)."""

    terminate_on_close: bool = True
    """Send MCP session-terminate on cleanup so the server releases resources."""

    idle_timeout: timedelta | None = None
    """Maximum time an idle keep-alive connection may remain open before the
    HTTP client closes it.  Wired via ``httpx.Limits.keepalive_expiry`` on
    the underlying httpx client pool — the installed MCP SDK (1.27.2) has no
    client-side idle-timeout parameter of its own.  ``None`` (default) leaves
    httpx's built-in pool defaults unchanged, preserving existing behaviour."""

    httpx_client: httpx.AsyncClient | None = field(default=None, repr=False)
    """Pre-built ``httpx.AsyncClient`` to use for all HTTP requests.

    Configure timeouts and pool limits on the client itself —
    ``timeout_seconds`` and the other client-construction fields do not
    apply on this path (``headers``/``idle_timeout`` are rejected
    outright when combined with a pre-built client).

    When set, the server's ``connect()`` passes the client directly to
    ``streamable_http_client(http_client=...)`` (the non-deprecated SDK
    entry point) instead of constructing a new client via
    ``_build_http_client``.  The caller is responsible for the client's
    lifecycle — the SDK will close it as part of the transport context.

    Mutually exclusive with ``httpx_client_factory``: setting both raises
    ``ValueError`` at construction time.

    ``repr=False`` so credentials embedded in custom clients cannot leak
    through ``str(params)`` or default logging."""

    httpx_client_factory: Any | None = None
    """Optional factory callable used to construct the ``httpx.AsyncClient``.

    Conforms to ``mcp.shared._httpx_utils.McpHttpClientFactory`` —
    ``(headers, timeout, auth) -> httpx.AsyncClient``.  When ``None``
    (the default), the server builds its own client via
    ``_build_http_client`` which installs the dynamic-header event hook
    for per-request token rotation.

    Mutually exclusive with ``httpx_client``: setting both raises
    ``ValueError`` at construction time."""

    def __post_init__(self) -> None:
        if self.httpx_client is not None and self.httpx_client_factory is not None:
            raise ValueError(
                "MCPServerStreamableHttpParams: 'httpx_client' and 'httpx_client_factory' "
                "are mutually exclusive — supply at most one."
            )
        if self.httpx_client is not None:
            # A pre-built client carries its own headers/limits; these
            # fields would be silently ignored on that path, so reject
            # the ambiguous combination outright.
            ignored = [
                name
                for name, value in (("headers", self.headers), ("idle_timeout", self.idle_timeout))
                if value is not None
            ]
            if len(ignored) > 0:
                raise ValueError(
                    f"MCPServerStreamableHttpParams: {', '.join(ignored)} cannot be combined with "
                    f"'httpx_client' — configure them on the client you pass in instead."
                )


class MCPServerStreamableHttp(MCPServerWithClientSession):
    """MCP server over the streamable HTTP transport.

    Per-request header injection lets a single connection serve
    rotating tokens. The HTTP client is built once at ``connect``
    time with both static ``headers`` and the request event hook
    that consults ``active_header_provider`` on every call.

    Example::

        server = MCPServerStreamableHttp(
            name="github",
            params=MCPServerStreamableHttpParams(
                url="https://api.example.com/mcp",
                header_provider=lambda: {"Authorization": f"Bearer {fresh_token()}"},
            ),
        )
        async with server:
            tools = await server.list_tools()
    """

    def __init__(
        self,
        *,
        name: str,
        params: MCPServerStreamableHttpParams,
        cache_tools_list: bool = True,
        use_structured_content: bool = False,
        llm: Any | None = None,
        elicitation_callback: Any | None = None,
    ) -> None:
        """Initialise the streamable HTTP MCP server.

        Args:
            name: Human-readable server name used in log messages and
                error reports.
            params: Connection and timeout parameters for the HTTP
                transport.
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
            header_provider=params.header_provider,
            llm=llm,
            elicitation_callback=elicitation_callback,
        )
        self._params = params
        self._exit_stack: AsyncExitStack | None = None

    @override
    async def connect(self) -> None:
        """Open the HTTP client and initialise the MCP session.

        Three client-construction paths are available:

        1. ``params.httpx_client`` is set — the pre-built client is
           forwarded directly to the non-deprecated
           ``streamable_http_client(http_client=...)`` entry point.
           The SDK manages the client's context; the caller owns its
           lifecycle.
        2. ``params.httpx_client_factory`` is set — the provided
           factory is passed to the ``streamablehttp_client`` wrapper
           (the SDK marks it deprecated; it remains the only entry
           point that accepts a factory, so the factory path uses it).
           Use this when client construction needs the SDK-composed
           headers/timeout rather than a pre-built client.
        3. Neither is set (default) — ``_build_http_client`` is used,
           which installs the dynamic-header event hook for per-request
           token rotation via ``active_header_provider``.
        """
        async with self._connect_lock:
            if self._session is not None:
                return
            await fire_on_mcp_connect(self._name)
            stack = AsyncExitStack()
            try:
                if self._params.httpx_client is not None:
                    # Path 1: caller supplied a pre-built client.
                    # Use the non-deprecated streamable_http_client which
                    # accepts http_client= directly; the SDK manages the
                    # context and closes the client on exit.
                    streams = await stack.enter_async_context(
                        streamable_http_client(
                            self._params.url,
                            http_client=self._params.httpx_client,
                            terminate_on_close=self._params.terminate_on_close,
                        )
                    )
                else:
                    # Path 2 / 3: factory-based (deprecated wrapper keeps
                    # timeout/headers/SSE params in one call site).
                    factory = self._params.httpx_client_factory or self._build_http_client
                    streams = await stack.enter_async_context(
                        streamablehttp_client(
                            self._params.url,
                            headers=self._params.headers,
                            timeout=timedelta(seconds=self._params.timeout_seconds),
                            sse_read_timeout=timedelta(seconds=self._params.sse_read_timeout_seconds),
                            terminate_on_close=self._params.terminate_on_close,
                            httpx_client_factory=factory,
                        )
                    )
                read, write, *_ = streams
                session = await stack.enter_async_context(
                    self._make_client_session(
                        read,
                        write,
                        read_timeout_seconds=timedelta(seconds=self._params.sse_read_timeout_seconds),
                    )
                )
                await self._attach_session(session)
                self._exit_stack = stack
                logger.info("MCP HTTP server %r connected (%s)", self._name, self._params.url)
                await fire_on_mcp_connected(self._name)
            except (Exception, asyncio.CancelledError) as exc:
                # CancelledError handling: anyio task groups inside the
                # MCP SDK cancel the parent task on internal failure
                # (e.g. ``httpcore.ConnectError`` from a background HTTP
                # task). The cancellation surfaces here as
                # ``asyncio.CancelledError``; ``stack.aclose()`` then
                # propagates the underlying error as an
                # ``ExceptionGroup`` from the SDK's task group. We
                # close the stack defensively so a more specific
                # error from there supersedes the CancelledError, and
                # always end with ``MCPConnectionError`` so callers
                # have a stable exception type. External cancellation
                # (``task.cancelling() > 0``) is re-raised intact.
                root_exc: BaseException = exc
                try:
                    await stack.aclose()
                except (Exception, BaseExceptionGroup) as close_exc:
                    # Prefer the cleanup error when the outer was a
                    # bare CancelledError — the group typically carries
                    # the actual httpx / httpcore root cause.
                    if isinstance(exc, asyncio.CancelledError):
                        root_exc = extract_first_exception(close_exc)
                if isinstance(root_exc, asyncio.CancelledError):
                    task = asyncio.current_task()
                    if task is not None and task.cancelling() > 0:
                        raise
                logger.error("MCP HTTP server %r failed to connect: %s", self._name, root_exc)
                await fire_on_mcp_error(self._name, root_exc)
                raise MCPConnectionError(f"MCP HTTP server '{self._name}' failed to connect: {root_exc}") from root_exc

    @override
    async def cleanup(self) -> None:
        """Close the session and HTTP client; idempotent.

        Anyio's ``RuntimeError("Attempted to exit ...cancel scope...")``
        is demoted to DEBUG. Same rationale as
        ``MCPServerStdio.cleanup``: the SDK opens an internal task
        group whose background cancellation surfaces as a scope
        mismatch on close. ``startswith("Attempted to exit")`` is
        the deliberate narrow filter — the MCP SDK's
        ``"No active cancel scope"`` is left at WARNING.
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
                        "MCP HTTP server %r cleanup: anyio cancel-scope mismatch (HTTP client already torn down)",
                        self._name,
                    )
                else:
                    logger.warning("MCP HTTP server %r cleanup raised", self._name, exc_info=True)
            except Exception:
                logger.warning("MCP HTTP server %r cleanup raised", self._name, exc_info=True)
            finally:
                self._exit_stack = None
                logger.info("MCP HTTP server %r disconnected", self._name)

    # ------------------------------------------------------------------
    # HTTP client construction with per-request header injection
    # ------------------------------------------------------------------

    def _build_http_client(
        self,
        headers: dict[str, str] | None = None,
        timeout: httpx.Timeout | None = None,
        auth: httpx.Auth | None = None,
    ) -> httpx.AsyncClient:
        """``McpHttpClientFactory`` implementation.

        The MCP SDK owns the client lifecycle and supplies composed
        ``headers`` / ``timeout`` / ``auth`` derived from the
        ``streamablehttp_client(...)`` call. We install the dynamic
        header event hook so per-request headers from
        :data:`active_header_provider` apply on every outbound call —
        a single client handles per-call token rotation without
        forcing a reconnect.

        When :attr:`MCPServerStreamableHttpParams.idle_timeout` is set,
        an ``httpx.Limits`` object is passed with ``keepalive_expiry`` equal
        to that duration.  When ``idle_timeout`` is ``None`` (the default),
        the ``limits`` kwarg is omitted entirely so httpx applies its own
        built-in pool defaults (``max_connections=100``,
        ``max_keepalive_connections=20``).

        Args:
            headers: Static headers composed by the MCP SDK from the
                ``streamablehttp_client`` call parameters.
            timeout: Composed timeout object from the SDK; ``None``
                lets httpx use its default.
            auth: Composed auth object from the SDK; ``None`` for no
                authentication.

        Returns:
            An ``httpx.AsyncClient`` with the dynamic-header event
            hook installed.
        """
        # Only pass `limits` when idle_timeout is explicitly set.  Passing
        # httpx.Limits() (even with all defaults) overrides httpx's built-in
        # pool limits (max_connections=100, max_keepalive_connections=20) with
        # unlimited values, which is a silent resource regression for callers
        # who do not set idle_timeout.
        if self._params.idle_timeout is not None:
            # httpx.Limits defaults max_connections / max_keepalive_connections
            # to None (unbounded) unless given explicitly, so we must restate
            # httpx's own bounded defaults alongside keepalive_expiry — passing
            # keepalive_expiry alone would silently uncap the pool.
            return httpx.AsyncClient(
                headers=headers or {},
                timeout=timeout,
                auth=auth,
                limits=httpx.Limits(
                    max_connections=100,
                    max_keepalive_connections=20,
                    keepalive_expiry=self._params.idle_timeout.total_seconds(),
                ),
                event_hooks={"request": [_inject_dynamic_headers]},
            )
        return httpx.AsyncClient(
            headers=headers or {},
            timeout=timeout,
            auth=auth,
            event_hooks={"request": [_inject_dynamic_headers]},
        )


async def _inject_dynamic_headers(request: httpx.Request) -> None:
    """Apply headers from the active ``HeaderProvider`` to ``request``.

    Sync providers are called inline; async providers are awaited.
    Provider exceptions are logged and swallowed so a transient
    auth failure does not crash the entire request lifecycle —
    callers see the underlying server response (typically 401).

    Args:
        request: The outbound ``httpx.Request`` whose headers will be
            mutated in place with values from the provider.
    """
    provider = active_header_provider.get()
    if provider is None:
        return
    try:
        result = provider()
        if inspect.isawaitable(result):
            headers = await result
        else:
            headers = result
    except Exception:
        logger.warning(
            "MCP header_provider raised; sending request without dynamic headers",
            exc_info=True,
        )
        return
    for key, value in headers.items():
        request.headers[key] = value
