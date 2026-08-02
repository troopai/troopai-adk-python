"""MCP server abstraction.

Two layers:

- ``MCPServer`` — the abstract contract every transport must satisfy.
  Defines async ``connect``/``cleanup``, ``list_tools``/``call_tool``,
  ``list_prompts``/``get_prompt``, and ``capabilities``.

- ``MCPServerWithClientSession`` — concrete base class that owns a
  ``mcp.ClientSession``, a serialised connect lock, a tool-list
  cache, and a notification subscriber. Concrete transports
  (``MCPServerStdio``, ``MCPServerStreamableHttp``) inherit from it
  and only implement the transport-specific ``connect``/``cleanup``
  pair.

The split keeps the public ABC tiny (so user-defined transports stay
small) while giving the framework's own transports a single source
of truth for caching, locking, and lifecycle.
"""

from __future__ import annotations

import abc
import asyncio
import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, override

from troopai.adk.mcp.auth import HeaderProvider, active_header_provider
from troopai.adk.mcp.exceptions import MCPConnectionError
from troopai.adk.mcp.notifications import make_message_handler
from troopai.adk.mcp.otel import build_mcp_meta

try:
    from mcp import ClientSession, Tool as MCPTool
    from mcp.types import (
        CallToolResult,
        GetPromptResult,
        ListPromptsResult,
        ListResourcesResult,
        ListResourceTemplatesResult,
        ReadResourceResult,
        ServerCapabilities,
    )
except ImportError as ie:
    raise ImportError(
        "Please install the 'mcp' extra to use MCPServer. You can use the `mcp` optional group - pip install troopai-adk-python[mcp]"
    ) from ie

if TYPE_CHECKING:
    from contextlib import AbstractAsyncContextManager

    from troopai.adk.llms import LLM

logger = logging.getLogger(__name__)


class MCPServer(abc.ABC):
    """Internal abstract base for MCP server transports.

    The abstract methods return mcp SDK wire types (``CallToolResult``,
    ``GetPromptResult``, etc.). User-facing framework types live in
    ``troopai.adk.types``; conversion happens in ``conversion.py``.

    Do NOT subclass ``MCPServer`` directly — extend
    ``MCPServerWithClientSession`` instead, which provides the
    ``ClientSession`` lifecycle, caching, and notification handling.
    ``MCPServer`` exists only so the concrete base class can be
    typed against the minimal contract.
    """

    def __init__(self, use_structured_content: bool = False):
        """
        Args:
            use_structured_content: Whether to use `tool_result.structured_content` when calling an
                MCP tool. Defaults to False because most MCP servers still include the structured
                content in the `tool_result.content`, so using it by default would cause duplicate
                content. You can set this to True if you know the
                server will not duplicate the structured content in the `tool_result.content`.
        """
        self.use_structured_content = use_structured_content

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """A readable name for the server."""

    @abc.abstractmethod
    async def connect(self) -> None:
        """Connect to the server. For example, this might mean spawning a subprocess or
        opening a network connection. The server is expected to remain connected until
        `cleanup()` is called.
        """

    @abc.abstractmethod
    async def cleanup(self) -> None:
        """Cleanup the server. For example, this might mean closing a subprocess or
        closing a network connection.
        """

    @abc.abstractmethod
    async def call_tool(
        self,
        name: str,
        args: Mapping[str, Any] | None = None,
    ) -> CallToolResult:
        """Invoke a tool on the server.

        Args:
            name: The name of the tool to invoke.
            args: Optional arguments to pass to the tool.

        Returns:
            The raw ``CallToolResult`` from the server, including
            ``isError`` and ``content``.
        """

    @abc.abstractmethod
    async def list_tools(self) -> list[MCPTool]:
        """List the tools available on the server."""

    @abc.abstractmethod
    async def list_prompts(self) -> ListPromptsResult:
        """List the prompts available on the server."""

    @abc.abstractmethod
    async def get_prompt(
        self,
        name: str,
        args: Mapping[str, Any] | None = None,
    ) -> GetPromptResult:
        """Get a specific prompt by name from the server.

        Args:
            name: The name of the prompt to retrieve.
            args: Optional arguments to pass when rendering the prompt.

        Returns:
            The rendered ``GetPromptResult`` containing the prompt messages.
        """

    @abc.abstractmethod
    def capabilities(self) -> ServerCapabilities:
        """Get the server capabilities."""

    async def list_resources(self) -> ListResourcesResult:
        """List the resources advertised by the server.

        Default implementation raises ``NotImplementedError`` so the
        ABC can stay minimal; ``MCPServerWithClientSession`` provides
        the concrete delegation.

        Returns:
            A ``ListResourcesResult`` containing the server's resource
            descriptors.
        """
        raise NotImplementedError

    async def list_resource_templates(self) -> ListResourceTemplatesResult:
        """List the resource templates advertised by the server.

        Default implementation raises ``NotImplementedError`` so the
        ABC can stay minimal; ``MCPServerWithClientSession`` provides
        the concrete delegation.

        Returns:
            A ``ListResourceTemplatesResult`` containing the server's
            resource template descriptors.
        """
        raise NotImplementedError

    async def read_resource(self, uri: str) -> ReadResourceResult:
        """Read a resource by URI.

        Default implementation raises ``NotImplementedError``;
        ``MCPServerWithClientSession`` provides the concrete delegation.

        Args:
            uri: The URI of the resource to read.

        Returns:
            A ``ReadResourceResult`` containing the resource's contents.
        """
        raise NotImplementedError


# ---------------------------------------------------------------------
# MCPServerWithClientSession — concrete shared base for stdio / HTTP
# ---------------------------------------------------------------------


class MCPServerWithClientSession(MCPServer):
    """Concrete base class that owns a ``ClientSession`` and a tool cache.

    Concrete transports (``MCPServerStdio``, ``MCPServerStreamableHttp``)
    extend this class and implement only ``connect`` (open the
    transport, call ``_attach_session``) and ``cleanup`` (cancel the
    notification task, close the session). All other methods are
    implemented here.

    Cache semantics: ``cache_tools_list=True`` (the default for the
    ``MCPToolset``) keeps the result of the most recent
    ``list_tools`` until either ``invalidate_tools_cache()`` is
    called manually or a server-side ``notifications/tools/list_changed``
    push notification triggers invalidation. Cache reads and writes
    are serialised by an ``asyncio.Lock`` so concurrent agent turns
    do not race on the dirty flag.

    Lifecycle invariant: ``cleanup`` MUST be called from the same
    asyncio task that called ``connect`` so the underlying anyio
    cancel-scope ends in the task it started in. The runner's
    auto-disposal path (``Runner._dispose_agent_toolsets``) preserves
    this invariant because both ``arun`` and ``_run_streamed_impl``
    run in a single task each.
    """

    def __init__(
        self,
        *,
        name: str,
        cache_tools_list: bool = True,
        use_structured_content: bool = False,
        header_provider: HeaderProvider | None = None,
        llm: LLM | None = None,
        elicitation_callback: Any | None = None,
    ) -> None:
        """Initialise the shared client-session base.

        Args:
            name: Human-readable server name used in log messages and
                error reports.
            cache_tools_list: When ``True`` (default), the tool list
                returned by ``list_tools`` is cached between turns and
                invalidated by push notifications or an explicit
                ``invalidate_tools_cache()`` call.
            use_structured_content: When ``True``, pass
                ``structuredContent`` through to the framework's
                artifact channel. Default ``False`` avoids duplicate
                content from servers that echo structured content into
                ``content`` as well.
            header_provider: Optional per-request callable returning
                auth headers. Stored as the constructor-time fallback
                for ``active_header_provider`` when no task-local
                provider is set.
            llm: Optional ``LLM`` instance to serve MCP
                ``sampling/createMessage`` requests. When ``None``,
                no sampling capability is advertised.
            elicitation_callback: Optional async callable invoked when
                the server sends an ``elicitation/create`` request.
                When ``None``, the session returns a "not implemented"
                response to the server.
        """
        super().__init__(use_structured_content=use_structured_content)
        self._name = name
        self._cache_tools_list = cache_tools_list
        self._header_provider = header_provider
        self._llm = llm
        self._elicitation_callback = elicitation_callback

        # Lifecycle state
        self._session: ClientSession | None = None
        self._connect_lock = asyncio.Lock()
        self._cache_lock = asyncio.Lock()
        self._tools_cache: list[MCPTool] | None = None
        self._cache_dirty: bool = False
        # Monotonic count of invalidate_tools_cache() calls. list_tools
        # snapshots it before the (awaiting) fetch to detect an invalidation
        # that arrives DURING the fetch — which a plain _cache_dirty=False
        # after the fetch would otherwise clobber.
        self._invalidations: int = 0
        self._transport_cm: AbstractAsyncContextManager[Any] | None = None

    # ------------------------------------------------------------------
    # Public ABC implementations
    # ------------------------------------------------------------------

    @property
    @override
    def name(self) -> str:
        return self._name

    @property
    def is_connected(self) -> bool:
        """Whether the underlying ``ClientSession`` is initialised.

        ``True`` between a successful ``connect()`` and the matching
        ``cleanup()``. Surfaces the otherwise-private
        ``_session is not None`` predicate so callers can poll
        connection state without touching the underscore field.
        """
        return self._session is not None

    @override
    async def list_tools(self) -> list[MCPTool]:
        """Return the server's tool list, hitting the cache when warm.

        Cache is consulted under ``_cache_lock``. On miss (first call,
        ``invalidate_tools_cache``, or ``notifications/tools/list_changed``)
        a fresh ``ClientSession.list_tools()`` is awaited and its
        ``tools`` array stored.
        """
        session = self._require_session()
        async with self._cache_lock:
            if self._cache_tools_list and self._tools_cache is not None and self._cache_dirty is False:
                return list(self._tools_cache)
            # Snapshot the invalidation counter BEFORE the await: an
            # invalidate_tools_cache() arriving DURING the fetch (e.g. a
            # tools/list_changed notification on another task) bumps the
            # counter, so the freshly-fetched list is already stale. Keep the
            # cache dirty in that case rather than clobbering the signal.
            invalidations_before = self._invalidations
            result = await session.list_tools()
            tools = list(result.tools)
            if self._cache_tools_list:
                self._tools_cache = tools
                self._cache_dirty = self._invalidations != invalidations_before
            return tools

    @override
    async def call_tool(
        self,
        name: str,
        args: Mapping[str, Any] | None = None,
    ) -> CallToolResult:
        """Invoke a tool, attaching W3C trace context to ``_meta``.

        The optional active ``HeaderProvider`` (set via the
        ``active_header_provider`` ContextVar) is consulted for
        per-call headers; HTTP transports read it on outbound
        requests. ``isError=True`` is left intact in the returned
        result; ``call_tool_result_to_str`` raises ``MCPToolCallError``
        when the converter sees the flag.

        Args:
            name: The name of the tool to invoke.
            args: Optional arguments forwarded to the server.

        Returns:
            The raw ``CallToolResult`` from the server. Callers inspect
            ``isError`` and ``content`` rather than this layer raising
            on errors.
        """
        session = self._require_session()
        meta = build_mcp_meta()
        meta_param = meta if len(meta) > 0 else None
        arguments = _args_dict_or_none(args)

        if self._header_provider is not None:
            token = active_header_provider.set(self._header_provider)
            try:
                return await session.call_tool(name, arguments=arguments, meta=meta_param)
            finally:
                active_header_provider.reset(token)
        return await session.call_tool(name, arguments=arguments, meta=meta_param)

    @override
    async def list_prompts(self) -> ListPromptsResult:
        return await self._require_session().list_prompts()

    @override
    async def list_resources(self) -> ListResourcesResult:
        return await self._require_session().list_resources()

    @override
    async def list_resource_templates(self) -> ListResourceTemplatesResult:
        return await self._require_session().list_resource_templates()

    @override
    async def read_resource(self, uri: str) -> ReadResourceResult:
        from pydantic import AnyUrl

        return await self._require_session().read_resource(AnyUrl(uri))

    @override
    async def get_prompt(
        self,
        name: str,
        args: Mapping[str, Any] | None = None,
    ) -> GetPromptResult:
        return await self._require_session().get_prompt(
            name,
            arguments=_args_dict_or_none(args),
        )

    @override
    def capabilities(self) -> ServerCapabilities:
        session = self._require_session()
        caps = session.get_server_capabilities()
        if caps is None:
            raise MCPConnectionError(f"MCP server '{self._name}' has no capabilities yet — session is not initialised.")
        return caps

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def invalidate_tools_cache(self) -> None:
        """Mark the tool cache stale so the next ``list_tools`` re-fetches.

        Safe to call from any task; sets a flag rather than mutating
        the cache itself, so concurrent ``list_tools`` calls remain
        consistent. Also bumps a monotonic counter so a ``list_tools``
        currently awaiting a fetch can detect that an invalidation arrived
        mid-fetch and avoid serving the now-stale result.
        """
        self._invalidations += 1
        self._cache_dirty = True

    # ------------------------------------------------------------------
    # Helpers for concrete subclasses
    # ------------------------------------------------------------------

    def _make_client_session(
        self,
        read: Any,
        write: Any,
        *,
        read_timeout_seconds: Any | None = None,
    ) -> ClientSession:
        """Construct an ``mcp.ClientSession`` wired to the configured callbacks.

        Concrete transports use this helper rather than calling
        ``ClientSession`` directly so a single call site decides
        which optional callbacks (sampling, elicitation, capabilities,
        message handler) flow into the session.

        Args:
            read: Readable stream from the MCP SDK transport context
                (type depends on the transport).
            write: Writable stream from the MCP SDK transport context.
            read_timeout_seconds: Optional read timeout forwarded to
                ``ClientSession``; ``None`` lets the SDK use its default.

        Returns:
            A configured ``ClientSession`` ready for ``initialize()``.
        """
        message_handler = make_message_handler(self)
        elicitation_cb = None
        if self._elicitation_callback is not None:
            from troopai.adk.mcp.elicitation import make_elicitation_callback

            elicitation_cb = make_elicitation_callback(self._elicitation_callback)

        if self._llm is not None:
            from troopai.adk.mcp.sampling import make_sampling_callback

            return ClientSession(
                read,
                write,
                read_timeout_seconds=read_timeout_seconds,
                sampling_callback=make_sampling_callback(self._llm),
                elicitation_callback=elicitation_cb,
                message_handler=message_handler,
            )
        return ClientSession(
            read,
            write,
            read_timeout_seconds=read_timeout_seconds,
            elicitation_callback=elicitation_cb,
            message_handler=message_handler,
        )

    async def _perform_handshake(self, session: ClientSession) -> None:
        """Run the session-establishment exchange before the session goes live.

        Called once per ``connect()`` cycle, just before the session is
        stored and the tool cache is cleared. The default implementation
        calls ``session.initialize()``, which is required for the current
        stateful MCP protocol.

        Subclasses may override this method to run a different or no-op
        handshake when connecting to a transport that does not require
        the initialize/initialized exchange.

        Args:
            session: A freshly constructed ``ClientSession`` from
                ``_make_client_session``, not yet initialised.
        """
        await session.initialize()

    async def _attach_session(self, session: ClientSession) -> None:
        """Initialise and store the session. Called by concrete ``connect``.

        The push-notification handler was injected at construction
        time via ``_make_client_session``; this method only initialises
        the session and clears any stale cache from a prior connect.

        Args:
            session: A freshly constructed (not yet initialised)
                ``ClientSession`` from ``_make_client_session``.
        """
        await self._perform_handshake(session)
        self._session = session
        self._tools_cache = None
        self._cache_dirty = False
        logger.info("MCP server %r session initialised", self._name)

    async def _detach_session(self) -> None:
        """Tear down session-level state. Called by concrete ``cleanup``.

        Idempotent: re-calling after detach is a no-op.
        Push notifications are driven inline by the mcp SDK via the
        ``message_handler`` callback injected in ``_make_client_session``;
        no background task is required here.
        """
        self._session = None
        self._tools_cache = None
        self._cache_dirty = False

    def _require_session(self) -> ClientSession:
        """Return the active session or raise ``MCPConnectionError``.

        Returns:
            The live ``ClientSession``.

        Raises:
            MCPConnectionError: If the server is not connected.
        """
        if self._session is None:
            raise MCPConnectionError(
                f"MCP server '{self._name}' is not connected. Call connect() (or use MCPToolset's auto_connect=True)."
            )
        return self._session

    # ------------------------------------------------------------------
    # Async context-manager protocol — supports `async with server:` DX
    # ------------------------------------------------------------------

    async def __aenter__(self) -> MCPServerWithClientSession:
        await self.connect()
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.cleanup()


def _args_dict_or_none(
    args: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Coerce a ``Mapping`` to ``dict`` (or pass ``None`` through).

    The MCP SDK's ``call_tool`` signature requires ``dict[str, Any]``;
    callers may pass any ``Mapping`` and we normalise here to keep
    the public interface flexible.

    Args:
        args: A mapping of tool arguments, or ``None`` for a
            parameterless tool call.

    Returns:
        A plain ``dict`` copy of ``args``, or ``None`` when ``args``
        is ``None``.
    """
    if args is None:
        return None
    return dict(args)


def extract_first_exception(exc: BaseException) -> BaseException:
    """Walk into a ``BaseExceptionGroup`` and return the first leaf.

    Used by every transport's ``connect()`` error path: the SDK's
    anyio task groups wrap underlying I/O errors
    (``httpcore.ConnectError``, ``OSError``, …) in
    ``ExceptionGroup`` / ``BaseExceptionGroup``. Surfacing the leaf
    gives callers a concrete root cause via
    ``raise MCPConnectionError(...) from extract_first_exception(eg)``.

    Walks recursively (groups can be nested). Returns ``exc``
    unchanged when it is already a leaf.

    Args:
        exc: A ``BaseException`` — either a plain exception or a
            (possibly nested) ``BaseExceptionGroup``.

    Returns:
        The first non-group leaf exception in the tree, or ``exc``
        itself when it is already a leaf.
    """
    while isinstance(exc, BaseExceptionGroup) and len(exc.exceptions) > 0:
        exc = exc.exceptions[0]
    return exc
