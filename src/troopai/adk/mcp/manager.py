"""Multi-server MCP lifecycle holder.

``MCPServerManager`` connects and cleans up a fixed set of
``MCPServer`` instances together. It is the right tool when an
agent talks to several MCP servers and you want a single
``async with`` to orchestrate them — typically when sharing a
manager across multiple agents in a swarm or graph topology.

Surfaces:

- ``connect_all`` / ``cleanup_all`` — fan-out lifecycle for every
  registered server; the connect path attempts every server before
  raising so partial fleets do not mask failures further down the list.
- ``acquire`` / ``release`` — ref-counted sharing across multiple
  ``MCPToolset`` instances; the underlying connection only closes
  when the last holder releases.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field

from troopai.adk.mcp.exceptions import MCPConnectionError
from troopai.adk.mcp.mcp_server import MCPServer

logger = logging.getLogger(__name__)


@dataclass
class MCPServerManager:
    """Lifecycle holder for a fixed list of ``MCPServer`` instances.

    Use as an async context manager for explicit lifecycle, or call
    ``connect_all()`` and ``cleanup_all()`` directly when integrating
    with a non-context-manager surface (e.g. tests).

    Example::

        manager = MCPServerManager(servers=[stdio_server, http_server])
        async with manager:
            agent = Agent(
                name="x",
                tools=[
                    MCPToolset(server=stdio_server, auto_connect=False),
                    MCPToolset(server=http_server, auto_connect=False),
                ],
            )
            await Runner.arun(agent, "hello")

    Attributes:
        servers: The MCP servers under management. Order determines
            connect order on parallel-False; on parallel-True every
            server starts concurrently.
    """

    servers: list[MCPServer] = field(default_factory=list)
    """MCP servers under management; order is the sequential connect order."""

    _connected: bool = field(default=False, init=False, repr=False)
    """Backing field for :attr:`is_active` — surfaced via the public
    property below. Callers read ``manager.is_active``, never
    ``manager._connected``."""

    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)
    """Synchronisation primitive — never observable; no public
    accessor by design. Same reasoning as ``MCPToolset._connect_lock``."""

    _ref_counts: dict[int, int] = field(default_factory=dict, init=False, repr=False)
    """Backing field for :meth:`get_ref_count` — per-server reference
    count keyed by ``id(server)``. Surfaced via the public method
    below so callers can audit shared-server topology without
    coupling to the dict layout."""

    # ------------------------------------------------------------------
    # Public state-observation surface
    # ------------------------------------------------------------------

    @property
    def is_active(self) -> bool:
        """Whether this manager is in the ``connect_all`` lifecycle state.

        ``True`` between a successful ``connect_all()`` and
        ``cleanup_all()``. Mutually exclusive with the
        ``acquire`` / ``release`` ref-counted lifecycle —
        ``acquire`` raises if ``is_active`` is ``True``.
        """
        return self._connected

    def get_ref_count(self, server: MCPServer) -> int:
        """Return the number of outstanding ``acquire`` calls for ``server``.

        Returns ``0`` when the server has no active references (or
        is not registered with this manager). Useful for auditing
        shared-server topology in multi-toolset deployments.
        """
        return self._ref_counts.get(id(server), 0)

    async def acquire(self, server: MCPServer) -> None:
        """Connect ``server`` (if not already) and bump its ref count.

        The companion ``release`` decrements; the underlying
        connection only closes when the last holder releases. Use
        when several ``MCPToolset`` instances share one server.

        ``acquire`` and ``connect_all`` operate on disjoint lifecycle
        models — do not mix them on the same manager. Calling
        ``acquire`` while the manager is in the ``connect_all``
        state raises ``MCPConnectionError`` to prevent a second
        connect on an already-live server.

        Args:
            server: The server to acquire. Must be registered in
                ``self.servers``.

        Raises:
            MCPConnectionError: If ``server`` is not registered with
                this manager, or if the manager is currently in the
                ``connect_all`` lifecycle state.
        """
        if server not in self.servers:
            raise MCPConnectionError(f"Server {server.name!r} is not registered with this manager")
        async with self._lock:
            if self._connected:
                raise MCPConnectionError(
                    f"MCPServerManager is in connect_all state; "
                    f"cannot acquire {server.name!r} via ref counting. "
                    "Use either connect_all/cleanup_all OR acquire/release, not both."
                )
            key = id(server)
            count = self._ref_counts.get(key, 0)
            if count == 0:
                await server.connect()
            self._ref_counts[key] = count + 1

    async def release(self, server: MCPServer) -> None:
        """Decrement ``server``'s ref count, cleaning up at zero.

        Calling ``release`` on a server with count 0 is a no-op
        (logged at DEBUG) so accidental double-releases are safe.

        Args:
            server: The server to release. Must have been previously
                acquired via ``acquire``.
        """
        async with self._lock:
            key = id(server)
            count = self._ref_counts.get(key, 0)
            if count <= 0:
                logger.debug(
                    "MCPServerManager.release on %r below zero — ignoring",
                    server.name,
                )
                return
            count -= 1
            if count == 0:
                self._ref_counts.pop(key, None)
                try:
                    await server.cleanup()
                except Exception:
                    logger.warning(
                        "MCPServerManager release cleanup failed for %r",
                        server.name,
                        exc_info=True,
                    )
            else:
                self._ref_counts[key] = count

    async def connect_all(self, *, parallel: bool = True) -> None:
        """Connect every server sequentially in declaration order.

        On exception, every successfully-connected server is cleaned
        up so the manager does not enter a half-connected state.
        Errors are accumulated across servers (every server is
        attempted) before raising the first failure.

        Args:
            parallel: Accepted for API compatibility but has no effect.
                Each ``MCPServer.connect()`` opens an anyio cancel scope
                anchored to the calling task; ``cleanup()`` must run in
                that same task or anyio raises a cross-task scope error.
                Spawning connect() calls in separate tasks violates this
                invariant, so all connects run sequentially in the
                calling task regardless of this flag.
        """
        del parallel  # Accepted for API compatibility; see docstring.
        async with self._lock:
            if self._connected:
                return
            errors: list[Exception] = []
            connected: list[MCPServer] = []

            for server in self.servers:
                try:
                    await server.connect()
                    connected.append(server)
                except asyncio.CancelledError:
                    # Honour structured-concurrency cancellation: roll back
                    # already-connected servers in LIFO order (the anyio
                    # cancel-scope invariant), then re-raise so the
                    # CancelledError propagates instead of being swallowed
                    # into an MCPConnectionError.
                    for s in reversed(connected):
                        try:
                            await s.cleanup()
                        except Exception:
                            logger.warning(
                                "MCPServerManager rollback cleanup failed for %r during cancellation",
                                s.name,
                                exc_info=True,
                            )
                    raise
                except Exception as exc:
                    errors.append(exc)

            if len(errors) > 0:
                # Roll back successfully-connected servers so the manager
                # does not advertise a half-connected fleet. Reversed
                # iteration honours the anyio LIFO cancel-scope invariant:
                # the last-connected server's scope sits highest on the
                # stack and must be exited first.
                for server in reversed(connected):
                    try:
                        await server.cleanup()
                    except Exception:
                        logger.warning(
                            "MCPServerManager rollback cleanup failed for %r",
                            server.name,
                            exc_info=True,
                        )
                raise MCPConnectionError(
                    f"MCPServerManager: {len(errors)} of {len(self.servers)} server(s) failed to connect: {errors[0]}"
                ) from errors[0]

            self._connected = True

    async def cleanup_all(self) -> None:
        """Clean up every server. Idempotent and exception-safe.

        Drains both lifecycles: the ``connect_all`` fleet and any
        server still held via ``acquire`` / ``ensure_connected`` ref
        counting. Servers cleaned up here have their ref counts
        cleared so no live connection is silently left open. Cleans up
        even if individual cleanups fail — the resulting state is "all
        servers asked to release". The first error encountered is
        logged at WARNING; cleanup never raises, mirroring the
        contract on ``Toolset.adispose``.

        Iteration is REVERSED so a server connected later (whose
        anyio cancel scope sits higher on the task's scope stack)
        is closed first. FIFO cleanup would attempt to pop a scope
        from underneath the active head and trigger
        ``RuntimeError: Attempted to exit a cancel scope that isn't
        the current tasks's current cancel scope``. Mirrors OpenAI
        Agents SDK ``MCPServerManager.cleanup_all``
        (``agents/mcp/manager.py``).
        """
        async with self._lock:
            if not self._connected and len(self._ref_counts) == 0:
                return
            # Reversed so a server connected later (whose anyio cancel scope
            # sits higher on the task's scope stack) is closed first. Drains
            # both lifecycles: the connect_all fleet and any server still
            # held via acquire/ensure_connected ref counting, so no live
            # connection is silently left open.
            for server in reversed(self.servers):
                if not self._connected and self._ref_counts.get(id(server), 0) == 0:
                    continue
                self._ref_counts.pop(id(server), None)
                try:
                    await server.cleanup()
                except Exception:
                    logger.warning(
                        "MCPServerManager cleanup failed for %r",
                        server.name,
                        exc_info=True,
                    )
            self._connected = False

    async def ensure_connected(self, server: MCPServer) -> None:
        """Connect ``server`` only if it is not already connected.

        Idempotent: if the server is already live (via ``acquire``
        or ``connect_all``), this is a no-op. Identity-checked
        against ``self.servers`` to guard against cross-manager
        confusion when multiple managers exist in one process.

        Args:
            server: The server to connect if not already live. Must
                be registered in ``self.servers``.

        Raises:
            MCPConnectionError: If ``server`` is not registered with
                this manager.
        """
        if server not in self.servers:
            raise MCPConnectionError(f"Server {server.name!r} is not registered with this manager")
        async with self._lock:
            if self._connected or self._ref_counts.get(id(server), 0) > 0:
                return
            await server.connect()
            self._ref_counts[id(server)] = 1

    async def __aenter__(self) -> MCPServerManager:
        await self.connect_all()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.cleanup_all()
