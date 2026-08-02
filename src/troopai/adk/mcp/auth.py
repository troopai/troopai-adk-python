"""Per-request auth header injection for HTTP MCP transports.

The ``HeaderProvider`` callable type and ``active_header_provider``
``ContextVar`` together give MCP HTTP transports a way to attach
fresh credentials to every outbound request without coupling
authentication to connection lifecycle.

Why a ContextVar (and not a session-scoped attribute):

A single ``MCPServerStreamableHttp`` instance may serve multiple
concurrent agent runs (different users, different tokens). A
session-scoped attribute would race; a per-call kwarg would force
every method on the public surface to plumb headers through. The
``ContextVar`` approach lets the toolset set the provider on the
calling task before delegating to the transport — concurrent tasks
each see their own provider with no shared state.

Adapted from Microsoft Agent Framework's ``_mcp_call_headers``
ContextVar pattern (`agent_framework/_mcp.py`), which solved the
same multi-tenant problem the same way.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextvars import ContextVar

from troopai.adk.utils import MaybeAwaitable

HeaderProvider = Callable[[], MaybeAwaitable[Mapping[str, str]]]
"""Callable returning headers to attach to the next MCP HTTP request.

Sync or async. Called per request so token rotation, request signing,
and other per-call credential schemes work without re-creating the
HTTP session. The returned mapping is merged into the transport's
default headers; provider keys win on collision.
"""


active_header_provider: ContextVar[HeaderProvider | None] = ContextVar(
    "troopai_mcp_active_header_provider",
    default=None,
)
"""Active ``HeaderProvider`` for the calling asyncio task.

Set transiently by ``MCPServerWithClientSession.call_tool`` when only a
constructor-time provider is configured, or by user code wrapping a tool
call; the transport reads it via ``get()``. Default is ``None`` —
transports fall back to the constructor-time ``header_provider`` common
to all transport types.

The variable is task-local: ``asyncio.create_task()`` inherits the
parent's value at creation time but updates do not propagate back,
so concurrent tool calls do not stomp on each other's headers.
"""
