"""ContextVar bridge from the runner to MCP server lifecycle.

The MCP server's ``connect`` / ``cleanup`` methods need to fire
``RunHooks.on_mcp_*`` callbacks but never hold a direct reference to
the active hooks (the hooks belong to the run, not the server). This
module supplies a ``ContextVar`` the runner sets at the top of each
``arun()`` call so the server can pick them up without explicit
plumbing.

The same task that calls ``arun`` also runs lazy-connect inside
``MCPToolset.get_tools`` and disposal inside the runner's finally
block, so the ``ContextVar`` is automatically inherited across
those call sites.
"""

from __future__ import annotations

import logging
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from troopai.adk.hooks.hooks import RunHooks
    from troopai.adk.run.context import RunContext

logger = logging.getLogger(__name__)


active_run_hooks: ContextVar[RunHooks[Any] | None] = ContextVar("troopai_mcp_active_run_hooks", default=None)
"""Active ``RunHooks`` for the calling task. ``None`` outside a run."""

active_run_context: ContextVar[RunContext[Any] | None] = ContextVar("troopai_mcp_active_run_context", default=None)
"""Active ``RunContext`` for the calling task. ``None`` outside a run."""


async def fire_on_mcp_connect(server_name: str) -> None:
    """Fire ``on_mcp_connect`` if hooks are active on the current task.

    Args:
        server_name: The name of the MCP server about to connect.
    """
    hooks = active_run_hooks.get()
    ctx = active_run_context.get()
    if hooks is None or ctx is None:
        return
    try:
        await hooks.on_mcp_connect(ctx, server_name)
    except Exception:
        logger.warning("MCP hook on_mcp_connect raised", exc_info=True)


async def fire_on_mcp_connected(server_name: str) -> None:
    """Fire ``on_mcp_connected`` if hooks are active on the current task.

    Args:
        server_name: The name of the MCP server that just connected.
    """
    hooks = active_run_hooks.get()
    ctx = active_run_context.get()
    if hooks is None or ctx is None:
        return
    try:
        await hooks.on_mcp_connected(ctx, server_name)
    except Exception:
        logger.warning("MCP hook on_mcp_connected raised", exc_info=True)


async def fire_on_mcp_error(server_name: str, error: BaseException) -> None:
    """Fire ``on_mcp_error`` if hooks are active on the current task.

    Args:
        server_name: The name of the MCP server that encountered an error.
        error: The exception that caused the connection failure.
    """
    hooks = active_run_hooks.get()
    ctx = active_run_context.get()
    if hooks is None or ctx is None:
        return
    try:
        await hooks.on_mcp_error(ctx, server_name, error)
    except Exception:
        logger.warning("MCP hook on_mcp_error raised", exc_info=True)
