"""Push-driven invalidation of the cached tool list.

When an MCP server's tool catalogue changes (a plugin loaded, a
permission revoked), the server emits a ``notifications/tools/list_changed``
JSON-RPC notification. ``handle_tools_list_changed`` is the message
handler that flips ``MCPServerWithClientSession._cache_dirty`` so the next
``list_tools`` call re-fetches.

Compared with polling, push invalidation is zero-cost on the steady
state and immediate on changes — the most expensive operation
(re-listing) only runs when the server says it has changed. The MCP
SDK's ``ClientSession`` accepts a ``message_handler`` at
construction; this module supplies one.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from troopai.adk.mcp.mcp_server import MCPServerWithClientSession

logger = logging.getLogger(__name__)


def make_message_handler(
    server: MCPServerWithClientSession,
) -> Any:
    """Return a ``ClientSession`` message handler bound to ``server``.

    The handler watches for ``notifications/tools/list_changed`` and
    flips ``server._cache_dirty`` so the next ``list_tools`` call
    re-fetches. All other notifications and request types are passed
    through unchanged. Exceptions are logged at WARNING and swallowed
    so a malformed notification cannot crash the session.

    Args:
        server: The server whose cache should be invalidated on push.

    Returns:
        A coroutine matching the MCP SDK's ``MessageHandlerFnT``
        signature.
    """

    async def _handler(message: Any) -> None:
        try:
            method = _extract_notification_method(message)
            if method == "notifications/tools/list_changed":
                server.invalidate_tools_cache()
                logger.debug(
                    "MCP server %r tools cache invalidated by push notification",
                    server.name,
                )
        except Exception:
            logger.warning("MCP message handler raised on server %r", server.name, exc_info=True)

    return _handler


def _extract_notification_method(message: Any) -> str | None:
    """Return the ``method`` field of an MCP notification, or ``None``.

    The MCP SDK's incoming-message wrapper exposes the inner
    notification under different attribute names across versions
    (``message.root``, ``message.notification``, or directly on the
    object). This helper tries each shape and returns ``None`` when
    none match — the caller treats unknown shapes as "not the
    notification we care about".

    Args:
        message: An incoming MCP message object as received by the
            ``message_handler`` callback.

    Returns:
        The ``method`` string (e.g. ``"notifications/tools/list_changed"``)
        when found, or ``None`` when the message does not carry a
        recognised method field.
    """
    candidate = getattr(message, "root", None)
    if candidate is not None and hasattr(candidate, "method"):
        return getattr(candidate, "method", None)
    if hasattr(message, "method"):
        return getattr(message, "method", None)
    notification = getattr(message, "notification", None)
    if notification is not None:
        return getattr(notification, "method", None)
    return None
