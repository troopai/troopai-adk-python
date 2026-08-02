"""MCP elicitation — let an MCP server request user input mid-call.

Some MCP servers need to ask the user a question while a tool call
is in progress (e.g. "which file did you mean?"). The MCP protocol
exposes this as an ``elicitation/create`` request from server to
client. This module supplies a callable Protocol the developer
implements to render the request and return the user's answer.

The default behaviour without an elicitation handler: the
``ClientSession`` returns a generic "not implemented" error to the
server. Servers that depend on elicitation will surface that
error; servers that treat it as optional will fall back gracefully.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mcp import types as mcp_types

logger = logging.getLogger(__name__)


ElicitationHandler = Callable[
    ["mcp_types.ElicitRequestParams"],
    Awaitable[Any],
]
"""Async callable invoked when an MCP server requests user input.

Receives ``ElicitRequestParams`` carrying the prompt and an optional
JSON Schema describing the expected response shape. Return the user's
response — a dict matching the schema, or a string when no schema is
set — to accept; return ``None`` to decline (the user refused). An
accepted value becomes ``ElicitResult(action="accept", content=...)``;
``None`` becomes ``ElicitResult(action="decline")`` so the server does
NOT proceed as though the request were approved.
"""


def make_elicitation_callback(handler: ElicitationHandler) -> Any:
    """Wrap a developer-supplied ``ElicitationHandler`` for the MCP SDK.

    The MCP SDK's ``ClientSession`` expects a callback returning
    ``ElicitResult | ErrorData``. This wrapper converts the
    framework-typed handler return value to that shape: a non-``None``
    return becomes ``action="accept"``; a ``None`` return becomes
    ``action="decline"`` (the user refused). Handler exceptions become
    ``ErrorData`` so the session stays alive.

    Args:
        handler: The developer-supplied async callable that renders the
            elicitation request to the user and returns their response.

    Returns:
        An async coroutine matching the MCP SDK's elicitation callback
        signature, ready to pass as ``elicitation_callback`` to
        ``ClientSession``.
    """

    async def _callback(
        ctx: Any, params: mcp_types.ElicitRequestParams
    ) -> mcp_types.ElicitResult | mcp_types.ErrorData:
        del ctx
        from mcp import types as mcp_types_

        try:
            raw = await handler(params)
            if raw is None:
                # An explicit refusal: the handler returning None MUST NOT be
                # coerced into an "accept" carrying the literal text "None" —
                # that would make the server proceed as if the user approved
                # the operation. Surface a decline so the server can react.
                logger.debug("MCP elicitation handler declined the request (returned None).")
                return mcp_types_.ElicitResult(action="decline")
            content: dict[str, Any] = raw if isinstance(raw, dict) else {"text": str(raw)}
            # ``ElicitResult.content`` is typed as a TypedDict with
            # narrow value types; the framework accepts ``dict[str, Any]``
            # and lets the SDK validate at the network boundary.
            return mcp_types_.ElicitResult(action="accept", content=content)  # type: ignore[arg-type]
        except Exception as exc:
            # Log full exception locally; do NOT include the message
            # text in the wire response. A malicious MCP server could
            # otherwise harvest internal stack details from exception
            # messages.
            logger.warning("MCP elicitation handler raised: %s", exc, exc_info=True)
            return mcp_types_.ErrorData(
                code=mcp_types_.INTERNAL_ERROR,
                message="elicitation failed",
            )

    return _callback
