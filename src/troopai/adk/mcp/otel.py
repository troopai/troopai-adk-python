"""OpenTelemetry trace-context injection for MCP requests.

The MCP protocol reserves a top-level ``_meta`` field on every request
for transport-agnostic metadata. This module's ``build_mcp_meta()``
returns a ``_meta`` dict pre-populated with W3C Trace Context headers
(``traceparent`` / ``tracestate``) when an OTel span is active on
the calling task.

Soft-import discipline mirrors ``tracing/otel/`` — the function
returns an empty dict when ``opentelemetry`` is not installed or
when no span is active, so MCP code never branches on whether OTel
is present.

Reference: https://www.w3.org/TR/trace-context/
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def build_mcp_meta(extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return a ``_meta`` dict for an outgoing MCP request.

    When an OTel span is active on the current task, injects the W3C
    Trace Context headers via the global text-map propagator. When
    OTel is not installed or no span is active, returns ``extra`` (or
    an empty dict). The ``_meta`` field is the protocol-level
    extensibility surface MCP servers honour for trace propagation.

    Args:
        extra: Optional dict merged into the result before injection.
            Provider-injected keys (``traceparent``, ``tracestate``)
            win on collision with ``extra``.

    Returns:
        A new dict suitable for passing as the ``_meta`` field of an
        MCP ``CallToolRequest`` or other request type. Empty when no
        trace context is available and ``extra`` is ``None``.
    """
    meta: dict[str, Any] = dict(extra) if extra is not None else {}

    try:
        from opentelemetry import propagate
    except ImportError:
        return meta

    try:
        propagate.get_global_textmap().inject(meta)
    except Exception:
        logger.debug("OTel trace-context injection failed", exc_info=True)

    return meta
