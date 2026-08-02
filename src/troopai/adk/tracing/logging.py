"""Canonical structured-logging event names and a small emit helper.

Standardizes the *fields* downstream modules attach to log records — not
whether they log. Use with each module's own ``logging.getLogger(__name__)``.
"""

from __future__ import annotations

import logging
from typing import Any

EVENT_AGENT_TURN_START = "agent.turn.start"
EVENT_AGENT_TURN_END = "agent.turn.end"
EVENT_LLM_REQUEST = "llm.request"
EVENT_TOOL_CALL = "tool.call"
EVENT_HANDOFF = "handoff"
EVENT_GRAPH_NODE = "graph.node"
EVENT_SWARM_TURN = "swarm.turn"


def log_event(
    logger: logging.Logger,
    event: str,
    *,
    level: int = logging.INFO,
    **fields: Any,
) -> None:
    """Emit a structured log record carrying ``event`` + ``fields``.

    Fields land on the ``LogRecord`` via ``extra`` so structured handlers
    (JSON formatters, OTel log bridges) can key off them. ``**fields`` is
    the legitimate variadic use for arbitrary structured-log metadata
    (``agent_name``, ``turn``, ``model``, …) — not core parameters.

    Args:
        logger: The module logger to emit on.
        event: A canonical event name (use the ``EVENT_*`` constants).
        level: Logging level (default ``INFO``).
        **fields: Structured key/value pairs attached to the record via
            ``extra``; field names must not collide with reserved
            ``LogRecord`` attributes (e.g. ``name``, ``msg``, ``args``,
            ``levelname``, ``message``, ``created``) — stdlib ``logging``
            raises ``KeyError`` at emit time if they do.

            Canonical fields (used by framework call sites):

            - ``agent_name`` (str) — the agent emitting the event.
            - ``turn`` (int) — the turn index within the agent loop.
            - ``model`` (str) — the resolved model name for LLM events.
            - ``tenant_id`` (str | None) — the opaque tenant identifier
              from ``RunConfig.tenant_id``; enables per-tenant log
              filtering and routing. ``None`` on untenanted runs.
    """
    logger.log(level, event, extra={"event": event, **fields})
