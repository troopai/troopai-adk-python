"""Per-tool rate-limit configuration.

Sliding-window rate limiter for ``FunctionTool`` invocations. The
config is provider-agnostic and lives under ``types/tools/`` because
every framework type lives under ``types/``.

The runtime state (timestamp deque, asyncio Lock) lives on the
``FunctionTool`` instance itself, accessed via the
``acquire_rate_slot()`` method. This config is the configuration
surface the developer interacts with.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ToolRateLimitBehavior = Literal["wait", "error"]
"""Behavior on rate-limit saturation.

- ``"wait"``: ``await asyncio.sleep(retry_after)`` until a slot opens,
  then proceed. The LLM does not see the saturation.
- ``"error"``: return an error result the LLM can react to (and
  potentially back off in its own logic).
"""


@dataclass(frozen=True, kw_only=True)
class ToolRateLimit:
    """Sliding-window rate-limit configuration for a ``FunctionTool``.

    The window is fixed at 60 seconds; rate is expressed as
    ``rpm`` (requests per minute). The implementation maintains a
    ``deque`` of monotonic-clock timestamps per tool instance, drops
    timestamps older than 60 seconds on each call, and admits the
    call when the deque length is below ``rpm``.

    Per-instance scope: each ``FunctionTool`` has its own deque. Two
    tools that share a backend rate budget should share a single
    ``FunctionTool`` instance (or compose middleware once that lands).

    Attributes:
        rpm: Calls allowed in any 60-second sliding window.
            Must be a positive integer.
        behavior: What happens when the window is saturated.
            Defaults to ``"wait"`` — keeps the LLM unaware. Set to
            ``"error"`` to surface saturation to the model so it can
            adapt (e.g. switch tools, summarise, defer).

    Example::

        from troopai.adk.tools import ToolRateLimit, function_tool


        @function_tool(
            name="search_api",
            description="Query the search API.",
            rate_limit=ToolRateLimit(rpm=30),
        )
        def search_api(query: str) -> str: ...
    """

    rpm: int
    """Calls allowed in any 60-second sliding window. Must be > 0."""

    behavior: ToolRateLimitBehavior = "wait"
    """What happens on saturation: ``"wait"`` (default) or ``"error"``."""

    max_wait_seconds: float | None = None
    """Optional cap on the cumulative wait this acquire may incur.

    When ``behavior="wait"`` and the executor's accumulated sleep would
    exceed this cap, the acquire falls back to ``"error"`` semantics —
    returns ``False`` so the LLM sees a clear rate-limit message rather
    than the run blocking indefinitely.

    Default ``None`` preserves the unbounded-wait behavior. Set this
    in multi-tenant deployments and any place where a long sleep would
    starve other coroutines on the event loop. Concrete worst-case
    without a cap: ``max_turns × 60s`` per saturated tool, multiplied
    by the number of concurrent waiters under parallel tool execution.

    Ignored when ``behavior="error"`` (which already short-circuits
    without sleeping).
    """

    def __post_init__(self) -> None:
        if self.rpm <= 0:
            raise ValueError(f"ToolRateLimit.rpm must be positive, got {self.rpm}")
        if self.max_wait_seconds is not None and self.max_wait_seconds <= 0:
            raise ValueError(f"ToolRateLimit.max_wait_seconds must be positive, got {self.max_wait_seconds}")
