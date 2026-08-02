"""Tool wrapper for durable Restate execution.

Provides :func:`restate_tool` — a decorator/factory that routes tool
function calls through ``ctx.run()`` when inside a Restate handler, and
calls the function directly otherwise.

This mirrors the Temporal ``activity_tool`` pattern but uses Restate's
journaling primitive (``ctx.run``) instead of Temporal activities.

References:
    Restate Python SDK ctx.run docs:
    https://docs.restate.dev/develop/python/durable-execution#journaling-results
"""

from __future__ import annotations

import functools
import logging
from collections.abc import Callable
from typing import Any

from troopai.adk.workflows.restate.llm import get_restate_context

logger = logging.getLogger(__name__)


def restate_tool(fn: Callable[..., Any], *, name: str = "") -> Callable[..., Any]:
    """Wrap a tool function to route calls through Restate ``ctx.run()`` when inside a handler.

    When called from inside a Restate handler the function is executed inside
    ``ctx.run()`` so that its result is journaled.  On replay, Restate returns
    the recorded result without re-executing the function.

    Outside a handler the function is called directly — no overhead is added
    in non-durable paths.

    The wrapper preserves the original function's ``__name__``, ``__doc__``,
    ``__annotations__``, ``__module__``, ``__qualname__``, and ``__wrapped__``
    attributes so that schema generation and introspection tools see the
    original signature.

    Args:
        fn: An async callable to wrap with durable routing.
        name: Journal entry name used for ``ctx.run(name, ...)``.
            Defaults to ``fn.__name__`` when empty.

    Returns:
        An async callable with the same signature as *fn* that routes through
        Restate's journal when inside a handler.

    References:
        Restate ctx.run journaling:
        https://docs.restate.dev/develop/python/durable-execution#journaling-results
    """
    journal_name = name if len(name) > 0 else fn.__name__

    @functools.wraps(fn)
    async def _wrapper(*args: Any, **kwargs: Any) -> Any:
        ctx = get_restate_context()

        if ctx is None:
            logger.debug(
                "restate_tool %r: calling directly (outside Restate handler)",
                journal_name,
            )
            return await fn(*args, **kwargs)

        logger.info(
            "restate_tool %r: routing through Restate ctx.run()",
            journal_name,
        )

        async def _invoke() -> Any:
            return await fn(*args, **kwargs)

        return await ctx.run(journal_name, _invoke)

    _wrapper.__wrapped__ = fn  # type: ignore[attr-defined]
    return _wrapper
