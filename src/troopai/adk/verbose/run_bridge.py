"""ContextVar bridge from the runner to the streaming chunk emitter.

The Rich ``Live`` widget that drives the CrewAI-style streaming panel
needs the current run's ``RunHooks`` and ``Agent`` instances on every
LLM stream chunk, but the streaming hot path
(:func:`troopai.adk.run.llm_calls.call_llm_streamed`) intentionally
does not take a verbose-specific argument — adding one would ripple
into every middleware and break the API. This module supplies two
``ContextVar`` slots the runner sets at the top of its
``emit_stream_start`` call so the streaming loop can read them
without explicit plumbing.

The same task that calls ``call_llm_streamed`` also runs the loop's
``finally`` clause, so the ContextVar is automatically inherited
across yields. Mirrors the MCP run-hooks bridge at
:mod:`troopai.adk.mcp.run_hooks_bridge`.
"""

from __future__ import annotations

import logging
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from troopai.adk.agents.agent import Agent

logger = logging.getLogger(__name__)


active_verbose_hooks: ContextVar[Any | None] = ContextVar(
    "troopai_verbose_active_hooks",
    default=None,
)
"""Active ``RunHooks`` chain for the calling task. ``None`` outside a
verbose-enabled streaming bracket."""


active_verbose_agent: ContextVar[Agent | None] = ContextVar(
    "troopai_verbose_active_agent",
    default=None,
)
"""Active :class:`Agent` for the calling task. ``None`` outside a
verbose-enabled streaming bracket."""


def fire_stream_chunk(accumulated_text: str, call_type: str) -> None:
    """Forward an accumulated streaming chunk to :func:`emit_stream_chunk`.

    Safe to call from inside the streaming loop without checking
    whether verbose is active — a no-op when either ContextVar is
    ``None`` (outside a bracket, or verbose disabled).

    .. warning::

        ContextVars are inherited by ``asyncio.create_task()`` children
        but the parent's ``finally`` ``ContextVar.reset()`` only resets
        the parent task's view. A tool handler that spawns a background
        task and outlives the streaming bracket may see stale
        ``active_verbose_hooks`` / ``active_verbose_agent``. The
        ``emit_stream_chunk`` path defensively swallows exceptions at
        DEBUG, so a stale fire is at worst a spurious Live update on a
        finished run — never a crash. Do NOT call ``fire_stream_chunk``
        from tool-spawned background tasks; the ContextVar contract is
        scoped to the streaming bracket only.

    Args:
        accumulated_text: Full text accumulated since the stream
            opened (not just the latest delta).
        call_type: ``"text"`` or ``"tool_call"``; forwarded to the
            Live panel to pick title and border colour.
    """
    hooks = active_verbose_hooks.get()
    agent = active_verbose_agent.get()
    if hooks is None or agent is None:
        return
    try:
        from troopai.adk.verbose.hooks import emit_stream_chunk

        emit_stream_chunk(hooks, agent, accumulated_text, call_type)
    except Exception:
        logger.debug("fire_stream_chunk raised", exc_info=True)
