"""RestateDurableEngine — concrete DurableEngine implementation for Restate.

Provides the :class:`RestateDurableEngine` facade that satisfies the
:class:`~troopai.adk.workflows.engine.DurableEngine` Protocol.

References:
    Restate Python SDK durable execution:
    https://docs.restate.dev/develop/python/durable-execution
    Restate ctx.run journaling:
    https://docs.restate.dev/develop/python/durable-execution#journaling-results
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from troopai.adk.workflows.engine import ModelActivityConfig, ToolActivityConfig

if TYPE_CHECKING:
    from troopai.adk.llms.llm import LLM
    from troopai.adk.tools.function_tool import FunctionTool

logger = logging.getLogger(__name__)


class RestateDurableEngine:
    """Concrete :class:`~troopai.adk.workflows.engine.DurableEngine` for Restate.

    Wraps LLMs via :class:`~troopai.adk.workflows.restate.llm.RestateLLM`.
    Tool wrapping re-uses the existing Restate tool helpers
    (:mod:`troopai.adk.workflows.restate.tools`) when available; callers may
    also wrap tools manually using ``ctx.run`` for full journaling semantics.

    Reports whether the current call stack is inside a Restate handler via
    :func:`~troopai.adk.workflows.restate.llm.get_restate_context`.

    References:
        Restate Python SDK:
        https://docs.restate.dev/develop/python
        Restate ctx.run:
        https://docs.restate.dev/develop/python/durable-execution#journaling-results
    """

    def wrap_llm(
        self,
        llm: LLM,
        *,
        config: ModelActivityConfig,
    ) -> LLM:
        """Wrap *llm* in a :class:`~troopai.adk.workflows.restate.llm.RestateLLM`.

        Args:
            llm: The :class:`~troopai.adk.llms.llm.LLM` instance to wrap.
            config: Timeout and retry policy carried for Protocol compatibility.

        Returns:
            A :class:`~troopai.adk.workflows.restate.llm.RestateLLM` that
            routes calls through ``ctx.run()`` when inside a Restate handler.

        References:
            Restate ctx.run journaling:
            https://docs.restate.dev/develop/python/durable-execution#journaling-results
        """
        from troopai.adk.workflows.restate.llm import RestateLLM

        return RestateLLM(wrapped=llm, activity_config=config)

    def wrap_tool(
        self,
        tool: FunctionTool,
        *,
        config: ToolActivityConfig,
    ) -> FunctionTool:
        """Return *tool* unchanged; Restate tool durability is caller-managed.

        Restate does not have a generic per-tool activity dispatch equivalent to
        Temporal's ``execute_activity``.  Tool calls inside a Restate handler
        are made durable by wrapping them individually in ``ctx.run()`` at the
        handler level.  This method returns the original tool so callers can
        compose their own wrapping strategy.

        Args:
            tool: The :class:`~troopai.adk.tools.function_tool.FunctionTool`
                to evaluate.
            config: Retained for Protocol compatibility; ignored.

        Returns:
            The original *tool* unmodified.

        References:
            Restate ctx.run:
            https://docs.restate.dev/develop/python/durable-execution#journaling-results
        """
        _ = config
        logger.debug(
            "RestateDurableEngine.wrap_tool: returning tool %r unchanged; "
            "wrap tool calls in ctx.run() at the handler level for durability",
            getattr(tool, "name", repr(tool)),
        )
        return tool

    def in_durable_context(self) -> bool:
        """Return ``True`` when called from inside a Restate handler.

        Returns:
            ``True`` if the current call stack is executing inside an active
            Restate handler; ``False`` if outside a handler or if the
            ``restate`` SDK is not installed.

        References:
            Restate current_context:
            https://docs.restate.dev/develop/python/durable-execution
        """
        from troopai.adk.workflows.restate.llm import get_restate_context

        return get_restate_context() is not None
