"""TemporalDurableEngine — concrete DurableEngine implementation for Temporal.

Provides the :class:`TemporalDurableEngine` facade that satisfies the
:class:`~troopai.adk.workflows.engine.DurableEngine` Protocol.

References:
    Temporal Python SDK activity docs:
    https://docs.temporal.io/develop/python/core-application#develop-activities
    Temporal workflow API:
    https://python.temporal.io/temporalio.workflow.html
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from troopai.adk.workflows.engine import ModelActivityConfig, ToolActivityConfig

if TYPE_CHECKING:
    from troopai.adk.llms.llm import LLM
    from troopai.adk.tools.function_tool import FunctionTool

logger = logging.getLogger(__name__)


class TemporalDurableEngine:
    """Concrete :class:`~troopai.adk.workflows.engine.DurableEngine` for Temporal.

    Wraps LLMs via :class:`~troopai.adk.workflows.temporal.llm.TemporalLLM` and
    tools via :func:`~troopai.adk.workflows.temporal.tools.activity_tool`, and
    reports whether the current call stack is inside a Temporal workflow via
    :func:`~temporalio.workflow.in_workflow`.

    References:
        Temporal Python SDK:
        https://docs.temporal.io/develop/python
        Temporal workflow API:
        https://python.temporal.io/temporalio.workflow.html#in_workflow
    """

    def wrap_llm(
        self,
        llm: LLM,
        *,
        config: ModelActivityConfig,
    ) -> LLM:
        """Wrap *llm* in a :class:`~troopai.adk.workflows.temporal.llm.TemporalLLM`.

        Args:
            llm: The :class:`~troopai.adk.llms.llm.LLM` instance to wrap.
            config: Timeout and retry policy for the activity.

        Returns:
            A :class:`~troopai.adk.workflows.temporal.llm.TemporalLLM` that
            routes calls through Temporal when inside a workflow.

        References:
            Temporal activity options:
            https://python.temporal.io/temporalio.workflow.html#execute_activity
        """
        from troopai.adk.workflows.temporal.llm import TemporalLLM

        return TemporalLLM(wrapped=llm, activity_config=config)

    def wrap_tool(
        self,
        tool: FunctionTool,
        *,
        config: ToolActivityConfig,
    ) -> FunctionTool:
        """Wrap *tool* so it executes as a Temporal activity when inside a workflow.

        Delegates to
        :func:`~troopai.adk.workflows.temporal.tools.to_durable_tool`, which
        clones *tool* to preserve its real name and JSON schema and re-routes
        invocation through :func:`~temporalio.workflow.execute_activity` **by
        the tool's name** when inside a workflow.  The worker must register an
        activity under ``tool.name`` for the dispatch to resolve — the same
        contract used by
        :class:`~troopai.adk.workflows.temporal.mcp.TemporalMCPToolSet`.

        Args:
            tool: The :class:`~troopai.adk.tools.function_tool.FunctionTool` to wrap.
            config: Timeout and retry policy for the activity.

        Returns:
            A clone of *tool* whose invocation is durable inside a Temporal
            workflow, with the original name and schema intact.

        References:
            Temporal activity options:
            https://docs.temporal.io/develop/python/core-application#develop-activities
        """
        from datetime import timedelta

        from troopai.adk.workflows.temporal.tools import to_durable_tool

        return to_durable_tool(
            tool,
            start_to_close_timeout=timedelta(seconds=config.start_to_close_timeout),
            maximum_attempts=config.maximum_attempts,
        )

    def in_durable_context(self) -> bool:
        """Return ``True`` when called from inside a Temporal workflow.

        Returns:
            ``True`` if the current call stack is executing inside an active
            Temporal workflow; ``False`` if outside a workflow or if
            ``temporalio`` is not installed.

        References:
            Temporal in_workflow:
            https://python.temporal.io/temporalio.workflow.html#in_workflow
        """
        try:
            from temporalio import workflow

            return workflow.in_workflow()
        except ImportError:
            return False
