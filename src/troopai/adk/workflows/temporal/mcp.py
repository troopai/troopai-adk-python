"""MCP tool execution routed through Temporal activities.

:class:`TemporalMCPToolSet` treats each MCP server as a pair of Temporal
activities — one for listing tools and one for calling a tool — so that MCP
I/O is durable, retried, and tracked in the Temporal event history.

Each instance holds the MCP server identifier, its connection parameters, and
the activity timeout / retry settings.  At runtime the two
``*_in_workflow`` methods dispatch through
:func:`temporalio.workflow.execute_activity` using the per-instance activity
name convention.

References:
    Temporal Python SDK — execute_activity:
    https://python.temporal.io/temporalio.workflow.html#execute_activity
    MCP Python SDK — client overview:
    https://github.com/modelcontextprotocol/python-sdk
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class TemporalMCPToolSet:
    """MCP server descriptor that routes tool calls through Temporal activities.

    Each MCP server is represented as two Temporal activities whose names are
    derived from :attr:`name`.  Calling :meth:`list_tools_in_workflow` or
    :meth:`call_tool_in_workflow` dispatches through
    :func:`~temporalio.workflow.execute_activity`, making MCP calls durable
    and replay-safe.

    Attributes:
        name: MCP server identifier.  Used as the prefix for both activity
            names (see :attr:`list_tools_activity_name` and
            :attr:`call_tool_activity_name`).
        connection_params: MCP connection parameters forwarded to the
            activity worker when establishing the MCP session.
        start_to_close_timeout: Maximum seconds allowed for a single activity
            attempt.  Defaults to 30.
        maximum_attempts: Total attempts, including the first, for each MCP
            activity.  ``1`` disables retries.  Defaults to ``1`` (no
            retries); raise it to opt into automatic re-runs of the list /
            call activity.
    """

    name: str
    """MCP server identifier used as the activity name prefix."""

    connection_params: dict[str, Any] = field(default_factory=dict)
    """MCP connection parameters forwarded to the activity worker."""

    start_to_close_timeout: int = 30
    """Maximum seconds for a single activity attempt."""

    maximum_attempts: int = 1
    """Total attempts including the first for each MCP activity.

    ``1`` (default) disables retries so an MCP call is never re-run unless
    the developer opts in; raise it to enable automatic re-runs.
    """

    def __post_init__(self) -> None:
        if self.start_to_close_timeout <= 0:
            raise ValueError(f"start_to_close_timeout must be > 0, got {self.start_to_close_timeout}")
        if self.maximum_attempts < 1:
            raise ValueError(f"maximum_attempts must be >= 1, got {self.maximum_attempts}")

    @property
    def list_tools_activity_name(self) -> str:
        """Activity name for listing tools on this MCP server.

        Returns:
            ``"{name}-mcp-list-tools"``
        """
        return f"{self.name}-mcp-list-tools"

    @property
    def call_tool_activity_name(self) -> str:
        """Activity name for calling a tool on this MCP server.

        Returns:
            ``"{name}-mcp-call-tool"``
        """
        return f"{self.name}-mcp-call-tool"

    async def list_tools_in_workflow(self) -> list[dict[str, Any]]:
        """List available tools by dispatching through a Temporal activity.

        Must be called from inside a Temporal workflow context.

        Returns:
            A list of tool descriptors returned by the MCP server.

        References:
            Temporal execute_activity:
            https://python.temporal.io/temporalio.workflow.html#execute_activity
        """
        from temporalio import workflow
        from temporalio.common import RetryPolicy

        logger.info(
            "TemporalMCPToolSet %r: dispatching list-tools via activity %r",
            self.name,
            self.list_tools_activity_name,
        )
        result: list[dict[str, Any]] = await workflow.execute_activity(
            self.list_tools_activity_name,
            start_to_close_timeout=timedelta(seconds=self.start_to_close_timeout),
            retry_policy=RetryPolicy(maximum_attempts=self.maximum_attempts),
        )
        logger.debug(
            "TemporalMCPToolSet %r: list-tools returned %d tool(s)",
            self.name,
            len(result),
        )
        return result

    async def call_tool_in_workflow(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> Any:
        """Call a named tool by dispatching through a Temporal activity.

        Must be called from inside a Temporal workflow context.

        Args:
            tool_name: The name of the MCP tool to invoke.
            arguments: Key-value arguments forwarded to the tool.

        Returns:
            The result returned by the MCP tool.

        References:
            Temporal execute_activity:
            https://python.temporal.io/temporalio.workflow.html#execute_activity
        """
        from temporalio import workflow
        from temporalio.common import RetryPolicy

        logger.info(
            "TemporalMCPToolSet %r: dispatching call-tool %r via activity %r",
            self.name,
            tool_name,
            self.call_tool_activity_name,
        )
        result: Any = await workflow.execute_activity(
            self.call_tool_activity_name,
            args=[tool_name, arguments],
            start_to_close_timeout=timedelta(seconds=self.start_to_close_timeout),
            retry_policy=RetryPolicy(maximum_attempts=self.maximum_attempts),
        )
        logger.debug(
            "TemporalMCPToolSet %r: call-tool %r returned",
            self.name,
            tool_name,
        )
        return result
