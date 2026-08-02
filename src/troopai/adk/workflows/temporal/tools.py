"""Tool wrappers for durable Temporal execution.

Provides :func:`activity_tool` — a factory that promotes an
:func:`~temporalio.activity.defn`-decorated async function into a
:class:`~troopai.adk.tools.function_tool.FunctionTool` whose invocation is
routed through :func:`~temporalio.workflow.execute_activity` when called
from inside a Temporal workflow, and called directly otherwise.

Also provides :class:`TemporalToolWrapper` — a lightweight dataclass that
maps tool names to per-tool :class:`~troopai.adk.workflows.engine.ToolActivityConfig`
overrides and provides a uniform ``should_wrap`` / ``get_config`` interface
used by workflow builders when deciding which tools to promote into
Temporal activities.

References:
    Temporal Python SDK activity docs:
    https://docs.temporal.io/develop/python/core-application#develop-activities
    Temporal execute_activity docs:
    https://python.temporal.io/temporalio.workflow.html#execute_activity
"""

from __future__ import annotations

import inspect
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import timedelta
from typing import TYPE_CHECKING, Any

from troopai.adk.tools.function_tool import FunctionTool, function_tool
from troopai.adk.workflows.engine import ToolActivityConfig

if TYPE_CHECKING:
    from troopai.adk.tools.tool_context import ToolContext

logger = logging.getLogger(__name__)


def activity_tool(
    fn: Callable[..., Any],
    *,
    start_to_close_timeout: timedelta = timedelta(seconds=30),
    maximum_attempts: int = 1,
) -> FunctionTool:
    """Promote an activity-decorated async function into a :class:`~troopai.adk.tools.function_tool.FunctionTool`.

    The returned tool preserves the original function's signature for JSON
    schema generation.  At invocation time the routing decision is made
    lazily:

    - **Inside a Temporal workflow**: the call is dispatched through
      :func:`~temporalio.workflow.execute_activity` with the supplied
      timeout and retry settings, making the tool call durable.
    - **Outside a workflow** (tests, CLI, non-Temporal runners): the
      function is called directly with no overhead.

    Args:
        fn: An ``async`` function decorated with
            :func:`~temporalio.activity.defn`.  The function's name,
            docstring, and parameter annotations are used to build the tool
            schema, so they must be present and accurate.
        start_to_close_timeout: Maximum wall-clock time allowed for a single
            activity attempt.  Defaults to 30 seconds.
        maximum_attempts: Total attempts including the first.  ``1``
            disables retries.  Defaults to ``1`` (no retries); raise it to
            opt into automatic, token-billed re-runs of the activity.

    Returns:
        A :class:`~troopai.adk.tools.function_tool.FunctionTool` wrapping
        *fn* with durable routing when inside a Temporal workflow.

    References:
        Temporal activity options:
        https://python.temporal.io/temporalio.workflow.html#execute_activity
    """
    tool_name: str = fn.__name__

    # Temporal's execute_activity forwards only positional ``args`` to the
    # activity (it calls ``fn(*decoded_args)``); there is no channel for
    # keyword arguments.  ``to_call_args`` routes KEYWORD_ONLY parameters (and
    # **kwargs) into the keyword dict, so a function with such a parameter
    # would have those values silently dropped at dispatch time — committing a
    # wrong result to durable history, or raising for a missing required
    # argument.  Reject them at construction so the failure is loud and early,
    # before any workflow history is written.
    keyword_only_params = [
        name
        for name, param in inspect.signature(fn).parameters.items()
        if param.kind in (inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.VAR_KEYWORD)
    ]
    if len(keyword_only_params) > 0:
        raise ValueError(
            f"activity_tool {tool_name!r}: keyword-only parameters "
            f"{keyword_only_params} cannot be routed through a Temporal "
            "activity, which accepts positional arguments only. Restructure "
            "the function to use positional-or-keyword parameters."
        )

    # Build a thin async wrapper that checks whether we are inside a Temporal
    # workflow and routes accordingly.  Passing the wrapper to function_tool
    # keeps the full schema + invoke chain (argument parsing, validation,
    # error handling) intact while substituting the execution dispatch.
    async def _dispatch_wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            from temporalio import workflow

            in_workflow = workflow.in_workflow()
        except ImportError:
            in_workflow = False

        if not in_workflow:
            logger.debug(
                "activity_tool %r: calling directly (outside Temporal workflow)",
                tool_name,
            )
            return await fn(*args, **kwargs)

        logger.info(
            "activity_tool %r: routing through Temporal execute_activity",
            tool_name,
        )
        from temporalio import workflow as wf
        from temporalio.common import RetryPolicy

        retry_policy = RetryPolicy(maximum_attempts=maximum_attempts)
        return await wf.execute_activity(
            fn,
            args=list(args),
            start_to_close_timeout=start_to_close_timeout,
            retry_policy=retry_policy,
        )

    # Transfer the original function's metadata so function_tool picks up
    # the correct name, docstring, and annotations for schema generation.
    _dispatch_wrapper.__name__ = fn.__name__
    _dispatch_wrapper.__doc__ = fn.__doc__
    _dispatch_wrapper.__annotations__ = fn.__annotations__
    _dispatch_wrapper.__module__ = fn.__module__
    _dispatch_wrapper.__qualname__ = fn.__qualname__
    # Copy __wrapped__ so introspection tools can follow the chain.
    _dispatch_wrapper.__wrapped__ = fn  # type: ignore[attr-defined]

    wrapped_tool: FunctionTool = function_tool(_dispatch_wrapper)
    logger.debug(
        "activity_tool: registered FunctionTool %r (timeout=%s, max_attempts=%d)",
        wrapped_tool.name,
        start_to_close_timeout,
        maximum_attempts,
    )
    return wrapped_tool


def to_durable_tool(
    tool: FunctionTool,
    *,
    start_to_close_timeout: timedelta = timedelta(seconds=30),
    maximum_attempts: int = 1,
) -> FunctionTool:
    """Promote an existing :class:`FunctionTool` to durable Temporal execution.

    Unlike :func:`activity_tool` — which builds a fresh tool from a raw
    activity function — this wraps an *already-constructed* tool, preserving
    its real ``name`` and JSON ``schema``.  Only the invocation is re-routed:

    - **Inside a Temporal workflow**: the call is dispatched through
      :func:`~temporalio.workflow.execute_activity` **by the tool's name**
      (the same string-name dispatch used by
      :class:`~troopai.adk.workflows.temporal.mcp.TemporalMCPToolSet`), so
      the worker must register an activity under ``tool.name``.
    - **Outside a workflow**: the tool's original ``on_invoke`` runs directly.

    The tool arguments cross the boundary as the LLM-produced JSON string
    (already serializable); the run-scoped ``ToolContext`` is never shipped —
    it has no worker-side equivalent.

    Building the durable tool by cloning (rather than re-deriving the schema
    from the generic ``on_invoke(ctx, input)`` wrapper) is load-bearing:
    re-deriving would collapse every tool to the wrapper's name and a
    two-field ``(ctx, input)`` schema, and hand ``execute_activity`` an
    undecorated closure it rejects.

    Args:
        tool: The :class:`FunctionTool` to promote.  Its ``on_invoke`` must
            be set.
        start_to_close_timeout: Per-attempt wall-clock ceiling.  Defaults to
            30 seconds.
        maximum_attempts: Total attempts including the first.  ``1`` disables
            retries.  Defaults to ``1`` (no retries); raise it to opt into
            automatic re-runs.

    Returns:
        A clone of *tool* — identical ``name``, ``schema``, and metadata —
        whose invocation is durable inside a Temporal workflow.

    Raises:
        ValueError: When *tool* has no ``on_invoke`` callable to route.

    References:
        Temporal execute_activity:
        https://python.temporal.io/temporalio.workflow.html#execute_activity
    """
    if tool.on_invoke is None:
        raise ValueError(
            f"to_durable_tool: tool {tool.name!r} has no on_invoke callable to route through a Temporal activity."
        )

    tool_name: str = tool.name
    original_on_invoke = tool.on_invoke

    async def _durable_on_invoke(ctx: ToolContext[Any], input_json: str) -> Any:
        try:
            from temporalio import workflow

            in_workflow = workflow.in_workflow()
        except ImportError:
            in_workflow = False

        if not in_workflow:
            logger.debug(
                "to_durable_tool %r: calling on_invoke directly (outside workflow)",
                tool_name,
            )
            return await original_on_invoke(ctx, input_json)

        logger.info(
            "to_durable_tool %r: routing through Temporal execute_activity",
            tool_name,
        )
        from temporalio import workflow as wf
        from temporalio.common import RetryPolicy

        return await wf.execute_activity(
            tool_name,
            args=[input_json],
            start_to_close_timeout=start_to_close_timeout,
            retry_policy=RetryPolicy(maximum_attempts=maximum_attempts),
        )

    return tool.clone(on_invoke=_durable_on_invoke)


@dataclass
class TemporalToolWrapper:
    """Per-tool Temporal activity config registry for workflow builders.

    Stores a mapping from tool names to either a
    :class:`~troopai.adk.workflows.engine.ToolActivityConfig` (custom config)
    or ``False`` (keep tool in-workflow, no activity wrapping), plus a default
    config applied to all tools not explicitly listed.

    Typical usage::

        wrapper = TemporalToolWrapper(
            tool_configs={
                "fast_lookup": False,  # run in-workflow
                "expensive_api": ToolActivityConfig(
                    start_to_close_timeout=120,
                    maximum_attempts=3,
                ),
            },
        )

        for tool in agent.tools:
            if wrapper.should_wrap(tool.name):
                config = wrapper.get_config(tool.name)
                agent_tools.append(activity_tool(tool.fn, ...))

    Attributes:
        tool_configs: Per-tool overrides.  A :class:`ToolActivityConfig`
            value installs custom timeout / retry settings.  ``False``
            keeps the tool running inside the workflow boundary without
            promotion to a Temporal activity.
        default_config: Config applied to tools whose name is absent from
            *tool_configs*.  Defaults to :class:`ToolActivityConfig` defaults
            (30 s timeout, 1 total attempt — retries off).
    """

    tool_configs: dict[str, ToolActivityConfig | bool] = field(default_factory=dict)
    """Per-tool activity config overrides.

    ``False`` disables wrapping for that tool; any :class:`ToolActivityConfig`
    instance enables wrapping with the supplied policy.
    """

    default_config: ToolActivityConfig = field(default_factory=ToolActivityConfig)
    """Default :class:`ToolActivityConfig` applied to tools not listed in
    *tool_configs*."""

    def should_wrap(self, tool_name: str) -> bool:
        """Return ``True`` if *tool_name* should be promoted to a Temporal activity.

        Args:
            tool_name: The :attr:`~troopai.adk.tools.function_tool.FunctionTool.name`
                of the tool being evaluated.

        Returns:
            ``False`` when *tool_name* is explicitly configured to ``False``
            in :attr:`tool_configs`; ``True`` in all other cases (specific
            :class:`ToolActivityConfig` or not listed at all).
        """
        config = self.tool_configs.get(tool_name)
        return config is not False

    def get_config(self, tool_name: str) -> ToolActivityConfig:
        """Return the :class:`ToolActivityConfig` to use for *tool_name*.

        Args:
            tool_name: The :attr:`~troopai.adk.tools.function_tool.FunctionTool.name`
                of the tool being evaluated.

        Returns:
            The :class:`ToolActivityConfig` stored under *tool_name* in
            :attr:`tool_configs` when present, otherwise :attr:`default_config`.
        """
        config = self.tool_configs.get(tool_name)
        if isinstance(config, ToolActivityConfig):
            return config
        return self.default_config
