"""Predicate-based filtering toolset wrapper.

The filter is evaluated **per turn** against the live ``RunContext``,
so the visible tool set can change between turns without rebuilding
the agent::

    def admin_only(ctx, tool):
        return ctx.context.get("role") == "admin"


    sensitive = FunctionToolset(tools=[delete, restart]).filtered(admin_only)

This is the headline reason for keeping toolsets as a *live*
abstraction (``await get_tools(ctx)`` per turn) rather than
materialising at construction time.

Predicates are treated as untrusted user code: an exception inside
``filter_func`` is caught, logged at WARNING level, and the tool is
**fail-closed** (excluded from this turn's list). Failing closed is
the safe default for ``FilteredToolset``'s primary use case
(role-based access control) — silently exposing a tool because the
ACL check threw would be the worse failure mode.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from inspect import isawaitable
from typing import TYPE_CHECKING, Any, override

from troopai.adk.tools.toolsets.abstract import Toolset, ToolsetFilter

if TYPE_CHECKING:
    from troopai.adk.run.context import RunContext
    from troopai.adk.tools.function_tool import FunctionTool


logger = logging.getLogger(__name__)


@dataclass
class FilteredToolset(Toolset):
    """A toolset that drops tools whose predicate returns False.

    Attributes:
        wrapped: The toolset whose tools are subject to filtering.
        filter_func: Predicate ``(ctx, tool) -> bool`` (sync or async).
            Tools where the predicate returns ``False`` — or where the
            predicate raises — are excluded from this turn's tool
            list. Predicate exceptions are logged at WARNING level
            and treated as ``False`` (fail-closed).
    """

    wrapped: Toolset
    """The toolset whose tools are subject to filtering."""

    filter_func: ToolsetFilter
    """Predicate evaluated for each tool each turn."""

    @override
    async def get_tools(
        self,
        ctx: RunContext[Any] | None = None,
    ) -> dict[str, FunctionTool]:
        """Materialise the wrapped toolset and drop tools failing the predicate.

        When ``ctx`` is ``None`` (e.g. construction-time validation),
        the predicate is skipped and the full inner toolset is
        returned. Predicates that depend on ``ctx`` will get a real
        context once execution begins.
        """
        inner = await self.wrapped.get_tools(ctx)
        if ctx is None:
            return dict(inner)
        out: dict[str, FunctionTool] = {}
        for name, tool in inner.items():
            try:
                verdict = self.filter_func(ctx, tool)
                if isawaitable(verdict):
                    verdict = await verdict
            except Exception as exc:
                # Fail-closed: exclude the tool. Role-based ACLs are
                # the primary use case; silently granting access on a
                # transient predicate failure would be the worse mode.
                logger.warning(
                    "FilteredToolset predicate raised for tool '%s'; excluding from this turn: %s",
                    name,
                    exc,
                )
                continue
            if bool(verdict):
                out[name] = tool
        return out

    @override
    async def adispose(self) -> None:
        """Forward disposal to the wrapped toolset."""
        await self.wrapped.adispose()
