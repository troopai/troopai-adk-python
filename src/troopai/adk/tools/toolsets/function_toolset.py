"""Leaf toolset wrapping a flat sequence of ``FunctionTool`` instances.

This is the primitive every other toolset variant ultimately wraps.
Constructing a ``FunctionToolset`` is the entry point for users who
want toolset composition over a list of plain function tools::

    from troopai.adk.tools.toolsets import FunctionToolset

    weather = FunctionToolset(tools=[get_temp, get_conditions]).prefixed("weather")
    db = FunctionToolset(tools=[query, insert]).prefixed("db")
    agent = Agent(name="ops", system_prompt="...", tools=[weather, db])

The toolset is **eagerly populated** — every tool is held by the
``FunctionToolset`` instance and returned each turn. Filtering and
renaming happen in wrapper layers, not here.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, override

from troopai.adk.tools.toolsets.abstract import Toolset

if TYPE_CHECKING:
    from troopai.adk.run.context import RunContext
    from troopai.adk.tools.function_tool import FunctionTool


logger = logging.getLogger(__name__)


@dataclass
class FunctionToolset(Toolset):
    """A toolset over a flat list of ``FunctionTool`` instances.

    Attributes:
        tools: The function tools this toolset exposes. Order is
            preserved in the materialised dict's iteration order
            (insertion-ordered), but downstream consumers should not
            rely on ordering for correctness.
    """

    tools: Sequence[FunctionTool] = field(default_factory=list)
    """The function tools this toolset exposes."""

    @override
    async def get_tools(
        self,
        ctx: RunContext[Any] | None = None,
    ) -> dict[str, FunctionTool]:
        """Return the tool dict, keyed by ``tool.name``.

        On duplicate names within the same toolset, the **last**
        occurrence wins and a warning is logged. Cross-toolset
        conflicts are surfaced by ``build_tools()`` with the full
        list of contributing sources.

        Args:
            ctx: Run context for the current turn. Not used by this
                implementation but accepted to satisfy the
                :class:`Toolset` interface.

        Returns:
            A mapping of ``tool_name -> FunctionTool`` for all tools in
            this toolset.
        """
        out: dict[str, FunctionTool] = {}
        for tool in self.tools:
            if tool.name in out:
                logger.warning(
                    "FunctionToolset has duplicate tool name '%s'; later definition wins",
                    tool.name,
                )
            out[tool.name] = tool
        return out
