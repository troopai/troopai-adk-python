"""Composition toolset that materialises multiple toolsets as one.

The combined toolset materialises each child in declaration order
and merges the resulting dicts. Conflict detection lives in
``build_tools()`` rather than here — surfacing the conflict at
agent-build time gives a single error per turn with every source
listed, instead of a per-toolset partial failure.

No ``+`` operator is shipped on ``Toolset`` deliberately: the
ordered-sequence form ``CombinedToolset(toolsets=[a, b, c])`` makes
conflict-error messages clearer ("contributed by toolsets[0],
toolsets[2]") and avoids surprising right-associativity quirks if
``__add__`` and ``__radd__`` were ever to disagree on subclass
priority.
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
class CombinedToolset(Toolset):
    """A toolset that flattens multiple child toolsets into one.

    Attributes:
        toolsets: The child toolsets, materialised in declaration
            order. The last writer wins on duplicate names within
            this combined toolset; cross-toolset conflicts at the
            agent level are surfaced by ``build_tools()`` with the
            full list of contributing sources.
    """

    toolsets: Sequence[Toolset] = field(default_factory=list)
    """The child toolsets to flatten."""

    @override
    async def get_tools(
        self,
        ctx: RunContext[Any] | None = None,
    ) -> dict[str, FunctionTool]:
        """Materialise every child and return the merged dict."""
        out: dict[str, FunctionTool] = {}
        for child in self.toolsets:
            child_tools = await child.get_tools(ctx)
            out.update(child_tools)
        return out

    @override
    async def adispose(self) -> None:
        """Forward disposal to every child; warn-and-continue on errors.

        Iteration is REVERSED so anyio cancel scopes opened by later
        children (e.g. a second ``MCPToolset`` whose ``get_tools``
        ran after the first) are closed before earlier ones. FIFO
        disposal here would trigger the same
        ``RuntimeError: Attempted to exit a cancel scope ...``
        observed when multiple ``MCPToolset`` instances are
        composed at the top level of ``Agent.tools``. See
        ``runner._dispose_agent_toolsets`` for the same rationale.
        """
        for child in reversed(self.toolsets):
            try:
                await child.adispose()
            except Exception:
                logger.warning(
                    "CombinedToolset child adispose raised; child=%r",
                    child,
                    exc_info=True,
                )
