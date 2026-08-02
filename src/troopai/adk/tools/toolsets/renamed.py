"""Per-name rename toolset wrapper.

Applies a ``{old_name: new_name}`` map to the wrapped toolset.
Names not present in the map pass through unchanged. Useful for
renaming a single tool out of a larger toolset (e.g. an MCP server
exposing ``execute_query`` that the agent should see as ``sql``)::

    db_tools.renamed({"execute_query": "sql"})
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, override

from troopai.adk.tools.toolsets.abstract import Toolset
from troopai.adk.tools.toolsets.clone import clone_with_name

if TYPE_CHECKING:
    from troopai.adk.run.context import RunContext
    from troopai.adk.tools.function_tool import FunctionTool


@dataclass
class RenamedToolset(Toolset):
    """A toolset that renames a subset of the wrapped toolset's tools.

    Attributes:
        wrapped: The toolset whose tools are subject to renaming.
        name_map: Mapping of original name to new name. Names not in
            the map are passed through unchanged.
    """

    wrapped: Toolset
    """The toolset whose tools are subject to renaming."""

    name_map: Mapping[str, str]
    """Mapping of original tool name to new tool name."""

    def __post_init__(self) -> None:
        if len(self.name_map) == 0:
            raise ValueError("RenamedToolset.name_map cannot be empty")
        for old, new in self.name_map.items():
            if len(new) == 0:
                raise ValueError(f"RenamedToolset rename target for '{old}' cannot be empty")

    @override
    async def get_tools(
        self,
        ctx: RunContext[Any] | None = None,
    ) -> dict[str, FunctionTool]:
        """Materialise the wrapped toolset and apply the rename map."""
        inner = await self.wrapped.get_tools(ctx)
        out: dict[str, FunctionTool] = {}
        for name, tool in inner.items():
            new_name = self.name_map.get(name, name)
            if new_name in out:
                raise ValueError(f"RenamedToolset: rename collision — two tools map to '{new_name}'")
            out[new_name] = clone_with_name(tool, new_name)
        return out

    @override
    async def adispose(self) -> None:
        """Forward disposal to the wrapped toolset."""
        await self.wrapped.adispose()
