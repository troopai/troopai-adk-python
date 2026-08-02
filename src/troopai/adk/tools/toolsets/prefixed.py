"""Namespace-prefixing toolset wrapper.

Adds a fixed prefix to every tool name in the wrapped toolset so
multiple toolsets can coexist on the same agent without name
collisions::

    weather = FunctionToolset(tools=[get_temp]).prefixed("weather")
    db = FunctionToolset(tools=[query]).prefixed("db")
    # The agent sees: weather_get_temp, db_query

The default ``_`` separator matches Pydantic-AI's convention and is
compatible with every provider's tool-name regex (which typically
allows ``[a-zA-Z0-9_-]``). Using ``.`` would be clearer as a
namespace separator but breaks on Anthropic and OpenAI Chat
Completions, which reject dots in tool names.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, override

from troopai.adk.tools.toolsets.abstract import Toolset
from troopai.adk.tools.toolsets.clone import clone_with_name

if TYPE_CHECKING:
    from troopai.adk.run.context import RunContext
    from troopai.adk.tools.function_tool import FunctionTool


@dataclass
class PrefixedToolset(Toolset):
    """A toolset that namespaces every wrapped tool with a fixed prefix.

    Attributes:
        wrapped: The toolset whose tools will be prefixed.
        prefix: The namespace prefix (e.g. ``"weather"``).
        separator: The character(s) joining ``prefix`` and the original
            name. Defaults to ``"_"`` for cross-provider compatibility.
    """

    wrapped: Toolset
    """The toolset whose tools will be prefixed."""

    prefix: str
    """The namespace prefix to prepend to every tool name."""

    separator: str = "_"
    """The character(s) joining ``prefix`` and the original tool name."""

    def __post_init__(self) -> None:
        if len(self.prefix) == 0:
            raise ValueError("PrefixedToolset.prefix cannot be empty")

    @override
    async def get_tools(
        self,
        ctx: RunContext[Any] | None = None,
    ) -> dict[str, FunctionTool]:
        """Materialise the wrapped toolset and return prefixed clones."""
        inner = await self.wrapped.get_tools(ctx)
        out: dict[str, FunctionTool] = {}
        for name, tool in inner.items():
            new_name = f"{self.prefix}{self.separator}{name}"
            out[new_name] = clone_with_name(tool, new_name)
        return out

    @override
    async def adispose(self) -> None:
        """Forward disposal to the wrapped toolset.

        Without this, an ``MCPToolset`` reached only through a prefix
        wrapper would leak its connection — the runner's disposal
        loop sees only the wrapper, whose default ``adispose`` is a
        no-op.
        """
        await self.wrapped.adispose()
