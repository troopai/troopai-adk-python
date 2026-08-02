"""Base class for user-authored toolset wrappers.

``WrapperToolset`` is the user-extension surface for toolset
behaviour that the shipped ``Prefixed`` / ``Renamed`` / ``Filtered``
/ ``Combined`` variants do not cover. Subclass and override
``get_tools()`` to return whatever ``dict[str, FunctionTool]`` the
custom logic produces — typically a transformation of the wrapped
toolset's output.

The ``middleware: list[ToolMiddleware]`` field wraps each materialised
tool's ``on_invoke`` with a toolset-scoped middleware chain.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, override

from troopai.adk.tools.toolsets.abstract import Toolset

if TYPE_CHECKING:
    from troopai.adk.run.context import RunContext
    from troopai.adk.tools.function_tool import FunctionTool
    from troopai.adk.tools.tool_middleware import ToolMiddleware


@dataclass
class WrapperToolset(Toolset):
    """Pass-through toolset wrapper that user subclasses can override.

    Attributes:
        wrapped: The toolset whose output is the basis for this
            wrapper's behaviour. Subclasses typically materialise
            this in ``get_tools`` then transform the result.
        middleware: Toolset-scoped tool middleware applied to every
            materialised tool's ``on_invoke``. Composes with
            ``Agent.middleware.tools`` (the agent-global chain) — the
            executor wraps the toolset-scoped chain inside the
            agent-global chain. Empty by default, in which case
            ``WrapperToolset`` is a pure pass-through.
    """

    wrapped: Toolset
    """The toolset whose output is the basis for this wrapper."""

    middleware: list[ToolMiddleware] = field(default_factory=list)
    """Toolset-scoped tool middleware applied to materialised tools."""

    @override
    async def get_tools(
        self,
        ctx: RunContext[Any] | None = None,
    ) -> dict[str, FunctionTool]:
        """Default: materialise the wrapped toolset, apply toolset middleware.

        Override this method to add custom transformations. When
        overriding, decide whether to call ``super().get_tools(ctx)``
        first (preserves middleware wrapping) or build the dict
        manually and skip middleware for this wrapper.
        """
        inner = await self.wrapped.get_tools(ctx)
        if len(self.middleware) == 0:
            return dict(inner)
        # Defer middleware application to the dedicated helper so the
        # wrapping logic stays in one place (also reused by Agent-global
        # middleware in the executor).
        from troopai.adk.tools.tool_middleware import wrap_tool_with_middleware

        out: dict[str, FunctionTool] = {}
        for name, tool in inner.items():
            out[name] = wrap_tool_with_middleware(tool, self.middleware)
        return out

    @override
    async def adispose(self) -> None:
        """Forward disposal to the wrapped toolset.

        Subclasses overriding ``adispose`` SHOULD call ``super().adispose()``
        so the wrapped toolset's resources are released.
        """
        await self.wrapped.adispose()
