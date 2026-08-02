"""Toolset abstract base class.

A ``Toolset`` is a live abstraction over a group of tools: it returns
its concrete ``FunctionTool`` instances on demand each turn via
``get_tools(ctx)``. This is symmetric with the existing per-turn
dynamism of ``FunctionTool.enabled`` and ``FunctionTool.prepare`` —
toolsets compose with the run context the same way individual tools
do.

Design choice: the surface is **live** (``await get_tools(ctx)``
each turn) rather than **materialised at construction**, so
``FilteredToolset(filter_func=lambda ctx, t: ...)`` can react to
``RunContext`` changes between turns. Materialise-at-construction
would have been simpler but lost the headline use case.

The five concrete subclasses are split across sibling modules:

- ``FunctionToolset`` — leaf primitive wrapping a flat list of tools.
- ``PrefixedToolset`` — namespace-prefixes every tool's name.
- ``RenamedToolset`` — applies a per-name rename map.
- ``FilteredToolset`` — drops tools whose predicate is False.
- ``CombinedToolset`` — composes multiple toolsets into one.
- ``WrapperToolset`` — base for user subclasses that want to override
  ``get_tools`` while delegating to a wrapped toolset.

Tools whose name is changed by ``PrefixedToolset`` /
``RenamedToolset`` are returned as **renamed clones** (the ``name``
field is replaced via ``dataclasses.replace`` and internal state is
copied). The executor's ``resolve_function_tool`` finds them by
their renamed name end-to-end — no un-rename step is needed at call
time.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

from troopai.adk.utils import MaybeAwaitable

if TYPE_CHECKING:
    from troopai.adk.run.context import RunContext
    from troopai.adk.tools.function_tool import FunctionTool


ToolsetFilter = Callable[["RunContext[Any]", "FunctionTool"], MaybeAwaitable[bool]]
"""Predicate signature for ``FilteredToolset``.

Receives the live ``RunContext`` and a ``FunctionTool`` from the
wrapped toolset; returns ``True`` to keep the tool for this turn,
``False`` to drop it. Sync or async — the executor awaits when
needed.
"""


class Toolset(ABC):
    """Abstract base class for tool collections.

    Concrete subclasses implement ``get_tools(ctx)`` to return the
    materialised tool dict for the current turn. Subclasses inherit
    builder methods (``prefixed``, ``renamed``, ``filtered``,
    ``combined_with``) that produce wrapped variants without needing
    to know each wrapper's import path.
    """

    @abstractmethod
    async def get_tools(
        self,
        ctx: RunContext[Any] | None = None,
    ) -> dict[str, FunctionTool]:
        """Return the tool dict for this turn.

        Implementations decide what runs each turn — a static toolset
        returns the same dict every call; a filtered toolset evaluates
        its predicate against the current ``ctx``; a combined toolset
        recursively materialises its children.

        Args:
            ctx: Run context for the current turn. May be ``None``
                during construction-time validation.

        Returns:
            A mapping ``tool_name -> FunctionTool`` of every tool the
            toolset contributes for this turn. Keys MUST match the
            ``name`` field of the corresponding tool — ``build_tools``
            relies on this for conflict detection.
        """

    async def adispose(self) -> None:
        """Release any resources held by this toolset.

        Called by the runner's ``finally`` block after a run completes
        — normally or with exception. Default is a no-op; override in
        toolsets that hold network connections, subprocesses, or other
        resources requiring deterministic cleanup.

        Implementations MUST be idempotent (safe to call multiple times)
        and MUST NOT raise — the runner swallows exceptions from this
        method so cleanup of one toolset cannot block cleanup of others.
        Wrap concrete cleanup in ``try/except`` and ``logger.warning``.
        """

    # ------------------------------------------------------------------
    # Builder methods — return wrapped variants without import-path leak
    # ------------------------------------------------------------------

    def prefixed(self, prefix: str, *, separator: str = "_") -> Toolset:
        """Return a new toolset where every tool's name is prefixed.

        Example: ``ts.prefixed("weather")`` turns ``get_temp`` into
        ``weather_get_temp``. The default ``_`` separator is the
        Pydantic-AI convention and is compatible with every
        provider's tool-name regex.

        Args:
            prefix: The namespace prefix to prepend to every tool name.
            separator: Character(s) joining ``prefix`` and the original
                name. Defaults to ``"_"``.

        Returns:
            A :class:`PrefixedToolset` wrapping ``self``.
        """
        from troopai.adk.tools.toolsets.prefixed import PrefixedToolset

        return PrefixedToolset(wrapped=self, prefix=prefix, separator=separator)

    def renamed(self, name_map: Mapping[str, str]) -> Toolset:
        """Return a new toolset that renames tools per the given map.

        Names not in ``name_map`` are passed through unchanged. The
        rename is applied at materialisation — the executor sees only
        the renamed tool, so call-time lookup matches the renamed name.

        Args:
            name_map: Mapping of original tool name to new tool name.

        Returns:
            A :class:`RenamedToolset` wrapping ``self``.
        """
        from troopai.adk.tools.toolsets.renamed import RenamedToolset

        return RenamedToolset(wrapped=self, name_map=name_map)

    def filtered(self, filter_func: ToolsetFilter) -> Toolset:
        """Return a new toolset that drops tools where ``filter_func`` returns False.

        The predicate is evaluated each turn against the current
        ``RunContext``, so the visible tool set can change as the run
        progresses (e.g. expose admin tools only when the context's
        ``role == "admin"``).

        Args:
            filter_func: Predicate ``(ctx, tool) -> bool`` (sync or
                async). Returns ``True`` to include the tool this turn.

        Returns:
            A :class:`FilteredToolset` wrapping ``self``.
        """
        from troopai.adk.tools.toolsets.filtered import FilteredToolset

        return FilteredToolset(wrapped=self, filter_func=filter_func)

    def combined_with(self, other: Toolset) -> Toolset:
        """Return a new toolset that materialises ``self`` then ``other``.

        Equivalent to ``CombinedToolset([self, other])``. Use the
        constructor directly for three or more toolsets — the builder
        method is the two-toolset shorthand.

        Args:
            other: The second toolset to compose.

        Returns:
            A :class:`CombinedToolset` containing ``[self, other]``.
        """
        from troopai.adk.tools.toolsets.combined import CombinedToolset

        return CombinedToolset(toolsets=[self, other])
