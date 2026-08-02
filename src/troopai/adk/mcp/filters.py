"""Tool-filter types for MCP toolsets.

A ``ToolFilter`` is a per-tool predicate evaluated each turn against
the *converted* ``FunctionTool`` (not the raw MCP ``Tool``), giving
filters access to the framework's tool name, description, and schema
in their final form. The companion ``ToolFilterContext`` carries
server-aware metadata that the generic ``Toolset.filtered(...)``
predicate cannot see — specifically the originating MCP server's
name and the live ``RunContext``.

This is deliberately separate from ``ToolsetFilter`` in
``tools/toolsets/abstract.py``: ``ToolsetFilter`` operates on a
``RunContext`` + ``FunctionTool`` pair, while ``ToolFilter`` adds
``server_name`` so multi-server agents can filter tools by their
origin (e.g. "only allow read tools from `prod-api`").
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from troopai.adk.utils import MaybeAwaitable

if TYPE_CHECKING:
    from troopai.adk.run.context import RunContext
    from troopai.adk.tools.function_tool import FunctionTool


@dataclass(frozen=True, kw_only=True)
class ToolFilterContext:
    """Context passed to a ``ToolFilter`` predicate.

    Attributes:
        server_name: The ``MCPServer.name`` of the originating server.
            Lets multi-server agents filter by tool origin.
        run_context: The live ``RunContext`` for this turn, or
            ``None`` during construction-time validation. Predicates
            that read user/session state SHOULD handle ``None`` by
            returning ``True`` (admit-all) so the tool list is not
            empty during introspection.
    """

    server_name: str
    """The originating MCP server's name."""

    run_context: RunContext[Any] | None
    """Live run context for this turn; may be ``None`` during validation."""


ToolFilter = Callable[[ToolFilterContext, "FunctionTool"], MaybeAwaitable[bool]]
"""Per-tool predicate. Sync or async; returns ``True`` to keep the tool.

Receives a ``ToolFilterContext`` (server name + run context) and the
already-converted ``FunctionTool``. Predicates raising an exception
are treated as fail-closed (tool excluded, warning logged) so a
buggy filter cannot leak unauthorised tools to the LLM.
"""
