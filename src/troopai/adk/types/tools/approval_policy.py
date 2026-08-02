"""Approval policy for human-in-the-loop tool gating.

``ApprovalPolicy`` is the type of ``FunctionTool.requires_approval``
and every surface that forwards into it (MCP tool conversion, MCP
toolsets): a static ``bool``, or a per-call callable receiving the
``ToolContext`` and returning a ``bool`` (sync or async) so approval
can be decided per invocation.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from troopai.adk.utils import MaybeAwaitable

if TYPE_CHECKING:
    from troopai.adk.tools.tool_context import ToolContext

type ApprovalPolicy = bool | Callable[[ToolContext[Any]], MaybeAwaitable[bool]]
"""Static flag or per-call decision callable for tool approval."""

__all__ = ["ApprovalPolicy"]
