"""Internal helper for cloning ``FunctionTool`` instances under a new name.

Toolsets that rename tools (``PrefixedToolset``, ``RenamedToolset``)
need to materialise the wrapped tool with a new ``name`` field while
preserving the internal slots a naive ``dataclasses.replace`` would
reset to factory defaults — the delegate-agent reference set by
``Agent.as_tool()``, the result cache, the rate-limit window state,
and the deferred-tool reveal closure.

This module is a thin adapter over ``FunctionTool.clone(name=...)``
that adds a same-name short-circuit so toolsets do not have to
remember to check before calling. The underlying state-copy
contract (cache and rate-state shared by reference; see why in
``FunctionTool.clone``'s docstring) lives on ``FunctionTool``
itself, not here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from troopai.adk.tools.function_tool import FunctionTool


def clone_with_name(tool: FunctionTool, new_name: str) -> FunctionTool:
    """Return a clone of ``tool`` whose ``name`` field is ``new_name``.

    Same-name calls return ``tool`` unchanged so the toolset call
    sites do not have to special-case "no rename" themselves.

    Args:
        tool: The source ``FunctionTool``.
        new_name: The new name for the cloned tool.

    Returns:
        A new ``FunctionTool`` instance whose ``name`` is ``new_name``
        and whose internal state mirrors the original. Returns the
        original ``tool`` unchanged when ``new_name == tool.name``.
    """
    if tool.name == new_name:
        return tool
    return tool.clone(name=new_name)
