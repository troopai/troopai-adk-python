"""Internal helpers for Runner handoff processing.

Provides module-level functions that the Runner calls to normalize,
build tools for, and look up LLM-orchestrated handoffs. Mirrors
OpenAI's pattern of building a handoff_map for O(1) lookup.

These are internal — not exported from the ``handoffs`` package.
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from troopai.adk.exceptions import HandoffDefinitionError
from troopai.adk.handoffs.handoff import Handoff
from troopai.adk.run.context import RunContext, TContext
from troopai.adk.utils import MaybeAwaitable

if TYPE_CHECKING:
    from troopai.adk.agents import Agent
    from troopai.adk.tools.function_tool import FunctionTool


async def normalize_handoffs(
    handoffs: list[Agent[Any] | Handoff[Any, Any, Any]],
) -> list[Handoff[Any, Any, Any]]:
    """Wrap bare Agents in ``Handoff(target=...)`` for uniform handling.

    Also rejects duplicate tool names at setup time. Two handoffs that
    resolve to the same ``get_name()`` would otherwise emit two function
    tools with identical names (which most providers reject) and make
    :func:`find_handoff_target` route to whichever happened to be last.
    Collisions are easy to hit accidentally because ``get_name()``
    lowercases and snake-cases the target name, so targets differing only
    in case or spacing — or the same agent listed twice — collapse to one
    name.

    Args:
        handoffs: Mixed list of Agent instances and Handoff objects.

    Returns:
        List of Handoff objects (bare Agents wrapped automatically).

    Raises:
        HandoffDefinitionError: two handoffs resolve to the same tool name.
    """
    normalized: list[Handoff[Any, Any, Any]] = []
    for item in handoffs:
        if isinstance(item, Handoff):
            normalized.append(item)
        else:
            # Bare Agent — wrap in a default Handoff
            normalized.append(Handoff(target=item))

    seen: dict[str, str] = {}
    for h in normalized:
        name = h.get_name()
        if name in seen:
            raise HandoffDefinitionError(
                name,
                f"Duplicate handoff tool name '{name}': targets "
                f"'{seen[name]}' and '{h.target.name}' resolve to the same "
                "tool name. Use a unique Handoff(name=...) to disambiguate.",
            )
        seen[name] = h.target.name

    return normalized


async def build_handoff_tools(
    handoffs: list[Handoff[Any, Any, Any]],
    context: RunContext[TContext] | None = None,
) -> list[FunctionTool]:
    """Build transfer tool definitions for all enabled Handoff targets.

    Args:
        handoffs: Normalized list of Handoff objects.
        context: Optional run context for evaluating ``enabled`` callbacks.

    Returns:
        List of ``FunctionTool`` instances for handoff transfer tools.
    """
    tools: list[FunctionTool] = []
    for handoff in handoffs:
        if await is_handoff_enabled(handoff, context):
            tools.append(handoff.to_tool())
    return tools


async def find_handoff_target(
    handoffs: list[Handoff[Any, Any, Any]],
    tool_name: str,
    context: RunContext[TContext] | None = None,
) -> Handoff[Any, Any, Any] | None:
    """Find a Handoff by matching tool_name.

    Builds a ``{tool_name: handoff}`` dict for O(1) lookup.

    Args:
        handoffs: Normalized list of Handoff objects.
        tool_name: The function name from the LLM tool call.
        context: Optional run context for evaluating ``enabled`` callbacks.

    Returns:
        The matching Handoff if found and enabled, None otherwise.
    """
    handoff_map = {h.get_name(): h for h in handoffs}
    target = handoff_map.get(tool_name)
    if target is not None and await is_handoff_enabled(target, context):
        return target
    return None


async def is_handoff_enabled(
    handoff: Handoff[Any, Any, Any],
    context: RunContext[TContext] | None = None,
) -> bool:
    """Check if an LLM-orchestrated Handoff is enabled.

    Thin wrapper around :func:`evaluate_enabled` that supplies the
    handoff's target ``Agent`` as the second positional argument to
    2-arg callables, so the gate can depend on the destination.

    Args:
        handoff: The Handoff to check.
        context: Run context (required when ``enabled`` is callable).

    Returns:
        True if the handoff is enabled, False otherwise.

    Raises:
        HandoffDefinitionError: see :func:`evaluate_enabled`.
    """
    return await evaluate_enabled(
        handoff.enabled,
        context,
        handoff.target,
        handoff_name=handoff.get_name(),
    )


async def evaluate_enabled(
    enabled: bool | Callable[..., MaybeAwaitable[bool]],
    context: RunContext[Any] | None,
    second_arg: Any,
    *,
    handoff_name: str,
) -> bool:
    """Evaluate a handoff ``enabled`` flag (shared by LLM-orch + code-orch).

    Bool values pass through unchanged. Callable values are dispatched
    by their introspected positional arity:

    - 0 positional params: ``enabled()`` — useful for global feature flags.
    - 1 positional param:  ``enabled(context)``.
    - 2+ positional params (or ``*args``): ``enabled(context, second_arg)``.
       For LLM-orchestrated handoffs ``second_arg`` is the target
       ``Agent``; for code-orchestrated routing it is the matched
       ``Intent``.

    Sync returns are validated directly; coroutine returns are awaited
    first. The return value MUST be a bool — non-bool returns surface
    as :class:`HandoffDefinitionError` so silent True/False coercions
    don't hide misconfigured gates.

    Args:
        enabled: The bool or callable to evaluate.
        context: The run context. REQUIRED when ``enabled`` is callable.
        second_arg: The mode-specific second positional argument.
        handoff_name: Human-readable handoff name for error messages.

    Returns:
        True if enabled, False otherwise.

    Raises:
        HandoffDefinitionError: callable + ``context is None``;
            signature not introspectable (e.g. some C-implemented
            builtins); ``**kwargs``-only signature with no positional
            slot; the callable returned an async generator; the
            (awaited) return value is not a bool.
    """
    if not callable(enabled):
        if not isinstance(enabled, bool):
            raise HandoffDefinitionError(
                handoff_name,
                f"Handoff '{handoff_name}' 'enabled' must be bool or callable; got {type(enabled).__name__}.",
            )
        return enabled

    if context is None:
        raise HandoffDefinitionError(
            handoff_name,
            f"Handoff '{handoff_name}' has a callable 'enabled' but "
            "no RunContext was supplied. Pass a context to "
            "build_handoff_tools() / find_handoff_target() / "
            "HandoffRoute.resolve() — callers always supply one at "
            "evaluation time.",
        )

    try:
        sig = inspect.signature(enabled)
    except (ValueError, TypeError) as exc:
        raise HandoffDefinitionError(
            handoff_name,
            f"Handoff '{handoff_name}' 'enabled' callable has a "
            f"signature that cannot be introspected ({exc}). Use a "
            "Python function, lambda, or bound method instead of a "
            "C-implemented callable.",
        ) from exc

    positional_kinds = (
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    )
    positional_count = 0
    has_var_positional = False
    for p in sig.parameters.values():
        if p.name == "self":
            continue
        if p.kind in positional_kinds:
            positional_count += 1
        elif p.kind is inspect.Parameter.VAR_POSITIONAL:
            has_var_positional = True

    # `enabled` is variadic for the type checker; the dispatch below
    # picks the actual arity. Widen through Any so mypy/pyright don't
    # try to reconcile each branch against the union of arities.
    callback: Any = enabled

    if has_var_positional or positional_count >= 2:
        result = callback(context, second_arg)
    elif positional_count == 1:
        result = callback(context)
    elif positional_count == 0 and not any(
        p.kind is inspect.Parameter.KEYWORD_ONLY and p.default is inspect.Parameter.empty
        for p in sig.parameters.values()
    ):
        result = callback()
    else:
        raise HandoffDefinitionError(
            handoff_name,
            f"Handoff '{handoff_name}' 'enabled' callable signature is "
            "unsupported. Accept 0, 1, or 2 positional args (or *args). "
            "Required keyword-only parameters are not supported.",
        )

    if inspect.isasyncgen(result):
        raise HandoffDefinitionError(
            handoff_name,
            f"Handoff '{handoff_name}' 'enabled' callable returned an "
            "async generator. Use 'async def f(...) -> bool: return ...' "
            "(not 'yield').",
        )

    if asyncio.iscoroutine(result):
        result = await result

    if not isinstance(result, bool):
        raise HandoffDefinitionError(
            handoff_name,
            f"Handoff '{handoff_name}' 'enabled' callable must return bool; got {type(result).__name__} ({result!r}).",
        )

    return result
