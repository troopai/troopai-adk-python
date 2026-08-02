from __future__ import annotations

import asyncio
import inspect
import logging
import typing
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic

from typing_extensions import TypeVar

from troopai.adk.handoffs.handoff_config import HandoffConfig, apply_callback_error_policy
from troopai.adk.handoffs.handoff_input_data import HandoffInputData
from troopai.adk.run.context import RunContext, TContext
from troopai.adk.types.intents import Intent
from troopai.adk.types.items import RunItem
from troopai.adk.utils import MaybeAwaitable

if TYPE_CHECKING:
    from troopai.adk.agents import Agent, BaseAgent

logger = logging.getLogger(__name__)

TAgent = TypeVar("TAgent", bound="BaseAgent[Any]", default="Agent[Any]")

THandoffInput = TypeVar("THandoffInput", default=Any)
"""The type of structured data passed when a handoff tool is called."""

# Type alias for handoff input filter function
type HandoffInputFilter = Callable[[HandoffInputData], HandoffInputData]
"""Function that transforms HandoffInputData before passing to target agent."""

type OnHandoffWithInput = Callable[[RunContext[Any], Any], MaybeAwaitable[None]]
"""Callback invoked with typed input: ``(ctx, validated_input) -> None``.

Used when the handoff has an ``input_type`` set. The second argument is
the validated Pydantic model (for LLM-orch) or the Intent (for code-orch)."""

type OnHandoffWithoutInput = Callable[[RunContext[Any]], MaybeAwaitable[None]]
"""Callback invoked without input: ``(ctx) -> None``.

Used for parameterless handoffs where the callback only needs the context."""

type OnHandoffWithData = Callable[[RunContext[Any], HandoffInputData], MaybeAwaitable[None]]
"""Callback invoked with full handoff data: ``(ctx, data) -> None``.

Used for advanced use cases where the callback needs access to temporal
slices (context, output) and the full audit trail."""

type OnHandoffCallback = OnHandoffWithInput | OnHandoffWithoutInput | OnHandoffWithData
"""Callback invoked when a handoff occurs. Side-effect only (logging, metrics, etc.).

Accepts either:
- ``(ctx, input)`` — receives the validated typed input (or raw intent)
- ``(ctx)`` — parameterless, only receives the run context
- ``(ctx, data: HandoffInputData)`` — receives the full handoff data with
  temporal slices

The framework detects the signature automatically and dispatches accordingly."""

type HandoffEnabledCallback = bool | Callable[..., MaybeAwaitable[bool]]
"""A boolean or a callable that determines if a handoff is enabled.

The callable form is dispatched by ``_is_handoff_enabled`` based on its
arity:

- 0 args: ``() -> bool`` — rare; useful for global feature flags.
- 1 arg:  ``(run_context) -> bool`` — works for both modes.
- 2 args: dispatched by orchestration mode:
   - LLM-orchestrated (``Handoff`` class): ``(run_context, target_agent)``.
     ``target_agent`` is the ``Agent`` the handoff would transfer to.
   - Code-orchestrated (``HandoffTarget`` via ``HandoffRoute``):
     ``(run_context, intent)``. ``intent`` is the matched ``Intent``
     dataclass.

Sync and async callables are both supported."""


def _handoff_callback_positionals(
    callback: OnHandoffCallback,
) -> tuple[bool, inspect.Parameter | None]:
    """Classify an ``on_handoff`` callback's positional arity.

    Only ``POSITIONAL_ONLY`` / ``POSITIONAL_OR_KEYWORD`` parameters can
    receive the positionally-passed ``(ctx, intent|data)`` arguments;
    keyword-only params and ``**kwargs`` cannot, and ``*args`` absorbs any
    count. Filtering by ``Parameter.kind`` (mirroring ``evaluate_enabled``)
    means a callback like ``(ctx, *, flag)`` or ``(ctx, **kw)`` is treated
    as a one-positional ``(ctx)`` callback rather than being called with a
    spurious second positional argument (which raises ``TypeError``).

    Args:
        callback: The user's ``on_handoff`` function.

    Returns:
        ``(wants_second_arg, second_positional_param)``. ``wants_second_arg``
        is ``True`` when the callback accepts a second positional argument
        (two or more positionals, or ``*args``). ``second_positional_param``
        is the second positional ``Parameter`` when one is named (for
        ``HandoffInputData`` type detection), else ``None``.
    """
    sig = inspect.signature(callback)
    positional_kinds = (
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    )
    positional: list[inspect.Parameter] = []
    has_var_positional = False
    for p in sig.parameters.values():
        if p.name == "self":
            continue
        if p.kind in positional_kinds:
            positional.append(p)
        elif p.kind is inspect.Parameter.VAR_POSITIONAL:
            has_var_positional = True
    second = positional[1] if len(positional) >= 2 else None
    wants_second = has_var_positional or len(positional) >= 2
    return wants_second, second


def _second_param_is_handoff_data(
    callback: OnHandoffCallback,
    second_param: inspect.Parameter | None,
) -> bool:
    """Detect whether the second positional param is typed ``HandoffInputData``.

    Resolves annotations via ``get_type_hints`` and falls back to the raw
    string annotation on ``NameError`` (an unresolvable forward reference).
    Only a bare ``HandoffInputData`` annotation qualifies — composite
    annotations (``list[HandoffInputData]``, ``HandoffInputData | None``)
    are intent callbacks, matching the resolved-path ``ann is
    HandoffInputData`` behaviour.

    Args:
        callback: The user's ``on_handoff`` function.
        second_param: The second positional ``Parameter``, or ``None`` when
            the callback has no named second positional (e.g. ``*args``).

    Returns:
        ``True`` when the second positional param is annotated exactly as
        ``HandoffInputData``.
    """
    if second_param is None:
        return False
    try:
        hints = typing.get_type_hints(callback)
        ann = hints.get(second_param.name, inspect.Parameter.empty)
        return ann is HandoffInputData
    except NameError as exc:
        # NameError is the expected failure mode when a forward-reference
        # string annotation cannot be resolved in the callback's module.
        # Other exceptions (ImportError, AttributeError from a broken
        # annotation) propagate so they are not silently swallowed.
        logger.debug(
            "get_type_hints failed for %s, falling back to string check: %s",
            callback,
            exc,
        )
        raw = second_param.annotation
        return raw is HandoffInputData or (
            isinstance(raw, str)
            and raw.strip()
            in (
                "HandoffInputData",
                "troopai.adk.handoffs.handoff_input_data.HandoffInputData",
            )
        )


async def invoke_on_handoff(
    callback: OnHandoffCallback,
    context: RunContext[Any],
    intent: Any,
    handoff_data: HandoffInputData | None = None,
) -> None:
    """Dispatch an on_handoff callback, detecting its signature.

    Inspects the callback's *positional* parameters to pick a variant:

    1. Zero or one positional → ``(ctx)`` — call with context only.
    2. Two+ positionals (or ``*args``), second typed ``HandoffInputData``
       → ``(ctx, data)``.
    3. Two+ positionals (or ``*args``) otherwise → ``(ctx, intent)``.

    Keyword-only params and ``**kwargs`` are ignored for arity — they
    cannot receive the positionally-passed second argument, so a callback
    like ``(ctx, *, flag)`` is a ``(ctx)`` callback, not ``(ctx, X)``.

    Handles both sync and async callbacks.

    Args:
        callback: The user's on_handoff function.
        context: The run context.
        intent: The validated typed input (or raw string/Intent).
        handoff_data: The full HandoffInputData (passed when available).
    """
    wants_second, second_param = _handoff_callback_positionals(callback)

    # ``callback`` is a union of three differently-arity ``Callable``
    # shapes; the dispatch below picks the right one at runtime. Bind
    # through a widened alias so the type checker does not try (and
    # fail) to reconcile the union against each concrete arity.
    dispatch: Callable[..., Any] = callback

    if wants_second:
        if handoff_data is not None and _second_param_is_handoff_data(callback, second_param):
            result = dispatch(context, handoff_data)
        else:
            result = dispatch(context, intent)
    else:
        result = dispatch(context)

    if asyncio.iscoroutine(result):
        await result


@dataclass(frozen=True)
class HandoffTarget(Generic[TAgent, TContext]):
    """
    Configuration for a single handoff destination.

    Created by HandoffRoute.when().to() and represents all settings for
    routing to a specific agent.

    Note: Source and target agent references are managed by the
    orchestrator, not stored in HandoffInputData. This keeps the
    data layer clean and focused on filterable content.

    Attributes:
        target: The agent to hand off to.
        on_handoff: Optional callback invoked when this handoff occurs.
        input_filter: Optional function to transform handoff data before passing to the target agent.
        enabled: Whether this route is active (bool or callable).
        config: Additional handoff configuration (strategy, window, budget).

    Example:
        from datetime import datetime
        from troopai.adk.handoffs.handoff_filters import remove_tool_calls
        from troopai.adk.run.context import RunContext

        # on_handoff with input — receives context and the Intent
        def log_refund(ctx: RunContext, intent: RefundIntent) -> None:
            logger.info(f"Refund for order {intent.order_id}")

        # on_handoff without input — receives context only
        def log_handoff(ctx: RunContext) -> None:
            logger.info("Handoff occurred")

        # on_handoff with full data — receives temporal slices
        def audit_handoff(ctx: RunContext, data: HandoffInputData) -> None:
            logger.info(f"Context: {len(data.context)} msgs, Output: {len(data.output)} msgs")

        # enabled callback - receives context and intent
        def during_business_hours(ctx: RunContext, intent: Intent) -> bool:
            current_hour = datetime.now().hour
            return 9 <= current_hour < 17

        HandoffTarget(
            target=refunds_agent,
            on_handoff=log_refund,
            input_filter=remove_tool_calls,
            enabled=during_business_hours
        )
    """

    target: Agent[TContext]
    """The agent to hand off to."""

    on_handoff: OnHandoffCallback | None = None
    """Optional callback invoked when this handoff occurs."""

    input_filter: HandoffInputFilter | None = None
    """Optional function to transform handoff data before passing to target agent."""

    enabled: HandoffEnabledCallback = True
    """Whether this route is active (bool or callable)."""

    config: HandoffConfig = HandoffConfig()
    """Additional handoff configuration."""

    async def invoke(
        self,
        intent: Intent,
        context: tuple[RunItem, ...],
        output: tuple[RunItem, ...],
        run_context: RunContext[TContext],
    ) -> tuple[Agent[TContext], HandoffInputData]:
        """Execute this handoff target.

        This is the internal execution logic that:

        1. Builds HandoffInputData from intent + context + output
        2. Applies input_filter if configured
        3. Calls on_handoff callback if provided

        Note: Lifecycle hooks are NOT called here - they remain in the runner
        for proper orchestration and observability.

        Args:
            intent: The Intent that triggered this handoff.
            context: Messages before the current agent's turn.
            output: Messages generated during the current agent's turn.
            run_context: The run context.

        Returns:
            Tuple of (target agent, filtered HandoffInputData).

        Example:
            target = await route.resolve(intent, run_context)
            if target:
                agent, data = await target.invoke(
                    intent, context_msgs, output_msgs, run_context,
                )
        """
        # 1. Build HandoffInputData
        handoff_data = HandoffInputData(
            intent=intent,
            context=context,
            output=output,
        )

        # 2. Apply input filter if configured. A raising callback is routed to
        # the configured config.on_error policy — identical to Handoff.invoke —
        # instead of crashing the run. BaseException (CancelledError) propagates.
        if self.input_filter is not None:
            try:
                handoff_data = self.input_filter(handoff_data)
            except Exception as exc:
                apply_callback_error_policy(
                    name=self.target.name,
                    config=self.config,
                    exc=exc,
                    callback_kind="input_filter",
                )

        # 3. Execute on_handoff callback if provided
        if self.on_handoff is not None:
            try:
                await invoke_on_handoff(
                    self.on_handoff,
                    run_context,
                    intent,
                    handoff_data=handoff_data,
                )
            except Exception as exc:
                apply_callback_error_policy(
                    name=self.target.name,
                    config=self.config,
                    exc=exc,
                    callback_kind="on_handoff",
                )

        return self.target, handoff_data
