"""FlowStepContext — typed argument passed to step gate callables.

Step-level gates on :class:`Flow` decorators (``requires_approval``,
``enabled``) accept either a bool or a callable. The callable form
receives a :class:`FlowStepContext` describing the step about to fire
and the current state of the run, and returns a bool.

Mirrors the :class:`ToolContext` pattern on :class:`FunctionTool` —
each gate evaluator gets a small, typed snapshot of the call site so
the decision can be state-dependent without leaking framework
internals.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from troopai.adk.flows.triggers import FlowTriggerEvent
from troopai.adk.utils.typedef import MaybeAwaitable

if TYPE_CHECKING:
    from troopai.adk.run.context import RunContext


type FlowStepGate = bool | Callable[[FlowStepContext[Any]], MaybeAwaitable[bool]]
"""Typed alias for step-level gate values.

A gate is either a literal ``bool`` or a callable that receives the
:class:`FlowStepContext` for the step about to fire and returns a
bool (sync or async).

Used as the typed shape for :attr:`FlowStep.__flow_requires_approval__`
and :attr:`FlowStep.__flow_enabled__` (mirrors the
:attr:`FunctionTool.requires_approval` and :attr:`FunctionTool.enabled`
pattern at the step layer).
"""


@dataclass(frozen=True, kw_only=True)
class FlowStepContext[StateT]:
    """Snapshot of the call site passed to step-level gate callables.

    Generic over the flow's typed state so gate callables receive a
    properly-narrowed :attr:`flow_state` (e.g. annotate the gate as
    ``def gate(ctx: FlowStepContext[MyState]) -> bool: ...`` and the
    type checker resolves ``ctx.flow_state`` to ``MyState``).

    Constructed by the executor immediately before a step's gates are
    evaluated. Frozen so gate implementations can keep references
    across awaits without worrying about mutation under their feet.

    Attributes:
        step_name: Method name of the step about to be invoked.
        flow_id: Stable identifier of the running flow instance.
        flow_state: The developer-typed shared state
            (``Flow.state``). Same object the step body would access
            via ``self.state``; typed via the ``StateT`` parameter.
        context: The :class:`RunContext` attached by
            ``Runner.arun_flow`` for the duration of the run, or
            ``None`` outside of a run.
        completed_steps: Tuple of step method names that have already
            run in this flow (in completion order). Empty before the
            first step fires.
        triggers: Tuple of :class:`FlowTriggerEvent` instances that
            caused this step to be scheduled. Cardinality reflects
            the listener shape:

            - Empty tuple for ``@flow_start`` (fires unconditionally).
            - Single-element tuple for a direct ``@flow_listen("step")``
              or for an OR-gate listener (which fires on the first
              arrival).
            - Multi-element tuple for an AND-gate listener — every
              required trigger appears, in arrival order.

            Each event carries the trigger's ``name``, the
            ``source_step`` that produced it, and the ``kind`` of
            production — so a gate can branch on provenance
            (``"step_completion"`` vs ``"route_label"``) without
            re-walking the registry.

            Always a tuple; never ``None`` (use ``len(triggers) == 0``
            to test for ``@flow_start``).
    """

    step_name: str
    """Method name of the step about to be invoked."""

    flow_id: str
    """Stable identifier of the running flow instance."""

    flow_state: StateT
    """Developer-typed shared state (``Flow.state``)."""

    context: RunContext[Any] | None
    """:class:`RunContext` for the run, or ``None`` outside a run."""

    completed_steps: tuple[str, ...]
    """Step names that have already completed in this run."""

    triggers: tuple[FlowTriggerEvent, ...]
    """:class:`FlowTriggerEvent`\\ s that scheduled this step (see class docstring)."""
