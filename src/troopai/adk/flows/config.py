"""FlowConfig — cost-conservative bounds for a :class:`Flow` execution.

All defaults are cost-conservative (off / smallest / bounded):
developers opt INTO higher caps; the framework never forces them to opt
OUT of one.

This config is a sibling to :class:`troopai.adk.run.config.RunConfig`. Both
travel through :meth:`Runner.arun_flow`. ``RunConfig`` covers concerns
shared with single-agent runs (guardrails, max_total_turns, hooks);
``FlowConfig`` covers Flow-specific bounds (step count, listener
fan-out, error policy).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

FlowErrorPolicy = Literal["halt", "route_to_error_handler"]
"""Policy for what happens when a Flow step body raises an exception.

- ``"halt"`` — surfaces the exception immediately as
  :class:`troopai.adk.flows.exceptions.FlowStepError`. Cost-conservative
  default: a single failure stops the whole flow.
- ``"route_to_error_handler"`` — looks up a listener decorated as
  ``@flow_listen(FLOW_ERROR_TRIGGER)`` (the ``"__error__"`` route
  literal; see :data:`troopai.adk.flows.triggers.FLOW_ERROR_TRIGGER`)
  and fires it with the exception accessible via ``self.state``. If no
  error handler exists, falls back to ``"halt"`` semantics. Use for
  resilient flows where one branch's failure should not abort the
  whole run.
"""


@dataclass(frozen=True, kw_only=True)
class FlowConfig:
    """Cost-conservative bounds for a :class:`Flow` execution.

    Every field defaults to a safe, bounded value. Raise a default
    only when the workflow genuinely needs more room — never to escape
    a perceived inconvenience.

    Attributes:
        max_steps: Hard cap on the total number of step invocations
            across the entire flow run. Default ``100``. When the
            cap is hit, :class:`troopai.adk.flows.exceptions.FlowMaxStepsExceeded`
            is raised. Bound exists to prevent unbounded fan-out in
            mis-wired flows; raise it for legitimately long workflows.
        max_total_tokens: Optional cumulative cap on tokens consumed by
            inner :meth:`Runner.arun` calls inside step bodies. Default
            ``None`` (no cap). Set to a positive integer to enable; the
            executor checks after each step's usage is recorded. Sister
            to :attr:`troopai.adk.run.config.RunConfig.max_total_tokens`.
        max_listeners_per_step: Maximum number of downstream methods
            (direct listeners + routers + gates) that may fire on a
            single trigger arrival. Default ``20``. Enforced
            structurally at transition-table build time, raising
            :class:`troopai.adk.flows.exceptions.FlowDefinitionError` if
            exceeded.
        error_policy: How to handle exceptions raised by step bodies.
            Default ``"halt"``. See :data:`FlowErrorPolicy`.
    """

    max_steps: int = 100
    """Hard cap on total step invocations across the run."""

    max_total_tokens: int | None = None
    """Optional cumulative-token cap; ``None`` means no cap."""

    max_listeners_per_step: int = 20
    """Maximum downstream methods per trigger; enforced at table build."""

    error_policy: FlowErrorPolicy = "halt"
    """Behavior when a step body raises."""

    def __post_init__(self) -> None:
        """Validate :class:`FlowConfig` field values.

        Raises:
            ValueError: When any numeric bound is non-positive.
        """
        if self.max_steps < 1:
            raise ValueError(
                f"FlowConfig.max_steps must be >= 1; got {self.max_steps}.",
            )
        if self.max_listeners_per_step < 1:
            raise ValueError(
                f"FlowConfig.max_listeners_per_step must be >= 1; got {self.max_listeners_per_step}.",
            )
        if self.max_total_tokens is not None and self.max_total_tokens < 1:
            raise ValueError(
                f"FlowConfig.max_total_tokens must be >= 1 when set; got {self.max_total_tokens}.",
            )
