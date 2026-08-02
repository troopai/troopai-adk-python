"""FlowStepRateLimit — per-step sliding-window rate limit.

Mirrors :class:`troopai.adk.types.tools.tool_rate_limit.ToolRateLimit`
at the Flow step layer. Each step with a non-``None``
:attr:`FlowStep.__flow_rate_limit__` carries its own 60-second
sliding window: the executor maintains a per-step deque of monotonic
timestamps, admits each step invocation when ``len(deque) < rpm``,
and either waits (``behavior="wait"``) or raises
``FlowStepRateLimitExceeded`` (``behavior="error"``) on saturation.

Per-step scope: every concrete subclass that declares the gate has
its own bucket. Two steps that share a backend rate budget should
share a single helper or compose a future middleware once that
lands.

**Distributed-execution caveat**: enforcement is per-executor
(per-batch claim under :class:`FlowWorkerBackend`). When the
same step fires across batches claimed by different workers,
the bucket resets between claims — the documented ``rpm`` cap
applies *per batch window* in that mode, not globally across
all workers. Set ``rate_limit=None`` and enforce the limit at
the worker pool's boundary when a globally coordinated cap is
required.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

FlowStepRateLimitBehavior = Literal["wait", "error"]
"""What happens when a step's sliding window is saturated.

- ``"wait"``: the executor sleeps until a slot opens, then proceeds.
  Cost-conservative default — keeps the flow advancing without
  surfacing the saturation to the developer.
- ``"error"``: the executor raises :class:`FlowStepRateLimitExceeded`
  which routes through :attr:`FlowConfig.error_policy` exactly like
  any other step exception. Set this when downstream consumers want
  to observe rate-limit failures.
"""


@dataclass(frozen=True, kw_only=True)
class FlowStepRateLimit:
    """Per-step sliding-window rate limit.

    Identical surface to
    :class:`troopai.adk.types.tools.tool_rate_limit.ToolRateLimit`,
    re-typed for the Flow step layer so cross-layer type confusion
    is impossible. The 60-second window is fixed; the cap is
    :attr:`rpm`.

    Attributes:
        rpm: Step invocations allowed in any 60-second sliding window.
            Must be a positive integer.
        behavior: What happens when the window is saturated. Defaults
            to ``"wait"``.
        max_wait_seconds: Optional cap on the cumulative wait one
            ``acquire`` may incur when ``behavior="wait"``. When the
            cap would be exceeded, the acquire falls back to error
            semantics (raises) rather than blocking the event loop
            indefinitely. ``None`` (default) preserves the
            unbounded-wait behaviour. Ignored when ``behavior="error"``.
    """

    rpm: int
    """Step invocations allowed in any 60-second sliding window. Must be > 0."""

    behavior: FlowStepRateLimitBehavior = "wait"
    """``"wait"`` (default) or ``"error"`` on saturation."""

    max_wait_seconds: float | None = None
    """Optional cumulative-wait cap when ``behavior="wait"`` (must be > 0 when set)."""

    def __post_init__(self) -> None:
        if self.rpm <= 0:
            raise ValueError(f"FlowStepRateLimit.rpm must be positive, got {self.rpm}")
        if self.max_wait_seconds is not None and self.max_wait_seconds <= 0:
            raise ValueError(
                f"FlowStepRateLimit.max_wait_seconds must be positive, got {self.max_wait_seconds}",
            )
