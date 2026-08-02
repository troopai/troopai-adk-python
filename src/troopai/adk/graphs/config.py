"""``GraphConfig`` — cost levers and behavioural knobs for a graph run.

Composable with :class:`~troopai.adk.run.config.RunConfig`. A
:class:`~troopai.adk.graphs.graph.Graph` is the container that binds
``GraphConfig`` to the node roster; the ``RunConfig`` passed to
``Runner.arun_graph`` still applies the absolute safety nets
(``max_total_turns``, ``usage_limits``, tracing toggles).

Design principle. Every field here is either (a) graph-wide (e.g.
default node timeout) or (b) a budget the graph loop enforces across
supersteps (e.g. ``max_supersteps``). Per-node overrides live on
:class:`~troopai.adk.graphs.node.GraphNode` via typed ``retry`` and
``timeout`` fields; when those fields are ``None``, the graph-level
``default_retry`` and ``per_node_timeout`` apply.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from troopai.adk.graphs.node import NodeErrorHandlerFn


class NodeInputStrategy(StrEnum):
    """How a downstream node's :class:`ExecutableInput.content` is built.

    This is orthogonal to :class:`Merge` strategies:

    - :class:`Merge` decides WHAT the merged payload looks like
      (concat, last-wins, extend-items).
    - :class:`NodeInputStrategy` decides WHICH upstream context the
      downstream node sees (only the direct parent? the full run's
      history? the last N items across the graph?).

    Default is :attr:`LAST_OUTPUT` — the downstream node sees only
    its direct upstream (merged per the node's merge strategy). This
    mirrors ``Swarm``'s ``SCOPED`` default and keeps graph runs cheap
    by default.
    """

    LAST_OUTPUT = "last_output"
    """Downstream sees only merged upstream :class:`NodeResult` values.

    Default and cheapest. Each node is essentially a function from
    upstream output to its own output. No global history broadcast.
    """

    MERGED_OUTPUTS = "merged_outputs"
    """Downstream sees every node-completion's ``final_text`` so far.

    Useful when a late node needs a running document of everything
    produced. Token cost grows linearly with the number of completed
    nodes.
    """

    FULL_HISTORY = "full_history"
    """Downstream sees every Layer 3 :class:`RunItem` produced in the run.

    Expensive — full replay of every agent turn's history, every
    tool call, every reasoning block. Reserved for workflows that
    genuinely need it (auditing pipelines, compliance review).
    """


@dataclass(frozen=True)
class NodeRetryPolicy:
    """Per-node retry configuration.

    Applied by the graph loop per node firing: each attempt is retried
    (subject to ``retry_on`` and ``max_attempts``) with exponential
    backoff.

    Attributes:
        max_attempts: Maximum number of times to run the node,
            including the first attempt. ``1`` means no retries.
            Default ``1``.
        initial_backoff: Seconds to wait before the first retry.
            Doubles each subsequent retry. Default ``1.0``.
        max_backoff: Cap on the backoff duration. Default ``30.0``.
        retry_on: Tuple of exception classes to retry on. Any other
            exception propagates immediately. Default empty → retry
            on every ``Exception`` (subject to ``max_attempts``).
    """

    max_attempts: int = 1
    """Maximum attempts. ``1`` disables retries."""

    initial_backoff: float = 1.0
    """Seconds before the first retry."""

    max_backoff: float = 30.0
    """Cap on backoff duration."""

    retry_on: tuple[type[Exception], ...] = ()
    """Exception classes to retry on. Empty = all exceptions."""

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError(f"NodeRetryPolicy.max_attempts must be >= 1, got {self.max_attempts}")
        if self.initial_backoff <= 0:
            raise ValueError(f"NodeRetryPolicy.initial_backoff must be > 0, got {self.initial_backoff}")
        if self.max_backoff < self.initial_backoff:
            raise ValueError(
                f"NodeRetryPolicy.max_backoff ({self.max_backoff}) must be >= initial_backoff ({self.initial_backoff})"
            )


@dataclass(frozen=True)
class GraphConfig:
    """Graph-wide budgets, defaults, and behaviour knobs.

    Attributes:
        max_supersteps: Hard cap on the number of BSP supersteps the
            driver may execute before raising. Protects against
            unbounded cycles in a misconfigured graph. Default
            ``50`` — generous for production pipelines, tight enough
            to catch runaway loops in development. Distinct from
            :attr:`RunConfig.max_total_turns` which counts LLM calls
            (a single node may make many LLM calls in its inner run).
        max_total_tokens: Optional graph-wide cumulative token cap.
            Checked against :attr:`GraphState.cumulative_usage` at the
            top of each superstep. ``None`` = no cap.
        node_input: Default :class:`NodeInputStrategy`. A per-node
            override via ``GraphNode.metadata["input"]`` is not yet
            implemented.
        default_retry: Default per-node :class:`NodeRetryPolicy`; a
            node may override it via :attr:`GraphNode.retry`.
        per_node_timeout: Default per-attempt node timeout in seconds.
            ``None`` = no timeout. Override per node via
            ``GraphNode.timeout``.
        fail_fast: When ``True`` (default), the first node error
            cancels all sibling tasks in the same superstep and
            surfaces on :class:`GraphRunResult`. When ``False``,
            siblings finish and their results are preserved;
            downstream nodes depending on the failed node do not
            fire.
    """

    max_supersteps: int = 50
    """BSP superstep cap. Default 50."""

    max_total_tokens: int | None = None
    """Optional cumulative token cap across the whole graph run."""

    node_input: NodeInputStrategy = NodeInputStrategy.LAST_OUTPUT
    """Default per-node input strategy. ``LAST_OUTPUT`` is cheapest."""

    default_retry: NodeRetryPolicy = field(default_factory=NodeRetryPolicy)
    """Default per-node retry policy; a node may override it via ``GraphNode.retry``."""

    per_node_timeout: float | None = None
    """Default per-attempt node timeout in seconds. ``None`` = no timeout. Override per node via ``GraphNode.timeout``."""

    fail_fast: bool = True
    """First error cancels siblings and surfaces on the result."""

    default_error_handler: NodeErrorHandlerFn | None = None
    """Graph-level fallback error handler.  Applied to any node whose
    :attr:`~troopai.adk.graphs.node.GraphNode.on_error` is ``None``.
    When both are ``None`` the original exception propagates unchanged.
    A per-node ``on_error`` always takes precedence over this field."""

    def __post_init__(self) -> None:
        if self.max_supersteps <= 0:
            raise ValueError(f"GraphConfig.max_supersteps must be > 0, got {self.max_supersteps}")
        if self.per_node_timeout is not None and self.per_node_timeout <= 0:
            raise ValueError(f"GraphConfig.per_node_timeout must be > 0 when set, got {self.per_node_timeout}")


__all__ = [
    "GraphConfig",
    "NodeInputStrategy",
    "NodeRetryPolicy",
]
