"""``GraphNode`` and ``GraphEdge`` — the basic structural units of a graph.

A :class:`GraphNode` is a named wrapper around an
:class:`~troopai.adk.orchestration.executable.Executable`. A
:class:`GraphEdge` describes a directed connection between two nodes
and carries the optional predicate that decides whether the edge
fires.

Both are frozen dataclasses. Mutable per-run data (who arrived, what
result did the node produce) lives on :class:`GraphState` and
:class:`JoinBarrier` — the nodes themselves are read-only at runtime.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

from troopai.adk.graphs.join import JoinSemantics
from troopai.adk.graphs.merge import DEFAULT_MERGE, MergeFn

if TYPE_CHECKING:
    from troopai.adk.graphs.config import NodeRetryPolicy
    from troopai.adk.orchestration.executable import Executable, NodeResult


logger = logging.getLogger(__name__)


class NodeErrorHandler(Protocol):
    """Protocol for per-node error handler callbacks.

    Invoked by the graph loop when a node's execution fails after all
    retry attempts are exhausted. A handler MAY return a fallback
    :class:`~troopai.adk.orchestration.executable.NodeResult` to substitute
    for the failed node's output; returning ``None`` causes the original
    exception to propagate unchanged.

    The handler signature is::

        async def my_handler(node_id: str, exc: BaseException) -> NodeResult | None: ...

    or its sync equivalent (the loop awaits only when the return value is
    awaitable).

    Exceptions raised INSIDE the handler propagate immediately — there is no
    silent double-failure suppression.
    """

    async def __call__(
        self,
        node_id: str,
        exc: BaseException,
    ) -> NodeResult[Any] | None:
        """Handle a post-retry node failure.

        Args:
            node_id: Id of the node that failed.
            exc: The exception raised after all retry attempts were exhausted.

        Returns:
            A fallback :class:`~troopai.adk.orchestration.executable.NodeResult`
            to substitute for the failed output, or ``None`` to let the
            original exception propagate.
        """
        ...  # pragma: no cover


# Actual callable type accepted by the handler fields — a plain Protocol
# annotation would not survive frozen-dataclass field storage, so we use a
# type alias that the type-checker can verify at call sites.
NodeErrorHandlerFn = Callable[
    [str, BaseException],
    "Awaitable[NodeResult[Any] | None] | NodeResult[Any] | None",
]
"""Callable type alias for a per-node error handler.

Accepts the failing node id and the exhausted exception; returns a fallback
:class:`~troopai.adk.orchestration.executable.NodeResult` or ``None`` to
propagate.  May be sync or async — the loop awaits when the return value is
awaitable.
"""


# Node ids feed OTel span names, log messages, and the ``graph_path``
# tuple on stream events. Keep them slug-safe so they don't need
# escaping in any of those sinks.
_NODE_ID_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_\-]*$")


# Predicate signature for a conditional edge. Takes the upstream
# :class:`NodeResult` and returns ``True`` iff the edge should fire.
# Can be sync or async — the graph loop awaits if needed. Pure function
# — MUST NOT mutate the result or produce side effects.
EdgeCondition = Callable[["NodeResult"], "bool | Any"]
"""Predicate called on an upstream :class:`NodeResult` to decide edge firing.

The ``Any`` in the return union accommodates awaitables (``Coroutine[Any,
Any, bool]``). The graph loop does ``result = predicate(...); if
inspect.isawaitable(result): result = await result`` — see
``graph_loop.py::_edge_fires``."""


@dataclass(frozen=True)
class GraphNode:
    """A named node wrapping an :class:`Executable`.

    Attributes:
        id: Unique node id within the owning graph. MUST match
            ``[a-zA-Z0-9][a-zA-Z0-9_-]*``. Used as the label on
            :meth:`Merge.concat_text` chunks, as the source id in
            :class:`JoinBarrier` arrivals, and as the
            :attr:`GraphStreamEvent.node_id` on every emitted event.
        executable: The :class:`Executable` the graph loop invokes
            when this node fires. Usually one of the adapters from
            :mod:`troopai.adk.graphs.adapters` (``AgentExecutable``,
            ``SwarmExecutable``, ``CallableExecutable``) or a nested
            :class:`~troopai.adk.graphs.graph.Graph` itself.
        merge: How to fold multiple upstream :class:`NodeResult`
            values into this node's :class:`ExecutableInput`.
            Defaults to :data:`Merge.concat_text` — the readable
            fan-in default.
        join: AND (default) or OR semantics for the fan-in barrier
            when this node has multiple incoming edges. Ignored when
            the node has zero or one incoming edges.
        description: Optional human-readable blurb. Emitted on
            :class:`GraphStreamEvent`\\ s and OTel spans for richer
            observability.
        metadata: Free-form dict — escape hatch for attaching tags
            (cost-center, owner team) to a node.
            Surfaced on events and hooks.
        retry: Optional per-node :class:`NodeRetryPolicy`. ``None``
            inherits the graph-level :attr:`GraphConfig.default_retry`.
        timeout: Optional per-node per-attempt timeout in seconds.
            ``None`` inherits :attr:`GraphConfig.per_node_timeout`.
        on_error: Optional error handler called when the node's execution
            fails AFTER all retry attempts are exhausted. The handler
            MAY return a fallback :class:`~troopai.adk.orchestration.executable.NodeResult`
            to substitute for the failed output, or ``None`` to propagate
            the original exception. ``None`` ⇒ inherit
            :attr:`GraphConfig.default_error_handler`, then raise if
            both are ``None``. Exceptions raised inside the handler
            propagate immediately — no silent double-failure.
    """

    id: str
    """Unique node id within the graph."""

    executable: Executable[Any]
    """The :class:`Executable` the graph loop runs when this node fires."""

    merge: MergeFn = DEFAULT_MERGE
    """Fan-in strategy. Ignored when this node has ≤1 upstream edges."""

    join: JoinSemantics = JoinSemantics.AND
    """Fan-in semantics. Default AND — the safer, more-readable option."""

    description: str | None = None
    """Human-readable blurb (emitted on spans and events)."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Free-form tags (cost-center, owner team) attached to the node."""

    retry: NodeRetryPolicy | None = None
    """Per-node retry override. ``None`` ⇒ inherit
    :attr:`GraphConfig.default_retry`."""

    timeout: float | None = None
    """Per-node per-attempt timeout (seconds) override. ``None`` ⇒
    inherit :attr:`GraphConfig.per_node_timeout`."""

    on_error: NodeErrorHandlerFn | None = None
    """Per-node error handler. ``None`` ⇒ inherit
    :attr:`GraphConfig.default_error_handler`, then raise if both are
    ``None``."""

    def __post_init__(self) -> None:
        """Validate the node id and that the executable is non-``None``."""
        if _NODE_ID_PATTERN.match(self.id) is None:
            raise ValueError(
                f"GraphNode.id={self.id!r} is invalid. Node ids must match "
                f"{_NODE_ID_PATTERN.pattern} so they are safe as span names, "
                "log tags, and graph_path elements."
            )
        if self.executable is None:
            raise ValueError(f"GraphNode(id={self.id!r}).executable must not be None.")


@dataclass(frozen=True)
class GraphEdge:
    """A directed edge from ``source`` to ``target``.

    Attributes:
        source: Upstream node id. MUST refer to a node in the same
            :class:`~troopai.adk.graphs.graph.Graph`.
        target: Downstream node id. MUST refer to a node in the same
            graph.
        when: Optional predicate on the upstream :class:`NodeResult`.
            When ``None`` the edge always fires. Otherwise the graph
            loop calls the predicate (awaiting if coroutine) and
            fires only on truthy return. The predicate MUST be pure.
        label: Optional label propagated to
            :attr:`ExecutableInput.edge_label`. Useful when the
            downstream adapter wants to know which branch fired
            (e.g. "approved" vs "rejected").
        priority: When two edges from the same source could fire in
            the same superstep and their targets would otherwise
            execute concurrently, higher ``priority`` fires first.
            Default ``0``. Ties broken by lexicographic target id.
            The loop currently executes all ready nodes in parallel
            within a superstep, so priority is advisory only; it
            takes effect only under non-concurrent step scheduling,
            which is not yet implemented.
    """

    source: str
    """Upstream node id."""

    target: str
    """Downstream node id."""

    when: EdgeCondition | None = None
    """Optional pure predicate on the upstream result."""

    label: str | None = None
    """Optional label propagated to the downstream executable."""

    priority: int = 0
    """Higher-first tie-break; advisory until non-concurrent scheduling is implemented."""

    def __post_init__(self) -> None:
        """Forbid self-loops; they're legal in a DAG sense but almost
        always a typo. If a user genuinely wants a self-loop they can
        route through an intermediate node (readability wins).
        """
        if self.source == self.target:
            raise ValueError(
                f"GraphEdge source and target are both {self.source!r}. "
                "Self-loops are not allowed — route through an "
                "intermediate node or use a CustomPolicy on a Swarm."
            )


__all__ = [
    "EdgeCondition",
    "GraphEdge",
    "GraphNode",
    "NodeErrorHandler",
    "NodeErrorHandlerFn",
]
