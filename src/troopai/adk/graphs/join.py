"""Join barrier — AND-join / OR-join fan-in primitive.

LangGraph implements joins implicitly through ``NamedBarrierValue`` in
``libs/langgraph/channels/named_barrier_value.py``: the channel
itself tracks the set of names it expects and becomes "available"
only when all expected names have written. The engine checks channel
availability to decide what to schedule next.

This module ports the idea as a standalone class the graph loop
queries directly. A :class:`JoinBarrier` belongs to a single target
node and tracks which upstream node ids have "arrived" (produced a
:class:`NodeResult`) since the last time the target executed.

Why a first-class class instead of engine-level edge counting:

- **Self-contained.** The barrier owns the decision "am I ready?".
  The graph loop asks :meth:`is_ready`. No edge-counting / visited-set
  bookkeeping leaks into the loop.

- **Resettable.** A barrier in a cycle (node A → B → A) resets after
  each firing. :meth:`consume` clears the arrival set so the next
  loop iteration starts clean.

- **Two semantics one class.** AND-join and OR-join differ only in
  the :meth:`is_ready` predicate. Passing the semantics in via
  :class:`JoinSemantics` avoids a class hierarchy.

Only one-time-per-superstep arrivals are tracked. If the same
upstream node re-fires before the target consumes, the second
arrival overwrites the first in :attr:`arrivals` — correct here
because a superstep cannot fire the same node twice. Dynamic
fan-out (Send-equivalent), which would need per-arrival queues, is
not supported.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from troopai.adk.orchestration.executable import NodeResult


logger = logging.getLogger(__name__)


class JoinSemantics(StrEnum):
    """How a :class:`JoinBarrier` decides it is ready to fire."""

    AND = "and"
    """Fire only when EVERY expected upstream has arrived.

    Default. Safer than OR because it prevents the downstream node
    from running with partial inputs (Strands' footgun — OR is the
    default there and silently executes targets with half their
    upstreams). AND-join matches what most developers mean when they
    draw a fan-in arrow.
    """

    OR = "or"
    """Fire as soon as ANY expected upstream has arrived.

    Use for "first-to-respond wins" patterns. The barrier still
    resets after firing, so a late-arriving upstream in the same
    cycle queues up the next firing.
    """


@dataclass
class JoinBarrier:
    """Fan-in barrier for a single target node.

    A :class:`JoinBarrier` is owned by the graph loop — one per
    graph node that has more than one incoming edge (single-parent
    nodes don't need a barrier; their input is the single upstream
    result).

    Attributes:
        target: Id of the node this barrier feeds. Used for logging
            and for the debug repr.
        expected: Set of upstream node ids that this barrier waits
            on. Derived at compile time from the graph's edge set —
            every edge ``(u, target)`` contributes ``u`` to this set.
        semantics: :class:`JoinSemantics.AND` (default) or
            :class:`JoinSemantics.OR`.
        arrivals: Map of arrived upstream id → the :class:`NodeResult`
            they produced. Populated by :meth:`record` during a
            superstep. Cleared by :meth:`consume` right before the
            target executes.
        skipped: Upstream ids whose conditional edge predicate returned
            ``False`` this cycle. An AND-join treats ``arrivals ∪
            skipped == expected`` as satisfied (while requiring at
            least one real arrival) so that conditional edges don't
            deadlock the barrier. Cleared by :meth:`consume`.
        failed: Upstream ids whose conditional edge predicate raised an
            exception this cycle. Distinct from :attr:`skipped`
            (predicate returned ``False``) — a failed predicate
            indicates an error, not an intentional branch skip.
            An AND-join with any failed upstream is NOT considered ready
            (fail-closed). Cleared by :meth:`consume`.
    """

    target: str
    expected: frozenset[str]
    semantics: JoinSemantics = JoinSemantics.AND
    arrivals: dict[str, NodeResult] = field(default_factory=dict)
    skipped: set[str] = field(default_factory=set)
    failed: set[str] = field(default_factory=set)

    def record(self, source: str, result: NodeResult) -> None:
        """Record an upstream arrival.

        Ignores arrivals from sources not in :attr:`expected` — this
        shouldn't happen in a well-formed graph but if it does
        (a conditional edge fires an unexpected path) we log a
        warning and drop the arrival rather than silently corrupting
        the barrier.

        Args:
            source: Upstream node id that just produced a result.
            result: The :class:`NodeResult` it produced.
        """
        if source not in self.expected:
            logger.warning(
                "JoinBarrier(target=%s): unexpected arrival from %r (expected %s); dropping.",
                self.target,
                source,
                sorted(self.expected),
            )
            return
        self.arrivals[source] = result
        self.skipped.discard(source)
        self.failed.discard(source)

    def record_skip(self, source: str) -> None:
        """Record that an upstream edge fired its predicate as ``False``.

        Without this call, an AND-join with a conditional upstream edge
        would deadlock: the barrier would wait forever for a source
        that will never arrive this cycle. Recording a skip means the
        barrier treats ``arrivals ∪ skipped == expected`` as satisfied,
        but still requires at least one real arrival to fire (all-
        skipped is a no-op).

        If the same source later arrives (cycles), ``record`` clears
        it from :attr:`skipped`. A clean skip also clears a prior
        :meth:`record_fail` mark for the same source — symmetric with
        :meth:`record_fail`, which clears :attr:`skipped` — so a transient
        predicate error in an earlier superstep does not permanently lock
        the barrier once the upstream resolves to an intentional skip.
        """
        if source not in self.expected:
            return
        if source in self.arrivals:
            return
        self.failed.discard(source)
        self.skipped.add(source)

    def record_fail(self, source: str) -> None:
        """Record that an upstream edge predicate raised an exception.

        Distinct from :meth:`record_skip` (predicate returned ``False``):
        a failed predicate is an error condition, not an intentional branch
        skip. An AND-join with any failed upstream is NOT ready (fail-closed)
        — the downstream node must not fire with potentially corrupt inputs.

        If the same source later arrives with a real result, it is cleared
        from :attr:`failed` by :meth:`record`. The caller
        (``graph_loop.py``) is responsible for logging the exception before
        calling this method.

        Args:
            source: Upstream node id whose predicate raised.
        """
        if source not in self.expected:
            return
        if source in self.arrivals:
            return
        self.skipped.discard(source)
        self.failed.add(source)

    def is_ready(self) -> bool:
        """``True`` when the barrier's semantics are satisfied.

        An AND-join with any failed upstream (predicate raised) is never
        ready — this ensures a downstream node does not fire with
        partial or corrupt inputs when an upstream predicate errors.
        """
        if len(self.failed) > 0:
            return False
        if self.semantics == JoinSemantics.AND:
            return (set(self.arrivals.keys()) | self.skipped) >= self.expected and len(self.arrivals) > 0
        # OR
        return len(self.arrivals) > 0

    def consume(self) -> tuple[list[NodeResult], list[str]]:
        """Return arrived results (ordered by source id) and reset.

        The loop calls this right before invoking the target node.
        The return order is deterministic (source ids sorted) so
        merge strategies don't see completion-order nondeterminism —
        LangGraph's ``_algo.py::prepare_next_tasks`` uses the same
        path-sorted approach.

        Returns:
            ``(results, sources)``: Parallel lists of arrived
            :class:`NodeResult` values and their source node ids,
            both sorted by source id.
        """
        sorted_sources = sorted(self.arrivals.keys())
        results = [self.arrivals[s] for s in sorted_sources]
        self.arrivals = {}
        self.skipped = set()
        self.failed = set()
        return results, sorted_sources


__all__ = [
    "JoinBarrier",
    "JoinSemantics",
]
