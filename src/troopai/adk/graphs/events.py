"""Graph stream events — ``TypedEvent(dict)`` subclasses for zero-cost streaming.

Why ``dict`` subclasses instead of ``@dataclass(frozen=True)`` (as
``swarms/events.py`` uses)?

On a streaming hot path the framework routinely emits thousands of
events per second — per-token deltas from nested agents inside a graph
will dominate throughput. A Pydantic model (``.model_dump()``) or a
frozen dataclass (``asdict()``) pays a per-event serialisation cost
every time a consumer wants to write the event to a log, a message
queue, or a DB. A plain ``dict`` is already in its serialised shape —
``json.dumps(ev)`` works directly.

The pattern comes from Strands' ``TypedEvent`` (see
``src/strands/event_loop/streaming.py``): event classes subclass
``dict``, the ``__init__`` validates required keys and populates them,
consumers can index either via attribute (``ev.node_id``) or via
key (``ev["node_id"]``). Zero conversion cost on every path.

Nesting. Every event carries ``graph_path: tuple[str, ...]`` — a
single-element tuple containing the id of the emitting graph
(e.g. ``(graph_id,)``). While the type allows for multi-element
paths to support future nested-trace propagation, current
implementations of the graph loop do not extend or prepend to this
path; nested graphs run non-streaming and surface only their
structural boundaries (``graph.node_start`` / ``graph.node_end``)
to the outer run.

Discriminator. Every event has a ``type`` key with a ``Literal``
value (e.g. ``"graph.node_start"``). Consumers can switch on
``ev["type"]`` for wire-format handling, or on ``isinstance(ev,
NodeStartEvent)`` for Python-side dispatch.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from troopai.adk.graphs.interrupt import Interrupt
    from troopai.adk.graphs.result import GraphRunStatus
    from troopai.adk.orchestration.executable import ExecutableInput, NodeResult


# Event type discriminator constants. Kept as module-level strings so
# consumers can switch on them without importing the event classes.
GRAPH_START = "graph.start"
GRAPH_END = "graph.end"
SUPERSTEP_START = "graph.superstep_start"
SUPERSTEP_END = "graph.superstep_end"
NODE_START = "graph.node_start"
NODE_END = "graph.node_end"
NODE_ERROR = "graph.node_error"
NODE_STREAM = "graph.node_stream"
NODE_INTERRUPT = "graph.node_interrupt"


class GraphStreamEvent(dict):
    """Base class for every graph-emitted stream event.

    All subclasses populate a ``type`` discriminator key and a
    ``graph_path`` tuple in their ``__init__``. Callers can read keys
    either via item access (``ev["type"]``) or via the named
    attributes the subclasses expose.

    This ``dict`` base is not instantiated directly; it exists so
    consumers can ``isinstance(ev, GraphStreamEvent)`` to filter
    graph-scoped events out of a multiplexed stream.
    """


class GraphStartEvent(GraphStreamEvent):
    """Emitted once at the top of a graph run, before the first superstep.

    Keys:
        type: Always :data:`GRAPH_START`.
        graph_path: Single-element tuple ``(graph_id,)`` identifying
            the emitting graph.
        graph_id: Id of the graph that is starting.
        description: Optional human-readable description.
        entry_node: The entry node id.
        terminal_nodes: The terminal node ids, sorted.
    """

    def __init__(
        self,
        *,
        graph_path: tuple[str, ...],
        graph_id: str,
        description: str | None,
        entry_node: str,
        terminal_nodes: tuple[str, ...],
    ) -> None:
        super().__init__(
            type=GRAPH_START,
            graph_path=graph_path,
            graph_id=graph_id,
            description=description,
            entry_node=entry_node,
            terminal_nodes=terminal_nodes,
        )


class GraphEndEvent(GraphStreamEvent):
    """Emitted exactly once at the end of a graph run.

    Keys:
        type: Always :data:`GRAPH_END`.
        graph_path: Same as on :class:`GraphStartEvent`.
        graph_id: Id of the graph that is ending.
        status: The terminal :class:`GraphRunStatus`.
        final_output: Aggregate graph output.
        total_supersteps: Number of supersteps executed.
    """

    def __init__(
        self,
        *,
        graph_path: tuple[str, ...],
        graph_id: str,
        status: GraphRunStatus,
        final_output: Any,
        total_supersteps: int,
    ) -> None:
        super().__init__(
            type=GRAPH_END,
            graph_path=graph_path,
            graph_id=graph_id,
            status=status,
            final_output=final_output,
            total_supersteps=total_supersteps,
        )


class SuperstepStartEvent(GraphStreamEvent):
    """Emitted at the top of each superstep.

    Keys:
        type: Always :data:`SUPERSTEP_START`.
        graph_path: Enclosing graph path.
        superstep: 1-indexed superstep counter.
        ready_nodes: Node ids about to fire in this superstep (sorted).
    """

    def __init__(
        self,
        *,
        graph_path: tuple[str, ...],
        superstep: int,
        ready_nodes: tuple[str, ...],
    ) -> None:
        super().__init__(
            type=SUPERSTEP_START,
            graph_path=graph_path,
            superstep=superstep,
            ready_nodes=ready_nodes,
        )


class SuperstepEndEvent(GraphStreamEvent):
    """Emitted after a superstep completes.

    Keys:
        type: Always :data:`SUPERSTEP_END`.
        graph_path: Enclosing graph path.
        superstep: 1-indexed superstep counter.
        fired_nodes: Node ids that fired successfully this superstep.
        errored_nodes: Node ids that raised this superstep (empty when
            no errors, or when ``fail_fast`` tripped and siblings were
            cancelled before they could error).
    """

    def __init__(
        self,
        *,
        graph_path: tuple[str, ...],
        superstep: int,
        fired_nodes: tuple[str, ...],
        errored_nodes: tuple[str, ...],
    ) -> None:
        super().__init__(
            type=SUPERSTEP_END,
            graph_path=graph_path,
            superstep=superstep,
            fired_nodes=fired_nodes,
            errored_nodes=errored_nodes,
        )


class NodeStartEvent(GraphStreamEvent):
    """Emitted before a node's :meth:`Executable.invoke`.

    Keys:
        type: Always :data:`NODE_START`.
        graph_path: Enclosing graph path.
        node_id: Id of the node about to fire.
        superstep: Current superstep counter.
        from_nodes: Upstream node ids that triggered this firing.
        edge_label: Optional label of the triggering edge.
    """

    def __init__(
        self,
        *,
        graph_path: tuple[str, ...],
        node_id: str,
        superstep: int,
        from_nodes: tuple[str, ...],
        edge_label: str | None,
        input: ExecutableInput | None = None,
    ) -> None:
        super().__init__(
            type=NODE_START,
            graph_path=graph_path,
            node_id=node_id,
            superstep=superstep,
            from_nodes=from_nodes,
            edge_label=edge_label,
            input=input,
        )


class NodeEndEvent(GraphStreamEvent):
    """Emitted after a node's :meth:`Executable.invoke` returns cleanly.

    Keys:
        type: Always :data:`NODE_END`.
        graph_path: Enclosing graph path.
        node_id: Id of the node that just fired.
        superstep: Current superstep counter.
        result: The produced :class:`NodeResult`.
    """

    def __init__(
        self,
        *,
        graph_path: tuple[str, ...],
        node_id: str,
        superstep: int,
        result: NodeResult,
    ) -> None:
        super().__init__(
            type=NODE_END,
            graph_path=graph_path,
            node_id=node_id,
            superstep=superstep,
            result=result,
        )


class NodeErrorEvent(GraphStreamEvent):
    """Emitted when a node raises an exception.

    Keys:
        type: Always :data:`NODE_ERROR`.
        graph_path: Enclosing graph path.
        node_id: Id of the node that raised.
        superstep: Current superstep counter.
        error_type: Class name of the exception.
        error_message: String form of the exception.
    """

    def __init__(
        self,
        *,
        graph_path: tuple[str, ...],
        node_id: str,
        superstep: int,
        error_type: str,
        error_message: str,
    ) -> None:
        super().__init__(
            type=NODE_ERROR,
            graph_path=graph_path,
            node_id=node_id,
            superstep=superstep,
            error_type=error_type,
            error_message=error_message,
        )


class NodeStreamEvent(GraphStreamEvent):
    """Wrapper around a streamed event bubbling up from a node.

    When a node's :meth:`Executable.stream_async` yields an event, the
    graph loop re-emits it wrapped in this envelope with ``graph_path``
    and ``node_id`` prepended. This is used primarily for agents
    forwarding per-token deltas; nested graphs currently run
    non-streaming and do not emit interior events.

    Keys:
        type: Always :data:`NODE_STREAM`.
        graph_path: Enclosing graph path.
        node_id: Id of the node whose inner stream produced this event.
        inner: The original event yielded by the inner executable.
    """

    def __init__(
        self,
        *,
        graph_path: tuple[str, ...],
        node_id: str,
        inner: Any,
    ) -> None:
        super().__init__(
            type=NODE_STREAM,
            graph_path=graph_path,
            node_id=node_id,
            inner=inner,
        )


class NodeInterruptEvent(GraphStreamEvent):
    """A node raised InterruptException; the run is suspending.

    The pending :class:`~troopai.adk.graphs.interrupt.Interrupt` is carried
    so consumers can prompt the human and resume via
    :class:`~troopai.adk.graphs.interrupt.GraphResume` and the
    checkpoint-resume surface.

    Keys:
        type: Always :data:`NODE_INTERRUPT`.
        graph_path: Identifies the emitting graph (single-element tuple
            ``(graph.id,)``).
        node_id: Id of the node that requested the interrupt.
        interrupt: The :class:`~troopai.adk.graphs.interrupt.Interrupt`
            describing the pending decision.
    """

    def __init__(
        self,
        *,
        graph_path: tuple[str, ...],
        node_id: str,
        interrupt: Interrupt,
    ) -> None:
        super().__init__(
            type=NODE_INTERRUPT,
            graph_path=graph_path,
            node_id=node_id,
            interrupt=interrupt,
        )


__all__ = [
    "GRAPH_END",
    "GRAPH_START",
    "NODE_END",
    "NODE_ERROR",
    "NODE_INTERRUPT",
    "NODE_START",
    "NODE_STREAM",
    "SUPERSTEP_END",
    "SUPERSTEP_START",
    "GraphEndEvent",
    "GraphStartEvent",
    "GraphStreamEvent",
    "NodeEndEvent",
    "NodeErrorEvent",
    "NodeInterruptEvent",
    "NodeStartEvent",
    "NodeStreamEvent",
    "SuperstepEndEvent",
    "SuperstepStartEvent",
]
