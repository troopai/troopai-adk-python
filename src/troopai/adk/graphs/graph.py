"""``Graph`` — the compiled, immutable, composable orchestration primitive.

A :class:`Graph` is the frozen artifact produced by
:meth:`GraphBuilder.compile`. It carries the full definition of a
multi-node run:

- :attr:`nodes` — the node roster (a tuple of :class:`GraphNode`\\ s).
- :attr:`edges` — the directed-edge set (a tuple of :class:`GraphEdge`\\ s).
- :attr:`entry` — the single entry-node id.
- :attr:`terminals` — one or more terminal-node ids.
- :attr:`config` — the graph-wide :class:`GraphConfig`.
- :attr:`hooks` — attached :class:`GraphHooks` observers.
- :attr:`id` / :attr:`description` / :attr:`metadata` — identity and
  free-form tags for tracing and logging.

A ``Graph`` implements :class:`Executable` directly. The graph loop
treats a nested ``Graph`` exactly like any other node — calling
:meth:`invoke` runs the inner loop and returns the resulting
:class:`NodeResult`. This is what lets a ``Graph`` live inside
another ``Graph`` without an intermediate adapter. The "Graph of
Graphs" composition pattern.

Construction. Callers MUST go through :meth:`Graph.new` →
:class:`GraphBuilder` → :meth:`GraphBuilder.compile`. The direct
constructor is not part of the public API — it exists so the builder
can instantiate a frozen ``Graph`` at compile time. Building a
``Graph`` by hand is possible but bypasses the builder's validation
and should be reserved for tests.

Identity. :meth:`Graph.new(id=None)` auto-generates a ``graph-XXXX``
id when none is supplied, mirroring CrewAI and OpenAI's Agents
Python SDK. Supply an explicit id when you want deterministic trace
span names, or when you expect to persist and later restore a
checkpoint for the same graph (the checkpointer validates
``graph.id`` matches the saved ``graph_id``).
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypeVar, override

from troopai.adk.graphs.config import GraphConfig
from troopai.adk.orchestration.executable import (
    Executable,
    ExecutableInput,
    NodeResult,
)

if TYPE_CHECKING:
    from troopai.adk.graphs.builder import GraphBuilder
    from troopai.adk.graphs.hooks import GraphHooks
    from troopai.adk.graphs.node import GraphEdge, GraphNode
    from troopai.adk.run.config import RunConfig
    from troopai.adk.run.context import RunContext
    from troopai.adk.visualization.dot import DotRankdir
    from troopai.adk.visualization.mermaid import MermaidDirection


logger = logging.getLogger(__name__)


TContext = TypeVar("TContext")


def _generate_graph_id() -> str:
    """Return a slug-safe auto-generated graph id.

    Format: ``graph-<12 hex chars>``. Uses :func:`uuid.uuid4` for
    collision safety and trims to 12 hex characters for readability
    in logs and OTel spans — consistent with CrewAI's ``crewai-...``
    auto-ids and OpenAI Agents SDK's run ids. 12 hex chars give ~48
    bits of entropy which is more than enough for in-process uniqueness.
    """
    return f"graph-{uuid.uuid4().hex[:12]}"


@dataclass(frozen=True)
class Graph[TContext](Executable[TContext]):
    """Frozen, compiled graph ready for execution.

    ``Graph`` inherits from :class:`Executable` so nested graphs
    compose without adapters — a :class:`GraphNode` can hold a
    ``Graph`` directly and the outer loop will call
    :meth:`Graph.invoke` on it the same way it calls
    :meth:`AgentExecutable.invoke`.

    All fields are frozen. Mutation is impossible after
    :meth:`GraphBuilder.compile` returns, which keeps the graph safely
    shareable across concurrent runs and serialisation-friendly.

    Attributes:
        id: Unique graph identifier. Used as the OTel span name
            ``graph:<id>`` and as an element of the ``graph_path``
            propagated on every :class:`GraphStreamEvent`.
            Auto-generated ``graph-XXXX`` when :meth:`Graph.new` is
            called without an explicit id.
        description: Optional human-readable blurb. Emitted on
            :class:`GraphStartEvent` and on OTel span attributes.
        metadata: Free-form user tag dict. Escape hatch for attaching
            cost-center, team owner, or SLO tier to a graph.
        nodes: Tuple of :class:`GraphNode`\\ s — the node roster.
            Immutable; use :meth:`get_node` for id-based lookup.
        edges: Tuple of :class:`GraphEdge`\\ s. Ordering is
            construction order; the graph loop sorts on read when
            determinism matters.
        entry: Id of the single entry node. Guaranteed to name a node
            in :attr:`nodes` (validated at compile time).
        terminals: Tuple of terminal node ids (at least one).
        config: The graph-wide :class:`GraphConfig`.
        hooks: Tuple of :class:`GraphHooks` observers attached to the
            graph. Merged with any run-level hooks passed to
            :meth:`Runner.arun_graph`.
    """

    id: str
    """Unique graph identifier."""

    nodes: tuple[GraphNode, ...]
    """Node roster."""

    edges: tuple[GraphEdge, ...]
    """Directed-edge set."""

    entry: str
    """Id of the single entry node."""

    terminals: tuple[str, ...]
    """Terminal node ids (at least one)."""

    description: str | None = None
    """Human-readable description."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Free-form tags."""

    config: GraphConfig = field(default_factory=GraphConfig)
    """Graph-wide config (budgets, defaults, behaviour knobs)."""

    hooks: tuple[GraphHooks[Any], ...] = field(default_factory=tuple)
    """Attached :class:`GraphHooks` observers."""

    # -- Construction -----------------------------------------------

    @staticmethod
    def new(
        id: str | None = None,
        *,
        description: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> GraphBuilder[Any]:
        """Open a :class:`GraphBuilder` with fresh identity.

        This is the primary entry point to building a graph. The
        returned builder accumulates nodes and edges via method
        chaining; :meth:`GraphBuilder.compile` produces the frozen
        :class:`Graph` artifact.

        Args:
            id: Optional explicit graph id. When ``None``, an id of
                the form ``graph-<12 hex chars>`` is auto-generated
                (CrewAI / OpenAI Agents SDK pattern). Supply an
                explicit id when you want deterministic trace names
                or when you plan to persist checkpoints across
                process restarts.
            description: Optional human-readable description emitted
                on :class:`GraphStartEvent` and OTel spans.
            metadata: Optional free-form tag dict. Not validated;
                surfaced verbatim on events and spans.

        Returns:
            A :class:`GraphBuilder` ready to receive
            :meth:`node`/:meth:`edge`/:meth:`pipe` calls.

        Example::

            from troopai.adk.graphs import Graph

            g = (
                Graph.new("legal-review", description="Compliance + sign-off")
                .node("checker", compliance_agent)
                .node("approver", legal_agent)
                .pipe("checker", "approver")
                .entry("checker")
                .terminal("approver")
                .compile()
            )
        """
        # Local import — builder.py imports Graph, so importing it at
        # module top would create a circular dependency on first use.
        from troopai.adk.graphs.builder import GraphBuilder

        resolved_id = id if id is not None else _generate_graph_id()
        return GraphBuilder(
            id=resolved_id,
            description=description,
            metadata=dict(metadata) if metadata is not None else {},
        )

    # -- Lookup -----------------------------------------------------

    def get_node(self, node_id: str) -> GraphNode:
        """Return the :class:`GraphNode` with id ``node_id``.

        Raises ``KeyError`` when no node with that id exists.
        """
        for node in self.nodes:
            if node.id == node_id:
                return node
        raise KeyError(
            f"Graph(id={self.id!r}) has no node with id {node_id!r}. Known ids: {sorted(n.id for n in self.nodes)}"
        )

    def incoming_edges(self, node_id: str) -> tuple[GraphEdge, ...]:
        """Return the tuple of edges whose ``target`` is ``node_id``."""
        return tuple(e for e in self.edges if e.target == node_id)

    def outgoing_edges(self, node_id: str) -> tuple[GraphEdge, ...]:
        """Return the tuple of edges whose ``source`` is ``node_id``."""
        return tuple(e for e in self.edges if e.source == node_id)

    # -- Visualisation -----------------------------------------------

    def to_mermaid(self, *, direction: MermaidDirection = "LR") -> str:
        """Return a Mermaid ``flowchart`` rendering of this Graph's topology.

        Thin ergonomic wrapper around
        :func:`troopai.adk.visualization.graph_to_mermaid`. Pure
        function: no I/O. Walks ``self.nodes`` and ``self.edges`` —
        the immutable frozen artifact built by :meth:`GraphBuilder.compile`.

        Args:
            direction: Mermaid layout direction. ``"LR"`` (default)
                reads left-to-right; ``"TD"`` / ``"TB"`` reads top-down.
                Invalid values raise ``ValueError``.

        Returns:
            A complete Mermaid ``flowchart`` block ready to paste into
            GitHub Markdown or any Mermaid renderer.
        """
        from troopai.adk.visualization.mermaid import graph_to_mermaid

        return graph_to_mermaid(self, direction=direction)

    def to_dot(self, *, rankdir: DotRankdir = "LR") -> str:
        """Return a Graphviz DOT rendering of this Graph's topology.

        Thin ergonomic wrapper around
        :func:`troopai.adk.visualization.graph_to_dot`. Pure function:
        no I/O. Walks ``self.nodes`` and ``self.edges``.

        Args:
            rankdir: Graphviz layout direction. ``"LR"`` (default)
                reads left-to-right; ``"TB"`` reads top-down. Invalid
                values raise ``ValueError``.

        Returns:
            A complete DOT digraph string.
        """
        from troopai.adk.visualization.dot import graph_to_dot

        return graph_to_dot(self, rankdir=rankdir)

    # -- Executable surface -----------------------------------------

    @override
    async def invoke(
        self,
        input: ExecutableInput,
        context: RunContext[TContext],
        config: RunConfig,
    ) -> NodeResult[TContext]:
        """Run the graph end-to-end when embedded as a node in another graph.

        Delegates to :func:`run_graph_loop` (the same driver
        :meth:`Runner.arun_graph` uses) and converts the resulting
        :class:`GraphRunResult` into a :class:`NodeResult` so the outer
        graph treats the inner graph indistinguishably from any other
        node.

        The ``input.content`` becomes the inner graph's initial prompt.
        If the inner graph has multiple terminals, the aggregate dict
        is placed on :attr:`NodeResult.output`; ``final_text`` is set
        when the terminal output is a single string.

        Args:
            input: Uniform input envelope from the outer graph.
            context: Shared :class:`RunContext`. Usage from the inner
                run does NOT auto-propagate; the outer graph loop's
                :meth:`GraphState.record` aggregates via
                :attr:`NodeResult.usage`.
            config: :class:`RunConfig` threaded from the outer run.

        Returns:
            A :class:`NodeResult` wrapping the inner
            :class:`GraphRunResult`.
        """
        # Local import — run/graph_loop.py imports types from graphs/;
        # importing it at module top would create a circular dependency
        # at package import time.
        from troopai.adk.run.graph_loop import run_graph_loop

        logger.debug(
            "Graph.invoke: nested graph_id=%s from_node=%s edge_label=%s",
            self.id,
            input.from_node,
            input.edge_label,
        )

        inner_result = await run_graph_loop(
            graph=self,
            user_prompt=input.content,
            context=context,
            config=config,
        )

        from troopai.adk.graphs.interrupt import (
            Interrupt,
            InterruptException,
            NestedAgentInterrupt,
            NestedGraphInterrupt,
        )
        from troopai.adk.graphs.result import GraphRunStatus

        if inner_result.status == GraphRunStatus.INTERRUPTED:
            # Lift the lexicographically-first inner pending interrupt
            # to one outer NestedAgentInterrupt. The inner GraphState
            # rides on the exception's _nested_graph_state attribute
            # so the outer BSP catch can park it for resume.
            inner_state = inner_result.state
            if inner_state is None:
                raise RuntimeError(
                    f"inner graph {self.id!r} returned INTERRUPTED with no state — cannot lift the inner interrupt"
                )
            if len(inner_state.pending_interrupts) == 0:
                raise RuntimeError(
                    f"inner graph {self.id!r} is INTERRUPTED but has no pending interrupts — inconsistent state"
                )
            inner_id, inner_interrupt = sorted(inner_state.pending_interrupts.items())[0]
            outer_iv: Interrupt
            if isinstance(inner_interrupt, NestedAgentInterrupt):
                outer_iv = NestedAgentInterrupt(
                    node_id="",
                    question=inner_interrupt.question,
                    metadata={
                        **inner_interrupt.metadata,
                        "inner_graph_id": self.id,
                        "inner_node_id": inner_id,
                    },
                    agent_name=inner_interrupt.agent_name,
                    tool_call_ids=inner_interrupt.tool_call_ids,
                )
            else:
                # Inner plain Interrupt — lift as a distinct NestedGraphInterrupt
                # (no agent_name / tool_call_ids). A NestedAgentInterrupt with an
                # empty agent_name would be REJECTED by GraphState.from_dict on
                # rehydrate, making the outer graph permanently non-resumable
                # after a checkpoint. The reply is a plain value forwarded into
                # the inner graph's GraphResume.
                outer_iv = NestedGraphInterrupt(
                    node_id="",
                    question=inner_interrupt.question,
                    metadata={
                        **inner_interrupt.metadata,
                        "inner_graph_id": self.id,
                        "inner_node_id": inner_id,
                    },
                )
            exc = InterruptException(outer_iv)
            exc._nested_graph_state = inner_state  # type: ignore[attr-defined]
            raise exc

        if inner_result.status == GraphRunStatus.FAILED:
            # Re-raise the inner failure verbatim. A dedicated error-lift
            # path wires this fully; a generic wrapper keeps the test scope
            # clear.
            raise RuntimeError(f"inner graph {self.id!r} failed: {inner_result.error or '(no error message)'}")

        if inner_result.status != GraphRunStatus.COMPLETED:
            # MAX_SUPERSTEPS / MAX_TOKENS / NO_READY_NODES: the inner graph
            # hit a budget cap or deadlocked without producing a terminal.
            # Surface it explicitly — returning its partial/empty
            # ``final_output`` would let the outer graph record this node as a
            # clean success and silently swallow a budget or routing exit.
            raise RuntimeError(f"inner graph {self.id!r} did not complete: status={inner_result.status.value}.")

        final_output = inner_result.final_output
        final_text = final_output if isinstance(final_output, str) else None

        return NodeResult(
            output=final_output,
            new_items=list(inner_result.new_items),
            usage=inner_result.cumulative_usage,
            final_text=final_text,
            metadata={
                "adapter": "graph",
                "graph_id": self.id,
                "status": inner_result.status.value,
                "total_supersteps": inner_result.total_supersteps,
                "per_node_usage": dict(inner_result.per_node_usage),
            },
        )


__all__ = ["Graph"]
