"""``GraphBuilder`` — fluent, opinionated API for constructing a :class:`Graph`.

Design goals, in priority order:

1. **Readability first.** A graph definition should read top-to-bottom
   like a spec, not like a diagram encoded in string fragments.
   Compare:

   - LangGraph: ``StateGraph(State).add_node("x", fn).add_edge(START, "x").add_conditional_edges(...)``
     with ``Annotated[list, add_messages]`` reducers declared on state
     fields. Readable once you've read the docs three times.
   - Strands: ``GraphBuilder().add_node(agent, "x").add_edge("x", "y")``
     with OR-default joins and a broken-parallel bug in older
     versions.
   - TroopAI (this file): ``Graph.new("x").node("a", agent).node("b", agent).pipe("a", "b").entry("a").terminal("b").compile()``.
     The receiving node declares merge semantics inline (``merge=Merge.concat_text``);
     no ``Annotated`` magic.

2. **Single routing mechanism.** Every routing decision is one
   :meth:`edge` call with an optional ``when=`` predicate. No
   ``Command(goto=)`` duality, no ``path_map`` dicts, no LLM routing
   tokens baked into the graph layer — those belong to
   :class:`Swarm` / :class:`StructuredRoutingPolicy`.

3. **Fail at compile time, not at run time.** :meth:`compile`
   validates the graph shape (entry exists, terminals exist, every
   edge references a known node, no duplicate node ids, entry has no
   incoming edges, terminals have no outgoing edges, the graph is
   non-empty). Pipelines that would crash mid-run surface as
   ``ValueError`` at build time.

4. **Zero adapter boilerplate.** :meth:`node` auto-wraps
   ``Agent``/``Swarm``/``Graph``/``Callable`` via
   :func:`to_executable`. Users never write
   ``AgentExecutable(agent=...)`` unless they want to pass a
   non-default ``max_turns``.

The builder is mutable during construction and frozen-at-compile. It
does not share state with the compiled :class:`Graph`; mutating a
builder after :meth:`compile` has no effect on the produced graph.
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypeVar

from troopai.adk.graphs.adapters import to_executable
from troopai.adk.graphs.config import GraphConfig
from troopai.adk.graphs.graph import Graph
from troopai.adk.graphs.join import JoinSemantics
from troopai.adk.graphs.merge import DEFAULT_MERGE, MergeFn
from troopai.adk.graphs.node import EdgeCondition, GraphEdge, GraphNode, NodeErrorHandlerFn

if TYPE_CHECKING:
    from troopai.adk.graphs.config import NodeRetryPolicy
    from troopai.adk.graphs.hooks import GraphHooks


logger = logging.getLogger(__name__)


TContext = TypeVar("TContext")


@dataclass
class GraphBuilder[TContext]:
    """Mutable, fluent builder producing a frozen :class:`Graph`.

    Every method except :meth:`compile` returns ``self`` so callers
    can chain. Construction order matters only for :meth:`pipe` — the
    first argument is the upstream node, subsequent arguments each
    receive an edge from the prior.

    **Node defaults**: :meth:`set_node_defaults` sets fallback values for
    ``retry``, ``timeout``, ``metadata``, and ``on_error`` that apply to
    all *subsequently* added nodes.  Explicit per-node arguments always
    win — layering is field-by-field, not dict-merge overwrite.  Defaults
    only apply to nodes added AFTER the :meth:`set_node_defaults` call;
    nodes added before that call retain their original values.

    Attributes:
        id: Target :attr:`Graph.id`.
        description: Target :attr:`Graph.description`.
        metadata: Target :attr:`Graph.metadata`.
        _nodes_by_id: Working set of :class:`GraphNode`\\ s keyed by
            id. Rejects duplicates on insert to catch typos early.
        _edges: Working edge list. Validated at compile time.
        _entry: Working entry id. Set via :meth:`entry`; must be set
            before :meth:`compile`.
        _terminals: Working terminal id set. Set via :meth:`terminal`;
            at least one must be set before :meth:`compile`.
        _config: Working :class:`GraphConfig`. Defaults to
            ``GraphConfig()`` — mutate via :meth:`with_config`.
        _hooks: Working tuple of :class:`GraphHooks` attached via
            :meth:`with_hooks`.
        _default_retry: Default retry policy for subsequently added nodes.
        _default_timeout: Default per-attempt timeout for subsequently
            added nodes.
        _default_metadata: Default metadata merged into subsequently added
            nodes (explicit per-node metadata shadows individual keys).
        _default_on_error: Default error handler for subsequently added
            nodes.
    """

    id: str
    """Graph id (non-empty; pre-set by :meth:`Graph.new`)."""

    description: str | None = None
    """Optional description."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Optional metadata."""

    _nodes_by_id: dict[str, GraphNode] = field(default_factory=dict)
    """Working node set, keyed by id."""

    _edges: list[GraphEdge] = field(default_factory=list)
    """Working edge list."""

    _entry: str | None = None
    """Working entry id."""

    _terminals: set[str] = field(default_factory=set)
    """Working terminal ids."""

    _config: GraphConfig = field(default_factory=GraphConfig)
    """Working :class:`GraphConfig`."""

    _hooks: tuple[GraphHooks[Any], ...] = field(default_factory=tuple)
    """Attached :class:`GraphHooks` observers."""

    _default_retry: NodeRetryPolicy | None = None
    """Default retry policy; applied to subsequently added nodes when
    the per-node ``retry`` argument is not supplied."""

    _default_timeout: float | None = None
    """Default per-attempt timeout (seconds); applied to subsequently
    added nodes when the per-node ``timeout`` argument is not supplied."""

    _default_metadata: dict[str, Any] = field(default_factory=dict)
    """Default metadata merged into subsequently added nodes.  Per-node
    ``metadata`` shadows individual keys."""

    _default_on_error: NodeErrorHandlerFn | None = None
    """Default error handler; applied to subsequently added nodes when
    the per-node ``on_error`` argument is not supplied."""

    # -- Node / edge API -------------------------------------------

    def node(
        self,
        node_id: str,
        executable: Any,
        *,
        merge: MergeFn | None = None,
        join: JoinSemantics | None = None,
        description: str | None = None,
        metadata: dict[str, Any] | None = None,
        retry: NodeRetryPolicy | None = None,
        timeout: float | None = None,
        on_error: NodeErrorHandlerFn | None = None,
    ) -> GraphBuilder[TContext]:
        """Register a node.

        The ``executable`` can be any of:

        - an :class:`~troopai.adk.orchestration.executable.Executable`
          (including a :class:`Graph`) — used as-is,
        - an :class:`~troopai.adk.agents.agent.Agent` — auto-wrapped in
          :class:`~troopai.adk.graphs.adapters.AgentExecutable`,
        - a :class:`~troopai.adk.swarms.swarm.Swarm` — auto-wrapped in
          :class:`~troopai.adk.graphs.adapters.SwarmExecutable`,
        - a plain callable — auto-wrapped in
          :class:`~troopai.adk.graphs.adapters.CallableExecutable`.

        Per-node arguments always win over builder-level node defaults set
        via :meth:`set_node_defaults`.  Metadata is merged field-by-field:
        the per-node dict shadows individual keys from the default.

        Args:
            node_id: Unique id for this node within the graph. MUST
                match the node-id pattern (alphanumerics, ``_``,
                ``-``); validated on :class:`GraphNode` construction.
            executable: The object to run when this node fires. See
                the list above for auto-wrap behaviour.
            merge: Optional override for the default
                :mod:`troopai.adk.graphs.merge` strategy. Applied when
                multiple upstream edges feed this node.
            join: Optional override for the fan-in semantics.
                Defaults to :attr:`JoinSemantics.AND`.
            description: Optional human-readable blurb.
            metadata: Optional free-form tag dict. Merged with the
                builder-level default metadata; per-node keys win.
            retry: Optional per-node retry policy. When ``None``,
                inherits the builder-level default set via
                :meth:`set_node_defaults`.
            timeout: Optional per-node per-attempt timeout in seconds.
                When ``None``, inherits the builder-level default.
            on_error: Optional per-node error handler. When ``None``,
                inherits the builder-level default.

        Returns:
            ``self`` for chaining.

        Raises:
            ValueError: When ``node_id`` is already in the builder.
        """
        if node_id in self._nodes_by_id:
            raise ValueError(
                f"GraphBuilder(id={self.id!r}) already has a node with id {node_id!r}. Node ids must be unique."
            )

        # Merge metadata: builder defaults provide the base; per-node
        # dict shadows individual keys.  Both may be empty.
        effective_metadata: dict[str, Any] = dict(self._default_metadata)
        if metadata is not None:
            effective_metadata.update(metadata)

        wrapped = to_executable(executable)
        node = GraphNode(
            id=node_id,
            executable=wrapped,
            merge=merge if merge is not None else DEFAULT_MERGE,
            join=join if join is not None else JoinSemantics.AND,
            description=description,
            metadata=effective_metadata,
            retry=retry if retry is not None else self._default_retry,
            timeout=timeout if timeout is not None else self._default_timeout,
            on_error=on_error if on_error is not None else self._default_on_error,
        )
        self._nodes_by_id[node_id] = node
        logger.debug(
            "GraphBuilder(%s).node: added %s (merge=%s, join=%s)",
            self.id,
            node_id,
            node.merge.__name__ if hasattr(node.merge, "__name__") else repr(node.merge),
            node.join.value,
        )
        return self

    def edge(
        self,
        source: str,
        target: str,
        *,
        when: EdgeCondition | None = None,
        label: str | None = None,
        priority: int = 0,
    ) -> GraphBuilder[TContext]:
        """Register a directed edge from ``source`` to ``target``.

        Args:
            source: Upstream node id. MUST already be registered via
                :meth:`node`.
            target: Downstream node id. MUST already be registered
                via :meth:`node`.
            when: Optional pure predicate on the upstream
                :class:`NodeResult`. When ``None`` the edge always
                fires.
            label: Optional label propagated to
                :attr:`ExecutableInput.edge_label`.
            priority: Higher-first ordering; advisory until
                non-concurrent scheduling is implemented.

        Returns:
            ``self`` for chaining.

        Raises:
            ValueError: When either endpoint is unknown. Self-loops
                are rejected inside :class:`GraphEdge.__post_init__`.
        """
        if source not in self._nodes_by_id:
            raise ValueError(
                f"GraphBuilder(id={self.id!r}).edge: source {source!r} is "
                f"not a registered node. Known ids: "
                f"{sorted(self._nodes_by_id.keys())}"
            )
        if target not in self._nodes_by_id:
            raise ValueError(
                f"GraphBuilder(id={self.id!r}).edge: target {target!r} is "
                f"not a registered node. Known ids: "
                f"{sorted(self._nodes_by_id.keys())}"
            )

        self._edges.append(
            GraphEdge(
                source=source,
                target=target,
                when=when,
                label=label,
                priority=priority,
            )
        )
        logger.debug(
            "GraphBuilder(%s).edge: %s -> %s (label=%s, conditional=%s)",
            self.id,
            source,
            target,
            label,
            when is not None,
        )
        return self

    def pipe(self, *ids: str) -> GraphBuilder[TContext]:
        """Chain a linear sequence of nodes with unconditional edges.

        ``builder.pipe("a", "b", "c")`` is sugar for::

            builder.edge("a", "b").edge("b", "c")

        At least two ids are required. Each adjacent pair becomes an
        unconditional edge. The nodes MUST already be registered.

        Returns:
            ``self`` for chaining.

        Raises:
            ValueError: When fewer than two ids are supplied.
        """
        if len(ids) < 2:
            raise ValueError(f"GraphBuilder(id={self.id!r}).pipe: needs at least 2 node ids to chain, got {len(ids)}.")
        for src, dst in itertools.pairwise(ids):
            self.edge(src, dst)
        return self

    # -- Graph-shape declarations ----------------------------------

    def entry(self, node_id: str) -> GraphBuilder[TContext]:
        """Declare ``node_id`` as the graph's entry node.

        Exactly one entry is allowed. Calling :meth:`entry` twice
        with different ids raises ``ValueError``; calling it twice
        with the same id is idempotent.

        Returns:
            ``self`` for chaining.

        Raises:
            ValueError: When ``node_id`` is unknown or when a
                different entry is already declared.
        """
        if node_id not in self._nodes_by_id:
            raise ValueError(
                f"GraphBuilder(id={self.id!r}).entry: {node_id!r} is not a "
                f"registered node. Known ids: "
                f"{sorted(self._nodes_by_id.keys())}"
            )
        if self._entry is not None and self._entry != node_id:
            raise ValueError(
                f"GraphBuilder(id={self.id!r}) already has entry "
                f"{self._entry!r}; refusing to overwrite with {node_id!r}."
            )
        self._entry = node_id
        return self

    def terminal(self, *node_ids: str) -> GraphBuilder[TContext]:
        """Declare one or more terminal node ids.

        Terminals are nodes whose completion finishes the graph run.
        A graph may have multiple terminals — when it does,
        :attr:`GraphRunResult.final_output` becomes a dict keyed by
        terminal id.

        Returns:
            ``self`` for chaining.

        Raises:
            ValueError: When any id is unknown.
        """
        if len(node_ids) == 0:
            raise ValueError(f"GraphBuilder(id={self.id!r}).terminal: needs at least one node id.")
        for nid in node_ids:
            if nid not in self._nodes_by_id:
                raise ValueError(
                    f"GraphBuilder(id={self.id!r}).terminal: {nid!r} is "
                    f"not a registered node. Known ids: "
                    f"{sorted(self._nodes_by_id.keys())}"
                )
            self._terminals.add(nid)
        return self

    def set_node_defaults(
        self,
        *,
        retry: NodeRetryPolicy | None = None,
        timeout: float | None = None,
        metadata: dict[str, Any] | None = None,
        on_error: NodeErrorHandlerFn | None = None,
    ) -> GraphBuilder[TContext]:
        """Set default values applied to all *subsequently* added nodes.

        ``None`` for any argument leaves that default unchanged —
        once set, a default cannot be cleared back to unset on this
        builder; construct a new builder if you need a clean slate.

        Field-by-field layering — per-node explicit arguments always win
        over these defaults.  For ``metadata`` the defaults provide the
        base and per-node keys shadow individual entries.

        **Ordering matters**: nodes added BEFORE this call are not
        affected; only nodes added AFTER inherit the defaults.

        Args:
            retry: Default :class:`~troopai.adk.graphs.config.NodeRetryPolicy`.
                Applied when a node's ``retry`` argument is ``None``.
            timeout: Default per-attempt timeout in seconds. Applied
                when a node's ``timeout`` argument is ``None``.
            metadata: Default metadata merged into subsequently added
                nodes.  Per-node ``metadata`` shadows individual keys.
            on_error: Default error handler. Applied when a node's
                ``on_error`` argument is ``None``.

        Returns:
            ``self`` for chaining.
        """
        if retry is not None:
            self._default_retry = retry
        if timeout is not None:
            self._default_timeout = timeout
        if metadata is not None:
            self._default_metadata = dict(metadata)
        if on_error is not None:
            self._default_on_error = on_error
        logger.debug(
            "GraphBuilder(%s).set_node_defaults: retry=%s timeout=%s metadata_keys=%s on_error=%s",
            self.id,
            retry,
            timeout,
            sorted(metadata.keys()) if metadata else [],
            on_error is not None,
        )
        return self

    def with_config(self, config: GraphConfig) -> GraphBuilder[TContext]:
        """Attach a :class:`GraphConfig` to the graph.

        Overrides the default ``GraphConfig()``. Returns ``self`` for
        chaining.
        """
        self._config = config
        return self

    def with_hooks(
        self,
        *hooks: GraphHooks[Any],
    ) -> GraphBuilder[TContext]:
        """Attach one or more :class:`GraphHooks` observers.

        Multiple calls accumulate. Returns ``self`` for chaining.
        """
        self._hooks = self._hooks + tuple(hooks)
        return self

    # -- Terminal --------------------------------------------------

    def compile(self) -> Graph[TContext]:
        """Validate and return an immutable :class:`Graph`.

        Validation errors raise ``ValueError`` with an actionable
        message. After :meth:`compile` returns, further builder
        mutations do NOT affect the returned graph (node / edge /
        terminal lists are copied into the frozen dataclass).

        Validation performed:

        1. At least one node is registered.
        2. :meth:`entry` has been called.
        3. :meth:`terminal` has been called at least once.
        4. The entry node exists.
        5. Every terminal node exists.
        6. The entry node has no incoming edges (would loop forever).
        7. Every terminal node has no outgoing edges (would never
           finish).
        8. Every non-entry, non-terminal node is reachable from the
           entry and can reach at least one terminal.

        Returns:
            A frozen :class:`Graph` ready for execution.
        """
        if len(self._nodes_by_id) == 0:
            raise ValueError(f"GraphBuilder(id={self.id!r}).compile: graph has no nodes.")
        if self._entry is None:
            raise ValueError(
                f"GraphBuilder(id={self.id!r}).compile: no entry declared; call .entry(node_id) before .compile()."
            )
        if len(self._terminals) == 0:
            raise ValueError(
                f"GraphBuilder(id={self.id!r}).compile: no terminal declared; "
                "call .terminal(node_id) at least once before .compile()."
            )

        # Entry must exist (guarded in entry() but re-check for safety).
        if self._entry not in self._nodes_by_id:
            raise ValueError(f"GraphBuilder(id={self.id!r}).compile: entry {self._entry!r} is not a registered node.")

        # Terminal existence (redundant guard; same rationale).
        for term in self._terminals:
            if term not in self._nodes_by_id:
                raise ValueError(f"GraphBuilder(id={self.id!r}).compile: terminal {term!r} is not a registered node.")

        # Entry has no incoming.
        incoming_to_entry = [e for e in self._edges if e.target == self._entry]
        if len(incoming_to_entry) > 0:
            sources = sorted(e.source for e in incoming_to_entry)
            raise ValueError(
                f"GraphBuilder(id={self.id!r}).compile: entry node "
                f"{self._entry!r} has incoming edges from {sources}. "
                "An entry cannot receive input from any other node."
            )

        # Terminals have no outgoing.
        for term in self._terminals:
            outgoing = [e for e in self._edges if e.source == term]
            if len(outgoing) > 0:
                targets = sorted(e.target for e in outgoing)
                raise ValueError(
                    f"GraphBuilder(id={self.id!r}).compile: terminal node "
                    f"{term!r} has outgoing edges to {targets}. "
                    "Terminals cannot forward to other nodes."
                )

        # Reachability from entry.
        reachable = _reachable_from(self._entry, self._edges)
        reachable.add(self._entry)
        unreachable = set(self._nodes_by_id.keys()) - reachable
        if len(unreachable) > 0:
            raise ValueError(
                f"GraphBuilder(id={self.id!r}).compile: nodes "
                f"{sorted(unreachable)} are not reachable from entry "
                f"{self._entry!r}. Remove them or add connecting edges."
            )

        # Reverse reachability to terminals.
        reversed_edges = [(e.target, e.source) for e in self._edges]
        can_reach_terminal: set[str] = set()
        for term in self._terminals:
            can_reach_terminal.update(_reverse_reachable(term, reversed_edges))
            can_reach_terminal.add(term)
        dead_ends = set(self._nodes_by_id.keys()) - can_reach_terminal
        if len(dead_ends) > 0:
            raise ValueError(
                f"GraphBuilder(id={self.id!r}).compile: nodes "
                f"{sorted(dead_ends)} cannot reach any terminal "
                f"({sorted(self._terminals)}). The graph would run "
                "forever or exit via NO_READY_NODES."
            )

        compiled = Graph[TContext](
            id=self.id,
            nodes=tuple(self._nodes_by_id.values()),
            edges=tuple(self._edges),
            entry=self._entry,
            terminals=tuple(sorted(self._terminals)),
            description=self.description,
            metadata=dict(self.metadata),
            config=self._config,
            hooks=self._hooks,
        )
        logger.info(
            "GraphBuilder(%s).compile: ok (%d nodes, %d edges, entry=%s, terminals=%s)",
            self.id,
            len(compiled.nodes),
            len(compiled.edges),
            compiled.entry,
            list(compiled.terminals),
        )
        return compiled


def _reachable_from(
    start: str,
    edges: list[GraphEdge],
) -> set[str]:
    """Return the set of node ids reachable from ``start`` via ``edges``.

    Forward reachability used by :meth:`compile` to flag unreachable
    nodes. Ignores edge predicates — a conditionally-firing edge
    still contributes to structural reachability; predicate-always-false
    edges are a runtime concern, not a compile concern.
    """
    adj: dict[str, list[str]] = {}
    for e in edges:
        adj.setdefault(e.source, []).append(e.target)

    visited: set[str] = set()
    stack = [start]
    while len(stack) > 0:
        n = stack.pop()
        for nxt in adj.get(n, []):
            if nxt not in visited:
                visited.add(nxt)
                stack.append(nxt)
    return visited


def _reverse_reachable(
    start: str,
    reversed_edges: list[tuple[str, str]],
) -> set[str]:
    """Return the set of node ids that can reach ``start``.

    Takes edges already flipped to ``(target, source)`` so the same
    DFS works. Used by :meth:`compile` to flag dead-end nodes.
    """
    adj: dict[str, list[str]] = {}
    for src, dst in reversed_edges:
        adj.setdefault(src, []).append(dst)

    visited: set[str] = set()
    stack = [start]
    while len(stack) > 0:
        n = stack.pop()
        for nxt in adj.get(n, []):
            if nxt not in visited:
                visited.add(nxt)
                stack.append(nxt)
    return visited


__all__ = ["GraphBuilder"]
