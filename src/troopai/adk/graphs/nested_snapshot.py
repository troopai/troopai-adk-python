"""Snapshot wrapper for nodes paused on a nested defer.

A graph node whose executable is an :class:`~troopai.adk.agents.agent.Agent`
parks a :class:`~troopai.adk.run.state.RunState` when the agent defers a
tool call. A graph node whose executable is itself a
:class:`~troopai.adk.graphs.graph.Graph` parks a
:class:`~troopai.adk.graphs.state.GraphState` when the inner graph suspends.
:class:`NestedSnapshot` unifies these two cases behind a single discriminator
so the BSP loop's dispatch site can route by ``snap.kind`` rather than by
inspecting the executable's type at resume time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from troopai.adk.graphs.graph import Graph
    from troopai.adk.graphs.state import GraphState
    from troopai.adk.run.state import RunState


@dataclass(frozen=True, kw_only=True)
class NestedSnapshot:
    """Wrapper for sub-state parked at a nested-defer node.

    Exactly one of ``run_state`` (kind=``"agent"``) or ``graph_state``
    (kind=``"graph"``) is populated. Construction enforces the
    invariant; consumers read ``kind`` to route.

    Attributes:
        kind: Discriminator — ``"agent"`` (agent-paused) or
            ``"graph"`` (inner-graph-paused).
        run_state: The paused agent's :class:`~troopai.adk.run.state.RunState`.
            Set iff ``kind == "agent"``.
        graph_state: The paused inner graph's
            :class:`~troopai.adk.graphs.state.GraphState`. Set iff
            ``kind == "graph"``.
    """

    kind: Literal["agent", "graph"]
    """Discriminator: ``"agent"`` (agent-paused) or ``"graph"`` (inner-graph-paused)."""

    run_state: RunState | None = None
    """The paused agent's :class:`~troopai.adk.run.state.RunState`. Set iff
    ``kind == "agent"``."""

    graph_state: GraphState[Any] | None = None
    """The paused inner graph's :class:`~troopai.adk.graphs.state.GraphState`.
    Set iff ``kind == "graph"``."""

    def __post_init__(self) -> None:
        if self.kind == "agent":
            if self.run_state is None:
                raise ValueError("NestedSnapshot(kind='agent') requires run_state to be set.")
            if self.graph_state is not None:
                raise ValueError("NestedSnapshot(kind='agent') must not carry graph_state.")
        elif self.kind == "graph":
            if self.graph_state is None:
                raise ValueError("NestedSnapshot(kind='graph') requires graph_state to be set.")
            if self.run_state is not None:
                raise ValueError("NestedSnapshot(kind='graph') must not carry run_state.")

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict.

        Both kinds delegate to the wrapped sub-state's own ``to_dict``
        when applicable. The discriminator + payload is reconstructed by
        :meth:`from_dict` against the parent ``Graph`` for graph-kind
        entries.

        Returns:
            A JSON-serialisable dict with a ``"kind"`` discriminator and
            either a ``"run_state"`` or ``"graph_state"`` payload.
        """
        if self.kind == "agent":
            if self.run_state is None:
                raise ValueError(
                    "NestedSnapshot(kind='agent').to_dict(): run_state is None — __post_init__ invariant violated"
                )
            return {"kind": "agent", "run_state": self.run_state.to_dict()}
        if self.graph_state is None:
            raise ValueError(
                "NestedSnapshot(kind='graph').to_dict(): graph_state is None — __post_init__ invariant violated"
            )
        return {"kind": "graph", "graph_state": self.graph_state.to_dict()}

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        graph: Graph[Any],
        node_id: str | None = None,
    ) -> NestedSnapshot:
        """Reconstruct a :class:`NestedSnapshot` from :meth:`to_dict` output.

        For a graph-kind snapshot the inner
        :class:`~troopai.adk.graphs.state.GraphState`'s node ids belong to
        the INNER graph — the executable of the parent node — not to the
        parent graph itself. Rehydrating the inner state against the parent
        graph would reject every inner node id as unknown. So ``node_id``
        identifies the parent node whose executable is that inner graph, and
        the inner state is validated against it.

        Args:
            data: Payload produced by :meth:`to_dict`.
            graph: The parent :class:`~troopai.adk.graphs.graph.Graph`. For
                a graph-kind snapshot the inner graph is resolved off this
                graph's node ``node_id``.
            node_id: Id of the parent node whose executable is the inner
                graph. Required when ``data["kind"] == "graph"``; unused for
                the agent kind.

        Returns:
            A fully rehydrated :class:`NestedSnapshot`.

        Raises:
            ValueError: When ``data["kind"]`` is neither ``"agent"`` nor
                ``"graph"``; when a graph-kind snapshot is given no
                ``node_id``; or when ``node_id`` is unknown or its executable
                is not a :class:`~troopai.adk.graphs.graph.Graph`.
        """
        from troopai.adk.graphs.graph import Graph as GraphCls
        from troopai.adk.graphs.state import GraphState
        from troopai.adk.run.state import RunState

        kind_raw = data.get("kind")
        if kind_raw == "agent":
            return cls(
                kind="agent",
                run_state=RunState.from_dict(data["run_state"]),
            )
        if kind_raw == "graph":
            if node_id is None:
                raise ValueError(
                    "NestedSnapshot.from_dict(kind='graph') requires node_id to resolve "
                    "the inner graph off the parent node's executable."
                )
            try:
                executable = graph.get_node(node_id).executable
            except KeyError as exc:
                raise ValueError(f"NestedSnapshot.from_dict: node_id {node_id!r} is not a known node id") from exc
            if not isinstance(executable, GraphCls):
                raise ValueError(
                    f"NestedSnapshot.from_dict: node {node_id!r} executable is "
                    f"{type(executable).__name__}, not a Graph — cannot rehydrate the inner graph_state."
                )
            return cls(
                kind="graph",
                graph_state=GraphState.from_dict(data["graph_state"], executable),
            )
        raise ValueError(f"NestedSnapshot.from_dict: unknown kind {kind_raw!r}. Expected one of 'agent', 'graph'.")


__all__ = ["NestedSnapshot"]
