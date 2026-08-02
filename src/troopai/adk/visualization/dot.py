"""Graphviz DOT emitters for :class:`Flow` and :class:`Graph`.

Pure functions translating immutable topology data into Graphviz DOT
strings ready for the ``dot`` CLI or any DOT-aware renderer.

Node shapes:

- ``@flow_start`` and graph entries → ``oval``.
- ``@flow_listen`` and intermediate graph nodes → ``box``.
- ``@flow_router`` → ``diamond``.
- AND / OR gates → ``circle`` labelled ``AND`` / ``OR``.

Edge shapes:

- Conditional edges (Graph ``when=`` predicates) → ``style=dashed``.
- Edge labels appear as ``[label="..."]``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from troopai.adk.flows.registry import build_transition_table
from troopai.adk.visualization.helpers import (
    assert_no_collision,
    build_step_lookup,
    escape_label,
    gate_node_id,
    node_label_from_desc,
    safe,
)

if TYPE_CHECKING:
    from troopai.adk.flows.definition import FlowDefinition
    from troopai.adk.flows.flow import Flow
    from troopai.adk.flows.flow_wrappers import FlowStep
    from troopai.adk.flows.registry import FlowTransitionTable
    from troopai.adk.graphs.graph import Graph
    from troopai.adk.graphs.node import GraphEdge


DotRankdir = Literal["LR", "TB", "RL", "BT"]
"""Layout direction accepted by Graphviz ``rankdir``."""

_VALID_RANKDIRS: frozenset[str] = frozenset({"LR", "TB", "RL", "BT"})
"""Runtime-validated rankdir values matching :data:`DotRankdir`."""


def flow_to_dot(flow: Flow, *, rankdir: DotRankdir = "LR") -> str:
    """Emit a Graphviz DOT digraph describing the Flow's topology.

    Pure function: same input → same output, no I/O.

    Args:
        flow: A constructed :class:`Flow` instance. Only the class
            registry is read; no run is triggered.
        rankdir: Graphviz layout direction (``"LR"`` default).

    Returns:
        A complete DOT digraph string starting with ``digraph "flow" {``
        and ending with ``}``.

    Raises:
        ValueError: When ``rankdir`` is not a Graphviz rankdir
            literal, or when two distinct identifiers collapse to the
            same sanitised node id.
    """
    if rankdir not in _VALID_RANKDIRS:
        raise ValueError(
            f"rankdir must be one of {sorted(_VALID_RANKDIRS)}; got {rankdir!r}.",
        )
    registry = flow.get_registry()
    table = build_transition_table(registry)
    step_lookup = build_step_lookup(flow)
    lines: list[str] = ['digraph "flow" {', f"    rankdir={rankdir};", '    node [fontname="Helvetica"];']
    declared: set[str] = set()
    sanitised: dict[str, str] = {}
    inverse: dict[str, str] = {}

    for name in sorted(registry.starts):
        assert_no_collision(sanitised, name, safe(name), inverse)
        lines.append(f"    {_node_decl(name, 'start', step_lookup)}")
        declared.add(name)
    for name in sorted(registry.listeners.keys()):
        assert_no_collision(sanitised, name, safe(name), inverse)
        lines.append(f"    {_node_decl(name, 'listen', step_lookup)}")
        declared.add(name)
    for name in sorted(registry.routers.keys()):
        assert_no_collision(sanitised, name, safe(name), inverse)
        lines.append(f"    {_node_decl(name, 'router', step_lookup)}")
        declared.add(name)

    lines.extend(_flow_direct_edges(table, sanitised, inverse))
    lines.extend(_flow_router_edges(table, sanitised, inverse))
    lines.extend(_flow_gate_edges(table, declared, sanitised, inverse))

    lines.append("}")
    return "\n".join(lines)


def definition_to_dot(defn: FlowDefinition, *, rankdir: DotRankdir = "LR") -> str:
    """Emit a Graphviz DOT digraph from a :class:`FlowDefinition`.

    Produces the same diagram as :func:`flow_to_dot` but accepts a
    :class:`~troopai.adk.flows.definition.FlowDefinition` directly, so
    visualisation is possible without constructing or running a
    :class:`~troopai.adk.flows.flow.Flow` instance.

    Pure function: no I/O, no side effects.

    Args:
        defn: A frozen :class:`FlowDefinition` produced by
            :meth:`~troopai.adk.flows.flow.Flow.get_definition` or
            :func:`~troopai.adk.flows.definition.build_flow_definition`.
        rankdir: Graphviz layout direction (``"LR"`` default).

    Returns:
        A complete DOT digraph string starting with ``digraph "flow" {``.

    Raises:
        ValueError: When ``rankdir`` is invalid or when two distinct
            identifiers collapse to the same sanitised node id.
    """
    if rankdir not in _VALID_RANKDIRS:
        raise ValueError(
            f"rankdir must be one of {sorted(_VALID_RANKDIRS)}; got {rankdir!r}.",
        )

    desc_lookup: dict[str, str | None] = {s.name: s.description for s in defn.steps}
    lines: list[str] = ['digraph "flow" {', f"    rankdir={rankdir};", '    node [fontname="Helvetica"];']
    declared: set[str] = set()
    sanitised: dict[str, str] = {}
    inverse: dict[str, str] = {}

    for step in defn.steps:
        assert_no_collision(sanitised, step.name, safe(step.name), inverse)
        lines.append(f"    {_dot_node_decl(step.name, step.role, desc_lookup)}")
        declared.add(step.name)

    lines.extend(_defn_direct_edges_dot(defn, sanitised, inverse))
    lines.extend(_defn_router_edges_dot(defn, sanitised, inverse))
    lines.extend(_defn_gate_edges_dot(defn, declared, sanitised, inverse))

    lines.append("}")
    return "\n".join(lines)


def _dot_node_decl(name: str, role: str, desc_lookup: dict[str, str | None]) -> str:
    """Build a DOT node declaration using a pre-built description lookup.

    Args:
        name: Step method name (node id).
        role: ``"start"`` / ``"listen"`` / ``"router"``.
        desc_lookup: Mapping from step name to optional description string.
            Passed to :func:`~troopai.adk.visualization.helpers.node_label_from_desc`
            for explicit ``is not None`` label resolution.

    Returns:
        DOT node declaration string ending with ``;``.
    """
    label = node_label_from_desc(name, desc_lookup)
    shape = _role_shape(role)
    return f'"{safe(name)}" [label="{escape_label(label)}", shape={shape}];'


def _defn_direct_edges_dot(
    defn: FlowDefinition,
    sanitised: dict[str, str],
    inverse: dict[str, str],
) -> list[str]:
    """Emit DOT direct-trigger edges from a :class:`FlowDefinition`.

    Args:
        defn: The compiled flow definition.
        sanitised: Collision accumulator shared with the caller.
        inverse: Inverse mapping for O(1) collision detection.

    Returns:
        Sorted list of DOT edge declarations.
    """
    trigger_to_listeners: dict[str, list[str]] = {}
    for listener_name, triggers in defn.direct_triggers.items():
        for trigger in triggers:
            trigger_to_listeners.setdefault(trigger, []).append(listener_name)
    edges: list[str] = []
    for trigger in sorted(trigger_to_listeners.keys()):
        assert_no_collision(sanitised, trigger, safe(trigger), inverse)
        for listener in sorted(trigger_to_listeners[trigger]):
            assert_no_collision(sanitised, listener, safe(listener), inverse)
            edges.append(f'    "{safe(trigger)}" -> "{safe(listener)}";')
    return edges


def _defn_router_edges_dot(
    defn: FlowDefinition,
    sanitised: dict[str, str],
    inverse: dict[str, str],
) -> list[str]:
    """Emit DOT router-trigger edges from a :class:`FlowDefinition`.

    Args:
        defn: The compiled flow definition.
        sanitised: Collision accumulator shared with the caller.
        inverse: Inverse mapping for O(1) collision detection.

    Returns:
        Sorted list of DOT edge declarations.
    """
    trigger_to_routers: dict[str, list[str]] = {}
    for router_name, triggers in defn.router_triggers.items():
        for trigger in triggers:
            trigger_to_routers.setdefault(trigger, []).append(router_name)
    edges: list[str] = []
    for trigger in sorted(trigger_to_routers.keys()):
        assert_no_collision(sanitised, trigger, safe(trigger), inverse)
        for router in sorted(trigger_to_routers[trigger]):
            assert_no_collision(sanitised, router, safe(router), inverse)
            edges.append(f'    "{safe(trigger)}" -> "{safe(router)}";')
    return edges


def _defn_gate_edges_dot(
    defn: FlowDefinition,
    declared: set[str],
    sanitised: dict[str, str],
    inverse: dict[str, str],
) -> list[str]:
    """Emit DOT gate nodes and their edges from a :class:`FlowDefinition`.

    Args:
        defn: The compiled flow definition.
        declared: Mutable set of already-declared node ids.
        sanitised: Collision accumulator shared with the caller.
        inverse: Inverse mapping for O(1) collision detection.

    Returns:
        Lines for gate node declarations and wired edges.
    """
    lines: list[str] = []
    for gate in defn.gates:
        node_id = gate_node_id(gate.gate_id)
        assert_no_collision(sanitised, gate.gate_id, node_id, inverse)
        if node_id not in declared:
            label = "AND" if gate.kind == "and" else "OR"
            lines.append(f'    "{node_id}" [label="{label}", shape=circle];')
            declared.add(node_id)
        for trigger in sorted(gate.triggers):
            lines.append(f'    "{safe(trigger)}" -> "{node_id}";')
        lines.append(f'    "{node_id}" -> "{safe(gate.listener_name)}";')
    return lines


def _flow_direct_edges(table: FlowTransitionTable, sanitised: dict[str, str], inverse: dict[str, str]) -> list[str]:
    """Emit DOT edges for direct (non-gated, non-routed) triggers.

    Each trigger and listener id flows through :func:`assert_no_collision`
    so route labels colliding with step names surface as typed errors.

    Args:
        table: The compiled transition table.
        sanitised: Collision accumulator shared with the caller.
        inverse: Inverse mapping (sanitised → original) maintained for
            O(1) collision detection.

    Returns:
        Sorted list of edge declarations.
    """
    edges: list[str] = []
    for trigger in sorted(table.direct_listeners.keys()):
        assert_no_collision(sanitised, trigger, safe(trigger), inverse)
        for listener in sorted(table.direct_listeners[trigger]):
            assert_no_collision(sanitised, listener, safe(listener), inverse)
            edges.append(f'    "{safe(trigger)}" -> "{safe(listener)}";')
    return edges


def _flow_router_edges(table: FlowTransitionTable, sanitised: dict[str, str], inverse: dict[str, str]) -> list[str]:
    """Emit DOT edges from triggers into routers.

    Args:
        table: The compiled transition table.
        sanitised: Collision accumulator shared with the caller.
        inverse: Inverse mapping (sanitised → original) maintained for
            O(1) collision detection.

    Returns:
        Sorted list of edge declarations.
    """
    edges: list[str] = []
    for trigger in sorted(table.routers_for.keys()):
        assert_no_collision(sanitised, trigger, safe(trigger), inverse)
        for router in sorted(table.routers_for[trigger]):
            assert_no_collision(sanitised, router, safe(router), inverse)
            edges.append(f'    "{safe(trigger)}" -> "{safe(router)}";')
    return edges


def _flow_gate_edges(
    table: FlowTransitionTable,
    declared: set[str],
    sanitised: dict[str, str],
    inverse: dict[str, str],
) -> list[str]:
    """Emit DOT synthesised gate nodes and their incoming / outgoing edges.

    Gate node ids are also registered in ``sanitised`` so a step name
    that happens to slug to the same string surfaces as a typed
    collision instead of silently merging diagram nodes.

    Args:
        table: The compiled transition table.
        declared: Names of nodes already declared.
        sanitised: Collision accumulator shared with the caller.
        inverse: Inverse mapping (sanitised → original) maintained for
            O(1) collision detection.

    Returns:
        Lines for gate declarations + edges.
    """
    lines: list[str] = []
    for gid in sorted(table.and_gates.keys()):
        spec = table.and_gates[gid]
        node_id = gate_node_id(gid)
        assert_no_collision(sanitised, gid, node_id, inverse)
        if node_id not in declared:
            lines.append(f'    "{node_id}" [label="AND", shape=circle];')
            declared.add(node_id)
        for trigger in sorted(spec.triggers):
            lines.append(f'    "{safe(trigger)}" -> "{node_id}";')
        lines.append(f'    "{node_id}" -> "{safe(spec.listener_name)}";')
    for gid in sorted(table.or_gates.keys()):
        spec = table.or_gates[gid]
        node_id = gate_node_id(gid)
        assert_no_collision(sanitised, gid, node_id, inverse)
        if node_id not in declared:
            lines.append(f'    "{node_id}" [label="OR", shape=circle];')
            declared.add(node_id)
        for trigger in sorted(spec.triggers):
            lines.append(f'    "{safe(trigger)}" -> "{node_id}";')
        lines.append(f'    "{node_id}" -> "{safe(spec.listener_name)}";')
    return lines


def graph_to_dot(graph: Graph, *, rankdir: DotRankdir = "LR") -> str:
    """Emit a Graphviz DOT digraph describing the Graph's topology.

    Pure function: no I/O. Walks ``graph.nodes`` + ``graph.edges``.

    Args:
        graph: A compiled :class:`Graph` instance.
        rankdir: Graphviz layout direction (``"LR"`` default).

    Returns:
        A complete DOT digraph string.

    Raises:
        ValueError: When ``rankdir`` is invalid or when two distinct
            node ids collapse to the same sanitised slug.
    """
    if rankdir not in _VALID_RANKDIRS:
        raise ValueError(
            f"rankdir must be one of {sorted(_VALID_RANKDIRS)}; got {rankdir!r}.",
        )
    terminal_ids = set(graph.terminals)
    entry_id = graph.entry
    lines: list[str] = [
        f'digraph "{escape_label(graph.id)}" {{',
        f"    rankdir={rankdir};",
        '    node [fontname="Helvetica"];',
    ]
    sanitised: dict[str, str] = {}
    inverse: dict[str, str] = {}
    for node in sorted(graph.nodes, key=lambda n: n.id):
        assert_no_collision(sanitised, node.id, safe(node.id), inverse)
        label = node.description if node.description is not None else node.id
        shape = _graph_shape(node.id, entry_id, terminal_ids)
        lines.append(f'    "{safe(node.id)}" [label="{escape_label(label)}", shape={shape}];')
    for edge in sorted(graph.edges, key=lambda e: (e.source, e.target, e.label or "", e.priority)):
        lines.append(f"    {_render_graph_edge(edge)}")
    lines.append("}")
    return "\n".join(lines)


def _render_graph_edge(edge: GraphEdge) -> str:
    """Format a single :class:`GraphEdge` as a DOT edge declaration.

    Args:
        edge: A :class:`GraphEdge` instance.

    Returns:
        A DOT edge declaration string ending with ``;``.
    """
    attrs: list[str] = []
    if edge.label is not None and len(edge.label) > 0:
        attrs.append(f'label="{escape_label(edge.label)}"')
    elif edge.when is not None:
        attrs.append('label="when"')
    if edge.when is not None:
        attrs.append("style=dashed")
    suffix = f" [{', '.join(attrs)}]" if len(attrs) > 0 else ""
    return f'"{safe(edge.source)}" -> "{safe(edge.target)}"{suffix};'


def _graph_shape(node_id: str, entry: str, terminals: set[str]) -> str:
    """Return the DOT shape name for a graph node id.

    Args:
        node_id: Node id under test.
        entry: The graph's entry node id.
        terminals: Set of terminal node ids.

    Returns:
        ``"oval"`` for entry; ``"doublecircle"`` for terminals;
        ``"box"`` for everything else.
    """
    if node_id == entry:
        return "oval"
    if node_id in terminals:
        return "doublecircle"
    return "box"


def _node_decl(name: str, role: str, lookup: dict[str, FlowStep]) -> str:
    """Build a DOT node declaration for a Flow step.

    Args:
        name: Step method name (also the node id).
        role: One of ``"start"`` / ``"listen"`` / ``"router"``.
        lookup: Mapping from name to bound :class:`FlowStep` for
            description lookup.

    Returns:
        A DOT node declaration string ending with ``;``.
    """
    step = lookup.get(name)
    label = step.description if step is not None and step.description is not None else name
    shape = _role_shape(role)
    return f'"{safe(name)}" [label="{escape_label(label)}", shape={shape}];'


def _role_shape(role: str) -> str:
    """Map a flow step role to a DOT shape name.

    Args:
        role: ``"start"`` / ``"listen"`` / ``"router"``.

    Returns:
        ``"oval"`` / ``"box"`` / ``"diamond"``.
    """
    if role == "start":
        return "oval"
    if role == "router":
        return "diamond"
    return "box"
